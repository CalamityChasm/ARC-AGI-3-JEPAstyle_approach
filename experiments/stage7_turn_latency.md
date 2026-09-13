# Stage 7 — Duck analyzer turn latency: it is queueing, but not for the reason we thought

Date: 2026-09-12. Branch: `stage7-turn-latency`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (run
artifacts, bundle source, or a locally-run computation over them);
**[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made.** Everything here is offline analysis of
an already-completed free run, plus one free `kaggle kernels push`. The daily
submission slot was untouched.

Reproduce every number with:

```
venv/Scripts/python.exe scripts/analyze_turn_latency.py  <kernel-output-dir>
venv/Scripts/python.exe scripts/analyze_batch_scaling.py <kernel-output-dir>
venv/Scripts/python.exe scripts/project_turns_rhae.py    <kernel-output-dir>/benchmark.json
```

Raw output committed at `experiments/stage7_turn_latency_measurements.txt`;
machine-readable at `stage7_turn_latency_data.json` / `stage7_turn_latency_rhae.json`;
the `/metrics` scrape the decomposition is built from at
`stage7_turn_latency_metrics.prom`.

Source run: `calamitychasm/arc3-duck-nvfp4-baseline` v2, `COMPLETE`, public-25
path, 2026-09-10, mean self-eval 10.69, profile `kv5-bf16-mtp3-c8-cg32`.

---

## TL;DR

1. **Turn latency is 87% queue wait.** [VERIFIED] Mean request e2e 145.70 s =
   queue **126.71 s (86.97%)** + inference 18.61 s (prefill 1.60 s = 1.10%,
   decode 17.01 s = 11.67%). Client/tool overhead is ~1.5%.
2. **Oversubscription is confirmed, but `TAAF_VLLM_MAX_NUM_SEQS = 8` is not the
   cause and raising it will do nothing.** [VERIFIED] The scheduler never
   reached 8 — `Running` was 2/3/6 (min/p50/max) across **all 792** scheduler
   snapshots, 0 of them at the cap. The binding constraint is the **KV pool**:
   105,202 tokens against 21,608 tokens resident per request = **4.87 requests
   fit**, at 80.3% mean KV utilisation with **191 preemptions**. The real
   oversubscription ratio is **7.87×** against ~3.13 effective slots, not 3.5×
   against 8.
3. **The throughput-vs-batch curve cannot be recovered from the existing
   artifacts** — `Running` sits at 3 in 89% of snapshots precisely because the
   KV pool pins it. That is what the free sweep is for; it is pushed and
   running.
4. **The RHAE upside is large and, unusually for this project, robustly
   signed.** Reading the harness's own scorer from source (not CLAUDE.md's
   stated formula, which is ambiguous in a sign-flipping way) shows the
   efficiency denominator runs over **all** levels, so more levels can never
   lower the score. At **2× turns** the projected public-25 mean rises from
   **10.69 to 17.7–37.7 (+66% to +253%)**, headline **24.8 (+132%)**.
5. **Recommendation: pursue it, via KV capacity — not via `MAX_NUM_SEQS`.**

---

## 0. What was asked, and where the premise was wrong

The lead was: *"Is turn latency dominated by queueing, and would raising
`TAAF_VLLM_MAX_NUM_SEQS` give each game materially more turns?"*

The first half is **confirmed and then some**. The second half is **refuted**
before any GPU time was spent, from the run's own server log. Both answers are
below, with the measurement that produced each.

---

## 1. The turn decomposition [VERIFIED]

Source: `vllm-metrics-final.prom`, the cumulative Prometheus scrape the bundle
writes at teardown. These are histogram `_sum / _count` pairs over the whole
run, so they are means over all 1,339 completed requests, not samples.

| phase | mean s | share of e2e | p50 ≤ | p90 ≤ | p99 ≤ |
|---|---:|---:|---:|---:|---:|
| **e2e request latency** | **145.70** | 100.00% | 240 | 240 | 480 |
| ‣ **queue (WAITING)** | **126.71** | **86.97%** | 240 | 240 | 240 |
| ‣ inference (RUNNING) | 18.61 | 12.77% | 15 | 40 | 120 |
| ‣ ‣ prefill | 1.60 | 1.10% | 2 | 2 | 5 |
| ‣ ‣ decode | 17.01 | 11.67% | 15 | 40 | 120 |
| time to first token | 128.69 | 88.32% | 160 | 640 | 640 |

Percentiles are bucket upper bounds, not interpolations — vLLM's buckets are
coarse and interpolating would invent precision. Read them as "≤".

**Closure checks, all passing:**

- `queue + inference = 145.32 s` against `e2e = 145.70 s`, residual **+0.38 s**.
- `queue + prefill = 128.31 s` against `TTFT = 128.69 s`. Consistent.
- Little's law on the throughput `X = 1339 / 7920 s = 0.1691 req/s`:

  | | predicted | log-observed |
  |---|---:|---:|
  | `X · queue` → requests waiting | 21.42 | **21.62** |
  | `X · inference` → requests running | 3.15 | **3.13** |
  | `X · e2e` → requests inside the server | 24.63 | 25 games |

  Every term closes. The last row is the answer to **(d) client-side / tool
  overhead**: 24.63 of 25 game threads have a request inside the server at any
  instant, so the harness spends **~1.5%** of a turn outside vLLM. Tool
  execution, parsing and the game step are not the problem.

Cross-checked against the transcripts, which carry independent per-turn
wall-clock: 1,312 analyzer turns, inter-turn gap p10/p50/p90 = **98 / 153 /
205 s**, and **1.021 LLM requests per analyzer turn**. A turn is one request
plus about four seconds.

---

## 2. Oversubscription: confirmed as queueing, refuted as a sequence-slot problem [VERIFIED]

If 25–28 threads contended for 8 sequence slots, `Running` would sit at 8. It
never does. From `vllm-openai-server.log`, across **792** scheduler snapshots
at a 10-second cadence covering the whole 7,920 s:

```
Running  min/p50/max        2 / 3 / 6      snapshots with Running >= 8:  0 of 792
Waiting  min/p50/max       18 / 22 / 22
KV usage min/p50/max     47.1% / 79.3% / 100.0%   (mean 80.3%)
mean Running 3.13   mean Waiting 21.62   preemptions 191 (14.3% of requests)
```

The scheduler's own capacity line says why, verbatim:

```
GPU KV cache size: 105,202 tokens, Maximum concurrency for 32,768 tokens per request: 3.21x
```

and the workload says how much each request needs:

```
mean prompt 20,175 tok + mean generation 1,433 tok = 21,608 tokens resident
105,202 / 21,608 = 4.87 requests fit
```

**So the constraint is KV-cache capacity, and `max_num_seqs = 8` is slack by a
factor of ~2.5.** Setting `TAAF_VLLM_MAX_NUM_SEQS` to 16, 28 or 40 changes a
limit that is not being reached. [VERIFIED for the mechanism; the sweep still
measures it directly, because "obviously a no-op" is exactly the kind of claim
this project has been wrong about before.]

The real oversubscription number is therefore **24.63 requests against ~3.13
effective slots = 7.87×**, not the 3.1–3.5× in the brief. And the queue-wait
share follows it exactly: `e2e / inference = 145.70 / 18.61 = 7.83`.

### 2.1 Why the KV pool is only 5 GiB, and why it cannot simply be doubled [VERIFIED]

```
Model loading took 81.8 GiB memory
Initial free memory 94.43 GiB, reserved 5.0 GiB memory for KV Cache as specified by
  kv_cache_memory_bytes config and skipped memory profiling
[AutoTuner] memory allocation failed with OOM on device 0 while trying to allocate
  511705088 bytes (free: 42336256, total: 101973950464)      <- at graph capture
  ... and once more mid-run at 16:16:03, wanting 2 MiB with 2.4 MiB free
```

81.8 GiB of weights plus a 5 GiB KV pool on a 95 GiB card leaves roughly 7.6 GiB
for activations, CUDA graphs, the MTP draft state and the fused-MoE autotuner
workspace — and **two real CUDA OOM events were logged**, one during graph
capture and one mid-run. [INFERRED] The upstream author's choice of 5 GiB looks
like a memory constraint rather than a tuning preference. How much headroom
actually exists is an empirical question, which is why the sweep tries 8 GiB and
12 GiB and treats a boot failure as a result rather than an error.

### 2.2 The hybrid-allocator detail that makes the fp8 lever non-obvious [VERIFIED]

```
Setting attention block size to 1600 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 0.25% to ensure that mamba page size and attention page size are exactly equal.
```

This is a hybrid GDN linear-attention model (`qwen_gdn_linear_attn.py` in the
log), so vLLM's hybrid allocator forces the attention page and the mamba page to
the same byte size. The pool is therefore ~65 pages of ~82.6 MB each
(5 GiB / (105,202 / 1600)), and a request costs ~13 attention pages plus its
mamba page ≈ 14 → **65 / 14 ≈ 4.6 resident requests**, matching the observed
`Running` of 3–6.

[INFERRED] Halving attention KV bytes via `--kv-cache-dtype fp8` should let
vLLM re-equalise at ~3,200 attention tokens per page, dropping a request to
~7 attention pages + 1 mamba page and roughly **1.7×**-ing residency at zero
extra GPU memory. That is a prediction, not a measurement — the mamba page cost
does not halve, and whether this model tolerates fp8 KV at all is untested. It
is the sweep's primary hypothesis.

### 2.3 Prefix caching is entirely off [VERIFIED]

`vllm:prefix_cache_queries_total = 0`, `vllm:prefix_cache_hits_total = 0`
(`TAAF_VLLM_ENABLE_PREFIX_CACHING=0`). Every one of the 1,339 requests
re-prefilled its ~20,175-token prompt from scratch, and turns within one game
share a very large prefix. The direct prefill saving would be small (prefill is
1.10% of a turn), but **shared blocks would cut KV footprint**, which is the
thing that actually binds. Also in the sweep.

### 2.4 MTP is working, and is not a suspect [VERIFIED]

688,180 drafts, 2,064,525 draft tokens, 1,231,695 accepted → **59.66%
acceptance, 1.79 of 3 accepted per draft**, i.e. 2.79 tokens per engine step.
This is the regime `stage7_duck_throughput.md` predicted would favour `mtp3`
(small batch, memory-bandwidth-bound), against the 45.6% our FP8 stack got at
concurrency 37. Leave it alone.

---

## 3. What could NOT be measured from existing artifacts, and why [VERIFIED]

The brief asked for the queue/prefill/decode split from existing artifacts
"or say so". The split **was** recoverable (section 1). The
**throughput-vs-batch curve was not**, and this is worth recording rather than
papering over, because it is the precondition for the whole lever.

`scripts/analyze_batch_scaling.py` buckets the 792 scheduler snapshots by
`Running`:

```
Running distribution: {2: 2, 3: 705, 4: 64, 5: 19, 6: 2}
89% of all snapshots sit at Running=3.
```

The batch barely varies — *because the KV pool pins it*. The off-mode buckets
are tiny (n = 2..19) and confounded with prefill bursts and preemption recovery;
the decode-dominated `Running=4` bucket even shows aggregate generation
*falling*, which no batching model predicts. Quoting a scaling number from this
log would be precisely the "confident, plausible-looking, entirely wrong table"
failure this project already paid three GPU runs for. It needs a controlled
sweep.

---

## 4. The controlled sweep (free kernel, pushed)

Kernel `calamitychasm/arc3-duck-nvfp4-kv-benchmark`, generated by
`scripts/_build_duck_kv_diag.py`. Cells 0/4/6/8/10 of the NVFP4 submission
notebook are copied **byte-identically** (the builder asserts it), so the
baseline server is the real `kv5-bf16-mtp3-c8-cg32` production configuration.

**Workload matched to production**, not to the older benchmarks: 25 concurrent
(public-25's game count), 50 requests through a semaphore so the pipe stays
full the way a closed loop does, ~20,175-token prompts (calibrated in-kernel
from a real `usage.prompt_tokens` report rather than a guessed chars/token
constant), 1,433 output tokens with `ignore_eos`. Known limitation, stated
rather than buried: production prompts are multimodal
(`MULTIMODAL_CONTEXT=current_grid`), these are text-only. KV footprint per token
is identical either way and prefill is 1.10% of a turn, so this affects the
vision-encoder cost and nothing else in this question.

**Configurations**, ordered most-informative-first so the deadline can only cut
the least important:

| # | name | change | question |
|---|---|---|---|
| 1 | `baseline` | none | reference |
| 2 | `seqs40` | `--max-num-seqs 40` **alone** | falsification test of the brief's hypothesis |
| 3 | `kvfp8-seqs40` | `--kv-cache-dtype fp8` + seqs 40 | more residency at zero GPU-memory cost |
| 4 | `kv8-seqs40` | KV pool 5 → 8 GiB + seqs 40 | does the card have room at all? |
| 5 | `kv8-fp8-seqs40` | both + seqs 40 | combined ceiling |
| 6 | `seqs16` | `--max-num-seqs 16` | fills the sequence-cap axis |
| 7 | `prefix-seqs40` | prefix caching ON + seqs 40 | block sharing across a game's common prefix |
| 8 | `kv12-seqs40` | KV pool 12 GiB + seqs 40 | how far does memory go before OOM |

Each row reports aggregate output tok/s, per-sequence tok/s, TTFT, **queue /
prefill / decode from the server's own histogram sums delta'd across the level**
(the identical quantities section 1 used, so rows are directly comparable to
production), mean `Running` / `Waiting`, KV pool size, preemptions, MTP
acceptance, and **turns per game**.

Three hazards were handled explicitly, each of which would otherwise have
produced a confident wrong table:

- **The bundle's own watchdog** polls `/v1/models` every 15 s and restarts the
  server after 4 failures (`vllm-watchdog-events.jsonl`, verified). Every config
  change takes the server down for minutes, so an active watchdog would
  relaunch the *original* argv underneath a mutated config. It is killed first,
  and the cell prints whether that worked.
- **Killing only the API-server pid leaves the workers alive.** This stack runs
  the engine core, GPU worker and PLE-offload worker as separate processes
  (`vllm-server-identity.json` lists pids 155/223/224/242/305/388 under one
  pgid), and the ~82 GiB allocation lives in the worker. `_stop_server` signals
  the whole process group and then waits on `nvidia-smi` free memory, so a later
  config cannot be misreported as "does not fit".
- **Argv mutation is unit-tested against the real baseline argv**
  (`scripts/_test_kv_diag_argv_mutation.py`), asserting no duplicated flags, no
  lost flags, and a correct prefix-caching flip. It caught a first `_set_flag`
  that silently *deleted* a boolean flag already present — which would have
  measured prefix caching OFF in the config whose entire point is turning it ON.

**Status at the time of writing: pushed, `QUEUED`.** Results are not in this
document. Nothing in sections 1, 2, 5 or 6 depends on them; section 6's
recommendation names the sweep as the gate.

---

## 5. Turns per game — the arithmetic, and a correction to the brief

The brief warns that "a configuration that doubles aggregate throughput but
halves per-sequence speed may deliver no extra turns per game". **That is not
how it works here, and the difference matters for how the sweep is read.**

A game is a serial loop, so it gets `T / turn_latency` turns; under saturation
Little's law gives `turn_latency = N_games / X`. Both reduce to the same
identity, which closes on the measured data [VERIFIED]:

```
turns_per_game = agg_generation_tok_s * T / (tokens_per_turn * N_games)
               = 242.1 * 7920 / (1463 * 25)
               = 52.4            (observed 52.5)

turns_per_game = X * T / N_games = 0.1691 * 7920 / 25 = 53.6
turn_latency   = N_games / X     = 25 / 0.1691       = 147.9 s  (measured e2e 145.7 s)
```

A turn's token count is a property of the harness — its prompt and its reasoning
budget — not of the server configuration. So **turns per game track aggregate
generation throughput and nothing else.** A configuration that doubles aggregate
throughput by running twice as many sequences at half the per-sequence rate
delivers exactly **twice** the turns: the halved per-sequence rate is already
inside the aggregate figure. Per-sequence latency would only matter if a single
turn could outgrow the analyzer timeout (900 s) or the per-game budget; at
145.7 s it is 6.2× under the former, and every configuration that raises server
capacity *lowers* turn latency rather than raising it.

| aggregate × | agg tok/s | turns/game | turn latency s |
|---:|---:|---:|---:|
| 1.00 | 242.1 | 52.4 | 145.7 |
| 1.25 | 302.6 | 65.5 | 116.6 |
| 1.50 | 363.1 | 78.6 | 97.1 |
| 2.00 | 484.2 | 104.9 | 72.9 |
| **2.55** | **617.3** | **133.7** | **57.1** |
| 3.00 | 726.2 | 157.3 | 48.6 |

**2.55× is the ceiling from residency alone** — mean `Running` 3.13 → the
configured `max_num_seqs` of 8 — *if* per-sequence decode speed survived that
batch increase unchanged. On a 512-expert MoE with 10 active experts it very
plausibly will not: expert-weight reads grow with batch in a way dense decode's
do not. That single number is what the sweep exists to pin down. [INFERRED]

For the real 110-game rerun the same identity applies with `N_games` = the
concurrency (28 today), giving `0.1691 × 7920 / 28 = 47.8` turns per game —
slightly fewer than public-25's 53.6, same structure, same lever.

---

## 6. RHAE upside

### 6.1 The formula had to be read from source, and CLAUDE.md's version is ambiguous in a sign-flipping way [VERIFIED]

CLAUDE.md records RHAE as

```
E_e = min( Sum_solved w_l / Sum_all w_n ,  Sum w_l*S_l / Sum w_l )
```

and flags it "high confidence, not confirmed". The unresolved bit —
**whether the efficiency term's denominator runs over solved levels or over all
levels** — decides whether one more slowly-won level raises the score or lowers
it, i.e. the entire sign of this investigation.

It is resolvable, and nobody had. The serving bundle ships the scorer that
produced this run's `score.json`:
`src/tufa-arc-agi-framework/src/taaf/game.py`,
`GameRun._compute_final_score`, whose own docstring says it mirrors
`arc_agi.scorecard.EnvironmentScoreCalculator` (v0.9.8). Verbatim:

```python
for level_idx in range(self.number_of_levels):        # ALL levels
    weight = level_idx + 1
    total_weights += weight
    completed = level_idx < self.levels_completed
    actions  = self.actions_per_level[level_idx] ...
    baseline = self.base_actions_per_level[level_idx]
    if completed and actions > 0:
        level_score = min(115.0, (baseline / actions) ** 2 * 100)
    else:
        level_score = 0.0
    if level_score > 0:
        max_weights += weight
    total_score += level_score * weight
score     = total_score / total_weights
max_score = max_weights / total_weights * 100
return min(score, max_score)
```

**The denominator is over ALL levels and is therefore fixed.** Three
consequences the CLAUDE.md form does not make visible:

- the efficiency term is **monotone non-decreasing in levels solved** — an extra
  level adds `w_l · S_l ≥ 0` to a constant denominator. **More turns can raise
  the score or leave it flat; it cannot lower it.**
- the completion cap counts only levels that actually *scored*, not merely
  levels completed;
- `min(115, …)` means a level solved faster than the human baseline is worth up
  to 1.15× a perfect one, which is why 16 of 25 games sit exactly on their
  completion cap.

**This is worth flagging back to CLAUDE.md**: a first draft of
`project_turns_rhae.py`, written to the solved-denominator reading, produced a
confident **−29% at 2× turns** — a clean, plausible-looking result with exactly
the wrong sign. It is documented in the script's own docstring so nobody
re-derives it.

Better still, **`benchmark.json` carries `base_actions_per_level` per game** —
the real human baselines — so `h_l` does not have to be inferred at all. The
scorer is reimplemented verbatim in `scripts/project_turns_rhae.py` and
**reproduces all 25 recorded `final_score` values bit-exactly** (max abs error 0.00e+00). Every projection
model reproduces the observed run exactly at ×1.0; that is the correctness test.

### 6.2 Which term binds, and where the headroom is [VERIFIED]

**Completion binds in 16 of 25 games, efficiency in 9.** Where completion binds,
an extra level pays in full. Where efficiency binds, an extra level still pays
(monotonicity above) but at the squared-efficiency discount.

The structural fact that makes this worth pursuing: **all 25 games hit the wall
with levels unsolved, and every one of them had actions already sunk into a
level it had not finished.** From `benchmark.json`'s `actions_per_level`, the
unfinished level had already absorbed e.g. `ar25` 348 actions, `s5i5` 201,
`sp80` 56, `r11l` 77. That work is currently worth exactly zero.

### 6.3 Projection [INFERRED from VERIFIED inputs]

Three bracketing models. All cost a not-yet-won level as
`max(ratio × baseline_l, actions_already_spent + 1)` — the human baseline gives
the *shape* of the cost curve (baselines vary 5–20× within one game: `m0r0` is
30, 111, 203, 26, 500, 237), and the second term is the observed-resistance
floor that keeps the model honest. They differ only in `ratio`:

- **optimistic** — ratio from won levels only. Too kind on a stuck game: it
  projects `r11l` (ratio 0.27, one level won in 6 actions) through five further
  levels at ~14 actions each while ignoring that level 2 has absorbed 77 and not
  fallen.
- **stuck-aware** — `max(that, actions_on_the_unfinished_level / its baseline)`.
  Current evidence of difficulty overrides historical efficiency. Headline.
- **+1 only** — stuck-aware costs, but at most one further level per game.
  A deliberate floor: credits no level the agent was not already working on.

Games that completed zero levels stay at zero under every model.

| × turns | optimistic | stuck-aware | +1 only | levels (stuck-aware) |
|---:|---:|---:|---:|---:|
| 1.00 | 10.689 (+0.0%) | 10.689 (+0.0%) | 10.689 (+0.0%) | 48 |
| 1.25 | 16.830 (+57.5%) | **14.595 (+36.5%)** | 14.595 (+36.5%) | 61 |
| 1.50 | 24.578 (+129.9%) | **18.741 (+75.3%)** | 15.879 (+48.6%) | 69 |
| 2.00 | 37.690 (+252.6%) | **24.784 (+131.9%)** | 17.701 (+65.6%) | 84 |
| 2.55 | 45.096 (+321.9%) | **30.647 (+186.7%)** | 17.701 (+65.6%) | 97 |
| 3.00 | 51.031 (+377.4%) | **34.042 (+218.5%)** | 17.701 (+65.6%) | 106 |

`+1 only` saturates at ×2.0 — by then every game that can finish its in-progress
level has.

**The floor is the number to hold onto: even crediting nothing but the levels
already in progress, 2× turns is +65.6% on the public-25 mean.** That is two
orders of magnitude more than the ≤2.12% the analyzer-timeout investigation
found next door, and unlike that one it is monotone by construction rather than
by assumption.

### 6.4 Translating to the hidden set — where this must NOT be oversold [INFERRED]

Our public-25-to-hidden ratio on this stack has been ~0.27 (10.69 local →
2.84/2.95 real). Applying the *relative* deltas at 2× turns gives
**~4.8 (floor) to ~10.3 (optimistic)**, headline ~6.8, against a top-10% bar of
3.31.

**Do not believe those numbers as stated**, for three specific reasons:

1. **The local→hidden relationship is demonstrably non-linear.** Our FP8 stack
   went 3.37 local → 2.57 real (ratio 0.76); the NVFP4 stack goes 10.69 → ~2.9
   (ratio 0.27). A ratio that falls this fast as the local score rises will not
   hold a 2.4× relative gain intact.
2. **Extra turns help most where a game is already making progress.** Public-25
   has 24 of 25 games with ≥1 level won and a level in progress. On 110 largely
   novel hidden games that fraction is certainly lower, and games at zero levels
   get nothing under every model here.
3. **A projected ~7 would be roughly second place globally** (#1 is 11.04). A
   serving change producing that is not credible on its face, which is itself
   evidence the proportional extrapolation is too generous.

The defensible statement is the *direction and the floor*: **the upside is
robustly positive, monotone by construction, and large enough that even a
heavily discounted transfer (say a quarter of the relative gain surviving)
clears the 3.31 bar from ~2.9.** That is a materially different conclusion from
the analyzer-timeout result, and it is the reason to spend engineering here.

---

## 7. Recommendation

**Pursue turn latency. Do not pursue it via `TAAF_VLLM_MAX_NUM_SEQS`.**

1. **Do not ship a `MAX_NUM_SEQS` change.** [VERIFIED] The scheduler never
   reaches 8 in 792 snapshots. Raising the cap changes a limit that is not
   binding. `seqs40` is in the sweep purely as the falsification test.
2. **The lever is KV-pool residency**, in this order of expected
   value-per-risk:
   - `TAAF_VLLM_KV_CACHE_DTYPE=fp8` — costs zero GPU memory, predicted ~1.7×
     residency. Must be validated for output quality, not just speed, because
     nothing here tests accuracy.
   - `TAAF_VLLM_KV_CACHE_MEMORY_BYTES` 5 → 8 GiB — needs headroom the card may
     not have (two logged OOMs). The sweep answers this by booting it.
   - `TAAF_VLLM_ENABLE_PREFIX_CACHING=1` — currently 0 queries, 0 hits; shares
     blocks across a game's large common prefix. Upstream disabled it, plausibly
     because prefix caching on hybrid Mamba/GDN models is flagged experimental
     by vLLM itself, so treat a win here with suspicion until output quality is
     checked.
   - Whatever KV change wins must be paired with a `MAX_NUM_SEQS` raise, or the
     cap that is slack today becomes the new binding constraint at ~8.
3. **Free validation path, no quota:** the sweep already running reports
   turns-per-game per configuration directly. If a configuration shows a real
   turns-per-game gain, the next step is **another free public-25 run** on the
   winning profile — that measures the end-to-end quantity (mean self-eval score
   and levels completed on the same 25 games) against a 10.69 baseline with no
   modelling in between, and it also catches any output-quality regression from
   fp8 KV or prefix caching, which a throughput benchmark structurally cannot.
   Only after that is a submission slot warranted.
4. **Do not touch MTP.** 59.66% acceptance at 1.79/3 tokens per draft is the
   regime `mtp3` was chosen for. Note the sweep raises batch size, which is the
   axis MTP's benefit falls off along (our FP8 stack got 45.6% at batch 37) — so
   MTP acceptance is reported per configuration, and a large drop there is a
   reason to re-test `mtp2`/`mtp1` at the new operating point rather than a
   reason to abandon the KV change.

### What would falsify this

A single number: if `kvfp8-seqs40` and `kv8-seqs40` both come back with
turns-per-game within noise of `baseline` — i.e. residency rose but aggregate
generation throughput did not, because decode was already compute-bound at
batch 3 — then the 2.55× ceiling in section 5 is not reachable and this whole
line closes. That is a real possibility on a 512-expert MoE and is exactly why
section 3 refused to estimate it from the production log.
