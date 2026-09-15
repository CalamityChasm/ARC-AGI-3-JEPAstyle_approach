# Stage 7 — Model search: is there a better model than Qwen3.8-Flash-Next for this kernel?

Date: 2026-09-14/15. Branch: `stage7-model-search`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (a run log, a
real API response, notebook/bundle source, or a locally-run computation over
them); **[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made.** Everything here is offline analysis of
already-completed runs, read-only API calls, and one free `kaggle kernels push`.
The daily submission slot was untouched.

Reproduce every number with:

```
venv/Scripts/python.exe scripts/analyze_model_tradeoff.py \
    fp8=<dir> nvfp4=<dir> ctx16k=<dir> dedupe=<dir> \
    --json experiments/stage7_model_search_runs.json
venv/Scripts/python.exe scripts/analyze_leaderboard_maxstat.py \
    experiments/stage7_leaderboard_snapshot.csv \
    --anchors wuliao0,keithtyser,akhileshgodugu,tantan0327,dantelok,woguoat,calamitychasm
venv/Scripts/python.exe scripts/size_kaggle_models.py --json experiments/stage7_model_sizes.json
venv/Scripts/python.exe scripts/_build_duck_nvfp4_anim.py <pulled-src-dir> \
    --out kaggle_submission_duck_nvfp4_anim/notebook
```

`<dir>` is an unpacked `kaggle kernels output` directory for, respectively,
`lb-9-arc3-duck-v12-with-qwen-3-8-27b`, `arc3-duck-nvfp4-baseline`,
`arc3-duck-nvfp4-ctx16k` and `arc3-duck-nvfp4-dedupe`.

---

## TL;DR

1. **The FP8-large-KV lead is dead, and not because it is unpromising — because
   it has already been run.** The premise (FP8 pinned KV to 5 GiB, leaving
   ~55 GiB unused) is false in both halves. Our FP8 stack got **44.57 GiB /
   342,144 KV tokens**, ran at **Waiting p50 = 0, Running p50 = 25** — every
   game resident, zero queueing, 3.25x the NVFP4 pool — and delivered **0.45x
   the actions and 0.32x the score.** [VERIFIED, §1]
2. **Residency does not buy turns on this harness, and there are now three
   independent runs proving it.** Turns obey
   `agg_gen_tok_s x T / (tokens_per_action x N_games)` to within 0.1% on every
   run — but `tokens_per_action` **varies 3.5x** (515 → 1,782) and it cancels
   every throughput gain. [VERIFIED, §2. Whether that variation is a property of
   the model or of the solver is *not* settled by these four runs — the fp8 row
   changes both at once, §2.0 — and separating them is the second reason to run
   §6.]
3. **The ctx16k result, left blank in `stage7_context_budget.md`, is recovered:
   score 2.01, actions 3,435.** Projected 1.93x turns; delivered **0.95x**.
   That is a sixth dead lever and, more importantly, a direct falsification of
   the residency→turns model the whole throughput line was built on. [VERIFIED]
4. **A model swap on the winning stack is impossible.** The NVFP4 serving bundle
   is a sealed appliance: it verifies the checkpoint manifest hash, repo,
   revision, file count (419), byte total, **every one of 419 per-file SHA-256s**,
   `config.json`, the quant config, the vLLM runtime layers, the PLE patch's
   pre/post target hashes — and its own SHA-256. [VERIFIED, §3]
5. **Nothing mountable is better at the capability that matters.** Of 409
   enumerated Kaggle models, the ones that fit 95 GiB are all *weaker* coders
   than the incumbent (LiveCodeBench 91.9 vs 81.19 / 78.93); the only model with
   better published agentic numbers, GLM-5.3-Flash, is **184.29 GiB at NVFP4**
   and does not fit. [VERIFIED sizes / mixed-quality capability evidence, §4]
6. **The real finding is not a model at all.** Our score is *average* for this
   stack, not broken — our n=3 is 2.84 / 2.95 / 2.42 (mean **2.74**, sd 0.28) —
   and the leaderboard is substantially a max-of-n statistic. But the one
   degree of freedom the seal leaves open is the **solver**, and a public graft
   (`thui-animfast-b71`) that swaps in the *anim* solver is used by teams at
   3.49 and 3.74. That is the recommendation, and it is free-run validated
   below. [§5, §6]

---

## 1. The lead: verified false, from the FP8 run's own server log

The brief's premise: *"the FP8 stack's weights are ~30.9 GiB, and its config ALSO
set `kv_cache_memory_bytes = 5368709120` (5 GiB) — apparently leaving ~55 GiB of
the card unused."* Both halves fail.

**Half one — those variables are inert on our stack.** Already established in
`stage7_duck_throughput.md` §2.1 and re-confirmed here: `TAAF_VLLM_*` is read
**only** by the NVFP4 fork's own `serving_setup.py`. Our bundle
(`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`) contains **zero**
references. The only place `5368709120` appears in our codebase is the
`fork-profile` entry in `kaggle_submission_duck/vllm_serving.py` — a *benchmark*
profile that never shipped. The profile that ships is `mtp1+flags`.

**Half two — the real argv, printed by the run itself** [VERIFIED,
`vllm-openai-server.log` of `lb-9-arc3-duck-v12-with-qwen-3-8-27b`]:

```
serving profile= mtp1+flags argv= ... --max-model-len 65536
  --speculative-config {"method":"mtp","num_speculative_tokens":1}
  --async-scheduling --no-enable-prefix-caching --kv-cache-dtype fp8
```

No `--kv-cache-memory-bytes`, no `--max-num-seqs`. vLLM profiled the card and
took what was left:

```
Model loading took 28.95 GiB memory and 99.66 seconds
Available KV cache memory: 44.57 GiB
GPU KV cache size: 342,144 tokens
Maximum concurrency for 65,536 tokens per request: 18.06x
```

**The large-KV small-model configuration is not untested. It is what our 2.57
submission ran.** And it is not merely "large KV" — it is *saturated*: across all
795 scheduler snapshots, `Running` p50 = **25** (all 25 games resident at once),
`Waiting` p50 = **0** (763 of 795 snapshots had an empty queue), KV usage 62%,
and the metrics scrape records **zero** preemptions.

Against the NVFP4 stack, which the same script reads from its own artifacts:

| | FP8 (our 2.57 stack) | NVFP4 (current, 2.74 mean) |
|---|---:|---:|
| weights resident | **28.95 GiB** | 81.80 GiB |
| KV pool | **44.57 GiB / 342,144 tok** | 5.00 GiB / 105,202 tok |
| `--max-model-len` | 65,536 | 32,768 |
| `Running` p50 | **25** | 3 |
| `Waiting` p50 | **0** | 22 |
| preemptions | 0 (log) | 191 |
| aggregate generation | **365.4 tok/s** | 236.2 tok/s |
| **actions** | **1,629** | **3,633** |
| generated tokens / action | **1,782** | **515** |
| **public-25 score** | **3.37** | **10.69** |

The small model has 3.25x the KV pool, 8x the residency, no queue at all, and
1.55x the raw token throughput — and it produced **45% of the turns and 32% of
the score.** The trade the brief wanted measured has been measured, and the
smaller model lost on **both** axes simultaneously, not just on capability.

---

## 2. Why: `tokens_per_action` is a model property, and it eats the throughput

`stage7_turn_latency.md` §5 derives
`turns_per_game = agg_gen_tok_s x T / (tokens_per_turn x N_games)` and notes that
turns therefore "track aggregate generation throughput and nothing else." The
identity is right; the conclusion drawn from it is not, because it silently
treats `tokens_per_turn` as a constant of the harness. It is not.

`scripts/analyze_model_tradeoff.py` closes the identity on every run:

| run | agg gen tok/s | tok/action | turns/game predicted | observed | residual |
|---|---:|---:|---:|---:|---:|
| fp8 | 365.4 | 1,782 | 65.2 | 65.2 | −0.07% |
| nvfp4 | 236.2 | 515 | 145.3 | 145.3 | +0.01% |
| ctx16k | 273.9 | 632 | 137.4 | 137.4 | +0.02% |
| dedupe | 228.3 | 959 | 75.4 | 75.4 | −0.01% |

Read the middle column. Across four real runs on the same hardware and the same
25 games, **generated tokens per action spans 515 → 1,782, a factor of 3.5** —
larger than any throughput factor any lever in this whole investigation has
moved. Every configuration that traded capability or context for residency paid
for it here:

- **FP8 (smaller model):** +55% throughput, +246% tokens/action → **0.45x turns.**
- **ctx16k (half the context):** +16% throughput, +23% tokens/action → **0.95x
  turns**, against a projected **1.93x**.
- **dedupe (27% smaller prompt):** −3% throughput, +86% tokens/action → **0.52x
  turns.**

This is the general form of the two results `stage7_context_budget.md` recorded
one at a time. It is not "prompt content cannot be treated as a free variable";
it is stronger and more useful: **on this harness, anything that makes the model
less able to decide makes it generate more tokens before deciding, and the extra
tokens cost more than the freed memory buys.** The residency→turns model is
falsified, in the same direction, on three independent interventions.

### 2.0 A confound in the fp8 row, stated rather than buried

The ctx16k and dedupe rows are clean: same model, same solver, one variable.
**The fp8 row is not.** It differs from the NVFP4 baseline in *two* ways, not
one — the model, and the solver. Our FP8 kernel mounts
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (the **anim** solver); the
NVFP4 baseline runs the June duck solver that ships inside Keith Tyser's bundle.
So "1,782 vs 515 generated tokens per action" is a model-plus-solver difference,
and attributing all of it to the model would be exactly the kind of unearned
attribution this project has been burned by.

What survives the confound unharmed is the §1 conclusion, because it does not
need attribution: *some* configuration with 8x the residency and zero queueing
produced 45% of the turns. Whatever the cause, freed memory did not become
turns. What does *not* survive is any claim about which of the two changes
caused it.

The 2x2 is one cell short, and that cell is the run in §6:

| | June duck solver | anim solver |
|---|---|---|
| **Qwen3.8-27B FP8** | never run | 3.37, 1,629 actions, 1,782 tok/action |
| **Qwen3.8-Flash-Next NVFP4** | 10.69, 3,633 actions, 515 tok/action | ← `arc3-duck-nvfp4-anim` |

Filling it in separates a model property from a solver property, which is the
second reason to run it and was not the reason it was chosen.

That also retires the brief's own framing for the model search — *"a model that
is half the size can be substantially weaker and still win on total levels
solved, because turns scale linearly"*. Turns do not scale linearly with freed
memory. On the one occasion this was actually tried, halving the model **halved
the turns as well as the score**.

### 2.1 The ctx16k result, recovered

`stage7_context_budget.md`'s results section is headed "both candidates
regressed" but tabulates only the dedupe run. Pulling
`calamitychasm/arc3-duck-nvfp4-ctx16k`'s output [VERIFIED]: **mean score 2.01,
total actions 3,435, 2,169,265 generated tokens**, against the baseline's 10.69 /
3,633 / 1,870,896. Score **−81.2%**, actions **−5.5%**. The candidate's own
projection was 1.93x turns. It delivered 0.95x. Recorded here so the next
session does not re-derive it — and because it is the single cleanest
falsification of the residency model in the whole line.

---

## 3. The feasibility gate: the winning stack cannot serve any other model

`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`'s `serving_setup.py` (3,109
lines) is a cryptographically sealed appliance. Verbatim excerpts archived at
`experiments/stage7_model_search_artifacts/nvfp4_bundle_model_pin.txt`
[VERIFIED — read out of the downloaded bundle]:

```python
MODEL_HF_REPO            = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MODEL_HF_REVISION        = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
MODEL_MANIFEST_SHA256    = "a09bdad3fe...dd9a5b"
MODEL_CONFIG_SHA256      = "e765305dab...a994d624"
MODEL_QUANT_CONFIG_SHA256= "7e69ef4b94...721b98e8"
MODEL_FILE_COUNT         = 419
MODEL_TOTAL_BYTES        = 135_253_622_894
```

and the gate that enforces them: manifest hash (`:665`), repo+revision (`:670`),
file count (`:673`), byte total (`:679`), **the SHA-256 of every individual model
file** (`:695`), `config.json` and the quant config (`:716`, `:718`), the vLLM
runtime image's layer digests (`:848`–`:891`), the PLE patch's stock and patched
target hashes (`:1020`, `:1036`) — and, at `:603`, **the SHA-256 of
`serving_setup.py` itself**, so the gate cannot be edited out without
re-deriving `SOURCE_IDENTITY.json`.

**Consequence: every candidate model in §4 is only servable on our own FP8-era
stack** (stock vLLM 0.19.0 from `driessmit1/arc3-vllm-h100-wheelhouse-v3`, a
hardcoded argv the notebook already rewrites) — the stack measured in §1 at
3.37 local / 2.57 real against the NVFP4's 10.69 / 2.74. Any new model starts
~0.8 real points in the hole and has to make that back before it is even even.

---

## 4. The candidate table

409 distinct Kaggle Models were enumerated across 16 search terms; every
identifier below returned **HTTP 200 on `models/get`** for our account
[VERIFIED]. Sizes are the summed, **paged** per-instance file listing
(`scripts/size_kaggle_models.py`), calibrated against two known ground truths:
it returns 125.96 GiB for the NVFP4 checkpoint — exactly the bundle's own
`MODEL_TOTAL_BYTES` of 135,253,622,894 — and 28.77 GiB for the FP8 one, against
28.95 GiB resident in its run log.

**Memory arithmetic.** The card reports 97,887 MiB (95.6 GiB) with 94.43 GiB
free at init. The NVFP4 run sits at 81.8 weights + 5.0 KV = 86.8 GiB and *still*
logged two CUDA OOMs; the FP8 run at 28.95 + 44.57 = 73.5 GiB logged none. Take
**weights + KV ≤ ~87 GiB** as the hard ceiling. Resident ≈ on-disk for every row
except the incumbent, whose 125.96 GiB is served from a 95 GiB card only because
the sealed bundle CPU-offloads its FP8 PLE embeddings — an offload path that
exists for that one checkpoint and cannot be pointed at another.

| model (mount) | GiB | resident | KV left | fits? | coding capability |
|---|---:|---:|---:|:--:|---|
| **`keithtyser/qwen3-8-flash-next-nvfp4` — INCUMBENT** Qwen3.8-Flash-Next, 125B MoE / 6B active, NVFP4 | 125.96 | **81.8** (PLE offload) | 5.0 | via sealed bundle only | **LCB v6 91.9, SWE-bench Pro 62.5, SWE-bench Multilingual 81.0** |
| `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` — PRIOR, dense FP8 | 28.77 | 28.95 | **44.57** | yes (measured) | measured: 3.37 local / 2.57 real |
| `impactganyu/qwen38-27b-radixark-nvfp4` — Qwen3.8-27B dense, NVFP4 | 20.44 | ~20 | ~60 | yes | ≤ the FP8 27B above (same model, lower precision) |
| `michaelpoluektov/qwen3-8-27b-nvfp4` | 24.59 | ~25 | ~55 | yes | as above |
| `cryptozenith/qwen-27b-nvfp4` | 23.45 | ~23 | ~57 | yes | as above |
| `michaelpoluektov/qwen3-6-35b-a3b-nvfp4` — Qwen3.6-35B-A3B | 21.85 | ~22 | ~58 | yes | older generation |
| `google/gemma-4` `other/gemma-4-31b-it-qat-w4a16-ct/2` | 21.70 | ~22 | ~58 | yes | LB 3.24 (`ko0kip`), but on a *different* vLLM (`ko0kip/vllm-0230-offline`) |
| `qwen-lm/qwen-3-5` `qwen3.5-27b` (bf16) | 51.77 | ~52 | ~28 | yes | older generation |
| `kekshibata/qwen3-next-80b-awq-4bit` — Qwen3-Next-80B-A3B | 45.89 | ~46 | ~34 | yes | older generation |
| `konstantinboyko/...-80b-a3b-thinking-awq-4bit-cpatonn` | 45.89 | ~46 | ~34 | yes | older generation |
| `barnobarno/nvidia-nemotron-3-super-120b-a12b-nvfp4` | 74.85 | ~75 | ~5 | marginal | **LCB 81.19** — below the incumbent's 91.9 |
| `qwen-lm/qwen3-coder-next` `-fp8` | 74.89 | ~75 | ~5 | marginal | coder-specialised but an older generation |
| `surasan092/qwen3-5-122b-a10b-nvfp4` | 77.21 | ~77 | ~3 | marginal | **LCB 78.93** — below the incumbent |
| `pranshubahadur/deepseek-v3.2-reap-...-w4a16` | 94.62 | — | **negative** | **no** | — |
| `woochangsim/qwen38-flash-next-w4a16-autoround-...` | 168.79 | — | negative | **no** | same model, but 2.1x the NVFP4 footprint |
| `ram2121/qwen3-8-flash-next-gptq-4bit` | 174.84 | — | negative | **no** | as above |
| `russcore/glm53-flash-nvfp4-redhatai-...` — GLM-5.3-Flash, 320B / 18B active | 184.29 | — | negative | **no** | **the one model with better published agentic numbers** (78.75% vs 70% on an 8-question independent suite) |
| `qwen-lm/qwen-3-5` `qwen3.5-397b-a17b-fp8` | 378.31 | — | negative | **no** | — |

**Verdict on the model question: the incumbent is the best model available to
this kernel, by the metric Tufa says matters.** Qwen3.8-Flash-Next leads every
mountable alternative on LiveCodeBench (91.9 vs 81.19 and 78.93) and on
SWE-bench Pro (62.5, ahead of Claude Opus 4.6 Max's 53.4). The two rows that
could plausibly have beaten it are both excluded by arithmetic, not by
preference: GLM-5.3-Flash needs 184.29 GiB, and the Flash-Next re-quantisations
that would dodge the sealed bundle are *larger* than the NVFP4 build, not
smaller.

Capability numbers in this table are **[INFERRED from secondary sources]** —
vendor-reported benchmarks and one independent 8-question suite, relayed through
model-review sites. They are directionally consistent across sources and the
margin is wide, but they are not measurements we made.

---

## 5. What the leaderboard actually says — and what our 2.95 actually is

### 5.1 We are not misconfigured; we are average, with n=3

From the competition submissions API [VERIFIED], our runs on the **unmodified**
NVFP4 baseline (`arc3-duck-nvfp4-baseline`, one unchanged kernel version):

| date | ref | score |
|---|---|---:|
| 2026-09-11 | 56153820 | 2.84 |
| 2026-09-12 | 56174348 | 2.95 |
| 2026-09-14 | 56233759 | **2.42** |
| 2026-09-15 | 56243996 | pending |

**Mean 2.74, sd 0.28.** The headline "2.95" is a max-of-3, not a typical run.

### 5.2 The stack we run is byte-identical to the ones scoring 4.33

Four public forks pulled and compared [VERIFIED]:

| kernel | notebook md5 | mounts | LB best | subs |
|---|---|---|---:|---:|
| `wuliao0/duck-qwen3-8-anim-base` | `5025d1d3…b1870c` | identical | **4.33** | 62 |
| `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` | `5025d1d3…b1870c` | identical | 4.17 | 82 |
| `akhileshgodugu/…-aacb81` | `5025d1d3…b1870c` | identical | 4.09 | 54 |
| `tantan0327/arc3-flashnext-asis` | `5025d1d3…b1870c` | identical | 3.92 | 48 |
| **ours** (`arc3-duck-nvfp4-baseline`) | same + 1 probe cell | identical | **2.95** | 3 on this stack |

A full diff of our notebook against `wuliao0`'s shows **one** difference: our
added print-only hardware probe. Metadata differs only in `id`, `title`,
`id_no`, `code_file` and `is_private`; `model_sources`, `dataset_sources`,
`competition_sources`, `docker_image` and `machine_shape` are identical.
**There is no configuration defect to find.**

### 5.3 The leaderboard is substantially a max-of-n statistic

`scripts/analyze_leaderboard_maxstat.py` on the 3,049-team snapshot [VERIFIED]:

- Spearman(submission count, best score) = **+0.591** over all teams, **+0.355**
  restricted to teams scoring ≥ 2.0.
- Median best score by submission bucket: **0.12** (n=1) → 0.21 (2–3) → 0.30
  (4–7) → 0.90 (8–15) → 1.51 (16–31) → **3.21** (32–63) → **3.45** (64+).
- Within the identical-code control group above, Spearman = **+0.857**, and that
  value is pure resampling — the code cannot differ.

Our own position: **rank 472 / 3,049**, score 2.95. Top-10% is now **3.39**
(rank 305), top-5% is 3.79.

**[INFERRED] How much of the gap is resampling.** Fitting
`E[max of n] ≈ mu + sigma·Φ⁻¹(n/(n+1))` to the control group gives roughly
mu ≈ 2.85, sigma ≈ 0.58 — and our three observed runs (2.84 / 2.95 / 2.42,
mean 2.74) land essentially on that mu, which is a real if modest corroboration.
But our *own* measured sd is 0.28, not 0.58, and at sd 0.28 the best of 62 draws
is only ≈ 3.31 — it cannot reach 4.33. So resampling explains a large part of the
spread and **not all of it**. The most likely remainder, and it is not
flattering to the simple story: these teams' *best* scores need not come from the
public byte-identical notebook at all. `keithtyser` has 82 submissions across
several kernels; `wuliao0` 62. The public fork is the baseline they published,
not necessarily the configuration that scored their best.

The falsifiable version of that claim is already running: see §6.

**Practical consequence either way.** At our measured mu = 2.74 / sd = 0.28,
P(a single unchanged run ≥ 3.39) ≈ 1.5%. **Resubmitting the unchanged stack is
not a path to top-10%.** That is worth stating plainly, because §5.1–5.3 could
easily be misread as "just roll the dice more".

---

## 6. The one degree of freedom the seal leaves open — and the recommendation

The bundle seals the **model**. It does not seal the **solver**: the notebook
puts that on `sys.path` from a separate, freely-chosen dataset mount.

And `stage7_duck_nvfp4.md` already flagged the relevant confound, and explicitly
declined to wave it away:

> the bundle's `git_status.txt` pins a *different* solver revision from our own
> FP8 fork's bundle: `ARC3-Inference aa69123 (DIRTY, branch
> add-kaggle-share-flag)` … where ours pins `9158303` on
> `feature/animation-awareness`. … So "the solver is unchanged" is true relative
> to *its own* Tufa baseline, but this is not literally the same solver snapshot
> our 2.57 ran. That is an uncontrolled difference between the two stacks … and
> it should not be waved away.

**It was never resolved, and someone else has already built the experiment.**
`yocybercode/thui-animfast-b71-full25-r1` (public, 10 votes; same source as
`sahasawatt/thui-animfast-v1`) is Keith Tyser's NVFP4 serving stack, unmodified,
with the **anim** solver — `jakobbrggen/taaf-kaggle-source-anim-20260807-anim`,
the very bundle our FP8 stack runs — grafted over the June duck. Diffed against
the plain fork, 8 of 18 cells differ, and the substantive changes are
[VERIFIED, from the pulled notebooks]:

| cell | change |
|---|---|
| 7 | attaches the anim bundle as a third dataset; resolves each bundle by its `benchmark_label` (both carry the marker file, so "first marker wins" would be a coin flip) |
| 9 | drops `ARC3-Inference` + `tufa-arc-agi-framework` from *his* `sys.path` entries and substitutes the anim tree; sets `LOCAL_ANALYZER_SEED=20260825` and **`LOCAL_ANALYZER_YIELD_SECONDS=180`** (the bundle persists 60) *after* serving setup and *before* the solver import, since `tool_agent` reads them at import time |
| 11 | takes `deploy_target.pkl` and `benchmark_initial.pkl` from the **anim** bundle; asserts the solver carries `animation_awareness=True` and `hard_noop_guard=True` |
| 3 | full diagnostics on an interactive run, minimal in a real rerun |
| 5, 15 | resolves the competition mount instead of hardcoding it (Kaggle serves two layouts) |

Teams running this graft: **`dantelok` 3.49 on 3 submissions** — which at our
measured mu/sigma is a z of 2.7, i.e. ~1% likely from the plain fork's
distribution — and **Thuitanium (`yocybercode`/`sahasawatt`) 3.74**. The
3-submission data point is the interesting one, because it is the one that
resampling cannot explain away.

### 6.1 What we built

`kaggle_submission_duck_nvfp4_anim/`, generated by
`scripts/_build_duck_nvfp4_anim.py`, which:

- asserts the source notebook's md5 (`87bd2660…65859d`), so a silent upstream
  re-push cannot change what we think we measured;
- copies **all 8 code cells byte-for-byte and asserts each one** — the
  experiment is worthless if any drifts;
- replaces only the two leading markdown cells, which are that team's own
  provenance narrative and would read as our claims under our kernel id, with an
  attribution header;
- prepends the same print-only RTX-PRO-6000 probe our NVFP4 baseline carries;
- carries every mount, the docker image and `machine_shape` over untouched.

Attribution for all four upstreams — Tufa Labs, Jakob Brüggen, Keith Tyser /
wuliao_0, Thuitanium / Knowless Crew — is in
`kaggle_submission_duck_nvfp4_anim/THIRD_PARTY_NOTICE.md`. **No score any of them
reports is claimed as ours.**

### 6.2 Free-run result

Kernel `calamitychasm/arc3-duck-nvfp4-anim` v1, pushed free (no submission
quota), public-25 path. Baseline to beat: **10.69 mean, 3,633 actions.**

<!-- RESULT: filled in below when the run completes -->

*Pending at the time of writing.* Read the **last** progressive summary block,
never the first — the harness emits one roughly every 10 games' worth of
progress and the first reads 0.80 (see `stage7_duck_nvfp4.md`, "A reporting trap
worth recording"). Cross-check by recomputing from the per-game `[finished]`
lines. Report **score and actions together**: §2 is the reason a score change
without an actions change means something completely different from a score
change with one.

---

## 7. Ranked recommendation

1. **`arc3-duck-nvfp4-anim` (the anim-solver graft) — the only candidate worth a
   slot, conditional on its free run.** It is the one untested degree of freedom
   on the best stack, it resolves a confound our own documentation flagged and
   left open, and two teams using it sit above the top-10% bar, one of them on
   three submissions. Gate on the free run: ship it if public-25 ≥ ~10.69 with
   actions not collapsed; do **not** ship on a score gain that comes with an
   actions collapse, because §2 shows that pattern is how a capability
   regression disguises itself.
2. **Do not swap the model.** Not because nothing is better in principle, but
   for two independent reasons that each suffice: the winning stack is
   cryptographically sealed to one checkpoint (§3), and of everything that fits
   95 GiB, nothing beats the incumbent on coding capability (§4).
3. **Do not revisit the FP8 stack or any "smaller model buys residency" variant.**
   Measured: 8x the residency, zero queueing, 45% of the turns, 32% of the score
   (§1). This is now the seventh dead throughput lever and the only one that was
   dead *before* anyone thought to look.
4. **Resubmitting the unchanged baseline is worth roughly 1.5% per slot** at our
   measured mu = 2.74 / sd = 0.28 (§5.3). Reasonable as a default when nothing
   better is ready; not a strategy.
5. **If a future session does want a model swap**, the only tractable host is our
   own stock-vLLM FP8 stack, and the *first* thing to check — cheaply, on a free
   push — is whether vLLM 0.19.0 in `driessmit1/arc3-vllm-h100-wheelhouse-v3`
   supports the target architecture and its quantisation at all. Every 4-bit row
   in §4 assumes `modelopt_fp4` / AWQ / W4A16 support that has never been
   verified on that wheelhouse, and discovering it is absent after building a
   notebook would be the expensive order to find out.

---

## 8. Corrections to earlier Stage 7 documents

- `stage7_kv_residency_levers.md` closes with "**A smaller model** — frees memory
  for KV, buys residency, but costs capability … A direct trade against the thing
  that actually drives score." The trade is real but the sign on the *first* half
  is wrong: on the one run where it was tried, the smaller model bought **fewer
  turns**, not more (§1, §2).
- `stage7_turn_latency.md` §5: "turns per game track aggregate generation
  throughput and nothing else." Only true at fixed `tokens_per_turn`, which
  varies 3.5x across our four runs and dominates (§2).
- `stage7_context_budget.md` §7: the results table omits the ctx16k run.
  It scored **2.01** with **3,435** actions (§2.1).
- The submission ledger in CLAUDE.md predates the 2026-09-14 `2.42` sample; our
  n on the unmodified NVFP4 baseline is 3, not 2, and the mean is 2.74 (§5.1).
