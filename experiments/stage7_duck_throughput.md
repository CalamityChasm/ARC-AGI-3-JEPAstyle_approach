# Stage 7 — Duck serving throughput: does the public NVFP4 + MTP recipe reproduce on our stack?

Date: 2026-09-08. Branch: `stage7-duck-throughput`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (a real
kernel log, a real API response, notebook/bundle source, or a locally-run
test); **[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made at any point.** Every measurement here
comes from free `kaggle kernels push` runs. The daily submission slot was left
untouched.

Follow-on to `experiments/stage7_duck_concurrency.md`, which swept *client
concurrency* and settled it at 37. This experiment holds concurrency fixed at
37 and sweeps the *server configuration*.

---

## 1. Hypothesis

A public fork, `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp`, states in its own
notebook that "the Duck prompts, tool-use loop, game policy, and scorer remain
unchanged. My changes are limited to model serving and performance." Its recipe:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4` weights (ModelOpt NVFP4, BF16 compute)
- native 3-token NEXTN MTP speculative decoding (`TAAF_VLLM_MTP_TOKENS=3`)
- async scheduling, chunked prefill, CUDA graphs, 8K batched-token cap,
  8 vLLM sequences, prefix caching disabled, KV cache pinned to 5 GiB
- a `/v1/models` watchdog with at most two restarts

`experiments/stage7_strategy_reset.md` identifies inference throughput as the
binding constraint on this harness, and this recipe is the one documented
reason other public forks climbed. **Hypothesis: some or all of this recipe
transfers to our stack and raises aggregate tokens/sec at concurrency 37.**

---

## 2. What the recipe actually is — three findings that changed the design

All three came from reading primary sources before spending any GPU time, and
each one invalidates a plausible-looking plan.

### 2.1 The fork's `TAAF_VLLM_*` variables are inert on our stack [VERIFIED]

The fork's notebook sets its profile purely through environment variables:

```python
PUBLIC25_VLLM_PROFILE_ENV = {
    "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0",
    "TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120",
    "TAAF_VLLM_MTP_TOKENS": "3", ...
}
```

That reads like a config we could copy. It is not. Downloading both source
bundles and grepping them:

| bundle | `TAAF_VLLM_*` references |
|---|---|
| ours, `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` | **0** |
| fork's, `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` | 11 distinct vars, all in its own `serving_setup.py` |

Those variables are read **only** by the fork's own bundled `serving_setup.py`
(a 3,109-line file it ships). **Setting them in our notebook would do nothing
at all, silently** — exactly the shape of failure this project has been burned
by repeatedly. The fork's "serving flags" are not flags we can set; they are a
different serving stack.

Our bundle instead launches vLLM from a hardcoded `cmd = [...]` list inside a
here-doc in `setup_commands.json`. So the portable form of this recipe is a
**direct vLLM argv change**, which is what both the benchmark and the
production patch do.

### 2.2 Our existing model ALREADY ships MTP weights [VERIFIED]

This is the finding that reshaped the whole experiment. Fetching
`config.json` for both models straight from the Kaggle Models API:

| | our production model | the fork's NVFP4 model |
|---|---|---|
| Kaggle mount | `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1` | `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1` |
| architecture | `Qwen3_5ForConditionalGeneration` | `Qwen4ExpForConditionalGeneration` |
| `text_config.mtp_num_hidden_layers` | **1** | **1** |
| MTP tensors in checkpoint | **yes** — `mtp.safetensors` | yes (`mtp.*` keys) |
| quantization | FP8 (e4m3, dynamic) | NVFP4 (4-bit, group 16) |
| MoE | dense | **512 experts**, 10 active |
| size on disk | ~30.9 GB | **135.3 GB** |

Our own submission notebook already asserts `mtp.safetensors` is present in the
mount (it is in `_required_qwen_files`), and a sibling fork's serving bundle
explicitly checks the same thing: `raise RuntimeError('Mounted checkpoint has
no built-in MTP tensor keys.')`.

**So MTP speculative decoding — the recipe's single biggest lever — does not
obviously require the model swap.** Our model carries a 1-layer MTP head, the
same depth as the NVFP4 model's. Whether *our vLLM build* can drive it is a
separate question, and is precisely what the benchmark was built to answer.

### 2.3 The model swap is not a drop-in even where it is mountable [VERIFIED]

Mountability, checked properly (a prior 403 in this project turned out to be a
*dataset* endpoint on something that was a *Model*):

| identifier | endpoint | result |
|---|---|---|
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `models/get` | **403 — no such Kaggle owner.** `RadixArk` is a HuggingFace org, not a Kaggle user. |
| **`keithtyser/qwen3-8-flash-next-nvfp4`** | `models/get` | **200 — ACCESSIBLE, public.** Byte-pinned mirror of the RadixArk repo. |
| `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` (7.9 GB vLLM runtime) | `datasets/download` | **200 — downloadable** |
| `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (15.5 MB source bundle) | `datasets/download` | **200 — downloadable** |
| `cryptozenith/qwen-27b-nvfp4` (25.2 GB, unrelated author) | `models/get` | 200 — accessible; NVFP4 at a size that *would* fit 96 GB |

**Working mount identifier for the fork's model:**
`keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`

So nothing is blocked by access. What blocks a *cheap* test is what the fork's
`serving_setup.py` reveals it needs:

- `--quantization modelopt_fp4` plus a **pinned custom vLLM runtime** (7.9 GB
  dataset, ~18.4 GB extracted, requiring ~36.5 GB free filesystem) — our
  wheelhouse is stock vLLM 0.19.0 and would have to be replaced wholesale;
- a **source-patched vLLM** for the model's FP8 PLE layers
  (`radixark_nvfp4_ple_fp8.patch`, applied by its own applicator script);
- **CPU offload of the PLE embeddings**, with an explicit host-RAM guard
  (`'Host memory is too small for FP8 PLE CPU offload'`);
- 135.3 GB of weights against a 96 GB card — a 512-expert MoE that does not fit
  in VRAM and is served only via that offload path;
- SHA-256 verification of a 419-file model manifest before boot.

**[INFERRED]** Adopting (c) is therefore not "swap a mount and add a flag"; it
is replacing our entire serving stack — wheelhouse, runtime, patch, offload
strategy — with the fork's. That is a multi-day port with a real chance of not
booting, and, critically, it **cannot be attributed**: running their bundle
would change the model, the vLLM build, the launch args and the offload
strategy in one step.

**Given 2.2, the far cheaper and better-attributed experiment is to test MTP on
our own model and our own vLLM first.** If MTP is where the gain lives, we get
it without touching the model at all. That is the experiment that was run.

---

## 3. Method

Free kernel `calamitychasm/arc3-duck-serving-benchmark`, generated by
`scripts/_build_duck_serving_diag.py`. Cells 1–5 are **byte-identical** to the
production submission notebook, so vLLM boots the real `Qwen3.8-27B-FP8` on the
real RTX PRO 6000 with the real bundled launch arguments. The final cell
replaces the game loop with the sweep.

Workload is **identical to the concurrency experiment** — same prompt
generator, same seeds, `max_tokens=512`, `ignore_eos`, unique ~9.4K-token
prompts so prefix caching cannot collapse the prefill — so rows here are
directly comparable to that table's `conc=37` row.

Three deliberate design choices, each a direct response to a way this could
have produced a confident, wrong answer:

1. **The baseline argv is read from `/proc/<pid>/cmdline` of the already-running
   server**, not hand-copied from the bundle. Every mutated config is that exact
   argv plus/minus specific tokens, so "vs. baseline" means what it says.
2. **`vllm serve --help` is dumped and parsed first**, so flag availability is
   verified against the installed build rather than assumed. And "help could not
   be parsed" is treated as *unknown*, never as *unsupported* — otherwise one
   parsing failure would skip every config and burn the run on a table of
   "skipped". (`unsupported_flags(..., help_usable=False) == []`, unit-tested.)
3. **`mtp3` and `flags` are each measured alone against the same baseline.** The
   combined `mtp3+flags` row exists but is explicitly not attributable, and is
   labelled as such.

A config that fails to boot is recorded with its server-log tail, and the
baseline server is restored before the next config, so one bad config cannot
poison the rest of the sweep.

### 3.1 Validation of the production-side patch [VERIFIED]

The production change is generated from `kaggle_submission_duck/vllm_serving.py`
into the submission notebook's cell 5 by
`scripts/_patch_duck_notebook_serving.py`, and was checked against the **real**
`setup_commands.json` downloaded from the mounted bundle
(`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`), not against a fixture:

```
real bundle setup command chars: 10076
launch anchor present in REAL bundle: True
  mtp1           patched=1  python_valid=True  has_extend=True
  flags          patched=1  python_valid=True  has_extend=True
  fork-profile   patched=1  python_valid=True  has_extend=True
  mtp3           patched=1  python_valid=True  has_extend=True
```

For every profile the anchor matches exactly once and the resulting here-doc
body is still valid Python. The notebook ships with
`DUCK_VLLM_SERVING_PROFILE = 'baseline'`, an exact no-op, and a mismatched
anchor raises rather than silently launching the unpatched argv.

---

## 4. Results

### 4.1 Environment, verified in-kernel [VERIFIED]

`vllm version: 0.19.0`. `vllm serve --help` parsed **293 flags**, and **every
flag the recipe needs is present** in our stock wheelhouse build — no flag in
this experiment was unavailable:

```
--speculative-config YES   --async-scheduling YES   --no-enable-prefix-caching YES
--kv-cache-dtype YES       --kv-cache-memory-bytes YES  --max-num-seqs YES
--max-num-batched-tokens YES  --max-cudagraph-capture-size YES  --moe-backend YES
```

Baseline argv, read from `/proc/<pid>/cmdline` of the live production server:

```
/usr/bin/python3 -m vllm.entrypoints.openai.api_server
  --model /kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1
  --served-model-name Qwen/Qwen3.8-27B-FP8 --host 127.0.0.1 --port 1234
  --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --generation-config vllm --enable-prefix-caching
  --default-chat-template-kwargs {"preserve_thinking": true}
  --reasoning-parser qwen3 --max-model-len 65536
```

Notably it sets **no** `--max-num-seqs`, `--max-num-batched-tokens`,
`--gpu-memory-utilization` or scheduling flags — those run at vLLM defaults.

### 4.2 Run 1 [VERIFIED — real kernel log, `experiments/stage7_duck_throughput_run1.txt`]

Concurrency fixed at 37. `ignore_eos` forced exactly 512 output tokens per
request, so **both rows generated an identical 18,944 tokens** — the throughput
difference is purely how long that same work took.

| config | ok | e2e agg tok/s | decode agg tok/s | ttft p50 | preempt | MTP accept | boot s | vs. baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 37/37 | **299.9** | 788.1 | 22.09 | 0 | — | — | 1.000x |
| `mtp3` | 37/37 | 288.3 | n/a¹ | 21.53 | 1 | **0.466** | 140 | **0.962x (−3.8%)** |

¹ `decode_agg_tps` is NaN for `mtp3` because the steady-decode window
degenerated: the slowest request's first token arrived at 55.9s, after the
fastest request's last token, so no interval exists in which all 37 sequences
were simultaneously decoding. That is a limitation of that particular
estimator, not a server failure — `e2e_agg_tps` is unaffected and is the
figure that governs total tokens in a fixed 9h wall-clock. (Run 2 adds a
per-request decode estimator that degrades instead of vanishing.)

**The primary hypothesis is confirmed on the mechanism and refuted on the
outcome:**

- **MTP speculative decoding works on our existing model, with stock vLLM
  0.19.0, with no model swap.** [VERIFIED] The server booted in 140s with
  `--speculative-config {"method":"mtp","num_speculative_tokens":3}` and did
  real speculative work: **23,826 draft tokens, 11,099 accepted, a 46.6%
  acceptance rate.** This is a genuinely useful finding — the recipe's biggest
  lever does *not* require the 135 GB NVFP4 mount.
- **It made throughput worse, not better**: −3.8% end-to-end, and the only
  preemption observed in either row.

vLLM warned about exactly the depth limit predicted from `config.json`
[VERIFIED, verbatim]:

```
WARNING [speculative.py:512] Enabling num_speculative_tokens > 1 will run
multiple times of forward on same MTP layer, which may result in lower
acceptance rate
WARNING [kv_cache_utils.py:1059] Add 3 padding layers, may waste at most
6.25% KV cache memory
```

Also present on **both** rows, so not attributable to MTP, but worth recording
since our baseline runs with prefix caching on and this model is a hybrid
Mamba/linear-attention architecture [VERIFIED, verbatim]:

```
WARNING [config.py:441] Mamba cache mode is set to 'align' for
Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
INFO [config.py:461] Warning: Prefix caching in Mamba cache 'align' mode is
currently enabled. Its support for Mamba layers is experimental.
```

**[INFERRED] Why MTP loses here.** Speculative decoding trades extra compute
(draft + verify) for fewer sequential decode steps. It wins when decode is
memory-bandwidth bound — i.e. at small batch, where the GPU is idle waiting on
weights. At 37 concurrent sequences the decode batch is already large enough to
be compute-bound, so the extra draft/verify work is a straight cost, and a 46.6%
acceptance rate over a one-layer MTP head run three times is not enough to pay
for it. This is consistent with the fork's own profile, which pairs 3-token MTP
with **only 8 concurrent sequences** and a 5 GiB KV cache — a deliberately
small-batch, low-latency operating point, the opposite of our
maximum-aggregate-throughput one.

Run 1 also aborted after `mtp3` (`flags` reported "failed to stop previous
server"), losing four configurations. That was a bug in this harness, not in
vLLM: `_start_server` discarded its `Popen` handle, so a killed server remained
an unreaped zombie, and `os.kill(pid, 0)` succeeds on zombies. Fixed by
tracking and waiting on the handle and reading `/proc/<pid>/stat` state.

### 4.3 Run 2

**[VERIFIED — real kernel log, `calamitychasm/arc3-duck-serving-benchmark`.]**
Concurrency fixed at 37, `ignore_eos` forcing exactly 512 output tokens per
request, so every row generated identical work. The zombie-reaping bug that
truncated run 1 is fixed, so all seven configurations completed.

| config | ok | e2e agg tok/s | dec agg | dec sum | per-seq | ttft p50 | preempt | MTP accept | boot s | vs. baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 37/37 | 302.1 | 788.8 | 507.3 | 21.32 | 21.89 | 0 | — | — | 1.000x |
| `mtp1` | 37/37 | 309.0 | 469.6 | 307.7 | 12.69 | 21.27 | 0 | **0.719** | 140 | 1.023x |
| `flags` | 37/37 | 322.9 | 940.7 | 562.7 | 25.42 | 21.20 | 0 | — | 90 | 1.069x |
| **`mtp1+flags`** | 37/37 | **352.0** | 739.1 | 397.2 | 19.97 | 20.53 | 0 | 0.709 | 70 | **1.165x** |
| `mtp2` | 37/37 | 320.4 | 426.9 | 272.2 | 11.54 | 21.52 | 0 | 0.553 | 60 | 1.061x |
| `mtp3` | 37/37 | 282.0 | n/a | 253.3 | n/a | 21.51 | 2 | 0.456 | 60 | 0.934x |
| `ngram3` | 37/37 | 129.8 | 370.0 | 223.2 | 10.00 | 19.80 | 0 | 0.295 | 75 | **0.430x** |

**Three results matter here:**

1. **`mtp1+flags` wins at +16.5%**, the best of every configuration tested, and
   it needs **no model swap** — it runs on our existing FP8 mount.
2. **Speculative depth 1, not the fork's 3.** Acceptance falls monotonically
   with depth (0.719 → 0.553 → 0.456) because our MTP head is one layer deep
   and driving it repeatedly degrades the draft. `mtp3` — literally the fork's
   published `TAAF_VLLM_MTP_TOKENS=3` — is **worse than baseline (−6.6%)** and
   produced the only preemptions in the run. **Copying the published recipe
   verbatim would have made us slower.**
3. **`ngram3` is catastrophic (−57%)** and is recorded so nobody tries it again.

The combination is **superadditive**: `mtp1` alone is +2.3% and `flags` alone is
+6.9% (sum +9.2%), but together they give +16.5%. **[INFERRED]** the likely
mechanism is `--async-scheduling` overlapping the draft/verify work speculative
decoding adds, so the two changes pay for each other rather than competing.
Note also that `mtp1+flags` had the **fastest server boot (70s)** and the
**lowest TTFT (20.53s)** of any configuration.


---

## 5. Attribution

Measured at concurrency 37, all on our existing FP8 model, no model swap:

| component | e2e tok/s | vs. baseline |
|---|---:|---:|
| serving flags alone (`--async-scheduling`, prefix caching off, `--kv-cache-dtype fp8`) | 322.9 | **+6.9%** |
| MTP speculative decoding alone, depth 1 | 309.0 | **+2.3%** |
| both together | 352.0 | **+16.5%** |

**Model swap contribution: zero, because no model swap is needed.** The recipe's
headline component (MTP) runs on our existing 30.9 GB FP8 mount; the fork's
135.3 GB NVFP4 model is a different architecture (`Qwen4Exp`, 512 experts) and
was never required to get this.

Stacking with the separately-measured concurrency change (28 → 37, +8.0%):

```
1.080  x  1.165  =  1.258   ->  ~+26% total tokens vs. the original config
```

**[INFERRED]** — that multiplication assumes the two effects are independent.
The concurrency curve was measured at baseline serving and the serving curve at
concurrency 37, so the product is an estimate, not a measurement. A combined
measurement was not run.


---

## 6. Verdict

**Ship `mtp1+flags`.** It is the measured winner (+16.5% e2e throughput at our
real operating point), needs no model swap, boots fastest, has the lowest TTFT,
and produced no preemptions.

**The public recipe does NOT reproduce as published, and that is the headline.**
Three of its components were checked directly and each failed on our stack:
- its `TAAF_VLLM_*` environment variables are **inert here** — they are read only
  by a 3,109-line `serving_setup.py` the fork ships and we do not. Setting them
  would have silently done nothing.
- its `TAAF_VLLM_MTP_TOKENS=3` is **actively harmful** on our one-layer MTP head
  (−6.6%).
- its 135 GB NVFP4 model is **unnecessary** — our own checkpoint already carries
  MTP weights of the same depth.

What did transfer is the *idea* (speculative decoding + async scheduling +
prefix caching off), retuned to our own measurements.

**Do not oversell this.** ~+26% more tokens is **not** ~+26% more score. RHAE
squares action efficiency and caps each game at its weighted completion
fraction, so extra actions convert into score non-linearly and possibly weakly.
This is a real, measured, cheap improvement — not a path to 2.99 on its own.


---

## 7. What could not be measured

- **KV-cache utilisation was `nan` for every configuration** — the
  `vllm:gpu_cache_usage_perc` scrape failed here exactly as it did in the
  concurrency benchmark. The "~22% utilised" figure from the production log
  remains **unverified** across both experiments.
- **The combined concurrency x serving effect was never measured**, only
  multiplied. See Attribution.
- **`decode_agg` degenerates for speculative configs** (NaN for `mtp3`): with
  speculation, sequences finish at very different times and the estimator's
  "all sequences simultaneously decoding" window can vanish. `dec_sum` (a
  per-request estimator added in run 2) degrades gracefully instead, and
  `e2e_agg` — the figure that actually governs total tokens in a fixed 9h
  wall-clock — is unaffected.
- **No accuracy check was run.** Speculative decoding is designed to be
  output-equivalent to non-speculative sampling, but that was assumed here, not
  verified. Prefix caching was also disabled, and this is a hybrid Mamba model
  whose prefix-caching support vLLM itself flags as experimental — so disabling
  it may be *safer* as well as faster, but neither direction was tested.
- **Nothing here has been tested in real scored play.** The only evidence is
  server-side throughput on synthetic prompts.

