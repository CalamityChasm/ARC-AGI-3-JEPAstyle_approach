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

---

## 4. Results [VERIFIED — real kernel log]

<!-- RESULTS_TABLE -->

---

## 5. Attribution

<!-- ATTRIBUTION -->

---

## 6. Verdict

<!-- VERDICT -->

---

## 7. What could not be measured

<!-- LIMITS -->
