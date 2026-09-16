# Stage 7 — Porting the public NVFP4 Duck fork (`wuliao0/duck-qwen3-8-anim-base`)

Date: 2026-09-10. Branch: `stage7-duck-nvfp4`. Labels follow this repo's
convention: **[VERIFIED]** = read directly from a primary artifact (a real
kernel log, a real API response, notebook/bundle source, or a locally-run
test); **[INFERRED]** = reasoning over verified facts, not directly observed.

**No competition submission was made at any point.** Everything here comes from
free `kaggle kernels push` runs and read-only API calls. The daily submission
slot was left untouched.

---

## 1. What this stack is, and why it is worth a slot at all

| | our current best | this stack |
|---|---|---|
| kernel | `calamitychasm/lb-9-arc3-duck-v12-with-qwen-3-8-27b` | `calamitychasm/arc3-duck-nvfp4-baseline` |
| upstream | `foysalemonshanto/...` (Tufa Duck fork) | `wuliao0/duck-qwen3-8-anim-base` |
| model | Qwen3.8-27B **FP8**, dense, ~30.9 GB | Qwen3.8-Flash-Next **NVFP4**, `Qwen4Exp`, 512 experts, **135.25 GB** |
| vLLM | stock wheelhouse **0.19.0** | pinned custom build **`0.1.dev20073+g8e685d198`** |
| speculative decoding | 1-layer MTP, `mtp1` (our measured winner) | native MTP, **3 tokens** |
| vLLM `max_num_seqs` | unset (vLLM default) | **8** |
| Duck game concurrency | **37** (our measured winner) | **28** (upstream default) |
| public LB | **2.57** | **4.33** |

Top-10% is 3.20 (rank 294). Our own 2.57 does not clear it; 4.33 does. The
notebook's own "About this fork" cell states the delta is serving-only:

> The Duck prompts, tool-use loop, game policy, and scorer remain unchanged. My
> changes are limited to model serving and performance.

**[VERIFIED]** All three of its mounts are accessible to this account
(2026-09-10, Kaggle API, HTTP 200 on each): the model
`keithtyser/qwen3-8-flash-next-nvfp4` (23 votes, public), the 61.5 MB serving
bundle `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (v10), and the 7.91 GB
pinned runtime `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` (v1). The
notebook itself is public (`isPrivateNullable: false`, `currentVersionNumber = 6`).

### 1.1 What the serving bundle actually does [VERIFIED — read from the downloaded bundle]

`setup_commands.json` runs exactly one command: the bundle's own
`serving_setup.py` (3,109 lines). Reading it:

- It launches **vLLM**, not sglang — `serving_setup.py` mentions `vllm` 194
  times and `sglang` **0** times, despite the bundle vendoring an
  `sglang-rtxpro6000` source tree.
- Launch argv is built in-file and pins
  `--load-format safetensors --dtype bfloat16 --quantization modelopt_fp4
  --tensor-parallel-size 1 --distributed-executor-backend mp`, then
  `--max-model-len 32768 --max-num-seqs <env> --max-num-batched-tokens <env>
  --async-scheduling`, a chunked-prefill flag, optional `--kv-cache-dtype`,
  `--max-cudagraph-capture-size`, and a prefix-caching flag.
- It reads **11** `TAAF_VLLM_*` environment variables — the same ones the
  notebook's cell 3 sets. On *our* FP8 stack those variables are inert (see
  `experiments/stage7_duck_throughput.md` §2.1); here they are the real
  configuration surface, because this bundle is the code that reads them.
- Model identity is pinned hard: `MODEL_HF_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"`,
  `MODEL_HF_REVISION = "7b71922…"`, `MODEL_FILE_COUNT = 419`,
  `MODEL_TOTAL_BYTES = 135_253_622_894`.
- It applies a source patch to the pinned vLLM for the model's FP8 PLE layers
  (`vllm-patches/radixark_nvfp4_ple_fp8.patch`, applied by
  `apply_radixark_nvfp4_ple_fp8_patch.py`), and CPU-offloads those embeddings —
  which is how 135 GB of weights are served from a 96 GB card.
- Hard preconditions it enforces before serving:
  `MIN_HOST_AVAILABLE_BYTES = 64 GiB` host RAM,
  `MIN_RUNTIME_FILESYSTEM_FREE_BYTES ≈ 36.5 GB` free filesystem,
  `MIN_GPU_FREE_MIB = 4096`. `SETUP_TOTAL_TIMEOUT = 1800 s`,
  `SERVER_READY_TIMEOUT = 1500 s`.
- **It fails loudly, not silently, if MTP does not engage.** After boot it
  scrapes `/metrics` and asserts the speculative-decode counters moved:

  ```python
  if drafts <= 0 or draft_tokens <= 0 or accepted_tokens <= 0:
      raise RuntimeError(
          "Native MTP telemetry deltas were not positive: "
          f"drafts={drafts} draft_tokens={draft_tokens} accepted={accepted_tokens}")
  ```

  and records `acceptance_rate = accepted_tokens / draft_tokens` into
  `/kaggle/working/vllm-server-preflight.json`. It also writes
  `loaded-gpu-compute.json`, `vllm-setup-provenance.json`,
  `vllm-openai-server.log` and `vllm-metrics-after-preflight.prom`. This matters
  for this project specifically: the failure mode CLAUDE.md has been burned by
  repeatedly — a run that completes, produces a submission, and scores near the
  floor — is *structurally* harder to hit here, because setup raises and cell 9
  runs it with `check=True`.

**[VERIFIED]** The bundle's `git_status.txt` pins a *different* solver revision
from our own FP8 fork's bundle: `ARC3-Inference aa69123 (DIRTY, branch
add-kaggle-share-flag)` + `tufa-arc-agi-framework fe9f7c4 (branch
submission-share-mode-bugfix)`, where ours pins `9158303` on
`feature/animation-awareness`. **[INFERRED]** So "the solver is unchanged" is
true relative to *its own* Tufa baseline, but this is not literally the same
solver snapshot our 2.57 ran. That is an uncontrolled difference between the two
stacks, on top of the serving differences, and it should not be waved away.

---

## 2. Why our own measured tunings were deliberately NOT applied

This is the single most important design decision in this port, so it is stated
before the results rather than after.

We hold two measured serving results, both real, both from free benchmark
kernels on the production hardware:

1. **`mtp1` beats `mtp3` by 6.6%** (`experiments/stage7_duck_throughput.md`).
2. **Game concurrency 37 beats 28 by ~8.0% effective tokens**
   (`experiments/stage7_duck_concurrency.md`).

**Neither transfers, and porting either would corrupt this baseline.** The
reasons are specific, not generic caution:

- **`mtp1` was measured against a 1-layer MTP head on an FP8 dense model at 37
  concurrent sequences** — a compute-bound decode regime, where a 3-token draft
  costs more verification compute than it saves. This stack runs
  `TAAF_VLLM_MAX_NUM_SEQS = 8` on a 512-expert MoE: a small-batch,
  memory-bandwidth-bound regime, which is exactly where speculative decoding
  pays. The sign of the effect is expected to flip. **[INFERRED]** — we have not
  measured `mtp1` on *this* stack, and will not, because that is a tuning
  experiment, not a baseline reproduction.
- **`--speculative-config` here is not even the same mechanism.** Our benchmark
  drove vLLM 0.19.0's generic speculative path; this bundle drives a pinned
  custom build with a `TAAF_VLLM_MTP_DYNAMIC_BATCH_SCHEDULE` (per-batch-size
  speculative token counts) that our build has no equivalent for.
- **Concurrency 37 was derived from a wave-balance argument** (110 games ÷ 37 =
  three near-equal waves) *plus* a measured aggregate-throughput curve on our
  server. Both inputs are stack-specific. With `max_num_seqs = 8`, 28 game
  threads already oversubscribe the server 3.5×; pushing to 37 changes the
  queueing regime in a direction we have not measured here.
- **Our rerun time-budget fix is also not ported.** That fix addressed a *missing*
  soft deadline in our fork. This notebook already computes a real
  `soft_end = start + (budget − 600 s)` and passes it to `bm.run(soft_end_time=…)`
  **[VERIFIED — cell 15]**. There is nothing to fix; adding our patch would be a
  regression.

**The purpose of this branch is to establish their configuration as a clean,
unmodified baseline.** Any tuning comes after a measured baseline exists, not
before it.

### 2.1 What *was* changed, exhaustively

Commit 1 (`769e728`) is the notebook **byte-identical** to the Kaggle pull
(md5 `5025d1d3e4e6c3719e2b0ea2cbb1870c`), plus `THIRD_PARTY_NOTICE.md`.

Commit 2 (`06d03a6`) makes exactly two changes:

1. `kernel-metadata.json`: `id` → `calamitychasm/arc3-duck-nvfp4-baseline`,
   `title` matched to the slug, `is_private` → `true` (matching every other
   kernel in this repo), and **`id_no` removed** — it was the upstream author's
   numeric kernel id and must not follow a copy. `machine_shape`
   (`NvidiaRtxPro6000`), `enable_gpu` (`true`), `enable_internet` (`false`),
   `docker_image`, and all four mounts are untouched.
2. One added first code cell that prints `nvidia-smi`, host RAM, free disk and
   the rerun flag. Print-only. Its existence is *required* by the upstream
   README's own warning ("if you make a copy of this notebook, you will have to
   manually select the proper GPU (RTX Pro 6000)") — getting that wrong silently
   is exactly how a submission gets wasted.

Verified programmatically that all 18 upstream cells are byte-identical after
the inserted probe and that notebook-level metadata/nbformat are unchanged.

---

## 3. The free run

<!-- RUN RESULTS: filled in below once the free kernel completes -->

*(pending)*

---

## 4. Local tooling notes (not defects in the stack)

Two Windows-only snags hit while staging, recorded so the next session does not
re-diagnose them:

- **`kaggle kernels push` fails with `'charmap' codec can't decode byte 0x9d`.**
  The upstream notebook contains 41 non-ASCII bytes (typographic apostrophes in
  its own markdown); Kaggle CLI 2.2.3 opens the `.ipynb` with the Windows locale
  codec. Fix: `PYTHONUTF8=1 kaggle kernels push …`. This is a CLI/locale issue,
  not something introduced by our edit — the byte count is identical in the
  upstream file.
- **`kaggle datasets download --unzip` silently half-extracts the serving
  bundle** on Windows: the vendored `sglang-rtxpro6000` tree contains paths past
  the 260-char `MAX_PATH` limit, extraction aborts partway, and the CLI then
  reports the archive "was not found". The half-extracted directory looks like a
  bundle that is *missing* `taaf-kaggle-bundle.json`, `vllm_server_watchdog.py`
  and the whole `tufa-arc-agi-framework` tree — i.e. it looks exactly like a
  broken mount that would fail the notebook at cell 7. It is not: reading the
  archive's namelist directly shows all 4,408 entries present. Read members from
  the zip rather than extracting on Windows.

---

## Free-run result (2026-09-10) — VERIFIED

Kernel `calamitychasm/arc3-duck-nvfp4-baseline` v2, `COMPLETE`, ~3 h wall-clock,
public-25 path (`KAGGLE_IS_COMPETITION_RERUN` unset).

### Hardware: the manual-GPU warning does NOT apply to an API push [VERIFIED]

The upstream notebook warns *"if you make a copy of this notebook, you will have
to manually select the proper GPU (RTX Pro 6000)"*. Our probe cell shows the
metadata request **is** honoured when pushed via the API:

```
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 12.0, 580.159.04
HW_PROBE host_ram_gib=176.9   cpu_count=48
```

### Serving stack came up as intended [VERIFIED]

```
PUBLIC25_VLLM_PROFILE name=kv5-bf16-mtp3-c8-cg32
Resolved architecture: Qwen3_8FlashNextMTP
Overriding draft model max model len from 262144 to 32768
```

The `TAAF_VLLM_*` variables **are live here** because the bundle that reads them
(`serving_setup.py`, in `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`) is
mounted. This is exactly what our own stack lacked, and is why those same
variables were inert in `experiments/stage7_duck_throughput.md`.

### Score: 10.69 vs. our own stack's 3.37 [VERIFIED]

**Mean public-25 self-eval `10.69`** (median 4.76, 3,633 actions), against
**3.37** for our FP8 stack on the identical 25 games — **3.2x better**.

| game | NVFP4 | our FP8 stack |
|---|---:|---:|
| `sb26` | **58.33** (6/8 levels) | 2.78 |
| `lp85` | **41.67** (5/8) | 8.33 |
| `tr87` | **28.57** (3/6) | 4.76 |
| `ft09` | **23.81** (3/6) | 8.07 |
| `re86` | 16.67 (3/8) | 1.83 |
| `vc33` | 16.37 (3/7) | 10.71 |

### A reporting trap worth recording

**The log emits a progressive summary block roughly every 10 games' worth of
progress, not one final result.** The first block reads `mean score: 0.80,
total actions: 236`; successive blocks climb 1.78 → 3.41 → 4.79 → 6.71 → 8.73 →
**10.69** at 3,633 actions. Reading the first block as "the result" inverts the
conclusion completely — it would say this stack is 4x *worse* than ours when it
is 3.2x *better*. This was very nearly reported that way. **Always take the
LAST summary block, and cross-check it by recomputing from the per-game
`[finished]` lines** (which reproduce 10.69 exactly).

### Known issues, not blocking [VERIFIED]

- **Many `analyzer request failed ... Read timed out`** across games (timeouts
  9-155 s). The run completed and scored well regardless, so these are degraded
  requests, not fatal — but they are lost actions and represent real headroom.
- `RuntimeError: vLLM teardown did not reach the bounded terminal gate` fires
  **after** play finishes, during teardown only. No scoring impact observed.
- Qwen3VL video-processor `[ERROR]` lines at load are cosmetic kwarg-docstring
  warnings from the bundled runtime.

### Verdict

**Recommend submitting this stack unmodified on the next slot.** It is the
strongest local evidence this project has produced: 3.2x our current stack on a
like-for-like measurement, with the hardware, model mount, and speculative
decoding all verified working on our own account.

**Do not port our tunings onto it** (unchanged from this document's earlier
reasoning, now with the operating point confirmed from the log:
`kv5-bf16-mtp3-c8-cg32` — 8 sequences, 3-token MTP). Our `mtp1`/concurrency-37
results were measured at 37 concurrent on a one-layer FP8 head and do not
transfer.

**Expectation management:** public-25 is not the hidden set. Our own stack went
3.37 local → 2.57 real (~0.76x), and the notebook's author reports **4.33** real.
Do not expect 10.69 to survive contact with the hidden games.
