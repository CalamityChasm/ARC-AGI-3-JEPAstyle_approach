# Third-party notice — the AVO arms

Everything in this directory except `THIRD_PARTY_NOTICE.md`, the build script
`scripts/_build_duck_nvfp4_avo.py`, the notebook's first markdown cell, its
print-only hardware-probe cell and the two inserted "6b. The AVO arm" cells is
**other people's work**, reproduced on our own Kaggle account to measure it.

Fourteen of the eighteen inherited cells are byte-identical to
`kaggle_submission_duck_nvfp4_anim`, and the build script asserts every one of
the eighteen by sha256 on every build. The four that differ (0, 7, 11, 13) are
listed exhaustively in that script's docstring.

**No score reported by any upstream is ours, and none is quoted here as ours.**
In particular: NVIDIA's AVO team reported 100.00 RHAE on the ARC-AGI-3 public
set with Claude Opus 5, and the one public run of *this* bundle scored 4.32 on
public-25 — on the older FP8-27B chassis, token-capped, with no surviving
matched control. Neither figure is a prediction for this notebook.

## The AVO solver bundle

`raist321/taaf-avo-v27-bundle` (public, Kaggle, CC0, v1) — a **frozen pin** of
`jakobbrggen/taaf-kaggle-source` v27 (`ARC3-Inference 74ff3df`, branch
`experiment/avo-v2`), by Jakob Brüggen, on top of the Tufa Labs ARC-AGI-3 Duck
harness (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal
Tesnar, Stefano Viel).

We mount the **pin**, not the rolling `jakobbrggen` slug, so a v28 push cannot
silently change what we ran. Re-verified byte-identical against a fresh download
of the rolling slug on 2026-09-19 (`diff -rq`: no differences other than
raist321's own added `README.md`).

`inference/avo/` (733 lines) is Tufa Labs' **reimplementation, from the
published description**, of the architecture NVIDIA's AVO team described.
NVIDIA released no code and no ablations.

## Unchanged upstreams, carried over from the anim graft

- **The graft** — `yocybercode/thui-animfast-b71-full25-r1` (Thuitanium /
  Knowless Crew): the notebook structure that puts a chosen solver bundle on
  top of the NVFP4 serving stack, with `LOCAL_ANALYZER_SEED=20260825` and
  `LOCAL_ANALYZER_YIELD_SECONDS=180`. Cell 9, which holds all of that, is
  inherited byte-identically.
- **The serving stack** — `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` and
  `wuliao0/duck-qwen3-8-anim-base` (Keith Tyser, wuliao_0). Mounted unmodified
  and sealed; this notebook changes nothing about it.
- **The weights** — `RadixArk/Qwen3.8-Flash-Next-NVFP4`, an NVFP4 quantisation
  of `Qwen/Qwen3.8-Flash-Next`. **Qwen's own licence terms apply.**

## What is ours

- `scripts/_build_duck_nvfp4_avo.py` — the builder, its per-cell sha256
  assertions, the `max_runtime_s` override and the arm switch.
- The notebook's attribution header and the two "6b. The AVO arm" cells.
- `kernel-metadata.json`'s `id`, `title`, `is_private` and `competition_sources`.
  Mounts (other than the solver bundle), `docker_image` and `machine_shape` are
  carried over unchanged.
- The measurement and its write-up in `experiments/stage7_avo.md`.
