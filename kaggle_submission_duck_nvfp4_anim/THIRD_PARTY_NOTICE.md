# Third-party notice — `kaggle_submission_duck_nvfp4_anim`

Everything in this directory except `THIRD_PARTY_NOTICE.md`, the build script
`scripts/_build_duck_nvfp4_anim.py`, the notebook's first markdown cell and its
print-only hardware-probe cell is **other people's work**, reproduced on our own
Kaggle account to measure it. Every code cell is byte-identical to the source
notebook, and the build script asserts that on every build.

**No score reported by any upstream is ours, and none is quoted here as ours.**

## The graft

`yocybercode/thui-animfast-b71-full25-r1` (public, Kaggle, v1) — Thuitanium /
Knowless Crew. Same source as `sahasawatt/thui-animfast-v1`. This is the
notebook we copied: the change that puts the anim solver on top of the NVFP4
serving stack, together with the `LOCAL_ANALYZER_SEED` / `LOCAL_ANALYZER_YIELD_SECONDS`
knobs and the anim deploy target.

## The serving stack

`keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` and `wuliao0/duck-qwen3-8-anim-base`
(public, Kaggle) — Keith Tyser and wuliao_0. The pinned offline vLLM runtime
(`keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`), the serving bundle
(`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`), the NVFP4 FP8-PLE
compatibility patch, the MTP-3 speculative-decoding profile and the server
watchdog are all theirs and are mounted unmodified.

## The solver

The Tufa Labs ARC-AGI-3 Duck harness — Harold Bessis, Jeroen Cottaar, Isaiah
Pressman, Andries Smit, Michal Tesnar, Stefano Viel — as published at
`jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`, on Jakob
Brüggen's `feature/animation-awareness` branch, distributed as the Kaggle dataset
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`. Mounted and executed
unmodified; no solver file is edited by this notebook.

## The weights

`RadixArk/Qwen3.8-Flash-Next-NVFP4` — an NVFP4 quantisation of
`Qwen/Qwen3.8-Flash-Next`, mirrored on Kaggle as
`keithtyser/qwen3-8-flash-next-nvfp4`. **Qwen's own licence terms apply to the
weights.**

## What is ours

- `scripts/_build_duck_nvfp4_anim.py` — the builder and its byte-identity assertions.
- The notebook's first markdown cell (attribution header) and its second cell
  (a print-only `nvidia-smi` / RAM / disk probe).
- `kernel-metadata.json`'s `id`, `title`, `is_private` and `competition_sources`.
  Mounts, `docker_image` and `machine_shape` are carried over unchanged.
- The measurement and its write-up in `experiments/stage7_model_search.md`.
