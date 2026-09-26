# Third-party notice — `kaggle_submission_duck_nvfp4_hud`

Everything in this directory except `THIRD_PARTY_NOTICE.md`, the build script
`scripts/_build_duck_nvfp4_hud.py`, the notebook's first markdown cell, the two
inserted arm cells and the three patched lines in cell 7 is **other people's
work**, reproduced on our own Kaggle account to measure it. Sixteen of the
eighteen inherited code cells are byte-identical to the source notebook, and the
build script asserts that on every build.

**No score reported by any upstream is ours, and none is quoted here as ours.**

## The base

`calamitychasm/arc3-duck-nvfp4-anim` — our own build of the NVFP4 + anim graft,
whose own attribution is in
`kaggle_submission_duck_nvfp4_anim/THIRD_PARTY_NOTICE.md` and applies here
unchanged: the graft is Thuitanium / Knowless Crew's
(`yocybercode/thui-animfast-b71-full25-r1`), the serving stack and runtime are
Keith Tyser's, the weights are RadixArk's NVFP4 quantisation of
`Qwen/Qwen3.8-Flash-Next` under Qwen's own licence, and the solver is the Tufa
Labs ARC-AGI-3 Duck harness on Jakob Brüggen's `feature/animation-awareness`
branch.

## The solver bundle, and why it is republished under our account

This kernel mounts `calamitychasm/taaf-anim-hud-v1` instead of
`jakobbrggen/taaf-kaggle-source-anim-20260807-anim`. That dataset is the
upstream bundle with **four files patched and 71 byte-identical**; the patch
edits solver source, which a notebook can only mount, not modify. The solver is
MIT-licensed (`src/ARC3-Inference/pyproject.toml`), so redistribution with
attribution is permitted. The dataset carries its own `THIRD_PARTY_NOTICE.md`
and `HUD_PATCH.md`, and the full unified diff is in this repo at
`experiments/stage7_hud_perception_artifacts/bundle_source_diff.patch`.

## The HUD detection rules

Evgenii Rudakov, Ryan Shock and Nathan Cowley, *"Graph-Based Exploration for
ARC-AGI-3 Interactive Reasoning Tasks"* (AAAI 2026 Workshop on AI for Scientific
Research, [arXiv:2512.24156](https://arxiv.org/abs/2512.24156)), original code
at [github.com/dolphin-in-a-coma/arc-agi-3-just-explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore),
MIT-licensed. Their `identify_status_bars_with_rule` and its helpers are the
source of the rules and of all three thresholds (edge distance 3, aspect ratio
5, twin count 3). The required copyright notice and licence text is carried
verbatim at
`ARC-AGI-3-Agents/agents/templates/graph_explorer_THIRD_PARTY_LICENSE` and is
shipped inside the mounted dataset as `GRAPH_EXPLORER_THIRD_PARTY_LICENSE`.

## What is ours

- `arc3_hud/hud_detect.py` — the port onto the bundle's node representation, and
  its two documented deviations from upstream (hash-based twins; bbox from
  `boundary`).
- `arc3_hud/splice.py` — the four anchored source edits.
- `scripts/_build_hud_bundle.py`, `scripts/_build_duck_nvfp4_hud.py`,
  `scripts/count_hud_mentions.py`, `scripts/_extract_hud_frames.py`.
- The prompt wording for the new field.
- `kernel-metadata.json`'s `id`, `title` and `dataset_sources[2]`. Everything
  else — mounts 0 and 1, `model_sources`, `docker_image`, `machine_shape`,
  `competition_sources` — is carried over unchanged.
- The measurement and its write-up in `experiments/stage7_hud_perception.md`.
