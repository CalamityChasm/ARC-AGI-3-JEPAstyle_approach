# Third-party notebook: attribution and licensing notes

`duck-qwen3-8-anim-base.ipynb` in this directory is a **byte-identical copy**
of a public Kaggle notebook, not code written for this project. It is staged
here as an unmodified baseline of a *higher-scoring* public Duck fork (public
LB **4.33** at the time of staging, vs. this project's own best real score of
**2.57**), so that the stack can be evaluated on a free kernel run before any
submission slot is spent on it.

Only the `kernel-metadata.json` `id` field is changed (to our own namespace, so
we push to our own kernel rather than overwriting someone else's) — see the
commit immediately following the baseline commit. `machine_shape`, `enable_gpu`,
`enable_internet`, `docker_image`, and every mount are deliberately left exactly
as the upstream author set them.

## Attribution

- **Kaggle notebook (direct source):**
  [`wuliao0/duck-qwen3-8-anim-base`](https://www.kaggle.com/code/wuliao0/duck-qwen3-8-anim-base),
  author Kaggle user `wuliao0`. Public (`isPrivateNullable: false`, confirmed
  via the Kaggle API `kernels/pull` endpoint, HTTP 200, `currentVersionNumber
  = 6`), fetched via `kaggle kernels pull` on 2026-09-10.
- **NVFP4 serving fork lineage:** the notebook's own "About this fork" markdown
  cell states that "the Duck prompts, tool-use loop, game policy, and scorer
  remain unchanged. My changes are limited to model serving and performance."
  Its three mounts are all owned by Kaggle user
  [`keithtyser`](https://www.kaggle.com/keithtyser) (display name `ktyser`), and
  this project's own earlier investigation
  (`experiments/stage7_duck_throughput.md`, section 2) identified
  `keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp` as the origin of this NVFP4 +
  MTP serving recipe. `wuliao0/duck-qwen3-8-anim-base` is therefore a
  downstream copy of that fork, not its origin.
- **Underlying solver ("duck-harness") and TAAF framework:** the original Duck
  solver and harness are by **Jeroen Cottaar and Tufa Labs**, credited by name
  in the notebook's own first markdown cell, which links to
  [`jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner).
  The notebook's second markdown cell names the full Tufa Labs solver team, in
  the authors' own alphabetical order: Harold Bessis, Jeroen Cottaar, Isaiah
  Pressman, Andries Smit, Michal Tesnar, and Stefano Viel. This corresponds to
  the public GitHub repository
  [`Tufalabs/duck-harness`](https://github.com/Tufalabs/duck-harness), the same
  upstream identified in `kaggle_submission_duck/notebook/THIRD_PARTY_NOTICE.md`.
- **Model weights:** `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`,
  a public Kaggle Model (HTTP 200 via `models/.../get`, 23 votes as of
  2026-09-10), described by its own subtitle as a *"Pinned RadixArk ModelOpt
  NVFP4 checkpoint"* — i.e. a byte-pinned Kaggle mirror of the HuggingFace repo
  `RadixArk/Qwen3.8-Flash-Next-NVFP4`. Mounted via `model_sources`; not
  modified or redistributed by this repo.
- **Supporting datasets:** `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`
  (61.5 MB — the TAAF source/serving bundle, containing `serving_setup.py`, the
  pickled benchmark, and `setup_commands.json`) and
  `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1` (7.91 GB — a pinned
  offline vLLM runtime built for NVFP4 plus the model's FP8 PLE compatibility
  patch). Both returned HTTP 200 on `datasets/view` for this account on
  2026-09-10. Mounted via `dataset_sources`; not modified or redistributed by
  this repo.

## Licensing status (checked directly, not assumed)

- The Kaggle notebook is **public**. Per this competition's own rules
  (`rules.md`, "Code & Submission Requirements": *"public sharing on Kaggle
  forums/notebooks is fine and is treated as open-sourced under an OSI
  license"*), a public Kaggle notebook is therefore treated as OSI-licensed for
  the purposes of this competition, regardless of whether the notebook or its
  underlying repos carry an explicit `LICENSE` file.
- **`Tufalabs/duck-harness` itself carries no declared license** — checked
  directly via `gh api repos/Tufalabs/duck-harness` on 2026-09-07 during the
  earlier `kaggle_submission_duck/` staging: the API's `license` field is
  `null`. There is therefore no upstream `LICENSE` text to carry forward, which
  is why this notice exists in its place. Same situation, same handling, as
  `kaggle_submission_duck/notebook/THIRD_PARTY_NOTICE.md`.
- **Both mounted datasets report `licenseNameNullable: "Unknown"`** on the
  Kaggle API. They are public and mountable by this account, but their authors
  have not attached a named license to them. Nothing from either dataset is
  vendored into this repository — they are mounted at runtime by reference only.
- No file from `Tufalabs/duck-harness`'s source tree, from the NVFP4 vLLM
  runtime, or from the serving bundle is vendored into this repository. The
  only third-party artifact committed here is the public notebook itself.

## Scope disclaimer

- Sits inside `kaggle_submission_duck_nvfp4/`, separate from this project's own
  JEPA (`kaggle_submission/`), GraphExplorer
  (`kaggle_submission_graph_explorer*/`), CodeWorld
  (`kaggle_submission_llm_world_engine/`) and FP8-Duck
  (`kaggle_submission_duck/`) submission pipelines — no shared code, no shared
  checkpoints, no shared mounts.
- **This project's own measured serving tunings were deliberately NOT applied
  to this stack.** See `experiments/stage7_duck_nvfp4.md` for the reasoning:
  our `mtp1`-beats-`mtp3` and `concurrency = 37` results were measured on a
  *different* model, a *different* vLLM build and a *different* operating
  point, and porting them here would corrupt the baseline this staging exists
  to establish.
