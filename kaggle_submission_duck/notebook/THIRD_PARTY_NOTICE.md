# Third-party notebook: attribution and licensing notes

`lb-9-arc3-duck-v12-with-qwen-3-8-27b.ipynb` in this directory is a
**byte-identical copy** of a public Kaggle notebook, not code written for
this project. It is staged here (a) so this project's own real submission
`55769792` (public score 1.77) is reproducible, and (b) as the unmodified
baseline that `stage7-duck-budget-fix`'s next commit diffs a fix against.

## Attribution

- **Kaggle notebook (direct source):**
  [`foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b`](https://www.kaggle.com/code/foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b),
  author Kaggle user `foysalemonshanto` (team FOYSAL). Public, `is_private: false`,
  fetched via the Kaggle API (`kaggle kernels pull`) on 2026-09-07.
- **Underlying solver ("duck-harness") and TAAF framework, per lineage:**
  the notebook's own bundled `git_status.txt` (captured in a run of our
  fork, `calamitychasm/lb-9-arc3-duck-v12-with-qwen-3-8-27b`) names two
  source repos checked out inside the run bundle: `ARC3-Inference` and
  `tufa-arc-agi-framework`, both at commit `9158303` on branch
  `feature/animation-awareness`. This matches
  [`Tufalabs/duck-harness`](https://github.com/Tufalabs/duck-harness)
  ("The Duck: ARC-AGI-3 inference harness -- winning solution to ARC-AGI-3
  Milestone 1"), a public GitHub repository (confirmed via the GitHub API,
  2026-09-07: `visibility: public`). Two of the Kaggle fork's dataset
  contributors (`driessmit1`, `jeroencottaar`) are members of the Tufa
  Labs team, consistent with this lineage.
- **Model weights:** `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, a
  public Kaggle Model (Qwen3.8-27B, FP8-quantized, 62 votes as of
  2026-09-07). Mounted via `model_sources`, not modified or redistributed
  by this repo.
- **Supporting datasets:** `driessmit1/arc3-vllm-h100-wheelhouse-v3` (vLLM
  wheelhouse for the RTX PRO 6000 image) and
  `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` (the pickled TAAF
  benchmark/solver bundle the notebook unpickles at runtime). Mounted via
  `dataset_sources`, not modified or redistributed by this repo.

## Licensing status (checked directly, not assumed)

- The Kaggle notebook itself is **public** (`is_private: false`). Per this
  competition's own rules (`rules.md`, "Code & Submission Requirements":
  *"public sharing on Kaggle forums/notebooks is fine and is treated as
  open-sourced under an OSI license"*), a public Kaggle notebook is
  therefore treated as OSI-licensed for the purposes of this competition,
  regardless of whether the notebook or its underlying repos carry an
  explicit `LICENSE` file.
- **`Tufalabs/duck-harness` itself carries no declared license.** Checked
  directly via `gh api repos/Tufalabs/duck-harness` on 2026-09-07: the
  API's own `license` field is `null`. Unlike this project's
  `graph_explorer_THIRD_PARTY_LICENSE` case (an MIT-licensed upstream
  repo, whose license text is reproduced verbatim there per the MIT
  license's own terms), **there is no upstream `LICENSE` file to carry
  forward here** -- the repository is public and viewable, but its owner
  has not attached an OSI license to it on GitHub. This notice exists in
  place of a reproduced license text for that reason: there is nothing to
  reproduce, only the fact of public availability (which is what
  `rules.md` actually conditions "OSI-licensed" treatment on for this
  competition, not a repo-level `LICENSE` file).
- No file from `Tufalabs/duck-harness`'s own source tree is vendored into
  this repository -- only the derived, pickled `benchmark_initial.pkl` /
  `solver.pkl` bundle and this notebook's own glue code, both distributed
  by the public Kaggle notebook and datasets named above.

## Scope disclaimer

- Sits inside `kaggle_submission_duck/`, entirely separate from this
  project's own JEPA (`kaggle_submission/`) and GraphExplorer
  (`kaggle_submission_graph_explorer*/`) submission pipelines -- no shared
  code, no shared checkpoints.
- This project's only original contribution here is (a) the budget fix in
  `stage7-duck-budget-fix`'s follow-up commit (see
  `experiments/stage7_duck_budget_fix.md`) and (b) the free-diagnostic and
  scoring investigation documented in `CLAUDE.md` and
  `experiments/stage7_duck_budget_fix.md`. The 25-game public evaluation
  shape, the TAAF/duck-harness solver itself, the model, and the vLLM
  serving stack are all third-party, unmodified except where this
  project's own commits explicitly say otherwise.
