# Stage 7 -- CodeWorldAgent live test: can the coder model actually write a world model?

**Status: RESULTS PENDING** (this file is committed with the method first so
that a session interruption cannot lose the reproducible setup; the results
and verdict sections are filled in from the real kernel output).

## The question

`CodeWorldAgent` (`kaggle_submission_llm_world_engine/`) has scored **0.00 on
every real Kaggle submission**. PR #8 found and fixed a verified root cause:
`draft_world_model` had an *unconditional* fallback that returned the template
skeleton, which the agent then installed without checking `outcome.ok`; plus a
sandbox allowlist missing `super` (so a correct model calling
`super().__init__()` was rejected at load time), an `extract_code` bug on
unterminated closing fences, and retry prompts that discarded the transcript.

All of that is real. **None of it matters if the prior question is "no":**

> Can Qwen3-Coder-30B-A3B actually write a WorldModel that reproduces a real
> 64x64 ARC-AGI-3 game's observed transitions exactly?

Nobody had ever checked. The local box is an RTX 2070 (8GB) and cannot host
the model, so this is only answerable on Kaggle hardware.

This experiment answers it with printed evidence, not inference.

## Why this needed a new kernel and a new dataset

**New dataset.** The live shared dataset `calamitychasm/llm-world-engine-agent`
does **not** contain PR #8's fixes -- verified by download, not assumed:

| check | live shared dataset | this branch's `dataset_stage/` |
|---|---|---|
| `"super"` in sandbox allowlist | absent | present |
| `fallback = load_world_model(WORLD_MODEL_SKELETON)` | **still present** (`drafting.py:178`) | removed |

It is also a *diverged fork*, not merely stale: it carries three modules that
do not exist on `master` at all -- `llm_engine/viewport.py` (115 lines),
`llm_engine/abstraction.py` (98), `llm_engine/level_diff.py` (67) -- and they
are wired deep into its own copies of `drafting.py`, `replay.py`,
`planner.py`, `__init__.py` and `code_world_agent.py`. A separate
(Gemini-driven) workstream is iterating on it. **That divergence will need
reconciling later; this experiment deliberately does not touch it.** Instead
it publishes a separate dataset, `calamitychasm/llm-world-engine-agent-fixed`,
built from `master`'s `dataset_stage/`, so the two workstreams cannot clobber
each other.

**New kernel.** Every previous free push validated nothing, because all of the
real setup and agent code sits behind
`if os.getenv('KAGGLE_IS_COMPETITION_RERUN')`, which is unset on a free push.
The diagnostic kernel runs **unconditionally**.

**Why not `main.py`.** `main.py` sources its game list from a live HTTP call to
`{ROOT_URL}/api/games` before anything else runs, *regardless of
`OPERATION_MODE`*. A free push has `enable_internet: false` and no gateway
sidecar, so that call always fails and `main.py` exits with "No games
available to play". `diag_driver.py` keeps every other part of the real path
identical -- same `Swarm`, same `Arcade`-created environments, same
`CodeWorldAgent`, same `TransformersClient` -- and replaces only the
game-listing step with a direct read of the scanned OFFLINE environments.

## Method

Artifacts (all on branch `stage7-codeworld-live-test`):

- `kaggle_submission_llm_world_engine/notebook_diag/diag_driver.py` -- the
  instrumented driver.
- `kaggle_submission_llm_world_engine/notebook_diag/build_diag_notebook.py` --
  inlines the driver into the notebook as a base64 blob, so a driver change
  needs only a free `kaggle kernels push`, never a dataset re-version.
- `kaggle_submission_llm_world_engine/notebook_diag/arc3-codeworld-live-diag.ipynb`
  (generated) and `kernel-metadata.json`.

Kernel config: `machine_shape: NvidiaRtxPro6000`, `enable_gpu: true`,
`enable_internet: false`, competition attached (for the wheels and
`environment_files`), models attached (Qwen3-Coder-30B-A3B and Gemma-3-12B-it),
dataset `calamitychasm/llm-world-engine-agent-fixed`. **Never submitted for
scoring.**

Run parameters (deliberately modest, so the run fits a free GPU session):
2 games, `MAX_ACTIONS=40`, `CODER_LLM_CALL_BUDGET=3` draft/repair rounds per
game, `DRAFT_MAX_ATTEMPTS=5`, `REPAIR_MAX_ATTEMPTS=2`.

### Instrumentation

`drafting._safe_complete`, `drafting.load_world_model` and `drafting.replay`
are monkeypatched in the driver, so every LLM round-trip is captured at source
without editing the deployed package. `code_world_agent`'s own module-level
references to `draft_world_model`/`repair_world_model` are wrapped separately
(it binds them at import time, so patching `drafting.*` alone would miss them).

Printed per attempt: the raw LLM response verbatim, the extracted candidate
source verbatim, the load result (`ok` / exact error), whether the candidate is
**byte-identical to `WORLD_MODEL_SKELETON`** (the decisive stub check), which
builtins beyond the original allowlist it uses (`super`/`map`/`filter`/...,
i.e. whether the pre-PR#8 sandbox would have rejected it), any `import`
statements, and the replay result (`pass_count/total` plus the first
divergence). Per game at the end: actions taken, levels completed, final
state, coder calls used, whether a world model was installed, and its full
source.

Everything is also written to `/kaggle/working/diag_evidence.json`, flushed
after every LLM call so a timeout still leaves partial evidence.

### Deliberate deviations from the real submission

All are printed at runtime, and all are noted here rather than buried:

1. `OPERATION_MODE=offline` against the competition's own `environment_files`
   (no gateway is available on a free push). The 25 public local games, not
   the ~110 hidden ones.
2. 2 games and a reduced action/LLM budget, as above.
3. `ACTION_MODEL_DIR` points at the **coder** model directory, so
   `get_shared_transformers_client` hands back the already-loaded 30B model
   instead of putting Gemma-3-12B in VRAM alongside it. The action head is
   *separately* disabled (`ACTION_LLM_CALL_BUDGET=0`) so it is never consulted;
   this deviation therefore only removes a VRAM/latency confound from the
   question being asked. The Gemma mount path is still resolved and printed.

### Pre-flight validation

The driver was validated locally before any GPU time was spent, via a
`DIAG_SELFTEST=1` mode that drives the full instrumentation and reporting path
from a synthetic transcript and a fake client. That confirmed, on the local
box: the fixed-code checks pass, the round wrapper fires, `is_exact_skeleton`
correctly identifies a skeleton response, replay divergence is reported, and
PR #8's behaviour holds -- a skeleton candidate is **rejected and nothing is
installed** (`outcome.world_model is None`), where the old code would have
installed it.

## Results

_(filled in from the real kernel output)_

## Verdict

_(filled in from the real kernel output)_
