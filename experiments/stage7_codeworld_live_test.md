# Stage 7 -- CodeWorldAgent live test: can the coder model actually write a world model?

**Status: COMPLETE. Verdict: NO -- see the Verdict section.** (The method was
committed before the run so a session interruption could not lose the reproducible
setup; results and verdict are filled in from the real kernel output.)

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

### Divergence audit (VERIFIED by downloading the live dataset, 2026-09-09)

Neither fork is a superset of the other. Both directions matter for the
reconciliation that will be needed later:

**`master` has, live does not** (all from PR #8):
- sandbox allowlist entries `super`, `map`, `filter`, `reversed`, `type`,
  `object`, ... -- live's is missing all of them.
- `draft_world_model` returning **no** model on failure -- live still has
  `fallback = load_world_model(WORLD_MODEL_SKELETON)` at `drafting.py:178`
  and installs it.
- `DRAFT_MAX_TOKENS = 4096`. **Live passes `max_tokens=768` at all three
  draft/repair call sites** (`drafting.py:120,215,260`).
- an `extract_code` that recovers the tail of an unterminated fence, strips
  `<think>` blocks, and prefers the last block defining `WorldModel`. Live
  uses `_CODE_BLOCK_RE.search()` (first fence only) and, when no *closed*
  fence is found, `return response_text.strip()`.
- transcript-preserving retry prompts.

Those last three compound into a near-certain failure mode in the currently
deployed dataset, independent of anything the model does: a complete
`WorldModel` for a 64x64 game does not fit in 768 tokens, so the response is
truncated; a truncated response has an opening fence and no closing one, so
`_CODE_BLOCK_RE.search()` does not match; the fallback then returns the whole
response, which *begins with* ```` ```python ```` -- a guaranteed
`SyntaxError` on line 1, every time. This is stated as VERIFIED from the
downloaded live files, not inferred from the 0.00 scores.

**Live has, `master` does not** -- three modules and their wiring:
- `viewport.py`: `detect_viewport_bounds` (bounding box of non-padding cells)
  and `generate_action6_candidates` (targeted click candidates instead of
  blind 64x64 sampling; used by its `planner.py` and agent).
- `abstraction.py`: `format_compact_grid` (crops to the active viewport
  before hex-rendering) and `summarize_transition_components` (object-level
  change summary).
- `level_diff.py`: `analyze_level_transition` / `LevelTransitionAnalysis`
  (viewport-changed / new-colours detection across a level boundary).

Live's `_render_transcript` uses `format_compact_grid` plus a "semantic
summary" line per transition, where `master` still pastes the full 64x64 hex
grid. That is a real prompt-size mitigation `master` lacks, and it is directly
relevant to this experiment: `master`'s prompt carries ~4.2 KB of raw hex per
transition.

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

Kernel `calamitychasm/arc3-codeworld-live-diag` v1,
`KernelWorkerStatus.COMPLETE`, 2026-09-09. Wall clock 2677.1 s (44.6 min).
Raw evidence in `stage7_codeworld_live_test_artifacts/diag_evidence.json`
(204 KB, includes every raw LLM response in full).

Games selected (first two by sorted game_id): **`ar25-0c556536`** and
**`bp35-0a0ad940`**. Both ran to the 40-action budget.

**Note on the stdout log:** `kaggle kernels output` returned
`arc3-codeworld-live-diag.log` at **0 bytes**. Everything below therefore
comes from `diag_evidence.json`, which the driver flushes to disk after every
LLM call precisely so a lost log cannot cost the run. One consequence:
`_safe_complete`'s `logger.warning` for the single failed LLM call is
unrecoverable, so that call's *cause* is undetermined (see "What could not be
determined").

### Headline numbers -- VERIFIED

| measurement | value |
|---|---|
| draft rounds attempted | 6 (3 per game -- the full coder budget) |
| LLM completions that returned text | **25** |
| candidates that compiled and loaded | **14** |
| candidates that failed to load | **11** |
| candidates that **passed replay** | **0** |
| best replay match ever achieved | **2 / 9 transitions** (`ar25`) |
| world model installed, either game | **no** |
| installed model was the template stub | **n/a -- nothing was installed** |
| levels completed | **0** (both games) |
| actions taken | 41 (both games) |
| exceptions escaping to the heartbeat | **0** (`errors: []`) |

`fixed_code_checks` all passed, confirming the kernel really was running
PR #8's code: `super`/`map`/`filter` in the allowlist, no unconditional
skeleton fallback, transcript-preserving retries, `DRAFT_MAX_TOKENS >= 4096`.

**PR #8's core fix behaved exactly as designed.** Every one of the six draft
rounds returned `ok=False` with `world_model=None`, and
`world_model_installed` is `false` for both games with
`world_model_is_template_stub` also `false` -- i.e. nothing was installed,
rather than a stub being installed. The old code would have installed
`WORLD_MODEL_SKELETON` six times here. That fix is real and is now confirmed
against a real game. **It simply does not help**, because there was never a
passing model to install in the first place.

### Failure mode 1 -- 11/11 load failures are generation truncation, exactly

Every failing candidate's raw response contains **exactly one** ```` ``` ````
(an opening fence, never closed). Every loadable candidate contains **two**.
The correlation is perfect, 25/25:

```
load failures (11):  fences=1  errors: "'(' was never closed" x8,
                               "'[' was never closed" x1,
                               "expected 'else' after 'if' expression" x1,
                               "expected an indented block" x1
loadable (14):       fences=2  (Counter({2: 14}))
```

Not one load failure was caused by a missing builtin, a forbidden import, a
bad interface, or anything `extract_code` mishandled. PR #8's unterminated-
fence recovery worked correctly -- it salvaged the tail every time -- but a
truncated class is still a truncated class.

The raw tail of one such response, verbatim:

```
..., (39,23,9), (40,23,9), (41,23,9)],
            "ACTION5": [(63,4,5)],
            "ACTION7": [(15,15,5), (16,15,5), (17,15,5), (24,15,9), ... (47,15,4),
                       (15
```

It stops mid-tuple. `DRAFT_MAX_TOKENS` is 4096, which sounds generous, but the
model writes dense coordinate literals that tokenize at roughly **1.2
chars/token** (`(40,22,9), ` is ~11 chars and close to 10 tokens), so 4096
tokens buys only ~5 KB of that style. The longer failures (16764 and 15148
raw chars) are prose-ier nested conditionals at ~4 chars/token, i.e. also
right at 4096 tokens. Both hit the same ceiling from different directions.
_(Character-per-token ratios are computed from the response lengths, not from
a tokenizer run -- INFERRED, though the arithmetic is tight and consistent
across both styles.)_

**The retry loop is inert against this failure.** Four consecutive `ar25`
attempts produced **byte-identical** 6018-character truncated output
(`len(set(...)) == 1`), each rejected with the identical
`SyntaxError: '(' was never closed (<world_model>, line 48)`. Telling the
model its answer was truncated did not change the answer.

### Failure mode 2 -- the model memorises the transcript instead of inferring a rule

The system prompt says, explicitly, "Infer the general RULE behind each
observed change, not the specific numbers in the examples." Both observable
strategies ignore that:

**(a) Hardcode the observed cells.** Per-action tables of literal
`(x, y, colour)` triples, as in the truncation example above. That is what
blows the token budget -- and it would not generalise one step beyond the
transcript even if it fit.

**(b) Write a plausible shell and leave the rule unimplemented.** 8 of the 14
loadable candidates contain an explicit "not actually implemented" marker
(`pass`, "placeholder", "simplified", "would need"). This includes the
best-scoring candidate of the entire run.

Here is that best candidate **in full, verbatim** -- the 2/9 one, the closest
anything came to a world model
(`stage7_codeworld_live_test_artifacts/best_candidate_ar25_2of9.py`):

```python
class WorldModel:
    def __init__(self):
        self.active_cells = set()
        self.clickable_cells = set()

    def predict(self, state, action_name, x=None, y=None):
        # Create a deep copy of the state to avoid modifying the original
        next_state = [layer[:] for layer in state]
        
        # For ACTION6, check if the cell is clickable
        if action_name == "ACTION6":
            if (x, y) in self.clickable_cells:
                # Toggle the cell value (this is a guess based on the pattern)
                current_value = next_state[0][y][x]
                # Based on the pattern, it seems to toggle between 4/9 and 5/0
                if current_value == 4:
                    next_state[0][y][x] = 9
                elif current_value == 9:
                    next_state[0][y][x] = 4
                elif current_value == 5:
                    next_state[0][y][x] = 0
                elif current_value == 0:
                    next_state[0][y][x] = 5
                # Update clickable cells based on new state
                self.update_clickable_cells(next_state[0])
            return (next_state, 0, False)
        
        # For fill actions, update the state according to their pattern
        # Based on the patterns seen, these actions likely change specific regions
        if action_name in ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7"]:
            # Apply the fill pattern for the specific action
            # This is a simplified model based on the observed patterns
            self.apply_fill_pattern(next_state[0], action_name)
            # Update clickable cells based on new state
            self.update_clickable_cells(next_state[0])
            
        return (next_state, 0, False)

    def apply_fill_pattern(self, layer, action_name):
        # This is a simplified implementation based on observed patterns
        # In reality, we'd need to reverse-engineer the exact pattern
        # But since we can't import anything, we'll implement based on the examples
        
        # For demonstration purposes, let's assume a simple pattern
        # This would need to be more sophisticated to match the exact behavior
        pass

    def update_clickable_cells(self, layer):
        # Cells that can be clicked are those that are in specific states
        # Based on the pattern, it seems like cells with values 4, 5, 9, 0 are potentially clickable
        self.clickable_cells.clear()
        for y in range(len(layer)):
            for x in range(len(layer[y])):
                val = layer[y][x]
                # Based on the examples, cells that are 4, 5, 9, 0 might be clickable
                if val in [4, 5, 9, 0]:
                    self.clickable_cells.add((x, y))

    def goal_hint(self, state):
        # Simple heuristic: count how many cells are in target state (e.g., 5)
        # This is a placeholder - in reality would be more sophisticated
        count = 0
        total = 0
        for layer in state:
            for row in layer:
                for cell in row:
                    if cell == 5:
                        count += 1
                    total += 1
        return count / max(total, 1)
```

`apply_fill_pattern` -- the entire dynamics of the game -- is `pass`. It scored
2/9 because two of those nine transitions happened to be no-ops. That is the
high-water mark of the run.

(It also has a latent bug: `[layer[:] for layer in state]` copies the list of
rows but shares the row objects, so mutating `next_state[0][y][x]` mutates the
caller's `state`. Not why it failed, but worth noting -- the "no imports, so no
`copy` module" constraint pushes models toward exactly this mistake.)

### Failure mode 3 -- a real prompt bug found in `master`, not the model's fault

`bp35-0a0ad940` scored 0/9, 0/17 and 0/25 -- never a single matching
transition. Rather than write that off as "the model is bad", the recordings
were parsed directly. **`bp35`'s frames genuinely have a varying number of
layers** -- measured over its 41 recorded frames:

```
bp35-0a0ad940: layer-count distribution {5: 14, 4: 3, 3: 3, 2: 21}
ar25-0c556536: layer-count distribution {1: 41}
```

So the candidate that reasoned "ACTION1: Changes layer count from 2 to 5" was
making a **correct observation**, not hallucinating.

But `master`'s transcript rendering cannot express those transitions.
`format_diff` calls `grid_shape_mismatch` first, which short-circuits on a
layer-count change; and `_render_transcript` renders only `layer 0` of the
before-frame. Replaying the real recordings through `master`'s own
`format_diff`:

```
ar25-0c556536: 40 pairs | shape-mismatch (diff unusable)=0  | no-change=7 | real cell diff=33
bp35-0a0ad940: 40 pairs | shape-mismatch (diff unusable)=25 | no-change=0 | real cell diff=15
   sample: <shape mismatch: layer count differs: 5 vs 4>
   sample: <shape mismatch: layer count differs: 4 vs 3>
   sample: <shape mismatch: layer count differs: 3 vs 5>
```

**62.5% of `bp35`'s transitions reach the model as the single line
`<shape mismatch: layer count differs: N vs M>` and nothing else** -- no cell
information at all -- while the replay gate demands bit-exact reproduction of
the full multi-layer next state. That is unwinnable by construction, and it is
a `master` bug, not a model limitation.

`ar25` has zero shape mismatches and is therefore the clean test of model
capability. On `ar25` the model still went **0/10 candidates passing, 2 of 10
even compiling, best 2/9**.

### Failure mode 4 -- throughput

VERIFIED from the recorded per-call timings:

- 25 completions in 44.6 min of wall clock for **2 games and 40 actions each**.
- Individual calls: 34-215 s.
- Prompt size grows from ~40 K chars at the first draft to **142,145 chars** by
  the third round, because `_render_transcript` pastes a full 64x64 hex grid
  (~4.2 KB) per transition for up to 40 transitions.
- The single call that returned no text at all was the largest-prompt call
  (142,145 chars).

The real competition is up to 110 games inside a 9-hour cap. This run consumed
the entire per-game coder budget on 2 games in 45 minutes and produced nothing
usable.

### What could not be determined

- **Why one LLM call returned `None`.** `_safe_complete` swallows the exception
  into a `logger.warning`, and the stdout log came back empty. It was the
  largest-prompt call (142,145 chars), which makes a context/KV-cache limit the
  obvious suspect, but that is **INFERRED and unconfirmed** -- it could equally
  be a transient CUDA OOM. A future run should record `type(e).__name__` into
  the evidence JSON rather than only the log.
- **Whether Gemma-3-12B loads and runs here.** Its mount path was resolved and
  printed, but the action head was deliberately disabled, so nothing exercised
  it.
- **Whether this generalises past 2 games.** n=2, one clean and one hitting the
  layer bug. `ar25` alone is the only clean capability datapoint.
- **Behaviour against the real online gateway**, which no free push can reach.

## Verdict

**VERIFIED: no. Qwen3-Coder-30B-A3B did not write a single replay-passing
world model for a real ARC-AGI-3 game -- 0 out of 25 attempts, across 2 games
and 6 full draft rounds. The agent finished both games with no world model at
all and 0 levels completed.**

Stated plainly, because a negative here is the valuable result: **PR #8's
fixes cannot rescue `CodeWorldAgent`, and neither can another sprint of fixes
in the same shape.** The fixes were correct and are confirmed working against a
real game -- the skeleton is no longer installed, the fence recovery salvages
truncated output, the allowlist is right. It changes nothing, because the
agent never obtains a model to install. And notably, the allowlist fix
addressed a problem that **never once occurred** in this run: not one of the 25
candidates used `super`, `map`, `filter`, `reversed`, or any `import`
(`builtins_used_beyond_original_allowlist` is `[]` on all 25).

Three of the four failure modes are real bugs on `master` and are fixable
(`DRAFT_MAX_TOKENS` vs. the model's chosen output style; the `bp35`
shape-mismatch blackout; the ~4.2 KB-per-transition prompt). Fixing them would
buy a fairer retest and is worth doing. **But they are not the reason to
doubt the design**, because on `ar25` -- where none of them applies, the
prompt is clean and single-layer -- the model still produced 10 candidates, of
which 2 compiled and the best reproduced 2 of 9 transitions using a `predict()`
whose rule function is literally `pass`.

The deeper problem is the contract itself. `replay()` requires **bit-exact
reproduction of every observed transition** -- the full state, every layer,
all 4096+ cells, plus `levels_delta` and `done` -- and a single wrong cell
rejects the whole candidate. Against that bar the model does not attempt rule
inference at all: it either transcribes the observed cells as literal tables
(which cannot fit the budget and would not generalise if it did) or writes a
well-commented shell with the rule left unimplemented. Neither strategy can
ever pass an exact-equality gate, so more attempts, more tokens and better
retry prompts do not converge on anything.

**Recommendation: stop fixing `CodeWorldAgent` in its current form.** If the
approach is pursued further, the exact-replay gate is the thing to change, not
the plumbing around it -- e.g. accept a model that predicts the *changed
region* rather than the entire state, score candidates on partial replay match
instead of pass/fail, or drop code-generation for full dynamics in favour of
generating a much narrower predicate (which cells are clickable, what a single
action does to one object). That is a redesign, not a bug fix.

For context on what the alternative already achieves: the training-free
`GraphExplorerAgent` reaches 46 levels across 8x25 local repeats with no model,
no GPU and no LLM. `CodeWorldAgent` here spent 45 GPU-minutes on 2 games to
complete 0 levels while playing effectively at random.

### Secondary conclusions

1. **The heartbeat hardening works.** Zero exceptions escaped
   (`errors: []`), across 82 actions, 25 LLM calls, 11 malformed candidates
   and a `None` completion. `_init_failed` was `false` for both games.
2. **The deployed shared dataset is worse still.** Independently of anything
   above, it caps drafts at `max_tokens=768` and its `extract_code` returns the
   raw response (leading ```` ```python ```` included) when no *closed* fence is
   found -- a guaranteed `SyntaxError` on essentially every draft. If the
   Gemini-driven workstream is still iterating on it, that is worth telling
   them regardless of what happens to this design.
3. **Instrumentation earned its keep twice.** The 0-byte stdout log would have
   destroyed the run had the driver not flushed evidence to JSON after every
   call; and parsing the recordings instead of trusting the "layer count"
   candidate turned an apparent hallucination into a verified `master` bug.
