# Stage 7 — CodeWorldAgent: locally-diagnosable defects, root-caused and fixed

**Branch:** `stage7-codeworld-fixes` (off `origin/master`, with the rescue commit
`3765cf7` cherry-picked so the code under repair is actually present — `origin/master`
does not carry `kaggle_submission_llm_world_engine/`).
**Scope:** GitHub issue #3, defects 1, 3 and 4. Defect 2 (throughput) is explicitly out
of scope; defect 5 was already fixed before the scored run.
**Status:** fixes implemented and unit-tested locally. **Nothing here has been validated
in real scored play, or in any Kaggle kernel, free or otherwise.** No submission was
made. The dataset backing the submission kernel has *not* been re-uploaded, so the
version on Kaggle is still the broken one.

---

## 1. What was being explained

`CodeWorldAgent` scored **0.00** on Kaggle submission `56084133` (2026-09-07). The
retrievable `llm-world-engine-test-run` kernel output showed the persisted per-game
world models were the untouched template stub — `predict()` returning `state, 0, False`
and `goal_hint()` returning `0.0` — plus repeated `repair failed after 3 attempts` and
an unhandled `TypeError: object of type 'int' has no len()` at `llm_engine/diff.py:18`
that killed a whole game thread.

I did not have the raw kernel log; I had the summary in issue #3 and
`experiments/stage7_strategy_reset.md` §4. Everything below that is labelled VERIFIED
was established by reading the rescued source and reproducing the behaviour locally
against the code exactly as submitted (`git archive 3765cf7`), not inferred from that
summary.

---

## 2. Root cause of defect 1

### 2.1 The persistence mechanism — VERIFIED

There is exactly one code path that can write the template stub to disk.

`drafting.py:151-162` (pre-fix) ended `draft_world_model` with an unconditional
fallback:

```python
fallback = load_world_model(WORLD_MODEL_SKELETON)
return DraftOutcome(ok=False, source=WORLD_MODEL_SKELETON,
                    world_model=fallback.world_model, ...)
```

and `code_world_agent.py:190-193` (pre-fix) installed and persisted whatever came back,
**without checking `outcome.ok`**:

```python
self.model = outcome.world_model
self.model_source = outcome.source
if self.model_source:
    save_revision(..., note="draft" if outcome.ok else "draft-fallback-skeleton")
```

So a persisted stub is the fingerprint of `draft_world_model` returning `ok=False` — the
docstring's stated intent, "so callers always get *something* loadable rather than
needing to special-case None". That intent is the bug. The skeleton is not a degraded
model, it is an anti-model, and installing it is strictly worse than installing nothing,
for three compounding reasons:

1. `planner.plan()` scores every candidate by `(levels_gained, goal_hint)`. Against the
   stub both terms are constant, so the beam search is choosing uniformly among
   identical scores — with the extra cost of a depth-2 × beam-4 search to do it.
2. `_choose_engine_action` only drafts when `self.model is None`
   (`code_world_agent.py:130-131`). Installing the stub makes that permanently false, so
   **no further draft is ever attempted**, no matter how much new evidence accumulates.
3. `_handle_new_transition` then diverges from "nothing changes" on essentially every
   real transition, so the entire remaining coder budget is spent calling
   `repair_world_model` on a source that cannot pass replay by construction — which is
   defect 3 (see §4).

The stub is also *not* the same as random play: `Random` at least samples freely, while
this agent pays full search cost per step to arrive at the same place, at 0.02–0.27
actions/sec.

### 2.2 Why every draft attempt failed — four verified contributors, no single winner

The mechanism above only fires when all five attempts fail. I found four independent
defects that each make that likely, reproduced all four locally, and **cannot rank them
from the evidence available** — the raw per-attempt log is not retrievable, so which one
fired first in the real run is unknown. All four are genuine bugs and all four are
fixed.

**(a) The exec sandbox rejects ordinary generated Python.** VERIFIED.
`world_model.py`'s `_SAFE_BUILTIN_NAMES` omitted `super`, `map`, `filter`, `reversed`,
`type`, `object`, and every exception class except `ValueError`/`TypeError`/
`IndexError`/`KeyError`/`Exception`/`StopIteration`. `super()` is the sharpest: a class
whose `__init__` calls `super().__init__()` — completely standard in LLM-written Python
— raises `NameError` at instantiation, so `load_world_model` reports "constructor
failed" and the attempt is burned. A model using `map()` inside `predict()` fails
later, at replay, in a way indistinguishable from having inferred the wrong rule. A
model that wraps its own logic in `try/except AttributeError` raises `NameError` while
handling the error.

This one is demonstrable end-to-end: a **correct, replay-passing** world model for a toy
game is rejected by the pre-fix pipeline purely because it calls `super().__init__()` —
see `[FAIL] a correct model is accepted` in §5.

**(b) `extract_code` cannot recover a truncated code fence.** VERIFIED.
`_CODE_BLOCK_RE` required a *closing* fence. A response cut off by `max_tokens` has an
opening fence and no closing one, so the regex did not match, and the "fall back to the
whole response" branch returned text *starting with* ```` ```python ```` — a guaranteed
`SyntaxError` on line 1, for a candidate that may have been nearly complete.
`max_tokens` was 2048 for a class that has to model a 64×64 grid game, with no headroom
for any preamble.

**(c) `extract_code` took the *first* fence.** VERIFIED. Qwen3 models emit
`<think>…</think>`. A throwaway sketch inside the reasoning block was extracted and the
real answer discarded. Reproduced: the pre-fix extractor returns
`'class WorldModel: pass'` from a response whose actual answer is a full model.

**(d) Retry prompts discarded the transcript.** VERIFIED. `drafting.py:126-130` and
`143-149` *replaced* `user_prompt` with just the error plus the broken code. From
attempt 2 onward the model was asked to infer a rule for data it could no longer see.
So the 5-attempt loop was really 1 informed attempt plus 4 blind ones — which is why a
single recoverable slip (a), (b) or (c) could sink the whole draft.

**INFERRED, not verified:** that `max_tokens=2048` truncation actually occurred in the
real run; and which of (a)–(d) fired first. Both need the raw kernel log or a real
model, and I had neither. It is also possible more than one fired on different games.

**Ruled out:** `MockLLMClient` (which returns the skeleton verbatim) was not in play —
it requires `LLM_ENGINE_USE_MOCK=1`, the submission notebook sets only
`LLM_BACKEND=transformers`, and the measured 0.02–0.27 actions/sec is inconsistent with
a client that returns instantly.

---

## 3. Root cause of defect 4 — VERIFIED and reproduced exactly

`diff.py:18` was:

```python
if len(la) != len(lb) or (la and lb and len(la[0]) != len(lb[0])):
```

`grid_shape_mismatch` assumed both arguments were well-formed `[layer][row][col]` grids.
One of them routinely is not: `replay._check_one` calls
`format_diff(predicted_state, t.frame_after)` where `predicted_state` is the raw return
value of **LLM-authored code**, which is under no obligation to be well-formed. A
`predict()` that returns one row instead of a grid makes `la[0]` an int and `len(la[0])`
raise `TypeError: object of type 'int' has no len()` — at that exact line.

Reproduced locally through the real `replay()` path, against the code as submitted:

```
[FAIL] malformed predict() -> replay failure, not TypeError:
       RAISED TypeError: object of type 'int' has no len() @ diff.py:18
```

Note what this failure mode costs beyond the crash itself: an LLM model that is *nearly*
right but returns a slightly wrong shape — the most repairable kind of near-miss — killed
the game thread instead of producing repair feedback.

Compounding it, `code_world_agent.py` contained **zero** `try`/`except`, so the exception
propagated out of `choose_action` into the harness and ended the game.

---

## 4. Defect 3 is downstream of defect 1 — VERIFIED

The repeated `repair failed after 3 attempts` needs no separate root cause. Once the
stub was installed, `self.model_source` was the skeleton; `repair_world_model` replays
it (always fails, by construction), asks the model to patch it, and requires the patch
to pass replay on the full transcript. Every repair round is three LLM calls charged as
one budget unit (`coder_budget.record("repair")` is called once, outside the loop), on a
prompt that — per §2.2(d) — did not contain the transcript either. It cannot succeed,
and it never stopped trying, because a failed repair left `self.model_source` unchanged
and the next transition re-triggered it.

I did **not** invent a separate fix for the repair algorithm. The changes it gets are
(i) the same transcript-preserving prompt as drafting, and (ii) a stop condition: after
`MAX_CONSECUTIVE_REPAIR_FAILURES = 2` failed rounds the model is discarded and drafting
restarts from the (now much longer) transcript, instead of patching forever.

---

## 5. Before/after, same seven scenarios, real output

`scripts/demo_codeworld_prefix_vs_postfix.py` runs the scenarios the unit tests assert,
against whichever `llm_engine` you point it at. The unit tests themselves cannot do this
— they import APIs the pre-fix code does not have — so this script exists specifically to
show the delta against the code as submitted.

```
$ git archive 3765cf7 kaggle_submission_llm_world_engine/dataset_stage | tar -x -C /tmp/prefix
$ python scripts/demo_codeworld_prefix_vs_postfix.py /tmp/prefix/kaggle_submission_llm_world_engine/dataset_stage

########## PRE-FIX (git 3765cf7, code as submitted) ##########
[FAIL] failed draft does NOT hand back the template stub: world_model=INSTALLED, source_is_template_stub=True
[FAIL] extract_code recovers a truncated fence: extracted starts with 'Sure:\n\n```python\nclass'
[FAIL] extract_code ignores a <think> sketch: extracted 'class WorldModel: pass'
[FAIL] retry prompt still contains the transcript: draft_ok=False, retry_has_transcript=False
[FAIL] sandbox accepts super(): load ok=False err=WorldModel() constructor failed: NameError: name 'super' is not defined
[FAIL] a correct model is accepted: ok=False attempts=3
[FAIL] malformed predict() -> replay failure, not TypeError: RAISED TypeError: object of type 'int' has no len() @ diff.py:18

########## POST-FIX (this branch) ##########
[PASS] failed draft does NOT hand back the template stub: world_model=None, source_is_template_stub=False
[PASS] extract_code recovers a truncated fence: extracted starts with 'class WorldModel:\n    '
[PASS] extract_code ignores a <think> sketch: extracted 'class WorldModel:\n    def __in'
[PASS] retry prompt still contains the transcript: draft_ok=True, retry_has_transcript=True
[PASS] sandbox accepts super(): load ok=True err=None
[PASS] a correct model is accepted: ok=True attempts=1
[PASS] malformed predict() -> replay failure, not TypeError: replay returned cleanly, reason="predict()'s next_state layer 0 row 0 is int, expected a list of ints -"
```

The `a correct model is accepted` row is the one worth pausing on: pre-fix, a world model
that genuinely reproduces the whole transcript is rejected in 3 attempts and the agent
falls back to the stub, purely because of §2.2(a) plus §2.2(d). That is a complete,
self-contained account of how the 0.00 run could reach the stub even with a competent
coder model behind it.

---

## 6. What changed

| File | Change |
|---|---|
| `llm_engine/diff.py` | Every function is now total. New `describe_malformed()` classifies bad grids; `grid_shape_mismatch`/`diff_cells`/`format_diff`/`format_grid` return a `<…>` description instead of raising. Fixes defect 4 at source. |
| `llm_engine/world_model.py` | Sandbox gains `super`, `map`, `filter`, `reversed`, `type`, `object`, `staticmethod`/`classmethod`/`property`, `divmod`/`pow`/`iter`/`next`/`chr`/`ord`/…, and the common exception classes. `__import__`, `open`, `eval`, `exec`, `compile` stay out. `safe_predict` now type-checks the *return value*, so a malformed prediction becomes actionable repair feedback instead of reaching a formatter. |
| `llm_engine/llm_client.py` | `extract_code` rewritten: strips reasoning blocks, prefers the last fence defining `WorldModel`, recovers the tail of an unterminated fence, strips stray fence markers from unfenced text. |
| `llm_engine/drafting.py` | `DRAFT_MAX_TOKENS = 4096` (was 2048). New `_retry_prompt()` keeps the transcript in every retry, for both draft and repair. **`draft_world_model` no longer returns the skeleton on failure** — it returns `ok=False` with no model, and the rejected candidate in `last_candidate_source` for the on-disk trail only. `repair_world_model` re-drafts rather than repairing un-loadable source. |
| `code_world_agent.py` | Heartbeat hardening mirroring `hypothesis_agent.py`: `_init_failed` flag set in a hardened `__init__` (client construction can raise — a missing `*_MODEL_DIR`, a multi-GB load), top-level `try/except` on `choose_action` and `is_done`, and `_safe_fallback_action` depending on nothing but `self._rng`. `_draft_initial_model` → `_maybe_draft_model`: installs a model **only if it passed replay**, retries drafting after `REDRAFT_AFTER_NEW_TRANSITIONS = 8` new transitions, discards a model after `MAX_CONSECUTIVE_REPAIR_FAILURES = 2` failed repair rounds. Transitions with empty frames are dropped rather than recorded. |
| `tests/`, `pytest.ini` | 52 tests, all fakes. |
| `scripts/demo_codeworld_prefix_vs_postfix.py` | The before/after harness above. |

Diff: 9 files, +1414 / −96.

---

## 7. Test output

```
$ venv/Scripts/python.exe -m pytest -q
....................................................                     [100%]
52 passed in 0.32s
```

Coverage against issue #3's "definition of done", plus the extras this work turned up:

- **A drafted world model, not the stub, is what gets persisted and reloaded** —
  `test_drafted_model_is_persisted_and_is_not_the_template_stub`,
  `test_failed_draft_does_not_hand_back_the_template_stub`,
  `test_agent_never_installs_a_stub_and_replans_after_a_good_draft`.
- **Round-trip persist → load → `predict()` is not the identity stub** —
  `test_round_trip_persist_load_predict_is_not_the_identity_stub` (asserts the state
  actually changes *and* that `goal_hint` is not constant, since a constant hint gives
  the beam search nothing even if `predict` works).
- **`choose_action` never propagates an exception** — six tests, covering a world model
  that raises, a planner that raises, a `diff.py` that raises (monkeypatched back to
  broken, so the agent's own guard is what is under test, not `diff.py`'s), an LLM client
  that raises, a garbage frame object, and `_init_failed`. Plus `is_done`.
- **The `diff.py` input that produced the `TypeError` no longer raises** —
  `test_the_exact_kernel_traceback_input_no_longer_raises` drives it through the real
  `replay()` path, and 24 parametrised cases across `format_diff`/`grid_shape_mismatch`/
  `format_grid`.
- Extras: `extract_code` truncation/think-block/multi-fence, retry-prompt transcript
  retention, sandbox acceptance of `super()`/`map()`/`reversed()` and continued refusal
  of `import`, and the repair stop condition.

No test loads a model or touches the network. The toy game the fake coder writes a model
for is a 6×6 grid defined in `tests/conftest.py`.

---

## 8. What is NOT fixed, and what is not known

**Throughput (defect 2) — untouched, and still the binding constraint.** 0.02–0.27
actions/sec via `transformers` against ~110 games in a 9-hour cap. Everything in this
document could be perfectly correct and the agent would still cover almost nothing.
This was explicitly out of scope (it needs the Kaggle serving path) and is being handled
separately. **A world model that works is worth nothing at 0.02 actions/sec** — treat
this branch as a prerequisite, not a fix for the score.

**No validation in real play, at all.** Local box is an RTX 2070 (8 GB) and cannot host
Qwen3-Coder-30B, so every LLM in this work is a canned-response fake. What is verified
is that *given a competent coder model*, the pipeline now accepts a correct world model
instead of rejecting it, and that malformed output degrades to feedback instead of a
crash. Whether Qwen3-Coder-30B-A3B can actually write a replay-passing `WorldModel` for a
real 64×64 ARC-3 game from nine opening probes is **completely unknown** and is the next
thing that needs testing — a free Kaggle push with a couple of games would answer it.

**Which of §2.2's four contributors actually fired in the scored run is unknown.** The
per-attempt log is not retrievable. All four are real, all four are fixed, but this
write-up should not be read as "we found *the* bug and fixed it".

**Unmeasured risk in the redraft policy.** `REDRAFT_AFTER_NEW_TRANSITIONS = 8` and
`MAX_CONSECUTIVE_REPAIR_FAILURES = 2` are reasoned starting points, not swept values. On
a 20-call coder budget they bound worst-case spend, but the right values depend on real
per-call latency, which is defect 2's territory.

**The replay bar itself is unexamined.** `replay()` demands bit-exact reproduction of
every 64×64 frame in the transcript. For a real game that is a very high bar, and if the
coder model routinely gets 8 of 9 transitions right, the current all-or-nothing gate
throws that away. A partial-credit variant (accept a model that passes ≥ N transitions,
keep repairing) is a plausible next lever — but it is a *design* change, not a defect,
and inventing it without evidence that the bar is what is binding would be guessing.

**The Kaggle dataset has not been re-uploaded.** `calamitychasm/llm-world-engine-agent`
still holds the broken files. Nothing here reaches a kernel until it is re-versioned.

---

## 9. Reproducing

```
git worktree add ../wt-codeworld-fixes stage7-codeworld-fixes
cd ../wt-codeworld-fixes
<repo-venv>/Scripts/python.exe -m pytest -q

# before/after against the code as submitted
git archive 3765cf7 kaggle_submission_llm_world_engine/dataset_stage | tar -x -C /tmp/prefix
<repo-venv>/Scripts/python.exe scripts/demo_codeworld_prefix_vs_postfix.py \
    /tmp/prefix/kaggle_submission_llm_world_engine/dataset_stage
<repo-venv>/Scripts/python.exe scripts/demo_codeworld_prefix_vs_postfix.py \
    kaggle_submission_llm_world_engine/dataset_stage
```
