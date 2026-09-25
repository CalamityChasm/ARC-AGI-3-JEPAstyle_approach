"""The backtest itself: can this model write a world model that replays?

The question this instrument answers, and nothing wider:

    Given the first N observed steps of one real level, can the served
    model write a `WorldModel` whose `predict()` reproduces all N exactly?

That is a **counted** quantity. It has no score variance, no hidden-set
sampling, and no dependence on the +/-2.46 SE that makes a single free
public-25 run unable to rank anything (CLAUDE.md, "THE MEASUREMENT RULE
THAT MATTERS MOST"). A run of this backtest either produces a
replay-passing model for a given segment or it does not.

Two design choices worth stating plainly, because they set what a result
here does and does not license:

**Replay is teacher-forced, not free-running.** `llm_engine.replay`
scores every step from that step's own real `frame_before`, never from
the model's own previous prediction. Errors therefore cannot compound.
This is a strictly *easier* test than the closed-loop rollout a planner
would actually need, which is deliberate: it is meant to be a floor. A
model that fails here cannot possibly support beam search over imagined
futures, so a negative result is decisive. A positive result is
necessary, not sufficient.

**The identity baseline is always reported.** `WORLD_MODEL_SKELETON` is
an "assume nothing ever changes" model; on a step where the board did not
change it passes for free. Real segments here run 0.56-1.00 changed
fraction, so that baseline is not zero and a pass rate quoted without it
would be unreadable. This repo has made exactly that mistake before, at
length, in Stage 1 -- the whole `changed-patches` metric exists because
whole-grid MSE was trivially beatable by predicting no change.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

from ._engine import (
    WORLD_MODEL_SKELETON,
    GameTranscript,
    ReplayResult,
    load_world_model,
    replay,
)
from .extract import LevelSegment
from .render import MAX_STEPS, build_repair_prompt, build_user_prompt

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are writing a Python simulator for one level of an \
unfamiliar grid-based game. You will be shown the level's opening grid and a \
list of steps: the action taken and exactly which cells changed. Infer the \
general RULE behind those changes, not the specific numbers -- your code is \
replayed against those same steps and must reproduce every one exactly.

Grids are lists of layers; each layer is a 64x64 list of integers 0-15 \
(colors). These games use exactly one layer. Coordinates are (x, y), \
0-indexed, origin top-left, so cell (x,y) is row y, column x.

Actions: ACTION1-ACTION5 and ACTION7 take no arguments. ACTION6 takes (x, y). \
What each action does is game-specific and only knowable from the steps shown.

Your code runs with NO imports available (no `import` statements at all -- \
plain Python using only builtins like list/dict/range/len/enumerate/min/max/abs). \
Do not use numpy, copy, or math.

THE RETURN CONTRACT -- get this exactly right or the answer is discarded \
before your rule is even considered:

  predict() MUST return a 3-tuple  (next_state, levels_delta, done)
    next_state   a LIST OF LAYERS, same shape as `state`.
                 `state` is [layer]; you must return [new_layer].
                 Returning `new_layer` on its own is WRONG and is the
                 single most common way these answers fail.
    levels_delta an int, almost always 0
    done         a bool, almost always False

Work on a copy. Never mutate `state` in place.

Here is a COMPLETE, VALID answer -- a model that predicts "nothing \
changes". Copy this structure exactly and replace only the marked line \
with your inferred rule:

```python
class WorldModel:
    def __init__(self):
        self.counters = {}

    def predict(self, state, action_name, x=None, y=None):
        layer = [row[:] for row in state[0]]      # copy, never mutate state

        # ---- your inferred rule goes here, editing `layer` in place ----
        # e.g.  if action_name == "ACTION3": layer[y][x] = 5

        return [layer], 0, False                  # NOTE: [layer], not layer

    def goal_hint(self, state):
        return 0.0
```

Before you answer, check your own code:
  1. does `predict` return `[layer], 0, False` -- a list, an int, a bool?
  2. is every index inside 0..63, guarded so no IndexError is possible?
  3. does it run without `import`?
  4. is the class complete, with both `predict` and `goal_hint`?

Respond with ONLY a single Python code fence containing the `WorldModel` \
class. No prose before or after it.
"""

#: Response budget per attempt. `llm_engine.drafting` settled on 4096
#: after a 2048 cap was found to truncate models mid-class, which is not a
#: partial answer but a guaranteed compile failure costing a whole attempt.
DRAFT_MAX_TOKENS = 4096


class SupportsComplete(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str: ...


# -- failure taxonomy -------------------------------------------------
#
# Kept coarse on purpose: each bucket implies a different fix, and a
# taxonomy finer than the fixes it distinguishes is just noise.
NO_RESPONSE = "no_response"        # client returned nothing / unreachable
NO_CODE = "no_code"                # reply contained no `class WorldModel` at all
LOAD_ERROR = "load_error"          # syntax / missing class / constructor raised
PREDICT_RAISED = "predict_raised"  # predict() threw at replay time
BAD_SHAPE = "bad_shape"            # predict() returned a malformed triple
GRID_MISMATCH = "grid_mismatch"    # ran fine, wrong board
SIGNAL_MISMATCH = "signal_mismatch"  # board right, levels_delta/done wrong
PASSED = "passed"


def classify_failure(result: Optional[ReplayResult]) -> str:
    """Bucket a failed replay by the first thing that went wrong."""
    if result is None or result.first_failure is None:
        return PASSED if result is not None and result.passed else NO_RESPONSE
    reason = result.first_failure.reason or ""
    if "raised" in reason:
        return PREDICT_RAISED
    if "expected" in reason or "malformed" in reason or "nesting level" in reason:
        return BAD_SHAPE
    if "grid mismatch" in reason or "shape mismatch" in reason:
        return GRID_MISMATCH
    if "levels_delta mismatch" in reason or "done mismatch" in reason:
        return SIGNAL_MISMATCH
    return GRID_MISMATCH


@dataclass
class AttemptRecord:
    index: int
    outcome: str
    prefix: int            # steps passing before the first failure
    elapsed_s: float
    response_chars: int


@dataclass
class SegmentResult:
    """Outcome for one (game, level) segment."""

    segment_key: str
    game_id: str
    level: int
    n_transitions: int
    changed_fraction: float

    passed: bool = False
    attempts: int = 0
    outcome: str = NO_RESPONSE
    #: Longest prefix of steps reproduced exactly, across all attempts.
    best_prefix: int = 0
    #: Same measure for the do-nothing model -- the null control.
    identity_prefix: int = 0
    identity_passed: bool = False
    elapsed_s: float = 0.0
    attempt_records: list[AttemptRecord] = field(default_factory=list)
    source: Optional[str] = None

    @property
    def best_prefix_fraction(self) -> float:
        return self.best_prefix / self.n_transitions if self.n_transitions else 0.0

    @property
    def beats_identity(self) -> bool:
        """Did the model reproduce strictly more than 'nothing changes'?"""
        return self.best_prefix > self.identity_prefix

    def as_dict(self) -> dict:
        d = asdict(self)
        d["best_prefix_fraction"] = self.best_prefix_fraction
        d["beats_identity"] = self.beats_identity
        # The source is kept on the object for saving separately, but it
        # would swamp a JSON summary.
        d.pop("source", None)
        return d


@dataclass
class BacktestConfig:
    max_steps: int = MAX_STEPS
    max_attempts: int = 3
    max_tokens: int = DRAFT_MAX_TOKENS


def _as_transcript(segment: LevelSegment) -> GameTranscript:
    transcript = GameTranscript(game_id=segment.game_id)
    for t in segment.transitions:
        transcript.append(t)
    return transcript


def identity_baseline(segment: LevelSegment) -> tuple[int, bool]:
    """(prefix, passed) for the do-nothing `WORLD_MODEL_SKELETON`.

    Needs no LLM at all, so it is free and exact. Any segment where this
    passes is a segment the backtest cannot learn anything from, and is
    reported as such rather than counted as a win.
    """
    load = load_world_model(WORLD_MODEL_SKELETON)
    if not load.ok or load.world_model is None:
        # The skeleton is a constant in the engine; if it stops loading,
        # that is a real regression in the sandbox, not a data condition.
        raise RuntimeError(f"WORLD_MODEL_SKELETON failed to load: {load.error}")
    result = replay(_as_transcript(segment), load.world_model)
    prefix = result.total if result.passed else (
        result.first_failure.index if result.first_failure else 0
    )
    return prefix, result.passed


def run_segment(
    client: SupportsComplete,
    segment: LevelSegment,
    config: Optional[BacktestConfig] = None,
) -> SegmentResult:
    """Draft-and-replay one segment, up to `config.max_attempts` times.

    `segment` is expected to be already windowed -- see
    `LevelSegment.window()`. The steps rendered into the prompt and the
    steps replayed against are the same list, by construction.
    """
    from llm_engine.llm_client import extract_code

    config = config or BacktestConfig()
    identity_prefix, identity_passed = identity_baseline(segment)

    result = SegmentResult(
        segment_key=segment.key,
        game_id=segment.game_id,
        level=segment.level,
        n_transitions=len(segment.transitions),
        changed_fraction=segment.changed_fraction,
        identity_prefix=identity_prefix,
        identity_passed=identity_passed,
    )

    transcript = _as_transcript(segment)
    user_prompt = build_user_prompt(segment)
    started = time.monotonic()

    for attempt in range(1, config.max_attempts + 1):
        result.attempts = attempt
        attempt_started = time.monotonic()

        try:
            response = client.complete(
                SYSTEM_PROMPT, user_prompt, max_tokens=config.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 -- a dead server is a data point
            logger.warning("%s attempt %d: client failed: %s", segment.key, attempt, exc)
            result.outcome = NO_RESPONSE
            result.attempt_records.append(
                AttemptRecord(attempt, NO_RESPONSE, 0, time.monotonic() - attempt_started, 0)
            )
            break

        if not response:
            result.outcome = NO_RESPONSE
            result.attempt_records.append(
                AttemptRecord(attempt, NO_RESPONSE, 0, time.monotonic() - attempt_started, 0)
            )
            break

        source = extract_code(response)
        # `extract_code` deliberately falls back to the raw response when
        # there is no fence at all (a model may emit unfenced code), so
        # "empty" is not the right test for "produced no artifact". A
        # reply with no `class WorldModel` in it anywhere did not attempt
        # the thing that was asked for, which is a different failure from
        # attempting it and getting it wrong -- and it wants a different
        # fix (prompt compliance, not code repair).
        if not source.strip() or "class WorldModel" not in source:
            result.outcome = NO_CODE
            result.attempt_records.append(
                AttemptRecord(
                    attempt, NO_CODE, 0, time.monotonic() - attempt_started, len(response)
                )
            )
            user_prompt = build_repair_prompt(
                segment, response[:2000], "you did not return a Python code fence"
            )
            continue

        load = load_world_model(source)
        if not load.ok or load.world_model is None:
            result.outcome = LOAD_ERROR
            result.attempt_records.append(
                AttemptRecord(
                    attempt, LOAD_ERROR, 0, time.monotonic() - attempt_started, len(response)
                )
            )
            user_prompt = build_repair_prompt(
                segment, source, f"it failed to load: {load.error}"
            )
            continue

        replay_result = replay(transcript, load.world_model)
        prefix = (
            replay_result.total
            if replay_result.passed
            else (replay_result.first_failure.index if replay_result.first_failure else 0)
        )
        result.best_prefix = max(result.best_prefix, prefix)

        if replay_result.passed:
            result.passed = True
            result.outcome = PASSED
            result.source = source
            result.attempt_records.append(
                AttemptRecord(
                    attempt, PASSED, prefix, time.monotonic() - attempt_started, len(response)
                )
            )
            break

        outcome = classify_failure(replay_result)
        result.outcome = outcome
        result.attempt_records.append(
            AttemptRecord(
                attempt, outcome, prefix, time.monotonic() - attempt_started, len(response)
            )
        )

        failure = replay_result.first_failure
        user_prompt = build_repair_prompt(
            segment,
            source,
            f"it loaded but does not reproduce the steps. Step #{failure.index} "  # type: ignore[union-attr]
            f"is wrong: {failure.reason}",  # type: ignore[union-attr]
        )

    result.elapsed_s = time.monotonic() - started
    return result
