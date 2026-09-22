"""A guaranteed-passing WorldModel, generated from a segment.

This is the backtest's **positive control**, and it exists for one
reason: a 0% pass rate is uninterpretable unless the instrument is known
to be able to report a pass at all. Without this, "the model never
produced a replay-passing world model" and "the harness cannot recognise
one" look identical in the output. This repo has been burned by exactly
that shape of ambiguity before -- a benchmark harness that reported a
healthy vLLM server as dead and produced a complete, plausible-looking,
entirely empty results table (CLAUDE.md, the `delta.reasoning` gotcha).

The generated model is a **lookup table, not an inference**. It keys a
fingerprint of (state, action) to the exact cell changes observed, so it
reproduces the transcript by memorisation. That is precisely what the
drafting prompt forbids a real answer from doing. It is useful only as a
wiring check on replay, the sandbox, and the scoring path -- never as
evidence that the task is learnable, and it is never offered to a model
as an example.
"""

from __future__ import annotations

from typing import Optional

from ._engine import load_world_model, replay
from .extract import LevelSegment


class OracleCollision(RuntimeError):
    """Two distinct observed steps share a fingerprint.

    Raised rather than papered over: a colliding table would silently
    mispredict, which would make the positive control report a *failure*
    and send someone debugging the model instead of the fingerprint.
    """


def _fingerprint(grid: list[list[list[int]]], action_name: str, x, y) -> tuple:
    """Collision-checked key over (state, action).

    One position-weighted checksum **per row** rather than one for the
    whole board. A single global sum collided on real data (two different
    boards in m0r0 hashing equal), which the collision check caught -- but
    a control that trips on its own hash is a bad control, so the key was
    strengthened to 64 independent row checksums.
    """
    layer = grid[0]
    rows = tuple(
        sum(value * (col_index + 1) for col_index, value in enumerate(row) if value)
        for row in layer
    )
    return (action_name, x, y, rows)


_CHECKSUM_KEY_BODY = """\
        layer = state[0]
        rows = tuple(
            sum(value * (col_index + 1) for col_index, value in enumerate(row) if value)
            for row in layer
        )
        return (action_name, x, y, rows)"""

_EXACT_KEY_BODY = """\
        return (action_name, x, y, tuple(tuple(row) for row in state[0]))"""


def _changes(before: list[list[int]], after: list[list[int]]) -> list[tuple[int, int, int]]:
    return [
        (x, y, after[y][x])
        for y, (row_before, row_after) in enumerate(zip(before, after))
        for x, (old, new) in enumerate(zip(row_before, row_after))
        if old != new
    ]


def _exact_key(grid, action_name, x, y) -> tuple:
    """Whole-board key. Cannot collide, at the cost of a much larger table."""
    return (action_name, x, y, tuple(tuple(row) for row in grid[0]))


def build_oracle_source(segment: LevelSegment, exact: bool = False) -> str:
    """Emit Python source for a WorldModel that replays `segment` exactly.

    Starts with compact per-row checksums. Those are only *probably*
    unique, and on real data one segment (m0r0 L2) did collide across two
    genuinely different boards, so `exact=True` switches to whole-board
    keys, which cannot collide at all. A collision surviving even that is
    a real contradiction in the data rather than a hashing artifact --
    exactly the distinction `verify_oracle` has to draw.
    """
    key_of = _exact_key if exact else _fingerprint
    table: dict[tuple, tuple] = {}

    for t in segment.transitions:
        key = key_of(t.frame_before, t.action.name, t.action.x, t.action.y)
        entry = (_changes(t.frame_before[0], t.frame_after[0]), t.levels_delta, t.done)
        if key in table and table[key] != entry:
            detail = (
                "an identical board and action" if exact else "a fingerprint"
            )
            raise OracleCollision(
                f"{segment.key}: two steps share {detail} with different outcomes"
            )
        table[key] = entry

    key_body = _EXACT_KEY_BODY if exact else _CHECKSUM_KEY_BODY

    return f'''\
class WorldModel:
    """Positive control: a memorised lookup table, not an inferred rule."""

    TABLE = {table!r}

    def __init__(self):
        pass

    def _key(self, state, action_name, x, y):
{key_body}

    def predict(self, state, action_name, x=None, y=None):
        entry = self.TABLE.get(self._key(state, action_name, x, y))
        if entry is None:
            return state, 0, False
        changes, levels_delta, done = entry
        layer = [list(row) for row in state[0]]
        for cx, cy, value in changes:
            layer[cy][cx] = value
        return [layer], levels_delta, done

    def goal_hint(self, state):
        return 0.0
'''


def verify_oracle(segment: LevelSegment) -> tuple[bool, Optional[str]]:
    """Build the oracle for `segment`, load it, and replay it.

    Returns (passed, error). Used by the tests and by the CLI's
    `--self-check`, so a real run can assert the instrument is sound on
    the very data it is about to measure.
    """
    try:
        source = build_oracle_source(segment)
    except OracleCollision:
        # Compact keys collided. Retry with whole-board keys, which
        # cannot -- so a second failure is a genuine contradiction.
        try:
            source = build_oracle_source(segment, exact=True)
        except OracleCollision as exc:
            return False, str(exc)

    load = load_world_model(source)
    if not load.ok or load.world_model is None:
        return False, f"oracle failed to load: {load.error}"

    from .harness import _as_transcript

    result = replay(_as_transcript(segment), load.world_model)
    if result.passed:
        return True, None
    failure = result.first_failure
    return False, (
        f"oracle failed replay at step {failure.index}: {failure.reason}"
        if failure
        else "oracle failed replay for an unknown reason"
    )
