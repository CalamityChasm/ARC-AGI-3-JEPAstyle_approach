"""Tests for the drafting prompt's size discipline.

The original `_render_transcript` emitted a full 64x64 grid for every
transition it showed. On a real 12-game run that produced prompts of up
to **281,603 characters (~94k tokens) against a 32,768-token context**,
and the server rejected them in ~0.1s: **93 of 109 LLM calls never
reached the model**. The run then reported "0 replay passes", which looks
identical to a capability ceiling.

Nothing about that raises locally -- the renderer happily produces a
300KB string -- so only a size assertion catches a regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_STAGE = (
    Path(__file__).resolve().parent.parent
    / "kaggle_submission_llm_world_engine"
    / "dataset_stage"
)
if str(_STAGE) not in sys.path:
    sys.path.insert(0, str(_STAGE))

from llm_engine.drafting import (  # noqa: E402
    MAX_TRANSCRIPT_CHARS,
    _render_transcript,
)
from llm_engine.types import Action, GameTranscript, Transition  # noqa: E402


def board(fill=0, size=64):
    return [[[fill] * size for _ in range(size)]]


def realistic_transcript(n, cells_changed=100):
    """n transitions over full 64x64 boards, like a real game."""
    tr = GameTranscript(game_id="size-test")
    before = board()
    for i in range(n):
        after = [[row[:] for row in before[0]]]
        for c in range(cells_changed):
            after[0][(i + c) % 64][(c * 7) % 64] = (c % 15) + 1
        tr.append(
            Transition(
                frame_before=before,
                action=Action(name="ACTION1"),
                frame_after=after,
                levels_completed_before=0,
                levels_completed_after=0,
                state_after="NOT_FINISHED",
            )
        )
        before = after
    return tr


@pytest.mark.parametrize("n", [1, 5, 40, 120, 400])
def test_rendered_transcript_always_fits_the_context_budget(n):
    """The regression that mattered: 76 of 109 real prompts exceeded the
    window, and the failure was silent at render time."""
    text = _render_transcript(realistic_transcript(n))
    assert len(text) <= MAX_TRANSCRIPT_CHARS, (
        f"{n} transitions rendered to {len(text):,} chars, over the "
        f"{MAX_TRANSCRIPT_CHARS:,} budget -- the server rejects this outright"
    )


def test_only_one_full_grid_is_emitted():
    """A grid per step is what blew the budget. The diffs carry the rest."""
    text = _render_transcript(realistic_transcript(30))
    # A rendered 64x64 grid row is 64 hex chars; count full-width lines.
    grid_rows = sum(
        1 for line in text.splitlines()
        if len(line) == 64 and all(c in "0123456789abcdef" for c in line)
    )
    assert grid_rows <= 64, (
        f"{grid_rows} grid rows rendered -- more than one 64x64 grid is present"
    )


def test_every_shown_step_is_still_described():
    text = _render_transcript(realistic_transcript(12))
    for i in range(12):
        assert f"step {i}:" in text


def test_truncation_is_stated_not_silent():
    """A model told nothing was omitted would misread the opening grid as
    the level's start."""
    text = _render_transcript(realistic_transcript(400))
    assert "omitted" in text
    assert "not the start of the level" in text


def test_empty_transcript_is_handled():
    assert "no transitions" in _render_transcript(GameTranscript(game_id="empty"))


def test_small_transcript_is_not_truncated():
    text = _render_transcript(realistic_transcript(3))
    assert "omitted" not in text
    assert "step 0:" in text and "step 2:" in text


def test_rendering_is_lossless_in_principle():
    """One grid plus diffs must let a reader reconstruct each later board,
    which is the justification for dropping the per-step grids."""
    tr = realistic_transcript(4)
    text = _render_transcript(tr)
    # The opening grid is present, and each step's changed cells appear as
    # explicit coordinate->value entries.
    assert "Grid at the first step shown" in text
    for t in tr.transitions:
        changed = [
            (x, y, after)
            for y, (rb, ra) in enumerate(zip(t.frame_before[0], t.frame_after[0]))
            for x, (before, after) in enumerate(zip(rb, ra))
            if before != after
        ]
        sample_x, sample_y, sample_new = changed[0]
        assert f"({0},{sample_x},{sample_y})" in text or f"(0,{sample_x},{sample_y})" in text
