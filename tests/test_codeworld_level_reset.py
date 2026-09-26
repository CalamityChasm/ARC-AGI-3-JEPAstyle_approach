"""Tests for CodeWorldAgent's per-level transcript reset.

The defect this covers is not a crash -- it is silent and permanent.
`WorldModel` is documented as a simulator for ONE level, but the
transcript was never segmented: every transition was appended for the
whole game, and `draft_world_model` requires a candidate to reproduce all
of it. Since a level-clearing transition's `frame_after` is the next
level's opening layout, clearing one level made the replay gate
unsatisfiable for the rest of that game -- on exactly the games that were
going well.

Nothing about that produces an error, so only a test that inspects the
transcript across a boundary can catch a regression.

`CodeWorldAgent.__init__` needs the competition framework, so these tests
drive `_handle_new_transition` / `_start_new_level` on an instance built
with `__new__` and the attributes they actually touch -- the same
technique the graph-explorer diagnostic kernel uses.
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

from llm_engine.types import Action, GameTranscript, Transition  # noqa: E402


def grid(*rows: str) -> list[list[list[int]]]:
    return [[[int(c, 16) for c in row] for row in rows]]


def transition(before, after, levels_before=0, levels_after=0, action="ACTION1"):
    return Transition(
        frame_before=before,
        action=Action(name=action),
        frame_after=after,
        levels_completed_before=levels_before,
        levels_completed_after=levels_after,
        state_after="NOT_FINISHED",
    )


def _load_code_world_agent():
    """Import `code_world_agent` outside the competition framework.

    The module does `from ..agent import Agent` and imports `arcengine`,
    neither of which exists here, so it is loaded as a submodule of a
    synthetic package with both stubbed. This imports the REAL file --
    the point is to test the shipped code, not a copy of it.
    """
    import importlib.util
    import types

    if "arcengine" not in sys.modules:
        arcengine = types.ModuleType("arcengine")

        class _Enum:
            NOT_PLAYED = "NOT_PLAYED"
            GAME_OVER = "GAME_OVER"
            WIN = "WIN"
            NOT_FINISHED = "NOT_FINISHED"

        class _GameAction:
            RESET = "RESET"

        arcengine.FrameData = object
        arcengine.GameAction = _GameAction
        arcengine.GameState = _Enum
        sys.modules["arcengine"] = arcengine

    pkg_name = "_cwa_test_pkg"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = []  # marks it a package so relative imports resolve
        sys.modules[pkg_name] = pkg

        agent_mod = types.ModuleType(f"{pkg_name}.agent")
        agent_mod.Agent = type("Agent", (), {})
        sys.modules[f"{pkg_name}.agent"] = agent_mod

        templates = types.ModuleType(f"{pkg_name}.templates")
        templates.__path__ = []
        sys.modules[f"{pkg_name}.templates"] = templates

    full = f"{pkg_name}.templates.code_world_agent"
    if full not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            full, _STAGE / "code_world_agent.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[full].CodeWorldAgent


@pytest.fixture()
def agent():
    """A CodeWorldAgent with only the attributes these paths touch."""
    CodeWorldAgent = _load_code_world_agent()

    a = CodeWorldAgent.__new__(CodeWorldAgent)
    a.game_id = "tst-0001"
    a.transcript = GameTranscript(game_id=a.game_id)
    a.model = None
    a.model_source = None
    a.model_version = 0
    a._last_draft_attempt_len = -1
    a._consecutive_repair_failures = 0
    a.levels_seen = 0
    a.coder_client = None
    return a


def test_ordinary_transition_is_appended(agent):
    agent._handle_new_transition(transition(grid("00"), grid("10")))
    assert len(agent.transcript) == 1
    assert agent.levels_seen == 0


def test_level_boundary_resets_the_transcript_and_is_not_appended(agent):
    agent._handle_new_transition(transition(grid("00"), grid("10")))
    agent._handle_new_transition(transition(grid("10"), grid("20")))
    assert len(agent.transcript) == 2

    # The boundary step: frame_after is a whole new level's layout.
    agent._handle_new_transition(
        transition(grid("20"), grid("ff"), levels_before=0, levels_after=1)
    )

    assert len(agent.transcript) == 0, (
        "the boundary transition must not survive into the next level's "
        "transcript -- no inferred rule can reproduce a fresh layout"
    )
    assert agent.levels_seen == 1


def test_the_next_level_accumulates_its_own_transcript(agent):
    agent._handle_new_transition(transition(grid("00"), grid("10")))
    agent._handle_new_transition(
        transition(grid("10"), grid("ff"), levels_before=0, levels_after=1)
    )
    agent._handle_new_transition(transition(grid("ff"), grid("fe")))
    agent._handle_new_transition(transition(grid("fe"), grid("fd")))

    assert len(agent.transcript) == 2
    assert agent.transcript.transitions[0].frame_before == grid("ff")


def test_the_old_model_is_dropped_at_a_boundary(agent):
    """A model fitted to the previous level is wrong for a fresh layout,
    and leaving it installed also suppresses re-drafting."""
    agent.model = object()
    agent.model_source = "class WorldModel: pass"

    agent._handle_new_transition(
        transition(grid("00"), grid("ff"), levels_before=0, levels_after=1)
    )

    assert agent.model is None
    assert agent.model_source is None


def test_redraft_cooldown_is_cleared_so_the_new_level_can_draft_at_once(agent):
    """`_last_draft_attempt_len` gates re-drafting on new evidence. Left
    stale across a reset, the fresh (empty) transcript would look like it
    had *lost* evidence and drafting would be delayed."""
    agent._last_draft_attempt_len = 40
    agent._consecutive_repair_failures = 2

    agent._handle_new_transition(
        transition(grid("00"), grid("ff"), levels_before=0, levels_after=1)
    )

    assert agent._last_draft_attempt_len == -1
    assert agent._consecutive_repair_failures == 0


def test_multiple_boundaries_are_counted(agent):
    for i in range(3):
        agent._handle_new_transition(transition(grid("00"), grid("10")))
        agent._handle_new_transition(
            transition(grid("10"), grid("ff"), levels_before=i, levels_after=i + 1)
        )
    assert agent.levels_seen == 3
    assert len(agent.transcript) == 0


def test_a_regression_would_make_the_gate_unsatisfiable(agent):
    """Guards the actual consequence, not just the mechanism.

    If a boundary transition ever lands back in the transcript, every
    future candidate must reproduce a 64x64 layout it has never seen --
    which is what made drafting fail permanently before the fix.
    """
    agent._handle_new_transition(
        transition(grid("00"), grid("ff"), levels_before=0, levels_after=1)
    )
    assert all(t.levels_delta == 0 for t in agent.transcript.transitions), (
        "no transition with a level change may remain in a per-level transcript"
    )
