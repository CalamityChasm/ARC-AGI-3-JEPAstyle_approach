"""Regression tests for the CodeWorldAgent defects diagnosed after the
0.00 scored submission (Kaggle ref 56084133, 2026-09-07) -- see GitHub
issue #3 and experiments/stage7_codeworld_fixes.md.

Everything here uses fakes. No model is loaded and no network call is
made: the "coder LLM" is a canned-response object, and the game being
modelled is a 6x6 toy defined in conftest.py.
"""

from __future__ import annotations

import pytest

from conftest import (  # type: ignore[import-not-found]
    CORRECT_WORLD_MODEL_SOURCE,
    FakeLLMClient,
    RaisingLLMClient,
    build_transcript,
    with_agent_at,
)

from llm_engine.diff import describe_malformed, format_diff, format_grid, grid_shape_mismatch
from llm_engine.drafting import draft_world_model, repair_world_model
from llm_engine.llm_client import extract_code
from llm_engine.persistence import save_revision
from llm_engine.replay import replay
from llm_engine.types import Action, GameTranscript, Transition
from llm_engine.world_model import (
    WORLD_MODEL_SKELETON,
    load_world_model,
    safe_predict,
)


def fenced(source: str) -> str:
    return f"Here is the model:\n\n```python\n{source}\n```\n"


def is_template_stub(source: str | None) -> bool:
    """True if `source` is the do-nothing template -- the thing that must
    never reach the planner."""
    if source is None:
        return False
    return source.strip() == WORLD_MODEL_SKELETON.strip()


# =====================================================================
# Defect 1: a drafted world model must be what gets persisted/installed,
# and the do-nothing template stub must never be.
# =====================================================================


def test_drafted_model_is_persisted_and_is_not_the_template_stub(tmp_path):
    transcript = build_transcript()
    client = FakeLLMClient([fenced(CORRECT_WORLD_MODEL_SOURCE)])

    outcome = draft_world_model(client, transcript, max_attempts=3)

    assert outcome.ok, f"draft rejected a correct model: {outcome.replay_result}"
    assert outcome.source is not None
    assert not is_template_stub(outcome.source)
    assert "def predict" in outcome.source

    path = save_revision("test-game", 1, outcome.source, note="draft", out_dir=tmp_path)
    reloaded = path.read_text(encoding="utf-8")
    assert reloaded == outcome.source
    assert not is_template_stub(reloaded), "the persisted revision is the template stub"

    loaded = load_world_model(reloaded)
    assert loaded.ok, loaded.error
    assert replay(transcript, loaded.world_model).passed


def test_failed_draft_does_not_hand_back_the_template_stub():
    """The 0.00 run's root cause: draft_world_model used to return a
    *loaded copy of the skeleton* on failure, which the agent then
    installed and persisted as if it were a model."""
    transcript = build_transcript()
    # Three responses that all load fine but predict "nothing changes".
    dud = "class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n        return s, 0, False\n    def goal_hint(self, s):\n        return 0.0\n"
    client = FakeLLMClient([fenced(dud)] * 3)

    outcome = draft_world_model(client, transcript, max_attempts=3)

    assert not outcome.ok
    assert outcome.world_model is None, "a rejected draft must not be installable"
    assert outcome.source is None, "a rejected draft must not be persisted as the model"
    assert not is_template_stub(outcome.source)
    # The rejected candidate is still retrievable for the on-disk trail.
    assert outcome.last_candidate_source is not None


def test_agent_never_installs_a_stub_and_replans_after_a_good_draft(code_world_agent_module, tmp_path, monkeypatch):
    """End-to-end at the agent level: a failing coder leaves `model` None
    (not a stub), and a subsequent good draft is installed."""
    import functools

    import llm_engine.persistence as persistence

    monkeypatch.setattr(
        code_world_agent_module,
        "save_revision",
        functools.partial(persistence.save_revision, out_dir=tmp_path),
    )

    agent = make_agent(code_world_agent_module, coder_responses=["not code at all"])
    agent.transcript = build_transcript()
    agent._last_draft_attempt_len = -1

    agent._maybe_draft_model()
    assert agent.model is None, "a failed draft must leave the agent model-less, not stubbed"
    assert agent.model_source is None

    # New evidence arrives, the coder now returns a working model.
    agent.coder_client = FakeLLMClient([fenced(CORRECT_WORLD_MODEL_SOURCE)])
    agent._last_draft_attempt_len = -1
    agent._maybe_draft_model()
    assert agent.model is not None
    assert not is_template_stub(agent.model_source)


def test_round_trip_persist_load_predict_is_not_the_identity_stub(tmp_path):
    """persist -> load -> predict() actually changes the state."""
    transcript = build_transcript()
    client = FakeLLMClient([fenced(CORRECT_WORLD_MODEL_SOURCE)])
    outcome = draft_world_model(client, transcript, max_attempts=2)
    assert outcome.ok

    path = save_revision("test-game", 1, outcome.source, note="draft", out_dir=tmp_path)
    loaded = load_world_model(path.read_text(encoding="utf-8"))
    assert loaded.ok, loaded.error

    state = with_agent_at(2, 2)
    next_state, delta, done, error = safe_predict(loaded.world_model, state, Action(name="ACTION4"))
    assert error is None, error
    assert next_state != state, "reloaded model behaves like the do-nothing stub"
    assert describe_malformed(next_state) is None
    assert len(next_state) == len(state) and len(next_state[0]) == len(state[0])
    assert (delta, done) == (0, False)

    # And its goal_hint is not the stub's constant 0.0.
    hints = {loaded.world_model.goal_hint(with_agent_at(x, x)) for x in range(5)}
    assert len(hints) > 1, "goal_hint is constant -- the planner would have no signal"


def test_stub_would_have_failed_replay_which_is_why_it_must_not_be_installed():
    """Documents the mechanism, not just the fix: the template stub cannot
    pass replay on any transcript where anything changes, so installing it
    guaranteed an endless repair loop."""
    transcript = build_transcript()
    stub = load_world_model(WORLD_MODEL_SKELETON)
    assert stub.ok
    result = replay(transcript, stub.world_model)
    assert not result.passed
    assert result.pass_count < result.total


# =====================================================================
# extract_code: the two parsing bugs that burned draft attempts
# =====================================================================


def test_extract_code_recovers_a_truncated_unterminated_fence():
    """A response cut off by max_tokens has an opening fence and no
    closing one. The old code fell through to 'return the whole response',
    which starts with '```python' and is a guaranteed SyntaxError."""
    truncated = "Sure:\n\n```python\nclass WorldModel:\n    def predict(self, s, a, x=None, y=None):\n        return s, 0, False\n"
    code = extract_code(truncated)
    assert not code.startswith("```")
    assert code.startswith("class WorldModel:")


def test_extract_code_ignores_a_sketch_inside_a_think_block():
    """Qwen3 emits <think>...</think>. A throwaway sketch in there is not
    the answer; the old 'first fence wins' rule extracted it anyway."""
    response = (
        "<think>\nrough idea:\n```python\nclass WorldModel: pass\n```\n</think>\n"
        "```python\n" + CORRECT_WORLD_MODEL_SOURCE + "```\n"
    )
    code = extract_code(response)
    assert "def goal_hint" in code
    assert code.strip() != "class WorldModel: pass"


def test_extract_code_prefers_the_block_defining_worldmodel():
    response = (
        "First, a helper sketch:\n```python\nGRID = 64\n```\n"
        "Now the model:\n```python\n" + CORRECT_WORLD_MODEL_SOURCE + "```\n"
    )
    assert "class WorldModel" in extract_code(response)


def test_extract_code_handles_a_plain_unfenced_response():
    assert extract_code(CORRECT_WORLD_MODEL_SOURCE).startswith("class WorldModel:")


# =====================================================================
# Retry prompts must keep the transcript (they used to drop it)
# =====================================================================


def test_retry_prompt_still_contains_the_transcript():
    transcript = build_transcript()
    client = FakeLLMClient(["```python\nthis is not python(\n```", fenced(CORRECT_WORLD_MODEL_SOURCE)])

    outcome = draft_world_model(client, transcript, max_attempts=3)

    assert outcome.ok, "should recover on the second attempt"
    assert len(client.prompts) == 2
    _system, second_user = client.prompts[1]
    assert "transition #0" in second_user, "the retry prompt dropped the transcript"
    assert "diff after action" in second_user


# =====================================================================
# exec sandbox: ordinary generated Python must not be rejected
# =====================================================================


@pytest.mark.parametrize("snippet, label", [
    ("class WorldModel:\n    def __init__(self):\n        super().__init__()\n"
     "    def predict(self, s, a, x=None, y=None):\n        return s, 0, False\n"
     "    def goal_hint(self, s):\n        return 0.0\n", "super()"),
    ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n"
     "        return [list(map(list, l)) for l in s], 0, False\n"
     "    def goal_hint(self, s):\n        return 0.0\n", "map()"),
    ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n"
     "        return [[list(reversed(r)) for r in l] for l in s], 0, False\n"
     "    def goal_hint(self, s):\n        return 0.0\n", "reversed()"),
])
def test_sandbox_accepts_ordinary_generated_python(snippet, label):
    loaded = load_world_model(snippet)
    assert loaded.ok, f"{label}: {loaded.error}"
    _s, _d, _dn, error = safe_predict(loaded.world_model, with_agent_at(1, 1), Action(name="ACTION1"))
    assert error is None, f"{label}: {error}"


def test_sandbox_still_refuses_imports():
    loaded = load_world_model("import os\nclass WorldModel:\n    pass\n")
    assert not loaded.ok
    assert "__import__" in (loaded.error or "")


# =====================================================================
# Defect 4: the diff.py TypeError
# =====================================================================


# The exact shapes that raised `TypeError: object of type 'int' has no
# len()` at diff.py:18 before this branch. Each is a plausible slip by
# LLM-authored predict() code.
MALFORMED_GRIDS = [
    pytest.param([[0, 1], [2, 3]][:1], id="layer-is-a-flat-row"),
    pytest.param([[0, 1]], id="single-layer-of-ints"),
    pytest.param(0, id="bare-int"),
    pytest.param([7], id="list-of-ints"),
    pytest.param([[[0, 1], [2, "x"]]], id="non-int-cell"),
    pytest.param([], id="empty"),
    pytest.param(None, id="none"),
    pytest.param("nope", id="string"),
]

REAL_GRID = [[[0, 1], [2, 3]]]


@pytest.mark.parametrize("bad", MALFORMED_GRIDS)
def test_format_diff_never_raises_on_a_malformed_grid(bad):
    out = format_diff(bad, REAL_GRID)
    assert isinstance(out, str) and out.startswith("<shape mismatch:")
    assert isinstance(format_diff(REAL_GRID, bad), str)


@pytest.mark.parametrize("bad", MALFORMED_GRIDS)
def test_grid_shape_mismatch_never_raises(bad):
    assert isinstance(grid_shape_mismatch(bad, REAL_GRID), str)


@pytest.mark.parametrize("bad", MALFORMED_GRIDS)
def test_format_grid_never_raises(bad):
    assert format_grid(bad).startswith("<unrenderable:")


def test_format_grid_still_renders_a_real_grid():
    assert format_grid([[[0, 1], [10, 15]]]) == "01\naf"


def test_the_exact_kernel_traceback_input_no_longer_raises():
    """The scored run died with TypeError at diff.py:18 inside replay(),
    when an LLM-authored predict() returned one row instead of a grid.
    Same path, same shape -- must now be an ordinary replay failure with
    an actionable reason."""
    source = (
        "class WorldModel:\n"
        "    def predict(self, state, action_name, x=None, y=None):\n"
        "        return [state[0][0]], 0, False\n"
        "    def goal_hint(self, state):\n        return 0.0\n"
    )
    loaded = load_world_model(source)
    assert loaded.ok

    transcript = GameTranscript(game_id="g")
    transcript.append(Transition(
        frame_before=REAL_GRID,
        action=Action(name="ACTION1"),
        frame_after=[[[9, 9], [9, 9]]],
        levels_completed_before=0,
        levels_completed_after=0,
        state_after="NOT_FINISHED",
    ))

    result = replay(transcript, loaded.world_model)  # must not raise
    assert not result.passed
    assert result.first_failure is not None
    assert "next_state" in (result.first_failure.reason or "")


def test_describe_malformed_accepts_a_well_formed_grid():
    assert describe_malformed(REAL_GRID) is None


def test_safe_predict_rejects_bad_return_shapes():
    for src, expect in [
        ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n        return s\n"
         "    def goal_hint(self, s):\n        return 0.0\n", "expected exactly 3"),
        ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n        return s, 0.5, False\n"
         "    def goal_hint(self, s):\n        return 0.0\n", "levels_delta"),
        ("class WorldModel:\n    def predict(self, s, a, x=None, y=None):\n        return s, 0, 1\n"
         "    def goal_hint(self, s):\n        return 0.0\n", "done"),
    ]:
        loaded = load_world_model(src)
        assert loaded.ok, loaded.error
        _s, _d, _dn, error = safe_predict(loaded.world_model, REAL_GRID, Action(name="ACTION1"))
        assert error is not None and expect in error, (src, error)


# =====================================================================
# Defect 4 (agent side): choose_action must never propagate an exception
# =====================================================================


def make_agent(module, coder_responses=None, action_responses=None):
    """A CodeWorldAgent with fake LLM clients, bypassing make_client."""
    agent = module.CodeWorldAgent.__new__(module.CodeWorldAgent)
    import random as _random

    agent.game_id = "test-game"
    agent.agent_name = "codeworldagent"
    agent._rng = _random.Random(0)
    agent._init_failed = False
    agent.transcript = __import__("llm_engine.types", fromlist=["GameTranscript"]).GameTranscript(game_id="test-game")
    agent._probe_plan = __import__("llm_engine.opening_probes", fromlist=["opening_probe_plan"]).opening_probe_plan()
    agent._probe_index = 0
    agent.model = None
    agent.model_source = None
    agent.model_version = 0
    agent._pending_action = None
    agent._pending_frame_before = None
    agent._pending_levels_before = 0
    agent._last_draft_attempt_len = -1
    agent._consecutive_repair_failures = 0
    from llm_engine.budget import LLMBudget

    agent.coder_budget = LLMBudget(max_calls_per_game=module.CodeWorldAgent.CODER_LLM_CALL_BUDGET)
    agent.action_budget = LLMBudget(max_calls_per_game=module.CodeWorldAgent.ACTION_LLM_CALL_BUDGET)
    agent.coder_client = FakeLLMClient(coder_responses or [])
    agent.action_client = FakeLLMClient(action_responses or [])
    return agent


def make_frame(state_value="NOT_FINISHED", grid=None, available=None):
    from arcengine import FrameData, GameState

    return FrameData(
        game_id="test-game",
        frame=grid if grid is not None else [[[0] * 64 for _ in range(64)]],
        state=GameState(state_value),
        levels_completed=0,
        available_actions=available if available is not None else [1, 2, 3, 4, 5, 6, 7],
    )


def assert_legal(action):
    from arcengine import GameAction

    assert isinstance(action, GameAction)


def test_choose_action_survives_a_raising_world_model(code_world_agent_module):
    module = code_world_agent_module
    agent = make_agent(module)
    agent._probe_index = len(agent._probe_plan)  # probes done

    class Exploding:
        def predict(self, *a, **k):
            raise RuntimeError("boom")

        def goal_hint(self, *a, **k):
            raise RuntimeError("boom")

    agent.model = Exploding()
    agent.model_source = "class WorldModel: pass"
    assert_legal(agent.choose_action([], make_frame()))


def test_choose_action_survives_a_raising_planner(code_world_agent_module, monkeypatch):
    module = code_world_agent_module
    agent = make_agent(module)
    agent._probe_index = len(agent._probe_plan)
    agent.model = load_world_model(CORRECT_WORLD_MODEL_SOURCE).world_model
    agent.model_source = CORRECT_WORLD_MODEL_SOURCE

    def explode(*_a, **_k):
        raise ValueError("planner exploded")

    monkeypatch.setattr(module, "next_action", explode)
    assert_legal(agent.choose_action([], make_frame()))


def test_choose_action_survives_a_raising_diff(code_world_agent_module, monkeypatch):
    """Even if diff.py were re-broken, the agent must not lose the game."""
    module = code_world_agent_module
    import llm_engine.diff as diff_mod

    def explode(*_a, **_k):
        raise TypeError("object of type 'int' has no len()")

    monkeypatch.setattr(diff_mod, "format_diff", explode)
    monkeypatch.setattr(diff_mod, "grid_shape_mismatch", explode)

    agent = make_agent(module, coder_responses=[fenced(CORRECT_WORLD_MODEL_SOURCE)])
    agent._probe_index = len(agent._probe_plan)
    assert_legal(agent.choose_action([], make_frame()))


def test_choose_action_survives_an_llm_client_that_raises(code_world_agent_module):
    module = code_world_agent_module
    agent = make_agent(module)
    agent.coder_client = RaisingLLMClient()
    agent.action_client = RaisingLLMClient()
    agent._probe_index = len(agent._probe_plan)
    agent.transcript = build_transcript()
    assert_legal(agent.choose_action([], make_frame()))
    assert agent.model is None


def test_choose_action_survives_a_garbage_frame(code_world_agent_module):
    module = code_world_agent_module
    agent = make_agent(module)
    agent._probe_index = len(agent._probe_plan)

    class Nonsense:
        state = "not-a-GameState"
        frame = 17
        levels_completed = "many"
        available_actions = None

    assert_legal(agent.choose_action([], Nonsense()))


def test_choose_action_falls_back_when_init_failed(code_world_agent_module):
    module = code_world_agent_module
    agent = make_agent(module)
    agent._init_failed = True
    action = agent.choose_action([], make_frame())
    assert_legal(action)
    assert "fallback" in (action.reasoning or "")


def test_is_done_never_raises(code_world_agent_module):
    agent = make_agent(code_world_agent_module)

    class Nonsense:
        pass

    assert agent.is_done([], Nonsense()) is False


def test_fallback_respects_available_actions(code_world_agent_module):
    from arcengine import GameAction

    agent = make_agent(code_world_agent_module)
    for _ in range(30):
        action = agent._safe_fallback_action(make_frame(available=[3]))
        assert action is GameAction.ACTION3


# =====================================================================
# Defect 3: repair failures are downstream of defect 1
# =====================================================================


def test_repeated_repair_failure_discards_the_model_instead_of_looping(code_world_agent_module):
    module = code_world_agent_module
    # A model that loads but is wrong about everything, and a coder that
    # never manages to fix it -- the exact shape of the scored run's
    # "repair failed after 3 attempts", repeatedly.
    wrong = ("class WorldModel:\n"
             "    def predict(self, s, a, x=None, y=None):\n        return s, 0, False\n"
             "    def goal_hint(self, s):\n        return 0.0\n")
    agent = make_agent(module, coder_responses=[fenced(wrong)] * 40)
    agent.model = load_world_model(wrong).world_model
    agent.model_source = wrong

    transcript = build_transcript()
    for t in transcript.transitions[: module.CodeWorldAgent.MAX_CONSECUTIVE_REPAIR_FAILURES]:
        agent._handle_new_transition(t)

    assert agent.model is None, "agent kept repairing an unfixable model"
    assert agent._consecutive_repair_failures == 0


def test_repair_returns_not_ok_without_clobbering_the_previous_model():
    transcript = build_transcript()
    good = CORRECT_WORLD_MODEL_SOURCE
    client = FakeLLMClient(["garbage("] * 5)
    # Give the repair loop a model that passes replay -> nothing to repair.
    outcome = repair_world_model(client, transcript, good, max_attempts=2)
    assert outcome.ok and outcome.source == good
