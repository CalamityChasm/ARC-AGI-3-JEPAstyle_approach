"""The commit floor: does it change exactly one thing, and nothing else?

The mechanism is two wrappers installed over
``inference.agent.tool_agent.ToolAgent.analyze`` and ``._build_user_prompt``
in the notebook's customization cell. After two consecutive ``analyze()`` turns
that executed no game action, the next turn's user prompt carries a directive
requiring that turn to end in ``action(...)``.

Unlike the wipe guard -- whose upstream method is 14 lines and was copied
byte-for-byte into its stub -- ``analyze`` is ~350 lines. The stub here is a
**faithful reduction**: it really calls ``_ensure_session``, really builds the
user prompt through ``_build_user_prompt``, and really returns an
``AnalyzerTurnResult`` carrying ``step_executed`` and ``yielded_control``, with
every line the cell's ``inspect.getsource`` assertions look for present
verbatim. Those lines are quoted from
``jakobbrggen/taaf-kaggle-source-anim-20260807-anim``
``src/ARC3-Inference/inference/agent/tool_agent.py`` (sha256
``856bf9b895d0ad8b959c8f828c7132b0e09eaa47f4c5cc6173785354090f8be7``) and
``inference/framework/solver.py`` (sha256
``2bef5d6bc23c0312675f0c7203194c94e93d056ac06bf6419acd5142a4ea7c8e``).

Run: venv/Scripts/python.exe -m pytest tests/test_commit_floor.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CELL = REPO / "scripts" / "commit_floor_cell.py"
NOTEBOOK = (
    REPO
    / "kaggle_submission_duck_nvfp4_anim_commitfloor"
    / "notebook"
    / "arc3-duck-nvfp4-anim-cf.ipynb"
)

# The upstream line the prompt directive is appended after, byte-for-byte
# (tool_agent.py:1484).
UPSTREAM_ACT_LINE = (
    "When ready, call `action(actions)` from inside the `python` tool with the best valid "
    "action or ordered batch selected by your code. If your code has found a reliable short "
    "sequence, prefer batching it in one call."
)

TOOL_AGENT_STUB = '''\
class AnalyzerTurnResult:
    def __init__(self, step_executed=False, yielded_control=False, retryable_failure=False):
        self.step_executed = step_executed
        self.yielded_control = yielded_control
        self.retryable_failure = retryable_failure


class ToolAgent:
    """A faithful reduction of the anim bundle's ToolAgent, for the parts the
    commit floor touches. `_script` drives the outcome of each turn."""

    def __init__(self, script=()):
        self._script = list(script)
        self._i = 0
        self._session_runtime_dir = None
        self._prompts = []
        self._raise_on = None

    def _ensure_session(self, state_path):
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir

    def _build_user_prompt(
        self,
        action_num,
        *,
        valid_actions=None,
        current_frame=None,
        history_entries=None,
        previous_step_summary=None,
    ):
        lines = [
            "The code executed 1 actions in the previous sequence.",
            {ACT_LINE!r},
        ]
        return "\\n".join(lines)

    def analyze(self, state_path, action_num=0, **kwargs):
        self._ensure_session(state_path)
        user_prompt = self._build_user_prompt(action_num, valid_actions=None)
        self._prompts.append(user_prompt)
        if self._raise_on is not None and self._i == self._raise_on:
            self._i += 1
            raise RuntimeError("synthetic upstream failure")
        outcome = self._script[self._i]
        self._i += 1
        if outcome is None:
            return None
        step_executed = bool(outcome)
        yielded_control_reason = None if step_executed else "turn_time_budget"
        return AnalyzerTurnResult(
            step_executed=step_executed,
            yielded_control=yielded_control_reason is not None,
        )
'''

SOLVER_STUB = '''\
def run():
    retry_analysis_step = None
    analysis_step = 0
    result = None
    if getattr(result, "yielded_control", False):
        retry_analysis_step = analysis_step
    return retry_analysis_step
'''


class _P:
    """Stands in for a state_path: the mechanism only reads `.parent`."""

    def __init__(self, parent):
        self.parent = parent


def _make_stub(tmp_path: Path, tool_agent_src: str | None = None, solver_src: str | None = None) -> Path:
    root = tmp_path / "stub"
    agent_pkg = root / "inference" / "agent"
    fw_pkg = root / "inference" / "framework"
    agent_pkg.mkdir(parents=True)
    fw_pkg.mkdir(parents=True)
    (root / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (agent_pkg / "__init__.py").write_text("", encoding="utf-8")
    (fw_pkg / "__init__.py").write_text("", encoding="utf-8")
    (agent_pkg / "tool_agent.py").write_text(
        tool_agent_src if tool_agent_src is not None else TOOL_AGENT_STUB.format(ACT_LINE=UPSTREAM_ACT_LINE),
        encoding="utf-8",
    )
    (fw_pkg / "solver.py").write_text(
        solver_src if solver_src is not None else SOLVER_STUB, encoding="utf-8"
    )
    return root


def _install(tmp_path, monkeypatch, tool_agent_src=None, solver_src=None):
    root = _make_stub(tmp_path, tool_agent_src, solver_src)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    ns: dict = {}
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), ns)  # noqa: S102
    import inference.agent.tool_agent as mod

    return mod, ns


def _drive(mod, script, sessions=None):
    agent = mod.ToolAgent(script)
    sessions = sessions or ["game-a"] * len(script)
    for s in sessions:
        agent.analyze(_P(s))
    return agent


def _fired(agent):
    return [i for i, p in enumerate(agent._prompts) if "HARNESS NOTICE" in p]


# --------------------------------------------------------------------------- install


def test_installs_and_rebinds_exactly_two_methods(tmp_path, monkeypatch, capsys):
    mod, ns = _install(tmp_path, monkeypatch)
    assert mod.ToolAgent.analyze is ns["_commit_floor_analyze"]
    assert mod.ToolAgent._build_user_prompt is ns["_commit_floor_prompt"]
    changed = [
        name
        for name in dir(mod.ToolAgent)
        if getattr(getattr(mod.ToolAgent, name, None), "_commit_floor_installed", False)
    ]
    assert sorted(changed) == ["_build_user_prompt", "analyze"], changed
    out = capsys.readouterr().out
    assert "COMMIT_FLOOR_INSTALLED" in out
    assert "probe=5/5" in out


def test_refuses_to_install_twice(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match="installed twice"):
        exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), {})  # noqa: S102


# --------------------------------------------------------------------------- the one change


def test_no_directive_below_the_threshold(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False, False])
    assert _fired(agent) == []


def test_fires_on_the_third_consecutive_dead_turn(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False, False, False, False])
    assert _fired(agent) == [2, 3]
    assert "The last 2 turns" in agent._prompts[2]
    assert "MUST end with a call to `action(...)`" in agent._prompts[2]


def test_an_executed_turn_resets_the_pressure(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False, False, False, True, False, False, False])
    assert _fired(agent) == [2, 3, 6]


def test_escalates_only_after_five(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False] * 8)
    strong = [i for i, p in enumerate(agent._prompts) if "Do not make any inspection-only" in p]
    assert strong == [5, 6, 7], strong


def test_a_new_session_resets_the_pressure(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False] * 4, sessions=["a", "a", "b", "b"])
    assert _fired(agent) == []


def test_upstream_returning_none_is_neutral(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False, None, None, False, False])
    assert _fired(agent) == [4]


def test_the_base_prompt_is_always_an_exact_prefix(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False] * 6)
    base = agent._prompts[0]
    assert UPSTREAM_ACT_LINE in base
    for p in agent._prompts:
        assert p.startswith(base), p[: len(base) + 40]
    # and the untouched turns are byte-identical to upstream's own output
    assert agent._prompts[0] == agent._prompts[1] == base


def test_directive_never_contradicts_the_prompts_own_act_instruction(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _drive(mod, [False] * 3)
    fired = agent._prompts[2]
    # it asks for an action; it must not forbid the tool the prompt requires
    assert "action(...)" in fired
    assert "do not call" not in fired.lower()


# --------------------------------------------------------------------------- counters


def test_counters_are_the_free_run_readout(tmp_path, monkeypatch, capsys):
    mod, ns = _install(tmp_path, monkeypatch)
    capsys.readouterr()
    agent = _drive(mod, [False, False, False, True, False, False, False])
    stats = ns["_CF_STATS"]
    assert stats["turns"] == 7
    assert stats["dead"] == 6
    assert stats["fired"] == 3          # turns 2, 3, 6
    assert stats["converted"] == 1      # turn 3 executed under pressure
    assert stats["unconverted"] == 2
    assert stats["errors"] == 0
    out = capsys.readouterr().out
    assert out.count("COMMIT_FLOOR_FIRED") == 3
    assert out.count("COMMIT_FLOOR_RESULT") == 3
    ns["_cf_report"]()
    assert "COMMIT_FLOOR_FINAL turns=7 dead=6 fired=3" in capsys.readouterr().out


# --------------------------------------------------------------------------- safety


def test_an_upstream_exception_propagates_unchanged(tmp_path, monkeypatch):
    """The solver raises on a None result, so swallowing here would change behaviour."""
    mod, ns = _install(tmp_path, monkeypatch)
    agent = mod.ToolAgent([False, False, False])
    agent._raise_on = 0
    with pytest.raises(RuntimeError, match="synthetic upstream failure"):
        agent.analyze(_P("a"))
    assert ns["_CF_STATS"]["errors"] == 0


def test_hostile_pressure_state_cannot_break_the_prompt(tmp_path, monkeypatch, capsys):
    mod, ns = _install(tmp_path, monkeypatch)
    agent = mod.ToolAgent([True])

    class _Boom:
        def __int__(self):
            raise RuntimeError("synthetic hostile pressure")

    agent._cf_pressure = _Boom()
    text = agent._build_user_prompt(0, valid_actions=None)
    assert text.endswith(UPSTREAM_ACT_LINE)
    assert ns["_CF_STATS"]["errors"] == 1
    assert "COMMIT_FLOOR_ERROR prompt" in capsys.readouterr().out


# --------------------------------------------------------------------------- fails loudly


@pytest.mark.parametrize(
    "drop",
    [
        "self._build_user_prompt(",
        "self._ensure_session(state_path)",
        "yielded_control=yielded_control_reason is not None",
        "step_executed=step_executed",
        "turn_time_budget",
    ],
)
def test_fails_loudly_when_analyze_drifts(tmp_path, monkeypatch, drop):
    src = TOOL_AGENT_STUB.format(ACT_LINE=UPSTREAM_ACT_LINE)
    if drop == "self._build_user_prompt(":
        src = src.replace("user_prompt = self._build_user_prompt(action_num, valid_actions=None)",
                          "user_prompt = 'inlined'")
    elif drop == "self._ensure_session(state_path)":
        src = src.replace("        self._ensure_session(state_path)\n", "")
    else:
        src = src.replace(drop, drop.replace("_", "X"))
    with pytest.raises(AssertionError, match="commit floor"):
        _install(tmp_path, monkeypatch, tool_agent_src=src)


def test_fails_loudly_when_the_prompt_builder_drifts(tmp_path, monkeypatch):
    src = TOOL_AGENT_STUB.format(ACT_LINE="some reworded instruction")
    with pytest.raises(AssertionError, match="commit floor"):
        _install(tmp_path, monkeypatch, tool_agent_src=src)


def test_fails_loudly_when_the_prompt_stops_returning_a_joined_list(tmp_path, monkeypatch):
    src = TOOL_AGENT_STUB.format(ACT_LINE=UPSTREAM_ACT_LINE).replace(
        'return "\\n".join(lines)', "return chr(10).join(lines)"
    )
    with pytest.raises(AssertionError, match="commit floor"):
        _install(tmp_path, monkeypatch, tool_agent_src=src)


def test_fails_loudly_when_the_solver_stops_retrying_a_yielded_step(tmp_path, monkeypatch):
    bad = SOLVER_STUB.replace("retry_analysis_step = analysis_step", "pass")
    with pytest.raises(AssertionError, match="no longer retries"):
        _install(tmp_path, monkeypatch, solver_src=bad)


def test_fails_loudly_when_the_solver_stops_branching_on_yield(tmp_path, monkeypatch):
    bad = SOLVER_STUB.replace('if getattr(result, "yielded_control", False):', "if False:")
    with pytest.raises(AssertionError, match="yielded_control"):
        _install(tmp_path, monkeypatch, solver_src=bad)


# --------------------------------------------------------------------------- one variable


def test_the_cell_touches_no_knob_and_no_serving_setting():
    src = CELL.read_text(encoding="utf-8")
    for forbidden in ("os.environ", "LOCAL_ANALYZER", "bm.solver", "setup_commands", "SETUP_ENV_PATH"):
        assert forbidden not in src, forbidden


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not built yet")
def test_the_notebook_carries_this_exact_cell():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    bodies = ["".join(c["source"]) for c in nb["cells"]]
    assert any(b.strip() == CELL.read_text(encoding="utf-8").strip() for b in bodies)
