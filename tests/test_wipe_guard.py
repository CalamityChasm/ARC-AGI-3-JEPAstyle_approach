"""The world-model wipe guard: does it change exactly one thing, and nothing else?

The guard is a wrapper installed over
``inference.agent.tool_agent.ToolAgent._update_summarized_knowledge_from_step_summary``
in the notebook's customization cell. These tests stand up a faithful stub of
that module -- the wipe method's body is copied **byte-for-byte** from
``jakobbrggen/taaf-kaggle-source-anim-20260807-anim``
(``src/ARC3-Inference/inference/agent/tool_agent.py`` lines 1343-1356, sha256
``856bf9b8...f8be7``) -- exec the real cell source against it, and check the
resulting behaviour case by case.

Run: venv/Scripts/python.exe -m pytest tests/test_wipe_guard.py -q
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parents[1] / "scripts" / "wipe_guard_cell.py"

# Copied byte-for-byte out of the anim bundle. If upstream rewords this, the
# guard's own assertions are supposed to fail loudly -- test_fails_loudly_*
# below is what proves they do.
UPSTREAM_WIPE = '''\
    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in (
                "world_model",
                "goal_model",
                "action_model",
                "recent_findings",
                "open_questions",
                "current_plan",
            ):
                self._summarized_knowledge[key] = ""
'''

WIPED_FIELDS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
)


def _make_stub(tmp_path: Path, wipe_src: str = UPSTREAM_WIPE) -> Path:
    """Write a minimal `inference.agent.tool_agent` carrying the real method."""
    root = tmp_path / "stub"
    pkg = root / "inference" / "agent"
    pkg.mkdir(parents=True)
    (root / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tool_agent.py").write_text(
        "class ToolAgent:\n"
        "    def __init__(self):\n"
        "        self._last_step_summary = None\n"
        "        self._session_runtime_dir = None\n"
        "        self._summarized_knowledge = {\n"
        + "".join(f'            "{k}": "",\n' for k in WIPED_FIELDS)
        + '            "cross_level_notes": "",\n'
        "        }\n"
        "\n" + wipe_src,
        encoding="utf-8",
    )
    return root


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wipe_src: str = UPSTREAM_WIPE):
    """Exec the real cell against a fresh stub; return (module, cell namespace)."""
    root = _make_stub(tmp_path, wipe_src)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    ns: dict = {}
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), ns)  # noqa: S102
    import inference.agent.tool_agent as mod

    return mod, ns


def _agent(mod, summary):
    agent = mod.ToolAgent()
    agent._summarized_knowledge = {k: f"<{k}>" for k in WIPED_FIELDS}
    agent._summarized_knowledge["cross_level_notes"] = "<cross_level_notes>"
    agent._last_step_summary = summary
    return agent


def _kept(agent) -> bool:
    return all(agent._summarized_knowledge[k] == f"<{k}>" for k in WIPED_FIELDS)


def _erased(agent) -> bool:
    return all(agent._summarized_knowledge[k] == "" for k in WIPED_FIELDS)


# --- the one thing it changes -------------------------------------------------


def test_in_level_game_over_keeps_the_world_model(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _agent(mod, {"game_over": True, "level": 3, "end_action_num": 42})
    agent._update_summarized_knowledge_from_step_summary()
    assert _kept(agent)


def test_unguarded_upstream_would_have_wiped_the_same_case(tmp_path, monkeypatch):
    """The control: without the guard, that exact summary erases six fields."""
    root = _make_stub(tmp_path)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    import inference.agent.tool_agent as mod

    agent = _agent(mod, {"game_over": True, "level": 3})
    agent._update_summarized_knowledge_from_step_summary()
    assert _erased(agent)


# --- everything it must NOT change --------------------------------------------


@pytest.mark.parametrize(
    "summary",
    [
        {"level_transition": True},
        {"run_complete": True},
        {"game_over": True, "level_transition": True},
        {"game_over": True, "run_complete": True},
        {"game_over": True, "level_transition": True, "run_complete": True},
    ],
)
def test_level_transition_and_run_completion_still_wipe(tmp_path, monkeypatch, summary):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _agent(mod, summary)
    agent._update_summarized_knowledge_from_step_summary()
    assert _erased(agent)


@pytest.mark.parametrize("summary", [None, {}, {"board_changed": True}, {"game_over": False}])
def test_ordinary_steps_are_untouched(tmp_path, monkeypatch, summary):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _agent(mod, summary)
    agent._update_summarized_knowledge_from_step_summary()
    assert _kept(agent)


def test_cross_level_notes_is_never_touched(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    for summary in ({"game_over": True}, {"level_transition": True}, {"run_complete": True}):
        agent = _agent(mod, summary)
        agent._update_summarized_knowledge_from_step_summary()
        assert agent._summarized_knowledge["cross_level_notes"] == "<cross_level_notes>"


# --- it can never kill a game -------------------------------------------------


def test_a_hostile_summary_does_not_raise(tmp_path, monkeypatch):
    mod, ns = _install(tmp_path, monkeypatch)

    class Hostile(dict):
        def get(self, *a, **kw):
            raise RuntimeError("boom")

    agent = _agent(mod, Hostile({"game_over": True}))
    agent._update_summarized_knowledge_from_step_summary()  # must not raise
    assert ns["_WG_STATS"]["errors"] >= 1


def test_a_broken_knowledge_dict_does_not_raise(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    agent = _agent(mod, {"level_transition": True})
    agent._summarized_knowledge = None  # upstream would TypeError here
    agent._update_summarized_knowledge_from_step_summary()  # must not raise


# --- it fails fast rather than degrading silently -----------------------------


def test_fails_loudly_if_upstream_stops_wiping_on_game_over(tmp_path, monkeypatch):
    reworded = UPSTREAM_WIPE.replace('or summary.get("game_over")', "")
    with pytest.raises(AssertionError, match="upstream wipe changed"):
        _install(tmp_path, monkeypatch, reworded)


def test_fails_loudly_if_upstream_renames_a_wiped_field(tmp_path, monkeypatch):
    reworded = UPSTREAM_WIPE.replace('"recent_findings"', '"findings"')
    with pytest.raises(AssertionError, match="upstream wipe changed"):
        _install(tmp_path, monkeypatch, reworded)


def test_fails_loudly_if_upstream_starts_wiping_cross_level_notes(tmp_path, monkeypatch):
    reworded = UPSTREAM_WIPE.replace(
        '                "current_plan",\n',
        '                "current_plan",\n                "cross_level_notes",\n',
    )
    with pytest.raises(AssertionError, match="cross_level_notes"):
        _install(tmp_path, monkeypatch, reworded)


def test_installing_twice_fails_loudly(tmp_path, monkeypatch):
    mod, _ = _install(tmp_path, monkeypatch)
    ns: dict = {}
    with pytest.raises(AssertionError, match="installed twice"):
        exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), ns)  # noqa: S102


# --- counters and accounting ---------------------------------------------------


def test_probe_traffic_does_not_pollute_the_run_counters(tmp_path, monkeypatch):
    _, ns = _install(tmp_path, monkeypatch)
    assert ns["_WG_STATS"] == {"kept": 0, "wiped": 0, "noop": 0, "errors": 0}


def test_counters_track_the_three_branches(tmp_path, monkeypatch):
    mod, ns = _install(tmp_path, monkeypatch)
    for summary in (
        {"game_over": True},
        {"game_over": True},
        {"level_transition": True},
        {"board_changed": True},
    ):
        _agent(mod, summary)._update_summarized_knowledge_from_step_summary()
    assert ns["_WG_STATS"] == {"kept": 2, "wiped": 1, "noop": 1, "errors": 0}


def test_the_guard_is_the_only_method_replaced(tmp_path, monkeypatch):
    """Nothing else on ToolAgent is rebound."""
    root = _make_stub(tmp_path)
    monkeypatch.syspath_prepend(str(root))
    for name in [m for m in sys.modules if m == "inference" or m.startswith("inference.")]:
        del sys.modules[name]
    import inference.agent.tool_agent as mod

    before = {k: v for k, v in vars(mod.ToolAgent).items() if callable(v)}
    exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), {})  # noqa: S102
    after = {k: v for k, v in vars(mod.ToolAgent).items() if callable(v)}
    changed = [k for k in after if before.get(k) is not after[k]]
    assert changed == ["_update_summarized_knowledge_from_step_summary"], changed


def test_no_knob_or_env_is_touched(tmp_path, monkeypatch):
    """One variable: the cell must not set env vars or analyzer knobs."""
    src = CELL.read_text(encoding="utf-8")
    for forbidden in ("os.environ", "LOCAL_ANALYZER", "bm.solver", "MULTIMODAL", "setdefault("):
        assert forbidden not in src, forbidden


def test_cell_text_matches_the_source_in_the_notebook(tmp_path, monkeypatch):
    """The notebook must carry this exact cell, not a drifted copy."""
    import json

    nb_path = (
        Path(__file__).resolve().parents[1]
        / "kaggle_submission_duck_nvfp4_anim_wipeguard"
        / "notebook"
        / "arc3-duck-nvfp4-anim-wg.ipynb"
    )
    if not nb_path.exists():
        pytest.skip("notebook not built yet")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert any(c.strip() == CELL.read_text(encoding="utf-8").strip() for c in cells), textwrap.shorten(
        "guard cell in the notebook does not match scripts/wipe_guard_cell.py", 200
    )


def test_built_arm_is_current_and_inherits_every_anim_cell():
    """The pushed notebook must be what the builder produces from today's anim arm."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_wg_builder", Path(__file__).resolve().parents[1] / "scripts" / "_build_duck_nvfp4_anim_wipeguard.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(check_only=True)  # raises if any inherited cell drifted or the build is stale
