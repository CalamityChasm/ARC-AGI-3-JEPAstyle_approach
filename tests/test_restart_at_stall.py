"""Restart-at-stall: does it change exactly one thing, and nothing else?

The mechanism is a wrapper installed over
``inference.agent.tool_agent.ToolAgent.analyze`` in the notebook's
customization cell. These tests stand up a faithful stub of that module --
``_ensure_session`` and ``_empty_world_model`` are copied **byte-for-byte** from
``jakobbrggen/taaf-kaggle-source-anim-20260807-anim``
(``src/ARC3-Inference/inference/agent/tool_agent.py``, sha256
``856bf9b895d0ad8b959c8f828c7132b0e09eaa47f4c5cc6173785354090f8be7``: lines
448-457 and 1140-1152), and ``analyze``'s signature and frame-resolution line
are reproduced from lines 2090-2110 -- then exec the real cell source against it
and check the resulting behaviour case by case.

Run: venv/Scripts/python.exe -m pytest tests/test_restart_at_stall.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parents[1] / "scripts" / "restart_at_stall_cell.py"

# --- byte-for-byte from the anim bundle ---------------------------------------------------
UPSTREAM_EMPTY_WORLD_MODEL = '''\
def _empty_world_model() -> dict[str, str]:
    return {
        "world_model": "",
        "goal_model": "",
        "action_model": "",
        "recent_findings": "",
        "open_questions": "",
        "current_plan": "",
        "cross_level_notes": "",
    }
'''

UPSTREAM_ENSURE_SESSION = '''\
    def _ensure_session(self, state_path: Path) -> None:
        runtime_dir = state_path.parent
        if self._session_runtime_dir != runtime_dir:
            self._session_runtime_dir = runtime_dir
            self._history_messages = []
            self._session_total_tokens = 0
            self._session_generated_tokens = 0
            self._last_step_summary = None
            self._last_action_result = None
            self._summarized_knowledge = _empty_world_model()
'''

# analyze(): the real signature, the real frame-resolution line, and the real
# per-request seed read. The body is reduced to what the wrapper inspects plus a
# recorder, so the stub has no network or filesystem dependencies.
UPSTREAM_ANALYZE = '''\
    def analyze(
        self,
        state_path,
        action_num: int,
        valid_actions=None,
        step_env=None,
        transcript_path=None,
        analysis_step=None,
        transcript_updated=None,
        request_timeout_seconds=None,
        should_stop=None,
    ):
        if not state_path.exists():
            return None
        current_frame, history_entries = load_runtime_state(state_path)
        payload = build_chat_payload(
            seed=_LOCAL_ANALYZER_SEED,
        )
        self.calls.append((action_num, analysis_step, payload["seed"]))
        return "RESULT"
'''

FIELDS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
    "cross_level_notes",
)


def _make_stub(
    tmp_path: Path,
    *,
    empty_world_model: str = UPSTREAM_EMPTY_WORLD_MODEL,
    ensure_session: str = UPSTREAM_ENSURE_SESSION,
    analyze: str = UPSTREAM_ANALYZE,
    seed: int = 20260825,
) -> Path:
    root = tmp_path / "stub"
    pkg = root / "inference" / "agent"
    pkg.mkdir(parents=True)
    (root / "inference" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tool_agent.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "\n"
        "class Frame:\n"
        "    def __init__(self, level):\n"
        "        self.level = level\n"
        "\n"
        "\n"
        "LEVELS = {}\n"
        "\n"
        "\n"
        "def load_runtime_state(state_path):\n"
        "    key = str(state_path)\n"
        "    if key not in LEVELS:\n"
        "        raise FileNotFoundError(key)\n"
        "    return Frame(LEVELS[key]), []\n"
        "\n"
        "\n"
        "def build_chat_payload(**kw):\n"
        "    return dict(kw)\n"
        "\n"
        "\n"
        f"_LOCAL_ANALYZER_SEED = {seed}\n"
        "\n"
        "\n" + empty_world_model + "\n"
        "\n"
        "class ToolAgent:\n"
        "    def __init__(self):\n"
        "        self.calls = []\n"
        "        self._session_runtime_dir = '/run/g1'\n"
        "        self._history_messages = [{'role': 'user'}]\n"
        "        self._session_total_tokens = 111\n"
        "        self._session_generated_tokens = 222\n"
        "        self._last_step_summary = {'game_over': True}\n"
        "        self._last_action_result = {'executed': True}\n"
        "        self._summarized_knowledge = {k: '<%s>' % k for k in "
        + repr(list(FIELDS))
        + "}\n"
        "\n" + ensure_session + "\n" + analyze,
        encoding="utf-8",
    )
    return root


class _StatePath:
    """Minimal Path stand-in the stub's load_runtime_state understands."""

    def __init__(self, key: str, exists: bool = True):
        self._key = key
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return self._key


def _install(tmp_path: Path, **kw):
    """Exec the real cell against a fresh stub; return (namespace, module)."""
    root = _make_stub(tmp_path, **kw)
    sys.path.insert(0, str(root))
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]
    try:
        import inference.agent.tool_agent as ta

        ta.LEVELS.clear()
        ta.LEVELS["/run/g1/state.json"] = 1
        ns: dict = {"__name__": "restart_at_stall_cell"}
        exec(compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec"), ns)
        return ns, ta
    finally:
        sys.path.remove(str(root))


@pytest.fixture
def installed(tmp_path):
    ns, ta = _install(tmp_path)
    yield ns, ta
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def _drive(ta, agent, n, *, start=1, key="/run/g1/state.json"):
    """n turns with distinct analysis_step values."""
    for s in range(start, start + n):
        ta.ToolAgent.analyze(agent, _StatePath(key), s, analysis_step=s)


# --- installation ---------------------------------------------------------------------------


def test_installs_and_rebinds_exactly_one_method(installed):
    ns, ta = installed
    assert ta.ToolAgent.analyze is ns["_restart_at_stall"]
    # Nothing else on the class was touched.
    assert ta.ToolAgent._ensure_session.__name__ == "_ensure_session"
    assert "_restart_at_stall" not in ta.ToolAgent._ensure_session.__qualname__


def test_probe_zeroes_its_own_counters(installed):
    ns, _ = installed
    assert ns["_RS_STATS"] == {"fired": 0, "turns": 0, "capped": 0, "no_step": 0, "errors": 0}
    assert ns["_RS_GAMES"] == set()


def test_refuses_to_install_twice(tmp_path):
    root = _make_stub(tmp_path)
    sys.path.insert(0, str(root))
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]
    try:
        import inference.agent.tool_agent as ta

        ta.LEVELS["/run/g1/state.json"] = 1
        src = compile(CELL.read_text(encoding="utf-8"), str(CELL), "exec")
        exec(src, {"__name__": "c1"})
        with pytest.raises(AssertionError, match="installed twice"):
            exec(src, {"__name__": "c2"})
    finally:
        sys.path.remove(str(root))
        for mod in [m for m in sys.modules if m.startswith("inference")]:
            del sys.modules[mod]


# --- the one thing that changes ---------------------------------------------------------------


def test_does_not_fire_below_threshold(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"] - 1)
    assert ns["_RS_STATS"]["fired"] == 0
    assert agent._history_messages == [{"role": "user"}]
    assert agent._summarized_knowledge["world_model"] == "<world_model>"


def test_fires_exactly_at_threshold(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    assert ns["_RS_STATS"]["fired"] == 1
    assert ns["_RS_STATS"]["turns"] == ns["RESTART_STALL_TURNS"]


def test_fire_clears_the_four_belief_attributes(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    assert agent._history_messages == []
    assert agent._last_step_summary is None
    assert agent._last_action_result is None
    for field in FIELDS:
        if field != "cross_level_notes":
            assert agent._summarized_knowledge[field] == ""


def test_fire_preserves_cross_level_notes(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    assert agent._summarized_knowledge["cross_level_notes"] == "<cross_level_notes>"


def test_fire_does_not_touch_token_accounting(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    assert agent._session_total_tokens == 111
    assert agent._session_generated_tokens == 222


def test_fire_bumps_the_seed_and_upstream_reads_it(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    seed0 = ta._LOCAL_ANALYZER_SEED
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    assert ta._LOCAL_ANALYZER_SEED == seed0 + 1
    # The seed the stub's payload builder actually saw on the firing turn.
    assert agent.calls[-1][2] == seed0 + 1
    assert agent.calls[0][2] == seed0


def test_upstream_is_called_on_every_turn_including_the_firing_one(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    n = ns["RESTART_STALL_TURNS"] + 3
    _drive(ta, agent, n)
    assert len(agent.calls) == n
    assert [c[1] for c in agent.calls] == list(range(1, n + 1))


def test_return_value_is_upstreams(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    assert ta.ToolAgent.analyze(agent, _StatePath("/run/g1/state.json"), 1, analysis_step=1) == "RESULT"


# --- the things that must NOT fire -------------------------------------------------------------


def test_retried_analysis_step_is_not_a_new_turn(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    # solver.py:285-302 retries the same analysis_step after a yield.
    for _rep in range(4):
        _drive(ta, agent, ns["RESTART_STALL_TURNS"] - 1)
    assert ns["_RS_STATS"]["fired"] == 0
    assert len(agent.calls) == 4 * (ns["RESTART_STALL_TURNS"] - 1)


def test_level_change_resets_the_counter(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"] - 1)
    ta.LEVELS["/run/g1/state.json"] = 2
    _drive(ta, agent, 5, start=100)
    assert ns["_RS_STATS"]["fired"] == 0
    assert agent._history_messages == [{"role": "user"}]


def test_new_session_resets_the_counter(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"] - 1)
    agent._session_runtime_dir = "/run/g2"
    ta.LEVELS["/run/g2/state.json"] = 1
    _drive(ta, agent, 3, start=100, key="/run/g2/state.json")
    assert ns["_RS_STATS"]["fired"] == 0


def test_cap_holds_at_max_per_level(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    t, cap = ns["RESTART_STALL_TURNS"], ns["RESTART_STALL_MAX_PER_LEVEL"]
    _drive(ta, agent, t * (cap + 2))
    assert ns["_RS_STATS"]["fired"] == cap
    assert ns["_RS_STATS"]["capped"] > 0


def test_missing_analysis_step_never_fires_and_is_counted(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    for _i in range(ns["RESTART_STALL_TURNS"] + 5):
        ta.ToolAgent.analyze(agent, _StatePath("/run/g1/state.json"), 1)
    assert ns["_RS_STATS"]["fired"] == 0
    assert ns["_RS_STATS"]["no_step"] == ns["RESTART_STALL_TURNS"] + 5


# --- it can never kill a game -----------------------------------------------------------------


def test_absent_runtime_state_does_not_raise(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    # Upstream returns None here (tool_agent.py:2102-2103); the wrapper must
    # pass that through and must not have counted an error getting there.
    out = ta.ToolAgent.analyze(agent, _StatePath("/run/g1/state.json", exists=False), 1, analysis_step=1)
    assert out is None
    assert ns["_RS_STATS"]["errors"] == 0


def test_corrupt_runtime_state_yields_level_none_without_raising(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    # load_runtime_state itself raising is the realistic corrupt-file case: the
    # path exists, the parse fails. _rs_level must absorb it, not propagate.
    assert ns["_rs_level"](agent, _StatePath("/run/NOPE/state.json")) is None
    assert ns["_RS_STATS"]["errors"] == 0


def test_bookkeeping_error_still_calls_upstream(installed):
    ns, ta = installed
    agent = ta.ToolAgent()

    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("hostile")

    agent.__dict__["_restart_at_stall"] = _Hostile()
    out = ta.ToolAgent.analyze(agent, _StatePath("/run/g1/state.json"), 1, analysis_step=1)
    assert out == "RESULT"
    assert ns["_RS_STATS"]["errors"] >= 1


def test_upstream_exception_is_not_swallowed(installed):
    ns, ta = installed
    agent = ta.ToolAgent()
    original = ns["_RS_ORIGINAL"]

    def _boom(self, *a, **k):
        raise ValueError("upstream failed")

    ns["_RS_ORIGINAL"] = _boom
    try:
        with pytest.raises(ValueError, match="upstream failed"):
            ns["_restart_at_stall"](agent, _StatePath("/run/g1/state.json"), 1, analysis_step=1)
    finally:
        ns["_RS_ORIGINAL"] = original


# --- it fails loudly on upstream drift ----------------------------------------------------------


def test_fails_loudly_if_analyze_loses_analysis_step(tmp_path):
    bad = UPSTREAM_ANALYZE.replace("        analysis_step=None,\n", "")
    with pytest.raises(AssertionError, match="analysis_step"):
        _install(tmp_path, analyze=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_fails_loudly_if_analyze_stops_resolving_the_frame(tmp_path):
    bad = UPSTREAM_ANALYZE.replace(
        "        current_frame, history_entries = load_runtime_state(state_path)\n",
        "        current_frame, history_entries = None, []\n",
    )
    with pytest.raises(AssertionError, match="load_runtime_state"):
        _install(tmp_path, analyze=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_fails_loudly_if_ensure_session_stops_resetting_a_belief_field(tmp_path):
    bad = UPSTREAM_ENSURE_SESSION.replace(
        "            self._last_action_result = None\n", ""
    )
    with pytest.raises(AssertionError, match="_last_action_result"):
        _install(tmp_path, ensure_session=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_fails_loudly_if_the_seed_stops_being_read_per_request(tmp_path):
    bad = UPSTREAM_ANALYZE.replace("            seed=_LOCAL_ANALYZER_SEED,\n", "            seed=7,\n")
    with pytest.raises(AssertionError, match="_LOCAL_ANALYZER_SEED"):
        _install(tmp_path, analyze=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_fails_loudly_if_cross_level_notes_disappears(tmp_path):
    bad = UPSTREAM_EMPTY_WORLD_MODEL.replace('        "cross_level_notes": "",\n', "")
    with pytest.raises(AssertionError, match="cross_level_notes|7 fields"):
        _install(tmp_path, empty_world_model=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_fails_loudly_if_the_world_model_gains_a_field(tmp_path):
    bad = UPSTREAM_EMPTY_WORLD_MODEL.replace(
        '        "cross_level_notes": "",\n',
        '        "cross_level_notes": "",\n        "extra": "",\n',
    )
    with pytest.raises(AssertionError, match="8 fields"):
        _install(tmp_path, empty_world_model=bad)
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


# --- one variable ---------------------------------------------------------------------------------


def test_cell_touches_no_config_knob():
    src = CELL.read_text(encoding="utf-8")
    for forbidden in (
        "os.environ",
        "bm.solver",
        "LOCAL_ANALYZER_YIELD",
        "LOCAL_ANALYZER_CONTEXT_WINDOW",
        "LOCAL_ANALYZER_ENABLE_THINKING",
        "LOCAL_ANALYZER_TEMPERATURE",
        "max_num_seqs",
        "concurrency",
        "SETUP_ENV_PATH",
    ):
        assert forbidden not in src, f"cell touches {forbidden} -- that is a second variable"


def test_notebook_carries_this_exact_cell():
    import json

    nb = (
        Path(__file__).resolve().parents[1]
        / "kaggle_submission_duck_nvfp4_anim_restart"
        / "notebook"
        / "arc3-duck-nvfp4-anim-rs.ipynb"
    )
    if not nb.exists():
        pytest.skip("restart arm not built yet")
    cells = json.loads(nb.read_text(encoding="utf-8"))["cells"]
    want = CELL.read_text(encoding="utf-8")
    assert any("".join(c["source"]) == want for c in cells), "notebook holds a drifted copy"


# --- the reader must actually match what the cell prints -----------------------------------------


def test_reader_regexes_match_the_real_printed_lines(tmp_path, capsys):
    """A drifted marker format would make the log reader silently report zero."""
    import importlib.util

    # Installed inside the test, not via the fixture, so the INSTALLED banner
    # printed at install time lands in this test's own capture.
    ns, ta = _install(tmp_path)
    agent = ta.ToolAgent()
    _drive(ta, agent, ns["RESTART_STALL_TURNS"])
    ns["_rs_report"]()
    out = capsys.readouterr().out

    spec = importlib.util.spec_from_file_location(
        "_rd", Path(__file__).resolve().parents[1] / "scripts" / "read_duck_public25_log.py"
    )
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    assert rd.RS_INSTALLED_RE.search(out), "RESTART_STALL_INSTALLED not matched by the reader"
    fired = list(rd.RS_FIRED_RE.finditer(out))
    # Exactly one: the synthetic probe must not leak marker lines into the log,
    # or the reader's firing count is inflated for every real run.
    assert len(fired) == 1, f"RESTART_STALL_FIRED not matched exactly once: {out!r}"
    assert fired[0].group("turns") == str(ns["RESTART_STALL_TURNS"])
    final = list(rd.RS_FINAL_RE.finditer(out))
    assert len(final) == 1, f"RESTART_STALL_FINAL not matched: {out!r}"
    assert final[0].group("fired") == "1"
    assert final[0].group("errors") == "0"
    for mod in [m for m in sys.modules if m.startswith("inference")]:
        del sys.modules[mod]


def test_reader_regexes_match_the_error_line(installed, capsys):
    import importlib.util

    ns, ta = installed
    agent = ta.ToolAgent()

    class _Hostile:
        def __getattr__(self, name):
            raise RuntimeError("hostile")

    agent.__dict__["_restart_at_stall"] = _Hostile()
    ta.ToolAgent.analyze(agent, _StatePath("/run/g1/state.json"), 1, analysis_step=1)
    out = capsys.readouterr().out
    assert out.count("RESTART_STALL_ERROR") == 1, out

    spec = importlib.util.spec_from_file_location(
        "_rd2", Path(__file__).resolve().parents[1] / "scripts" / "read_duck_public25_log.py"
    )
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    assert rd.RS_ERROR_RE.search(out), f"RESTART_STALL_ERROR not matched: {out!r}"


def test_reader_ignores_probe_lines_printed_before_the_banner():
    """Kernel v1 shipped a build whose probe was not silenced; the reader must
    still report the true firing count for that artifact."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rd3", Path(__file__).resolve().parents[1] / "scripts" / "read_duck_public25_log.py"
    )
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)

    fired = (
        "RESTART_STALL_FIRED n={n} level=1 turns=20 nth_on_level=1 seed=2026082{n} "
        "kept_cross_level_notes=19c session=/run/g{n}\n"
    )
    log = (
        fired.format(n=1) + fired.format(n=2) + fired.format(n=3)
        + "RESTART_STALL_ERROR RuntimeError: probe\n"
        + "RESTART_STALL_INSTALLED target=inference.agent.tool_agent.ToolAgent.analyze probe=9/9\n"
        + fired.format(n=4)
        + "RESTART_STALL_FINAL fired=1 turns_discarded=20 capped=0 no_step=0 "
          "errors=0 games=1 seed_now=20260826\n"
    )
    banner = rd.RS_INSTALLED_RE.search(log)
    after = log[banner.end():]
    assert len(rd.RS_FIRED_RE.findall(after)) == 1
    assert len(rd.RS_ERROR_RE.findall(after)) == 0
    assert len(rd.RS_FIRED_RE.findall(log)) == 4  # what a naive count would give
