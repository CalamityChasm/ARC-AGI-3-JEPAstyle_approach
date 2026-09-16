# ---- [calamitychasm] RESTART-AT-STALL --------------------------------------------------------
# The ONLY behavioural change in this notebook relative to arc3-duck-nvfp4-anim.
#
# WHAT IT DOES
# After RESTART_STALL_TURNS distinct analysis turns on one level with no level
# change, the agent's level-local belief state is cleared and the sampling seed
# is bumped, so the stalled level gets a fresh draw inside the same run. No
# prompt text is added, no action is taken on the model's behalf, and the game
# itself is untouched -- only what the agent remembers and what seed it samples
# with.
#
# WHY, MEASURED ON OUR OWN RUN (scripts/measure_*.py over the anim artifacts)
#   * 1,270 of 2,615 actions (48.6%) went into a level that never completed.
#   * Inside those stalls, 80.5% of actions exactly repeat an (action, level)
#     pair already tried on that level -- the environment is deterministic, so
#     those are telling the agent nothing it has not already seen.
#   * Five games scored 0.00 having never cleared level 1, spending 23-30
#     analysis turns each on it.
#
# WHY THIS THRESHOLD, AND WHY IT IS NOT FITTED TO THE OUTCOME
# Levels that DO clear take a median of 5 analysis turns, p84 = 12, p90 = 17,
# max 26. Levels that never clear run to a median of 10 and a max of 30. At 20
# turns, exactly 1 of 42 cleared levels would have been interrupted (2%) against
# 7 of 25 stalls (28%). 20 is also the value sahasawatt/thui-rs-v0 published
# (a team at 3.74), derived from the same distribution on their own chassis --
# so it is an externally-fixed number that our data independently endorses,
# not one swept for best result here.
#
# WHAT IS CLEARED, AND WHAT DELIBERATELY IS NOT
# `ToolAgent._ensure_session` (tool_agent.py:1140-1155) is upstream's own
# definition of forgetting. It clears seven things. We clear the four that are
# belief:
#     _history_messages, _summarized_knowledge, _last_step_summary,
#     _last_action_result
# and leave the three that are not:
#     _session_total_tokens / _session_generated_tokens -- accounting, read by
#         the `total_tokens` / `generated_tokens` properties into benchmark.json;
#         zeroing them would corrupt the run's own reported metrics.
#     _noop_guard / animation_counters / _reset_animation_hint_state() -- the
#         anim solver's safety machinery, not the agent's beliefs.
#
# `cross_level_notes` is preserved, which is a DELIBERATE DEVIATION from
# thui-rs-v0 (it assigns `_empty_world_model()`, zeroing all seven fields).
# Upstream never erases that field: the wipe at tool_agent.py:1343-1356 spares
# it on all three of its triggers, including a full level transition. It holds
# cross-level mechanics, not this level's failed plan, so a within-level restart
# has no reason to destroy it.
#
# THE SEED BUMP
# `_LOCAL_ANALYZER_SEED` is read at request-build time (tool_agent.py:1536,
# inside `build_chat_payload`), not captured at construction, so rebinding the
# module global takes effect on the very next call. Without it a restart against
# a deterministic environment would re-draw the same opening.
#
# TURN COUNTING
# `analyze` is called more than once per turn: `inference/framework/solver.py:
# 285-302` retries the SAME `analysis_step` after a turn yields without acting
# (`retry_analysis_step`). On our anim run sp80 made 41 `analyze` calls across
# 26 distinct steps. So turns are counted as the SET of distinct `analysis_step`
# values seen, never as a call count.
#
# Prior art: sahasawatt/thui-rs-v0, a team at 3.74.
import inspect
import threading
import types

import inference.agent.tool_agent as _rs_tool_agent

_RS_TARGET = "analyze"
_RS_ORIGINAL = getattr(_rs_tool_agent.ToolAgent, _RS_TARGET)

# --- tunables, both fixed before the run ---
RESTART_STALL_TURNS = 20        # distinct analysis turns on one level with no level change
RESTART_STALL_MAX_PER_LEVEL = 2  # restarts per level, as thui-rs-v0

# The belief attributes cleared on a restart. Asserted below to be exactly the
# belief subset of what upstream's own _ensure_session clears.
_RS_CLEARED = ("_history_messages", "_summarized_knowledge",
               "_last_step_summary", "_last_action_result")
_RS_PRESERVED_FIELD = "cross_level_notes"

# ---- fail fast if we are not patching what we think we are ----------------------------------
assert not getattr(_RS_ORIGINAL, "_restart_at_stall_installed", False), "restart-at-stall installed twice"

_RS_ANALYZE_SRC = inspect.getsource(_RS_ORIGINAL)
_RS_SIG = inspect.signature(_RS_ORIGINAL)
assert "analysis_step" in _RS_SIG.parameters, "restart-at-stall: analyze() lost its analysis_step parameter"
assert "state_path" in _RS_SIG.parameters, "restart-at-stall: analyze() lost its state_path parameter"
assert "load_runtime_state(state_path)" in _RS_ANALYZE_SRC, (
    "restart-at-stall: analyze() no longer resolves the frame via load_runtime_state"
)

# _ensure_session is the upstream reference for "forget this session".
_RS_ENSURE_SRC = inspect.getsource(_rs_tool_agent.ToolAgent._ensure_session)
for _attr in _RS_CLEARED:
    assert f"self.{_attr}" in _RS_ENSURE_SRC, (
        f"restart-at-stall: upstream _ensure_session no longer resets {_attr}; "
        "the set of things a restart should forget has changed"
    )
assert "_empty_world_model()" in _RS_ENSURE_SRC, (
    "restart-at-stall: upstream _ensure_session no longer rebuilds the world model"
)

# The seed must still be read per request, or bumping the global is inert.
_RS_PAYLOAD_SRC = inspect.getsource(_rs_tool_agent.ToolAgent)
assert "seed=_LOCAL_ANALYZER_SEED" in _RS_PAYLOAD_SRC, (
    "restart-at-stall: the request payload no longer reads _LOCAL_ANALYZER_SEED at call time"
)
assert isinstance(_rs_tool_agent._LOCAL_ANALYZER_SEED, int), _rs_tool_agent._LOCAL_ANALYZER_SEED

# cross_level_notes must exist and must be one of the world-model fields, or
# "preserve it" is meaningless.
_RS_EMPTY_WM = _rs_tool_agent._empty_world_model()
assert _RS_PRESERVED_FIELD in _RS_EMPTY_WM, (
    f"restart-at-stall: {_RS_PRESERVED_FIELD} is no longer a world-model field"
)
assert len(_RS_EMPTY_WM) == 7, f"restart-at-stall: world model has {len(_RS_EMPTY_WM)} fields, expected 7"

_RS_LOCK = threading.Lock()
# The synthetic probe drives the real wrapper, so its marker lines would land
# in the kernel log and be counted by scripts/read_duck_public25_log.py even
# though its counters are zeroed afterwards. Silence them at the source.
_RS_QUIET = [False]


def _rs_print(msg):
    if not _RS_QUIET[0]:
        print(msg, flush=True)


_RS_STATS = {"fired": 0, "turns": 0, "capped": 0, "no_step": 0, "errors": 0}
_RS_GAMES = set()


def _rs_bump(key, n=1):
    with _RS_LOCK:
        _RS_STATS[key] += n
        return _RS_STATS[key]


def _rs_state(self):
    """Per-instance bookkeeping, created on first use."""
    return self.__dict__.setdefault(
        "_restart_at_stall", {"level": None, "steps": set(), "restarts": 0, "session": None}
    )


def _rs_forget(self):
    """Clear the level-local belief state, keeping cross_level_notes."""
    carried = ""
    try:
        carried = (self._summarized_knowledge or {}).get(_RS_PRESERVED_FIELD, "") or ""
    except Exception:
        carried = ""
    self._history_messages = []
    fresh = _rs_tool_agent._empty_world_model()
    fresh[_RS_PRESERVED_FIELD] = carried
    self._summarized_knowledge = fresh
    self._last_step_summary = None
    self._last_action_result = None
    return len(carried)


def _rs_level(self, state_path):
    """Current level, or None if the runtime state is not readable yet."""
    try:
        if state_path is None or not state_path.exists():
            return None
        frame, _history = _rs_tool_agent.load_runtime_state(state_path)
        return getattr(frame, "level", None) if frame is not None else None
    except Exception:
        return None


def _restart_at_stall(self, state_path, action_num, *args, **kwargs):
    """Upstream analyze(), preceded by the stall check. Never swallows upstream."""
    try:
        st = _rs_state(self)

        # A new game reuses the class but a fresh runtime dir; reset with it so
        # counts never leak between concurrently-played games.
        session = getattr(self, "_session_runtime_dir", None)
        if session != st["session"]:
            st.update(session=session, level=None, steps=set(), restarts=0)

        level = _rs_level(self, state_path)
        if level != st["level"]:
            st.update(level=level, steps=set(), restarts=0)

        step = kwargs.get("analysis_step")
        if step is None:
            # The solver passes analysis_step by keyword (solver.py:299). If it
            # ever stops, the counter would silently never advance and the whole
            # arm would be a no-op -- so count it and make it visible.
            _rs_bump("no_step")
        else:
            st["steps"].add(step)

        if len(st["steps"]) >= RESTART_STALL_TURNS:
            if st["restarts"] >= RESTART_STALL_MAX_PER_LEVEL:
                _rs_bump("capped")
            else:
                turns = len(st["steps"])
                carried = _rs_forget(self)
                _rs_tool_agent._LOCAL_ANALYZER_SEED = int(_rs_tool_agent._LOCAL_ANALYZER_SEED) + 1
                st["restarts"] += 1
                # This turn becomes turn 1 of the fresh context.
                st["steps"] = {step} if step is not None else set()
                n = _rs_bump("fired")
                _rs_bump("turns", turns)
                with _RS_LOCK:
                    _RS_GAMES.add(str(session))
                _rs_print(
                    f"RESTART_STALL_FIRED n={n} level={level} turns={turns} "
                    f"nth_on_level={st['restarts']} seed={_rs_tool_agent._LOCAL_ANALYZER_SEED} "
                    f"kept_{_RS_PRESERVED_FIELD}={carried}c session={session}"
                )
    except Exception as exc:  # bookkeeping must never cost a turn
        _rs_bump("errors")
        _rs_print(f"RESTART_STALL_ERROR {type(exc).__name__}: {exc}")

    # Deliberately NOT wrapped: analyze() returns the turn's result to the
    # solver, so swallowing an exception here would change control flow rather
    # than protect it.
    return _RS_ORIGINAL(self, state_path, action_num, *args, **kwargs)


_restart_at_stall._restart_at_stall_installed = True


# ---- synthetic probe: prove the transform before it touches a real game ----------------------
def _rs_probe():
    calls = []

    class _StubPath:
        def __init__(self, ok=True):
            self._ok = ok

        def exists(self):
            return self._ok

    def _fresh():
        agent = types.SimpleNamespace()
        agent._history_messages = [{"role": "user", "content": "x"}]
        agent._summarized_knowledge = {k: f"<{k}>" for k in _RS_EMPTY_WM}
        agent._last_step_summary = {"game_over": True}
        agent._last_action_result = {"executed": True}
        agent._session_runtime_dir = "/run/g1"
        agent.__dict__["_probe"] = True
        return agent

    def _believes(agent):
        return (
            agent._history_messages == []
            and all(
                agent._summarized_knowledge[k] == ""
                for k in _RS_EMPTY_WM
                if k != _RS_PRESERVED_FIELD
            )
            and agent._last_step_summary is None
            and agent._last_action_result is None
        )

    _RS_QUIET[0] = True
    orig_original = globals()["_RS_ORIGINAL"]
    orig_level = globals()["_rs_level"]
    orig_seed = _rs_tool_agent._LOCAL_ANALYZER_SEED
    before = dict(_RS_STATS)
    probe_level = {"value": 1}

    globals()["_RS_ORIGINAL"] = lambda self, sp, an, *a, **k: calls.append((sp, an, k)) or "RESULT"
    globals()["_rs_level"] = lambda self, sp: probe_level["value"]
    try:
        # 1. below threshold -> no fire, upstream still called every time
        a = _fresh()
        for s in range(1, RESTART_STALL_TURNS):
            assert _restart_at_stall(a, _StubPath(), s, analysis_step=s) == "RESULT"
        assert _RS_STATS["fired"] == before["fired"], "probe: fired below threshold"
        assert a._history_messages, "probe: belief cleared below threshold"
        assert len(calls) == RESTART_STALL_TURNS - 1, "probe: upstream not called every turn"

        # 2. the threshold turn fires exactly once, clears belief, keeps notes, bumps seed
        seed_before = _rs_tool_agent._LOCAL_ANALYZER_SEED
        _restart_at_stall(a, _StubPath(), 99, analysis_step=RESTART_STALL_TURNS)
        assert _RS_STATS["fired"] == before["fired"] + 1, _RS_STATS
        assert _believes(a), f"probe: belief not cleared -> {a._summarized_knowledge}"
        assert a._summarized_knowledge[_RS_PRESERVED_FIELD] == f"<{_RS_PRESERVED_FIELD}>", (
            "probe: cross_level_notes was destroyed"
        )
        assert _rs_tool_agent._LOCAL_ANALYZER_SEED == seed_before + 1, "probe: seed not bumped"

        # 3. a duplicated analysis_step (the solver's retry path) is not a new turn
        a2 = _fresh()
        for _rep in range(3):
            for s in range(1, RESTART_STALL_TURNS):
                _restart_at_stall(a2, _StubPath(), s, analysis_step=s)
        assert _RS_STATS["fired"] == before["fired"] + 1, "probe: retries counted as fresh turns"

        # 4. a level change resets the counter
        a3 = _fresh()
        for s in range(1, RESTART_STALL_TURNS):
            _restart_at_stall(a3, _StubPath(), s, analysis_step=s)
        probe_level["value"] = 2
        _restart_at_stall(a3, _StubPath(), 99, analysis_step=RESTART_STALL_TURNS)
        assert _RS_STATS["fired"] == before["fired"] + 1, "probe: fired across a level change"
        assert a3._history_messages, "probe: belief cleared across a level change"
        probe_level["value"] = 1

        # 5. the per-level cap holds, and the (cap+1)th attempt counts as capped
        a4 = _fresh()
        step = 0
        for _r in range(RESTART_STALL_MAX_PER_LEVEL + 1):
            for _i in range(RESTART_STALL_TURNS):
                step += 1
                _restart_at_stall(a4, _StubPath(), step, analysis_step=step)
        assert _RS_STATS["fired"] == before["fired"] + 1 + RESTART_STALL_MAX_PER_LEVEL, _RS_STATS
        assert _RS_STATS["capped"] > before["capped"], "probe: cap never engaged"

        # 6. a new session (new game) resets the counter
        a5 = _fresh()
        for s in range(1, RESTART_STALL_TURNS):
            _restart_at_stall(a5, _StubPath(), s, analysis_step=s)
        a5._session_runtime_dir = "/run/g2"
        fired_now = _RS_STATS["fired"]
        _restart_at_stall(a5, _StubPath(), 99, analysis_step=RESTART_STALL_TURNS)
        assert _RS_STATS["fired"] == fired_now, "probe: counter leaked across games"

        # 7. a missing analysis_step is visible, not silent, and never fires
        a6 = _fresh()
        for _i in range(RESTART_STALL_TURNS + 5):
            _restart_at_stall(a6, _StubPath(), 1)
        assert _RS_STATS["no_step"] >= RESTART_STALL_TURNS, _RS_STATS
        assert _RS_STATS["fired"] == fired_now, "probe: fired without a turn counter"

        # 8. hostile state: bookkeeping raises, upstream is still called
        a7 = _fresh()
        del a7._session_runtime_dir
        globals()["_rs_level"] = lambda self, sp: (_ for _ in ()).throw(RuntimeError("probe"))
        errs = _RS_STATS["errors"]
        n_calls = len(calls)
        assert _restart_at_stall(a7, _StubPath(), 1, analysis_step=1) == "RESULT"
        assert _RS_STATS["errors"] == errs + 1, "probe: hostile state not counted as an error"
        assert len(calls) == n_calls + 1, "probe: upstream skipped after a bookkeeping error"

        # 9. an unreadable runtime state yields level None and never raises
        globals()["_rs_level"] = orig_level
        a8 = _fresh()
        assert _restart_at_stall(a8, _StubPath(ok=False), 1, analysis_step=1) == "RESULT"
    finally:
        _RS_QUIET[0] = False
        globals()["_RS_ORIGINAL"] = orig_original
        globals()["_rs_level"] = orig_level
        _rs_tool_agent._LOCAL_ANALYZER_SEED = orig_seed
        # Probe traffic must not pollute the run's own counters.
        with _RS_LOCK:
            for key in _RS_STATS:
                _RS_STATS[key] = 0
            _RS_GAMES.clear()


_rs_probe()
setattr(_rs_tool_agent.ToolAgent, _RS_TARGET, _restart_at_stall)
assert getattr(_rs_tool_agent.ToolAgent, _RS_TARGET) is _restart_at_stall
assert _rs_tool_agent._LOCAL_ANALYZER_SEED == int(_rs_tool_agent._LOCAL_ANALYZER_SEED)


def _rs_report():
    print(
        "RESTART_STALL_FINAL fired={fired} turns_discarded={turns} capped={capped} "
        "no_step={no_step} errors={errors} games={games} seed_now={seed}".format(
            games=len(_RS_GAMES), seed=_rs_tool_agent._LOCAL_ANALYZER_SEED, **_RS_STATS
        ),
        flush=True,
    )


import atexit

atexit.register(_rs_report)
print(
    f"RESTART_STALL_INSTALLED target={_rs_tool_agent.ToolAgent.__module__}.ToolAgent.{_RS_TARGET} "
    f"probe=9/9 turns={RESTART_STALL_TURNS} max_per_level={RESTART_STALL_MAX_PER_LEVEL} "
    f"cleared={len(_RS_CLEARED)} preserved={_RS_PRESERVED_FIELD} "
    f"seed0={_rs_tool_agent._LOCAL_ANALYZER_SEED}",
    flush=True,
)
