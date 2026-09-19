# ---- [calamitychasm] COMMIT FLOOR -----------------------------------------------------------
# The ONLY behavioural change in this notebook relative to arc3-duck-nvfp4-anim.
#
# WHAT IT DOES
# After two consecutive `analyze()` turns on the same analysis step that ended
# without executing any game action, the next turn's user prompt carries a
# directive requiring that turn to end in `action(...)`. It escalates after
# five. Nothing else changes: no forced action, no injected move, no reset, no
# knob, no serving change. The model still chooses what to do.
#
# WHY (measured on this exact chassis, scripts/analyze_dead_turns.py)
# Every `[ANALYZER STATUS]` block is one `analyze()` turn and reports
# `step_executed`. Across three anim-chassis public-25 runs:
#
#     run   turns  executed   dead        clock in dead turns
#     anim    940       586   354 (37.7%)  91,295s / 198,233s = 46.1%
#     wg      921       558   363 (39.4%)  96,363s / 198,128s = 48.6%
#     rs      929       553   376 (40.5%)  98,606s / 198,161s = 49.8%
#
# A dead turn is also the EXPENSIVE kind: median 258s against 173s for a turn
# that acts, because it runs until the 180s yield budget trips. When a turn
# yields, framework/solver.py:360-361 sets `retry_analysis_step = analysis_step`
# and replays the same step, so a model that stops committing burns the rest of
# its fixed 7,920s clock without touching the board. On the anim run the
# terminal streaks alone are 89 turns and 21,598s (10.9% of the whole run's
# clock): `r11l` spent its last 16 turns and 4,703s (59% of its clock) on one
# analysis step, `cn04` 18 turns and 4,750s (60%), and `ft09` -- the run's best
# game at 47.62 -- 1,771s (22%).
#
# WHY IT TARGETS DEPTH, NOT BREADTH
# RHAE weights a level by its index, so the value of a recovered turn is the
# value of the level it might finish. The games holding the largest terminal
# streaks are the deep ones: clearing one more level is worth +23.81 on `ft09`,
# +14.29 on `r11l`, +9.52 on `cn04`. It adds nothing to any game it does not
# already play.
#
# THRESHOLD, FIXED BEFORE THE RUN AND NOT SWEPT
# 2 consecutive dead turns, which is `sahasawatt/thui-af-v0`'s published action-
# floor trigger ("after 2 consecutive turn ends with no executed action"), from a
# team at 3.74. Our own streak histogram endorses it rather than fitting it: on
# the anim run 125 of 189 streaks are a single turn (ordinary deliberation, left
# alone), 34 are two, and the 29 streaks of 3+ hold 71 turns and 18,664s.
#
# HOW IT DIFFERS FROM THE TWO MECHANISMS ALREADY TRIED HERE
#   * the wipe guard PRESERVED state the harness erased -- it found there was
#     almost nothing to preserve (239 characters).
#   * restart-at-stall DISCARDS the belief state and re-draws; its free run went
#     42 -> 39 levels.
# This keeps the belief state and asks for a decision from it.
import inspect
import threading
import types

import inference.agent.tool_agent as _cf_tool_agent

# --- constants -------------------------------------------------------------------------------
_CF_AFTER = 2       # consecutive dead turns before the directive is attached
_CF_ESCALATE = 5    # consecutive dead turns before the stronger form is used

_CF_TARGET_ANALYZE = "analyze"
_CF_TARGET_PROMPT = "_build_user_prompt"
_CF_ORIG_ANALYZE = getattr(_cf_tool_agent.ToolAgent, _CF_TARGET_ANALYZE)
_CF_ORIG_PROMPT = getattr(_cf_tool_agent.ToolAgent, _CF_TARGET_PROMPT)

assert not getattr(_CF_ORIG_ANALYZE, "_commit_floor_installed", False), "commit floor installed twice"
assert not getattr(_CF_ORIG_PROMPT, "_commit_floor_installed", False), "commit floor installed twice"

# --- fail loudly if upstream is not what this mechanism assumes --------------------------------
_CF_SRC_ANALYZE = inspect.getsource(_CF_ORIG_ANALYZE)
for _needle in (
    "self._build_user_prompt(",              # the prompt we append to is built inside analyze()
    "self._ensure_session(state_path)",      # per-game session boundary we key the counter on
    "yielded_control=yielded_control_reason is not None",
    "step_executed=step_executed",
    "turn_time_budget",
):
    assert _needle in _CF_SRC_ANALYZE, f"commit floor: analyze() changed, {_needle!r} is gone"
_CF_SRC_PROMPT = inspect.getsource(_CF_ORIG_PROMPT)
assert _CF_SRC_PROMPT.rstrip().endswith('return "\\n".join(lines)'), (
    "commit floor: _build_user_prompt no longer returns a joined line list"
)
assert "call `action(actions)` from inside the `python` tool" in _CF_SRC_PROMPT, (
    "commit floor: the prompt's own act instruction is gone; re-derive the directive"
)
# The retry semantics the whole mechanism rests on: a yielded turn replays the
# SAME analysis step, so consecutive dead turns are consecutive attempts at one
# decision rather than progress through the game.
try:
    import inference.framework.solver as _cf_solver

    _CF_SRC_SOLVER = inspect.getsource(_cf_solver)
    assert 'if getattr(result, "yielded_control", False):' in _CF_SRC_SOLVER, (
        "commit floor: solver no longer branches on yielded_control"
    )
    assert "retry_analysis_step = analysis_step" in _CF_SRC_SOLVER, (
        "commit floor: solver no longer retries a yielded step"
    )
except ImportError:  # pragma: no cover - the wrapper does not need the solver at run time
    pass

# --- the directive ------------------------------------------------------------------------------
_CF_DIRECTIVE = (
    "\n"
    "HARNESS NOTICE -- TURN BUDGET (this block is from the harness, not from the game):\n"
    "The last {n} turns on this step ended without executing any action. A turn that executes"
    " no action leaves the board unchanged and still spends this game's fixed wall clock, which"
    " is the only budget that limits how many levels you can reach. Everything you worked out in"
    " those turns is already above in this conversation; you do not need to redo it.\n"
    "This turn MUST end with a call to `action(...)`. Pick the best action or short sequence your"
    " current world model supports. An imperfect action is strictly more informative than another"
    " inspection-only turn: it returns a new frame, and a new frame is evidence."
)
_CF_ESCALATION = (
    "\n"
    "Do not make any inspection-only `python` call this turn. Your first `python` call must itself"
    " call `action(...)`. If the mechanics are still unclear, execute the single most discriminating"
    " legal probe rather than continuing to analyse."
)

_CF_LOCK = threading.Lock()
_CF_STATS = {
    "turns": 0,
    "dead": 0,
    "fired": 0,
    "escalated": 0,
    "converted": 0,      # fired turns that then executed an action
    "unconverted": 0,
    "errors": 0,
}


def _cf_bump(key, by=1):
    with _CF_LOCK:
        _CF_STATS[key] += by
        return _CF_STATS[key]


def _cf_directive(n):
    text = _CF_DIRECTIVE.format(n=n)
    if n >= _CF_ESCALATE:
        text += _CF_ESCALATION
    return text


def _commit_floor_prompt(self, *args, **kwargs):
    """Upstream, plus the directive when this turn is carrying commit pressure."""
    text = _CF_ORIG_PROMPT(self, *args, **kwargs)
    try:
        n = int(getattr(self, "_cf_pressure", 0) or 0)
        if n >= _CF_AFTER:
            return text + "\n" + _cf_directive(n)
    except Exception as exc:  # never let the directive break prompt construction
        _cf_bump("errors")
        print(f"COMMIT_FLOOR_ERROR prompt {type(exc).__name__}: {exc}", flush=True)
    return text


def _commit_floor_analyze(self, state_path, *args, **kwargs):
    """Upstream, plus: count consecutive non-executing turns and arm the directive."""
    pressure = 0
    armed = False
    try:
        # A new game resets the count: the solver builds one ToolAgent per game,
        # but _ensure_session is what upstream itself treats as the boundary.
        session = getattr(state_path, "parent", None)
        if getattr(self, "_cf_session", "<unset>") != session:
            self._cf_session = session
            self._cf_consec_dead = 0
        pressure = int(getattr(self, "_cf_consec_dead", 0) or 0)
        self._cf_pressure = pressure
        if pressure >= _CF_AFTER:
            armed = True
            n = _cf_bump("fired")
            if pressure >= _CF_ESCALATE:
                _cf_bump("escalated")
            print(
                f"COMMIT_FLOOR_FIRED n={n} consec={pressure} "
                f"escalated={int(pressure >= _CF_ESCALATE)} session={session}",
                flush=True,
            )
    except Exception as exc:
        _cf_bump("errors")
        print(f"COMMIT_FLOOR_ERROR arm {type(exc).__name__}: {exc}", flush=True)

    result = _CF_ORIG_ANALYZE(self, state_path, *args, **kwargs)

    try:
        self._cf_pressure = 0
        if result is None:
            # Upstream returns None for a missing state file or an unexpected
            # error. Neither is the model declining to act, so it is neutral.
            return result
        _cf_bump("turns")
        executed = bool(getattr(result, "step_executed", False))
        if executed:
            self._cf_consec_dead = 0
        else:
            _cf_bump("dead")
            self._cf_consec_dead = pressure + 1
        if armed:
            _cf_bump("converted" if executed else "unconverted")
            print(
                f"COMMIT_FLOOR_RESULT consec={pressure} executed={int(executed)} "
                f"session={getattr(state_path, 'parent', None)}",
                flush=True,
            )
    except Exception as exc:  # bookkeeping must never affect the game
        _cf_bump("errors")
        print(f"COMMIT_FLOOR_ERROR account {type(exc).__name__}: {exc}", flush=True)
    return result


_commit_floor_analyze._commit_floor_installed = True
_commit_floor_prompt._commit_floor_installed = True


# ---- synthetic probe: prove the whole transform before it touches a real game ------------------
def _cf_probe():
    calls = {"n": 0}

    class _Stub:
        """A ToolAgent stand-in whose analyze/_build_user_prompt are the real wrappers."""

        def __init__(self, script):
            self._script = list(script)   # per-turn: True = executes, False = dead, None = upstream None
            self._prompts = []

        # stands in for _CF_ORIG_PROMPT
        def _base_prompt(self):
            return "BASE PROMPT"

        def analyze(self, state_path):
            return _commit_floor_analyze(self, state_path)

    def fake_prompt(self, *a, **kw):
        return self._base_prompt()

    def fake_analyze(self, state_path, *a, **kw):
        # what upstream analyze() does, reduced to: build the prompt, then report.
        self._prompts.append(_commit_floor_prompt(self, 0, valid_actions=None))
        outcome = self._script[calls["n"]]
        calls["n"] += 1
        if outcome is None:
            return None
        return types.SimpleNamespace(step_executed=bool(outcome))

    global _CF_ORIG_PROMPT, _CF_ORIG_ANALYZE
    real_prompt, real_analyze = _CF_ORIG_PROMPT, _CF_ORIG_ANALYZE
    _CF_ORIG_PROMPT, _CF_ORIG_ANALYZE = fake_prompt, fake_analyze
    before = dict(_CF_STATS)
    try:
        class _P:
            def __init__(self, parent):
                self.parent = parent

        # 1. the shape of the mechanism over one game.
        calls["n"] = 0
        agent = _Stub([False, False, False, True, False, False, False, False, False, False])
        for _ in range(10):
            agent.analyze(_P("game-a"))
        fired = [i for i, p in enumerate(agent._prompts) if "HARNESS NOTICE" in p]
        assert fired == [2, 3, 6, 7, 8, 9], fired         # 2 dead -> fire; an action resets it
        assert all(agent._prompts[i].startswith("BASE PROMPT") for i in range(10))
        assert "MUST end with a call to `action(...)`" in agent._prompts[2]
        assert "Do not make any inspection-only" not in agent._prompts[2]
        assert "The last 2 turns" in agent._prompts[2]
        # escalation only once five consecutive dead turns have happened
        assert "Do not make any inspection-only" not in agent._prompts[8]   # consec=4
        assert "Do not make any inspection-only" in agent._prompts[9]       # consec=5

        # 2. a new session resets the counter.
        calls["n"] = 0
        agent = _Stub([False, False, False, False])
        agent.analyze(_P("game-a"))
        agent.analyze(_P("game-a"))
        agent.analyze(_P("game-b"))      # new game: pressure must be 0 again
        agent.analyze(_P("game-b"))
        assert not any("HARNESS NOTICE" in p for p in agent._prompts), agent._prompts

        # 3. upstream returning None is neutral, not a dead turn.
        calls["n"] = 0
        agent = _Stub([False, None, None, False, False])
        for _ in range(5):
            agent.analyze(_P("game-c"))
        fired = [i for i, p in enumerate(agent._prompts) if "HARNESS NOTICE" in p]
        assert fired == [4], fired

        # 4. a turn that never had pressure is byte-identical to upstream.
        calls["n"] = 0
        agent = _Stub([True, True])
        agent.analyze(_P("game-d"))
        agent.analyze(_P("game-d"))
        assert agent._prompts == ["BASE PROMPT", "BASE PROMPT"], agent._prompts

        # 5. the conversion counters are what the free run will be read on.
        assert _CF_STATS["fired"] - before["fired"] == 7, _CF_STATS
        assert _CF_STATS["converted"] - before["converted"] == 1, _CF_STATS
        assert _CF_STATS["unconverted"] - before["unconverted"] == 6, _CF_STATS
        assert _CF_STATS["errors"] - before["errors"] == 0, _CF_STATS
    finally:
        _CF_ORIG_PROMPT, _CF_ORIG_ANALYZE = real_prompt, real_analyze
        # Probe traffic must not pollute the run's own tally.
        with _CF_LOCK:
            for key in _CF_STATS:
                _CF_STATS[key] = 0


_cf_probe()
setattr(_cf_tool_agent.ToolAgent, _CF_TARGET_ANALYZE, _commit_floor_analyze)
setattr(_cf_tool_agent.ToolAgent, _CF_TARGET_PROMPT, _commit_floor_prompt)
assert getattr(_cf_tool_agent.ToolAgent, _CF_TARGET_ANALYZE) is _commit_floor_analyze
assert getattr(_cf_tool_agent.ToolAgent, _CF_TARGET_PROMPT) is _commit_floor_prompt


def _cf_report():
    print(
        "COMMIT_FLOOR_FINAL turns={turns} dead={dead} fired={fired} escalated={escalated} "
        "converted={converted} unconverted={unconverted} errors={errors}".format(**_CF_STATS),
        flush=True,
    )


import atexit

atexit.register(_cf_report)
print(
    f"COMMIT_FLOOR_INSTALLED target={_cf_tool_agent.ToolAgent.__module__}.ToolAgent."
    f"{{{_CF_TARGET_ANALYZE},{_CF_TARGET_PROMPT}}} after={_CF_AFTER} escalate={_CF_ESCALATE} probe=5/5",
    flush=True,
)
