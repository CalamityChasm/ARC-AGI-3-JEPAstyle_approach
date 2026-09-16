# ---- [calamitychasm] WORLD-MODEL WIPE GUARD ------------------------------------------------
# The ONLY behavioural change in this notebook relative to arc3-duck-nvfp4-anim.
#
# inference/agent/tool_agent.py:1343-1356 (anim bundle; byte-identical to
# keithtyser's 1113-1126) erases six of the seven summarized-knowledge fields
# whenever the last executed step sequence reports a level transition, a run
# completion, OR a game over:
#
#     def _update_summarized_knowledge_from_step_summary(self) -> None:
#         summary = self._last_step_summary
#         if not summary:
#             return
#         if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
#             for key in ("world_model", "goal_model", "action_model",
#                         "recent_findings", "open_questions", "current_plan"):
#                 self._summarized_knowledge[key] = ""
#
# Only "cross_level_notes" survives. But a game over triggers an auto-RESET that
# replays the SAME level with the SAME mechanics -- verified in our own event
# streams, where `level` does not revert across a GAME_OVER -- so the six fields
# were still true when they were erased.
#
# The guard skips the wipe on exactly `game_over and not level_transition and
# not run_complete`. Level transitions and run completion wipe as upstream.
#
# Sized on our own runs (scripts/analyze_wipe_guard.py): the anim chassis has
# 16 such wipes across the public 25 (nvfp4 baseline: 37). NB the "74 GAME_OVER
# events" in stage7_sota_research.md is a 2x over-count -- the event stream
# mirrors every action row with a type=='analysis' row carrying the same state.
#
# Prior art: sahasawatt/thui-wm-v0 (+ a matched -ctl arm), a team at 3.74.
import inspect
import threading
import types

import inference.agent.tool_agent as _wg_tool_agent

_WG_TARGET = "_update_summarized_knowledge_from_step_summary"
_WG_ORIGINAL = getattr(_wg_tool_agent.ToolAgent, _WG_TARGET)

# Fail fast if we are not patching what we think we are, or if we already did.
assert not getattr(_WG_ORIGINAL, "_wipe_guard_installed", False), "wipe guard installed twice"
_WG_SRC = inspect.getsource(_WG_ORIGINAL)
for _needle in (
    'summary.get("level_transition")',
    'summary.get("run_complete")',
    'summary.get("game_over")',
    '"world_model"',
    '"goal_model"',
    '"action_model"',
    '"recent_findings"',
    '"open_questions"',
    '"current_plan"',
):
    assert _needle in _WG_SRC, f"wipe guard: upstream wipe changed, {_needle!r} is gone"
# cross_level_notes is the field upstream deliberately keeps; if it appears in
# the wipe list, the semantics we are guarding have changed.
assert '"cross_level_notes"' not in _WG_SRC, "wipe guard: upstream now wipes cross_level_notes"

_WG_WIPED_FIELDS = ("world_model", "goal_model", "action_model",
                    "recent_findings", "open_questions", "current_plan")
_WG_LOCK = threading.Lock()
_WG_STATS = {"kept": 0, "wiped": 0, "noop": 0, "errors": 0}


def _wg_bump(key: str) -> int:
    with _WG_LOCK:
        _WG_STATS[key] += 1
        return _WG_STATS[key]


def _wipe_guard(self) -> None:
    """Upstream, except: an in-level game over keeps the world model."""
    try:
        summary = self._last_step_summary
        game_over = bool(summary.get("game_over")) if summary else False
        level_tx = bool(summary.get("level_transition")) if summary else False
        run_done = bool(summary.get("run_complete")) if summary else False
        if game_over and not level_tx and not run_done:
            n = _wg_bump("kept")
            print(
                f"WIPE_GUARD_KEPT n={n} level={summary.get('level')} "
                f"action={summary.get('end_action_num')} "
                f"session={getattr(self, '_session_runtime_dir', None)}",
                flush=True,
            )
            return None
        _wg_bump("wiped" if (game_over or level_tx or run_done) else "noop")
    except Exception as exc:  # never kill a game over bookkeeping
        _wg_bump("errors")
        print(f"WIPE_GUARD_ERROR {type(exc).__name__}: {exc}", flush=True)
    try:
        return _WG_ORIGINAL(self)
    except Exception as exc:  # upstream cannot realistically raise; belt and braces
        _wg_bump("errors")
        print(f"WIPE_GUARD_UPSTREAM_ERROR {type(exc).__name__}: {exc}", flush=True)
        return None


_wipe_guard._wipe_guard_installed = True


# ---- synthetic probe: prove the transform before it touches a real game ---------------------
def _wg_probe() -> None:
    def fresh():
        agent = types.SimpleNamespace()
        agent._summarized_knowledge = {k: f"<{k}>" for k in _WG_WIPED_FIELDS}
        agent._summarized_knowledge["cross_level_notes"] = "<cross_level_notes>"
        agent._session_runtime_dir = None
        return agent

    def kept(agent):
        return all(agent._summarized_knowledge[k] == f"<{k}>" for k in _WG_WIPED_FIELDS)

    def erased(agent):
        return all(agent._summarized_knowledge[k] == "" for k in _WG_WIPED_FIELDS)

    class _Hostile(dict):
        def get(self, *a, **kw):
            raise RuntimeError("synthetic probe: hostile summary")

    cases = [
        # (summary, expect_kept, label)
        ({"game_over": True, "level": 2, "end_action_num": 41}, True, "in-level game over -> KEEP"),
        ({"game_over": True, "level_transition": True}, False, "game over + level up -> wipe"),
        ({"game_over": True, "run_complete": True}, False, "game over + run complete -> wipe"),
        ({"level_transition": True}, False, "level up -> wipe"),
        ({"run_complete": True}, False, "run complete -> wipe"),
        ({"board_changed": True}, True, "ordinary step -> untouched"),
        (None, True, "no summary -> untouched"),
        ({}, True, "empty summary -> untouched"),
        (_Hostile({"game_over": True}), True, "hostile summary -> no raise, no wipe"),
    ]
    before = dict(_WG_STATS)
    for summary, expect_kept, label in cases:
        agent = fresh()
        agent._last_step_summary = summary
        _wipe_guard(agent)  # must never raise
        ok = kept(agent) if expect_kept else erased(agent)
        assert ok, f"wipe guard probe failed: {label} -> {agent._summarized_knowledge}"
        assert agent._summarized_knowledge["cross_level_notes"] == "<cross_level_notes>", label
    # Exactly one case must have taken the guarded branch, and the hostile one
    # must have been counted as an error rather than silently swallowed.
    assert _WG_STATS["kept"] - before["kept"] == 1, _WG_STATS
    assert _WG_STATS["errors"] - before["errors"] >= 1, _WG_STATS
    assert _WG_STATS["wiped"] - before["wiped"] == 4, _WG_STATS
    # Reset: probe traffic must not pollute the run's own counters.
    with _WG_LOCK:
        for key in _WG_STATS:
            _WG_STATS[key] = 0


_wg_probe()
setattr(_wg_tool_agent.ToolAgent, _WG_TARGET, _wipe_guard)
assert getattr(_wg_tool_agent.ToolAgent, _WG_TARGET) is _wipe_guard


def _wg_report() -> None:
    print(
        "WIPE_GUARD_FINAL kept={kept} wiped={wiped} noop={noop} errors={errors}".format(**_WG_STATS),
        flush=True,
    )


import atexit

atexit.register(_wg_report)
print(
    f"WIPE_GUARD_INSTALLED target={_wg_tool_agent.ToolAgent.__module__}.ToolAgent.{_WG_TARGET} "
    f"probe=9/9 fields={len(_WG_WIPED_FIELDS)} survivor=cross_level_notes",
    flush=True,
)
