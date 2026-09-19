"""What the world-model wipe costs, and what a guard against it would buy.

`inference/agent/tool_agent.py` erases six of the seven summarized-knowledge
fields whenever the last executed step sequence reports a level transition, a
run completion **or a game over**:

    def _update_summarized_knowledge_from_step_summary(self) -> None:
        summary = self._last_step_summary
        if not summary:
            return
        if summary.get("level_transition") or summary.get("run_complete") or summary.get("game_over"):
            for key in ("world_model", "goal_model", "action_model",
                        "recent_findings", "open_questions", "current_plan"):
                self._summarized_knowledge[key] = ""

A game over triggers an auto-RESET that replays the *same* level with the same
mechanics, so the game-over wipes throw away knowledge that is still true. The
guard skips the wipe exactly when ``game_over and not level_transition and not
run_complete``.

This script sizes that from a completed run's own artifacts and reports the
outcome measures the guard has to move to be real.

The wipe fires once per *step summary*, not once per environment event: a
single `python` tool call can execute several game actions, and they are
summarised together. So the number of wipes is the number of distinct
(game, analysis_step) groups that contain a qualifying event -- not the raw
event count. Both are reported; the raw count is an upper bound.

Usage:
    venv/Scripts/python.exe scripts/analyze_wipe_guard.py \
        anim=<dir> guard=<dir> --json experiments/stage7_wipe_guard.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Fields erased by the wipe, and the one that survives it.
WIPED_FIELDS = (
    "world_model",
    "goal_model",
    "action_model",
    "recent_findings",
    "open_questions",
    "current_plan",
)
SURVIVING_FIELD = "cross_level_notes"

FINISHED_RE = re.compile(
    r"\[finished\]\s+(?P<game>\S+).*?score[= ]\s*(?P<score>[0-9.]+)", re.IGNORECASE
)


def _events(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line (disk-full, kill) must not abort the
                # whole accounting -- skip it, same as extract_level_up.
                continue
    return out


def _game_wipe_accounting(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Count wipes, split by which branch of the `if` fires.

    ``state`` on an event is the environment state after the action
    (``GAME_OVER``, ``WIN``, ``NOT_FINISHED``). A level transition is a change
    in ``level`` between consecutive action events -- verified against the
    event stream: ``level`` does **not** revert on a game over (the auto-RESET
    replays the same level), so a level change is a genuine completion.

    Only ``type == "action"`` rows are real actions. The stream also carries
    ``type == "analysis"`` rows that repeat the preceding action's ``action_num``
    and ``state``, and one ``type == "initial"`` row per game. Counting rows by
    ``action_num is not None`` (as ``analyze_rhae_binding.py`` does) therefore
    over-counts: 4,970 rows against ``benchmark.json``'s 3,633 real actions on
    the nvfp4 baseline, and double-counts every game over.
    """
    actions = [e for e in events if e.get("type") == "action"]

    # Per-event flags.
    prev_level: int | None = None
    flagged: list[tuple[Any, bool, bool, bool]] = []  # (step, game_over, level_tx, run_complete)
    for e in actions:
        lvl = e.get("level")
        state = str(e.get("state") or "")
        run_status = str(e.get("run_status") or "")
        game_over = state == "GAME_OVER"
        level_tx = prev_level is not None and lvl is not None and int(lvl) != int(prev_level)
        run_complete = state == "WIN" or run_status in ("win", "won", "complete")
        if lvl is not None:
            prev_level = int(lvl)
        if game_over or level_tx or run_complete:
            flagged.append((e.get("analysis_step"), game_over, level_tx, run_complete))

    # Group into step summaries: one wipe per (analysis_step) group. Events with
    # a null analysis_step (the initial RESET) are each their own group.
    groups: dict[Any, list[tuple[Any, bool, bool, bool]]] = {}
    for i, rec in enumerate(flagged):
        key = rec[0] if rec[0] is not None else f"_null{i}"
        groups.setdefault(key, []).append(rec)

    wipes = 0
    guardable = 0  # game_over and not level_tx and not run_complete
    for recs in groups.values():
        go = any(r[1] for r in recs)
        tx = any(r[2] for r in recs)
        rc = any(r[3] for r in recs)
        wipes += 1
        if go and not tx and not rc:
            guardable += 1

    return {
        "actions": len(actions),
        "raw_game_over_events": sum(1 for r in flagged if r[1]),
        "raw_level_transitions": sum(1 for r in flagged if r[2]),
        "wipes": wipes,
        "guardable_wipes": guardable,
    }


def _level_actions(events: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for e in events:
        if e.get("type") != "action":
            continue
        lvl = e.get("level")
        if lvl is None:
            continue
        counts[int(lvl)] = counts.get(int(lvl), 0) + 1
    return counts


def _calls(run_dir: Path) -> int:
    """LLM requests, from the vLLM server's own counter (ground truth)."""
    prom = run_dir / "vllm-metrics-final.prom"
    if not prom.exists():
        return 0
    total = 0.0
    for line in prom.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("vllm:request_generation_tokens_count"):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
    return int(total)


def _noop_turns(run_dir: Path) -> tuple[int, int]:
    """(analyzer turns, turns that executed no action) from the transcripts."""
    turns = executed = 0
    tdir = run_dir / "transcripts"
    if not tdir.is_dir():
        return (0, 0)
    for path in sorted(tdir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        turns += text.count("[ANALYZER STATUS]")
        executed += text.count("step_executed: True")
    return (turns, executed)


def analyse(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "benchmark.json").open(encoding="utf-8") as fh:
        runs = json.load(fh)["game_runs"]
    with (run_dir / "score.json").open(encoding="utf-8") as fh:
        scores = json.load(fh)["games"]

    per_game = []
    tot_wipes = tot_guardable = 0
    tot_actions = tot_solved_actions = tot_unsolved_actions = 0
    tot_levels_solved = 0
    for run in sorted(runs, key=lambda r: r["game_id"]):
        gid = run["game_id"]
        ev_path = run_dir / "artifacts" / f"{gid}_p0_events.jsonl"
        events = _events(ev_path) if ev_path.exists() else []
        acct = _game_wipe_accounting(events)
        per_level = _level_actions(events)
        reached = max(per_level) if per_level else 1
        # Levels strictly before the deepest one reached are solved; the
        # deepest is the one the run was still on when time ran out.
        solved_actions = sum(v for k, v in per_level.items() if k < reached)
        unsolved_actions = per_level.get(reached, 0)
        levels_solved = max(0, reached - 1)

        row = {
            "game_id": gid,
            "score": scores.get(gid, {}).get("score", 0.0),
            "levels_total": run.get("number_of_levels") or 0,
            "levels_solved": levels_solved,
            "deepest_level": reached,
            "actions": acct["actions"],
            "actions_solved_levels": solved_actions,
            "actions_unsolved_level": unsolved_actions,
            **{k: acct[k] for k in ("raw_game_over_events", "raw_level_transitions", "wipes", "guardable_wipes")},
        }
        per_game.append(row)
        tot_wipes += acct["wipes"]
        tot_guardable += acct["guardable_wipes"]
        tot_actions += acct["actions"]
        tot_solved_actions += solved_actions
        tot_unsolved_actions += unsolved_actions
        tot_levels_solved += levels_solved

    turns, executed = _noop_turns(run_dir)
    calls = _calls(run_dir)
    n_games = len(per_game) or 1
    mean_score = sum(r["score"] for r in per_game) / n_games

    return {
        "run_dir": str(run_dir),
        "games": n_games,
        "public25_mean": mean_score,
        "actions": tot_actions,
        "actions_per_game": tot_actions / n_games,
        "llm_calls": calls,
        "calls_per_game": calls / n_games,
        "levels_solved": tot_levels_solved,
        "levels_total": sum(r["levels_total"] for r in per_game),
        "wasted_action_fraction": (tot_unsolved_actions / tot_actions) if tot_actions else 0.0,
        "actions_on_solved_levels": tot_solved_actions,
        "actions_on_never_completed_level": tot_unsolved_actions,
        "wipes": tot_wipes,
        "guardable_wipes": tot_guardable,
        "analyzer_turns": turns,
        "turns_executed": executed,
        "noop_turn_fraction": (1 - executed / turns) if turns else 0.0,
        "per_game": per_game,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=dir pairs (or bare dirs)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results: dict[str, Any] = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            label, path = Path(spec).name, spec
        results[label] = analyse(Path(path))

    hdr = f"{'run':<10} {'pub25':>7} {'actions':>8} {'a/game':>7} {'calls':>6} {'c/game':>7} {'lvls':>5} {'wasted':>7} {'wipes':>6} {'guard':>6} {'noop%':>6}"
    print(hdr)
    print("-" * len(hdr))
    for label, r in results.items():
        print(
            f"{label:<10} {r['public25_mean']:>7.2f} {r['actions']:>8d} {r['actions_per_game']:>7.1f} "
            f"{r['llm_calls']:>6d} {r['calls_per_game']:>7.1f} {r['levels_solved']:>5d} "
            f"{r['wasted_action_fraction']*100:>6.1f}% {r['wipes']:>6d} {r['guardable_wipes']:>6d} "
            f"{r['noop_turn_fraction']*100:>5.1f}%"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
