"""Generalized version of scripts/aggregate_deadend_backtest.py -- same
methodology (total levels completed, distinct games with >=1 completion,
avg actions-to-first-completion, matching scripts/compare_agents.py), but
takes an arbitrary --dir containing before/ and after/ subdirectories
instead of a hardcoded path, so it works for any before/after backtest
produced by scripts/run_agent_backtest_fold.py.

Usage: python scripts/aggregate_agent_backtest.py --dir E:/jepa_overflow/lookahead_backtest
"""

import argparse
import json
from pathlib import Path


def analyze_file(path: Path) -> dict:
    game_id = None
    levels_completed = 0
    action_count = 0
    actions_to_first_level = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if "state" in data and "frame" in data:
                game_id = data.get("game_id") or game_id
                if data.get("action_input", {}).get("id") is not None:
                    action_count += 1
                new_levels = data.get("levels_completed", levels_completed)
                if new_levels > levels_completed and actions_to_first_level is None:
                    actions_to_first_level = action_count
                levels_completed = max(levels_completed, new_levels)

    return {
        "game_id": (game_id or "unknown").split("-")[0],
        "levels_completed": levels_completed,
        "actions_to_first_level": actions_to_first_level,
    }


def summarize(label: str, directory: Path) -> dict:
    files = sorted(directory.glob("*.recording.jsonl"))
    rows = [analyze_file(p) for p in files]
    total_levels = sum(r["levels_completed"] for r in rows)
    distinct_games = sorted({r["game_id"] for r in rows if r["levels_completed"] > 0})
    completions = [r["actions_to_first_level"] for r in rows if r["actions_to_first_level"] is not None]
    avg_actions = sum(completions) / len(completions) if completions else float("nan")
    print(f"\n=== {label} ({len(rows)} runs) ===")
    print(f"total levels completed: {total_levels}")
    print(f"distinct games with >=1 completion: {len(distinct_games)}  {distinct_games}")
    print(f"avg actions to first completion: {avg_actions:.1f}" if completions else "avg actions to first completion: n/a")
    return {"runs": len(rows), "total_levels": total_levels, "distinct_games": distinct_games, "avg_actions": avg_actions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path, help="Parent dir containing before/ and after/ subdirs.")
    args = parser.parse_args()

    before = summarize("BEFORE", args.dir / "before")
    after = summarize("AFTER", args.dir / "after")

    print("\n=== delta ===")
    print(f"total levels: {before['total_levels']} -> {after['total_levels']}  (delta {after['total_levels'] - before['total_levels']:+d})")
    print(f"distinct games: {len(before['distinct_games'])} -> {len(after['distinct_games'])}  (delta {len(after['distinct_games']) - len(before['distinct_games']):+d})")
    only_after = sorted(set(after['distinct_games']) - set(before['distinct_games']))
    only_before = sorted(set(before['distinct_games']) - set(after['distinct_games']))
    if only_after:
        print(f"games solved AFTER but not BEFORE: {only_after}")
    if only_before:
        print(f"games solved BEFORE but not AFTER: {only_before}")


if __name__ == "__main__":
    main()
