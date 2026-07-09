"""Stage 5 milestone check: does an agent reach its first level completion
in fewer actions than another, on average, across matched runs?

Reads every ARC-AGI-3-Agents/recordings/*.recording.jsonl file for the given
agent names and reports, per agent: total levels completed, distinct games
with at least one completion, and average actions-to-first-completion
(only over games where that agent completed at least one level in that run).

Usage: python scripts/compare_agents.py hypothesis curiosity
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"


def analyze_file(path: Path) -> dict:
    agent = path.name.split(".")[1] if "." in path.name else "unknown"
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
        "agent": agent,
        "game_id": game_id or "unknown",
        "levels_completed": levels_completed,
        "actions": action_count,
        "actions_to_first_level": actions_to_first_level,
    }


def main() -> None:
    agents = sys.argv[1:]
    if not agents:
        print("Usage: python scripts/compare_agents.py <agent1> [agent2 ...]")
        return

    rows = [analyze_file(p) for p in sorted(RECORDINGS_DIR.glob("*.recording.jsonl"))]
    rows = [r for r in rows if r["agent"] in agents]

    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    print(f"{'agent':<12} {'runs':<6} {'total_levels':<14} {'distinct_games':<16} {'avg_actions_to_1st_level':<26}")
    print("-" * 80)
    for agent in agents:
        runs = by_agent.get(agent, [])
        total_levels = sum(r["levels_completed"] for r in runs)
        distinct_games = {r["game_id"] for r in runs if r["levels_completed"] > 0}
        completions = [r["actions_to_first_level"] for r in runs if r["actions_to_first_level"] is not None]
        avg_actions = sum(completions) / len(completions) if completions else float("nan")
        print(f"{agent:<12} {len(runs):<6} {total_levels:<14} {len(distinct_games):<16} {avg_actions:<26.1f}")
        if distinct_games:
            print(f"    games solved: {sorted(distinct_games)}")


if __name__ == "__main__":
    main()
