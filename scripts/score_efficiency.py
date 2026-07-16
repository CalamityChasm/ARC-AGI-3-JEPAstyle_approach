"""Local approximation of the real Kaggle scoring shape (rules.md):
completion is primary, later levels are weighted more, and efficiency
(fewer actions per level) matters once a level is completed.

This is NOT a reproduction of the real 0-1 score -- that formula is
min(human_actions / agent_actions, 1.0) ** 2 per level, and we have no
human-baseline action counts locally. Instead this reports two honestly
separate, comparable numbers per agent/config:
  - total levels completed, distinct games (the completion axis)
  - a weighted efficiency-adjusted score: sum over completed levels of
    (level_index / actions_taken_for_that_level) -- rewards completing
    higher-numbered levels in fewer actions, same qualitative shape as
    the real formula (later levels count more, fewer actions is better),
    on an arbitrary scale meant only for A/B comparison between two runs
    of OUR OWN agent, never as a stand-in for the real percentage.

action count is cumulative across the whole recording file (including any
resets/deaths in between) -- a level's "actions taken" is the cumulative
count at that completion minus the cumulative count at the previous
level's completion, so a level reached only after several failed resets
is correctly charged for all of it, not just the final successful attempt.

Usage: python scripts/score_efficiency.py <agent> [agent2 ...]
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
    level_completion_actions: list[int] = []  # cumulative action count at each level-up

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if "state" not in data or "frame" not in data:
                continue
            game_id = data.get("game_id") or game_id
            if data.get("action_input", {}).get("id") is not None:
                action_count += 1
            new_levels = data.get("levels_completed", levels_completed)
            while new_levels > levels_completed:
                levels_completed += 1
                level_completion_actions.append(action_count)

    per_level_actions = []
    prev = 0
    for cum in level_completion_actions:
        per_level_actions.append(max(cum - prev, 1))  # avoid div-by-zero
        prev = cum

    weighted_score = sum(
        (idx + 1) / actions for idx, actions in enumerate(per_level_actions)
    )

    return {
        "agent": agent,
        "game_id": game_id or "unknown",
        "levels_completed": levels_completed,
        "actions": action_count,
        "per_level_actions": per_level_actions,
        "weighted_score": weighted_score,
    }


def main() -> None:
    agents = sys.argv[1:]
    if not agents:
        print("Usage: python scripts/score_efficiency.py <agent1> [agent2 ...]")
        return

    files = []
    for agent in agents:
        files.extend(RECORDINGS_DIR.glob(f"*.{agent}.*.recording.jsonl"))
    rows = [analyze_file(p) for p in sorted(files)]

    by_agent = defaultdict(list)
    for r in rows:
        by_agent[r["agent"]].append(r)

    print(f"{'agent':<12} {'runs':<6} {'total_levels':<14} {'distinct_games':<16} "
          f"{'total_actions':<14} {'weighted_eff_score':<20}")
    print("-" * 90)
    for agent in agents:
        runs = by_agent.get(agent, [])
        total_levels = sum(r["levels_completed"] for r in runs)
        distinct_games = {r["game_id"] for r in runs if r["levels_completed"] > 0}
        total_actions = sum(r["actions"] for r in runs)
        weighted = sum(r["weighted_score"] for r in runs)
        print(f"{agent:<12} {len(runs):<6} {total_levels:<14} {len(distinct_games):<16} "
              f"{total_actions:<14} {weighted:<20.4f}")
        if distinct_games:
            print(f"    games solved: {sorted(distinct_games)}")


if __name__ == "__main__":
    main()
