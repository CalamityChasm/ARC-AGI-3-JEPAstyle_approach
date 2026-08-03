"""One-off analysis for experiments/stage6_budget_tta_combo.md: aggregates
a set of scripts/run_scorecard.py output JSONs (logs/scorecards/<prefix>*.json)
into mean/std score, mean levels completed, total levels, and distinct
games solved (any environment whose levels_completed > 0, matching this
project's established "distinct games" metric used throughout CLAUDE.md's
Stage 5/6 sections).

Usage: python scripts/analyze_budget_tta_combo.py <prefix> [<prefix> ...]
"""

import json
import statistics
import sys
from pathlib import Path

SCORECARDS_DIR = Path(__file__).resolve().parent.parent / "logs" / "scorecards"


def game_key(env_id: str) -> str:
    # env ids look like "r11l-495a7899" -- strip the per-run suffix.
    return env_id.split("-")[0]


def main() -> None:
    prefixes = sys.argv[1:]
    for prefix in prefixes:
        files = sorted(SCORECARDS_DIR.glob(f"{prefix}*.json"))
        if not files:
            print(f"{prefix}: no matching files")
            continue
        scores, levels, actions = [], [], []
        solved_games = set()
        per_game_solves = {}
        for f in files:
            text = f.read_text()
            if not text.strip():
                print(f"  (skipping {f.name}: empty/corrupt file)")
                continue
            card = json.loads(text)
            scores.append(card.get("score", 0.0))
            levels.append(card.get("total_levels_completed", 0))
            actions.append(card.get("total_actions", 0))
            for env in card.get("environments", []):
                gk = game_key(env.get("id", "?"))
                if env.get("levels_completed", 0) > 0:
                    solved_games.add(gk)
                    per_game_solves[gk] = per_game_solves.get(gk, 0) + 1
        n = len(scores)
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if n > 1 else 0.0
        print(f"{prefix}: n={n}")
        print(f"  scores:  {[round(s, 5) for s in scores]}")
        print(f"  mean_score={mean:.5f}  std={std:.5f}  min={min(scores):.5f}  max={max(scores):.5f}")
        print(f"  levels:  {levels}  mean_levels={statistics.mean(levels):.3f}  total_levels={sum(levels)}")
        print(f"  distinct_games_solved={len(solved_games)} {sorted(solved_games)}")
        print(f"  per_game_solve_counts={per_game_solves}")
        print(f"  total_actions={actions}")
        print()


if __name__ == "__main__":
    main()
