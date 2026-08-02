"""Per-game breakdown across a set of scorecards, plus a distribution
comparison (mean/median/quartiles) between two label-prefix groups --
used for the large-scale novelty-beta-cap backtest where the pooled
mean is known to be dominated by a handful of outlier runs.

Usage: python scripts/summarize_per_game.py <prefix_a> <prefix_b>
"""

import argparse
import json
import statistics
from pathlib import Path

SCORECARDS_DIR = Path(__file__).resolve().parent.parent / "logs" / "scorecards"


def load_group(prefix: str):
    files = sorted(SCORECARDS_DIR.glob(f"{prefix}*.json"))
    scores, levels = [], []
    per_game_levels = {}
    per_game_solved_runs = {}
    for f in files:
        text = f.read_text()
        if not text.strip():
            continue
        card = json.loads(text)
        scores.append(card.get("score", 0.0))
        levels.append(card.get("total_levels_completed", 0))
        for env in card.get("environments", []):
            # strip the -<hash> suffix to get the plain game id
            gid = env["id"].split("-")[0]
            per_game_levels.setdefault(gid, []).append(env.get("levels_completed", 0))
            per_game_solved_runs.setdefault(gid, 0)
            if env.get("levels_completed", 0) > 0:
                per_game_solved_runs[gid] += 1
    return files, scores, levels, per_game_levels, per_game_solved_runs


def describe(scores):
    s = sorted(scores)
    n = len(s)
    if n == 0:
        return {}
    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "std": statistics.pstdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
        "q1": s[n // 4],
        "q3": s[(3 * n) // 4],
        "nonzero_frac": sum(1 for x in s if x > 0) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    args = parser.parse_args()

    for prefix in args.prefixes:
        files, scores, levels, per_game_levels, per_game_solved = load_group(prefix)
        print(f"=== {prefix} (n={len(files)}) ===")
        d = describe(scores)
        print(f"  score dist: {d}")
        dl = describe(levels)
        print(f"  levels dist: {dl}")
        print(f"  total levels completed: {sum(levels)}")
        print("  per-game (runs with >=1 level completed / total runs, total levels):")
        for gid in sorted(per_game_levels):
            runs = per_game_levels[gid]
            solved = per_game_solved[gid]
            total_lv = sum(runs)
            print(f"    {gid}: solved {solved}/{len(runs)} runs, total levels={total_lv}")
        print()


if __name__ == "__main__":
    main()
