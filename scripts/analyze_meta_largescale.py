"""Analysis for the meta-learning large-scale backtest
(scripts/run_meta_largescale_backtest.py): per-condition summary stats,
per-game breakdown, and the same Mann-Whitney U significance checks
experiments/stage6_novelty_aware_beta.md's Part 3 used (per-solved-run
score, per-solved-run action-count-to-solve) -- mirrors that
methodology exactly for direct comparability.

Usage: python scripts/analyze_meta_largescale.py <prefix_a> <prefix_b>
"""

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import mannwhitneyu

SCORECARDS_DIR = Path(__file__).resolve().parent.parent / "logs" / "scorecards"


def load_group(prefix: str):
    files = sorted(SCORECARDS_DIR.glob(f"{prefix}*.json"))
    scores, levels = [], []
    per_game_levels = {}
    per_game_solved_runs = {}
    solved_scores = []
    solved_actions = []
    for f in files:
        text = f.read_text()
        if not text.strip():
            print(f"  (skipping {f.name}: empty/corrupt file)")
            continue
        card = json.loads(text)
        score = card.get("score", 0.0)
        lv = card.get("total_levels_completed", 0)
        scores.append(score)
        levels.append(lv)
        for env in card.get("environments", []):
            gid = env["id"].split("-")[0]
            env_levels = env.get("levels_completed", 0)
            per_game_levels.setdefault(gid, []).append(env_levels)
            per_game_solved_runs.setdefault(gid, 0)
            if env_levels > 0:
                per_game_solved_runs[gid] += 1
                # action count to solve level 1 -- first entry in the run's
                # own level_actions list (env["runs"][0], not the env
                # dict itself -- level_actions lives one level deeper).
                runs = env.get("runs") or []
                lvl_actions = runs[0].get("level_actions") if runs else None
                if lvl_actions:
                    solved_actions.append(lvl_actions[0])
                solved_scores.append(env.get("score", score))
    return {
        "files": files,
        "scores": scores,
        "levels": levels,
        "per_game_levels": per_game_levels,
        "per_game_solved": per_game_solved_runs,
        "solved_scores": solved_scores,
        "solved_actions": solved_actions,
    }


def describe(xs):
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "std": statistics.pstdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix_a")
    parser.add_argument("prefix_b")
    args = parser.parse_args()

    a = load_group(args.prefix_a)
    b = load_group(args.prefix_b)

    for name, g in [(args.prefix_a, a), (args.prefix_b, b)]:
        print(f"=== {name} (n={len(g['files'])}) ===")
        print(f"  score dist: {describe(g['scores'])}")
        print(f"  levels dist: {describe(g['levels'])}")
        print(f"  total levels completed: {sum(g['levels'])}")
        print("  per-game (runs with >=1 level completed / total runs, total levels):")
        for gid in sorted(g["per_game_levels"]):
            runs = g["per_game_levels"][gid]
            solved = g["per_game_solved"][gid]
            total_lv = sum(runs)
            print(f"    {gid}: solved {solved}/{len(runs)} runs, total levels={total_lv}")
        print()

    print("=== significance checks ===")
    print(f"levels-completed (per-run), n={len(a['levels'])} vs n={len(b['levels'])}:")
    if len(set(a["levels"])) > 1 or len(set(b["levels"])) > 1:
        stat, p = mannwhitneyu(a["levels"], b["levels"], alternative="two-sided")
        print(f"  Mann-Whitney U={stat:.1f}, p={p:.4f}")
    else:
        print("  degenerate (all-identical values on at least one side) -- skipping MWU")

    print(f"per-run score, n={len(a['scores'])} vs n={len(b['scores'])}:")
    stat, p = mannwhitneyu(a["scores"], b["scores"], alternative="two-sided")
    print(f"  Mann-Whitney U={stat:.1f}, p={p:.4f}")

    if a["solved_scores"] and b["solved_scores"]:
        print(f"per-solved-run score, n={len(a['solved_scores'])} vs n={len(b['solved_scores'])}:")
        stat, p = mannwhitneyu(a["solved_scores"], b["solved_scores"], alternative="two-sided")
        print(f"  Mann-Whitney U={stat:.1f}, p={p:.4f}")
    else:
        print("per-solved-run score: not enough solved runs on one side to test")

    if a["solved_actions"] and b["solved_actions"]:
        print(f"solve-efficiency (actions to solve), n={len(a['solved_actions'])} vs n={len(b['solved_actions'])}:")
        stat, p = mannwhitneyu(a["solved_actions"], b["solved_actions"], alternative="two-sided")
        print(f"  Mann-Whitney U={stat:.1f}, p={p:.4f}")
        mean_a = statistics.mean(a["solved_actions"])
        mean_b = statistics.mean(b["solved_actions"])
        print(f"  mean actions: {mean_a:.1f} vs {mean_b:.1f}")
    else:
        print("solve-efficiency: not enough solved runs with action data on one side to test")


if __name__ == "__main__":
    main()
