"""Aggregate a set of scripts/run_scorecard.py outputs (logs/scorecards/*.json)
matching a label prefix into mean/std/min/max -- used to quantify run-to-run
variance under a fixed set of local games (any variance measured this way is
attributable to the agent's own stochastic policy, not to hidden-game
sampling, since the local 25-game suite is fixed run to run).

Usage: python scripts/summarize_scorecards.py var_eps025_r var_eps000_r ...
Each positional arg is a label PREFIX; every logs/scorecards/<prefix>*.json
file is grouped under it.
"""

import argparse
import json
import statistics
from pathlib import Path

SCORECARDS_DIR = Path(__file__).resolve().parent.parent / "logs" / "scorecards"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prefixes", nargs="+")
    args = parser.parse_args()

    for prefix in args.prefixes:
        files = sorted(SCORECARDS_DIR.glob(f"{prefix}*.json"))
        if not files:
            print(f"{prefix}: no matching files")
            continue
        scores, levels, actions = [], [], []
        for f in files:
            card = json.loads(f.read_text())
            scores.append(card.get("score", 0.0))
            levels.append(card.get("total_levels_completed", 0))
            actions.append(card.get("total_actions", 0))
        n = len(scores)
        mean = statistics.mean(scores)
        std = statistics.pstdev(scores) if n > 1 else 0.0
        print(f"{prefix}: n={n}")
        print(f"  scores:  {[round(s, 5) for s in scores]}")
        print(f"  mean={mean:.5f}  std={std:.5f}  min={min(scores):.5f}  max={max(scores):.5f}")
        print(f"  levels:  {levels}  (mean={statistics.mean(levels):.2f})")
        print(f"  actions: {actions}")


if __name__ == "__main__":
    main()
