"""Trains and cross-validates a small structural "hops to nearest known
win" regressor from scripts/harvest_graph_win_distance_data.py's output.

Deliberately NOT a torch model -- this project's own history (Stage 4's
MoE gate collapse, Stage 1's early predictor-learns-identity bug, this
session's own JEPA rollout-compounding findings) is full of ways a neural
net can silently degenerate; a small gradient-boosted tree on a
six-feature table is easier to sanity-check and matches the "cheap test"
scope this was scoped as.

Cross-validation is grouped BY GAME (leave-some-games-out), not a random
row split -- this project's Stage 6 addendum spent an entire investigation
establishing that a same-game split looks deceptively good and tells you
nothing about generalization to a game the model didn't train on. A
within-local-roster leave-games-out split is still not the same as
Kaggle's fully novel hidden games, but it's the honest local proxy this
project has used everywhere else.

Usage:
    python scripts/train_graph_win_distance_model.py --data data/graph_win_distance_dataset.csv --out checkpoints_graph_win_distance
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

FEATURE_NAMES = [
    "out_degree",
    "in_degree",
    "visit_count",
    "depth",
    "num_available_actions",
    "action6_available",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"loaded {len(df)} rows across {df['game'].nunique()} games")
    print(f"hops_to_win distribution:\n{df['hops_to_win'].describe()}")
    print(f"per-game row counts:\n{df['game'].value_counts()}")

    X = df[FEATURE_NAMES].values
    y = df["hops_to_win"].values
    groups = df["game"].values

    n_groups = df["game"].nunique()
    n_folds = min(args.n_folds, n_groups)
    if n_folds < 2:
        print(f"\nonly {n_groups} distinct game(s) in the data -- "
              f"leave-games-out cross-validation needs at least 2, skipping "
              f"CV and training on all data (report this honestly, don't fake a CV score).")
        fold_results = []
    else:
        gkf = GroupKFold(n_splits=n_folds)
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
            )
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            mae = mean_absolute_error(y[test_idx], pred)

            # Baseline: predict the TRAIN set's mean hops_to_win for every
            # test row -- the honest "does this model know anything beyond
            # the label's own marginal distribution" check, same spirit as
            # this project's own identity/zero-baseline comparisons
            # elsewhere (Stage 1's changed-patches methodology, Stage 5's
            # value-head zero-baseline).
            baseline_pred = np.full(len(test_idx), y[train_idx].mean())
            baseline_mae = mean_absolute_error(y[test_idx], baseline_pred)

            corr = np.corrcoef(y[test_idx], pred)[0, 1] if len(set(y[test_idx])) > 1 else float("nan")
            test_games = sorted(set(groups[test_idx]))
            fold_results.append({
                "fold": fold_idx, "test_games": test_games,
                "n_test": len(test_idx), "mae": mae, "baseline_mae": baseline_mae,
                "corr": corr,
            })
            print(f"fold {fold_idx} (games={test_games}): n={len(test_idx)} "
                  f"mae={mae:.3f} baseline_mae={baseline_mae:.3f} corr={corr:.3f}")

        mean_mae = np.mean([r["mae"] for r in fold_results])
        mean_baseline_mae = np.mean([r["baseline_mae"] for r in fold_results])
        mean_corr = np.nanmean([r["corr"] for r in fold_results])
        print(f"\n=== leave-games-out CV summary ({n_folds} folds) ===")
        print(f"mean MAE: {mean_mae:.3f}  (baseline/marginal-mean MAE: {mean_baseline_mae:.3f})")
        print(f"mean correlation (pred vs true, held-out games): {mean_corr:.3f}")
        improvement_pct = 100 * (mean_baseline_mae - mean_mae) / mean_baseline_mae
        print(f"improvement over marginal-mean baseline: {improvement_pct:.1f}%")

    # Final model trained on ALL data, for live deployment.
    final_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0
    )
    final_model.fit(X, y)
    importances = dict(zip(FEATURE_NAMES, final_model.feature_importances_.tolist()))
    print(f"\nfeature importances (final model, all data): {json.dumps(importances, indent=2)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final_model, out_dir / "win_distance_model.joblib")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "n_train_rows": len(df),
            "n_train_games": int(df["game"].nunique()),
            "cv_fold_results": [
                {k: (v if k != "test_games" else list(v)) for k, v in r.items()}
                for r in fold_results
            ],
            "feature_importances": importances,
        }, f, indent=2)

    print(f"\nsaved model + meta to {out_dir}")


if __name__ == "__main__":
    main()
