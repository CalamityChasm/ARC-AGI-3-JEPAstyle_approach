"""Stage 6 test-time adaptation AGENT follow-up: a wider tradeoff-curve
sweep over scripts/test_time_adaptation.py's three knobs (K, N_STEPS, LR)
across all 5 held-out games, to find a real operating point for
jepa/test_time_adapter.py / ARC-AGI-3-Agents/agents/templates/
hypothesis_agent.py's HYPOTHESIS_TEST_TIME_ADAPT integration, rather than
just replicating the original diagnostic's narrow 2-K/1-LR check.

A full K x STEPS x LR grid across 5 games is expensive for little extra
information (most combinations are far from the interesting region), so
this runs a coordinate-descent-style sweep instead -- standard practice
for a 3-knob tradeoff search:

  Phase A (cadence): fix STEPS=3, LR=5e-5 (the original diagnostic's own
    picked point), vary K widely. Pick the best K by the tradeoff score
    below.
  Phase B (step count): fix K at Phase A's winner, LR=5e-5, vary STEPS.
  Phase C (learning rate): fix K, STEPS at their running winners, vary LR.

"Best" is scored as held-out improvement at n=200 (mean across all 5
games) minus a penalty on trained-game interference (mean pooled
improvement drop from the pre-adaptation +9.8% baseline) -- since a
config that maximizes held-out gain while wrecking trained-game accuracy
is not actually a good real-deployment choice. The penalty weight is
deliberately mild (0.3): the task cares primarily about closing the
held-out gap, with interference as a secondary tie-breaker, not an
equally-weighted objective.

Usage:
    python scripts/sweep_test_time_adaptation.py
    python scripts/sweep_test_time_adaptation.py --phase A
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.test_time_adaptation as tta  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

K_GRID = [5, 10, 25, 50, 100, 200]
STEPS_GRID = [1, 2, 3, 5, 8, 12]
LR_GRID = [1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 4e-4]

DEFAULT_K = 10
DEFAULT_STEPS = 3
DEFAULT_LR = 5e-5

INTERFERENCE_PENALTY_WEIGHT = 0.3
# Pre-adaptation trained-games pooled improvement, from
# experiments/stage6_test_time_adaptation.md -- recomputed fresh below too
# (checkpoints may have been regenerated with a different random seed), but
# used here as a documented reference point.
PRE_ADAPT_TRAINED_REFERENCE = 9.8


def run_one_config(
    online, predictor_state, game_vocab, num_experts, feature_channels,
    games: list, k: int, n_steps: int, lr: float, device,
    trained_probe: list,
) -> dict:
    """Runs the K/STEPS/LR config across all `games`, returns per-game
    n=200 held-out improvement, mean held-out improvement, and pooled
    trained-game interference after adapting on each game (averaged)."""
    tta.N_STEPS = n_steps
    tta.LR = lr

    per_game_final = {}
    interference_posts = []
    for game in games:
        per_file = tta.load_game_transitions_per_file(game)
        if len(per_file) < 2:
            continue
        eval_set = per_file[-1]
        stream = [t for f in per_file[:-1] for t in f]
        traj, adapted_predictor = tta.run_adaptation_trajectory(
            online, predictor_state, game_vocab, num_experts, feature_channels,
            stream, eval_set, k, device,
        )
        final = traj.get(200) or traj.get(max(traj.keys())) if traj else None
        if final and final.get("n_changed", 0) > 0:
            per_game_final[game] = final["improvement_pct"]
        else:
            per_game_final[game] = None

        post_trained = tta.changed_patches_eval(online, adapted_predictor, trained_probe, game_vocab, device)
        if post_trained.get("n_changed", 0) > 0:
            interference_posts.append(post_trained["improvement_pct"])

    valid = [v for v in per_game_final.values() if v is not None]
    mean_heldout = statistics.mean(valid) if valid else float("nan")
    mean_trained_post = statistics.mean(interference_posts) if interference_posts else float("nan")
    return {
        "k": k, "n_steps": n_steps, "lr": lr,
        "per_game": per_game_final,
        "mean_heldout_improvement_pct": mean_heldout,
        "mean_trained_post_improvement_pct": mean_trained_post,
    }


def score_config(result: dict, pre_trained: float) -> float:
    heldout = result["mean_heldout_improvement_pct"]
    trained_drop = pre_trained - result["mean_trained_post_improvement_pct"]
    if heldout != heldout:  # NaN
        return float("-inf")
    return heldout - INTERFERENCE_PENALTY_WEIGHT * max(trained_drop, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--games", nargs="+", default=tta.ALL_HELDOUT_GAMES)
    args = parser.parse_args()

    device = tta.get_device()
    print(f"Device: {device}")
    online, predictor_state, game_vocab, num_experts, feature_channels = tta.load_baseline_checkpoint(device)
    trained_probe = tta.load_sample_transitions(tta.TRAINED_PROBE_GAMES, max_per_game=200)
    print(f"Trained-game interference probe: {len(trained_probe)} transitions")

    fresh_predictor = tta.build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
    pre_trained_eval = tta.changed_patches_eval(online, fresh_predictor, trained_probe, game_vocab, device)
    pre_trained = pre_trained_eval.get("improvement_pct", PRE_ADAPT_TRAINED_REFERENCE)
    print(f"Pre-adaptation trained-games baseline: {pre_trained:+.1f}%")

    all_results = []
    best_k, best_steps, best_lr = DEFAULT_K, DEFAULT_STEPS, DEFAULT_LR

    def run_grid(label, k_list, steps_list, lr_list):
        nonlocal best_k, best_steps, best_lr
        results = []
        for k in k_list:
            for n_steps in steps_list:
                for lr in lr_list:
                    t0 = time.time()
                    r = run_one_config(
                        online, predictor_state, game_vocab, num_experts, feature_channels,
                        args.games, k, n_steps, lr, device, trained_probe,
                    )
                    r["score"] = score_config(r, pre_trained)
                    r["wall_s"] = time.time() - t0
                    results.append(r)
                    all_results.append(r)
                    print(
                        f"[{label}] k={k:4d} steps={n_steps:2d} lr={lr:.0e} "
                        f"heldout={r['mean_heldout_improvement_pct']:+.2f}% "
                        f"trained_post={r['mean_trained_post_improvement_pct']:+.2f}% "
                        f"score={r['score']:+.3f} ({r['wall_s']:.1f}s)"
                    )
        best = max(results, key=lambda r: r["score"])
        return best["k"], best["n_steps"], best["lr"]

    if args.phase in ("A", "all"):
        best_k, _, _ = run_grid("A:K", K_GRID, [DEFAULT_STEPS], [DEFAULT_LR])
        print(f"\n>>> Phase A winner: K={best_k}\n")

    if args.phase in ("B", "all"):
        best_k2, best_steps, _ = run_grid("B:STEPS", [best_k], STEPS_GRID, [DEFAULT_LR])
        print(f"\n>>> Phase B winner: K={best_k2} STEPS={best_steps}\n")

    if args.phase in ("C", "all"):
        best_k3, best_steps3, best_lr = run_grid("C:LR", [best_k], [best_steps], LR_GRID)
        print(f"\n>>> Phase C winner: K={best_k3} STEPS={best_steps3} LR={best_lr}\n")

    print(f"\n=== FINAL RECOMMENDED OPERATING POINT: K={best_k} STEPS={best_steps} LR={best_lr} ===")

    out_path = REPO_ROOT / "logs" / "tta_sweep_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "pre_adaptation_trained_baseline_pct": pre_trained,
        "results": all_results,
        "recommended": {"k": best_k, "n_steps": best_steps, "lr": best_lr},
    }, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
