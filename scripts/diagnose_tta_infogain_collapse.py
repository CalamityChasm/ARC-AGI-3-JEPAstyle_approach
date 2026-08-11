"""Diagnostic: does test-time adaptation's prediction-accuracy gradient
suppress the Stage 5 hypothesis bundle's InfoGain exploration signal as a
side effect?

Motivation: TTA (jepa/test_time_adapter.py) shows a real, monotonic
changed-patches improvement on held-out games (experiments/
stage6_test_time_adaptation_agent.md), but that representation-level gain
has never once translated into a detectable agent-level win, across three
separate agent-level tests this session (plain TTA, TTA+MAX_ACTIONS=900
combo, TTA+Reptile meta-learning -- see CLAUDE.md's Stage 6 addendum).
One untested mechanistic explanation: TTA's loss is plain per-step
prediction MSE on the GATED blend across all 8 MoE experts. Since
averaging several experts' raw predictions reduces variance, the
gradient that lowers prediction error may also implicitly pull the
experts' raw (ungated) predictions toward agreement with each other --
directly suppressing InfoGain (expert disagreement,
jepa/hypothesis_bundle.py: info_gain). `stage6-encoder-holdout-diag`'s
own earlier finding showed InfoGain does NOT collapse on held-out games
under a FROZEN forward pass (ratio 0.999 vs trained games) and is exactly
what `Hypothesis`'s exploration runs on via NOVELTY_BETA_CAP when a game
is outside the trained vocabulary. If TTA's own gradient steps erode that
specifically, it would be a real, previously undocumented cost that a
pure changed-patches metric can't see -- and a mechanistic explanation
for why representation-level TTA gains never show up in real play: the
act of getting more accurate could be disarming the one signal this
project already confirmed survives on unfamiliar games.

Reuses scripts/test_time_adaptation.py's exact adaptation trajectory
machinery (same checkpoint, same held-out games, same production
operating point K=5/STEPS=8/LR=5e-5 from hypothesis_agent.py's
TTA_K/TTA_STEPS/TTA_LR defaults) -- no retraining, this is a pure
diagnostic layered on top of already-built infrastructure.

Usage:
    python scripts/diagnose_tta_infogain_collapse.py
    python scripts/diagnose_tta_infogain_collapse.py --games r11l bp35
"""

import argparse
import json
import statistics
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.test_time_adaptation as tta  # noqa: E402
from jepa.grid import arc3_frame_to_tensor  # noqa: E402
from jepa.hypothesis_bundle import info_gain  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# Production operating point (hypothesis_agent.py's TTA_K/TTA_STEPS/TTA_LR
# defaults), not the original diagnostic's exploratory K=10/K=50 pair.
PRODUCTION_K = 5
PRODUCTION_STEPS = 8
PRODUCTION_LR = 5e-5

NUM_CANDIDATE_ACTIONS = 4
INFOGAIN_SAMPLE_N = 100


@torch.no_grad()
def mean_info_gain_on_eval_set(online, predictor, eval_set: list, game_vocab: dict, device, sample_idxs: list) -> float:
    """Mean InfoGain (expert-disagreement variance, averaged over
    NUM_CANDIDATE_ACTIONS candidate actions) across a FIXED subset of
    eval_set's frame_t states -- fixed sample_idxs so successive calls at
    different adaptation checkpoints are directly comparable (no sampling
    noise confound)."""
    values = []
    for i in sample_idxs:
        frame_t, _action_id, _x, _y, _frame_t1, _changed, game_id = eval_set[i]
        tensor = arc3_frame_to_tensor(frame_t)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        feat = online(x)
        short = game_id.split("-")[0]
        game_idx = torch.full((1,), game_vocab.get(short, 0), dtype=torch.long, device=device)
        xy = torch.zeros((1, 2), dtype=torch.float32, device=device)

        per_action_ig = []
        for action_id in range(NUM_CANDIDATE_ACTIONS):
            action_t = torch.full((1,), action_id, dtype=torch.long, device=device)
            expert_preds = predictor.predict_all_experts(feat, action_t, xy, game_idx)  # (1, K, C, H, W)
            ig = info_gain(expert_preds.squeeze(0)).item()
            per_action_ig.append(ig)
        values.append(sum(per_action_ig) / len(per_action_ig))

    return statistics.mean(values) if values else float("nan")


def run_trajectory_with_infogain(
    online, predictor_state, game_vocab, num_experts, feature_channels,
    stream: list, eval_set: list, k: int, device,
) -> dict:
    """Mirrors tta.run_adaptation_trajectory exactly (same predictor
    construction, same adapter-param freezing, same adaptation_step calls
    at the same cadence/step-count/lr), but additionally measures mean
    InfoGain on a fixed eval_set subset at each checkpoint, alongside the
    existing changed-patches metric."""
    predictor = tta.build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
    trainable = tta.set_adapter_trainable(predictor)
    opt = torch.optim.AdamW(trainable, lr=tta.LR)

    import random
    rng = random.Random(0)
    sample_idxs = rng.sample(range(len(eval_set)), min(INFOGAIN_SAMPLE_N, len(eval_set)))

    results = {}

    def record(n_observed):
        cp = tta.changed_patches_eval(online, predictor, eval_set, game_vocab, device)
        ig = mean_info_gain_on_eval_set(online, predictor, eval_set, game_vocab, device, sample_idxs)
        results[n_observed] = {
            "changed_patches": cp,
            "mean_info_gain": ig,
        }

    if 0 in tta.EVAL_CHECKPOINTS:
        record(0)

    buffer = []
    max_needed = max(tta.EVAL_CHECKPOINTS)
    for i, t in enumerate(stream):
        if i >= max_needed:
            break
        buffer.append(t)
        n_observed = i + 1
        if n_observed % k == 0:
            tta.adaptation_step(online, predictor, opt, buffer, game_vocab, device, tta.N_STEPS)
        if n_observed in tta.EVAL_CHECKPOINTS:
            record(n_observed)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=tta.ALL_HELDOUT_GAMES)
    parser.add_argument("--k", type=int, default=PRODUCTION_K)
    parser.add_argument("--steps", type=int, default=PRODUCTION_STEPS)
    parser.add_argument("--lr", type=float, default=PRODUCTION_LR)
    args = parser.parse_args()

    k = args.k
    tta.N_STEPS = args.steps
    tta.LR = args.lr

    device = tta.get_device()
    print(f"Device: {device}")
    print(f"Operating point: K={k} STEPS={args.steps} LR={args.lr}")

    online, predictor_state, game_vocab, num_experts, feature_channels = tta.load_baseline_checkpoint(device)
    print(f"Loaded checkpoints_holdout_baseline (game_vocab has {len(game_vocab)} entries, num_experts={num_experts})")

    all_results = {}
    for game in args.games:
        print(f"\n{'=' * 70}\nGAME: {game}\n{'=' * 70}")
        per_file = tta.load_game_transitions_per_file(game)
        if len(per_file) < 2:
            print(f"  SKIPPING {game}: fewer than 2 recording files found")
            continue
        eval_set = per_file[-1]
        stream = [t for f in per_file[:-1] for t in f]
        print(f"  stream: {len(stream)} transitions, eval: {len(eval_set)} transitions")

        traj = run_trajectory_with_infogain(
            online, predictor_state, game_vocab, num_experts, feature_channels,
            stream, eval_set, k, device,
        )
        for n_obs in tta.EVAL_CHECKPOINTS:
            r = traj.get(n_obs)
            if r is None:
                continue
            cp = r["changed_patches"]
            cp_str = f"{cp['improvement_pct']:+.2f}%" if cp.get("n_changed", 0) > 0 else "n/a"
            print(f"    n_observed={n_obs:4d}: changed-patches={cp_str:>9}  mean_InfoGain={r['mean_info_gain']:.6e}")

        all_results[game] = {str(k): v for k, v in traj.items()}

    # Summary: does InfoGain trend down as adaptation accumulates, pooled
    # across all games that produced a full trajectory?
    print(f"\n{'=' * 70}\nSUMMARY (pooled mean InfoGain per checkpoint, games with full trajectories)\n{'=' * 70}")
    for n_obs in tta.EVAL_CHECKPOINTS:
        vals = [all_results[g][str(n_obs)]["mean_info_gain"] for g in all_results if str(n_obs) in all_results[g]]
        vals = [v for v in vals if v == v]  # drop NaN
        if vals:
            print(f"  n_observed={n_obs:4d}: mean InfoGain across {len(vals)} games = {statistics.mean(vals):.6e}")

    zero_vals = [all_results[g]["0"]["mean_info_gain"] for g in all_results if "0" in all_results[g]]
    final_cp = max(tta.EVAL_CHECKPOINTS)
    final_vals = [all_results[g][str(final_cp)]["mean_info_gain"] for g in all_results if str(final_cp) in all_results[g]]
    zero_vals = [v for v in zero_vals if v == v]
    final_vals = [v for v in final_vals if v == v]
    if zero_vals and final_vals:
        z, f = statistics.mean(zero_vals), statistics.mean(final_vals)
        ratio = f / z if z else float("nan")
        print(f"\n  pre-adaptation (n=0) mean InfoGain: {z:.6e}")
        print(f"  post-adaptation (n={final_cp}) mean InfoGain: {f:.6e}")
        print(f"  ratio: {ratio:.3f}")
        if ratio < 0.7:
            print("  VERDICT: InfoGain measurably erodes as TTA adaptation accumulates --")
            print("  consistent with the collapse-via-accuracy-gradient hypothesis.")
        elif ratio > 1.3:
            print("  VERDICT: InfoGain actually INCREASES with adaptation -- hypothesis not supported,")
            print("  adaptation may be sharpening genuine disagreement, not suppressing it.")
        else:
            print("  VERDICT: InfoGain stays roughly flat -- TTA's accuracy gain does not come")
            print("  at the expense of the exploration signal. Hypothesis not supported.")

    out_path = REPO_ROOT / "logs" / f"tta_infogain_collapse_results_k{k}_s{args.steps}_lr{args.lr:.0e}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
