"""Stage 6 meta-learning validation (`stage6-meta-learning`): does a
Reptile-meta-trained checkpoint (`jepa/train_meta_predictor.py`) adapt
BETTER at test time than a normally-trained checkpoint of the identical
recipe otherwise?

Compares two checkpoints -- `checkpoints_holdout_baseline` (ordinary joint
training, the exact recipe from experiments/stage6_game_holdout.md /
stage6_test_time_adaptation_agent.md) and `checkpoints_meta_fold1`
(this branch's Reptile meta-training, same corpus/curriculum, only the ARC
fine-tune phase's head-subset update rule differs) -- on the SAME fold-1
held-out games (`r11l`, `bp35`, `m0r0`, `tr87`, `ka59`), three ways:

  1. Zero-shot changed-patches on the FULL held-out-game population (no
     adaptation at all) -- does meta-training alone already generalize
     better, with no test-time adaptation applied?
  2. Post-adaptation changed-patches, using the SAME `TestTimeAdapter`
     procedure and validated operating point (K=5, STEPS=8, LR=5e-5,
     `experiments/stage6_test_time_adaptation_agent.md`) applied to each
     checkpoint -- THE central comparison this experiment exists to make:
     is post-adaptation improvement larger starting from the meta-learned
     base than from the normally-trained one?
  3. Trained-games zero-shot sanity check -- does the meta-objective cost
     much normal (pre-adaptation) accuracy on the 20 training-pool games?

Adaptation mechanics (per-game stream/eval split, adaptation_step,
changed_patches_eval) intentionally mirror scripts/test_time_adaptation.py
exactly (same K/steps/LR semantics, same per-file "last file = eval set,
remaining files = adaptation stream, in order" split) so results are
directly comparable to that script's own already-published numbers.

Usage:
    python scripts/eval_meta_learning.py
"""

import copy
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, _load_frame_lines
from jepa.device import get_device
from jepa.losses import per_region_error, weighted_prediction_loss
from jepa.models import CNNEncoder, MoEPredictor
from jepa.test_time_adapter import get_adapter_params

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"

HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
TRAINED_PROBE_GAMES = ["ft09", "s5i5", "vc33", "ar25", "cd82", "cn04", "lp85", "sp80"]

CHECKPOINTS = {
    "baseline": REPO_ROOT / "checkpoints_holdout_baseline",
    "meta-reptile": REPO_ROOT / "checkpoints_meta_fold1",
}

# The validated stage6-test-time-adaptation-agent operating point --
# see experiments/stage6_test_time_adaptation_agent.md's coordinate-descent
# sweep (K=5, STEPS=8, LR=5e-5: +0.84% mean held-out gain).
K = 5
N_STEPS = 8
LR = 5e-5
ADAPT_BATCH_SIZE = 16
EVAL_CHECKPOINTS = [0, 50, 200]


# --- Data loading (mirrors scripts/test_time_adaptation.py exactly) -------


def load_game_transitions_per_file(game: str) -> list:
    files = sorted(RECORDINGS_DIR.glob(f"{game}-*.recording.jsonl"))
    files = [f for f in files if ".random.80." in f.name]
    per_file = []
    for path in files:
        frames = _load_frame_lines(path)
        transitions = []
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x = xy.get("x", 0)
            y = xy.get("y", 0)
            changed = cur["frame"] != nxt["frame"]
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], changed, game_id))
        per_file.append(transitions)
    return per_file


def load_sample_transitions(games: list, max_per_game: int = 200) -> list:
    out = []
    for g in games:
        per_file = load_game_transitions_per_file(g)
        flat = [t for f in per_file for t in f]
        out.extend(flat[:max_per_game])
    return out


# --- Model loading -----------------------------------------------------------


def load_checkpoint(ckpt_dir: Path, device):
    game_vocab = json.loads((ckpt_dir / "game_vocab_moe.json").read_text())
    meta = json.loads((ckpt_dir / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)
    feature_channels = 64
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(ckpt_dir / "encoder_moe.pt", map_location=device))
    online.eval()
    for p in online.parameters():
        p.requires_grad = False
    predictor_state = torch.load(ckpt_dir / "moe_predictor.pt", map_location=device)
    return online, predictor_state, game_vocab, num_experts, feature_channels, meta


def build_predictor(state_dict, game_vocab, num_experts, feature_channels, device):
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(copy.deepcopy(state_dict))
    predictor.eval()
    return predictor


# --- Eval / adaptation ------------------------------------------------------


@torch.no_grad()
def changed_patches_eval(online, predictor, transitions: list, game_vocab: dict, device) -> dict:
    if not transitions:
        return {"n_changed": 0}
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    predictor.eval()
    total = {"pred": 0.0, "identity": 0.0, "n": 0}
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        cur_feat = online(cur)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        next_feat = online(nxt)
        pred_err = per_region_error(pred_feat, next_feat)
        identity_err = per_region_error(cur_feat, next_feat)
        for i in range(cur.shape[0]):
            m = patch_mask[i]
            if m.any():
                total["pred"] += pred_err[i][m].mean().item()
                total["identity"] += identity_err[i][m].mean().item()
                total["n"] += 1
    if total["n"] == 0:
        return {"n_changed": 0}
    return {
        "n_changed": total["n"],
        "pred_changed_mse": total["pred"] / total["n"],
        "identity_changed_mse": total["identity"] / total["n"],
        "improvement_pct": (total["identity"] - total["pred"]) / total["identity"] * 100,
    }


def adaptation_step(online, predictor, opt, buffer: list, game_vocab: dict, device, n_steps: int):
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(buffer, fallback_vocab)
    bs = min(ADAPT_BATCH_SIZE, len(buffer))
    predictor.train()
    for _ in range(n_steps):
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        cur, action_id, xy, nxt, patch_mask, game_idx = next(iter(loader))
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        with torch.no_grad():
            cur_feat = online(cur)
            next_feat = online(nxt)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        loss = weighted_prediction_loss(pred_feat, next_feat, patch_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    predictor.eval()


def run_adaptation_trajectory(
    online, predictor_state, game_vocab, num_experts, feature_channels,
    stream: list, eval_set: list, k: int, n_steps: int, lr: float, device,
) -> dict:
    predictor = build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
    trainable = get_adapter_params(predictor)
    for p in predictor.parameters():
        p.requires_grad = False
    for p in trainable:
        p.requires_grad = True
    opt = torch.optim.AdamW(trainable, lr=lr)

    results = {}
    if 0 in EVAL_CHECKPOINTS:
        results[0] = changed_patches_eval(online, predictor, eval_set, game_vocab, device)

    buffer = []
    max_needed = max(EVAL_CHECKPOINTS)
    for i, t in enumerate(stream):
        if i >= max_needed:
            break
        buffer.append(t)
        n_observed = i + 1
        if n_observed % k == 0:
            adaptation_step(online, predictor, opt, buffer, game_vocab, device, n_steps)
        if n_observed in EVAL_CHECKPOINTS:
            results[n_observed] = changed_patches_eval(online, predictor, eval_set, game_vocab, device)
    return results


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    all_results = {}

    trained_probe = load_sample_transitions(TRAINED_PROBE_GAMES, max_per_game=200)
    print(f"Trained-games probe: {len(trained_probe)} transitions across {TRAINED_PROBE_GAMES}")

    for ckpt_name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {ckpt_name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {ckpt_name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor_state, game_vocab, num_experts, feature_channels, meta = load_checkpoint(ckpt_dir, device)
        print(f"  algorithm={meta.get('algorithm', 'baseline')}  game_vocab={len(game_vocab)} entries "
              f"(should exclude all 5 held-out games)")
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  {n_in_vocab}/{len(HELDOUT_GAMES)} held-out games present in vocab (should be 0)")

        ckpt_results = {"zero_shot_full_heldout": {}, "post_adaptation_trajectories": {}, "trained_games_zero_shot": {}}

        # --- 1. Zero-shot, full held-out population, no adaptation. -------
        fresh_predictor = build_predictor(predictor_state, game_vocab, num_experts, feature_channels, device)
        per_game_zs = {}
        heldout_full = []
        for g in HELDOUT_GAMES:
            per_file = load_game_transitions_per_file(g)
            flat = [t for f in per_file for t in f]
            heldout_full.extend(flat)
            r = changed_patches_eval(online, fresh_predictor, flat, game_vocab, device)
            per_game_zs[g] = r
        zs = changed_patches_eval(online, fresh_predictor, heldout_full, game_vocab, device)
        print(f"\n  [1. zero-shot, full held-out population, n={zs.get('n_changed', 0)}]")
        print(f"    overall: improvement={zs.get('improvement_pct', float('nan')):+.2f}% "
              f"(pred={zs.get('pred_changed_mse', float('nan')):.6f} identity={zs.get('identity_changed_mse', float('nan')):.6f})")
        for g in HELDOUT_GAMES:
            r = per_game_zs.get(g, {})
            print(f"    {g}: improvement={r.get('improvement_pct', float('nan')):+.2f}% (n={r.get('n_changed', 0)})")
        ckpt_results["zero_shot_full_heldout"] = {"overall": zs, "per_game": per_game_zs}

        # --- 2. Trained-games zero-shot sanity check. ----------------------
        tg = changed_patches_eval(online, fresh_predictor, trained_probe, game_vocab, device)
        print(f"\n  [2. trained-games zero-shot sanity check, n={tg.get('n_changed', 0)}]")
        print(f"    improvement={tg.get('improvement_pct', float('nan')):+.2f}% "
              f"(pred={tg.get('pred_changed_mse', float('nan')):.6f} identity={tg.get('identity_changed_mse', float('nan')):.6f})")
        ckpt_results["trained_games_zero_shot"] = tg

        # --- 3. Post-adaptation, K=5/STEPS=8/LR=5e-5, per held-out game. --
        print(f"\n  [3. post-adaptation, K={K} STEPS={N_STEPS} LR={LR}]")
        per_game_post = {}
        for g in HELDOUT_GAMES:
            per_file = load_game_transitions_per_file(g)
            if len(per_file) < 2:
                print(f"    {g}: SKIPPED (fewer than 2 recording files)")
                continue
            eval_set = per_file[-1]
            stream = [t for f in per_file[:-1] for t in f]
            traj = run_adaptation_trajectory(
                online, predictor_state, game_vocab, num_experts, feature_channels,
                stream, eval_set, K, N_STEPS, LR, device,
            )
            per_game_post[g] = traj
            for n_obs in EVAL_CHECKPOINTS:
                r = traj.get(n_obs)
                if r is None:
                    print(f"    {g} n_observed={n_obs:4d}: insufficient stream data (< {n_obs} transitions)")
                elif r.get("n_changed", 0) > 0:
                    print(f"    {g} n_observed={n_obs:4d}: improvement={r['improvement_pct']:+.2f}% "
                          f"(pred={r['pred_changed_mse']:.6f} identity={r['identity_changed_mse']:.6f} n={r['n_changed']})")
                else:
                    print(f"    {g} n_observed={n_obs:4d}: no changed-patch examples")
        ckpt_results["post_adaptation_trajectories"] = per_game_post

        # --- Summary: mean held-out improvement at n=200. -------------------
        n200_vals = []
        pooled_pred = pooled_identity = 0.0
        for g, traj in per_game_post.items():
            r = traj.get(200)
            if r and r.get("n_changed", 0) > 0:
                n200_vals.append(r["improvement_pct"])
                pooled_pred += r["pred_changed_mse"] * r["n_changed"]
                pooled_identity += r["identity_changed_mse"] * r["n_changed"]
        if n200_vals:
            simple_mean = sum(n200_vals) / len(n200_vals)
            print(f"\n  [SUMMARY] post-adaptation n=200: simple per-game mean improvement = {simple_mean:+.2f}% "
                  f"(over {len(n200_vals)}/{len(HELDOUT_GAMES)} games with changed-patch examples)")
            ckpt_results["post_adaptation_n200_simple_mean_pct"] = simple_mean
        if pooled_identity > 0:
            pooled_pct = (pooled_identity - pooled_pred) / pooled_identity * 100
            print(f"  [SUMMARY] post-adaptation n=200: pooled (transition-weighted) improvement = {pooled_pct:+.2f}%")
            ckpt_results["post_adaptation_n200_pooled_pct"] = pooled_pct

        all_results[ckpt_name] = ckpt_results

    out_path = REPO_ROOT / "logs" / "meta_learning_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
