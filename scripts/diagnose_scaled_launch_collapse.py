"""Checks whether the scaled-curriculum run's final checkpoint
(E:/jepa_overflow/checkpoints_scaled_launch/) shows real representation
collapse, or whether the late-training changed-patches swing is just the
small-absolute-denominator artifact this project has hit before (see
CLAUDE.md item 5 under Stage 1: several games show triple-digit-percent
swings purely because the identity baseline is tiny).

Runs on CUDA (the training process has exited, GPU is free) for speed.
Directly measures encoder feature std on a held-out batch against the
same VARIANCE_FLOOR=1.0 threshold jepa/losses.py already uses, and
separately measures the predictor's own residual magnitude at changed
patches (the Stage 1 item 8 diagnostic: does it commit to real dynamics,
or coast on the identity skip-connection).

Usage: python scripts/diagnose_scaled_launch_collapse.py
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset
from jepa.models.encoder_scaled import ScaledCNNEncoder
from jepa.models.moe_predictor_scaled import ScaledMoEPredictor
from jepa.train_scaled_curriculum import build_game_vocab
from torch.utils.data import DataLoader

CKPT_DIR = Path("E:/jepa_overflow/checkpoints_scaled_launch")
VAL_FRACTION = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="", help="e.g. 'best_' to load best_encoder_scaled.pt etc.")
    parser.add_argument("--device", default=None, help="Override device (default: cuda if available, else cpu).")
    cli_args = parser.parse_args()

    device = torch.device(cli_args.device) if cli_args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    meta = json.loads((CKPT_DIR / f"{cli_args.prefix}curriculum_meta.json").read_text())
    args = meta["args"]
    print(f"checkpoint: completed_epochs={meta['completed_epochs']}/{meta['total_epochs']}"
          + (f"  best_pred_changed_mse={meta['best_pred_changed_mse']:.6f}" if "best_pred_changed_mse" in meta else ""))

    t0 = time.time()
    with open(CKPT_DIR / "corpus_cache.pkl", "rb") as f:
        sources = pickle.load(f)
    print(f"loaded corpus cache ({time.time() - t0:.1f}s)")

    game_vocab = build_game_vocab(sources)

    rng = np.random.default_rng(0)
    local_transitions, _cat = sources["arc3_local"]
    n_val = max(1, int(len(local_transitions) * VAL_FRACTION))
    idx = rng.permutation(len(local_transitions))[:n_val]
    val_transitions = [local_transitions[i] for i in idx]
    print(f"arc3_local held-out slice: {len(val_transitions)} transitions")

    encoder = ScaledCNNEncoder(
        out_channels=args["feature_channels"],
        width_mult=args["width_mult"],
        blocks_per_stage=args["blocks_per_stage"],
    ).to(device)
    predictor = ScaledMoEPredictor(
        feature_channels=encoder.out_channels,
        num_games=len(game_vocab),
        num_experts=args["num_experts"],
        expert_hidden=args["expert_hidden"],
        expert_depth=args["expert_depth"],
        top_k=args["top_k"],
    ).to(device)
    encoder.load_state_dict(torch.load(CKPT_DIR / f"{cli_args.prefix}encoder_scaled.pt", map_location=device))
    predictor.load_state_dict(torch.load(CKPT_DIR / f"{cli_args.prefix}moe_predictor_scaled.pt", map_location=device))
    encoder.eval()
    predictor.eval()

    dataset = TransitionDataset(val_transitions, game_vocab)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    all_cur_feat = []
    all_next_feat = []
    all_pred_feat = []
    all_patch_mask = []
    with torch.no_grad():
        for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
            cur_feat = encoder(cur)
            pred_feat, _gw = predictor(cur_feat, action_id, xy, game_idx)
            next_feat = encoder(nxt)
            all_cur_feat.append(cur_feat.cpu())
            all_next_feat.append(next_feat.cpu())
            all_pred_feat.append(pred_feat.cpu())
            all_patch_mask.append(patch_mask.cpu())

    cur_feat = torch.cat(all_cur_feat, dim=0)
    next_feat = torch.cat(all_next_feat, dim=0)
    pred_feat = torch.cat(all_pred_feat, dim=0)
    patch_mask = torch.cat(all_patch_mask, dim=0)

    # 1. Encoder feature collapse check (Stage 1's own VARIANCE_FLOOR=1.0 test).
    per_channel_std = cur_feat.std(dim=(0, 2, 3))
    print("\n=== encoder feature collapse check ===")
    print(f"per-channel feature std: mean={per_channel_std.mean().item():.4f} "
          f"min={per_channel_std.min().item():.4f} max={per_channel_std.max().item():.4f}")
    print("(VARIANCE_FLOOR=1.0 is the threshold jepa/losses.py already trains against -- "
          "well below this means the encoder itself has collapsed to a near-constant output)")

    # 2. Does the encoder register real changes at all? (Stage 1 item 8's first check.)
    feat_delta = (cur_feat - next_feat).pow(2).mean(dim=1)  # (B, 8, 8)
    changed_delta = feat_delta[patch_mask]
    unchanged_delta = feat_delta[~patch_mask]
    print("\n=== encoder change-sensitivity ===")
    if changed_delta.numel() and unchanged_delta.numel():
        ratio = changed_delta.mean().item() / max(unchanged_delta.mean().item(), 1e-12)
        print(f"mean feature delta at changed patches: {changed_delta.mean().item():.3e}")
        print(f"mean feature delta at unchanged patches: {unchanged_delta.mean().item():.3e}")
        print(f"ratio (changed/unchanged): {ratio:.2f}x")
    else:
        print("no changed or unchanged patches in this held-out slice to compare")

    # 3. Predictor residual-commitment check (Stage 1 item 8's second check, and
    #    exactly the mechanism the Stage 6 held-out-games collapse traced back to).
    residual = pred_feat - cur_feat
    true_delta = next_feat - cur_feat
    print("\n=== predictor residual-commitment check ===")
    print(f"mean |residual|^2 (all patches): {residual.pow(2).mean().item():.3e}")
    print(f"mean |true delta|^2 (all patches): {true_delta.pow(2).mean().item():.3e}")
    ratio2 = residual.pow(2).mean().item() / max(true_delta.pow(2).mean().item(), 1e-12)
    print(f"residual/true-delta ratio: {ratio2:.3f} "
          "(near 0 means the predictor is coasting on the identity skip-connection "
          "rather than committing to real dynamics -- the Stage 1 item 8 / Stage 6 "
          "held-out-games failure mode)")


if __name__ == "__main__":
    main()
