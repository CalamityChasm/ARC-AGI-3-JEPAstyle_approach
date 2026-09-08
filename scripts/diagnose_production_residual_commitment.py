"""Stage 6 diagnostic: does the production MoEPredictor commit to real
residuals, or coast on the identity skip-connection?

Same methodology as scripts/diagnose_scaled_launch_collapse.py (which ran
this check against the *scaled* 85M-param architecture and found a
0.219-0.382 residual/true-delta ratio after a targeted fine-tune), applied
here to the true production-scale checkpoint (checkpoints/encoder_moe.pt +
moe_predictor.pt, CNNEncoder/MoEPredictor) for the first time -- this
number doesn't exist yet for production. Read-only by default; pass
--checkpoint-dir to point at a fine-tuned checkpoint elsewhere (e.g.
checkpoints_residual_finetune/) for a direct before/after comparison
without ever touching checkpoints/ itself.

Data: local ARC-3 recordings only (ARC-AGI-3-Agents/recordings), the same
"real data" source used everywhere else in this task -- not the MiniGrid/
external corpora the production checkpoint was originally trained on.

Usage:
    python -m scripts.diagnose_production_residual_commitment
    python -m scripts.diagnose_production_residual_commitment --checkpoint-dir checkpoints_residual_finetune
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from jepa.data.trajectories import TransitionDataset, load_all_transitions
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_FRACTION = 0.1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    ckpt_dir = args.checkpoint_dir
    print(f"checkpoint dir: {ckpt_dir}")

    vocab = json.loads((ckpt_dir / "game_vocab_moe.json").read_text())
    print(f"{len(vocab)}-entry game vocab: {sorted(vocab.keys())}")

    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(ckpt_dir / "encoder_moe.pt", map_location=device))
    encoder.eval()
    predictor = MoEPredictor(num_games=max(len(vocab), 1)).to(device)
    predictor.load_state_dict(torch.load(ckpt_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()

    transitions = load_all_transitions(REPO_ROOT)
    print(f"loaded {len(transitions)} local ARC-3 transitions")
    dataset = TransitionDataset(transitions, vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"held-out slice: {len(val_ds)} transitions")

    all_cur_feat, all_next_feat, all_pred_feat, all_patch_mask = [], [], [], []
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

    per_channel_std = cur_feat.std(dim=(0, 2, 3))
    print("\n=== encoder feature collapse check ===")
    print(f"per-channel feature std: mean={per_channel_std.mean().item():.4f} "
          f"min={per_channel_std.min().item():.4f} max={per_channel_std.max().item():.4f}")

    feat_delta = (cur_feat - next_feat).pow(2).mean(dim=1)
    changed_delta = feat_delta[patch_mask]
    unchanged_delta = feat_delta[~patch_mask]
    print("\n=== encoder change-sensitivity ===")
    if changed_delta.numel() and unchanged_delta.numel():
        ratio = changed_delta.mean().item() / max(unchanged_delta.mean().item(), 1e-12)
        print(f"mean feature delta at changed patches: {changed_delta.mean().item():.3e}")
        print(f"mean feature delta at unchanged patches: {unchanged_delta.mean().item():.3e}")
        print(f"ratio (changed/unchanged): {ratio:.2f}x")

    # Residual-commitment check, restricted to changed patches -- same
    # framing as the fresh-vs-compounding rollout diagnostic: at patches
    # that actually changed, does the predictor's residual (pre-skip)
    # magnitude track the true delta, or stay near zero (coasting on the
    # `feat +` skip connection)?
    residual = pred_feat - cur_feat
    true_delta = next_feat - cur_feat
    print("\n=== predictor residual-commitment check (all patches) ===")
    r_all = residual.pow(2).mean().item()
    d_all = true_delta.pow(2).mean().item()
    print(f"mean |residual|^2: {r_all:.3e}   mean |true delta|^2: {d_all:.3e}   "
          f"ratio: {r_all / max(d_all, 1e-12):.3f}")

    print("\n=== predictor residual-commitment check (changed patches only) ===")
    if changed_delta.numel():
        # residual/true_delta are (N, C, H, W); patch_mask is (N, H, W).
        # Broadcast-select changed spatial positions across the channel dim.
        pm = patch_mask.unsqueeze(1).expand_as(residual)
        r_chg = residual[pm].pow(2).mean().item()
        d_chg = true_delta[pm].pow(2).mean().item()
        print(f"mean |residual|^2 (changed): {r_chg:.3e}   mean |true delta|^2 (changed): {d_chg:.3e}   "
              f"ratio: {r_chg / max(d_chg, 1e-12):.3f} "
              "(near 0 = coasting on identity skip-connection at exactly the patches "
              "that matter most)")
    else:
        print("no changed patches in this held-out slice")


if __name__ == "__main__":
    main()
