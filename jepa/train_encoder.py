"""Masked-patch JEPA pretraining for the CNN encoder, on ARC-1/2 grids.

Mask random 8x8 patches of the input grid, encode the masked grid with the
online encoder, encode the *unmasked* grid with the EMA target encoder
(stop-grad), and minimize latent MSE at the masked patches only. This is
the "encoder" half of Stage 1 (plan.md): near-zero domain gap to ARC-3
frames, thousands of examples, no action signal needed.

Usage: python -m jepa.train_encoder [--epochs N] [--out PATH]
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.arc_static import MaskedGridDataset, load_all_grids
from .losses import prediction_loss, variance_regularizer
from .models import CNNEncoder, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996


def train(epochs: int, out_path: Path, batch_size: int = 64, lr: float = 3e-4) -> CNNEncoder:
    grids = load_all_grids(REPO_ROOT)
    print(f"loaded {len(grids)} ARC-1/2 grids")
    dataset = MaskedGridDataset(grids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    online = CNNEncoder()
    target = make_ema_target(online)
    opt = torch.optim.AdamW(online.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for masked, unmasked, patch_mask in loader:
            pred_feat = online(masked)  # (B, C, 8, 8)
            with torch.no_grad():
                target_feat = target(unmasked)  # (B, C, 8, 8)

            m = patch_mask.unsqueeze(1)  # (B, 1, 8, 8)
            if m.any():
                pred_masked = pred_feat.permute(0, 2, 3, 1)[patch_mask]
                target_masked = target_feat.permute(0, 2, 3, 1)[patch_mask]
                loss = prediction_loss(pred_masked, target_masked)
            else:
                loss = prediction_loss(pred_feat, target_feat)
            loss = loss + variance_regularizer(pred_feat)

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch + 1}/{epochs}  loss={total_loss / n_batches:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(online.state_dict(), out_path)
    print(f"saved encoder weights to {out_path}")
    return online


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    args = parser.parse_args()
    train(args.epochs, args.out)
