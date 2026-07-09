"""Stage 5: trains the decoupled value head on local ARC-3 recordings.

Uses the frozen Stage 1 encoder (checkpoints/encoder_finetuned.pt) --
this is a small, fast regression on top of already-good features, not
worth another joint fine-tuning pass.

Usage: python -m jepa.train_value_head [--epochs N] [--out PATH]
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .data.value_targets import load_value_targets
from .device import get_device
from .grid import arc3_frame_to_tensor
from .models import CNNEncoder, ValueHead

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_FRACTION = 0.1


class ValueDataset(Dataset):
    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        frame, target, _game_id = self.samples[idx]
        tensor = arc3_frame_to_tensor(frame)
        return torch.from_numpy(tensor), torch.tensor(target, dtype=torch.float32)


def train(
    epochs: int, encoder_path: Path, out_dir: Path, batch_size: int = 64, lr: float = 1e-3
) -> None:
    device = get_device()
    print(f"training on {device}")
    samples = load_value_targets(REPO_ROOT)
    print(f"loaded {len(samples)} value-target samples")
    nonzero = sum(1 for _, t, _ in samples if t > 1e-3)
    print(f"{nonzero} samples ({nonzero / len(samples) * 100:.1f}%) have a nonzero value target")

    dataset = ValueDataset(samples)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    value_head = ValueHead().to(device)
    opt = torch.optim.AdamW(value_head.parameters(), lr=lr)

    for epoch in range(epochs):
        value_head.train()
        total_loss = 0.0
        n_batches = 0
        for cur, target in train_loader:
            cur, target = cur.to(device), target.to(device)
            with torch.no_grad():
                feat = encoder(cur)
            pred = value_head(feat)
            loss = F.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        val_loss, val_baseline = evaluate(encoder, value_head, val_loader, device)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"val_mse={val_loss:.4f}  zero-baseline_mse={val_baseline:.4f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in value_head.state_dict().items()}, out_dir / "value_head.pt")
    print(f"saved value head to {out_dir}")


@torch.no_grad()
def evaluate(encoder, value_head, loader, device: torch.device) -> tuple:
    """Also reports a "always predict 0" baseline MSE -- given how sparse
    the reward signal is, this is the real bar (mirrors Stage 1's identity
    baseline in spirit: is the value head learning anything beyond
    "nothing happens"?)."""
    value_head.eval()
    total_loss = 0.0
    total_baseline = 0.0
    n = 0
    for cur, target in loader:
        cur, target = cur.to(device), target.to(device)
        feat = encoder(cur)
        pred = value_head(feat)
        total_loss += F.mse_loss(pred, target, reduction="sum").item()
        total_baseline += (target**2).sum().item()
        n += target.numel()
    value_head.train()
    return total_loss / n, total_baseline / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder_finetuned.pt"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints")
    args = parser.parse_args()
    train(args.epochs, args.encoder, args.out)
