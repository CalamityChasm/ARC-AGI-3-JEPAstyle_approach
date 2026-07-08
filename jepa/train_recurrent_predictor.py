"""Stage 3 dynamics pretraining: recurrent memory core on top of the
Stage 1 predictor architecture.

Trains on ordered episode sequences (`jepa/data/sequences.py`), not the
i.i.d.-shuffled single transitions Stage 1 used -- a recurrent core has
nothing to learn from if consecutive training examples aren't actually
consecutive in time. Each fixed-length chunk (`--seq-len`, default 16)
starts with a zeroed hidden state and runs truncated BPTT across the whole
chunk; hidden state does not persist *across* chunks of the same episode
(a simplification -- still teaches the model to use recent history within
a chunk, which is what the Stage 3 milestone actually needs).

Usage: python -m jepa.train_recurrent_predictor [--epochs N] [--encoder PATH] [--out PATH]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data.sequences import EpisodeSequenceDataset, build_game_vocab, load_all_episodes
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, RecurrentActionConditionedPredictor, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1
SEQ_LEN = 16


def build_models(encoder_path: Path | None, num_games: int, device: torch.device) -> tuple:
    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = RecurrentActionConditionedPredictor(num_games=num_games).to(device)
    return online, target, predictor


def _run_sequence(
    online, predictor, cur, action_id, xy, nxt, patch_mask, game_idx, device, target=None
) -> list:
    """Runs one (B, T, ...) batch of sequence chunks through the encoder +
    recurrent predictor with truncated BPTT across the T steps. Returns a
    list of per-step (cur_feat, pred_feat, target_feat, patch_mask) tuples
    so callers can compute loss or eval metrics identically."""
    b, t = cur.shape[0], cur.shape[1]
    hidden = predictor.init_hidden(b, device)
    outputs = []
    for step in range(t):
        cur_feat = online(cur[:, step])
        pred_feat, hidden = predictor(cur_feat, action_id[:, step], xy[:, step], hidden, game_idx)
        if target is not None:
            with torch.no_grad():
                target_feat = target(nxt[:, step])
        else:
            target_feat = online(nxt[:, step])
        outputs.append((cur_feat, pred_feat, target_feat, patch_mask[:, step]))
    return outputs


def train(
    epochs: int,
    encoder_path: Path,
    out_dir: Path,
    batch_size: int = 8,
    lr: float = 3e-4,
    seq_len: int = SEQ_LEN,
) -> None:
    device = get_device()
    print(f"training on {device}")
    episodes = load_all_episodes(REPO_ROOT)
    print(f"loaded {len(episodes)} episodes")
    game_vocab = build_game_vocab(episodes)
    print(f"{len(game_vocab)} distinct games")
    dataset = EpisodeSequenceDataset(episodes, game_vocab, seq_len=seq_len)
    print(f"{len(dataset)} sequence chunks of length {seq_len}")

    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    num_workers = 4 if device.type == "cuda" else 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    online, target, predictor = build_models(encoder_path, num_games=len(game_vocab), device=device)
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)

            outputs = _run_sequence(
                online, predictor, cur, action_id, xy, nxt, patch_mask, game_idx, device, target=target
            )
            loss = 0.0
            for cur_feat, pred_feat, target_feat, mask in outputs:
                loss = loss + weighted_prediction_loss(pred_feat, target_feat, mask) + variance_regularizer(
                    cur_feat
                )
            loss = loss / len(outputs)

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            n_batches += 1

        stats = evaluate(online, predictor, val_loader, device=device)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_recurrent.pt")
    torch.save(
        {k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "recurrent_predictor.pt"
    )
    (out_dir / "game_vocab_recurrent.json").write_text(json.dumps(game_vocab, indent=2))
    print(f"saved encoder + recurrent predictor + game vocab to {out_dir}")


@torch.no_grad()
def evaluate(online, predictor, loader, device: torch.device) -> dict:
    """Same fair same-encoder-on-both-sides comparison as Stage 1's
    train_predictor.evaluate (see CLAUDE.md iteration #1 for why that
    matters), extended across a whole sequence chunk."""
    online.eval()
    predictor.eval()
    totals = {"pred": 0.0, "identity": 0.0, "pred_changed": 0.0, "identity_changed": 0.0}
    n_batches = 0
    n_changed_batches = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        outputs = _run_sequence(
            online, predictor, cur, action_id, xy, nxt, patch_mask, game_idx, device, target=None
        )
        for cur_feat, pred_feat, next_feat, mask in outputs:
            totals["pred"] += prediction_loss(pred_feat, next_feat).item()
            totals["identity"] += prediction_loss(cur_feat, next_feat).item()
            n_batches += 1
            if mask.any():
                pred_err = per_region_error(pred_feat, next_feat)[mask]
                identity_err = per_region_error(cur_feat, next_feat)[mask]
                totals["pred_changed"] += pred_err.mean().item()
                totals["identity_changed"] += identity_err.mean().item()
                n_changed_batches += 1

    online.train()
    predictor.train()
    n_changed_batches = max(n_changed_batches, 1)
    return {
        "pred": totals["pred"] / n_batches,
        "identity": totals["identity"] / n_batches,
        "pred_changed": totals["pred_changed"] / n_changed_batches,
        "identity_changed": totals["identity_changed"] / n_changed_batches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()
    train(args.epochs, args.encoder, args.out, seq_len=args.seq_len)
