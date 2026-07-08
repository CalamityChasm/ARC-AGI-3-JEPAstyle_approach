"""Dynamics pretraining: action-conditioned one-step latent predictor.

Starts from the ARC-1/2-pretrained encoder (warm start, per plan.md's
transfer-learning curriculum -- the encoder transfers, the dynamics mostly
don't, so we fine-tune the encoder lightly here too rather than freezing
it). Trains on the Stage 0 ARC-3 trajectory corpus.

Usage: python -m jepa.train_predictor [--epochs N] [--encoder PATH] [--out PATH]
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.trajectories import TransitionDataset, build_game_vocab, load_all_transitions
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import ActionConditionedPredictor, CNNEncoder, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1


def build_models(encoder_path: Path | None, num_games: int, device: torch.device) -> tuple:
    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = ActionConditionedPredictor(num_games=num_games).to(device)
    return online, target, predictor


def train(
    epochs: int,
    encoder_path: Path,
    out_dir: Path,
    batch_size: int = 32,
    lr: float = 3e-4,
    external_per_game: int | None = None,
) -> None:
    device = get_device()
    print(f"training on {device}")
    transitions = load_all_transitions(REPO_ROOT)
    n_local = len(transitions)
    print(f"loaded {n_local} local ARC-3 transitions")

    n_external = 0
    if external_per_game:
        external = load_external_transitions(REPO_ROOT, max_per_game=external_per_game)
        n_external = len(external)
        if external:
            print(
                f"loaded {n_external} external ARC-3 transitions "
                f"(arc-3-logs, capped at {external_per_game}/game)"
            )
            transitions += external
        else:
            print(
                "--external-per-game set but data/arc3_logs.zip is missing -- "
                "training on local transitions only (see CLAUDE.md to pull it)"
            )

    game_vocab = build_game_vocab(transitions)
    print(f"{len(game_vocab)} distinct games")
    dataset = TransitionDataset(transitions, game_vocab)

    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # Random-policy rollouts are dominated by no-op transitions (action had
    # no visible effect); oversample the ones that actually changed so the
    # predictor sees enough real-dynamics signal instead of mostly learning
    # "predict no change".
    all_weights = dataset.sample_weights()
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    # __getitem__ does CPU-side one-hot conversion per sample (see
    # trajectories.py); with a GPU doing the actual compute, single-process
    # loading (num_workers=0) becomes the bottleneck once the corpus is
    # large (e.g. with --external-per-game mixed in). Workers persist across
    # epochs so they aren't respawned every epoch.
    num_workers = 4 if device.type == "cuda" else 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    online, target, predictor = build_models(encoder_path, num_games=len(game_vocab), device=device)
    opt = torch.optim.AdamW(
        list(online.parameters()) + list(predictor.parameters()), lr=lr
    )

    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
            cur_feat = online(cur)
            pred_feat = predictor(cur_feat, action_id, xy, game_idx)
            with torch.no_grad():
                target_feat = target(nxt)

            loss = weighted_prediction_loss(pred_feat, target_feat, patch_mask) + variance_regularizer(
                cur_feat
            )

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
    torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_finetuned.pt")
    torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "predictor.pt")
    (out_dir / "game_vocab.json").write_text(json.dumps(game_vocab, indent=2))
    (out_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(device),
                "n_local_transitions": n_local,
                "n_external_transitions": n_external,
                "external_per_game": external_per_game,
                "n_games": len(game_vocab),
            },
            indent=2,
        )
    )
    print(f"saved encoder + predictor weights + game vocab to {out_dir}")


@torch.no_grad()
def evaluate(online, predictor, loader, device: torch.device | None = None) -> dict:
    """Fair comparison: both the predictor's target and the identity
    baseline's target are encoded with the *same* (online) encoder, so
    neither side gets an advantage/penalty from online/EMA weight drift.
    `target` (the EMA encoder) is only used during training's loss, not here.

    Also reports the changed-patches-only breakdown: the real test of
    "learned dynamics" is whether the predictor beats identity specifically
    on the handful of patches that actually moved, not just in an aggregate
    dominated by static patches.
    """
    online.eval()
    predictor.eval()
    totals = {"pred": 0.0, "identity": 0.0, "pred_changed": 0.0, "identity_changed": 0.0}
    n_batches = 0
    n_changed_batches = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        if device is not None:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        cur_feat = online(cur)
        pred_feat = predictor(cur_feat, action_id, xy, game_idx)
        next_feat = online(nxt)

        totals["pred"] += prediction_loss(pred_feat, next_feat).item()
        totals["identity"] += prediction_loss(cur_feat, next_feat).item()
        n_batches += 1

        if patch_mask.any():
            pred_err = per_region_error(pred_feat, next_feat)[patch_mask]
            identity_err = per_region_error(cur_feat, next_feat)[patch_mask]
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--external-per-game",
        type=int,
        default=None,
        help=(
            "Mix in up to this many transitions per game from the external "
            "arc-3-logs Kaggle dataset (data/arc3_logs.zip), on top of the "
            "local recordings. Omit to train on local recordings only."
        ),
    )
    args = parser.parse_args()
    train(args.epochs, args.encoder, args.out, external_per_game=args.external_per_game)
