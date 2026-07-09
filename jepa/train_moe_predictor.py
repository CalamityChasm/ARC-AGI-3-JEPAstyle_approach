"""Stage 4 dynamics pretraining: mixture-of-gated-experts predictor.

Supports plan.md's originally-specified transfer curriculum: pretrain on
MiniGrid (consistent action semantics across every layout, cheap to
generate in unlimited quantity -- see jepa/data/minigrid_data.py) via
`--pretrain-epochs N`, then fine-tune on the ARC-3 corpus (local
recordings + optional external arc-3-logs) for `--epochs`. Both phases
share one encoder/predictor/optimizer and one game vocabulary (the 25 ARC
games + a single shared "minigrid" entry, built up front so the game
embedding table is the same size in both phases) -- `--pretrain-epochs 0`
(the default) skips MiniGrid entirely and reproduces the original
ARC-3-only training.

MoE routing doesn't need Stage 3's temporal ordering the way the
recurrent core did -- both phases use i.i.d.-shuffled single transitions.

Usage:
    python -m jepa.train_moe_predictor --epochs 30 --num-experts 8
    python -m jepa.train_moe_predictor --pretrain-epochs 10 --epochs 30 --num-experts 8
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, GAME_ID as MINIGRID_GAME_ID, generate_transitions
from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, MoEPredictor, load_balance_loss, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1
# Small: the main task loss operates at ~1e-2 to 1e-4 scale (weighted MSE
# on tiny latent residuals), so even nominally-small aux-loss weights can
# dominate it and collapse the gate to a constant uniform distribution
# (exactly what happened at 0.01 -- confirmed via gate_weights having
# ~0 variance across a whole validation batch). This is deliberately much
# smaller than typical LLM-MoE recipes, which apply a similar-looking
# weight against a much larger main loss.
LOAD_BALANCE_WEIGHT = 0.001


def build_models(
    encoder_path: Path | None,
    num_games: int,
    num_experts: int,
    device: torch.device,
    top_k: int | None = None,
) -> tuple:
    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = MoEPredictor(num_games=num_games, num_experts=num_experts, top_k=top_k).to(device)
    return online, target, predictor


def _make_loaders(transitions: list, game_vocab: dict, batch_size: int, device: torch.device) -> tuple:
    dataset = TransitionDataset(transitions, game_vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    all_weights = dataset.sample_weights()
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)
    num_workers = 4 if device.type == "cuda" else 0
    # NOT persistent_workers here (unlike train_predictor.py): this script
    # builds a *second* set of DataLoaders for the ARC fine-tuning phase
    # after the MiniGrid pretrain phase's loaders go out of scope, and the
    # first attempt at that (persistent_workers=True on both) hung
    # indefinitely -- the old phase's persistent worker processes weren't
    # torn down before the new phase's workers started, some kind of
    # resource contention (CUDA context / pinned memory) rather than a
    # clean handoff. Not worth debugging further given the fix is free:
    # this script only ever has (at most) one phase transition, so the
    # per-epoch worker respawn cost persistent_workers avoids barely
    # matters here.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _run_epochs(
    online, target, predictor, opt, train_loader, val_loader, device, epochs: int, phase: str
) -> None:
    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = 0.0
        total_lb_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
            cur_feat = online(cur)
            pred_feat, gate_weights = predictor(cur_feat, action_id, xy, game_idx)
            with torch.no_grad():
                target_feat = target(nxt)

            lb_loss = load_balance_loss(gate_weights)
            loss = (
                weighted_prediction_loss(pred_feat, target_feat, patch_mask)
                + variance_regularizer(cur_feat)
                + LOAD_BALANCE_WEIGHT * lb_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            total_lb_loss += lb_loss.item()
            n_batches += 1

        stats = evaluate(online, predictor, val_loader, device=device)
        print(
            f"[{phase}] epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"lb_loss={total_lb_loss / n_batches:.3f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )


def train(
    epochs: int,
    encoder_path: Path,
    out_dir: Path,
    num_experts: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    external_per_game: int | None = None,
    pretrain_epochs: int = 0,
    minigrid_episodes_per_env: int = 40,
    minigrid_steps_per_episode: int = 80,
    top_k: int | None = None,
) -> None:
    device = get_device()
    gating = f"top-{top_k} noisy" if top_k is not None else "dense softmax"
    print(f"training on {device}, {num_experts} experts, {gating} gate")

    arc_transitions = load_all_transitions(REPO_ROOT)
    n_local = len(arc_transitions)
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
            arc_transitions += external
        else:
            print(
                "--external-per-game set but data/arc3_logs.zip is missing -- "
                "training on local ARC-3 transitions only (see CLAUDE.md to pull it)"
            )

    minigrid_transitions = []
    if pretrain_epochs > 0:
        minigrid_transitions = generate_transitions(
            env_names=DEFAULT_ENV_NAMES,
            episodes_per_env=minigrid_episodes_per_env,
            steps_per_episode=minigrid_steps_per_episode,
        )
        print(
            f"generated {len(minigrid_transitions)} MiniGrid transitions "
            f"across {len(DEFAULT_ENV_NAMES)} environments"
        )

    # One shared vocabulary across both phases -- built from the union of
    # ARC game_ids and (if pretraining) the single "minigrid" game_id, so
    # the game-embedding table is the same size/meaning in both phases and
    # weights carry over cleanly.
    game_ids = sorted({t[6] for t in arc_transitions} | ({MINIGRID_GAME_ID} if pretrain_epochs > 0 else set()))
    game_vocab = {g: i for i, g in enumerate(game_ids)}
    print(f"{len(game_vocab)} distinct games in the shared vocab")

    online, target, predictor = build_models(
        encoder_path, num_games=len(game_vocab), num_experts=num_experts, device=device, top_k=top_k
    )
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    if pretrain_epochs > 0:
        mg_train_loader, mg_val_loader = _make_loaders(minigrid_transitions, game_vocab, batch_size, device)
        _run_epochs(
            online, target, predictor, opt, mg_train_loader, mg_val_loader, device, pretrain_epochs, "minigrid-pretrain"
        )
        del mg_train_loader, mg_val_loader  # fully drop before building the next phase's loaders

    arc_train_loader, arc_val_loader = _make_loaders(arc_transitions, game_vocab, batch_size, device)
    _run_epochs(online, target, predictor, opt, arc_train_loader, arc_val_loader, device, epochs, "arc-finetune")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_moe.pt")
    torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "moe_predictor.pt")
    (out_dir / "game_vocab_moe.json").write_text(json.dumps(game_vocab, indent=2))
    (out_dir / "moe_training_meta.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "pretrain_epochs": pretrain_epochs,
                "n_minigrid_transitions": len(minigrid_transitions),
                "num_experts": num_experts,
                "top_k": top_k,
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
    print(f"saved encoder + MoE predictor + game vocab to {out_dir}")


@torch.no_grad()
def evaluate(online, predictor, loader, device: torch.device | None = None) -> dict:
    """Same fair same-encoder comparison as Stage 1's train_predictor.evaluate."""
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
        pred_feat, _gate_weights = predictor(cur_feat, action_id, xy, game_idx)
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
    parser.add_argument("--epochs", type=int, default=30, help="ARC-3 fine-tuning epochs.")
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="MiniGrid pretraining epochs (0 = skip, ARC-3-only training).",
    )
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--external-per-game",
        type=int,
        default=None,
        help="Mix in up to this many transitions per game from the external arc-3-logs dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Noisy top-k gating: route to only the top-k experts per example "
            "(softmax over just those) instead of a dense blend over all "
            "--num-experts. Omit for the original dense softmax gate."
        ),
    )
    args = parser.parse_args()
    train(
        args.epochs,
        args.encoder,
        args.out,
        num_experts=args.num_experts,
        external_per_game=args.external_per_game,
        pretrain_epochs=args.pretrain_epochs,
        top_k=args.top_k,
    )
