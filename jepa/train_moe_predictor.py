"""Stage 4 dynamics pretraining: mixture-of-gated-experts predictor.

Supports plan.md's originally-specified transfer curriculum: pretrain on
one or more synthetic environment families -- MiniGrid (consistent action
semantics across every layout, cheap to generate in unlimited quantity --
see jepa/data/minigrid_data.py) and, opt-in via `--sokoban-episodes-per-config`,
Sokoban (a distinct "push a movable object with persistent consequences"
mechanic neither ARC-3 nor MiniGrid reliably exercises -- see
jepa/data/sokoban_data.py) -- via `--pretrain-epochs N`, then fine-tune on
the ARC-3 corpus (local recordings + optional external arc-3-logs) for
`--epochs`. Both phases share one encoder/predictor/optimizer and one game
vocabulary (the 25 ARC games + whichever synthetic-source game_ids were
actually used, built up front so the game embedding table is the same size
in both phases) -- `--pretrain-epochs 0` (the default) skips synthetic
pretraining entirely and reproduces the original ARC-3-only training.
Sokoban is off by default (`--sokoban-episodes-per-config 0`) so a
MiniGrid-only run remains a one-flag baseline for comparison.

MoE routing doesn't need Stage 3's temporal ordering the way the
recurrent core did -- both phases use i.i.d.-shuffled single transitions.

Usage:
    python -m jepa.train_moe_predictor --epochs 30 --num-experts 8
    python -m jepa.train_moe_predictor --pretrain-epochs 10 --epochs 30 --num-experts 8
    python -m jepa.train_moe_predictor --pretrain-epochs 10 --epochs 30 --num-experts 8 --sokoban-episodes-per-config 60
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, GAME_ID as MINIGRID_GAME_ID, generate_transitions
from .data.sokoban_data import (
    DEFAULT_CONFIGS as SOKOBAN_DEFAULT_CONFIGS,
    GAME_ID as SOKOBAN_GAME_ID,
    generate_transitions as generate_sokoban_transitions,
)
from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .losses import (
    per_region_error,
    prediction_loss,
    same_color_contrastive_loss,
    variance_regularizer,
    weighted_prediction_loss,
)
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
# stage6-object-identity: same_color_contrastive_loss operates on cosine
# similarity, which lives on a much larger natural scale (0 to ~2 per
# pair) than the main task loss's tiny latent-MSE values -- same
# risk-calibration reasoning as LOAD_BALANCE_WEIGHT above. Default 0.0
# (off) so this script reproduces the original no-contrastive-loss
# recipe unless explicitly requested via --contrast-weight (stage6-
# game-holdout addition: a CLI toggle, not a hardcoded branch-specific
# constant, so one script can produce both the "production-style" and
# "object-identity-style" checkpoints needed for a clean ablation).
CONTRAST_WEIGHT_DEFAULT = 0.0


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
    # JEPA_NUM_WORKERS env var override (ported from stage6-object-identity/
    # stage6-selfplay-bootstrap): CLAUDE.md documents a MemoryError-inside-
    # a-DataLoader-worker's-pickle.load gotcha on Windows spawn-based
    # workers when running on a shared/contended machine -- falling back
    # to 0 sidesteps the spawn/pickle path entirely (slower, but doesn't
    # depend on pagefile/RAM headroom). Unset behaves exactly as before
    # (4 on cuda, 0 on cpu).
    _num_workers_override = os.getenv("JEPA_NUM_WORKERS")
    if _num_workers_override is not None:
        num_workers = int(_num_workers_override)
    else:
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
    online, target, predictor, opt, train_loader, val_loader, device, epochs: int, phase: str,
    contrast_weight: float = CONTRAST_WEIGHT_DEFAULT, checkpoint_cb=None, checkpoint_every: int = 0,
) -> None:
    """checkpoint_cb(epoch_1_indexed, phase), if given, is called every
    `checkpoint_every` epochs (ported from stage6-selfplay-bootstrap /
    stage6-object-identity) -- periodic mid-run saves so an interruption
    on a shared/contended GPU costs at most `checkpoint_every` epochs of
    progress, not the whole run."""
    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = 0.0
        total_lb_loss = 0.0
        total_contrast_loss = 0.0
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
            if contrast_weight > 0.0:
                contrast_loss = same_color_contrastive_loss(cur_feat, cur)
                loss = loss + contrast_weight * contrast_loss
                total_contrast_loss += contrast_loss.item()

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
            + (f"contrast_loss={total_contrast_loss / n_batches:.4f}  " if contrast_weight > 0.0 else "")
            + f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )
        if checkpoint_cb is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_cb(epoch + 1, phase)


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
    sokoban_episodes_per_config: int = 0,
    sokoban_steps_per_episode: int = 80,
    top_k: int | None = None,
    contrast_weight: float = CONTRAST_WEIGHT_DEFAULT,
    exclude_games: list | None = None,
    recording_substrings: list | None = None,
    checkpoint_every: int = 0,
) -> None:
    device = get_device()
    gating = f"top-{top_k} noisy" if top_k is not None else "dense softmax"
    print(f"training on {device}, {num_experts} experts, {gating} gate, contrast_weight={contrast_weight}")
    if exclude_games:
        print(f"excluding games from all local/external corpora: {exclude_games}")

    arc_transitions = load_all_transitions(
        REPO_ROOT, name_substrings=recording_substrings, exclude_games=exclude_games
    )
    n_local = len(arc_transitions)
    print(f"loaded {n_local} local ARC-3 transitions")

    n_external = 0
    if external_per_game:
        external = load_external_transitions(
            REPO_ROOT, max_per_game=external_per_game, exclude_games=exclude_games
        )
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
    sokoban_transitions = []
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
        if sokoban_episodes_per_config > 0:
            sokoban_transitions = generate_sokoban_transitions(
                configs=SOKOBAN_DEFAULT_CONFIGS,
                episodes_per_config=sokoban_episodes_per_config,
                steps_per_episode=sokoban_steps_per_episode,
            )
            print(
                f"generated {len(sokoban_transitions)} Sokoban transitions "
                f"across {len(SOKOBAN_DEFAULT_CONFIGS)} room configs"
            )
    synthetic_transitions = minigrid_transitions + sokoban_transitions

    # One shared vocabulary across both phases -- built from the union of
    # ARC game_ids and (if pretraining) whichever synthetic-source
    # game_ids were actually generated, so the game-embedding table is the
    # same size/meaning in both phases and weights carry over cleanly.
    synthetic_game_ids = {t[6] for t in synthetic_transitions}
    game_ids = sorted({t[6] for t in arc_transitions} | synthetic_game_ids)
    game_vocab = {g: i for i, g in enumerate(game_ids)}
    print(f"{len(game_vocab)} distinct games in the shared vocab")

    online, target, predictor = build_models(
        encoder_path, num_games=len(game_vocab), num_experts=num_experts, device=device, top_k=top_k
    )
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    def _save(tag: str) -> None:
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
                    "n_sokoban_transitions": len(sokoban_transitions),
                    "num_experts": num_experts,
                    "top_k": top_k,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": str(device),
                    "n_local_transitions": n_local,
                    "n_external_transitions": n_external,
                    "external_per_game": external_per_game,
                    "n_games": len(game_vocab),
                    "contrast_weight": contrast_weight,
                    "exclude_games": exclude_games,
                    "checkpoint_tag": tag,
                },
                indent=2,
            )
        )
        print(f"[checkpoint] saved encoder + MoE predictor + game vocab to {out_dir} (tag={tag})")

    def _checkpoint_cb(epoch_1_indexed: int, phase: str) -> None:
        _save(f"{phase}-epoch{epoch_1_indexed}-inprogress")

    if pretrain_epochs > 0:
        mg_train_loader, mg_val_loader = _make_loaders(synthetic_transitions, game_vocab, batch_size, device)
        phase_name = "synthetic-pretrain" if sokoban_transitions else "minigrid-pretrain"
        _run_epochs(
            online, target, predictor, opt, mg_train_loader, mg_val_loader, device, pretrain_epochs, phase_name,
            contrast_weight=contrast_weight, checkpoint_cb=_checkpoint_cb, checkpoint_every=checkpoint_every,
        )
        del mg_train_loader, mg_val_loader  # fully drop before building the next phase's loaders
        if checkpoint_every > 0:
            _save(f"{phase_name}-complete")  # cheap insurance at the phase boundary regardless of the interval

    arc_train_loader, arc_val_loader = _make_loaders(arc_transitions, game_vocab, batch_size, device)
    _run_epochs(
        online, target, predictor, opt, arc_train_loader, arc_val_loader, device, epochs, "arc-finetune",
        contrast_weight=contrast_weight, checkpoint_cb=_checkpoint_cb, checkpoint_every=checkpoint_every,
    )

    _save("final")


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
    parser.add_argument(
        "--sokoban-episodes-per-config",
        type=int,
        default=0,
        help=(
            "Episodes per Sokoban room config to add to the synthetic "
            "pretrain phase (0 = skip, MiniGrid-only pretraining -- see "
            "jepa/data/sokoban_data.py). Only used when --pretrain-epochs > 0."
        ),
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=CONTRAST_WEIGHT_DEFAULT,
        help=(
            "Weight on same_color_contrastive_loss (stage6-object-identity's "
            "encoder object-identity fix, jepa/losses.py). 0.0 (default) "
            "reproduces the original recipe with no contrastive loss; 0.05 "
            "matches the object-identity experiment's own setting. A single "
            "CLI toggle (stage6-game-holdout addition) so this one script "
            "can produce both a 'production-style' and an 'object-identity-"
            "style' checkpoint on identical data for a clean ablation."
        ),
    )
    parser.add_argument(
        "--exclude-games",
        type=str,
        default=None,
        help=(
            "Comma-separated short game codes (e.g. 'r11l,bp35,m0r0,tr87,ka59') "
            "-- skip these games entirely from both the local recordings and "
            "the external arc-3-logs corpus (stage6-game-holdout addition). "
            "Built for leave-some-games-out generalization tests; reusable "
            "for any future one, not a one-off hack."
        ),
    )
    parser.add_argument(
        "--recording-substrings",
        type=str,
        default=None,
        help=(
            "Comma-separated substrings (e.g. '.random.,.solver.') -- only load "
            "local recording files whose name contains at least one of these "
            "(stage6-object-identity addition). Use to pin the training corpus "
            "to a specific file set for an apples-to-apples comparison when the "
            "recordings directory holds files from other sources too."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help=(
            "Save an in-progress checkpoint to --out every N epochs, in "
            "both phases, plus once at the pretrain/finetune phase "
            "boundary (0 = only save once, at the very end -- original "
            "behavior). Ported from stage6-selfplay-bootstrap/stage6-"
            "object-identity: bounds how much progress an interruption on "
            "a shared/contended GPU can cost."
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
        sokoban_episodes_per_config=args.sokoban_episodes_per_config,
        top_k=args.top_k,
        contrast_weight=args.contrast_weight,
        exclude_games=args.exclude_games.split(",") if args.exclude_games else None,
        recording_substrings=args.recording_substrings.split(",") if args.recording_substrings else None,
        checkpoint_every=args.checkpoint_every,
    )
