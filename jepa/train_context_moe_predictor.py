"""Stage 6 continuous-game-embedding investigation, Phase 2B(b): trains
the MoE predictor with `context_mode="external"`, conditioned on a
`jepa.models.context_encoder.EpisodeContextEncoder` embedding computed
from K other transitions observed earlier in the same episode, instead
of a categorical game_id lookup or a single-frame content embedding
(Phase 2B(a)).

Co-trains three components together (online encoder, MoEPredictor,
EpisodeContextEncoder) -- the online encoder is used to embed BOTH the
target transition and all K context transitions each step, so context
and target always share one latent space, and gradients flow back
through the encoder from context frames too (not just the target).

Local-only (no MiniGrid pretrain, no external arc-3-logs augmentation) --
see experiments/stage6_continuous_game_embedding.md's Phase 2B(b) section
for why this deviation from Phase 2B(a)'s fully-matched recipe was
accepted for this scoped, single-fold preliminary test: episode-context
construction depends on jepa/data/sequences.py's per-episode ordering,
which (like Stage 3's recurrent predictor) only exists for local
recordings.

Usage:
    python -m jepa.train_context_moe_predictor --epochs 60 \
        --exclude-games r11l,bp35,m0r0,tr87,ka59 --out checkpoints_fold1_episode_context
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.episode_context import CONTEXT_WINDOW, load_episode_context_dataset
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, EpisodeContextEncoder, MoEPredictor, load_balance_loss, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1
LOAD_BALANCE_WEIGHT = 0.001  # same value/rationale as train_moe_predictor.py


def build_models(
    encoder_path: Path | None, num_experts: int, device: torch.device, context_window: int = CONTEXT_WINDOW,
) -> tuple:
    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    # num_games=1 is irrelevant in context_mode="external" (no game_embed
    # table is built at all -- see MoEPredictor.__init__), kept only
    # because the constructor still accepts the argument for API
    # consistency with "categorical"/"frame" modes.
    predictor = MoEPredictor(num_games=1, num_experts=num_experts, context_mode="external").to(device)
    context_encoder = EpisodeContextEncoder(feature_channels=64, embed_dim=16).to(device)
    return online, target, predictor, context_encoder


def _pooled_context_embed(online, context_encoder, ctx_cur, ctx_action, ctx_nxt, device) -> torch.Tensor:
    """ctx_cur, ctx_nxt: (B, K, 17, 64, 64); ctx_action: (B, K) long.
    Runs the online encoder over every context frame (both t and t+1,
    flattened into one batched forward pass for efficiency), then calls
    EpisodeContextEncoder on the resulting pooled features. Returns
    (B, embed_dim)."""
    b, k = ctx_action.shape
    ctx_cur_flat = ctx_cur.view(b * k, *ctx_cur.shape[2:])
    ctx_nxt_flat = ctx_nxt.view(b * k, *ctx_nxt.shape[2:])
    both = torch.cat([ctx_cur_flat, ctx_nxt_flat], dim=0)  # (2*B*K, 17, 64, 64)
    both_feat = online(both)  # (2*B*K, C, 8, 8)
    both_pooled = both_feat.mean(dim=(2, 3))  # (2*B*K, C)
    pooled_t, pooled_t1 = both_pooled.chunk(2, dim=0)  # each (B*K, C)
    pooled_t = pooled_t.view(b, k, -1)
    pooled_t1 = pooled_t1.view(b, k, -1)
    return context_encoder(pooled_t, ctx_action, pooled_t1)


def _make_loaders(dataset, batch_size: int, device: torch.device) -> tuple:
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    all_weights = dataset.sample_weights()
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    # Local-only, no persistent_workers/num_workers>0 by default here:
    # this is a scoped, single-fold preliminary test (see module
    # docstring), and each __getitem__ already does more work than
    # trajectories.py's (K+1 frames instead of 2) -- keeping the loader
    # simple avoids adding another moving part to debug under time
    # pressure. Revisit if this graduates beyond a preliminary test.
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train(
    epochs: int,
    encoder_path: Path,
    out_dir: Path,
    num_experts: int = 8,
    batch_size: int = 16,
    lr: float = 3e-4,
    context_window: int = CONTEXT_WINDOW,
    exclude_games: list | None = None,
) -> None:
    device = get_device()
    print(f"training on {device}, {num_experts} experts, context_mode=external, context_window={context_window}")
    if exclude_games:
        print(f"excluding games from local recordings: {exclude_games}")

    dataset, game_vocab = load_episode_context_dataset(
        REPO_ROOT, context_window=context_window, exclude_games=exclude_games
    )
    print(f"loaded {len(game_vocab)} distinct games, {len(dataset)} (target, context) examples")

    train_loader, val_loader = _make_loaders(dataset, batch_size, device)

    online, target, predictor, context_encoder = build_models(
        encoder_path, num_experts=num_experts, device=device, context_window=context_window
    )
    opt = torch.optim.AdamW(
        list(online.parameters()) + list(predictor.parameters()) + list(context_encoder.parameters()), lr=lr
    )

    def _save(tag: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_context.pt")
        torch.save(
            {k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "context_moe_predictor.pt"
        )
        torch.save(
            {k: v.cpu() for k, v in context_encoder.state_dict().items()},
            out_dir / "episode_context_encoder.pt",
        )
        (out_dir / "context_training_meta.json").write_text(
            json.dumps(
                {
                    "epochs": epochs,
                    "num_experts": num_experts,
                    "context_window": context_window,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": str(device),
                    "n_examples": len(dataset),
                    "n_games": len(game_vocab),
                    "exclude_games": exclude_games,
                    "context_mode": "external",
                    "checkpoint_tag": tag,
                },
                indent=2,
            )
        )
        print(f"[checkpoint] saved encoder + context MoE predictor + episode context encoder to {out_dir} (tag={tag})")

    for epoch in range(epochs):
        online.train()
        predictor.train()
        context_encoder.train()
        total_loss = 0.0
        total_lb_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, _game_idx, ctx_cur, ctx_action, ctx_nxt in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask = nxt.to(device), patch_mask.to(device)
            ctx_cur, ctx_action, ctx_nxt = ctx_cur.to(device), ctx_action.to(device), ctx_nxt.to(device)

            context_embed = _pooled_context_embed(online, context_encoder, ctx_cur, ctx_action, ctx_nxt, device)
            cur_feat = online(cur)
            pred_feat, gate_weights = predictor(cur_feat, action_id, xy, context_embed=context_embed)
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

        stats = evaluate(online, predictor, context_encoder, val_loader, device=device)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"lb_loss={total_lb_loss / n_batches:.3f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )

    _save("final")


@torch.no_grad()
def evaluate(online, predictor, context_encoder, loader, device: torch.device) -> dict:
    """Same fair same-encoder comparison as every other eval in this
    project (see CLAUDE.md iteration #1)."""
    online.eval()
    predictor.eval()
    context_encoder.eval()
    totals = {"pred": 0.0, "identity": 0.0, "pred_changed": 0.0, "identity_changed": 0.0}
    n_batches = 0
    n_changed_batches = 0
    for cur, action_id, xy, nxt, patch_mask, _game_idx, ctx_cur, ctx_action, ctx_nxt in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask = nxt.to(device), patch_mask.to(device)
        ctx_cur, ctx_action, ctx_nxt = ctx_cur.to(device), ctx_action.to(device), ctx_nxt.to(device)

        context_embed = _pooled_context_embed(online, context_encoder, ctx_cur, ctx_action, ctx_nxt, device)
        cur_feat = online(cur)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, context_embed=context_embed)
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
    context_encoder.train()
    n_changed_batches = max(n_changed_batches, 1)
    return {
        "pred": totals["pred"] / n_batches,
        "identity": totals["identity"] / n_batches,
        "pred_changed": totals["pred_changed"] / n_changed_batches,
        "identity_changed": totals["identity_changed"] / n_changed_batches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_episode_context")
    parser.add_argument("--context-window", type=int, default=CONTEXT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--exclude-games",
        type=str,
        default=None,
        help="Comma-separated short game codes to exclude from the local recordings corpus.",
    )
    args = parser.parse_args()
    train(
        args.epochs,
        args.encoder,
        args.out,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        context_window=args.context_window,
        exclude_games=args.exclude_games.split(",") if args.exclude_games else None,
    )
