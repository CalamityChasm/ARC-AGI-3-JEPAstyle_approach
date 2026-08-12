"""Reptile (first-order) meta-training for the RECURRENT predictor,
porting `jepa/train_meta_predictor.py`'s design (built for the MoE
predictor) to `RecurrentActionConditionedPredictor`.

## Motivation

`scripts/test_time_adaptation_recurrent.py` showed real test-time
adaptation substantially narrows the recurrent predictor's held-out-game
residual collapse on `bp35`/`ka59` specifically (changed-patches `bp35`:
-9.76% -> -0.51%, `ka59`: -6.51% -> -1.84%, both at n=200 observed
transitions) -- but `experiments/stage6_recurrent_exploration.md`'s
"Follow-up" section found that gain doesn't show up as real level
completions when wired into `RecurrentSearchTTA` (0/16 at n=8x2 games,
identical to the frozen-model 0/32 baseline). The checkpoint TTA starts
from was never trained with adaptability as an explicit goal -- same
situation `train_meta_predictor.py` addressed for the MoE predictor via
Reptile, with a real (if modest) amplification of TTA's own effect
(CLAUDE.md's Stage 6 addendum: standard-dose Reptile was a negative,
high-dose Reptile gave +0.98%/+1.28% vs. the un-meta-trained baseline's
+0.78%/+0.66%). This script tests the same idea for the recurrent
predictor, which has never been meta-trained before.

## Design differences from the MoE version

The MoE predictor trains on i.i.d.-shuffled single transitions; the
recurrent predictor trains on ORDERED per-episode sequence chunks with
truncated BPTT and a hidden state reset per chunk
(`jepa/train_recurrent_predictor.py`'s own recipe, reused unchanged here
for the joint-SGD phase). Consequences for the Reptile machinery:

- **Adapted head params**: `predictor.net[-1]` (the final Conv2d
  producing the residual, ~4.2K params) -- the SAME subset
  `scripts/test_time_adaptation_recurrent.py` already validated at real
  eval/play time. No `get_adapter_params`-style shared helper exists for
  the recurrent predictor (that module is MoE-specific); referenced
  directly here so the meta-objective and the real adaptation mechanism
  can never silently drift apart -- if the real TTA adapter subset ever
  changes, this must change with it.
- **Task pools are per-game EPISODE lists**, not flattened transitions --
  the inner loop needs real temporal continuity to sample a SEQ_LEN chunk
  from (a flattened cross-episode transition list would let a "chunk"
  span two unrelated episodes). Mirrors `scripts/
  test_time_adaptation_recurrent.py`'s own per-file-then-per-chunk
  sampling, just IID-batch sampled here instead of streamed.
- **Inner loop** mirrors `test_time_adaptation_recurrent.py`'s
  `adaptation_step`/`run_chunk` exactly (fresh zero hidden state per
  sampled chunk, `weighted_prediction_loss` per step, mean over the
  chunk) -- the meta-objective targets exactly the adaptation procedure
  real play will run, same principle the MoE version's docstring states.

## The same bug already fixed once, applied here from the start

`train_meta_predictor.py`'s own history found that excluding the head
from ordinary joint SGD (a pure ANIL split, head updated ONLY by Reptile)
causes catastrophic representation collapse -- the encoder learns to make
`cur_feat`/`target_feat` trivially close together instead of learning
real dynamics, since a head frozen near-zero can't produce a meaningful
residual regardless of what the body does. Applying that lesson directly
here rather than re-discovering it: `joint_opt` covers the WHOLE model
(encoder + predictor, head included) throughout, exactly like
`train_recurrent_predictor.py`'s own recipe; the Reptile update is a
periodic ADDITIONAL nudge on top of the head only, not a replacement for
its ordinary training signal.

Usage:
    python -m jepa.train_recurrent_meta_predictor --epochs 30 \
      --exclude-games r11l,bp35,m0r0,tr87,ka59 --out checkpoints_meta_recurrent_fold1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from .data.sequences import EpisodeSequenceDataset, build_game_vocab, load_all_episodes
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, RecurrentActionConditionedPredictor, make_ema_target, update_ema_target
from .train_recurrent_predictor import _run_sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1
SEQ_LEN = 16

# Matches scripts/test_time_adaptation_recurrent.py's validated operating
# point exactly -- the meta-objective must target the same adaptation
# procedure real play will actually run.
DEFAULT_INNER_STEPS = 8
DEFAULT_INNER_LR = 5e-5


def _group_episodes_by_game(episodes: list) -> dict:
    pools = defaultdict(list)
    for ep in episodes:
        game_id = ep[0][6].split("-")[0]
        pools[game_id].append(ep)
    return dict(pools)


def _chunk_loss(online, predictor, chunk: list, game_idx: int, device) -> torch.Tensor:
    """Mirrors scripts/test_time_adaptation_recurrent.py's run_chunk +
    loss exactly: fresh zero hidden state, per-step weighted_prediction_loss,
    mean over the chunk. Encoder features are computed under no_grad (the
    encoder itself is never part of the Reptile inner loop -- only
    predictor.net[-1] is adapted, matching the real TTA mechanism)."""
    import numpy as np

    from .grid import arc3_frame_to_tensor, patch_change_mask

    curs, actions, xys, nxts, masks = [], [], [], [], []
    for frame_t, action_id, x, y, frame_t1, _changed, _gid in chunk:
        curs.append(arc3_frame_to_tensor(frame_t))
        actions.append(action_id)
        xys.append([x / 63.0, y / 63.0])
        nxts.append(arc3_frame_to_tensor(frame_t1))
        masks.append(patch_change_mask(frame_t, frame_t1))
    cur = torch.from_numpy(np.stack(curs)).to(device)
    nxt = torch.from_numpy(np.stack(nxts)).to(device)
    action_t = torch.tensor(actions, dtype=torch.long, device=device)
    xy_t = torch.tensor(xys, dtype=torch.float32, device=device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    game_t = torch.full((len(chunk),), game_idx, dtype=torch.long, device=device)

    hidden = predictor.init_hidden(1, device)
    losses = []
    for step in range(len(chunk)):
        with torch.no_grad():
            cur_feat = online(cur[step : step + 1])
            next_feat = online(nxt[step : step + 1])
        pred_feat, hidden = predictor(
            cur_feat, action_t[step : step + 1], xy_t[step : step + 1], hidden, game_t[step : step + 1]
        )
        losses.append(weighted_prediction_loss(pred_feat, next_feat, mask_t[step].unsqueeze(0)))
    return sum(losses) / len(losses)


def _inner_loop_adapt(
    online, predictor, head_params: list, game_episodes: list, game_idx: int, device,
    n_steps: int, lr: float, seq_len: int, rng: random.Random,
) -> list:
    """One Reptile inner loop for one game-task: n_steps of real AdamW
    updates on ONLY head_params, each step resampling a fresh SEQ_LEN
    chunk from a randomly chosen episode of this game (real temporal
    continuity within a chunk, IID across steps). Returns the per-param
    delta and restores head_params to the pre-adapt snapshot."""
    snapshot = [p.detach().clone() for p in head_params]
    opt = torch.optim.AdamW(head_params, lr=lr)

    predictor.train()
    for _ in range(n_steps):
        ep = rng.choice(game_episodes)
        if len(ep) <= seq_len:
            chunk = ep
        else:
            start = rng.randrange(len(ep) - seq_len + 1)
            chunk = ep[start : start + seq_len]
        loss = _chunk_loss(online, predictor, chunk, game_idx, device)
        opt.zero_grad()
        loss.backward()
        opt.step()
    predictor.eval()

    delta = [p.detach().clone() - s for p, s in zip(head_params, snapshot)]
    with torch.no_grad():
        for p, s in zip(head_params, snapshot):
            p.copy_(s)
    return delta


def _reptile_outer_step(head_params: list, avg_delta: list, epsilon: float) -> None:
    with torch.no_grad():
        for p, d in zip(head_params, avg_delta):
            p.add_(epsilon * d)


def _run_meta_updates(
    online, predictor, head_params: list, game_pools: dict, game_vocab: dict, device,
    n_updates: int, tasks_per_update: int, inner_steps: int, inner_lr: float, seq_len: int,
    epsilon_start: float, epsilon_end: float, global_update_idx: int, total_updates: int, rng: random.Random,
) -> dict:
    games = [g for g, eps in game_pools.items() if eps]
    stats = {"n_updates": 0, "mean_delta_norm": 0.0, "mean_epsilon": 0.0}
    if not games:
        return stats

    for u in range(n_updates):
        step_idx = global_update_idx + u
        frac = step_idx / max(1, total_updates - 1)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * min(1.0, frac)

        sampled = rng.sample(games, min(tasks_per_update, len(games)))
        deltas = []
        for g in sampled:
            game_idx = game_vocab.get(g, 0)
            delta = _inner_loop_adapt(
                online, predictor, head_params, game_pools[g], game_idx, device,
                inner_steps, inner_lr, seq_len, rng,
            )
            deltas.append(delta)
        if not deltas:
            continue

        avg_delta = [
            torch.stack([d[i] for d in deltas], dim=0).mean(dim=0) for i in range(len(head_params))
        ]
        _reptile_outer_step(head_params, avg_delta, epsilon)

        norm = sum(d.norm().item() ** 2 for d in avg_delta) ** 0.5
        stats["n_updates"] += 1
        stats["mean_delta_norm"] += norm
        stats["mean_epsilon"] += epsilon

    if stats["n_updates"] > 0:
        stats["mean_delta_norm"] /= stats["n_updates"]
        stats["mean_epsilon"] /= stats["n_updates"]
    return stats


@torch.no_grad()
def evaluate(online, predictor, loader, device: torch.device) -> dict:
    online.eval()
    predictor.eval()
    totals = {"pred": 0.0, "identity": 0.0, "pred_changed": 0.0, "identity_changed": 0.0}
    n_batches = 0
    n_changed_batches = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        outputs = _run_sequence(online, predictor, cur, action_id, xy, nxt, patch_mask, game_idx, device, target=None)
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


def train(
    epochs: int,
    encoder_path: Path,
    out_dir: Path,
    batch_size: int = 8,
    lr: float = 3e-4,
    seq_len: int = SEQ_LEN,
    exclude_games: list | None = None,
    meta_iters_per_epoch: int = 20,
    meta_tasks_per_batch: int = 4,
    inner_steps: int = DEFAULT_INNER_STEPS,
    inner_lr: float = DEFAULT_INNER_LR,
    meta_epsilon: float = 1.0,
    epsilon_anneal: bool = True,
    meta_seed: int = 0,
) -> None:
    device = get_device()
    print(
        f"training on {device}, Reptile meta-learning for recurrent predictor "
        f"(inner_steps={inner_steps} inner_lr={inner_lr} meta_epsilon={meta_epsilon} "
        f"anneal={epsilon_anneal} meta_iters_per_epoch={meta_iters_per_epoch} "
        f"meta_tasks_per_batch={meta_tasks_per_batch})"
    )
    if exclude_games:
        print(f"excluding games from the episode corpus AND Reptile task pools: {exclude_games}")

    episodes = load_all_episodes(REPO_ROOT, exclude_games=exclude_games)
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
        num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=num_workers > 0
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    game_pools = _group_episodes_by_game(episodes)
    print(f"Reptile task pools: {len(game_pools)} games, episode counts {sorted(len(v) for v in game_pools.values())}")

    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = RecurrentActionConditionedPredictor(num_games=len(game_vocab)).to(device)
    joint_opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    head_params = list(predictor.net[-1].parameters())
    n_body = (
        sum(p.numel() for p in online.parameters())
        + sum(p.numel() for p in predictor.parameters())
        - sum(p.numel() for p in head_params)
    )
    print(f"head (joint-SGD + Reptile-nudged) params: {sum(p.numel() for p in head_params)}  body params: {n_body}")

    rng = random.Random(meta_seed)
    updates_per_epoch = max(1, meta_iters_per_epoch // meta_tasks_per_batch)
    total_updates = updates_per_epoch * epochs
    global_update_idx = 0

    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)

            outputs = _run_sequence(online, predictor, cur, action_id, xy, nxt, patch_mask, game_idx, device, target=target)
            loss = 0.0
            for cur_feat, pred_feat, target_feat, mask in outputs:
                loss = loss + weighted_prediction_loss(pred_feat, target_feat, mask) + variance_regularizer(cur_feat)
            loss = loss / len(outputs)

            joint_opt.zero_grad()
            loss.backward()
            joint_opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            n_batches += 1

        meta_stats = _run_meta_updates(
            online, predictor, head_params, game_pools, game_vocab, device,
            n_updates=updates_per_epoch, tasks_per_update=meta_tasks_per_batch,
            inner_steps=inner_steps, inner_lr=inner_lr, seq_len=seq_len,
            epsilon_start=meta_epsilon, epsilon_end=(0.0 if epsilon_anneal else meta_epsilon),
            global_update_idx=global_update_idx, total_updates=total_updates, rng=rng,
        )
        global_update_idx += meta_stats["n_updates"]

        stats = evaluate(online, predictor, val_loader, device=device)
        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"meta_updates={meta_stats['n_updates']}  mean_delta_norm={meta_stats['mean_delta_norm']:.6f}  "
            f"mean_epsilon={meta_stats['mean_epsilon']:.4f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_recurrent.pt")
    torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "recurrent_predictor.pt")
    (out_dir / "game_vocab_recurrent.json").write_text(json.dumps(game_vocab, indent=2))
    (out_dir / "recurrent_training_meta.json").write_text(
        json.dumps(
            {
                "algorithm": "reptile",
                "exclude_games": exclude_games,
                "epochs": epochs,
                "seq_len": seq_len,
                "meta_iters_per_epoch": meta_iters_per_epoch,
                "meta_tasks_per_batch": meta_tasks_per_batch,
                "inner_steps": inner_steps,
                "inner_lr": inner_lr,
                "meta_epsilon": meta_epsilon,
                "epsilon_anneal": epsilon_anneal,
                "meta_seed": meta_seed,
            },
            indent=2,
        )
    )
    print(f"saved encoder + recurrent predictor + game vocab to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_meta_recurrent")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument(
        "--exclude-games", type=str, default=None,
        help="Comma-separated short game codes to exclude from the episode corpus AND Reptile task pools.",
    )
    parser.add_argument("--meta-iters-per-epoch", type=int, default=20)
    parser.add_argument("--meta-tasks-per-batch", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    parser.add_argument("--inner-lr", type=float, default=DEFAULT_INNER_LR)
    parser.add_argument("--meta-epsilon", type=float, default=1.0)
    parser.add_argument("--no-epsilon-anneal", action="store_true")
    parser.add_argument("--meta-seed", type=int, default=0)
    args = parser.parse_args()
    train(
        args.epochs,
        args.encoder,
        args.out,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        exclude_games=args.exclude_games.split(",") if args.exclude_games else None,
        meta_iters_per_epoch=args.meta_iters_per_epoch,
        meta_tasks_per_batch=args.meta_tasks_per_batch,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        meta_epsilon=args.meta_epsilon,
        epsilon_anneal=not args.no_epsilon_anneal,
        meta_seed=args.meta_seed,
    )
