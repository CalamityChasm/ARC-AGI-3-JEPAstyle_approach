"""Stage 6 meta-learning: Reptile (first-order) meta-training for the MoE
predictor, explicitly optimizing for POST-test-time-adaptation performance
on a new game, not just ordinary pre-adaptation accuracy.

## Motivation

CLAUDE.md's Stage 6 addendum documents a long investigation (14 negative
interventions) into why the world model has no zero-shot prediction edge
over identity on ARC-3 games it wasn't trained on, and exactly one positive
result: test-time adaptation (`jepa/test_time_adapter.py: TestTimeAdapter`)
-- letting the model take a few real AdamW steps on a held-out game's own
observed transitions, during play, on a small ANIL-style parameter subset
(each MoE expert's LAST Conv2d + the gate's LAST Linear, ~33.8K params).
That mechanism works, but the checkpoint it starts from was never trained
with adaptability as a goal -- it was trained the ordinary way (minimize
prediction error on the training distribution) and happened to retain
enough structure to be somewhat adaptable. This script builds a checkpoint
that is EXPLICITLY optimized to be quick to adapt, via Reptile.

## Why Reptile, not MAML

Full (second-order) MAML backpropagates through the inner-loop optimization
itself, which is both compute-heavier and a real implementation-correctness
risk (differentiating through an AdamW inner loop is easy to get subtly
wrong). Reptile (Nichol et al. 2018) only needs the INNER-loop endpoint --
no second-order gradients, no differentiating through the adaptation
process -- and has a solid practical track record in few-shot settings.
Given this project's hardware (a single RTX 2070) and the priority on
correctness over squeezing out the last bit of sample efficiency, Reptile
is the defensible default here; nothing encountered while building this
suggested MAML would be worth its extra complexity/risk on this problem.

## Design: ANIL-style split between "body" and "head"

The predictor is split into two groups, mirroring EXACTLY the parameter
subset `TestTimeAdapter` adapts at real eval/play time (imported directly
from `jepa/test_time_adapter.py: get_adapter_params` so the two can never
silently drift apart):

- **Head** (the adapter subset, ~33.8K params): each expert's LAST Conv2d
  + the gate's LAST Linear. Updated ONLY via the Reptile meta-objective
  below -- never touched by the ordinary per-batch joint loss. This is the
  one thing this script exists to change relative to the existing recipe.
- **Body** (everything else -- encoder, action/xy/game embeddings, every
  expert's FIRST Conv2d, the gate's FIRST Linear): trained by ordinary
  joint multi-task gradient descent on i.i.d.-shuffled batches across every
  training game, IDENTICAL to `jepa/train_moe_predictor.py`'s own ARC
  fine-tune recipe (same loss terms, same LR, same EMA target). This keeps
  the shared representation itself trained the normal way -- what makes a
  "trained games" sanity check meaningful/comparable to the existing
  baseline, and avoids the encoder only ever seeing frozen-encoder,
  few-shot inner-loop episodes (which would starve it of ordinary
  supervised signal).

This is why a bare "run Reptile on the whole model" design was NOT used:
it would confound "does the meta-objective help the adaptable subset" with
"did multi-task training on the rest of the model get worse because it's
now only getting sparse few-shot gradient signal instead of dense
supervised batches." Splitting body/head isolates the ONE variable this
experiment is actually about.

## The Reptile update itself

For each of a small batch of sampled "tasks" (one training-pool ARC-3 game
each, per meta-update -- see `--meta-tasks-per-batch`):

1. Snapshot the head params.
2. Run `--inner-steps` real AdamW steps (`--inner-lr`, `--inner-batch-size`)
   on ONLY that task's own transitions (drawn from the TRAIN split of the
   ARC corpus, excluded-games respected the same way as everywhere else in
   this project) -- by default the SAME K=5/STEPS=8/LR=5e-5-derived
   operating point `stage6-test-time-adaptation-agent` validated for real
   play (see experiments/stage6_test_time_adaptation_agent.md), so the
   meta-objective targets exactly the adaptation procedure that will
   actually run at eval/play time.
3. Record the delta (adapted head params - snapshot), then restore the
   head params to the snapshot (the shared model object is reused across
   tasks within one meta-update -- adaptation during meta-TRAINING is
   throwaway per task, only the aggregated delta below is kept).
4. Average the deltas across the sampled task batch, then interpolate the
   real head params toward that average by `epsilon` (optionally linearly
   annealed to 0 over the course of the ARC-finetune phase, the standard
   Reptile schedule): `head += epsilon * mean_delta`.

Meta-tasks are sampled ONLY from the ARC-3 training-pool games' TRAIN
split (never the validation split, never MiniGrid/external/Sokoban, never
the held-out fold's games) -- this keeps the objective specifically
targeted at "adapt quickly to a new ARC-3 game," matching what
`TestTimeAdapter` actually does during real play, rather than diluting it
with a different kind of task (MiniGrid's own action semantics are a
different distribution the real held-out-game problem never sees).

MiniGrid pretraining (if `--pretrain-epochs > 0`) is UNCHANGED from
`train_moe_predictor.py`'s own recipe -- reused directly (joint training of
the WHOLE model, head included) so the meta-learned and baseline
checkpoints share an identical warm start; the ONLY deliberate difference
between this script's output and `train_moe_predictor.py`'s is how the ARC
fine-tune phase updates the head subset.

Usage:
    python -m jepa.train_meta_predictor --pretrain-epochs 20 --epochs 60 \
      --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 \
      --out checkpoints_meta_fold1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, generate_transitions
from .data.sokoban_data import (
    DEFAULT_CONFIGS as SOKOBAN_DEFAULT_CONFIGS,
    generate_transitions as generate_sokoban_transitions,
)
from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .losses import (
    same_color_contrastive_loss,
    variance_regularizer,
    weighted_prediction_loss,
)
from .models import load_balance_loss, update_ema_target
from .test_time_adapter import get_adapter_params
from .train_moe_predictor import (
    CONTRAST_WEIGHT_DEFAULT,
    EMA_MOMENTUM,
    LOAD_BALANCE_WEIGHT,
    VAL_FRACTION,
    build_models,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults chosen to match stage6-test-time-adaptation-agent's own validated
# operating point (experiments/stage6_test_time_adaptation_agent.md) -- the
# meta-objective should target the SAME adaptation procedure that actually
# runs at real eval/play time, not an arbitrary different one.
DEFAULT_INNER_STEPS = 8
DEFAULT_INNER_LR = 5e-5
DEFAULT_INNER_BATCH_SIZE = 16


# --- Data / task-pool setup -------------------------------------------------


def _split_and_build_loaders(
    transitions: list, game_vocab: dict, batch_size: int, device: torch.device
):
    """Same random_split (seed=0) + WeightedRandomSampler recipe as
    train_moe_predictor.py's own `_make_loaders`, but ALSO returns the
    train-split transitions grouped by short game code (for Reptile task
    pools) -- kept as one function (rather than calling `_make_loaders`
    and separately re-deriving the split) so there's no risk of the two
    splits silently drifting apart from two independent random_split calls."""
    import os

    dataset = TransitionDataset(transitions, game_vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    all_weights = dataset.sample_weights()
    train_weights = [all_weights[i] for i in train_ds.indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_weights), replacement=True)

    num_workers_override = os.getenv("JEPA_NUM_WORKERS")
    num_workers = int(num_workers_override) if num_workers_override is not None else (
        4 if device.type == "cuda" else 0
    )
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device.type == "cuda"), persistent_workers=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    game_pools = defaultdict(list)
    for i in train_ds.indices:
        t = transitions[i]
        game_short = t[6].split("-")[0]
        game_pools[game_short].append(t)

    return train_loader, val_loader, dict(game_pools)


# --- Reptile inner loop / outer step ----------------------------------------


def _inner_loop_adapt(
    online, predictor, head_params: list, task_transitions: list, game_vocab: dict,
    device: torch.device, n_steps: int, lr: float, batch_size: int,
) -> list:
    """One Reptile inner loop: n_steps of real AdamW updates on ONLY
    `head_params`, fit to `task_transitions` (one game's own train-split
    transitions). Mirrors scripts/test_time_adaptation.py's own
    `adaptation_step` exactly (same batch-resample-per-step pattern, same
    frozen-encoder feature source, same loss) so meta-training exercises
    the identical mechanics real test-time adaptation will use later.

    Returns the per-param DELTA (adapted - pre-adapt snapshot) and restores
    `head_params` to the snapshot before returning -- the caller decides how
    much of this delta to actually keep (the Reptile outer step)."""
    snapshot = [p.detach().clone() for p in head_params]
    opt = torch.optim.AdamW(head_params, lr=lr)
    fallback_vocab = defaultdict(int, game_vocab)
    bs = min(batch_size, len(task_transitions))
    ds = TransitionDataset(task_transitions, fallback_vocab)

    predictor.train()
    for _ in range(n_steps):
        loader = DataLoader(ds, batch_size=bs, shuffle=True)
        cur, action_id, xy, nxt, patch_mask, game_idx = next(iter(loader))
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
        with torch.no_grad():
            cur_feat = online(cur)
            next_feat = online(nxt)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        loss = weighted_prediction_loss(pred_feat, next_feat, patch_mask)

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
    """meta_weights += epsilon * (adapted_weights - meta_weights) -- the
    Reptile update, applied directly to the live head params using the
    ALREADY-AVERAGED delta across a batch of sampled tasks."""
    with torch.no_grad():
        for p, d in zip(head_params, avg_delta):
            p.add_(epsilon * d)


def _run_meta_updates(
    online, predictor, head_params: list, game_pools: dict, game_vocab: dict, device: torch.device,
    n_updates: int, tasks_per_update: int, inner_steps: int, inner_lr: float, inner_batch_size: int,
    epsilon_start: float, epsilon_end: float, global_update_idx: int, total_updates: int, rng: random.Random,
) -> dict:
    """Runs `n_updates` Reptile outer steps (each averaging deltas across
    `tasks_per_update` sampled games). Returns stats for logging + the
    number of the global schedule this consumed (for epsilon annealing to
    track true progress across the whole ARC-finetune phase, not just one
    epoch)."""
    games = [g for g, pool in game_pools.items() if len(pool) >= 2]
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
            delta = _inner_loop_adapt(
                online, predictor, head_params, game_pools[g], game_vocab, device,
                inner_steps, inner_lr, inner_batch_size,
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


# --- MiniGrid pretrain phase (unchanged from train_moe_predictor.py) -------


def _run_joint_epochs(online, target, predictor, opt, train_loader, val_loader, device, epochs, phase, checkpoint_cb=None, checkpoint_every=0):
    """Ordinary joint training of the WHOLE model (used only for the
    MiniGrid pretrain phase -- identical loss/logging to
    train_moe_predictor.py's `_run_epochs`, duplicated rather than imported
    because that function is private (`_run_epochs`) and this keeps this
    script's phase structure self-contained and easy to read top to
    bottom)."""
    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = total_lb_loss = 0.0
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
        if checkpoint_cb is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_cb(epoch + 1, phase)


# --- Main training entry point ----------------------------------------------


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
    contrast_weight: float = CONTRAST_WEIGHT_DEFAULT,
    exclude_games: list | None = None,
    recording_substrings: list | None = None,
    checkpoint_every: int = 0,
    meta_iters_per_epoch: int = 20,
    meta_tasks_per_batch: int = 4,
    inner_steps: int = DEFAULT_INNER_STEPS,
    inner_lr: float = DEFAULT_INNER_LR,
    inner_batch_size: int = DEFAULT_INNER_BATCH_SIZE,
    meta_epsilon: float = 1.0,
    epsilon_anneal: bool = True,
    meta_seed: int = 0,
) -> None:
    device = get_device()
    print(
        f"training on {device}, {num_experts} experts, Reptile meta-learning "
        f"(inner_steps={inner_steps} inner_lr={inner_lr} inner_batch_size={inner_batch_size} "
        f"meta_epsilon={meta_epsilon} anneal={epsilon_anneal} "
        f"meta_iters_per_epoch={meta_iters_per_epoch} meta_tasks_per_batch={meta_tasks_per_batch})"
    )
    if exclude_games:
        print(f"excluding games from all local/external corpora: {exclude_games}")

    arc_transitions = load_all_transitions(
        REPO_ROOT, name_substrings=recording_substrings, exclude_games=exclude_games
    )
    n_local = len(arc_transitions)
    print(f"loaded {n_local} local ARC-3 transitions")

    n_external = 0
    if external_per_game:
        external = load_external_transitions(REPO_ROOT, max_per_game=external_per_game, exclude_games=exclude_games)
        n_external = len(external)
        if external:
            print(f"loaded {n_external} external ARC-3 transitions (arc-3-logs, capped at {external_per_game}/game)")
            arc_transitions += external
        else:
            print("--external-per-game set but data/arc3_logs.zip is missing -- training on local ARC-3 transitions only")

    minigrid_transitions = []
    sokoban_transitions = []
    if pretrain_epochs > 0:
        minigrid_transitions = generate_transitions(
            env_names=DEFAULT_ENV_NAMES, episodes_per_env=minigrid_episodes_per_env,
            steps_per_episode=minigrid_steps_per_episode,
        )
        print(f"generated {len(minigrid_transitions)} MiniGrid transitions across {len(DEFAULT_ENV_NAMES)} environments")
        if sokoban_episodes_per_config > 0:
            sokoban_transitions = generate_sokoban_transitions(
                configs=SOKOBAN_DEFAULT_CONFIGS, episodes_per_config=sokoban_episodes_per_config,
                steps_per_episode=sokoban_steps_per_episode,
            )
            print(f"generated {len(sokoban_transitions)} Sokoban transitions across {len(SOKOBAN_DEFAULT_CONFIGS)} room configs")
    synthetic_transitions = minigrid_transitions + sokoban_transitions

    synthetic_game_ids = {t[6] for t in synthetic_transitions}
    game_ids = sorted({t[6] for t in arc_transitions} | synthetic_game_ids)
    game_vocab = {g: i for i, g in enumerate(game_ids)}
    print(f"{len(game_vocab)} distinct games in the shared vocab")

    online, target, predictor = build_models(encoder_path, num_games=len(game_vocab), num_experts=num_experts, device=device)
    joint_opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    def _save(tag: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_moe.pt")
        torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "moe_predictor.pt")
        (out_dir / "game_vocab_moe.json").write_text(json.dumps(game_vocab, indent=2))
        (out_dir / "moe_training_meta.json").write_text(
            json.dumps(
                {
                    "algorithm": "reptile",
                    "epochs": epochs,
                    "pretrain_epochs": pretrain_epochs,
                    "n_minigrid_transitions": len(minigrid_transitions),
                    "n_sokoban_transitions": len(sokoban_transitions),
                    "num_experts": num_experts,
                    "top_k": None,
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
                    "meta_iters_per_epoch": meta_iters_per_epoch,
                    "meta_tasks_per_batch": meta_tasks_per_batch,
                    "inner_steps": inner_steps,
                    "inner_lr": inner_lr,
                    "inner_batch_size": inner_batch_size,
                    "meta_epsilon": meta_epsilon,
                    "epsilon_anneal": epsilon_anneal,
                    "meta_seed": meta_seed,
                },
                indent=2,
            )
        )
        print(f"[checkpoint] saved encoder + MoE predictor + game vocab to {out_dir} (tag={tag})")

    def _checkpoint_cb(epoch_1_indexed: int, phase: str) -> None:
        _save(f"{phase}-epoch{epoch_1_indexed}-inprogress")

    # --- Phase 0: MiniGrid pretrain (unchanged joint recipe, whole model). ---
    if pretrain_epochs > 0:
        mg_train_loader, mg_val_loader, _mg_pools = _split_and_build_loaders(
            synthetic_transitions, game_vocab, batch_size, device
        )
        _run_joint_epochs(
            online, target, predictor, joint_opt, mg_train_loader, mg_val_loader, device,
            pretrain_epochs, "minigrid-pretrain", checkpoint_cb=_checkpoint_cb, checkpoint_every=checkpoint_every,
        )
        del mg_train_loader, mg_val_loader
        if checkpoint_every > 0:
            _save("minigrid-pretrain-complete")

    # --- Phase 1: ARC fine-tune, body via joint SGD + head via Reptile. ----
    arc_train_loader, arc_val_loader, game_pools = _split_and_build_loaders(
        arc_transitions, game_vocab, batch_size, device
    )
    print(
        f"ARC training-pool task pools: {len(game_pools)} distinct games, "
        f"sizes {sorted(len(v) for v in game_pools.values())}"
    )

    head_params = get_adapter_params(predictor)
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in list(online.parameters()) + list(predictor.parameters()) if id(p) not in head_ids]
    body_opt = torch.optim.AdamW(body_params, lr=lr)
    print(f"head (Reptile-only) params: {sum(p.numel() for p in head_params)}  body (joint-SGD) params: {sum(p.numel() for p in body_params)}")

    rng = random.Random(meta_seed)
    updates_per_epoch = max(1, meta_iters_per_epoch // meta_tasks_per_batch)
    total_updates = updates_per_epoch * epochs
    global_update_idx = 0

    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_loss = total_lb_loss = total_contrast_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in arc_train_loader:
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

            body_opt.zero_grad()
            loss.backward()
            body_opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            total_lb_loss += lb_loss.item()
            n_batches += 1

        meta_stats = _run_meta_updates(
            online, predictor, head_params, game_pools, game_vocab, device,
            n_updates=updates_per_epoch, tasks_per_update=meta_tasks_per_batch,
            inner_steps=inner_steps, inner_lr=inner_lr, inner_batch_size=inner_batch_size,
            epsilon_start=meta_epsilon, epsilon_end=(0.0 if epsilon_anneal else meta_epsilon),
            global_update_idx=global_update_idx, total_updates=total_updates, rng=rng,
        )
        global_update_idx += meta_stats["n_updates"]

        stats = evaluate(online, predictor, arc_val_loader, device=device)
        print(
            f"[arc-finetune-reptile] epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"lb_loss={total_lb_loss / n_batches:.3f}  "
            + (f"contrast_loss={total_contrast_loss / n_batches:.4f}  " if contrast_weight > 0.0 else "")
            + f"meta_updates={meta_stats['n_updates']}  mean_delta_norm={meta_stats['mean_delta_norm']:.6f}  "
            f"mean_epsilon={meta_stats['mean_epsilon']:.4f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            _checkpoint_cb(epoch + 1, "arc-finetune-reptile")

    _save("final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="ARC-3 fine-tuning (Reptile) epochs.")
    parser.add_argument("--pretrain-epochs", type=int, default=0, help="MiniGrid pretraining epochs (0 = skip).")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_meta")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Body (joint-SGD) learning rate.")
    parser.add_argument("--external-per-game", type=int, default=None)
    parser.add_argument("--sokoban-episodes-per-config", type=int, default=0)
    parser.add_argument("--contrast-weight", type=float, default=CONTRAST_WEIGHT_DEFAULT)
    parser.add_argument(
        "--exclude-games", type=str, default=None,
        help="Comma-separated short game codes to exclude from ALL corpora AND all Reptile task pools.",
    )
    parser.add_argument("--recording-substrings", type=str, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--meta-iters-per-epoch", type=int, default=20,
        help="Reptile task-samples per epoch (grouped into updates of --meta-tasks-per-batch each).",
    )
    parser.add_argument(
        "--meta-tasks-per-batch", type=int, default=4,
        help="Games averaged together per single Reptile outer step.",
    )
    parser.add_argument(
        "--inner-steps", type=int, default=DEFAULT_INNER_STEPS,
        help="AdamW steps per task in the Reptile inner loop (default matches the validated TestTimeAdapter operating point).",
    )
    parser.add_argument("--inner-lr", type=float, default=DEFAULT_INNER_LR)
    parser.add_argument("--inner-batch-size", type=int, default=DEFAULT_INNER_BATCH_SIZE)
    parser.add_argument(
        "--meta-epsilon", type=float, default=1.0,
        help="Reptile outer-step size (interpolation factor toward the averaged adapted weights).",
    )
    parser.add_argument(
        "--no-epsilon-anneal", action="store_true",
        help="Disable the default linear anneal of --meta-epsilon to 0 over the ARC-finetune phase.",
    )
    parser.add_argument("--meta-seed", type=int, default=0, help="RNG seed for Reptile task sampling.")
    args = parser.parse_args()
    train(
        args.epochs,
        args.encoder,
        args.out,
        num_experts=args.num_experts,
        batch_size=args.batch_size,
        lr=args.lr,
        external_per_game=args.external_per_game,
        pretrain_epochs=args.pretrain_epochs,
        sokoban_episodes_per_config=args.sokoban_episodes_per_config,
        contrast_weight=args.contrast_weight,
        exclude_games=args.exclude_games.split(",") if args.exclude_games else None,
        recording_substrings=args.recording_substrings.split(",") if args.recording_substrings else None,
        checkpoint_every=args.checkpoint_every,
        meta_iters_per_epoch=args.meta_iters_per_epoch,
        meta_tasks_per_batch=args.meta_tasks_per_batch,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        inner_batch_size=args.inner_batch_size,
        meta_epsilon=args.meta_epsilon,
        epsilon_anneal=not args.no_epsilon_anneal,
        meta_seed=args.meta_seed,
    )
