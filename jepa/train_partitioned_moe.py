"""Stage 6 experiment (stage6-partitioned-experts): hard-partitioned MoE
experts during ARC-3 finetune, to test whether forcing each expert to
specialize on a fixed subset of training games gives the gate something
concrete to route on -- and whether that routing rule generalizes to a
held-out game's id it's never seen.

Working theory (see experiments/stage6_partitioned_experts.md for the
full writeup): CLAUDE.md's Stage 6 addendum found that (a) raw per-expert
disagreement does NOT collapse on held-out games (InfoGain ratio 0.999),
but (b) the *gated* blend does collapse to ~identity there, and (c) gate
specialization is "a minority behavior, not the norm" even on trained
games (mean entropy ~98-99% of the uniform maximum). The gate has never
had a training signal that ties any expert to a distinguishable
competence region -- every expert sees every game's data via the same
soft, jointly-learned routing, so uniform blending (which nets
genuinely-different-but-uncoordinated expert opinions toward the mean)
is a perfectly good solution to the main task loss. This script tests
whether an explicit routing-supervision signal, with game-id dropout to
discourage a pure id-lookup shortcut, changes that.

Mechanism (ARC finetune phase only -- MiniGrid pretrain is left exactly
as in jepa/train_moe_predictor.py, since MiniGrid's single shared
game_id="minigrid" can't be meaningfully hard-partitioned and changing
pretrain too would confound the one variable this experiment tests):

1. The 20 fold-1 training games (5 held out: r11l, bp35, m0r0, tr87,
   ka59, exactly matching stage6-game-holdout) are deterministically
   round-robin-assigned to 8 fixed expert clusters (game_to_expert_idx).
2. Main task loss uses ONLY the assigned expert's raw prediction
   (`MoEPredictor.predict_all_experts`, gathered per-example) -- not the
   gate-blended combination. Only that expert's parameters (+ the shared
   encoder, since the assigned expert's forward pass still routes
   through it) get gradient from the main task loss for a given example.
3. Since the main loss no longer touches the gate, a separate
   cross-entropy routing-supervision loss (`MoEPredictor.gate_logits` vs
   the known one-hot assignment) trains the gate directly.
4. Game-id dropout (`GAME_ID_DROPOUT_P`, applied only to the routing-loss
   forward pass, not the main-loss one) forces the gate to predict the
   correct expert from pooled visual features alone on a fraction of
   steps, discouraging a pure game_id -> expert lookup-table shortcut
   that would be structurally useless on a held-out game's untrained id.
5. At eval time (jepa/benchmark-style evaluate(), same as
   train_moe_predictor.py), the *normal* learned soft gate is used --
   hard routing is a training-time-only mechanism, matching the real
   deployment path.

Usage (fold-1 recipe, matching stage6-game-holdout's own baseline
command for direct comparability):
    python -m jepa.train_partitioned_moe --pretrain-epochs 20 --epochs 60 \
        --num-experts 8 --external-per-game 2000 \
        --exclude-games r11l,bp35,m0r0,tr87,ka59 \
        --checkpoint-every 5 --out checkpoints_partitioned_experts
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, generate_transitions
from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, MoEPredictor, make_ema_target, update_ema_target

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1

# Routing-supervision cross-entropy loss weight. Chosen the same way
# LOAD_BALANCE_WEIGHT was in train_moe_predictor.py: the main task loss
# lives at ~1e-2 to 1e-4 scale, while an 8-class cross-entropy loss starts
# near ln(8)=2.08 and settles somewhere well above the main loss's scale
# even once the gate has learned something real -- an unscaled aux loss
# would swamp the main signal. 0.01 was picked from a smoke-test run (see
# experiments/stage6_partitioned_experts.md) that printed both losses'
# raw, unweighted magnitudes before committing to the full run.
ROUTING_LOSS_WEIGHT = 0.01
# Fraction of routing-loss forward passes that see game_idx zeroed out
# (the same fallback index a genuinely unseen game gets in production --
# game_vocab.get(game_id, 0)) -- forces the gate to learn at least partly
# from pooled visual features rather than purely memorizing a
# game_id -> expert lookup table, which would be structurally useless on
# a held-out game's untrained embedding index.
GAME_ID_DROPOUT_P = 0.25


def build_models(
    encoder_path: Path | None, num_games: int, num_experts: int, device: torch.device
) -> tuple:
    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = MoEPredictor(num_games=num_games, num_experts=num_experts).to(device)
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
    _num_workers_override = os.getenv("JEPA_NUM_WORKERS")
    if _num_workers_override is not None:
        num_workers = int(_num_workers_override)
    else:
        num_workers = 4 if device.type == "cuda" else 0
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def build_game_to_expert(game_vocab: dict, train_game_ids: list, num_experts: int, device) -> torch.Tensor:
    """Deterministic round-robin assignment of the (sorted) training games
    to expert clusters -- documented, reproducible, not semantically
    motivated per the experiment's own scope. With 20 training games and
    8 experts, round-robin gives experts 0-3 three games each and experts
    4-7 two games each (balanced within +/-1)."""
    sorted_games = sorted(train_game_ids)
    game_to_expert = {g: i % num_experts for i, g in enumerate(sorted_games)}
    lookup = torch.zeros(len(game_vocab), dtype=torch.long)
    for g, idx in game_vocab.items():
        if g in game_to_expert:
            lookup[idx] = game_to_expert[g]
    print("game -> expert assignment (round-robin):")
    by_expert: dict = {}
    for g, e in game_to_expert.items():
        by_expert.setdefault(e, []).append(g)
    for e in range(num_experts):
        print(f"  expert {e}: {sorted(by_expert.get(e, []))}")
    return lookup.to(device)


def _run_pretrain_epochs(online, target, predictor, opt, train_loader, val_loader, device, epochs: int) -> None:
    """Unmodified from train_moe_predictor.py's _run_epochs -- MiniGrid
    pretrain uses the ordinary gate-blended loss + load-balance term,
    deliberately NOT hard-partitioned (see module docstring: MiniGrid's
    single shared game_id can't be meaningfully split this way, and
    changing pretrain too would confound the one variable this experiment
    tests)."""
    from .models import load_balance_loss

    LOAD_BALANCE_WEIGHT = 0.001
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
            f"[minigrid-pretrain] epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"lb_loss={total_lb_loss / n_batches:.3f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )


def _run_partitioned_finetune_epochs(
    online, target, predictor, opt, train_loader, val_loader, device, epochs: int,
    game_to_expert_idx: torch.Tensor, routing_weight: float, game_id_dropout_p: float,
    checkpoint_cb=None, checkpoint_every: int = 0,
) -> None:
    """The hard-routed ARC finetune phase -- see module docstring for the
    full mechanism. Main task loss uses only the assigned expert's raw
    (skip-connected) prediction; a separate cross-entropy routing loss
    (with game-id dropout) trains the gate."""
    num_experts = predictor.num_experts
    for epoch in range(epochs):
        online.train()
        predictor.train()
        total_main_loss = 0.0
        total_routing_loss = 0.0
        total_routing_correct = 0
        total_routing_n = 0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
            b = cur.shape[0]

            cur_feat = online(cur)
            with torch.no_grad():
                target_feat = target(nxt)

            assigned_expert = game_to_expert_idx[game_idx]  # (B,)

            # --- main task loss: hard-routed to the assigned expert only ---
            expert_preds_all = predictor.predict_all_experts(cur_feat, action_id, xy, game_idx)  # (B,K,C,H,W)
            pred_feat = expert_preds_all[torch.arange(b, device=device), assigned_expert]  # (B,C,H,W)
            main_loss = weighted_prediction_loss(pred_feat, target_feat, patch_mask) + variance_regularizer(cur_feat)

            # --- routing-supervision loss: cross-entropy vs known assignment,
            # with game-id dropout on a fraction of examples so the gate can't
            # purely memorize a game_id -> expert lookup table. ---
            game_idx_routing = game_idx.clone()
            dropout_mask = torch.rand(b, device=device) < game_id_dropout_p
            game_idx_routing[dropout_mask] = 0
            gate_logits = predictor.gate_logits(cur_feat, action_id, xy, game_idx_routing)
            routing_loss = F.cross_entropy(gate_logits, assigned_expert)

            loss = main_loss + routing_weight * routing_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_main_loss += main_loss.item()
            total_routing_loss += routing_loss.item()
            total_routing_correct += (gate_logits.argmax(dim=-1) == assigned_expert).sum().item()
            total_routing_n += b
            n_batches += 1

        stats = evaluate(online, predictor, val_loader, device=device)
        routing_acc = total_routing_correct / max(total_routing_n, 1)
        print(
            f"[arc-finetune-partitioned] epoch {epoch + 1}/{epochs}  "
            f"main_loss={total_main_loss / n_batches:.4f}  "
            f"routing_loss={total_routing_loss / n_batches:.4f}  "
            f"routing_acc={routing_acc:.3f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )
        if checkpoint_cb is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_cb(epoch + 1, "arc-finetune-partitioned")


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
    exclude_games: list | None = None,
    routing_weight: float = ROUTING_LOSS_WEIGHT,
    game_id_dropout_p: float = GAME_ID_DROPOUT_P,
    checkpoint_every: int = 0,
    resume_from: Path | None = None,
) -> None:
    device = get_device()
    print(
        f"training on {device}, {num_experts} experts, HARD-PARTITIONED finetune "
        f"(routing_weight={routing_weight}, game_id_dropout_p={game_id_dropout_p})"
    )
    if exclude_games:
        print(f"excluding games from all local/external corpora: {exclude_games}")

    resume_meta = None
    if resume_from is not None:
        resume_meta = json.loads((resume_from / "moe_training_meta.json").read_text())
        print(
            f"RESUMING from {resume_from} (previous checkpoint_tag={resume_meta.get('checkpoint_tag')}) -- "
            f"skipping MiniGrid pretrain (already baked into the warm-started weights), "
            f"loading encoder_moe.pt + moe_predictor.pt + game_vocab_moe.json from there, "
            f"remaining --epochs run with a FRESH optimizer (no optimizer state was checkpointed)."
        )
        pretrain_epochs = 0  # already done, baked into the loaded weights

    arc_transitions = load_all_transitions(REPO_ROOT, exclude_games=exclude_games)
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
            print("--external-per-game set but data/arc3_logs.zip is missing -- local-only training")

    minigrid_transitions = []
    if pretrain_epochs > 0:
        minigrid_transitions = generate_transitions(
            env_names=DEFAULT_ENV_NAMES,
            episodes_per_env=minigrid_episodes_per_env,
            steps_per_episode=minigrid_steps_per_episode,
        )
        print(f"generated {len(minigrid_transitions)} MiniGrid transitions across {len(DEFAULT_ENV_NAMES)} environments")

    synthetic_game_ids = {t[6] for t in minigrid_transitions}
    arc_game_ids = {t[6] for t in arc_transitions}

    if resume_from is not None:
        # Reuse the EXACT vocab the resumed checkpoint's embeddings were
        # trained against -- must not be rebuilt, since embedding row i's
        # meaning is fixed by whatever vocab produced the loaded weights.
        game_vocab = json.loads((resume_from / "game_vocab_moe.json").read_text())
        assert arc_game_ids <= set(game_vocab), (
            f"resume vocab missing games present in this run's corpus: "
            f"{arc_game_ids - set(game_vocab)}"
        )
    else:
        game_ids = sorted(arc_game_ids | synthetic_game_ids)
        game_vocab = {g: i for i, g in enumerate(game_ids)}
    print(f"{len(game_vocab)} distinct games in the shared vocab")

    game_to_expert_idx = build_game_to_expert(game_vocab, sorted(arc_game_ids), num_experts, device)
    if resume_from is not None:
        # Sanity check: the resumed checkpoint's own recorded assignment
        # must match what this run's deterministic round-robin recomputes,
        # or "resuming" would silently retarget mid-training.
        prev_assignment = resume_meta.get("game_to_expert", {})
        for g in arc_game_ids:
            prev_e = prev_assignment.get(g)
            new_e = int(game_to_expert_idx[game_vocab[g]].item())
            assert prev_e == new_e, f"game_to_expert mismatch on resume for {g}: was {prev_e}, now {new_e}"
        print("game_to_expert assignment confirmed identical to the resumed checkpoint's own.")

    if resume_from is not None:
        online = CNNEncoder().to(device)
        online.load_state_dict(torch.load(resume_from / "encoder_moe.pt", map_location=device))
        predictor = MoEPredictor(num_games=len(game_vocab), num_experts=num_experts).to(device)
        predictor.load_state_dict(torch.load(resume_from / "moe_predictor.pt", map_location=device))
        target = make_ema_target(online)
        print(f"warm-started encoder + predictor from {resume_from}")
    else:
        online, target, predictor = build_models(
            encoder_path, num_games=len(game_vocab), num_experts=num_experts, device=device
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
                    "num_experts": num_experts,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": str(device),
                    "n_local_transitions": n_local,
                    "n_external_transitions": n_external,
                    "external_per_game": external_per_game,
                    "n_games": len(game_vocab),
                    "exclude_games": exclude_games,
                    "routing_weight": routing_weight,
                    "game_id_dropout_p": game_id_dropout_p,
                    "game_to_expert": {g: int(game_to_expert_idx[i].item()) for g, i in game_vocab.items() if g in arc_game_ids},
                    "checkpoint_tag": tag,
                    "resumed_from": str(resume_from) if resume_from is not None else None,
                    "resumed_from_tag": resume_meta.get("checkpoint_tag") if resume_meta is not None else None,
                },
                indent=2,
            )
        )
        print(f"[checkpoint] saved encoder + MoE predictor + game vocab to {out_dir} (tag={tag})")

    def _checkpoint_cb(epoch_1_indexed: int, phase: str) -> None:
        _save(f"{phase}-epoch{epoch_1_indexed}-inprogress")

    if pretrain_epochs > 0:
        mg_train_loader, mg_val_loader = _make_loaders(minigrid_transitions, game_vocab, batch_size, device)
        _run_pretrain_epochs(online, target, predictor, opt, mg_train_loader, mg_val_loader, device, pretrain_epochs)
        del mg_train_loader, mg_val_loader
        if checkpoint_every > 0:
            _save("minigrid-pretrain-complete")

    arc_train_loader, arc_val_loader = _make_loaders(arc_transitions, game_vocab, batch_size, device)
    _run_partitioned_finetune_epochs(
        online, target, predictor, opt, arc_train_loader, arc_val_loader, device, epochs,
        game_to_expert_idx, routing_weight, game_id_dropout_p,
        checkpoint_cb=_checkpoint_cb, checkpoint_every=checkpoint_every,
    )

    _save("final")


@torch.no_grad()
def evaluate(online, predictor, loader, device: torch.device | None = None) -> dict:
    """Identical to train_moe_predictor.py's evaluate -- deliberately uses
    the normal learned SOFT gate (predictor.forward), not hard routing.
    Hard routing is a training-time-only mechanism (see module docstring,
    item 5); this is what a real deployment/inference path would see."""
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
    parser.add_argument("--pretrain-epochs", type=int, default=0)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_partitioned_experts")
    parser.add_argument("--external-per-game", type=int, default=None)
    parser.add_argument("--exclude-games", type=str, default=None)
    parser.add_argument("--routing-weight", type=float, default=ROUTING_LOSS_WEIGHT)
    parser.add_argument("--game-id-dropout", type=float, default=GAME_ID_DROPOUT_P)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--resume-from", type=Path, default=None,
        help=(
            "Resume ARC finetune from an existing checkpoint dir's "
            "encoder_moe.pt + moe_predictor.pt + game_vocab_moe.json "
            "(e.g. after an interrupted run). Skips MiniGrid pretrain "
            "(assumed already baked into the loaded weights) and uses a "
            "FRESH optimizer -- no optimizer state is checkpointed. "
            "--epochs is the number of ADDITIONAL finetune epochs to run "
            "from here, not a total."
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
        exclude_games=args.exclude_games.split(",") if args.exclude_games else None,
        routing_weight=args.routing_weight,
        game_id_dropout_p=args.game_id_dropout,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume_from,
    )
