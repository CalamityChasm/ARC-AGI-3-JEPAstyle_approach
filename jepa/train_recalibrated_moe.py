"""Stage 6 follow-up (stage6-partitioned-experts): does removing the
routing-supervision crutch and letting the gate learn from its own real
objective -- after specialization is already established -- produce a
routing rule that transfers better to a genuinely unseen game?

`train_partitioned_moe.py`'s hard-partitioning run showed two things at
once: (a) forcing per-game expert specialization works (split-half
agreement 71.9% -> 95.5%, all 8 experts win something instead of 2-3
dominating), but (b) it only ever taught the gate to mimic a fixed
game_id -> expert lookup table via an artificial cross-entropy loss, and
that lookup is structurally useless on a held-out game's untrained id --
the gate ended up 100% deterministic on BOTH trained and held-out games,
confidently right on the former and confidently arbitrary on the latter.

This script tests a genuinely different three-phase curriculum (user-
proposed) instead of iterating further on the routing-supervision weight:

1. SPECIALIZE (hard-routed, unchanged mechanism from train_partitioned_
   moe.py) on a 15-game pool -- forces real per-expert competence.
2. RECALIBRATE: unfreeze everything, restore the normal SOFT gated
   forward() pass, drop the routing-supervision loss and hard-routing
   entirely -- train on the real task loss ALONE (plus the usual variance
   regularizer; no load-balance loss either, to see what the gate does
   under zero explicit routing pressure once it has genuinely-specialized
   experts to route between). Critically, this phase trains on a
   *different* 5-game pool than phase 1 specialized on, so the gate can't
   just re-derive the same game_id->expert lookup it would get from
   memorizing phase-1's own games -- it has to learn whatever routing
   rule the real loss rewards on games it's seeing for the first time
   post-specialization.
3. EVALUATE on the fold-1 held-out 5 games (r11l, bp35, m0r0, tr87,
   ka59) -- untouched by either phase, exactly as in every other stage6-*
   held-out-generalization experiment.

Fixed fold-1-compatible 20/5 split (deterministic, sorted, no semantic
motivation -- same convention as build_game_to_expert's own round-robin):
sorted list of the 20 non-held-out local games, first 15 -> SPECIALIZE_
GAMES, last 5 -> CALIBRATE_GAMES.

**v2 (this version): phase 2 got real structure back, targeting the two
clearest collapse signatures the unregularized v1 attempt showed
(see experiments/stage6_partitioned_experts.md's "Follow-up" section):**
1. **Encoder frozen during recalibration** (ANIL-style, matching
   `jepa/test_time_adapter.py`'s own restricted-parameter-subset
   philosophy) -- v1's `val_identity_mse` collapsed 16x over 30 epochs,
   the signature of the encoder itself drifting/overfitting on a narrow
   5-game pool. Freezing it structurally prevents that regardless of what
   the predictor/gate do.
2. **A light `load_balance_loss` (weight 0.001, same value used in the
   MiniGrid pretrain phase) stays active during recalibration** -- v1's
   gate went to 0.00% entropy (fully deterministic) on both trained and
   held-out games, exactly the failure mode this loss is designed to
   penalize (`K * sum_i f_i * P_i`, minimized at uniform usage, maximized
   at total collapse). It was previously reasoned that keeping this off
   would reveal a "pure" real-objective routing signal -- instead it just
   removed the only thing stopping the gate from collapsing.

Deliberately NOT reintroduced: the routing-supervision cross-entropy loss
(that's the game-id-lookup mechanism phase 2 exists specifically to move
away from) and hard-routing (phase 2's whole point is a soft, gate-driven
blend). An explicit diversity loss on raw per-expert outputs (targeting
v1's ~100x InfoGain collapse) is deliberately deferred -- try (1) and (2)
first, since they're both already-proven mechanisms, before adding a
third, novel one.

Usage:
    python -m jepa.train_recalibrated_moe --pretrain-epochs 20 \
        --specialize-epochs 60 --recalibrate-epochs 30 \
        --external-per-game 2000 --checkpoint-every 5 \
        --out checkpoints_recalibrated_experts_v2
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from .data.external_logs import load_external_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, generate_transitions
from .data.trajectories import TransitionDataset, load_all_transitions
from .device import get_device
from .losses import prediction_loss, variance_regularizer, weighted_prediction_loss
from .models import CNNEncoder, MoEPredictor, load_balance_loss, make_ema_target, update_ema_target
from .train_partitioned_moe import (
    _make_loaders,
    _run_partitioned_finetune_epochs,
    _run_pretrain_epochs,
    build_game_to_expert,
    evaluate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996

HELD_OUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
# Fixed 15/5 split of the 20 non-held-out fold-1 games (sorted, first 15 /
# last 5 -- deterministic, not semantically chosen, matching this
# project's established convention for arbitrary-but-reproducible splits).
_ALL_20_SORTED = [
    "ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "lf52", "lp85", "ls20", "re86",
    "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tu93", "vc33", "wa30",
]
SPECIALIZE_GAMES = _ALL_20_SORTED[:15]
CALIBRATE_GAMES = _ALL_20_SORTED[15:]


def _run_recalibration_epochs(
    online, target, predictor, opt, train_loader, val_loader, device, epochs: int,
    load_balance_weight: float = 0.001,
    checkpoint_cb=None, checkpoint_every: int = 0,
) -> None:
    """Phase 2 (v2): normal SOFT gated forward(), real task loss -- but
    now with two structural fixes v1 lacked, both targeting a specific
    diagnosed v1 collapse signature (see module docstring):

    1. Encoder frozen (online(cur) wrapped in no_grad(), online.eval(),
       and `opt` is constructed over predictor.parameters() only by the
       caller) -- prevents the representation collapse v1's
       val_identity_mse showed (16x drop over 30 epochs).
    2. A light load_balance_loss stays active -- prevents the gate
       collapsing to full determinism (0.00% entropy) v1 showed on both
       trained and held-out games.

    Still deliberately absent: routing-supervision loss, hard-routing.
    The point remains letting the gate learn routing from the real task
    objective, not a lookup table -- just without the two failure modes
    that made v1's version of "no structure" collapse instead of learn."""
    online.eval()  # frozen for the whole phase -- never re-toggled to train()
    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        total_lb_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in train_loader:
            cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
            nxt, patch_mask, game_idx = nxt.to(device), patch_mask.to(device), game_idx.to(device)
            with torch.no_grad():
                cur_feat = online(cur)
                target_feat = target(nxt)
            pred_feat, gate_weights = predictor(cur_feat, action_id, xy, game_idx)

            lb_loss = load_balance_loss(gate_weights)
            loss = (
                weighted_prediction_loss(pred_feat, target_feat, patch_mask)
                + variance_regularizer(cur_feat)
                + load_balance_weight * lb_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            # No EMA/encoder update -- online is frozen this phase, so
            # target has nothing new to track toward.

            total_loss += loss.item()
            total_lb_loss += lb_loss.item()
            n_batches += 1

        stats = evaluate(online, predictor, val_loader, device=device)
        print(
            f"[recalibrate-v2] epoch {epoch + 1}/{epochs}  train_loss={total_loss / n_batches:.4f}  "
            f"lb_loss={total_lb_loss / n_batches:.3f}  "
            f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
            f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}"
        )
        if checkpoint_cb is not None and checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            checkpoint_cb(epoch + 1, "recalibrate")


def train(
    encoder_path: Path,
    out_dir: Path,
    pretrain_epochs: int,
    specialize_epochs: int,
    recalibrate_epochs: int,
    num_experts: int = 8,
    batch_size: int = 32,
    lr: float = 3e-4,
    external_per_game: int | None = None,
    minigrid_episodes_per_env: int = 40,
    minigrid_steps_per_episode: int = 80,
    routing_weight: float = 0.01,
    game_id_dropout_p: float = 0.25,
    recalibrate_load_balance_weight: float = 0.001,
    checkpoint_every: int = 0,
) -> None:
    device = get_device()
    print(f"training on {device}: 3-phase specialize -> recalibrate curriculum")
    print(f"SPECIALIZE_GAMES ({len(SPECIALIZE_GAMES)}): {SPECIALIZE_GAMES}")
    print(f"CALIBRATE_GAMES ({len(CALIBRATE_GAMES)}): {CALIBRATE_GAMES}")
    print(f"HELD_OUT_GAMES ({len(HELD_OUT_GAMES)}, untouched by either phase): {HELD_OUT_GAMES}")

    # Build ONE game_vocab up front from the union of both training pools
    # (never the held-out games) so embedding indices stay meaningful and
    # checkpoint-compatible across both phases.
    all_arc_transitions = load_all_transitions(REPO_ROOT, exclude_games=HELD_OUT_GAMES)
    minigrid_transitions = generate_transitions(
        env_names=DEFAULT_ENV_NAMES,
        episodes_per_env=minigrid_episodes_per_env,
        steps_per_episode=minigrid_steps_per_episode,
    )
    print(f"generated {len(minigrid_transitions)} MiniGrid transitions across {len(DEFAULT_ENV_NAMES)} environments")
    game_ids = sorted({t[6] for t in all_arc_transitions} | {t[6] for t in minigrid_transitions})
    game_vocab = {g: i for i, g in enumerate(game_ids)}
    print(f"{len(game_vocab)} distinct games in the shared vocab (fixed across all phases)")

    specialize_transitions = load_all_transitions(REPO_ROOT, exclude_games=CALIBRATE_GAMES + HELD_OUT_GAMES)
    calibrate_transitions = load_all_transitions(REPO_ROOT, exclude_games=SPECIALIZE_GAMES + HELD_OUT_GAMES)
    if external_per_game:
        ext_specialize = load_external_transitions(REPO_ROOT, max_per_game=external_per_game, exclude_games=CALIBRATE_GAMES + HELD_OUT_GAMES)
        ext_calibrate = load_external_transitions(REPO_ROOT, max_per_game=external_per_game, exclude_games=SPECIALIZE_GAMES + HELD_OUT_GAMES)
        specialize_transitions += ext_specialize
        calibrate_transitions += ext_calibrate
    print(f"phase 1 (specialize) transitions: {len(specialize_transitions)} across {len(SPECIALIZE_GAMES)} games")
    print(f"phase 2 (recalibrate) transitions: {len(calibrate_transitions)} across {len(CALIBRATE_GAMES)} games")

    game_to_expert_idx = build_game_to_expert(game_vocab, SPECIALIZE_GAMES, num_experts, device)

    online = CNNEncoder().to(device)
    if encoder_path and encoder_path.exists():
        online.load_state_dict(torch.load(encoder_path, map_location=device))
        print(f"warm-started encoder from {encoder_path}")
    target = make_ema_target(online)
    predictor = MoEPredictor(num_games=len(game_vocab), num_experts=num_experts).to(device)
    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=lr)

    def _save(tag: str, phase: str) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in online.state_dict().items()}, out_dir / "encoder_moe.pt")
        torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, out_dir / "moe_predictor.pt")
        (out_dir / "game_vocab_moe.json").write_text(json.dumps(game_vocab, indent=2))
        (out_dir / "moe_training_meta.json").write_text(json.dumps({
            "num_experts": num_experts, "batch_size": batch_size, "lr": lr, "device": str(device),
            "n_games": len(game_vocab), "phase": phase, "checkpoint_tag": tag,
            "specialize_games": SPECIALIZE_GAMES, "calibrate_games": CALIBRATE_GAMES,
            "held_out_games": HELD_OUT_GAMES,
            "game_to_expert": {g: int(game_to_expert_idx[i].item()) for g, i in game_vocab.items() if g in SPECIALIZE_GAMES},
        }, indent=2))
        print(f"[checkpoint] saved to {out_dir} (phase={phase}, tag={tag})")

    # --- Phase 0: MiniGrid pretrain (unchanged mechanism) ---
    mg_train_loader, mg_val_loader = _make_loaders(minigrid_transitions, game_vocab, batch_size, device)
    _run_pretrain_epochs(online, target, predictor, opt, mg_train_loader, mg_val_loader, device, pretrain_epochs)
    del mg_train_loader, mg_val_loader
    _save("minigrid-pretrain-complete", "pretrain")

    # --- Phase 1: hard-partitioned specialization on SPECIALIZE_GAMES only ---
    spec_train_loader, spec_val_loader = _make_loaders(specialize_transitions, game_vocab, batch_size, device)
    _run_partitioned_finetune_epochs(
        online, target, predictor, opt, spec_train_loader, spec_val_loader, device, specialize_epochs,
        game_to_expert_idx, routing_weight, game_id_dropout_p,
        checkpoint_cb=lambda e, phase: _save(f"{phase}-epoch{e}-inprogress", "specialize"),
        checkpoint_every=checkpoint_every,
    )
    del spec_train_loader, spec_val_loader
    _save("specialize-complete", "specialize")

    # --- Phase 2 (v2): soft end-to-end recalibration on CALIBRATE_GAMES
    # only, encoder frozen -- fresh optimizer over predictor.parameters()
    # ONLY (online is frozen this phase; including it would let AdamW's
    # moment estimates accumulate for a parameter set that never
    # receives gradient, which is harmless but wasteful -- excluding it
    # is also the clearest signal in the code itself that the encoder is
    # deliberately not part of this phase). ---
    opt = torch.optim.AdamW(predictor.parameters(), lr=lr)
    cal_train_loader, cal_val_loader = _make_loaders(calibrate_transitions, game_vocab, batch_size, device)
    _run_recalibration_epochs(
        online, target, predictor, opt, cal_train_loader, cal_val_loader, device, recalibrate_epochs,
        load_balance_weight=recalibrate_load_balance_weight,
        checkpoint_cb=lambda e, phase: _save(f"{phase}-epoch{e}-inprogress", "recalibrate"),
        checkpoint_every=checkpoint_every,
    )
    _save("final", "recalibrate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--specialize-epochs", type=int, default=60)
    parser.add_argument("--recalibrate-epochs", type=int, default=30)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--encoder", type=Path, default=REPO_ROOT / "checkpoints" / "encoder.pt")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_recalibrated_experts_v2")
    parser.add_argument("--external-per-game", type=int, default=None)
    parser.add_argument("--routing-weight", type=float, default=0.01)
    parser.add_argument("--game-id-dropout", type=float, default=0.25)
    parser.add_argument("--recalibrate-load-balance-weight", type=float, default=0.001)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    args = parser.parse_args()
    train(
        args.encoder, args.out,
        pretrain_epochs=args.pretrain_epochs,
        specialize_epochs=args.specialize_epochs,
        recalibrate_epochs=args.recalibrate_epochs,
        num_experts=args.num_experts,
        external_per_game=args.external_per_game,
        routing_weight=args.routing_weight,
        game_id_dropout_p=args.game_id_dropout,
        recalibrate_load_balance_weight=args.recalibrate_load_balance_weight,
        checkpoint_every=args.checkpoint_every,
    )
