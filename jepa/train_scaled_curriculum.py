"""Stage 6 scaled-architecture prep: a genuine capacity increase
(`ScaledCNNEncoder` + `ScaledMoEPredictor`, see jepa/models/encoder_scaled.py
and jepa/models/moe_predictor_scaled.py) trained with a *continuous*
curriculum -- a smoothly-shifting sampling mixture from synthetic-heavy at
the start of training toward real-ARC-3-heavy at the end, rather than the
existing hard two-phase pretrain-then-finetune split in
`jepa/train_moe_predictor.py`.

Design rationale (see experiments/stage6_scaled_architecture_prep.md for the
full writeup):
- All non-ARC-3 sources (MiniGrid, Sokoban, MinAtar, OpenSpiel, the
  hand-rolled ARC-synthetic puzzles) are pooled into one "synthetic"
  category. Real ARC-3 data (local recordings, optionally external
  arc-3-logs, optionally a future search-harvested "winning round" corpus)
  is the "real" category.
- Within a category, each *source* (not each transition) gets equal
  sampling pull -- a source's per-example base weight is `1 / count_in_that
  _source`, so e.g. OpenSpiel's ~2M transitions don't drown out MiniGrid's
  ~67K within the synthetic pool just because there happen to be more of
  them.
- Across categories, `synthetic_frac(epoch)` interpolates linearly (or via
  `--curriculum-schedule cosine`) from `--synthetic-start` (default 0.97)
  down to `--synthetic-end` (default 0.20) over the run. Each epoch draws a
  fresh `WeightedRandomSampler`-equivalent index set from the *current*
  epoch's mixture -- a genuinely continuous shift, not a single switch.
- `--curriculum-schedule flat` reproduces a fixed-mixture bakeline (no
  shift) for comparison; `--pretrain-epochs`/`--epochs`-style hard
  two-phase behavior is *not* reproduced here on purpose (that's what
  jepa/train_moe_predictor.py already does) -- this script is specifically
  the continuous alternative.

Checkpointing: `--checkpoint-every N` saves encoder/predictor/optimizer
state + a `curriculum_meta.json` (completed epoch count, args) every N
epochs; `--resume-from <dir>` picks the run back up from there. Mirrors
`jepa/train_moe_predictor.py`'s own `--checkpoint-every`/`--resume-from`
pattern (built this session on a sibling branch), needed because this
environment's background-task supervisor kills long-running commands
somewhere north of ~45 minutes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.arc_synthetic_data import generate_transitions as generate_arc_synthetic_transitions
from .data.external_logs import load_external_transitions
from .data.minatar_data import DEFAULT_GAMES as MINATAR_DEFAULT_GAMES, generate_transitions as generate_minatar_transitions
from .data.minigrid_data import DEFAULT_ENV_NAMES, generate_transitions as generate_minigrid_transitions
from .data.openspiel_data import GAME_IDS as OPENSPIEL_GAME_IDS, generate_transitions as generate_openspiel_transitions
from .data.sokoban_data import DEFAULT_CONFIGS as SOKOBAN_DEFAULT_CONFIGS, generate_transitions as generate_sokoban_transitions
from .data.trajectories import TransitionDataset, load_all_transitions, load_transitions_from_dir
from .device import get_device
from .losses import per_region_error, prediction_loss, variance_regularizer, weighted_prediction_loss
from .models.encoder_scaled import ScaledCNNEncoder, count_params as count_encoder_params, make_ema_target, update_ema_target
from .models.moe_predictor_scaled import ScaledMoEPredictor, count_params as count_predictor_params
from .models.moe_predictor import load_balance_loss

REPO_ROOT = Path(__file__).resolve().parent.parent
EMA_MOMENTUM = 0.996
VAL_FRACTION = 0.1
LOAD_BALANCE_WEIGHT_DEFAULT = 0.001


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def assemble_sources(args) -> dict:
    """Returns {source_name: (transitions, category)} where category is
    "synthetic" or "real". Every source uses the shared (frame_t, action_id,
    x, y, frame_t1, changed, game_id) tuple schema."""
    sources: dict = {}

    if args.minigrid_episodes_per_env > 0:
        t0 = time.time()
        tr = generate_minigrid_transitions(
            env_names=DEFAULT_ENV_NAMES,
            episodes_per_env=args.minigrid_episodes_per_env,
            steps_per_episode=args.minigrid_steps_per_episode,
        )
        sources["minigrid"] = (tr, "synthetic")
        print(f"  minigrid: {len(tr)} transitions ({time.time() - t0:.1f}s)")

    if args.sokoban_episodes_per_config > 0:
        t0 = time.time()
        tr = generate_sokoban_transitions(
            configs=SOKOBAN_DEFAULT_CONFIGS,
            episodes_per_config=args.sokoban_episodes_per_config,
            steps_per_episode=args.sokoban_steps_per_episode,
        )
        sources["sokoban"] = (tr, "synthetic")
        print(f"  sokoban: {len(tr)} transitions ({time.time() - t0:.1f}s)")

    if args.minatar_episodes_per_game > 0:
        t0 = time.time()
        tr = generate_minatar_transitions(
            games=MINATAR_DEFAULT_GAMES,
            episodes_per_game=args.minatar_episodes_per_game,
            steps_per_episode=args.minatar_steps_per_episode,
            per_game_ids=True,  # the fixed default -- see jepa/data/minatar_data.py's own history
        )
        sources["minatar"] = (tr, "synthetic")
        print(f"  minatar: {len(tr)} transitions ({time.time() - t0:.1f}s)")

    if args.openspiel_episodes_per_game > 0:
        t0 = time.time()
        tr = []
        for game_name in OPENSPIEL_GAME_IDS:
            tr += generate_openspiel_transitions(
                game_name=game_name,
                num_episodes=args.openspiel_episodes_per_game,
                steps_per_episode=args.openspiel_steps_per_episode,
            )
        sources["openspiel"] = (tr, "synthetic")
        print(f"  openspiel: {len(tr)} transitions across {len(OPENSPIEL_GAME_IDS)} games ({time.time() - t0:.1f}s)")

    if args.arc_synthetic_episodes_per_type > 0:
        t0 = time.time()
        tr = generate_arc_synthetic_transitions(
            episodes_per_type=args.arc_synthetic_episodes_per_type,
            steps_per_episode=args.arc_synthetic_steps_per_episode,
        )
        sources["arc_synthetic"] = (tr, "synthetic")
        print(f"  arc_synthetic: {len(tr)} transitions ({time.time() - t0:.1f}s)")

    # -- real ARC-3 data --
    t0 = time.time()
    local = load_all_transitions(REPO_ROOT)
    sources["arc3_local"] = (local, "real")
    print(f"  arc3_local: {len(local)} transitions ({time.time() - t0:.1f}s)")

    if args.external_per_game:
        external = load_external_transitions(REPO_ROOT, max_per_game=args.external_per_game)
        if external:
            sources["arc3_external"] = (external, "real")
            print(f"  arc3_external: {len(external)} transitions")
        else:
            print("  --external-per-game set but data/arc3_logs.zip is missing -- skipping")

    if args.search_harvest_dir is not None:
        # Documented integration point for a sibling session's search-
        # harvested "winning round" ARC-3 corpus, not yet available in this
        # worktree at time of writing -- see experiments/
        # stage6_scaled_architecture_prep.md. Any directory of
        # *.recording.jsonl files in the same format as
        # ARC-AGI-3-Agents/recordings/ works here unchanged, no rework
        # needed once that corpus exists.
        harvest_dir = Path(args.search_harvest_dir)
        if harvest_dir.exists():
            harvested = load_transitions_from_dir(harvest_dir)
            if harvested:
                sources["arc3_search_harvest"] = (harvested, "real")
                print(f"  arc3_search_harvest: {len(harvested)} transitions from {harvest_dir}")
        else:
            print(f"  --search-harvest-dir {harvest_dir} does not exist -- skipping (integration point only)")

    return sources


def build_game_vocab(sources: dict) -> dict:
    game_ids = set()
    for transitions, _cat in sources.values():
        game_ids.update(t[6] for t in transitions)
    return {g: i for i, g in enumerate(sorted(game_ids))}


# ---------------------------------------------------------------------------
# Curriculum sampler
# ---------------------------------------------------------------------------


class Curriculum:
    """Precomputes per-example base weights (equal pull per *source* within
    a category) once, then hands out a fresh set of sampled indices for
    whatever category mixture a given epoch calls for."""

    def __init__(self, sources: dict, val_holdout_ids: set):
        self.all_transitions: list = []
        self.category = []  # 0 = synthetic, 1 = real
        self.base_weight = []
        for _name, (transitions, cat) in sources.items():
            train_transitions = [t for i, t in enumerate(transitions) if (_name, i) not in val_holdout_ids]
            if not train_transitions:
                continue
            w = 1.0 / len(train_transitions)
            cat_id = 0 if cat == "synthetic" else 1
            for t in train_transitions:
                self.all_transitions.append(t)
                self.category.append(cat_id)
                self.base_weight.append(w)
        self.category = np.array(self.category, dtype=np.int64)
        self.base_weight = np.array(self.base_weight, dtype=np.float64)
        # Normalize base_weight so each category's weights sum to 1 on its own
        # -- multiplying by synthetic_frac/real_frac afterward then gives a
        # clean probability distribution over the whole pool.
        for cat_id in (0, 1):
            mask = self.category == cat_id
            total = self.base_weight[mask].sum()
            if total > 0:
                self.base_weight[mask] /= total

    def weights_for(self, synthetic_frac: float) -> np.ndarray:
        frac = np.where(self.category == 0, synthetic_frac, 1.0 - synthetic_frac)
        return self.base_weight * frac

    def sample_indices(self, synthetic_frac: float, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        weights = self.weights_for(synthetic_frac)
        weights = weights / weights.sum()
        return rng.choice(len(self.all_transitions), size=num_samples, replace=True, p=weights)

    def realized_mix(self, indices: np.ndarray) -> dict:
        cats = self.category[indices]
        n = len(indices)
        return {
            "synthetic_frac_realized": float((cats == 0).sum()) / n,
            "real_frac_realized": float((cats == 1).sum()) / n,
        }


def synthetic_frac_schedule(epoch_1indexed: int, total_epochs: int, start: float, end: float, schedule: str) -> float:
    if total_epochs <= 1:
        return start
    t = (epoch_1indexed - 1) / (total_epochs - 1)  # 0 at epoch 1, 1 at final epoch
    if schedule == "flat":
        return start
    if schedule == "cosine":
        import math

        t = (1 - math.cos(math.pi * t)) / 2  # eases in/out instead of linear
    # "linear" (default) falls through unchanged
    return start + (end - start) * t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_models(
    num_games: int,
    feature_channels: int,
    width_mult: float,
    blocks_per_stage: int,
    num_experts: int,
    expert_hidden: int,
    expert_depth: int,
    top_k: int | None,
    device: torch.device,
) -> tuple:
    online = ScaledCNNEncoder(
        out_channels=feature_channels, width_mult=width_mult, blocks_per_stage=blocks_per_stage
    ).to(device)
    target = make_ema_target(online)
    predictor = ScaledMoEPredictor(
        feature_channels=online.out_channels,
        num_games=num_games,
        num_experts=num_experts,
        expert_hidden=int(round(expert_hidden * width_mult)),
        expert_depth=expert_depth,
        top_k=top_k,
    ).to(device)
    return online, target, predictor


def _make_val_loader(sources: dict, val_indices: dict, game_vocab: dict, batch_size: int, device: torch.device):
    val_transitions = []
    for name, (transitions, _cat) in sources.items():
        for i in val_indices.get(name, []):
            val_transitions.append(transitions[i])
    dataset = TransitionDataset(val_transitions, game_vocab)
    num_workers = 2 if device.type == "cuda" else 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda"))


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
        cur_feat = online(cur)
        pred_feat, _gw = predictor(cur_feat, action_id, xy, game_idx)
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


def train(args) -> None:
    device = get_device()
    print(f"training on {device}")

    print("assembling data sources...")
    sources = assemble_sources(args)
    game_vocab = build_game_vocab(sources)
    print(f"{len(game_vocab)} distinct games in the shared vocab")
    total_transitions = sum(len(t) for t, _cat in sources.values())
    print(f"total pooled transitions across all sources: {total_transitions}")

    # Held-out validation: a VAL_FRACTION slice of *each* source, held out of
    # the curriculum sampler entirely (mirrors every other script's honest
    # eval convention in this project).
    rng = np.random.default_rng(0)
    val_indices: dict = {}
    val_holdout_ids = set()
    for name, (transitions, _cat) in sources.items():
        n_val = max(1, int(len(transitions) * VAL_FRACTION)) if len(transitions) > 1 else 0
        idx = rng.choice(len(transitions), size=n_val, replace=False) if n_val else np.array([], dtype=np.int64)
        val_indices[name] = idx.tolist()
        val_holdout_ids.update((name, int(i)) for i in idx)

    curriculum = Curriculum(sources, val_holdout_ids)
    print(f"curriculum pool (train, val held out): {len(curriculum.all_transitions)} transitions")

    val_loader = _make_val_loader(sources, val_indices, game_vocab, args.batch_size, device)
    arc3_local_val_loader = None
    if "arc3_local" in sources:
        # A second, narrower val loader restricted to arc3_local only -- the
        # metric every prior stage's milestone was actually judged against,
        # kept separate from the pooled-across-all-sources val_loader above
        # (which includes synthetic-source validation examples too and
        # would otherwise dilute the number that matters for comparison
        # against production/prior checkpoints).
        arc3_local_transitions, _ = sources["arc3_local"]
        arc3_val = [arc3_local_transitions[i] for i in val_indices.get("arc3_local", [])]
        arc3_local_val_loader = DataLoader(
            TransitionDataset(arc3_val, game_vocab), batch_size=args.batch_size, shuffle=False
        )

    online, target, predictor = build_models(
        num_games=len(game_vocab),
        feature_channels=args.feature_channels,
        width_mult=args.width_mult,
        blocks_per_stage=args.blocks_per_stage,
        num_experts=args.num_experts,
        expert_hidden=args.expert_hidden,
        expert_depth=args.expert_depth,
        top_k=args.top_k,
        device=device,
    )
    n_enc = count_encoder_params(online)
    n_pred = count_predictor_params(predictor)
    print(f"encoder params: {n_enc:,}  predictor params: {n_pred:,}  total: {n_enc + n_pred:,}")

    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=args.lr)

    start_epoch = 0
    if args.resume_from is not None:
        meta = json.loads((args.resume_from / "curriculum_meta.json").read_text())
        start_epoch = meta["completed_epochs"]
        online.load_state_dict(torch.load(args.resume_from / "encoder_scaled.pt", map_location=device))
        predictor.load_state_dict(torch.load(args.resume_from / "moe_predictor_scaled.pt", map_location=device))
        opt.load_state_dict(torch.load(args.resume_from / "optimizer.pt", map_location=device))
        target = make_ema_target(online)
        print(f"resumed from {args.resume_from}: completed_epochs={start_epoch}, continuing to {args.epochs}")

    args.out.mkdir(parents=True, exist_ok=True)

    def _save(completed_epochs: int) -> None:
        torch.save({k: v.cpu() for k, v in online.state_dict().items()}, args.out / "encoder_scaled.pt")
        torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, args.out / "moe_predictor_scaled.pt")
        torch.save(opt.state_dict(), args.out / "optimizer.pt")
        (args.out / "game_vocab_scaled.json").write_text(json.dumps(game_vocab, indent=2))
        (args.out / "curriculum_meta.json").write_text(
            json.dumps(
                {
                    "completed_epochs": completed_epochs,
                    "total_epochs": args.epochs,
                    "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                    "n_encoder_params": n_enc,
                    "n_predictor_params": n_pred,
                    "source_sizes": {name: len(tr) for name, (tr, _cat) in sources.items()},
                },
                indent=2,
            )
        )
        print(f"  checkpoint saved: completed_epochs={completed_epochs}")

    npy_rng = np.random.default_rng(1234)
    epoch_times = []
    for epoch_1idx in range(start_epoch + 1, args.epochs + 1):
        t_epoch_start = time.time()
        frac = synthetic_frac_schedule(epoch_1idx, args.epochs, args.synthetic_start, args.synthetic_end, args.curriculum_schedule)
        indices = curriculum.sample_indices(frac, args.steps_per_epoch * args.batch_size, npy_rng)
        realized = curriculum.realized_mix(indices)

        log_this_epoch = epoch_1idx in (1, max(1, args.epochs // 2), args.epochs) or epoch_1idx % max(1, args.epochs // 5) == 0
        if log_this_epoch:
            print(
                f"[curriculum] epoch {epoch_1idx}/{args.epochs}  target_synthetic_frac={frac:.3f}  "
                f"realized_synthetic_frac={realized['synthetic_frac_realized']:.3f}  "
                f"realized_real_frac={realized['real_frac_realized']:.3f}"
            )

        dataset = TransitionDataset(curriculum.all_transitions, game_vocab)
        num_workers = 4 if device.type == "cuda" else 0
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=indices.tolist(),
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )

        online.train()
        predictor.train()
        total_loss = 0.0
        total_lb_loss = 0.0
        n_batches = 0
        for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
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
                + args.load_balance_weight * lb_loss
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema_target(target, online, EMA_MOMENTUM)

            total_loss += loss.item()
            total_lb_loss += lb_loss.item()
            n_batches += 1

        epoch_time = time.time() - t_epoch_start
        epoch_times.append(epoch_time)

        if log_this_epoch:
            stats = evaluate(online, predictor, arc3_local_val_loader or val_loader, device)
            print(
                f"  train_loss={total_loss / n_batches:.4f}  lb_loss={total_lb_loss / n_batches:.3f}  "
                f"val_pred_mse={stats['pred']:.5f}  val_identity_mse={stats['identity']:.5f}  |  "
                f"changed-patches: pred={stats['pred_changed']:.5f} identity={stats['identity_changed']:.5f}  "
                f"changed-patches improvement={(stats['identity_changed'] - stats['pred_changed']) / max(stats['identity_changed'], 1e-8) * 100:.1f}%  "
                f"epoch_time={epoch_time:.1f}s"
            )

        if args.checkpoint_every > 0 and epoch_1idx % args.checkpoint_every == 0:
            _save(epoch_1idx)

    _save(args.epochs)
    print(f"mean epoch time: {sum(epoch_times) / max(len(epoch_times), 1):.1f}s over {len(epoch_times)} epochs run this session")
    print(f"final checkpoints saved to {args.out}")


def _add_args(parser: argparse.ArgumentParser) -> None:
    # -- architecture --
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--blocks-per-stage", type=int, default=0)
    parser.add_argument("--feature-channels", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--expert-hidden", type=int, default=64)
    parser.add_argument("--expert-depth", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--load-balance-weight", type=float, default=LOAD_BALANCE_WEIGHT_DEFAULT)

    # -- curriculum --
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--synthetic-start", type=float, default=0.97)
    parser.add_argument("--synthetic-end", type=float, default=0.20)
    parser.add_argument("--curriculum-schedule", choices=["linear", "cosine", "flat"], default="linear")

    # -- data sources (0 = skip that source) --
    parser.add_argument("--minigrid-episodes-per-env", type=int, default=40)
    parser.add_argument("--minigrid-steps-per-episode", type=int, default=80)
    parser.add_argument("--sokoban-episodes-per-config", type=int, default=0)
    parser.add_argument("--sokoban-steps-per-episode", type=int, default=80)
    parser.add_argument("--minatar-episodes-per-game", type=int, default=0)
    parser.add_argument("--minatar-steps-per-episode", type=int, default=80)
    parser.add_argument("--openspiel-episodes-per-game", type=int, default=0)
    parser.add_argument("--openspiel-steps-per-episode", type=int, default=60)
    parser.add_argument("--arc-synthetic-episodes-per-type", type=int, default=0)
    parser.add_argument("--arc-synthetic-steps-per-episode", type=int, default=80)
    parser.add_argument("--external-per-game", type=int, default=None)
    parser.add_argument(
        "--search-harvest-dir",
        type=Path,
        default=None,
        help="Directory of *.recording.jsonl files (same format as ARC-AGI-3-Agents/recordings/) "
        "from a future search-harvested 'winning round' corpus. Not required to run this script.",
    )

    # -- checkpointing --
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_scaled")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--resume-from", type=Path, default=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _add_args(parser)
    args = parser.parse_args()
    train(args)
