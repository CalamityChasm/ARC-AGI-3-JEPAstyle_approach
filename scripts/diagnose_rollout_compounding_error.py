"""Stage 6 diagnostic: does compounding rollout through the learned MoE
predictor (or the Stage 3 recurrent predictor, via --model recurrent)
accumulate error faster than a fresh single step would, at the same
target depth?

Motivation and full writeup: experiments/stage6_rollout_compounding_error.md.
Read-only -- loads checkpoints and recording data, writes no checkpoints,
touches no agent code.

For each real episode segment (RESET-to-RESET) and each sampled start index
t, and for depth d = 1..MAX_DEPTH:
  (a) compounding rollout -- predictor fed its OWN previous output as the
      next `feat`, conditioned on the REAL action taken at each step, vs
      the REAL encoded frame at t+d. For --model recurrent, the hidden
      state is *also* carried forward purely from the model's own prior
      call (see below) -- this represents what a pure multi-step planner
      through the learned model, with no ground truth reinjected, would
      actually see.
  (b) fresh single-step baseline -- ONE predictor call from the REAL
      encoded frame at t+d-1 (ground truth, not the rollout's own guess)
      to predict t+d, vs the same REAL t+d target. For --model recurrent,
      the hidden state fed into this call is the "real-history" hidden --
      the state that results from advancing the GRU using only REAL
      observed frames from episode start through t+d-1 (exactly what
      `memory_agent.py`'s real deployment usage accumulates, since it
      always calls `_predict` with the actually-observed previous feat,
      never a self-generated one). This isolates "does compounding
      through *both* self-predicted features and self-predicted hidden
      state add error beyond a fresh call with perfect real history."
  (c) identity baseline -- the encoded frame at t held constant, vs the
      same REAL t+d target.

All three are scored both as full (unmasked) latent MSE and as this
project's own "changed-patches" metric (jepa/losses.py: per_region_error,
masked to patches that actually differ between frame_t and frame_{t+d} --
same masking convention jepa/train_moe_predictor.py's evaluate() uses for
a single step, applied here to the multi-step target).

Usage:
    python -m scripts.diagnose_rollout_compounding_error
    python -m scripts.diagnose_rollout_compounding_error --source local --max-depth 10
    python -m scripts.diagnose_rollout_compounding_error --model recurrent
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from jepa.grid import arc3_frame_to_tensor, patch_change_mask
from jepa.losses import per_region_error, prediction_loss
from jepa.models import CNNEncoder, MoEPredictor, RecurrentActionConditionedPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
WINNING_HARVEST_DIR = Path("E:/jepa_overflow/winning_harvest/recordings")
LOCAL_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"

DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_STARTS_PER_EPISODE = 6
SEED = 0


def load_frame_lines(path: Path) -> list:
    """Same predicate as jepa/data/trajectories.py: _load_frame_lines."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw)
            data = event.get("data", {})
            if "frame" in data and "action_input" in data and data["frame"]:
                lines.append(data)
    return lines


def split_episodes(frames: list) -> list:
    """Split an ordered frame list into RESET-delimited episodes.

    jepa/data/trajectories.py's own docstring establishes the convention:
    frame i's action_input is the action that *produced* frame i. So
    action_id == 0 (RESET) on frame i means frame i is the first frame of a
    new episode -- everything from there until the next RESET (or EOF) is
    one contiguous, action-conditioned segment safe to roll a prediction
    across.
    """
    episodes = []
    cur = []
    for f in frames:
        action_id = f["action_input"]["id"]
        if action_id == 0 and cur:
            episodes.append(cur)
            cur = []
        cur.append(f)
    if cur:
        episodes.append(cur)
    return episodes


def load_episodes(recordings_dir: Path) -> list:
    """Returns [(game_id, episode_frames), ...] across every recording file
    in `recordings_dir`."""
    episodes = []
    for path in sorted(Path(recordings_dir).glob("*.recording.jsonl")):
        # See jepa/data/trajectories.py's identical fix: recordings dirs
        # are concurrently modified by other sessions in practice, so a
        # file present at glob time can vanish before it's opened.
        try:
            frames = load_frame_lines(path)
        except FileNotFoundError:
            print(f"skipping {path.name}: vanished between glob and open (concurrent modification)")
            continue
        if not frames:
            continue
        game_id = frames[0].get("game_id", "unknown")
        for ep in split_episodes(frames):
            episodes.append((game_id, ep))
    return episodes


def load_models(device: torch.device, checkpoint_dir: Path = CHECKPOINT_DIR):
    vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device))
    encoder.eval()
    predictor = MoEPredictor(num_games=max(len(vocab), 1)).to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, vocab


def load_models_recurrent(device: torch.device):
    vocab = json.loads((CHECKPOINT_DIR / "game_vocab_recurrent.json").read_text())
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_recurrent.pt", map_location=device))
    encoder.eval()
    predictor = RecurrentActionConditionedPredictor(num_games=max(len(vocab), 1)).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "recurrent_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, vocab


def pick_starts(episode_len: int, max_depth: int, max_starts: int, rng: random.Random) -> list:
    """Up to `max_starts` distinct start indices t with t + max_depth < episode_len,
    evenly-ish spread rather than always t=0 (so a single episode can't
    dominate the sample with near-duplicate rollouts starting at the same
    point)."""
    n_valid = episode_len - max_depth
    if n_valid <= 0:
        return []
    if n_valid <= max_starts:
        return list(range(n_valid))
    # Evenly spaced starts, jittered slightly so we don't always land on the
    # exact same phase of an episode across many similarly-shaped episodes.
    starts = sorted(rng.sample(range(n_valid), max_starts))
    return starts


@torch.no_grad()
def run_episode(
    game_id: str,
    ep_frames: list,
    encoder,
    predictor,
    vocab: dict,
    max_depth: int,
    max_starts: int,
    rng: random.Random,
    device: torch.device,
    stats: dict,
) -> None:
    game_idx = vocab.get(game_id, 0)
    in_vocab = game_id in vocab
    group = "in_vocab" if in_vocab else "out_of_vocab"

    raw_frames = [f["frame"] for f in ep_frames]
    action_ids = [f["action_input"]["id"] for f in ep_frames]
    xys = []
    for f in ep_frames:
        d = f["action_input"].get("data", {}) or {}
        xys.append((d.get("x", 0), d.get("y", 0)))

    tensors = torch.stack([torch.from_numpy(arc3_frame_to_tensor(fr)) for fr in raw_frames]).to(device)
    feats = encoder(tensors)  # (T, C, 8, 8)
    game_t = torch.tensor([game_idx], dtype=torch.long, device=device)

    T = len(ep_frames)
    starts = pick_starts(T, max_depth, max_starts, rng)

    for t in starts:
        cur_feat = feats[t : t + 1]
        frame_t_raw = raw_frames[t]
        for d in range(1, max_depth + 1):
            idx = t + d
            action_id = action_ids[idx]
            x, y = xys[idx]
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)

            target_feat = feats[idx : idx + 1]
            mask_np = patch_change_mask(frame_t_raw, raw_frames[idx])
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
            has_change = bool(mask_t.any())

            # (a) compounding rollout
            comp_pred, _ = predictor(cur_feat, action_t, xy_t, game_t)
            comp_raw = prediction_loss(comp_pred, target_feat).item()
            comp_region = per_region_error(comp_pred, target_feat)
            comp_changed = comp_region[mask_t].mean().item() if has_change else None

            # (b) fresh single-step baseline: from the REAL previous frame
            real_prev_feat = feats[idx - 1 : idx]
            fresh_pred, _ = predictor(real_prev_feat, action_t, xy_t, game_t)
            fresh_raw = prediction_loss(fresh_pred, target_feat).item()
            fresh_region = per_region_error(fresh_pred, target_feat)
            fresh_changed = fresh_region[mask_t].mean().item() if has_change else None

            # (c) identity: frame at t held constant
            id_feat = feats[t : t + 1]
            id_raw = prediction_loss(id_feat, target_feat).item()
            id_region = per_region_error(id_feat, target_feat)
            id_changed = id_region[mask_t].mean().item() if has_change else None

            rec = stats[(group, d)]
            rec["n_total"] += 1
            rec["comp_raw"].append(comp_raw)
            rec["fresh_raw"].append(fresh_raw)
            rec["id_raw"].append(id_raw)
            if has_change:
                rec["n_changed"] += 1
                rec["comp_changed"].append(comp_changed)
                rec["fresh_changed"].append(fresh_changed)
                rec["id_changed"].append(id_changed)
            rec["games"].add(game_id)

            # Continue the compounding chain with the model's OWN prediction,
            # not ground truth -- this is the whole point of the measurement.
            cur_feat = comp_pred


@torch.no_grad()
def run_episode_recurrent(
    game_id: str,
    ep_frames: list,
    encoder,
    predictor,
    vocab: dict,
    max_depth: int,
    max_starts: int,
    rng: random.Random,
    device: torch.device,
    stats: dict,
) -> None:
    """Same methodology as run_episode, extended for the recurrent model's
    hidden state. See the module docstring's (a)/(b)/(c) description for
    exactly what "compounding" vs "fresh" mean here w.r.t. hidden state."""
    game_idx = vocab.get(game_id, 0)
    in_vocab = game_id in vocab
    group = "in_vocab" if in_vocab else "out_of_vocab"

    raw_frames = [f["frame"] for f in ep_frames]
    action_ids = [f["action_input"]["id"] for f in ep_frames]
    xys = []
    for f in ep_frames:
        d = f["action_input"].get("data", {}) or {}
        xys.append((d.get("x", 0), d.get("y", 0)))

    tensors = torch.stack([torch.from_numpy(arc3_frame_to_tensor(fr)) for fr in raw_frames]).to(device)
    feats = encoder(tensors)  # (T, C, 8, 8)
    game_t = torch.tensor([game_idx], dtype=torch.long, device=device)

    T = len(ep_frames)

    # Precompute the "real-history" hidden state at every position: the
    # hidden state a real deployed agent would actually be carrying at
    # step i, having advanced the GRU using only REAL observed frames and
    # REAL actions from episode start (see memory_agent.py's
    # _update_from_last_turn -- it always feeds the actually-observed
    # previous feat, never a self-generated one, into the predictor to
    # advance self._hidden). real_hidden_at[0] is the zero init.
    real_hidden_at = [predictor.init_hidden(1, device)]
    for i in range(1, T):
        action_t = torch.tensor([action_ids[i]], dtype=torch.long, device=device)
        x, y = xys[i]
        xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)
        _, new_hidden = predictor(feats[i - 1 : i], action_t, xy_t, real_hidden_at[i - 1], game_t)
        real_hidden_at.append(new_hidden)

    starts = pick_starts(T, max_depth, max_starts, rng)

    for t in starts:
        cur_feat = feats[t : t + 1]
        cur_hidden = real_hidden_at[t]
        frame_t_raw = raw_frames[t]
        for d in range(1, max_depth + 1):
            idx = t + d
            action_id = action_ids[idx]
            x, y = xys[idx]
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)

            target_feat = feats[idx : idx + 1]
            mask_np = patch_change_mask(frame_t_raw, raw_frames[idx])
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(device)
            has_change = bool(mask_t.any())

            # (a) compounding rollout: both feat AND hidden are chained
            # purely from the model's own prior outputs.
            comp_pred, comp_hidden_new = predictor(cur_feat, action_t, xy_t, cur_hidden, game_t)
            comp_raw = prediction_loss(comp_pred, target_feat).item()
            comp_region = per_region_error(comp_pred, target_feat)
            comp_changed = comp_region[mask_t].mean().item() if has_change else None

            # (b) fresh single-step baseline: REAL previous frame's feat,
            # combined with the REAL-history hidden state at idx-1 (not the
            # rollout's own drifted hidden) -- one predictor call given
            # perfect knowledge of everything up to now.
            real_prev_feat = feats[idx - 1 : idx]
            fresh_hidden = real_hidden_at[idx - 1]
            fresh_pred, _ = predictor(real_prev_feat, action_t, xy_t, fresh_hidden, game_t)
            fresh_raw = prediction_loss(fresh_pred, target_feat).item()
            fresh_region = per_region_error(fresh_pred, target_feat)
            fresh_changed = fresh_region[mask_t].mean().item() if has_change else None

            # (c) identity: frame at t held constant.
            id_feat = feats[t : t + 1]
            id_raw = prediction_loss(id_feat, target_feat).item()
            id_region = per_region_error(id_feat, target_feat)
            id_changed = id_region[mask_t].mean().item() if has_change else None

            rec = stats[(group, d)]
            rec["n_total"] += 1
            rec["comp_raw"].append(comp_raw)
            rec["fresh_raw"].append(fresh_raw)
            rec["id_raw"].append(id_raw)
            if has_change:
                rec["n_changed"] += 1
                rec["comp_changed"].append(comp_changed)
                rec["fresh_changed"].append(fresh_changed)
                rec["id_changed"].append(id_changed)
            rec["games"].add(game_id)

            # Continue the compounding chain with the model's OWN
            # prediction and OWN advanced hidden state -- not ground truth.
            cur_feat = comp_pred
            cur_hidden = comp_hidden_new


def summarize(stats: dict, max_depth: int) -> list:
    rows = []
    for group in ("in_vocab", "out_of_vocab"):
        for d in range(1, max_depth + 1):
            rec = stats.get((group, d))
            if rec is None or rec["n_total"] == 0:
                continue
            n_changed = rec["n_changed"]

            def mean(xs):
                return float(np.mean(xs)) if xs else float("nan")

            rows.append(
                {
                    "group": group,
                    "depth": d,
                    "n_total": rec["n_total"],
                    "n_changed": n_changed,
                    "n_games": len(rec["games"]),
                    "comp_raw_mse": mean(rec["comp_raw"]),
                    "fresh_raw_mse": mean(rec["fresh_raw"]),
                    "id_raw_mse": mean(rec["id_raw"]),
                    "comp_changed_mse": mean(rec["comp_changed"]),
                    "fresh_changed_mse": mean(rec["fresh_changed"]),
                    "id_changed_mse": mean(rec["id_changed"]),
                }
            )
    return rows


def print_table(rows: list, group: str) -> None:
    print(f"\n=== {group} ===")
    header = (
        f"{'d':>3} {'n_tot':>6} {'n_chg':>6} {'n_games':>7} | "
        f"{'comp_chg':>10} {'fresh_chg':>10} {'id_chg':>10} | "
        f"{'gap(c-f)':>9} {'gap%':>7} | "
        f"{'comp_raw':>10} {'fresh_raw':>10} {'id_raw':>10}"
    )
    print(header)
    for r in rows:
        if r["group"] != group:
            continue
        gap = r["comp_changed_mse"] - r["fresh_changed_mse"]
        gap_pct = 100.0 * gap / r["fresh_changed_mse"] if r["fresh_changed_mse"] else float("nan")
        print(
            f"{r['depth']:>3} {r['n_total']:>6} {r['n_changed']:>6} {r['n_games']:>7} | "
            f"{r['comp_changed_mse']:>10.6f} {r['fresh_changed_mse']:>10.6f} {r['id_changed_mse']:>10.6f} | "
            f"{gap:>9.6f} {gap_pct:>6.1f}% | "
            f"{r['comp_raw_mse']:>10.6f} {r['fresh_raw_mse']:>10.6f} {r['id_raw_mse']:>10.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["winning_harvest", "local", "both"], default="both")
    parser.add_argument("--model", choices=["moe", "recurrent"], default="moe")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint dir (moe model only) -- e.g. checkpoints_residual_finetune/.")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-starts-per-episode", type=int, default=DEFAULT_MAX_STARTS_PER_EPISODE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to dump raw per-depth JSON.")
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.model == "recurrent":
        encoder, predictor, vocab = load_models_recurrent(device)
        run_fn = run_episode_recurrent
        used_dir = CHECKPOINT_DIR
    else:
        used_dir = args.checkpoint_dir or CHECKPOINT_DIR
        encoder, predictor, vocab = load_models(device, used_dir)
        run_fn = run_episode
    print(f"model={args.model}; loaded checkpoints from {used_dir}, "
          f"{len(vocab)}-entry game vocab: {sorted(vocab.keys())}")

    sources = []
    if args.source in ("winning_harvest", "both"):
        sources.append(("winning_harvest", WINNING_HARVEST_DIR))
    if args.source in ("local", "both"):
        sources.append(("local", LOCAL_DIR))

    all_results = {}
    for name, path in sources:
        print(f"\n--- loading episodes from {name}: {path} ---")
        episodes = load_episodes(path)
        print(f"{len(episodes)} RESET-delimited episodes across {len(set(g for g, _ in episodes))} games")
        lengths = [len(ep) for _, ep in episodes]
        eligible = [l for l in lengths if l - args.max_depth > 0]
        print(
            f"episode length: min={min(lengths)} max={max(lengths)} mean={np.mean(lengths):.1f}; "
            f"{len(eligible)}/{len(episodes)} episodes long enough for max_depth={args.max_depth}"
        )

        rng = random.Random(args.seed)
        stats = defaultdict(
            lambda: {
                "n_total": 0,
                "n_changed": 0,
                "comp_raw": [],
                "fresh_raw": [],
                "id_raw": [],
                "comp_changed": [],
                "fresh_changed": [],
                "id_changed": [],
                "games": set(),
            }
        )
        for game_id, ep in episodes:
            run_fn(
                game_id, ep, encoder, predictor, vocab, args.max_depth,
                args.max_starts_per_episode, rng, device, stats,
            )

        rows = summarize(stats, args.max_depth)
        print_table(rows, "in_vocab")
        print_table(rows, "out_of_vocab")
        all_results[name] = rows

    if args.out:
        args.out.write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote raw results to {args.out}")


if __name__ == "__main__":
    main()
