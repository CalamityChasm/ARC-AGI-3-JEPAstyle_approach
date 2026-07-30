"""Stage 6 continuous-game-embedding investigation, Phase 2B(b): evaluates
the episode-context MoE predictor (jepa/train_context_moe_predictor.py)
on a fold's held-out games.

For each held-out game's full episode, walks forward through it and, at
every position i >= context_window, builds the context embedding from
that *same held-out episode's own* preceding transitions -- never from a
trained game -- exactly the scenario this investigation's context-encoder
hypothesis is meant to help with: an agent accumulating real experience
within one episode of a genuinely novel game.

Usage:
    python scripts/eval_episode_context_holdout.py --fold 1 \
        --heldout-games r11l,bp35,m0r0,tr87,ka59 \
        --ckpt checkpoints_fold1_episode_context
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.sequences import load_all_episodes
from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor, patch_change_mask
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, EpisodeContextEncoder, MoEPredictor
from jepa.train_context_moe_predictor import CONTEXT_WINDOW

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_heldout_episodes(heldout_games: list) -> list:
    substrings = [f"{g}-" for g in heldout_games]
    episodes = load_all_episodes(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({ep[0][6].split("-")[0] for ep in episodes if ep})
    assert set(games_seen) == set(heldout_games), (
        f"expected exactly {heldout_games}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return episodes


def load_episode_context_checkpoint(checkpoint_dir: Path, device):
    meta_path = checkpoint_dir / "context_training_meta.json"
    num_experts, context_window = 8, CONTEXT_WINDOW
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        num_experts = meta.get("num_experts", 8)
        context_window = meta.get("context_window", CONTEXT_WINDOW)
    online = CNNEncoder(out_channels=64).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_context.pt", map_location=device))
    online.eval()
    predictor = MoEPredictor(num_games=1, num_experts=num_experts, context_mode="external").to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "context_moe_predictor.pt", map_location=device))
    predictor.eval()
    context_encoder = EpisodeContextEncoder(feature_channels=64, embed_dim=16).to(device)
    context_encoder.load_state_dict(
        torch.load(checkpoint_dir / "episode_context_encoder.pt", map_location=device)
    )
    context_encoder.eval()
    return online, predictor, context_encoder, context_window


@torch.no_grad()
def changed_patches_per_game(
    online, predictor, context_encoder, episodes: list, context_window: int, device
) -> dict:
    per_game = defaultdict(lambda: {"pred": 0.0, "identity": 0.0, "n": 0})
    overall = {"pred": 0.0, "identity": 0.0, "n": 0}

    for ep in episodes:
        if len(ep) <= context_window:
            continue
        game_short = ep[0][6].split("-")[0]

        # Encode every frame in the episode once (cheap reuse across
        # positions -- each frame appears as both a context frame for
        # several later positions and, once, as a target frame).
        frames_t = [arc3_frame_to_tensor(t[0]) for t in ep]
        frames_t1 = [arc3_frame_to_tensor(t[4]) for t in ep]
        actions = [t[1] for t in ep]

        for pos in range(context_window, len(ep)):
            frame_t, action_id, x, y, frame_t1, _changed, _game_id = ep[pos]
            cur = torch.from_numpy(frames_t[pos]).unsqueeze(0).to(device)
            nxt = torch.from_numpy(frames_t1[pos]).unsqueeze(0).to(device)
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)
            mask = torch.from_numpy(patch_change_mask(frame_t, frame_t1)).unsqueeze(0).to(device)

            ctx_cur = torch.from_numpy(
                np.stack(frames_t[pos - context_window : pos])
            ).unsqueeze(0).to(device)
            ctx_nxt = torch.from_numpy(
                np.stack(frames_t1[pos - context_window : pos])
            ).unsqueeze(0).to(device)
            ctx_action = torch.tensor(
                [actions[pos - context_window : pos]], dtype=torch.long, device=device
            )

            b, k = ctx_action.shape
            ctx_cur_flat = ctx_cur.view(b * k, *ctx_cur.shape[2:])
            ctx_nxt_flat = ctx_nxt.view(b * k, *ctx_nxt.shape[2:])
            both = torch.cat([ctx_cur_flat, ctx_nxt_flat], dim=0)
            both_feat = online(both)
            both_pooled = both_feat.mean(dim=(2, 3))
            pooled_t, pooled_t1 = both_pooled.chunk(2, dim=0)
            pooled_t = pooled_t.view(b, k, -1)
            pooled_t1 = pooled_t1.view(b, k, -1)
            context_embed = context_encoder(pooled_t, ctx_action, pooled_t1)

            cur_feat = online(cur)
            pred_feat, _gate = predictor(cur_feat, action_t, xy_t, context_embed=context_embed)
            next_feat = online(nxt)

            if mask.any():
                pred_err = per_region_error(pred_feat, next_feat)[mask].mean().item()
                identity_err = per_region_error(cur_feat, next_feat)[mask].mean().item()
                per_game[game_short]["pred"] += pred_err
                per_game[game_short]["identity"] += identity_err
                per_game[game_short]["n"] += 1
                overall["pred"] += pred_err
                overall["identity"] += identity_err
                overall["n"] += 1

    def _finish(d):
        if d["n"] == 0:
            return {}
        return {
            "n_changed": d["n"],
            "pred_changed_mse": d["pred"] / d["n"],
            "identity_changed_mse": d["identity"] / d["n"],
            "improvement_pct": (d["identity"] - d["pred"]) / d["identity"] * 100 if d["identity"] > 0 else None,
        }

    return {
        "overall": _finish(overall),
        "per_game": {g: _finish(s) for g, s in per_game.items() if s["n"] > 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--heldout-games", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    heldout_games = args.heldout_games.split(",")
    device = get_device()
    print(f"Device: {device}")
    print(f"Fold {args.fold} held-out games: {heldout_games}")

    heldout_episodes = load_heldout_episodes(heldout_games)
    n_transitions = sum(len(ep) for ep in heldout_episodes)
    print(f"  {len(heldout_episodes)} episodes, {n_transitions} transitions across {len(heldout_games)} held-out games")

    ckpt_dir = REPO_ROOT / args.ckpt
    if not ckpt_dir.exists():
        print(f"SKIPPING: {ckpt_dir} does not exist")
        return

    online, predictor, context_encoder, context_window = load_episode_context_checkpoint(ckpt_dir, device)
    print(f"  context_window={context_window}")

    cp = changed_patches_per_game(online, predictor, context_encoder, heldout_episodes, context_window, device)
    print(f"\n[changed-patches] held-out games pooled (real within-episode context, never a trained game):")
    o = cp["overall"]
    if o:
        print(f"    pred_changed_mse={o['pred_changed_mse']:.6f}  identity_changed_mse={o['identity_changed_mse']:.6f}")
        print(f"    improvement over identity: {o['improvement_pct']:+.1f}%  (n={o['n_changed']})")
    print(f"\n[changed-patches] per held-out game:")
    for g in heldout_games:
        s = cp["per_game"].get(g)
        if s:
            print(f"    {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                  f"improvement={s['improvement_pct']:+.1f}% (n={s['n_changed']})")
        else:
            print(f"    {g}: no changed-patch examples found")

    out_path = REPO_ROOT / "logs" / "episode_context_holdout_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
    all_results[str(args.fold)] = {"heldout_games": heldout_games, "result": cp}
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved fold {args.fold} results to {out_path}")


if __name__ == "__main__":
    main()
