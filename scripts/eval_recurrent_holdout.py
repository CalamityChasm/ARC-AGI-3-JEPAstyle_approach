"""Stage 6 continuous-game-embedding investigation, Phase 1: does Stage
3's recurrent predictor (jepa/models/recurrent_predictor.py) generalize
better to held-out games than the MoE predictor did, thanks to its
observation-derived hidden state (in addition to, not instead of, its
own remaining categorical game_id conditioning)?

Unlike scripts/eval_multifold.py (which evaluates the MoE predictor on
i.i.d.-shuffled single transitions), the recurrent predictor's forward
pass depends on a hidden state accumulated from *preceding in-episode
transitions* -- a bare single-transition batch would always hand it a
zeroed hidden state, which is not how the model would ever actually be
used and would silently make the "does the hidden state help"
comparison meaningless. So this script runs each held-out game's full
episode sequentially (real recurrence, not the 16-step truncated-BPTT
chunks training uses -- eval doesn't need to backprop, so there's no
reason to truncate), maintaining real accumulated hidden state from
episode start, and measures changed-patches pred-vs-identity MSE at
every step. Steps are also split into "early" (the first
WARMUP_STEPS steps, where hidden state is still mostly zeros) vs.
"warmed up" (afterward) to see whether accumulated history specifically
helps, not just whether the model overall beats identity.

Usage:
    python scripts/eval_recurrent_holdout.py --fold 1 \
        --heldout-games r11l,bp35,m0r0,tr87,ka59 \
        --ckpt checkpoints_fold1_recurrent \
        [--ablate-game-id]

Appends its result to logs/recurrent_holdout_results.json (keyed by fold
number) so results accumulate across separate invocations.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.sequences import load_all_episodes
from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor, patch_change_mask
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
WARMUP_STEPS = 4  # steps considered "still mostly zeroed hidden state"


def load_heldout_episodes(heldout_games: list) -> list:
    substrings = [f"{g}-" for g in heldout_games]
    episodes = load_all_episodes(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({ep[0][6].split("-")[0] for ep in episodes if ep})
    assert set(games_seen) == set(heldout_games), (
        f"expected exactly {heldout_games}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return episodes


def load_recurrent_checkpoint(checkpoint_dir: Path, device):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_recurrent.json").read_text())
    feature_channels = 64
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_recurrent.pt", map_location=device))
    online.eval()
    predictor = RecurrentActionConditionedPredictor(
        feature_channels=feature_channels, num_games=len(game_vocab)
    ).to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "recurrent_predictor.pt", map_location=device))
    predictor.eval()
    return online, predictor, game_vocab


@torch.no_grad()
def changed_patches_per_game(
    online, predictor, episodes: list, game_vocab: dict, device, force_zero_game_idx: bool = False
) -> dict:
    """Runs each episode sequentially with real accumulated hidden state
    (zeroed only at episode start, exactly mirroring how a live agent
    would use this model within one episode). game_vocab lookups fall
    back to index 0 for unknown game_ids (held-out games are, by
    construction, never in the checkpoint's vocab) -- mirrors
    scripts/eval_multifold.py's / hypothesis_agent.py's real production
    behavior on a genuinely novel game."""
    fallback_vocab = defaultdict(int, game_vocab)

    per_game = defaultdict(lambda: {"pred": 0.0, "identity": 0.0, "n": 0})
    overall = {"pred": 0.0, "identity": 0.0, "n": 0}
    by_warmup = {
        "early": {"pred": 0.0, "identity": 0.0, "n": 0},
        "warmed_up": {"pred": 0.0, "identity": 0.0, "n": 0},
    }

    for ep in episodes:
        if not ep:
            continue
        game_id_full = ep[0][6]
        game_short = game_id_full.split("-")[0]
        game_idx_val = fallback_vocab[game_id_full]
        if force_zero_game_idx:
            game_idx_val = 0
        game_idx = torch.tensor([game_idx_val], dtype=torch.long, device=device)

        hidden = predictor.init_hidden(1, device)
        for step, (frame_t, action_id, x, y, frame_t1, _changed, _gid) in enumerate(ep):
            cur = torch.from_numpy(arc3_frame_to_tensor(frame_t)).unsqueeze(0).to(device)
            nxt = torch.from_numpy(arc3_frame_to_tensor(frame_t1)).unsqueeze(0).to(device)
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)
            mask = torch.from_numpy(patch_change_mask(frame_t, frame_t1)).unsqueeze(0).to(device)

            cur_feat = online(cur)
            pred_feat, hidden = predictor(cur_feat, action_t, xy_t, hidden, game_idx)
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

                bucket = "early" if step < WARMUP_STEPS else "warmed_up"
                by_warmup[bucket]["pred"] += pred_err
                by_warmup[bucket]["identity"] += identity_err
                by_warmup[bucket]["n"] += 1

    def _finish(d):
        if d["n"] == 0:
            return {}
        return {
            "n_changed": d["n"],
            "pred_changed_mse": d["pred"] / d["n"],
            "identity_changed_mse": d["identity"] / d["n"],
            "improvement_pct": (d["identity"] - d["pred"]) / d["identity"] * 100 if d["identity"] > 0 else None,
        }

    result = {
        "overall": _finish(overall),
        "per_game": {g: _finish(s) for g, s in per_game.items() if s["n"] > 0},
        "by_warmup": {k: _finish(v) for k, v in by_warmup.items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--heldout-games", type=str, required=True, help="comma-separated 4-char game codes")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument(
        "--ablate-game-id",
        action="store_true",
        help="Evaluate with game_idx forced to 0 (must match how the checkpoint was trained).",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default=None,
        help="Label for this checkpoint in the results file (default: 'baseline' or 'no-gameid').",
    )
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
    variant_name = args.variant_name or ("no-gameid" if args.ablate_game_id else "baseline")
    print(f"\n{'=' * 70}\nCHECKPOINT: {variant_name} ({ckpt_dir})  ablate_game_id={args.ablate_game_id}\n{'=' * 70}")

    online, predictor, game_vocab = load_recurrent_checkpoint(ckpt_dir, device)
    n_in_vocab = sum(1 for g in heldout_games if any(k.startswith(f"{g}-") for k in game_vocab))
    print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(heldout_games)} "
          f"held-out games present (should be 0 -- confirms true holdout)")

    cp = changed_patches_per_game(
        online, predictor, heldout_episodes, game_vocab, device, force_zero_game_idx=args.ablate_game_id
    )
    print(f"\n[changed-patches] held-out games pooled (full-episode real hidden state):")
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
    print(f"\n[changed-patches] by warmup bucket (early = first {WARMUP_STEPS} steps of each episode):")
    for k, s in cp["by_warmup"].items():
        if s:
            print(f"    {k}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                  f"improvement={s['improvement_pct']:+.1f}% (n={s['n_changed']})")

    out_path = REPO_ROOT / "logs" / "recurrent_holdout_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
    all_results.setdefault(str(args.fold), {})
    all_results[str(args.fold)]["heldout_games"] = heldout_games
    all_results[str(args.fold)].setdefault("variants", {})
    all_results[str(args.fold)]["variants"][variant_name] = cp
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved fold {args.fold} / {variant_name} results to {out_path}")


if __name__ == "__main__":
    main()
