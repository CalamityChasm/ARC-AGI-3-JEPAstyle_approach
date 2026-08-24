"""Stage 6 ARC-synthetic-puzzles generalization test: does pretraining on
genuinely ARC-3-*shaped* procedural puzzles (jepa/data/arc_synthetic_data.py)
close the held-out-*game* generalization gap any better than the established
no-diversity baseline or the best prior diverse-data attempt (the 2.14M-
transition, 26-game OpenSpiel roster -- see experiments/
stage6_expanded_roster.md)?

Compares two (or more) MoE checkpoints on transitions drawn ONLY from
HELDOUT_GAMES -- games neither checkpoint ever trained on, local or
external -- using the same changed-patches pred-vs-identity MSE metric
(and the same game_vocab.get(game_id, 0) fallback-to-index-0 behavior
hypothesis_agent.py uses on a genuinely novel Kaggle game) that
scripts/eval_game_holdout.py already established for this project's main
held-out-games test.

Usage:
    python scripts/eval_arc_synthetic_holdout.py
    python scripts/eval_arc_synthetic_holdout.py --checkpoint checkpoints_arc_synthetic_fold1:arc-synthetic --checkpoint checkpoints_holdout_baseline_fold1:baseline
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, load_all_transitions
from jepa.device import get_device
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent

# This project's own established fold-1 held-out set -- used throughout
# CLAUDE.md's Stage 6 addendum (game-holdout, multifold-cv, every
# diverse-data attempt including the OpenSpiel roster) so a new result
# here is directly comparable to all of those numbers.
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

DEFAULT_CHECKPOINTS = {
    "arc-synthetic": REPO_ROOT / "checkpoints_arc_synthetic_fold1",
    "no-diversity-baseline": REPO_ROOT / "checkpoints_holdout_baseline_fold1",
}


def load_heldout_transitions() -> list:
    substrings = [f"{g}-" for g in HELDOUT_GAMES]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(HELDOUT_GAMES), (
        f"expected exactly {HELDOUT_GAMES}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.* files"
    )
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, device):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    num_experts, feature_channels = 8, 64
    meta_path = checkpoint_dir / "moe_training_meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        num_experts = meta.get("num_experts", 8)
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device))
    online.eval()
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return online, predictor, game_vocab, meta


@torch.no_grad()
def changed_patches_per_game(online, predictor, transitions: list, game_vocab: dict, device) -> dict:
    fallback_vocab = defaultdict(int, game_vocab)  # unknown game_id -> 0, same as production agent
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    per_game = defaultdict(lambda: {"pred": 0.0, "identity": 0.0, "n": 0})
    overall = {"pred": 0.0, "identity": 0.0, "n": 0}

    idx = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        b = cur.shape[0]
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask = nxt.to(device), patch_mask.to(device)
        game_idx_dev = game_idx.to(device)
        cur_feat = online(cur)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx_dev)
        next_feat = online(nxt)

        pred_err = per_region_error(pred_feat, next_feat)
        identity_err = per_region_error(cur_feat, next_feat)

        for i in range(b):
            game_id_full = transitions[idx + i][6]
            game_short = game_id_full.split("-")[0]
            m = patch_mask[i]
            if m.any():
                p = pred_err[i][m].mean().item()
                idn = identity_err[i][m].mean().item()
                per_game[game_short]["pred"] += p
                per_game[game_short]["identity"] += idn
                per_game[game_short]["n"] += 1
                overall["pred"] += p
                overall["identity"] += idn
                overall["n"] += 1
        idx += b

    result = {"overall": {}, "per_game": {}}
    if overall["n"] > 0:
        result["overall"] = {
            "n_changed": overall["n"],
            "pred_changed_mse": overall["pred"] / overall["n"],
            "identity_changed_mse": overall["identity"] / overall["n"],
            "improvement_pct": (overall["identity"] - overall["pred"]) / overall["identity"] * 100,
        }
    for g, s in per_game.items():
        if s["n"] == 0:
            continue
        result["per_game"][g] = {
            "n_changed": s["n"],
            "pred_changed_mse": s["pred"] / s["n"],
            "identity_changed_mse": s["identity"] / s["n"],
            "improvement_pct": (s["identity"] - s["pred"]) / s["identity"] * 100,
        }
    return result


@torch.no_grad()
def trained_games_sanity_check(online, predictor, game_vocab: dict, device, max_per_game: int = 200) -> dict:
    """Quick pooled changed-patches on the 20 TRAINED games (everything not
    in HELDOUT_GAMES) -- a sanity check that the checkpoint learned
    something real at all (not a degenerate/collapsed run), same spirit as
    every other experiment in this project's Stage 6 addendum."""
    exclude = set(HELDOUT_GAMES)
    substrings = None  # load everything, then filter by game prefix below
    all_transitions = load_all_transitions(REPO_ROOT)
    trained = [t for t in all_transitions if t[6].split("-")[0] not in exclude]
    return changed_patches_per_game(online, predictor, trained, game_vocab, device)["overall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="dir:label pairs, e.g. checkpoints_arc_synthetic_fold1:arc-synthetic. Repeatable. "
             "Defaults to the arc-synthetic vs no-diversity-baseline comparison.",
    )
    parser.add_argument("--skip-trained-sanity", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        checkpoints = {}
        for spec in args.checkpoint:
            d, label = spec.split(":", 1)
            checkpoints[label] = REPO_ROOT / d
    else:
        checkpoints = DEFAULT_CHECKPOINTS

    device = get_device()
    print(f"Device: {device}")
    print(f"Held-out games (never in any training corpus this experiment used): {HELDOUT_GAMES}")

    print("\nLoading held-out-games-only transitions from local recordings...")
    heldout_transitions = load_heldout_transitions()
    print(f"  {len(heldout_transitions)} transitions across {len(HELDOUT_GAMES)} held-out games")

    results = {}
    for name, ckpt_dir in checkpoints.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor, game_vocab, meta = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} "
              f"held-out games present (should be 0 -- confirms true holdout)")
        if meta:
            print(f"  pretrain_source={meta.get('pretrain_source')} pretrain_epochs={meta.get('pretrain_epochs')} "
                  f"epochs={meta.get('epochs')} n_arc_synthetic={meta.get('n_arc_synthetic_transitions')} "
                  f"n_minigrid={meta.get('n_minigrid_transitions')}")

        cp = changed_patches_per_game(online, predictor, heldout_transitions, game_vocab, device)
        print(f"\n[changed-patches] overall (all {len(HELDOUT_GAMES)} held-out games pooled):")
        o = cp["overall"]
        if o:
            print(f"    pred_changed_mse={o['pred_changed_mse']:.6f}  identity_changed_mse={o['identity_changed_mse']:.6f}")
            print(f"    improvement over identity: {o['improvement_pct']:+.2f}%")
        print(f"\n[changed-patches] per held-out game:")
        for g in HELDOUT_GAMES:
            s = cp["per_game"].get(g)
            if s:
                print(f"    {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                      f"improvement={s['improvement_pct']:+.2f}% (n={s['n_changed']})")
            else:
                print(f"    {g}: no changed-patch examples found")

        entry = {"changed_patches_heldout": cp, "meta": meta}
        if not args.skip_trained_sanity:
            trained = trained_games_sanity_check(online, predictor, game_vocab, device)
            print(f"\n[sanity check, TRAINED games pooled]: "
                  f"pred={trained.get('pred_changed_mse', float('nan')):.6f} "
                  f"identity={trained.get('identity_changed_mse', float('nan')):.6f} "
                  f"improvement={trained.get('improvement_pct', float('nan')):+.2f}%")
            entry["changed_patches_trained_sanity"] = trained

        results[name] = entry

    out_path = REPO_ROOT / "logs" / "arc_synthetic_holdout_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
