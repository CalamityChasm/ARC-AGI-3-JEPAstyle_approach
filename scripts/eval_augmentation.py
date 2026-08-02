"""Stage 6 augmentation test: evaluates changed-patches improvement over
identity on (a) a fold's held-out games (the number that actually
matters) and (b) that fold's trained games (a regression check), for a
baseline (no augmentation) checkpoint vs. a color-augmented checkpoint.

Adapted from scripts/eval_multifold.py (which compares baseline vs.
no-game-id variants on held-out games only) -- this script compares
baseline vs. color-augment variants and adds the trained-games check
that eval_multifold.py didn't need (stage6-multifold-cv only ever cared
about the held-out side).

Evaluation never applies color augmentation itself, regardless of which
checkpoint is being scored -- the whole point is to measure how each
checkpoint performs on the real, unaugmented input distribution actual
gameplay produces (the same standard this project's own eval has always
held, e.g. never training the identity baseline).

Usage:
    python scripts/eval_augmentation.py --fold 1 \
        --heldout-games r11l,bp35,m0r0,tr87,ka59 \
        --trained-games ar25,cd82,... \
        --baseline-ckpt checkpoints_fold1_baseline \
        --augment-ckpt checkpoints_fold1_augment

Appends its result to logs/augmentation_results.json (keyed by fold
number) so results accumulate across separate invocations.
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


def load_games_transitions(games: list) -> list:
    """Loads local-recordings transitions for exactly the given games
    (via load_all_transitions's include-list use of name_substrings),
    asserting the recordings directory actually has all of them."""
    substrings = [f"{g}-" for g in games]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(games), (
        f"expected exactly {games}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
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
    """No color augmentation here, ever -- real eval always uses the raw,
    unaugmented frames (see module docstring)."""
    vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, vocab, color_augment=False)
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


def _print_block(label: str, cp: dict, games: list) -> None:
    print(f"\n[changed-patches] {label}:")
    o = cp["overall"]
    if o:
        print(
            f"    pred_changed_mse={o['pred_changed_mse']:.6f}  "
            f"identity_changed_mse={o['identity_changed_mse']:.6f}"
        )
        print(f"    improvement over identity: {o['improvement_pct']:+.2f}%  (n={o['n_changed']})")
    else:
        print("    no changed-patch examples found")
    print(f"  per game:")
    for g in games:
        s = cp["per_game"].get(g)
        if s:
            print(
                f"    {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                f"improvement={s['improvement_pct']:+.2f}% (n={s['n_changed']})"
            )
        else:
            print(f"    {g}: no changed-patch examples found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--heldout-games", type=str, required=True, help="comma-separated game codes")
    parser.add_argument("--trained-games", type=str, required=True, help="comma-separated game codes")
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--augment-ckpt", type=str, required=True)
    args = parser.parse_args()

    heldout_games = args.heldout_games.split(",")
    trained_games = args.trained_games.split(",")
    device = get_device()
    print(f"Device: {device}")
    print(f"Fold {args.fold} held-out games: {heldout_games}")
    print(f"Fold {args.fold} trained games (regression check): {trained_games}")

    heldout_transitions = load_games_transitions(heldout_games)
    print(f"  {len(heldout_transitions)} held-out-game transitions")
    trained_transitions = load_games_transitions(trained_games)
    print(f"  {len(trained_transitions)} trained-game transitions (local recordings only)")

    checkpoints = {
        "baseline": REPO_ROOT / args.baseline_ckpt,
        "color-augment": REPO_ROOT / args.augment_ckpt,
    }

    fold_result = {
        "heldout_games": heldout_games,
        "trained_games": trained_games,
        "variants": {},
    }
    for name, ckpt_dir in checkpoints.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nVARIANT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor, game_vocab, meta = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in heldout_games if any(k.startswith(f"{g}-") for k in game_vocab))
        print(
            f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(heldout_games)} "
            f"held-out games present (should be 0 -- confirms true holdout)"
        )
        print(f"  meta: color_augment={meta.get('color_augment')} contrast_weight={meta.get('contrast_weight')} "
              f"exclude_games={meta.get('exclude_games')}")

        cp_heldout = changed_patches_per_game(online, predictor, heldout_transitions, game_vocab, device)
        _print_block("held-out games (the number that matters)", cp_heldout, heldout_games)

        cp_trained = changed_patches_per_game(online, predictor, trained_transitions, game_vocab, device)
        _print_block("trained games (regression check)", cp_trained, trained_games)

        fold_result["variants"][name] = {"heldout": cp_heldout, "trained": cp_trained}

    out_path = REPO_ROOT / "logs" / "augmentation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
    all_results[str(args.fold)] = fold_result
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved fold {args.fold} results to {out_path}")

    print(f"\n{'=' * 70}\nSUMMARY: fold {args.fold}\n{'=' * 70}")
    for name in ["baseline", "color-augment"]:
        if name not in fold_result["variants"]:
            continue
        ho = fold_result["variants"][name]["heldout"]["overall"].get("improvement_pct")
        tr = fold_result["variants"][name]["trained"]["overall"].get("improvement_pct")
        ho_s = f"{ho:+.2f}%" if ho is not None else "n/a"
        tr_s = f"{tr:+.2f}%" if tr is not None else "n/a"
        print(f"{name:<15} held-out={ho_s:<10} trained={tr_s}")


if __name__ == "__main__":
    main()
