"""Stage 6 game-id reseed test: is the +64.9% (no-game-id) vs +8.0%
(with-game-id) changed-patches gap found in experiments/stage6_gameid_ablation.md
a robust, reproducible effect, or a lucky/unlucky single-run draw?

Reuses stage6-gameid-ablation's exact 20-game corpus/recipe and
scripts/eval_gameid_ablation.py's evaluation methodology (changed-patches
improvement over identity, pooled + per-game, on both the 20 TRAINED games
and the 5 held-out games), but generalizes it to N checkpoints instead of
a fixed 3, so it can score 3 independently-seeded runs per condition (6
checkpoints total) and report the distribution (mean/std/min/max) per
condition rather than a single number.

Usage:
    python scripts/eval_gameid_reseed.py
"""

import json
import statistics
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

# Same 5 games excluded from all training corpora as stage6-game-holdout /
# stage6-gameid-ablation -- see experiments/stage6_game_holdout.md.
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

# checkpoint name -> (directory, condition, force_zero_game_idx)
# force_zero_game_idx=True only for the no-game-id (ablated) condition,
# matching how each was actually trained (see jepa/train_moe_predictor.py
# --ablate-game-id).
CHECKPOINTS = {
    "gameid-seed0": (REPO_ROOT / "checkpoints_reseed" / "seed0_gameid", "with-game-id", False),
    "gameid-seed1": (REPO_ROOT / "checkpoints_reseed" / "seed1_gameid", "with-game-id", False),
    "gameid-seed2": (REPO_ROOT / "checkpoints_reseed" / "seed2_gameid", "with-game-id", False),
    "nogameid-seed0": (REPO_ROOT / "checkpoints_reseed" / "seed0_nogameid", "no-game-id", True),
    "nogameid-seed1": (REPO_ROOT / "checkpoints_reseed" / "seed1_nogameid", "no-game-id", True),
    "nogameid-seed2": (REPO_ROOT / "checkpoints_reseed" / "seed2_nogameid", "no-game-id", True),
}


def load_heldout_transitions() -> list:
    substrings = [f"{g}-" for g in HELDOUT_GAMES]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(HELDOUT_GAMES), (
        f"expected exactly {HELDOUT_GAMES}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return transitions


def load_trained_games_transitions() -> list:
    transitions = load_all_transitions(REPO_ROOT, exclude_games=HELDOUT_GAMES)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert not (set(games_seen) & set(HELDOUT_GAMES)), (
        f"held-out games leaked into the 'trained games' slice: "
        f"{set(games_seen) & set(HELDOUT_GAMES)}"
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
def changed_patches_per_game(
    online, predictor, transitions: list, game_vocab: dict, device, force_zero_game_idx: bool = False
) -> dict:
    if force_zero_game_idx:
        vocab = defaultdict(int)
    else:
        vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, vocab)
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


def _dist(vals: list) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "values": vals,
    }


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Held-out games (never in any training corpus): {HELDOUT_GAMES}")

    print("\nLoading held-out-games-only transitions...")
    heldout_transitions = load_heldout_transitions()
    print(f"  {len(heldout_transitions)} transitions across {len(HELDOUT_GAMES)} held-out games")

    print("\nLoading trained-games-only transitions (20 games)...")
    trained_transitions = load_trained_games_transitions()
    trained_games = sorted({t[6].split("-")[0] for t in trained_transitions})
    print(f"  {len(trained_transitions)} transitions across {len(trained_games)} trained games")

    results = {}
    for name, (ckpt_dir, condition, force_zero) in CHECKPOINTS.items():
        if not ckpt_dir.exists() or not (ckpt_dir / "moe_predictor.pt").exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} not ready")
            continue
        print(f"\n{'=' * 78}\nCHECKPOINT: {name} ({ckpt_dir})  condition={condition}  force_zero_game_idx={force_zero}\n{'=' * 78}")
        online, predictor, game_vocab, meta = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} held-out games present "
              f"(should be 0); seed={meta.get('seed')}  ablate_game_id={meta.get('ablate_game_id')}")

        ckpt_result = {"condition": condition, "seed": meta.get("seed"), "ablate_game_id": meta.get("ablate_game_id")}
        for slice_name, transitions in [
            ("held_out_games", heldout_transitions),
            ("trained_games", trained_transitions),
        ]:
            cp = changed_patches_per_game(
                online, predictor, transitions, game_vocab, device, force_zero_game_idx=force_zero
            )
            print(f"\n[changed-patches] {slice_name}:")
            o = cp["overall"]
            if o:
                print(f"    pred_changed_mse={o['pred_changed_mse']:.6f}  identity_changed_mse={o['identity_changed_mse']:.6f}")
                print(f"    improvement over identity: {o['improvement_pct']:+.1f}%  (n={o['n_changed']})")
            ckpt_result[f"changed_patches_{slice_name}"] = cp

        results[name] = ckpt_result

    out_path = REPO_ROOT / "logs" / "gameid_reseed_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")

    # Distribution summary per condition
    print(f"\n{'=' * 78}\nSUMMARY: changed-patches improvement over identity, by condition\n{'=' * 78}")
    by_condition_trained = defaultdict(list)
    by_condition_heldout = defaultdict(list)
    for name, r in results.items():
        cond = r["condition"]
        tr = r["changed_patches_trained_games"]["overall"].get("improvement_pct")
        ho = r["changed_patches_held_out_games"]["overall"].get("improvement_pct")
        if tr is not None:
            by_condition_trained[cond].append(tr)
        if ho is not None:
            by_condition_heldout[cond].append(ho)

    print("\n-- TRAINED games (20) --")
    for cond, vals in by_condition_trained.items():
        d = _dist(vals)
        print(f"  {cond:<16} n={d['n']}  mean={d['mean']:+.1f}%  std={d['std']:.1f}  min={d['min']:+.1f}%  max={d['max']:+.1f}%  values={[f'{v:+.1f}' for v in vals]}")

    print("\n-- HELD-OUT games (5) --")
    for cond, vals in by_condition_heldout.items():
        d = _dist(vals)
        print(f"  {cond:<16} n={d['n']}  mean={d['mean']:+.1f}%  std={d['std']:.1f}  min={d['min']:+.1f}%  max={d['max']:+.1f}%  values={[f'{v:+.1f}' for v in vals]}")


if __name__ == "__main__":
    main()
