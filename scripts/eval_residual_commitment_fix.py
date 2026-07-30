"""Stage 6 residual-commitment-fix, step 3: does the fix actually close
some of the held-out-game changed-patches gap, without regressing on the
20 trained games?

Same methodology as scripts/eval_gameid_ablation.py (itself built on
scripts/eval_game_holdout.py): changed-patches (pred vs identity MSE on
changed 8x8 patches), evaluated on TWO slices of the identical local
corpus every checkpoint in this experiment family trains on:

  1. held_out_games: the 5 games (r11l, bp35, m0r0, tr87, ka59) excluded
     from ALL training (local + external) for every checkpoint in this
     family -- the real test, since this fix's whole point is closing
     the ~0% collapse found there.
  2. trained_games: the other 20 games -- the regression check, using
     the FULL local corpus for those games (not a held-back val slice),
     matching eval_gameid_ablation.py's own convention exactly so the
     numbers are directly comparable to that experiment's own table.

Reports both the new residual-commitment-fix checkpoint AND (if present
on disk) the two stage6-game-holdout reference checkpoints
(checkpoints_holdout_baseline / checkpoints_holdout_objid) for an
apples-to-apples comparison in one run.

Usage:
    python scripts/eval_residual_commitment_fix.py
"""

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

HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

CHECKPOINTS = {
    "residual-commitment-fix": REPO_ROOT / "checkpoints_holdout_rescommit",
    "baseline-holdout (reference)": REPO_ROOT / "checkpoints_holdout_baseline",
    "object-identity-holdout (reference)": REPO_ROOT / "checkpoints_holdout_objid",
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
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        num_experts = meta.get("num_experts", 8)
        feature_channels = meta.get("feature_channels", 64)
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device))
    online.eval()
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(torch.load(checkpoint_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return online, predictor, game_vocab


@torch.no_grad()
def changed_patches_per_game(online, predictor, transitions: list, game_vocab: dict, device) -> dict:
    """Same fallback-to-index-0-on-unknown-game_id convention as
    eval_game_holdout.py / eval_gameid_ablation.py -- mirrors
    hypothesis_agent.py's real production behavior on a genuinely novel
    Kaggle game."""
    fallback_vocab = defaultdict(int, game_vocab)
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


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading held-out-games-only transitions...")
    heldout_transitions = load_heldout_transitions()
    print(f"  {len(heldout_transitions)} transitions across {len(HELDOUT_GAMES)} held-out games")

    print("\nLoading trained-games-only transitions...")
    trained_transitions = load_trained_games_transitions()
    trained_games = sorted({t[6].split("-")[0] for t in trained_transitions})
    print(f"  {len(trained_transitions)} transitions across {len(trained_games)} trained games")

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 78}\nCHECKPOINT: {name} ({ckpt_dir})\n{'=' * 78}")
        online, predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} "
              f"held-out games present (should be 0 -- confirms true holdout)")

        ckpt_result = {}
        for slice_name, transitions, games_list in [
            ("held_out_games", heldout_transitions, HELDOUT_GAMES),
            ("trained_games", trained_transitions, trained_games),
        ]:
            cp = changed_patches_per_game(online, predictor, transitions, game_vocab, device)
            print(f"\n[changed-patches] {slice_name}:")
            o = cp["overall"]
            if o:
                print(f"    pred_changed_mse={o['pred_changed_mse']:.6f}  identity_changed_mse={o['identity_changed_mse']:.6f}")
                print(f"    improvement over identity: {o['improvement_pct']:+.1f}%  (n={o['n_changed']})")
            if slice_name == "held_out_games":
                print("    per-game:")
                for g in HELDOUT_GAMES:
                    s = cp["per_game"].get(g)
                    if s:
                        print(f"      {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                              f"improvement={s['improvement_pct']:+.1f}% (n={s['n_changed']})")
                    else:
                        print(f"      {g}: no changed-patch examples found")
            ckpt_result[f"changed_patches_{slice_name}"] = cp

        results[name] = ckpt_result

    out_path = REPO_ROOT / "logs" / "residual_commitment_fix_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")

    print(f"\n{'=' * 78}\nSUMMARY: changed-patches improvement over identity\n{'=' * 78}")
    print(f"{'checkpoint':<38}{'held-out (5 games)':<22}{'trained (20 games)':<22}")
    for name in CHECKPOINTS:
        if name not in results:
            continue
        ho = results[name]["changed_patches_held_out_games"]["overall"].get("improvement_pct")
        tr = results[name]["changed_patches_trained_games"]["overall"].get("improvement_pct")
        ho_s = f"{ho:+.1f}%" if ho is not None else "n/a"
        tr_s = f"{tr:+.1f}%" if tr is not None else "n/a"
        print(f"{name:<38}{ho_s:<22}{tr_s:<22}")


if __name__ == "__main__":
    main()
