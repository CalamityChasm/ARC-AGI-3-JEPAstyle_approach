"""Stage 6 diverse-pretraining generalization test: does adding MinAtar
(and optionally Procgen) as a synthetic pretraining source, on top of the
existing MiniGrid source, close any of the held-out-ARC-games
generalization gap documented in CLAUDE.md's "Stage 6 addendum"?

Adapted from scripts/eval_multifold.py, but the two checkpoints being
compared here both use *normal* game-id conditioning (neither is the
--ablate-game-id variant) -- so unlike eval_multifold.py's
force_zero_game_idx flag (which exists to replicate the no-gameid
ablation's training-time behavior), both variants here are evaluated the
same way: the natural game_vocab.get(game_id, 0) fallback for a
never-seen held-out game_id, mirroring hypothesis_agent.py's real
production behavior on a genuinely novel Kaggle game.

Usage:
    python scripts/eval_diverse_pretraining.py --fold 1 \
        --heldout-games r11l,bp35,m0r0,tr87,ka59 \
        --baseline-ckpt checkpoints_diverse_baseline \
        --minatar-ckpt checkpoints_diverse_minatar

Appends its result to logs/diverse_pretraining_results.json (keyed by
fold number) so results accumulate across separate invocations.
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


def load_heldout_transitions(heldout_games: list) -> list:
    substrings = [f"{g}-" for g in heldout_games]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(heldout_games), (
        f"expected exactly {heldout_games}, found {games_seen} -- "
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
    vocab = defaultdict(int, game_vocab)  # unknown game_id -> 0, same as production agent
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--heldout-games", type=str, required=True, help="comma-separated 4-char game codes")
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--minatar-ckpt", type=str, required=True)
    args = parser.parse_args()

    heldout_games = args.heldout_games.split(",")
    device = get_device()
    print(f"Device: {device}")
    print(f"Fold {args.fold} held-out games: {heldout_games}")

    heldout_transitions = load_heldout_transitions(heldout_games)
    print(f"  {len(heldout_transitions)} transitions across {len(heldout_games)} held-out games")

    checkpoints = {
        "baseline (minigrid-only)": REPO_ROOT / args.baseline_ckpt,
        "minigrid+minatar": REPO_ROOT / args.minatar_ckpt,
    }

    fold_result = {"heldout_games": heldout_games, "variants": {}}
    for name, ckpt_dir in checkpoints.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nVARIANT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor, game_vocab, meta = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in heldout_games if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(heldout_games)} "
              f"held-out games present (should be 0 -- confirms true holdout)")
        if meta:
            print(f"  meta: n_minigrid={meta.get('n_minigrid_transitions')} "
                  f"n_minatar={meta.get('n_minatar_transitions')} "
                  f"n_local={meta.get('n_local_transitions')} n_external={meta.get('n_external_transitions')}")

        cp = changed_patches_per_game(online, predictor, heldout_transitions, game_vocab, device)
        print(f"\n[changed-patches] held-out games pooled:")
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

        fold_result["variants"][name] = cp

    out_path = REPO_ROOT / "logs" / "diverse_pretraining_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if out_path.exists():
        all_results = json.loads(out_path.read_text())
    all_results[str(args.fold)] = fold_result
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved fold {args.fold} results to {out_path}")

    print(f"\n{'=' * 70}\nSUMMARY: fold {args.fold}\n{'=' * 70}")
    for name in checkpoints:
        if name not in fold_result["variants"]:
            continue
        ho = fold_result["variants"][name]["overall"].get("improvement_pct")
        ho_s = f"{ho:+.1f}%" if ho is not None else "n/a"
        print(f"{name:<30}{ho_s}")


if __name__ == "__main__":
    main()
