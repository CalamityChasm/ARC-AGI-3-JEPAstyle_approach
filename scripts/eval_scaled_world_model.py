"""Stage 6 scaled-world-model evaluation: does capacity + genuinely
diverse pretraining data (jepa/data/openspiel_data.py,
jepa/data/arcade_data.py, plus the existing MiniGrid source -- see
experiments/stage6_scaled_world_model.md) close any of the held-out-ARC-
games generalization gap documented in CLAUDE.md's Stage 6 addendum
(~0.0% to -0.3% across 10 independent prior interventions, all on
much-smaller/less-diverse data)?

Generalizes scripts/eval_multifold.py's/eval_game_holdout.py's
`load_moe_checkpoint`/`changed_patches_per_game` (kept those files
untouched for historical reproducibility of their own results) with one
real fix: `load_moe_checkpoint` here reads `feature_channels` from
`moe_training_meta.json` when present, instead of hardcoding 64 -- needed
for this experiment's `--width-mult` checkpoints (a 128-channel encoder's
state dict won't load into a 64-channel model). Also adds a standard
trained-games sanity check (evaluated on the SAME fold's 20 non-held-out
games' local recordings) alongside the held-out-games number, since
Phase 4 of this experiment asks for both, not just the held-out number in
isolation.

Usage:
    python scripts/eval_scaled_world_model.py --ckpt checkpoints_scaled_fold1_w1 \
        --label scaled-w1-fold1 --heldout-games r11l,bp35,m0r0,tr87,ka59
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

ALL_25_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85",
    "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15",
    "tn36", "tr87", "tu93", "vc33", "wa30",
]


def load_transitions_for_games(games: list) -> list:
    substrings = [f"{g}-" for g in games]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(games), (
        f"expected exactly {sorted(games)}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, device):
    """Fixed vs. eval_multifold.py/eval_game_holdout.py's own copies of
    this function: reads feature_channels from moe_training_meta.json
    (written by train_moe_predictor.py's --width-mult support) instead of
    hardcoding 64 -- required for this experiment's wider checkpoints."""
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    num_experts, feature_channels = 8, 64
    meta_path = checkpoint_dir / "moe_training_meta.json"
    meta = {}
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
    return online, predictor, game_vocab, meta


@torch.no_grad()
def changed_patches_per_game(
    online, predictor, transitions: list, game_vocab: dict, device, unknown_game_fallback: bool = True
) -> dict:
    """unknown_game_fallback=True (the held-out-games case): unknown
    game_id -> vocab index 0, mirroring hypothesis_agent.py's real
    Kaggle-time behavior on a genuinely novel game -- NOT a bug, this is
    the faithful simulation of what actually happens. False (the
    trained-games sanity-check case): use the real game_vocab directly,
    every game here should already be a known key."""
    vocab = defaultdict(int, game_vocab) if unknown_game_fallback else game_vocab
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
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--heldout-games", type=str, required=True)
    args = parser.parse_args()

    heldout_games = args.heldout_games.split(",")
    trained_games = [g for g in ALL_25_GAMES if g not in heldout_games]
    ckpt_dir = Path(args.ckpt)
    if not ckpt_dir.is_absolute():
        ckpt_dir = REPO_ROOT / ckpt_dir

    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Held-out games: {heldout_games}")
    print(f"Trained-games sanity-check sample: {trained_games}")

    online, predictor, game_vocab, meta = load_moe_checkpoint(ckpt_dir, device)
    print(f"\nCheckpoint meta: {json.dumps(meta, indent=2)}")
    n_in_vocab = sum(1 for g in heldout_games if any(k.startswith(f"{g}-") for k in game_vocab))
    print(f"game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(heldout_games)} "
          f"held-out games present (should be 0 -- confirms true holdout)")

    heldout_transitions = load_transitions_for_games(heldout_games)
    print(f"\n{len(heldout_transitions)} held-out transitions across {len(heldout_games)} games")
    ho = changed_patches_per_game(online, predictor, heldout_transitions, game_vocab, device, unknown_game_fallback=True)
    print(f"\n[HELD-OUT GAMES -- the number that matters most] changed-patches:")
    o = ho["overall"]
    if o:
        print(f"    pred={o['pred_changed_mse']:.6f}  identity={o['identity_changed_mse']:.6f}  "
              f"improvement={o['improvement_pct']:+.2f}%  (n={o['n_changed']})")
    for g in heldout_games:
        s = ho["per_game"].get(g)
        if s:
            print(f"    {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                  f"improvement={s['improvement_pct']:+.2f}% (n={s['n_changed']})")
        else:
            print(f"    {g}: no changed-patch examples found")

    trained_transitions = load_transitions_for_games(trained_games)
    print(f"\n{len(trained_transitions)} trained-games transitions across {len(trained_games)} games")
    tr = changed_patches_per_game(online, predictor, trained_transitions, game_vocab, device, unknown_game_fallback=False)
    print(f"\n[TRAINED GAMES -- standard sanity check] changed-patches:")
    o = tr["overall"]
    if o:
        print(f"    pred={o['pred_changed_mse']:.6f}  identity={o['identity_changed_mse']:.6f}  "
              f"improvement={o['improvement_pct']:+.2f}%  (n={o['n_changed']})")

    out_path = REPO_ROOT / "logs" / "stage6_scaled_world_model_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(out_path.read_text()) if out_path.exists() else {}
    all_results[args.label] = {
        "ckpt": str(ckpt_dir), "meta": meta, "heldout_games": heldout_games,
        "heldout": ho, "trained_games_sample": trained_games, "trained": tr,
    }
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results (label={args.label}) to {out_path}")


if __name__ == "__main__":
    main()
