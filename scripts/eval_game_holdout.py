"""Stage 6 game-holdout generalization test: does the object-identity
checkpoint's advantage over production survive on games *never seen in
training* -- the best local proxy for Kaggle's hidden-game evaluation?

Compares two MoE checkpoints (a "production-style" baseline and an
"object-identity-style" checkpoint, both trained via
jepa.train_moe_predictor on the identical 20-game corpus, see
experiments/stage6_game_holdout.md) on transitions drawn ONLY from
HELDOUT_GAMES -- games neither checkpoint ever trained on, local or
external. Two things are measured, both restricted to the held-out
games only:

  1. changed-patches (pred vs identity MSE on changed 8x8 patches),
     overall and per-game -- does the prediction-quality edge (if any)
     generalize?
  2. diagnostic B (same-color vs different-color patch cosine
     similarity gap) -- does the *representation* advantage itself
     generalize, or is it specific to the 20 training games' own color
     statistics?

Held-out games are looked up in the checkpoint's game_vocab with a
fallback to index 0 for any game_id not present -- this exactly mirrors
ARC-AGI-3-Agents/agents/templates/hypothesis_agent.py's real behavior on
a genuinely novel Kaggle game (`game_vocab.get(self.game_id, 0)`), so the
eval faithfully simulates what happens on a hidden game rather than
crashing or silently cheating with a held-out game's own embedding.

Usage:
    python scripts/eval_game_holdout.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, load_all_transitions, _load_frame_lines
from jepa.device import get_device
from jepa.grid import PATCH, arc3_frame_to_tensor
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent

# The 5 games excluded from ALL training corpora for this experiment --
# see experiments/stage6_game_holdout.md for the selection rationale
# (3 "commonly solved"/well-studied games + 2 essentially untouched ones).
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

CHECKPOINTS = {
    "baseline-holdout": REPO_ROOT / "checkpoints_holdout_baseline",
    "object-identity-holdout": REPO_ROOT / "checkpoints_holdout_objid",
}

N_PROBE_FRAMES = 8  # frames sampled per game for diagnostic B, matching diagnose_encoder_vs_predictor.py


def load_heldout_transitions() -> list:
    """Local recordings for HELDOUT_GAMES only, via the file-prefix filter
    (name_substrings reused as an include-list: every held-out game's
    recording files start with '<game>-', so filtering to those
    substrings selects exactly and only their transitions)."""
    substrings = [f"{g}-" for g in HELDOUT_GAMES]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(HELDOUT_GAMES), (
        f"expected exactly {HELDOUT_GAMES}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, device):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    num_experts, feature_channels = 8, 64
    meta_path = checkpoint_dir / "moe_training_meta.json"
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
    return online, predictor, game_vocab


@torch.no_grad()
def changed_patches_per_game(online, predictor, transitions: list, game_vocab: dict, device) -> dict:
    """Overall + per-game changed-patches pred-vs-identity MSE, restricted
    to whatever `transitions` contains (here: held-out games only, full
    population -- no train/val split needed since NONE of this data was
    ever trained on by either checkpoint). game_vocab lookups fall back
    to index 0 for unknown game_ids, mirroring hypothesis_agent.py's real
    Kaggle-time behavior on a novel game."""
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

        pred_err = per_region_error(pred_feat, next_feat)  # (B, 8, 8)
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
def diagnostic_b_object_identity(online, device, games: list, n_frames: int, seed: int = 0) -> dict:
    """Same logic as scripts/diagnose_encoder_vs_predictor.py's function of
    the same name, restricted to `games` and pointed at this worktree's
    recordings dir (which holds all 25 games' verified random.80 files,
    including the held-out ones -- these are source *frames* to probe the
    representation on, not training data, so using the full corpus here
    is correct and doesn't leak training signal)."""
    rng = random.Random(seed)
    same_type_sims, diff_type_sims = [], []

    for game in games:
        files = sorted((REPO_ROOT / "ARC-AGI-3-Agents" / "recordings").glob(f"{game}-*.recording.jsonl"))
        if not files:
            continue
        frames_pool = _load_frame_lines(files[0])
        if not frames_pool:
            continue
        sample_idxs = rng.sample(range(len(frames_pool)), min(n_frames, len(frames_pool)))

        for idx in sample_idxs:
            grid = np.asarray(frames_pool[idx]["frame"][0], dtype=np.int64)  # (64, 64)
            tensor = torch.from_numpy(arc3_frame_to_tensor(frames_pool[idx]["frame"])).unsqueeze(0).to(device)
            feat = online(tensor)[0]  # (C, 8, 8)

            patch_colors = {}
            for r in range(8):
                for c in range(8):
                    block = grid[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
                    vals, counts = np.unique(block, return_counts=True)
                    patch_colors[(r, c)] = int(vals[counts.argmax()])

            color_counts = {}
            for col in patch_colors.values():
                color_counts[col] = color_counts.get(col, 0) + 1
            background = max(color_counts, key=color_counts.get)

            by_color = {}
            for pos, col in patch_colors.items():
                if col == background:
                    continue
                by_color.setdefault(col, []).append(pos)

            def cos_sim(p1, p2):
                v1, v2 = feat[:, p1[0], p1[1]], feat[:, p2[0], p2[1]]
                return torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

            for col, positions in by_color.items():
                if len(positions) < 2:
                    continue
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        same_type_sims.append(cos_sim(positions[i], positions[j]))

            all_fg_positions = [(p, col) for p, col in patch_colors.items() if col != background]
            for _ in range(min(10, len(all_fg_positions) * (len(all_fg_positions) - 1) // 2 or 0)):
                if len(all_fg_positions) < 2:
                    break
                (p1, c1), (p2, c2) = rng.sample(all_fg_positions, 2)
                if c1 != c2:
                    diff_type_sims.append(cos_sim(p1, p2))

    return {
        "B_same_type_mean_cos_sim": float(np.mean(same_type_sims)) if same_type_sims else None,
        "B_same_type_n": len(same_type_sims),
        "B_diff_type_mean_cos_sim": float(np.mean(diff_type_sims)) if diff_type_sims else None,
        "B_diff_type_n": len(diff_type_sims),
    }


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"Held-out games (never in any training corpus this experiment used): {HELDOUT_GAMES}")

    print("\nLoading held-out-games-only transitions from local recordings...")
    heldout_transitions = load_heldout_transitions()
    print(f"  {len(heldout_transitions)} transitions across {len(HELDOUT_GAMES)} held-out games")

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} "
              f"held-out games present (should be 0 -- confirms true holdout)")

        cp = changed_patches_per_game(online, predictor, heldout_transitions, game_vocab, device)
        print(f"\n[changed-patches] overall (all {len(HELDOUT_GAMES)} held-out games pooled):")
        o = cp["overall"]
        if o:
            print(f"    pred_changed_mse={o['pred_changed_mse']:.6f}  identity_changed_mse={o['identity_changed_mse']:.6f}")
            print(f"    improvement over identity: {o['improvement_pct']:+.1f}%")
        print(f"\n[changed-patches] per held-out game:")
        for g in HELDOUT_GAMES:
            s = cp["per_game"].get(g)
            if s:
                print(f"    {g}: pred={s['pred_changed_mse']:.6f} identity={s['identity_changed_mse']:.6f} "
                      f"improvement={s['improvement_pct']:+.1f}% (n={s['n_changed']})")
            else:
                print(f"    {g}: no changed-patch examples found")

        b = diagnostic_b_object_identity(online, device, HELDOUT_GAMES, N_PROBE_FRAMES)
        print(f"\n[diagnostic B] object-identity cosine similarity (held-out games only):")
        if b["B_same_type_mean_cos_sim"] is not None and b["B_diff_type_mean_cos_sim"] is not None:
            gap = b["B_same_type_mean_cos_sim"] - b["B_diff_type_mean_cos_sim"]
            print(f"    same-color, diff-location (n={b['B_same_type_n']}):  {b['B_same_type_mean_cos_sim']:.4f}")
            print(f"    diff-color, diff-location (n={b['B_diff_type_n']}): {b['B_diff_type_mean_cos_sim']:.4f}")
            print(f"    gap (same - different): {gap:+.4f}")
        else:
            print("    insufficient same/diff-color pairs found")

        results[name] = {"changed_patches": cp, "diagnostic_b": b}

    out_path = REPO_ROOT / "logs" / "game_holdout_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
