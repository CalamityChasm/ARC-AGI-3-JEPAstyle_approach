"""Stage 6 game-id-ablation test: does removing per-game embedding
conditioning entirely close the held-out-game generalization gap found in
experiments/stage6_game_holdout.md?

That experiment found both a "production-style" (no contrastive loss) and
an "object-identity-style" (same-color contrastive loss) MoE checkpoint --
both trained WITH per-game embedding conditioning -- collapse to ~identity
parity (changed-patches improvement ~0%) on 5 games held out of training
entirely, because a never-seen game_id falls back to game_idx=0, an
untrained (baseline/object-identity) or trivially-shared (this ablation)
embedding row. The suspected mechanism: the model may be leaning on
per-game identity as a crutch rather than learning game-agnostic dynamics,
so conditioning on an untrained/fallback game embedding at eval/inference
time produces a meaningless signal.

This script adds a third checkpoint -- trained via
`jepa.train_moe_predictor --ablate-game-id` on the IDENTICAL 20-game
corpus/recipe as the other two (only difference: every transition's
game_idx is forced to a constant 0 throughout training AND validation,
see jepa/train_moe_predictor.py's ablate_game_id docs) -- and compares
all three checkpoints on BOTH slices:

  1. The 5 held-out games (r11l, bp35, m0r0, tr87, ka59) -- never seen by
     any of the three checkpoints, local or external. The key question:
     does the no-game-id checkpoint's changed-patches improvement stay
     closer to its trained-games number here, instead of collapsing to
     ~0% like the other two?
  2. The 20 trained games -- the key regression check: does removing
     game conditioning cost anything on games the model DID train on?

For the no-game-id checkpoint, BOTH slices are evaluated with game_idx
forced to 0 (force_zero_game_idx=True) -- matching exactly how it was
trained, so this is a fair like-for-like comparison, not an accidental
mismatch between training and eval conditioning. For the other two
checkpoints, held-out games already fall back to game_idx=0 by
construution (game_vocab.get(game_id, 0), mirroring
hypothesis_agent.py's real behavior on an unseen Kaggle game) while
trained games use their real trained embedding index, exactly as in
experiments/stage6_game_holdout.md.

Usage:
    python scripts/eval_gameid_ablation.py
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

# Same 5 games excluded from ALL training corpora for the whole
# stage6-game-holdout / stage6-gameid-ablation experiment family -- see
# experiments/stage6_game_holdout.md for the selection rationale.
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

# checkpoint name -> (directory, whether eval must force game_idx=0 on
# EVERY transition regardless of slice -- True only for the ablated
# checkpoint, which was trained that way; False for the other two, which
# use their real trained per-game embedding on trained games and the
# natural game_vocab fallback (0) on held-out games).
CHECKPOINTS = {
    "baseline-holdout": (REPO_ROOT / "checkpoints_holdout_baseline", False),
    "object-identity-holdout": (REPO_ROOT / "checkpoints_holdout_objid", False),
    "no-gameid-holdout": (REPO_ROOT / "checkpoints_holdout_nogameid", True),
}

N_PROBE_FRAMES = 8  # frames sampled per game for diagnostic B, matching diagnose_encoder_vs_predictor.py


def load_heldout_transitions() -> list:
    """Local recordings for HELDOUT_GAMES only (include-list via the
    file-prefix substring filter)."""
    substrings = [f"{g}-" for g in HELDOUT_GAMES]
    transitions = load_all_transitions(REPO_ROOT, name_substrings=substrings)
    games_seen = sorted({t[6].split("-")[0] for t in transitions})
    assert set(games_seen) == set(HELDOUT_GAMES), (
        f"expected exactly {HELDOUT_GAMES}, found {games_seen} -- "
        f"check ARC-AGI-3-Agents/recordings/ has all 25 games' random.80 files"
    )
    return transitions


def load_trained_games_transitions() -> list:
    """Local recordings for the 20 TRAINED games only -- everything except
    HELDOUT_GAMES. This is the identical local corpus each of the three
    checkpoints actually trained its ARC-finetune phase on (9,600
    transitions, matching moe_training_meta.json's n_local_transitions)."""
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
def changed_patches_per_game(
    online, predictor, transitions: list, game_vocab: dict, device, force_zero_game_idx: bool = False
) -> dict:
    """Overall + per-game changed-patches pred-vs-identity MSE.

    force_zero_game_idx=True forces every transition's game_idx to 0
    regardless of game_vocab, matching how the ablated checkpoint was
    trained (jepa/train_moe_predictor.py --ablate-game-id). Otherwise,
    unknown game_ids fall back to index 0 via defaultdict (mirrors
    hypothesis_agent.py's real Kaggle-time behavior on a novel game)."""
    if force_zero_game_idx:
        vocab = defaultdict(int)  # every game_id -> 0, unconditionally
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
    """Same logic as scripts/diagnose_encoder_vs_predictor.py / eval_game_holdout.py's
    function of the same name -- encoder-only, no game_idx/predictor
    involvement, so this measures whatever the joint training run did to
    the ENCODER's representations, independent of the predictor's game
    conditioning ablation."""
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
    print(f"Held-out games (never in any training corpus this experiment family used): {HELDOUT_GAMES}")

    print("\nLoading held-out-games-only transitions...")
    heldout_transitions = load_heldout_transitions()
    print(f"  {len(heldout_transitions)} transitions across {len(HELDOUT_GAMES)} held-out games")

    print("\nLoading trained-games-only transitions (the 20 games all three checkpoints trained on)...")
    trained_transitions = load_trained_games_transitions()
    trained_games = sorted({t[6].split("-")[0] for t in trained_transitions})
    print(f"  {len(trained_transitions)} transitions across {len(trained_games)} trained games")

    results = {}
    for name, (ckpt_dir, force_zero) in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 78}\nCHECKPOINT: {name} ({ckpt_dir})  force_zero_game_idx={force_zero}\n{'=' * 78}")
        online, predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)
        n_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"  game_vocab has {len(game_vocab)} entries; {n_in_vocab}/{len(HELDOUT_GAMES)} "
              f"held-out games present (should be 0 -- confirms true holdout)")

        ckpt_result = {}

        for slice_name, transitions, games_list in [
            ("held_out_games", heldout_transitions, HELDOUT_GAMES),
            ("trained_games", trained_transitions, trained_games),
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

        b_heldout = diagnostic_b_object_identity(online, device, HELDOUT_GAMES, N_PROBE_FRAMES)
        b_trained = diagnostic_b_object_identity(online, device, trained_games, N_PROBE_FRAMES)
        for label, b in [("held_out_games", b_heldout), ("trained_games", b_trained)]:
            print(f"\n[diagnostic B] {label}:")
            if b["B_same_type_mean_cos_sim"] is not None and b["B_diff_type_mean_cos_sim"] is not None:
                gap = b["B_same_type_mean_cos_sim"] - b["B_diff_type_mean_cos_sim"]
                print(f"    same-color (n={b['B_same_type_n']}):  {b['B_same_type_mean_cos_sim']:.4f}")
                print(f"    diff-color (n={b['B_diff_type_n']}): {b['B_diff_type_mean_cos_sim']:.4f}")
                print(f"    gap (same - different): {gap:+.4f}")
            else:
                print("    insufficient same/diff-color pairs found")
        ckpt_result["diagnostic_b_held_out_games"] = b_heldout
        ckpt_result["diagnostic_b_trained_games"] = b_trained

        results[name] = ckpt_result

    out_path = REPO_ROOT / "logs" / "gameid_ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved full results to {out_path}")

    # Compact summary table
    print(f"\n{'=' * 78}\nSUMMARY: changed-patches improvement over identity\n{'=' * 78}")
    print(f"{'checkpoint':<28}{'held-out (5 games)':<22}{'trained (20 games)':<22}")
    for name in CHECKPOINTS:
        if name not in results:
            continue
        ho = results[name]["changed_patches_held_out_games"]["overall"].get("improvement_pct")
        tr = results[name]["changed_patches_trained_games"]["overall"].get("improvement_pct")
        ho_s = f"{ho:+.1f}%" if ho is not None else "n/a"
        tr_s = f"{tr:+.1f}%" if tr is not None else "n/a"
        print(f"{name:<28}{ho_s:<22}{tr_s:<22}")


if __name__ == "__main__":
    main()
