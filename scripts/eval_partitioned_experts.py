"""Stage 6 experiment (stage6-partitioned-experts): full evaluation of the
hard-partitioned-expert MoE checkpoint against the stage6-game-holdout
fold-1 baseline checkpoint, on the SAME 150-file verified corpus (all 25
games -- 20 trained + 5 held out: r11l, bp35, m0r0, tr87, ka59).

Four things are measured for each checkpoint, all restricted/pooled
consistently between checkpoints for direct comparability:

1. changed-patches (pred vs identity MSE on changed 8x8 patches), pooled
   over trained games and pooled over held-out games separately, plus a
   held-out per-game breakdown -- the standard metric every other
   stage6-* experiment in this project reports.
2. Gate entropy / dominant-expert frequency (methodology ported from
   scripts/diagnose_gate_entropy_holdout.py, stage6-residual-commitment-
   fix), on held-out games and a 2000-transition trained-games sample --
   is the *deployed* soft-gated blend actually using more of the 8
   experts on held-out games, not just cosmetically partitioned during
   training?
3. InfoGain (methodology ported from scripts/diagnose_infogain_holdout.py),
   on held-out vs trained games -- does raw per-expert disagreement (the
   Stage 5 hypothesis-bundle exploration signal) still survive, same
   question this project asked of the original baseline checkpoint.
4. Split-half best-expert agreement (methodology ported from
   scripts/diagnose_expert_specialization.py, a diagnostic run directly
   requested by the coordinator mid-task): among (game, action) contexts
   in the TRAINED games only, does one expert consistently predict best,
   and if so, is winning competence spread across more of the 8 experts
   under hard-partitioning than in the baseline, or does it just relocate
   the same "2 dominate, 6 are dead weight" pattern onto different game
   clusters?

Usage:
    python scripts/eval_partitioned_experts.py
"""

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, _load_frame_lines
from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor, patch_change_mask
from jepa.hypothesis_bundle import info_gain
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
DOMINANT_THRESHOLD = 0.3
MIN_GROUP_SIZE = 16
INFOGAIN_SAMPLE_N = 500
NUM_CANDIDATE_ACTIONS = 4
GATE_TRAINED_SAMPLE_N = 2000
SPLIT_HALF_TRIALS = 2000
SEED = 0

CHECKPOINTS = {
    "baseline-fold1": REPO_ROOT / "checkpoints_holdout_baseline",
    "partitioned-experts-fold1": REPO_ROOT / "checkpoints_partitioned_experts",
    "recalibrated-experts-fold1": REPO_ROOT / "checkpoints_recalibrated_experts",
    "recalibrated-experts-v2-fold1": REPO_ROOT / "checkpoints_recalibrated_experts_v2",
}


def load_all_verified_transitions() -> list:
    """(frame_t, action_id, x, y, frame_t1, changed, game_id_full) for all
    25 games' verified 150-file random.80 corpus."""
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified corpus files, found {len(files)}"
    transitions = []
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x, y = xy.get("x", 0), xy.get("y", 0)
            changed = cur["frame"] != nxt["frame"]
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], changed, game_id))
    return transitions


def load_checkpoint(ckpt_dir: Path, device):
    game_vocab = json.loads((ckpt_dir / "game_vocab_moe.json").read_text())
    meta = json.loads((ckpt_dir / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(ckpt_dir / "encoder_moe.pt", map_location=device))
    encoder.eval()
    predictor = MoEPredictor(num_games=len(game_vocab), num_experts=num_experts).to(device)
    predictor.load_state_dict(torch.load(ckpt_dir / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, game_vocab, num_experts


@torch.no_grad()
def changed_patches_per_game(encoder, predictor, transitions: list, game_vocab: dict, device) -> dict:
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    per_game = defaultdict(lambda: {"pred": 0.0, "identity": 0.0, "n": 0})
    idx = 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        b = cur.shape[0]
        cur, action_id, xy = cur.to(device), action_id.to(device), xy.to(device)
        nxt, patch_mask = nxt.to(device), patch_mask.to(device)
        game_idx_dev = game_idx.to(device)
        cur_feat = encoder(cur)
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx_dev)
        next_feat = encoder(nxt)
        pred_err = per_region_error(pred_feat, next_feat)
        identity_err = per_region_error(cur_feat, next_feat)
        for i in range(b):
            game_short = transitions[idx + i][6].split("-")[0]
            m = patch_mask[i]
            if m.any():
                per_game[game_short]["pred"] += pred_err[i][m].mean().item()
                per_game[game_short]["identity"] += identity_err[i][m].mean().item()
                per_game[game_short]["n"] += 1
        idx += b
    return per_game


def pooled_stats(per_game: dict, games: list) -> dict | None:
    pred_sum = identity_sum = n_sum = 0.0
    for g in games:
        s = per_game.get(g)
        if not s or s["n"] == 0:
            continue
        pred_sum += s["pred"]
        identity_sum += s["identity"]
        n_sum += s["n"]
    if n_sum == 0:
        return None
    pred_mse = pred_sum / n_sum
    identity_mse = identity_sum / n_sum
    return {
        "pred_changed_mse": pred_mse,
        "identity_changed_mse": identity_mse,
        "improvement_pct": (identity_mse - pred_mse) / identity_mse * 100,
        "n": n_sum,
    }


@torch.no_grad()
def gate_entropy_stats(encoder, predictor, game_vocab, transitions: list, device, num_experts: int) -> dict:
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    max_entropy = math.log(num_experts)
    entropies, max_weights = [], []
    for cur, action_id, xy, _nxt, _pm, game_idx in loader:
        cur = cur.to(device)
        action_id, xy, game_idx = action_id.to(device), xy.to(device), game_idx.to(device)
        cur_feat = encoder(cur)
        _pred, gate_weights = predictor(cur_feat, action_id, xy, game_idx)
        p = gate_weights.clamp(min=1e-12)
        ent = -(p * p.log()).sum(dim=-1)
        entropies.append(ent.cpu())
        max_weights.append(gate_weights.max(dim=-1).values.cpu())
    entropies = torch.cat(entropies)
    max_weights = torch.cat(max_weights)
    return {
        "n": entropies.numel(),
        "mean_entropy": entropies.mean().item(),
        "max_entropy": max_entropy,
        "mean_entropy_pct_of_max": entropies.mean().item() / max_entropy * 100,
        "entropy_std": entropies.std().item(),
        "frac_dominant_expert_gt_0.3": (max_weights > DOMINANT_THRESHOLD).float().mean().item(),
        "mean_max_weight": max_weights.mean().item(),
    }


@torch.no_grad()
def mean_info_gain(frames_and_games: list, encoder, predictor, game_vocab, device, sample_n: int, seed: int = 0):
    rng = random.Random(seed)
    idxs = list(range(len(frames_and_games)))
    rng.shuffle(idxs)
    idxs = idxs[:sample_n]
    values = []
    for i in idxs:
        frame, game_id = frames_and_games[i]
        tensor = torch.from_numpy(arc3_frame_to_tensor(frame)).unsqueeze(0).to(device)
        feat = encoder(tensor)
        game_idx = torch.full((1,), game_vocab.get(game_id, 0), dtype=torch.long, device=device)
        xy = torch.zeros((1, 2), dtype=torch.float32, device=device)
        per_action_ig = []
        for action_id in range(NUM_CANDIDATE_ACTIONS):
            action_t = torch.full((1,), action_id, dtype=torch.long, device=device)
            expert_preds = predictor.predict_all_experts(feat, action_t, xy, game_idx)
            per_action_ig.append(info_gain(expert_preds.squeeze(0)).item())
        values.append(sum(per_action_ig) / len(per_action_ig))
    values_t = torch.tensor(values)
    return values_t.mean().item(), values_t.std().item(), len(values)


@torch.no_grad()
def split_half_specialization(encoder, predictor, game_vocab, changed_transitions: list, device, num_experts: int, rng: random.Random):
    """changed_transitions: (frame_t, action_id, x, y, frame_t1, game_id_full),
    restricted by the caller to TRAINED games only (held-out games were
    never trained on, so a "does the model specialize per context" test
    on them isn't meaningful the same way)."""
    grouped = defaultdict(list)
    for frame_t, action_id, x, y, frame_t1, game_id in changed_transitions:
        mask = patch_change_mask(frame_t, frame_t1)
        if not mask.any():
            continue
        cur_t = torch.from_numpy(arc3_frame_to_tensor(frame_t)).unsqueeze(0).to(device)
        nxt_t = torch.from_numpy(arc3_frame_to_tensor(frame_t1)).unsqueeze(0).to(device)
        feat = encoder(cur_t)
        next_feat = encoder(nxt_t)
        game_idx = torch.full((1,), game_vocab.get(game_id, 0), dtype=torch.long, device=device)
        xy_norm = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)
        action_t = torch.full((1,), action_id, dtype=torch.long, device=device)
        expert_preds = predictor.predict_all_experts(feat, action_t, xy_norm, game_idx).squeeze(0)
        target = next_feat.squeeze(0)
        mask_t = torch.from_numpy(mask).to(device)
        diff2 = (expert_preds - target.unsqueeze(0)).pow(2).mean(dim=1)
        per_expert_err = diff2[:, mask_t].mean(dim=1)
        grouped[(game_id.split("-")[0], action_id)].append(per_expert_err.cpu().numpy())

    rows = []
    for (game, action), errors_list in sorted(grouped.items()):
        n = len(errors_list)
        if n < MIN_GROUP_SIZE:
            continue
        idxs = list(range(n))
        rng.shuffle(idxs)
        half = n // 2
        a_idxs, b_idxs = idxs[:half], idxs[half:]
        a_argmins = [int(np.argmin(errors_list[i])) for i in a_idxs]
        b_argmins = [int(np.argmin(errors_list[i])) for i in b_idxs]
        winner_a = max(set(a_argmins), key=a_argmins.count)
        winner_b = max(set(b_argmins), key=b_argmins.count)
        agree = winner_a == winner_b

        chance_agree = 0
        for _ in range(SPLIT_HALF_TRIALS):
            labels = [rng.randrange(num_experts) for _ in range(n)]
            aa, bb = labels[:half], labels[half:]
            wa = max(set(aa), key=aa.count)
            wb = max(set(bb), key=bb.count)
            chance_agree += int(wa == wb)
        chance_rate = chance_agree / SPLIT_HALF_TRIALS

        argmins_all = [int(np.argmin(e)) for e in errors_list]
        counts = np.bincount(argmins_all, minlength=num_experts).astype(np.float64)
        probs = counts / counts.sum()
        rows.append({
            "game": game, "action": action, "n": n, "agree": agree, "chance_rate": chance_rate,
            "mode_share": float(probs.max()), "mode_expert": int(probs.argmax()),
        })

    if not rows:
        return None
    observed_rate = sum(1 for r in rows if r["agree"]) / len(rows)
    mean_chance = sum(r["chance_rate"] for r in rows) / len(rows)
    mean_mode_share = sum(r["mode_share"] for r in rows) / len(rows)
    winner_counts: dict = defaultdict(int)
    for r in rows:
        winner_counts[r["mode_expert"]] += 1
    winner_dist = {k: v / len(rows) for k, v in sorted(winner_counts.items())}
    return {
        "n_groups": len(rows),
        "observed_agreement_rate": observed_rate,
        "mean_chance_agreement_rate": mean_chance,
        "mean_mode_share": mean_mode_share,
        "winner_distribution": winner_dist,
        "max_winner_share": max(winner_dist.values()) if winner_dist else None,
        "n_experts_ever_winning": len(winner_dist),
    }


def main() -> None:
    device = get_device()
    print(f"device: {device}")
    rng = random.Random(SEED)

    print("loading verified 150-file corpus (all 25 games)...")
    all_transitions = load_all_verified_transitions()
    print(f"{len(all_transitions)} transitions loaded")

    heldout_prefixes = tuple(f"{g}-" for g in HELDOUT_GAMES)
    heldout_transitions = [t for t in all_transitions if t[6].startswith(heldout_prefixes)]
    trained_transitions = [t for t in all_transitions if not t[6].startswith(heldout_prefixes)]
    print(f"held-out transitions: {len(heldout_transitions)}  trained transitions: {len(trained_transitions)}")

    heldout_frames = [(t[0], t[6]) for t in heldout_transitions]
    trained_frames = [(t[0], t[6]) for t in trained_transitions]
    trained_games_list = sorted({t[6].split("-")[0] for t in trained_transitions})
    trained_changed = [(t[0], t[1], t[2], t[3], t[4], t[6]) for t in trained_transitions if t[5]]
    print(f"trained-games changed transitions available for split-half test: {len(trained_changed)}")

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            print(f"\nSKIP {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\n{name} ({ckpt_dir})\n{'=' * 70}")
        encoder, predictor, game_vocab, num_experts = load_checkpoint(ckpt_dir, device)
        n_ho_in_vocab = sum(1 for g in HELDOUT_GAMES if any(k.startswith(f"{g}-") for k in game_vocab))
        print(f"game_vocab entries={len(game_vocab)}; held-out games present={n_ho_in_vocab}/5 (should be 0)")

        # 1. changed-patches
        per_game_ho = changed_patches_per_game(encoder, predictor, heldout_transitions, game_vocab, device)
        per_game_tr = changed_patches_per_game(encoder, predictor, trained_transitions, game_vocab, device)
        ho_pooled = pooled_stats(per_game_ho, HELDOUT_GAMES)
        tr_pooled = pooled_stats(per_game_tr, trained_games_list)
        print(f"[changed-patches] trained games pooled: {tr_pooled}")
        print(f"[changed-patches] held-out games pooled: {ho_pooled}")
        ho_per_game_out = {}
        for g in HELDOUT_GAMES:
            s = per_game_ho.get(g)
            if s and s["n"] > 0:
                imp = (s["identity"] - s["pred"]) / s["identity"] * 100
                ho_per_game_out[g] = {"pred": s["pred"] / s["n"], "identity": s["identity"] / s["n"], "improvement_pct": imp, "n": s["n"]}
                print(f"    {g}: pred={ho_per_game_out[g]['pred']:.6f} identity={ho_per_game_out[g]['identity']:.6f} improvement={imp:+.1f}% n={s['n']}")

        # 2. gate entropy
        g_ho = gate_entropy_stats(encoder, predictor, game_vocab, heldout_transitions, device, num_experts)
        trained_sample = trained_transitions[:]
        rng.shuffle(trained_sample)
        trained_sample = trained_sample[:GATE_TRAINED_SAMPLE_N]
        g_tr = gate_entropy_stats(encoder, predictor, game_vocab, trained_sample, device, num_experts)
        print(f"[gate entropy] held-out: mean={g_ho['mean_entropy']:.4f} ({g_ho['mean_entropy_pct_of_max']:.2f}% of max) "
              f"dominant-frac={g_ho['frac_dominant_expert_gt_0.3']*100:.2f}% (n={g_ho['n']})")
        print(f"[gate entropy] trained sample: mean={g_tr['mean_entropy']:.4f} ({g_tr['mean_entropy_pct_of_max']:.2f}% of max) "
              f"dominant-frac={g_tr['frac_dominant_expert_gt_0.3']*100:.2f}% (n={g_tr['n']})")

        # 3. InfoGain
        ig_ho_mean, ig_ho_std, ig_ho_n = mean_info_gain(heldout_frames, encoder, predictor, game_vocab, device, INFOGAIN_SAMPLE_N)
        ig_tr_mean, ig_tr_std, ig_tr_n = mean_info_gain(trained_frames, encoder, predictor, game_vocab, device, INFOGAIN_SAMPLE_N)
        ratio = ig_ho_mean / ig_tr_mean if ig_tr_mean else float("nan")
        print(f"[InfoGain] held-out mean={ig_ho_mean:.6e} (n={ig_ho_n})  trained mean={ig_tr_mean:.6e} (n={ig_tr_n})  ratio={ratio:.3f}")

        # 4. split-half specialization (trained games only)
        print("computing split-half specialization diagnostic (trained games, changed transitions)...")
        spec = split_half_specialization(encoder, predictor, game_vocab, trained_changed, device, num_experts, rng)
        print(f"[split-half specialization] {spec}")

        results[name] = {
            "changed_patches": {
                "trained_pooled": tr_pooled,
                "heldout_pooled": ho_pooled,
                "heldout_per_game": ho_per_game_out,
            },
            "gate_entropy": {"heldout": g_ho, "trained_sample": g_tr},
            "info_gain": {"heldout_mean": ig_ho_mean, "heldout_n": ig_ho_n, "trained_mean": ig_tr_mean, "trained_n": ig_tr_n, "ratio": ratio},
            "split_half_specialization": spec,
        }

    out_path = REPO_ROOT / "logs" / "partitioned_experts_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
