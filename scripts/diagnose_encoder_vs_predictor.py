"""Decompose "the world model doesn't predict change well" into its two
possible causes: the ENCODER not representing what's actually in a frame,
vs the PREDICTOR not modeling how actions change it -- re-running (on the
current checkpoints, with today's data) a version of the diagnostic Stage 1
ran once, long before the action-input bugfix, MoE, and everything since,
plus a genuinely new test neither Stage 1 nor anything since has run:
whether the encoder represents *object identity* at all, not just "did a
pixel here change."

Four sub-diagnostics, each isolating one claim:

  A. Encoder change-sensitivity: does encoder feature-space distance between
     frame_t and frame_t+1 differ at patches that actually changed vs ones
     that didn't? (Stage 1's original test, re-run fresh.)
  B. Encoder object-identity: do two instances of the same color/object at
     DIFFERENT grid locations map to more similar feature vectors than two
     DIFFERENT-colored patches do? Tests representation, not just
     sensitivity-to-change. Never tested before in this project.
  C. Predictor residual commitment: how does the predictor's own residual
     magnitude compare to the true feature-space delta, at patches that
     actually changed? (Stage 1's other original test, re-run fresh.)
  D. Predictor action-sensitivity: for the SAME starting state, does
     varying the hypothetical action id change the predicted output
     meaningfully, or does the predictor produce nearly the same output
     regardless of which action is asked about?

Usage:
    python scripts/diagnose_encoder_vs_predictor.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset, _load_frame_lines
from jepa.device import get_device
from jepa.grid import PATCH, arc3_frame_to_tensor
from jepa.losses import per_region_error
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
# Verified (stage6-capacity-sweep, four independent checks) as the exact
# 150-file / 12,000-transition corpus checkpoints/moe_predictor.pt was
# actually trained on -- the main checkout's own ARC-AGI-3-Agents/recordings/
# currently holds unrelated leftover files (see CLAUDE.md's Gotchas
# section), so pointing at that would silently evaluate against the wrong
# data.
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
SAMPLE_N = 2000  # transitions sampled for A/C/D -- fast, still >1500 changed examples
N_PROBE_FRAMES = 8  # frames sampled per game for diagnostic B

CHECKPOINTS = {
    "production": Path("checkpoints"),
    "search-harvest": Path("E:/jepa_overflow/checkpoints_search"),
    "object-identity": Path("C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/agent-a5728f08b06dab74a/checkpoints_object_identity"),
}

PROBE_GAMES = ["ft09", "m0r0", "r11l", "bp35", "s5i5", "tr87", "ka59", "vc33"]


def load_verified_transitions() -> list:
    transitions = []
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified random-corpus files, found {len(files)}"
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x, y = xy.get("x", 0), xy.get("y", 0)
            game_id = cur.get("game_id", "unknown")
            changed = cur["frame"] != nxt["frame"]
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], changed, game_id))
    assert len(transitions) == 12000, f"expected 12000 transitions, found {len(transitions)}"
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, num_games: int, device):
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
def diagnostics_a_c_d(online, predictor, game_vocab, loader, device):
    """A: encoder change-sensitivity. C: predictor residual commitment.
    D: predictor action-sensitivity (does varying the action change the
    prediction, holding the state fixed?)."""
    changed_deltas, unchanged_deltas = [], []
    residual_at_changed, true_delta_at_changed = [], []
    action_sensitivity, true_action_residual = [], []

    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, nxt = cur.to(device), nxt.to(device)
        action_id, xy, game_idx = action_id.to(device), xy.to(device), game_idx.to(device)
        patch_mask = patch_mask.to(device)  # (B, 8, 8) bool

        cur_feat = online(cur)   # (B, C, 8, 8)
        nxt_feat = online(nxt)
        true_delta = nxt_feat - cur_feat  # what the residual SHOULD approximate

        # --- A: per-patch feature delta at changed vs unchanged patches ---
        per_patch_delta_norm = true_delta.pow(2).mean(dim=1)  # (B, 8, 8)
        changed_deltas.append(per_patch_delta_norm[patch_mask].cpu())
        unchanged_deltas.append(per_patch_delta_norm[~patch_mask].cpu())

        # --- C: predictor's actual residual vs the true delta, at changed patches ---
        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        residual = pred_feat - cur_feat
        residual_norm = residual.pow(2).mean(dim=1)  # (B, 8, 8)
        residual_at_changed.append(residual_norm[patch_mask].cpu())
        true_delta_at_changed.append(per_patch_delta_norm[patch_mask].cpu())

        # --- D: action-sensitivity -- same state, sweep all 8 actions ---
        b = cur.shape[0]
        all_action_outputs = []
        for a in range(8):
            a_tensor = torch.full((b,), a, dtype=torch.long, device=device)
            out, _ = predictor(cur_feat, a_tensor, xy, game_idx)
            all_action_outputs.append(out)
        stacked = torch.stack(all_action_outputs, dim=0)  # (8, B, C, 8, 8)
        # variance ACROSS actions, per example, pooled over channels/space
        per_example_action_var = stacked.var(dim=0).mean(dim=(1, 2, 3))  # (B,)
        action_sensitivity.append(per_example_action_var.cpu())
        true_action_residual.append(residual_norm.mean(dim=(1, 2)).cpu())

    changed_deltas = torch.cat(changed_deltas)
    unchanged_deltas = torch.cat(unchanged_deltas)
    residual_at_changed = torch.cat(residual_at_changed)
    true_delta_at_changed = torch.cat(true_delta_at_changed)
    action_sensitivity = torch.cat(action_sensitivity)
    true_action_residual = torch.cat(true_action_residual)

    return {
        "A_changed_delta_mean": changed_deltas.mean().item(),
        "A_unchanged_delta_mean": unchanged_deltas.mean().item(),
        "A_ratio": (changed_deltas.mean() / unchanged_deltas.mean().clamp(min=1e-12)).item(),
        "C_residual_mean_at_changed": residual_at_changed.mean().item(),
        "C_true_delta_mean_at_changed": true_delta_at_changed.mean().item(),
        "C_commitment_ratio": (residual_at_changed.mean() / true_delta_at_changed.mean().clamp(min=1e-12)).item(),
        "D_action_variance_mean": action_sensitivity.mean().item(),
        "D_typical_residual_magnitude": true_action_residual.mean().item(),
        "D_sensitivity_ratio": (action_sensitivity.mean() / true_action_residual.mean().clamp(min=1e-12)).item(),
    }


@torch.no_grad()
def diagnostic_b_object_identity(online, device, games: list, n_frames: int, seed: int = 0):
    """For each of a few real frames: do same-color patches at DIFFERENT
    locations map to more similar encoder features than DIFFERENT-color
    patches do? Tests object-identity representation, not just
    sensitivity-to-change."""
    rng = random.Random(seed)
    same_type_sims, diff_type_sims = [], []

    for game in games:
        files = sorted((REPO_ROOT / "ARC-AGI-3-Agents" / "recordings").glob(f"{game}-*.recording.jsonl"))
        if not files:
            files = sorted(Path("E:/jepa_overflow/recordings").glob(f"{game}-*.recording.jsonl"))
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

            # per-8x8-patch dominant color
            patch_colors = {}
            for r in range(8):
                for c in range(8):
                    block = grid[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
                    vals, counts = np.unique(block, return_counts=True)
                    patch_colors[(r, c)] = int(vals[counts.argmax()])

            # background = the single most common color across all patches
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

            # same-color, different-location pairs
            for col, positions in by_color.items():
                if len(positions) < 2:
                    continue
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        same_type_sims.append(cos_sim(positions[i], positions[j]))

            # different-color pairs (control) -- sample to keep it bounded
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

    print("\nLoading verified 150-file / 12,000-transition random-policy corpus...")
    transitions = load_verified_transitions()
    game_vocab_full = sorted({t[6] for t in transitions})
    print(f"  {len(transitions)} transitions across {len(game_vocab_full)} games -- verified count matches.")

    random.Random(0).shuffle(transitions)
    sample = transitions[:SAMPLE_N]

    for name, ckpt_dir in CHECKPOINTS.items():
        print(f"\n{'=' * 70}\nCHECKPOINT: {name} ({ckpt_dir})\n{'=' * 70}")
        online, predictor, ckpt_game_vocab = load_moe_checkpoint(ckpt_dir, len(game_vocab_full), device)
        ds = TransitionDataset(sample, ckpt_game_vocab)
        loader = DataLoader(ds, batch_size=32, shuffle=False)

        acd = diagnostics_a_c_d(online, predictor, ckpt_game_vocab, loader, device)
        print(f"\n[A] Encoder change-sensitivity:")
        print(f"    feature-delta at CHANGED patches:   {acd['A_changed_delta_mean']:.6f}")
        print(f"    feature-delta at UNCHANGED patches: {acd['A_unchanged_delta_mean']:.6f}")
        print(f"    ratio (changed/unchanged):          {acd['A_ratio']:.2f}x")

        print(f"\n[C] Predictor residual commitment (at changed patches):")
        print(f"    predictor's residual magnitude:     {acd['C_residual_mean_at_changed']:.6f}")
        print(f"    true feature-delta magnitude:       {acd['C_true_delta_mean_at_changed']:.6f}")
        print(f"    commitment ratio (residual/true):   {acd['C_commitment_ratio']:.3f}  (1.0 = fully committing, 0 = predicting no change)")

        print(f"\n[D] Predictor action-sensitivity:")
        print(f"    variance across the 8 hypothetical actions: {acd['D_action_variance_mean']:.6f}")
        print(f"    typical residual magnitude (true action):   {acd['D_typical_residual_magnitude']:.6f}")
        print(f"    sensitivity ratio:                          {acd['D_sensitivity_ratio']:.3f}  (near 0 = predictor barely cares which action)")

        b = diagnostic_b_object_identity(online, device, PROBE_GAMES, N_PROBE_FRAMES)
        print(f"\n[B] Encoder object-identity (cosine similarity):")
        print(f"    same color, different location (n={b['B_same_type_n']}):  {b['B_same_type_mean_cos_sim']:.4f}" if b['B_same_type_mean_cos_sim'] is not None else "    (no same-color pairs found)")
        print(f"    different color, different location (n={b['B_diff_type_n']}): {b['B_diff_type_mean_cos_sim']:.4f}" if b['B_diff_type_mean_cos_sim'] is not None else "    (no diff-color pairs found)")
        if b['B_same_type_mean_cos_sim'] is not None and b['B_diff_type_mean_cos_sim'] is not None:
            gap = b['B_same_type_mean_cos_sim'] - b['B_diff_type_mean_cos_sim']
            print(f"    gap (same - different):             {gap:+.4f}  (positive = encoder distinguishes object identity)")


if __name__ == "__main__":
    main()
