"""Follow-up sanity checks for diagnose_encoder_holdout.py's surprising
result (ratio does NOT collapse on held-out games -- it's actually
LARGER: 80x/48x held-out vs 6x/30x trained). Before taking that at face
value, rule out the obvious confound: the encoder producing much
larger-magnitude (possibly unstable/OOD-blown-up) features on unfamiliar
games in general, which would inflate both the changed AND unchanged
deltas by roughly the same multiplicative factor without reflecting any
real "sensitivity to change" property. Also breaks the held-out ratio
down per-game (5 games) to check the result isn't driven by one outlier
game, and checks the raw pixel-level change magnitude per changed patch
(held-out vs trained) as a candidate innocent explanation for scale
differences.

Usage:
    python scripts/diagnose_encoder_holdout_followup.py
"""

import json
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
from jepa.models import CNNEncoder, MoEPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
HELDOUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

CHECKPOINTS = {
    "baseline-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_baseline"
    ),
    "object-identity-holdout": Path(
        "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
        "agent-a0f09770086c096a6/checkpoints_holdout_objid"
    ),
}
TRAINED_SAMPLE_N = 2000


def load_verified_transitions() -> list:
    transitions = []
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            action = nxt["action_input"]
            action_id = action["id"]
            xy = action.get("data", {}) or {}
            x, y = xy.get("x", 0), xy.get("y", 0)
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], action_id, x, y, nxt["frame"], cur["frame"] != nxt["frame"], game_id))
    assert len(transitions) == 12000
    return transitions


def load_moe_checkpoint(checkpoint_dir: Path, device):
    game_vocab = json.loads((checkpoint_dir / "game_vocab_moe.json").read_text())
    feature_channels = 64
    meta_path = checkpoint_dir / "moe_training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        feature_channels = meta.get("feature_channels", 64)
    online = CNNEncoder(out_channels=feature_channels).to(device)
    online.load_state_dict(torch.load(checkpoint_dir / "encoder_moe.pt", map_location=device))
    online.eval()
    return online, game_vocab


@torch.no_grad()
def analyze(online, game_vocab, transitions: list, device, label: str):
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    feat_norms = []
    changed_deltas, unchanged_deltas = [], []
    pixel_diff_counts_changed = []  # how many of the 64 pixels in a changed patch actually differ

    idx = 0
    for cur, _action_id, _xy, nxt, patch_mask, _game_idx in loader:
        b = cur.shape[0]
        cur_d, nxt_d = cur.to(device), nxt.to(device)
        patch_mask_d = patch_mask.to(device)

        cur_feat = online(cur_d)
        nxt_feat = online(nxt_d)
        feat_norms.append(cur_feat.pow(2).mean(dim=1).cpu().flatten())  # per-patch squared norm

        true_delta = (nxt_feat - cur_feat).pow(2).mean(dim=1)
        changed_deltas.append(true_delta[patch_mask_d].cpu())
        unchanged_deltas.append(true_delta[~patch_mask_d].cpu())

        # raw pixel-level diff count within each changed patch (via cur/nxt tensors, one-hot -> argmax color)
        cur_color = cur_d.argmax(dim=1)  # (B, 64, 64)
        nxt_color = nxt_d.argmax(dim=1)
        diff = (cur_color != nxt_color).float()  # (B, 64, 64)
        # pool into 8x8 patches, count differing pixels per patch
        diff_patches = diff.view(b, 8, 8, 8, 8).sum(dim=(2, 4))  # (B, 8, 8) count out of 64 pixels
        pixel_diff_counts_changed.append(diff_patches[patch_mask_d].cpu())

        idx += b

    feat_norms = torch.cat(feat_norms)
    changed_deltas = torch.cat(changed_deltas)
    unchanged_deltas = torch.cat(unchanged_deltas)
    pixel_diff_counts_changed = torch.cat(pixel_diff_counts_changed)

    print(f"  [{label}] n_transitions={len(transitions)}")
    print(f"    mean per-patch feature squared-norm (||cur_feat||^2, all patches): {feat_norms.mean().item():.6f}")
    print(f"    changed-patch mean delta: {changed_deltas.mean().item():.6f}  (n={changed_deltas.numel()})")
    print(f"    unchanged-patch mean delta: {unchanged_deltas.mean().item():.6f}  (n={unchanged_deltas.numel()})")
    print(f"    ratio: {(changed_deltas.mean() / unchanged_deltas.mean().clamp(min=1e-12)).item():.2f}x")
    print(f"    mean pixels-differing per changed patch (out of 64): {pixel_diff_counts_changed.mean().item():.2f}")
    return {
        "feat_norm_mean": feat_norms.mean().item(),
        "changed_delta_mean": changed_deltas.mean().item(),
        "unchanged_delta_mean": unchanged_deltas.mean().item(),
        "ratio": (changed_deltas.mean() / unchanged_deltas.mean().clamp(min=1e-12)).item(),
        "mean_pixels_differing_per_changed_patch": pixel_diff_counts_changed.mean().item(),
    }


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    all_transitions = load_verified_transitions()
    heldout_prefixes = tuple(f"{g}-" for g in HELDOUT_GAMES)
    heldout_transitions = [t for t in all_transitions if t[6].startswith(heldout_prefixes)]
    trained_transitions_full = [t for t in all_transitions if not t[6].startswith(heldout_prefixes)]
    random.Random(0).shuffle(trained_transitions_full)
    trained_sample = trained_transitions_full[:TRAINED_SAMPLE_N]

    results = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        if not ckpt_dir.exists():
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name}\n{'=' * 70}")
        online, game_vocab = load_moe_checkpoint(ckpt_dir, device)

        print("\nOverall (feature-norm-scale sanity check):")
        heldout_stats = analyze(online, game_vocab, heldout_transitions, device, "held-out (5 games pooled)")
        trained_stats = analyze(online, game_vocab, trained_sample, device, "trained (20-game sample)")

        print("\nPer-held-out-game diagnostic A breakdown:")
        per_game = {}
        for g in HELDOUT_GAMES:
            g_transitions = [t for t in heldout_transitions if t[6].startswith(f"{g}-")]
            stats = analyze(online, game_vocab, g_transitions, device, g)
            per_game[g] = stats

        results[name] = {"heldout_overall": heldout_stats, "trained_overall": trained_stats, "per_heldout_game": per_game}

    out_path = REPO_ROOT / "logs" / "encoder_holdout_diagnostic_a_followup.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
