"""Stage 6 residual-commitment-fix, step 3 (mechanistic check): did the
new residual-commitment loss actually raise the predictor's residual-
commitment ratio on HELD-OUT games specifically -- the exact diagnostic-C
metric that motivated this whole fix (stage6-encoder-holdout-diag: ratio
collapses to ~0.000 on held-out games vs 0.235/0.010 on trained games for
the pre-fix checkpoints)?

Reuses the identical methodology/data as
scripts/diagnose_encoder_holdout_predictor_check.py, applied to the new
checkpoints_holdout_rescommit checkpoint alongside the two pre-fix
reference checkpoints for direct comparison.

Usage:
    python scripts/diagnose_rescommit_check.py
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

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
    "residual-commitment-fix": REPO_ROOT / "checkpoints_holdout_rescommit",
    "baseline-holdout": REPO_ROOT / "checkpoints_holdout_baseline",
    "object-identity-holdout": REPO_ROOT / "checkpoints_holdout_objid",
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
def diagnostic_c(online, predictor, game_vocab, transitions: list, device) -> dict:
    fallback_vocab = defaultdict(int, game_vocab)
    ds = TransitionDataset(transitions, fallback_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    residual_at_changed, true_delta_at_changed = [], []
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, nxt = cur.to(device), nxt.to(device)
        action_id, xy, game_idx = action_id.to(device), xy.to(device), game_idx.to(device)
        patch_mask = patch_mask.to(device)

        cur_feat = online(cur)
        nxt_feat = online(nxt)
        true_delta = (nxt_feat - cur_feat).pow(2).mean(dim=1)

        pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
        residual = pred_feat - cur_feat
        residual_norm = residual.pow(2).mean(dim=1)

        residual_at_changed.append(residual_norm[patch_mask].cpu())
        true_delta_at_changed.append(true_delta[patch_mask].cpu())

    residual_at_changed = torch.cat(residual_at_changed)
    true_delta_at_changed = torch.cat(true_delta_at_changed)
    return {
        "n_changed_patches": residual_at_changed.numel(),
        "residual_mean": residual_at_changed.mean().item(),
        "true_delta_mean": true_delta_at_changed.mean().item(),
        "commitment_ratio": (residual_at_changed.mean() / true_delta_at_changed.mean().clamp(min=1e-12)).item(),
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
            print(f"\nSKIPPING {name}: {ckpt_dir} does not exist")
            continue
        print(f"\n{'=' * 70}\nCHECKPOINT: {name}\n{'=' * 70}")
        online, predictor, game_vocab = load_moe_checkpoint(ckpt_dir, device)

        c_heldout = diagnostic_c(online, predictor, game_vocab, heldout_transitions, device)
        print(f"\n[diagnostic C] predictor residual commitment on HELD-OUT games (n={c_heldout['n_changed_patches']}):")
        print(f"    residual mean: {c_heldout['residual_mean']:.6f}  true-delta mean: {c_heldout['true_delta_mean']:.6f}")
        print(f"    commitment ratio (residual/true): {c_heldout['commitment_ratio']:.3f}  (1.0=fully committing, 0=predicts no change)")

        c_trained = diagnostic_c(online, predictor, game_vocab, trained_sample, device)
        print(f"\n[diagnostic C] predictor residual commitment on TRAINED games (n={c_trained['n_changed_patches']}):")
        print(f"    residual mean: {c_trained['residual_mean']:.6f}  true-delta mean: {c_trained['true_delta_mean']:.6f}")
        print(f"    commitment ratio (residual/true): {c_trained['commitment_ratio']:.3f}")

        results[name] = {"heldout": c_heldout, "trained": c_trained}

    out_path = REPO_ROOT / "logs" / "rescommit_diagnostic_c_check.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
