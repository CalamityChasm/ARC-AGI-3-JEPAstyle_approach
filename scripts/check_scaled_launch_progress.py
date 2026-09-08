"""Standalone, read-only progress check for the in-flight scaled-curriculum
launch (E:/jepa_overflow/checkpoints_scaled_launch/). Loads the CURRENT
checkpoint on CPU (never CUDA -- the live training process is actively
using most of the card's VRAM, and a second process competing for it risks
crashing a 40-hour unattended run over a status check) and computes a real
changed-patches-vs-identity number on the arc3_local held-out split, the
same metric this project has used as its bar since Stage 1.

Exists because the training process's own stdout never got flushed to its
log file (block-buffered when redirected to a file, not a TTY) -- this is
the only way to get a real number mid-run without touching the live
process or its GPU memory.

Usage: python scripts/check_scaled_launch_progress.py
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import TransitionDataset
from jepa.losses import per_region_error, prediction_loss
from jepa.models.encoder_scaled import ScaledCNNEncoder
from jepa.models.moe_predictor_scaled import ScaledMoEPredictor
from jepa.train_scaled_curriculum import build_game_vocab
from torch.utils.data import DataLoader

CKPT_DIR = Path("E:/jepa_overflow/checkpoints_scaled_launch")
VAL_FRACTION = 0.1


def main() -> None:
    device = torch.device("cpu")

    meta = json.loads((CKPT_DIR / "curriculum_meta.json").read_text())
    args = meta["args"]
    print(f"checkpoint: completed_epochs={meta['completed_epochs']}/{meta['total_epochs']}")

    t0 = time.time()
    with open(CKPT_DIR / "corpus_cache.pkl", "rb") as f:
        sources = pickle.load(f)
    print(f"loaded corpus cache ({time.time() - t0:.1f}s)")

    game_vocab = build_game_vocab(sources)

    # Same seeded val split the training script itself uses -- reconstructing
    # it here, not re-deriving a different one, so this is the SAME held-out
    # set the run's own eval would use.
    rng = np.random.default_rng(0)
    local_transitions, _cat = sources["arc3_local"]
    n_val = max(1, int(len(local_transitions) * VAL_FRACTION))
    idx = rng.permutation(len(local_transitions))[:n_val]
    val_transitions = [local_transitions[i] for i in idx]
    print(f"arc3_local held-out slice: {len(val_transitions)} transitions")

    encoder = ScaledCNNEncoder(
        out_channels=args["feature_channels"],
        width_mult=args["width_mult"],
        blocks_per_stage=args["blocks_per_stage"],
    ).to(device)
    predictor = ScaledMoEPredictor(
        feature_channels=encoder.out_channels,
        num_games=len(game_vocab),
        num_experts=args["num_experts"],
        expert_hidden=args["expert_hidden"],
        expert_depth=args["expert_depth"],
        top_k=args["top_k"],
    ).to(device)

    t0 = time.time()
    encoder.load_state_dict(torch.load(CKPT_DIR / "encoder_scaled.pt", map_location=device))
    predictor.load_state_dict(torch.load(CKPT_DIR / "moe_predictor_scaled.pt", map_location=device))
    print(f"loaded current checkpoint weights ({time.time() - t0:.1f}s)")

    encoder.eval()
    predictor.eval()

    dataset = TransitionDataset(val_transitions, game_vocab)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    # Mirrors jepa/train_scaled_curriculum.py's own evaluate() exactly --
    # including encoding frame_t+1 with the SAME (online) encoder as
    # frame_t, not a lagging EMA target. Getting this wrong reintroduces
    # the exact EMA-vs-online asymmetry bug Stage 1 already found and fixed
    # once (see CLAUDE.md's gotcha) -- an easy mistake to make writing a
    # fresh diagnostic script, so it's called out here rather than silently
    # done "the obvious way".
    totals = {"pred": 0.0, "identity": 0.0, "pred_changed": 0.0, "identity_changed": 0.0}
    n_batches = 0
    n_changed_batches = 0
    t0 = time.time()
    with torch.no_grad():
        for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
            cur_feat = encoder(cur)
            pred_feat, _gate = predictor(cur_feat, action_id, xy, game_idx)
            next_feat = encoder(nxt)

            totals["pred"] += prediction_loss(pred_feat, next_feat).item()
            totals["identity"] += prediction_loss(cur_feat, next_feat).item()
            n_batches += 1

            if patch_mask.any():
                pred_err = per_region_error(pred_feat, next_feat)[patch_mask]
                identity_err = per_region_error(cur_feat, next_feat)[patch_mask]
                totals["pred_changed"] += pred_err.mean().item()
                totals["identity_changed"] += identity_err.mean().item()
                n_changed_batches += 1
    print(f"evaluated {n_batches} batches on CPU ({time.time() - t0:.1f}s)")

    n_changed_batches = max(n_changed_batches, 1)
    pred_mse = totals["pred"] / n_batches
    identity_mse = totals["identity"] / n_batches
    pred_changed = totals["pred_changed"] / n_changed_batches
    identity_changed = totals["identity_changed"] / n_changed_batches
    improvement = (identity_changed - pred_changed) / identity_changed * 100 if identity_changed else float("nan")

    print("\n=== live snapshot, arc3_local held-out ===")
    print(f"val_pred_mse={pred_mse:.5f}  val_identity_mse={identity_mse:.5f}")
    print(f"changed-patches: pred={pred_changed:.5f} identity={identity_changed:.5f}")
    print(f"changed-patches improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
