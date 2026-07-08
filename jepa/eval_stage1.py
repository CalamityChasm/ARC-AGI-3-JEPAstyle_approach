"""Stage 1 milestone check (plan.md): does the trained predictor beat the
"nothing changes" identity baseline on held-out transitions -- specifically
on the patches that actually changed, not just in an aggregate dominated by
static ones -- and does the per-region prediction error look like a sane
salience map?

Usage: python -m jepa.eval_stage1
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split

from .data.trajectories import TransitionDataset, load_all_transitions
from .losses import per_region_error
from .models import ActionConditionedPredictor, CNNEncoder
from .train_predictor import evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
VAL_FRACTION = 0.1


def main() -> None:
    transitions = load_all_transitions(REPO_ROOT)
    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab.json").read_text())
    dataset = TransitionDataset(transitions, game_vocab)
    n_val = max(1, int(len(dataset) * VAL_FRACTION))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    online = CNNEncoder()
    online.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_finetuned.pt", map_location="cpu"))
    online.eval()
    predictor = ActionConditionedPredictor(num_games=len(game_vocab))
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "predictor.pt", map_location="cpu"))
    predictor.eval()

    stats = evaluate(online, predictor, val_loader)

    print(f"overall predictor MSE          : {stats['pred']:.5f}")
    print(f"overall identity-baseline MSE  : {stats['identity']:.5f}")
    overall_improvement = (stats["identity"] - stats["pred"]) / stats["identity"] * 100
    print(f"overall improvement            : {overall_improvement:+.1f}%")
    print()
    print(f"changed-patches predictor MSE  : {stats['pred_changed']:.5f}")
    print(f"changed-patches identity MSE   : {stats['identity_changed']:.5f}")
    changed_improvement = (
        (stats["identity_changed"] - stats["pred_changed"]) / stats["identity_changed"] * 100
    )
    print(f"changed-patches improvement    : {changed_improvement:+.1f}%")
    print()
    if stats["pred_changed"] < stats["identity_changed"]:
        print("PASS: predictor beats identity specifically on patches that actually changed.")
    else:
        print("FAIL: predictor does not beat identity on patches that actually changed.")

    _render_salience_example(online, predictor, val_ds)


def _render_salience_example(online, predictor, val_ds) -> None:
    # Pick a transition with the most changed patches to make for a
    # legible example (most transitions are near-total no-ops).
    best_idx, best_count = 0, -1
    for i in range(len(val_ds)):
        *_ignore, patch_mask, _game_idx = val_ds[i]
        count = int(patch_mask.sum())
        if count > best_count:
            best_idx, best_count = i, count

    cur, action_id, xy, nxt, patch_mask, game_idx = val_ds[best_idx]
    with torch.no_grad():
        cur_feat = online(cur.unsqueeze(0))
        pred_feat = predictor(cur_feat, action_id.unsqueeze(0), xy.unsqueeze(0), game_idx.unsqueeze(0))
        next_feat = online(nxt.unsqueeze(0))
        error_map = per_region_error(pred_feat, next_feat)[0].numpy()  # (8, 8)

    heat = (error_map - error_map.min()) / (np.ptp(error_map) + 1e-8)
    heat_img = (heat * 255).astype(np.uint8)
    img = Image.fromarray(heat_img).resize((256, 256), Image.NEAREST)
    out_path = CHECKPOINT_DIR / "salience_example.png"
    img.save(out_path)
    print(
        f"saved salience map for a {best_count}/64-patch-changed example to {out_path}"
    )


if __name__ == "__main__":
    main()
