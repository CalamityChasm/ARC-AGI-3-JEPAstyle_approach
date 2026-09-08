"""Pretrains jepa/click_effect_model.py's ClickEffectModel on
scripts/harvest_click_effect_data.py's output.

Leave-games-out cross-validation (GroupKFold by game), same discipline as
scripts/train_graph_win_distance_model.py and this project's whole Stage 6
history -- a same-game split would look deceptively good and say nothing
about generalization to a game the model didn't train on. This is the
frozen, zero-shot-transfer number; the live agent additionally test-time-
adapts per game on top of whatever this checkpoint starts with (see
jepa/click_effect_adapter.py) -- CV here answers "how good is the starting
point," not "how good is the deployed system."

win=1 examples are rare (~2% overall, scripts/harvest_click_effect_data.py's
own printed rates) -- oversampled via a WeightedRandomSampler on the
training split only, same NONZERO_WEIGHT-style pattern Stage 5's value
head training used (jepa/train_value_head.py), for the same reason:
an unweighted loss would be dominated by the trivial majority and learn
nothing about the rare, meaningful class.

Usage:
    python scripts/train_click_effect_model.py --data data/click_effect_dataset.npz --out checkpoints_click_effect
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jepa.click_effect_model import ClickEffectModel  # noqa: E402
from jepa.device import get_device  # noqa: E402

WIN_OVERSAMPLE_WEIGHT = 20.0


def make_loader(patches, seg_feats, changed, win, batch_size, weighted: bool):
    patches_t = torch.from_numpy(patches).long()
    seg_t = torch.from_numpy(seg_feats).float()
    changed_t = torch.from_numpy(changed).float()
    win_t = torch.from_numpy(win).float()
    dataset = torch.utils.data.TensorDataset(patches_t, seg_t, changed_t, win_t)
    if weighted:
        weights = np.where(win > 0.5, WIN_OVERSAMPLE_WEIGHT, 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_one_model(train_idx, patches, seg_feats, changed, win, device, epochs=30, lr=1e-3, batch_size=64):
    model = ClickEffectModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = make_loader(patches[train_idx], seg_feats[train_idx], changed[train_idx], win[train_idx],
                          batch_size, weighted=True)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        for patch_b, seg_b, changed_b, win_b in loader:
            patch_b, seg_b = patch_b.to(device), seg_b.to(device)
            changed_b, win_b = changed_b.to(device), win_b.to(device)
            changed_logit, win_logit = model(patch_b, seg_b)
            loss = bce(changed_logit, changed_b) + bce(win_logit, win_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, idx, patches, seg_feats, changed, win, device):
    if len(idx) == 0:
        return {}
    patch_t = torch.from_numpy(patches[idx]).long().to(device)
    seg_t = torch.from_numpy(seg_feats[idx]).float().to(device)
    changed_logit, win_logit = model(patch_t, seg_t)
    changed_pred = torch.sigmoid(changed_logit).cpu().numpy()
    win_pred = torch.sigmoid(win_logit).cpu().numpy()
    y_changed, y_win = changed[idx], win[idx]

    result = {"n": len(idx)}
    if len(set(y_changed.tolist())) > 1:
        result["changed_auc"] = roc_auc_score(y_changed, changed_pred)
    else:
        result["changed_auc"] = None
    if len(set(y_win.tolist())) > 1:
        result["win_auc"] = roc_auc_score(y_win, win_pred)
    else:
        result["win_auc"] = None
    result["changed_acc"] = float(((changed_pred > 0.5) == (y_changed > 0.5)).mean())
    result["baseline_changed_acc"] = float(max(y_changed.mean(), 1 - y_changed.mean()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    data = np.load(args.data, allow_pickle=True)
    patches, seg_feats, changed, win, games = (
        data["patches"], data["seg_feats"], data["changed"], data["win"], data["games"]
    )
    print(f"loaded {len(changed)} rows across {len(set(games.tolist()))} games")

    n_groups = len(set(games.tolist()))
    n_folds = min(args.n_folds, n_groups)
    fold_results = []
    if n_folds >= 2:
        gkf = GroupKFold(n_splits=n_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(patches, changed, games)):
            model = train_one_model(train_idx, patches, seg_feats, changed, win, device, epochs=args.epochs)
            metrics = evaluate(model, test_idx, patches, seg_feats, changed, win, device)
            test_games = sorted(set(games[test_idx].tolist()))
            fold_results.append({"fold": fold_idx, "test_games": test_games, **metrics})
            print(f"fold {fold_idx} (games={test_games}): {metrics}")

        valid_aucs = [r["changed_auc"] for r in fold_results if r["changed_auc"] is not None]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else None
        mean_acc = float(np.mean([r["changed_acc"] for r in fold_results]))
        mean_baseline_acc = float(np.mean([r["baseline_changed_acc"] for r in fold_results]))
        print(f"\n=== leave-games-out CV summary ({n_folds} folds) ===")
        print(f"mean frame_changed AUC (held-out games): {mean_auc}")
        print(f"mean frame_changed accuracy: {mean_acc:.3f}  (majority-class baseline: {mean_baseline_acc:.3f})")

    # Final model trained on ALL data, for pretraining the live checkpoint
    # (the live agent additionally test-time-adapts per game on top of this).
    final_idx = np.arange(len(changed))
    final_model = train_one_model(final_idx, patches, seg_feats, changed, win, device, epochs=args.epochs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), out_dir / "click_effect_model.pt")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train_rows": len(changed),
            "n_train_games": n_groups,
            "cv_fold_results": fold_results,
            "overall_frame_changed_rate": float(changed.mean()),
            "overall_win_rate": float(win.mean()),
        }, f, indent=2)
    print(f"\nsaved model + meta to {out_dir}")


if __name__ == "__main__":
    main()
