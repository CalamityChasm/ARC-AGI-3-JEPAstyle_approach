"""Sweeps jepa/click_effect_adapter.py: ClickEffectAdapter's dose
(k/n_steps/lr) the same way this project's own JEPA-side test-time
adaptation was tuned (jepa/test_time_adapter.py's own sweep, see
experiments/stage6_test_time_adaptation.md) -- never swept before now,
the current defaults (k=10, n_steps=3, lr=5e-4) were just a reasonable
starting guess.

Methodology, leave-games-out + within-game adapt/eval split:
  1. Pretrain the base model on all games EXCEPT the held-out one (same
     as scripts/train_click_effect_model.py's own CV fold training).
  2. Split the held-out game's own data (already in natural, chronological
     append order from the harvest) into an "adapt stream" (first ~70%)
     and an "eval window" (last ~30%, never adapted on).
  3. For each candidate dose, clone the frozen base model, stream the
     adapt window through a ClickEffectAdapter with that dose, then
     evaluate the (now-adapted) model's AUC on the eval window -- compared
     directly against the frozen (no-adaptation) baseline on the same
     eval window.

Only games with enough rows for a meaningful eval split are included
(>= MIN_GAME_ROWS) -- the smallest local games don't have enough data for
this to mean anything, same reasoning as everywhere else in this project.

Usage:
    python scripts/sweep_click_effect_tta.py --data data/click_effect_dataset.npz
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jepa.click_effect_model import ClickEffectModel, get_adapter_params  # noqa: E402
from jepa.click_effect_adapter import ClickEffectAdapter  # noqa: E402
from jepa.device import get_device  # noqa: E402

MIN_GAME_ROWS = 60
ADAPT_FRACTION = 0.7
WIN_OVERSAMPLE_WEIGHT = 20.0

DOSES = {
    "off": None,
    "conservative": dict(k=20, n_steps=2, lr=1e-4),
    "default": dict(k=10, n_steps=3, lr=5e-4),
    "aggressive": dict(k=5, n_steps=8, lr=2e-3),
    "very_aggressive": dict(k=3, n_steps=15, lr=5e-3),
    "extreme": dict(k=2, n_steps=25, lr=1e-2),
    "max": dict(k=1, n_steps=40, lr=2e-2),
}


def train_base_model(train_idx, patches, seg_feats, changed, win, device, epochs=30, lr=1e-3, batch_size=64):
    model = ClickEffectModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    patches_t = torch.from_numpy(patches[train_idx]).long()
    seg_t = torch.from_numpy(seg_feats[train_idx]).float()
    changed_t = torch.from_numpy(changed[train_idx]).float()
    win_t = torch.from_numpy(win[train_idx]).float()
    dataset = torch.utils.data.TensorDataset(patches_t, seg_t, changed_t, win_t)
    weights = np.where(win[train_idx] > 0.5, WIN_OVERSAMPLE_WEIGHT, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
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
def eval_auc(model, idx, patches, seg_feats, changed, device):
    if len(idx) == 0 or len(set(changed[idx].tolist())) < 2:
        return None
    patch_t = torch.from_numpy(patches[idx]).long().to(device)
    seg_t = torch.from_numpy(seg_feats[idx]).float().to(device)
    changed_logit, _ = model(patch_t, seg_t)
    pred = torch.sigmoid(changed_logit).cpu().numpy()
    return roc_auc_score(changed[idx], pred)


def run_adaptation(base_model, dose_cfg, adapt_idx, patches, seg_feats, changed, win, device):
    model = copy.deepcopy(base_model)
    adapter = ClickEffectAdapter(model, device, k=dose_cfg["k"], n_steps=dose_cfg["n_steps"],
                                 lr=dose_cfg["lr"], min_buffer_for_adapt=min(6, max(2, len(adapt_idx) // 4)))
    for i in adapt_idx:
        adapter.buffer.append((patches[i], seg_feats[i], float(changed[i]), float(win[i])))
        adapter._n_observed += 1
        if len(adapter.buffer) >= adapter.min_buffer_for_adapt and adapter._n_observed % adapter.k == 0:
            adapter._adapt_step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}")

    data = np.load(args.data, allow_pickle=True)
    patches, seg_feats, changed, win, games = (
        data["patches"], data["seg_feats"], data["changed"], data["win"], data["games"]
    )
    game_list = games.tolist()
    unique_games = sorted(set(game_list))
    eligible_games = [g for g in unique_games if game_list.count(g) >= MIN_GAME_ROWS]
    print(f"eligible games (>= {MIN_GAME_ROWS} rows): {eligible_games}")

    results = {dose_name: [] for dose_name in DOSES}

    for held_out in eligible_games:
        held_out_idx = np.array([i for i, g in enumerate(game_list) if g == held_out])
        train_idx = np.array([i for i, g in enumerate(game_list) if g != held_out])

        n_adapt = int(len(held_out_idx) * ADAPT_FRACTION)
        adapt_idx = held_out_idx[:n_adapt]
        eval_idx = held_out_idx[n_adapt:]
        if len(eval_idx) < 10 or len(set(changed[eval_idx].tolist())) < 2:
            print(f"[{held_out}] skipping -- eval window too small or single-class ({len(eval_idx)} rows)")
            continue

        base_model = train_base_model(train_idx, patches, seg_feats, changed, win, device, epochs=args.pretrain_epochs)
        frozen_auc = eval_auc(base_model, eval_idx, patches, seg_feats, changed, device)
        print(f"[{held_out}] n_adapt={len(adapt_idx)} n_eval={len(eval_idx)} frozen_auc={frozen_auc}")
        results["off"].append(frozen_auc)

        for dose_name, dose_cfg in DOSES.items():
            if dose_cfg is None:
                continue
            adapted_model = run_adaptation(base_model, dose_cfg, adapt_idx, patches, seg_feats, changed, win, device)
            adapted_auc = eval_auc(adapted_model, eval_idx, patches, seg_feats, changed, device)
            results[dose_name].append(adapted_auc)
            print(f"  dose={dose_name:16s} adapted_auc={adapted_auc}")

    print("\n=== pooled results across held-out games ===")
    summary = {}
    for dose_name, aucs in results.items():
        valid = [a for a in aucs if a is not None]
        mean_auc = float(np.mean(valid)) if valid else None
        summary[dose_name] = {"mean_auc": mean_auc, "n_games": len(valid), "per_game": aucs}
        print(f"{dose_name:16s} mean_auc={mean_auc} (n={len(valid)} games)")

    out = Path("data/click_effect_tta_sweep.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
