"""Follow-up to diagnose_procgen_collapse.py: overall feature std wasn't
collapsed (batch-level variance is fine, even slightly higher than
baseline), yet identity_changed_mse (feature-space delta between frame_t
and frame_t+1 specifically at pixel-changed patches) is ~19x smaller in
absolute terms for the Procgen checkpoint than the baseline. Mirrors
Stage 1 item 8's diagnostic: compare mean feature-space delta at changed
vs. unchanged patches, for both checkpoints, to see whether the Procgen
checkpoint has lost *relative* temporal-change sensitivity (delta at
changed patches no longer much bigger than at unchanged ones) or just
shrunk everything proportionally (relative contrast preserved, absolute
scale smaller -- a much more benign difference)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from jepa.data.trajectories import TransitionDataset, load_all_transitions
from jepa.device import get_device
from jepa.losses import per_region_error
from jepa.models import CNNEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent
device = get_device()

transitions = load_all_transitions(REPO_ROOT, exclude_games=["r11l", "bp35", "m0r0", "tr87", "ka59"])

_baseline_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_diverse_baseline"
_procgen_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints_diverse_procgen"
for name, ckpt_dir in [("baseline", _baseline_dir), ("procgen", _procgen_dir)]:
    online = CNNEncoder(out_channels=64).to(device)
    online.load_state_dict(torch.load(REPO_ROOT / ckpt_dir / "encoder_moe.pt", map_location=device))
    online.eval()

    game_vocab = json.loads((REPO_ROOT / ckpt_dir / "game_vocab_moe.json").read_text())
    ds = TransitionDataset(transitions[:1024], game_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    changed_sum, changed_n = 0.0, 0
    unchanged_sum, unchanged_n = 0.0, 0
    for cur, action_id, xy, nxt, patch_mask, game_idx in loader:
        cur, nxt, patch_mask = cur.to(device), nxt.to(device), patch_mask.to(device)
        with torch.no_grad():
            cur_feat = online(cur)
            next_feat = online(nxt)
        err = per_region_error(cur_feat, next_feat)  # (B, 8, 8)
        changed_sum += err[patch_mask].sum().item()
        changed_n += patch_mask.sum().item()
        unchanged_sum += err[~patch_mask].sum().item()
        unchanged_n += (~patch_mask).sum().item()

    changed_mean = changed_sum / max(changed_n, 1)
    unchanged_mean = unchanged_sum / max(unchanged_n, 1)
    ratio = changed_mean / unchanged_mean if unchanged_mean > 0 else float("nan")
    print(f"{name}: changed_mean_delta={changed_mean:.6f} (n={changed_n})  "
          f"unchanged_mean_delta={unchanged_mean:.6f} (n={unchanged_n})  "
          f"ratio(changed/unchanged)={ratio:.2f}x")
