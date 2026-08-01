"""Quick diagnostic: does the Procgen-pretrained checkpoint's encoder show
classic representation collapse (near-constant features regardless of
input), the way val_identity_mse collapsing to ~0.00001 during training
suggests? Mirrors Stage 1 item 7's VARIANCE_FLOOR check."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from jepa.data.trajectories import TransitionDataset, load_all_transitions
from jepa.device import get_device
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
    ds = TransitionDataset(transitions[:512], game_vocab)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    cur, action_id, xy, nxt, patch_mask, game_idx = next(iter(loader))
    cur = cur.to(device)
    with torch.no_grad():
        feat = online(cur)
    std_per_channel = feat.std(dim=(0, 2, 3))
    print(f"{name}: feat shape={tuple(feat.shape)}  mean std/channel={std_per_channel.mean().item():.5f}  "
          f"min={std_per_channel.min().item():.5f}  max={std_per_channel.max().item():.5f}  "
          f"overall_std={feat.std().item():.5f}  overall_mean_abs={feat.abs().mean().item():.5f}")
