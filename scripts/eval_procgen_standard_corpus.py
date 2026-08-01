"""Standard-corpus sanity check (trained-games, held-out-transitions, not
held-out-games) at full float precision -- the training log's 5-decimal
print rounds to 0.00000 for the Procgen checkpoint's collapsed-magnitude
numbers, making the epoch-51-60 trailing mean unreadable directly from
the log. Rebuilds the *exact* same ARC-3 train/val split
train_moe_predictor.py's own train() used (same exclude_games, same
external-per-game cap, same VAL_FRACTION/split seed) and reuses its own
evaluate() function directly against both final checkpoints.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch

from jepa.data.external_logs import load_external_transitions
from jepa.data.trajectories import load_all_transitions
from jepa.device import get_device
from jepa.models import CNNEncoder, MoEPredictor
from jepa.train_moe_predictor import _make_loaders, evaluate

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]

device = get_device()
arc_transitions = load_all_transitions(REPO_ROOT, exclude_games=EXCLUDE_GAMES)
external = load_external_transitions(REPO_ROOT, max_per_game=2000, exclude_games=EXCLUDE_GAMES)
arc_transitions += external
print(f"total ARC-3 transitions (local+external, 20 trained games): {len(arc_transitions)}")

for name, ckpt_dir in [("baseline", "checkpoints_diverse_baseline"), ("procgen", "checkpoints_diverse_procgen")]:
    ckpt_dir = REPO_ROOT / ckpt_dir
    game_vocab = json.loads((ckpt_dir / "game_vocab_moe.json").read_text())
    meta = json.loads((ckpt_dir / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)

    _train_loader, val_loader = _make_loaders(arc_transitions, game_vocab, batch_size=32, device=device)

    online = CNNEncoder(out_channels=64).to(device)
    online.load_state_dict(torch.load(ckpt_dir / "encoder_moe.pt", map_location=device))
    predictor = MoEPredictor(num_games=len(game_vocab), num_experts=num_experts).to(device)
    predictor.load_state_dict(torch.load(ckpt_dir / "moe_predictor.pt", map_location=device))

    stats = evaluate(online, predictor, val_loader, device=device)
    improvement = (stats["identity_changed"] - stats["pred_changed"]) / stats["identity_changed"] * 100
    print(f"\n{name}:")
    print(f"  pred_changed_mse     = {stats['pred_changed']:.8e}")
    print(f"  identity_changed_mse = {stats['identity_changed']:.8e}")
    print(f"  improvement over identity: {improvement:+.2f}%")
    print(f"  (whole-grid) pred_mse={stats['pred']:.8e}  identity_mse={stats['identity']:.8e}")
