"""Stage 6 task: does fixing the production MoEPredictor's residual
under-commitment (see diagnose_production_residual_commitment.py --
baseline ratio 0.054 at changed patches) reduce multi-step rollout
compounding error (stage6_rollout_compounding_error.md)?

This is step 2 of that investigation: a short, targeted continued-training
run, loading the TRUE production checkpoint's weights as a warm start and
training further, concentrated on real local ARC-3 data (not a full
retrain, not touching MiniGrid/external corpora again) -- mirrors the
"mostly-real data mix, moderate epoch count" recipe used for the earlier
scaled-architecture residual fine-tune. Uses the exact same loss/eval
machinery as jepa/train_moe_predictor.py's arc-finetune phase (imported
directly, not reimplemented) so this is a fair continuation of the same
training objective, not a different recipe.

CRITICAL: writes ONLY to --out (default checkpoints_residual_finetune/ at
repo root) -- never touches checkpoints/, per this task's standing
instruction not to risk repeating the checkpoint-substitution incident
documented in CLAUDE.md.

Usage:
    python -m scripts.finetune_production_residual --epochs 20
"""

import argparse
import json
from pathlib import Path

import torch

from jepa.data.trajectories import load_all_transitions
from jepa.device import get_device
from jepa.models import CNNEncoder, MoEPredictor
from jepa.train_moe_predictor import _make_loaders, _run_epochs

REPO_ROOT = Path(__file__).resolve().parent.parent
PROD_CKPT_DIR = REPO_ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4, help="Lower than the original 3e-4 -- this is a continued fine-tune, not from-scratch training.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints_residual_finetune")
    args = parser.parse_args()

    device = get_device()
    print(f"training on {device}")

    vocab = json.loads((PROD_CKPT_DIR / "game_vocab_moe.json").read_text())
    print(f"warm-starting from {PROD_CKPT_DIR}, {len(vocab)}-entry vocab (unchanged -- same games, no new embeddings)")

    online = CNNEncoder().to(device)
    online.load_state_dict(torch.load(PROD_CKPT_DIR / "encoder_moe.pt", map_location=device))
    from jepa.models import make_ema_target
    target = make_ema_target(online)
    predictor = MoEPredictor(num_games=len(vocab)).to(device)
    predictor.load_state_dict(torch.load(PROD_CKPT_DIR / "moe_predictor.pt", map_location=device))
    print("warm-started encoder + MoE predictor from true production checkpoint")

    arc_transitions = load_all_transitions(REPO_ROOT)
    print(f"loaded {len(arc_transitions)} local ARC-3 transitions (real data only, no MiniGrid re-pretrain)")

    train_loader, val_loader = _make_loaders(arc_transitions, vocab, args.batch_size, device)

    opt = torch.optim.AdamW(list(online.parameters()) + list(predictor.parameters()), lr=args.lr)
    _run_epochs(online, target, predictor, opt, train_loader, val_loader, device, args.epochs, "residual-finetune")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in online.state_dict().items()}, args.out / "encoder_moe.pt")
    torch.save({k: v.cpu() for k, v in predictor.state_dict().items()}, args.out / "moe_predictor.pt")
    (args.out / "game_vocab_moe.json").write_text(json.dumps(vocab, indent=2))
    (args.out / "moe_training_meta.json").write_text(
        json.dumps(
            {
                "note": "continued-training fine-tune from true production checkpoint, real local ARC-3 data only",
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "n_local_transitions": len(arc_transitions),
                "n_games": len(vocab),
                "device": str(device),
                "warm_started_from": str(PROD_CKPT_DIR),
            },
            indent=2,
        )
    )
    print(f"saved fine-tuned checkpoint to {args.out}")


if __name__ == "__main__":
    main()
