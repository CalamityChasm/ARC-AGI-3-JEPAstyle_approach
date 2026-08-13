"""Does the recurrent predictor correctly predict the depleting countdown
bar specifically, on a game it was never trained on (ka59, held out of
checkpoints_recurrent_holdout)?

Motivation: `experiments/stage6_bp35_ka59_mechanics.md` found ka59's
bottom row is a literal, deterministic countdown bar (one segment
disappears roughly every ~1.5 actions, in lockstep with the already-
documented 100-action-per-attempt cap) -- and the user's own observation:
this kind of mechanic (a visible action/turn counter) plausibly appears
in many ARC-3 games, not just ka59, so a world model that's actually
learning transferable structure -- rather than memorizing each game's
specific pixel content -- should predict IT well even zero-shot on a
held-out game, even though CLAUDE.md's Stage 6 addendum already
established the model has ~no edge over identity in aggregate on held-out
games. This test isolates the bar specifically from the rest of the
board to see whether the collapse is uniform (fails on everything,
including things that should transfer) or selective (fails on
game-specific content but still gets generic conventions like a
countdown bar).

The bar occupies pixel row 63 -- entirely inside patch row 7 (patch rows
span 8 pixel-rows each: 0-7, 8-15, ..., 56-63), the model's own native
spatial resolution via `per_region_error`'s (B, 8, 8) output. Compares
patch-row-7 prediction/identity error against the other 7 patch rows,
using a REAL accumulated hidden state (not zeroed per-transition) to
match how the model is actually used during play.

Usage:
    python scripts/diagnose_countdown_bar_prediction.py --game ka59
"""

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.sequences import load_all_episodes  # noqa: E402
from jepa.device import get_device  # noqa: E402
from jepa.grid import arc3_frame_to_tensor  # noqa: E402
from jepa.losses import per_region_error  # noqa: E402
from jepa.models import CNNEncoder, RecurrentActionConditionedPredictor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints_recurrent_holdout"
BAR_PATCH_ROW = 7  # pixel rows 56-63, contains ka59's bottom-row countdown bar


def load_checkpoint(device):
    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_recurrent.json").read_text())
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_recurrent.pt", map_location=device))
    encoder.eval()
    predictor = RecurrentActionConditionedPredictor(num_games=max(len(game_vocab), 1)).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "recurrent_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, game_vocab


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="ka59")
    args = parser.parse_args()

    device = get_device()
    encoder, predictor, game_vocab = load_checkpoint(device)
    print(f"Loaded {CHECKPOINT_DIR} (game_vocab has {len(game_vocab)} entries, "
          f"'{args.game}' in vocab: {args.game in game_vocab})")

    all_episodes = load_all_episodes(REPO_ROOT)
    episodes = [ep for ep in all_episodes if ep[0][6].split("-")[0] == args.game]
    print(f"{len(episodes)} episodes for {args.game}")
    if not episodes:
        print("No episodes found -- nothing to measure.")
        return

    bar_pred, bar_identity, bar_n = 0.0, 0.0, 0
    rest_pred, rest_identity, rest_n = 0.0, 0.0, 0

    for ep in episodes:
        hidden = predictor.init_hidden(1, device)
        for frame_t, action_id, x, y, frame_t1, _changed, game_id in ep:
            short = game_id.split("-")[0]
            game_idx = torch.tensor([game_vocab.get(short, 0)], dtype=torch.long, device=device)
            cur = torch.from_numpy(arc3_frame_to_tensor(frame_t)).unsqueeze(0).to(device)
            nxt = torch.from_numpy(arc3_frame_to_tensor(frame_t1)).unsqueeze(0).to(device)
            cur_feat = encoder(cur)
            next_feat = encoder(nxt)
            action_t = torch.tensor([action_id], dtype=torch.long, device=device)
            xy_t = torch.tensor([[x / 63.0, y / 63.0]], dtype=torch.float32, device=device)

            pred_feat, hidden = predictor(cur_feat, action_t, xy_t, hidden, game_idx)

            pred_err = per_region_error(pred_feat, next_feat)[0]  # (8, 8)
            identity_err = per_region_error(cur_feat, next_feat)[0]  # (8, 8)

            bar_pred += pred_err[BAR_PATCH_ROW].sum().item()
            bar_identity += identity_err[BAR_PATCH_ROW].sum().item()
            bar_n += pred_err[BAR_PATCH_ROW].numel()

            other_mask = torch.ones(8, dtype=torch.bool)
            other_mask[BAR_PATCH_ROW] = False
            rest_pred += pred_err[other_mask].sum().item()
            rest_identity += identity_err[other_mask].sum().item()
            rest_n += pred_err[other_mask].numel()

    bar_pred_mse = bar_pred / bar_n
    bar_identity_mse = bar_identity / bar_n
    rest_pred_mse = rest_pred / rest_n
    rest_identity_mse = rest_identity / rest_n

    def pct(pred_mse, identity_mse):
        return (identity_mse - pred_mse) / identity_mse * 100 if identity_mse > 0 else float("nan")

    print(f"\n{'region':<20} {'pred_mse':>12} {'identity_mse':>14} {'improvement':>12}")
    print(f"{'bar row (patch 7)':<20} {bar_pred_mse:>12.6f} {bar_identity_mse:>14.6f} {pct(bar_pred_mse, bar_identity_mse):>11.2f}%")
    print(f"{'rest of board':<20} {rest_pred_mse:>12.6f} {rest_identity_mse:>14.6f} {pct(rest_pred_mse, rest_identity_mse):>11.2f}%")


if __name__ == "__main__":
    main()
