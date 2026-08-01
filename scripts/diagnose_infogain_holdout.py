"""Does the Stage 5 hypothesis bundle's InfoGain signal (expert-disagreement
across the 8 MoE experts, jepa/hypothesis_bundle.py: info_gain) carry any
real signal on games the model was never trained on -- or does it collapse
to near-zero right when it would matter most (a genuinely novel Kaggle
game), since it's built directly on the same experts whose predictive
advantage was just shown (stage6-multifold-cv) to collapse to identity
parity on held-out games?

Reuses the stage6-game-holdout baseline checkpoint (20 games trained,
5 held out: r11l, bp35, m0r0, tr87, ka59) -- no retraining needed, this is
a pure diagnostic over an existing checkpoint.

Usage:
    python scripts/diagnose_infogain_holdout.py
"""

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jepa.data.trajectories import _load_frame_lines
from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor
from jepa.hypothesis_bundle import info_gain
from jepa.models import CNNEncoder, MoEPredictor

ARCHIVE_DIR = Path("E:/ARC-AGI-3-JEPAstyle_data/recordings_archive")
CHECKPOINT_DIR = Path(
    "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
    "agent-a0f09770086c096a6/checkpoints_holdout_baseline"
)
HELD_OUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
SAMPLE_N = 500  # per group -- fast, still a solid sample
NUM_CANDIDATE_ACTIONS = 4  # score info_gain for action ids 0-3 per state, average


def load_verified_transitions() -> list:
    transitions = []
    files = sorted(ARCHIVE_DIR.glob("*.random.80.*.recording.jsonl"))
    assert len(files) == 150, f"expected 150 verified corpus files, found {len(files)}"
    for path in files:
        frames = _load_frame_lines(path)
        for i in range(len(frames) - 1):
            cur, nxt = frames[i], frames[i + 1]
            game_id = cur.get("game_id", "unknown")
            transitions.append((cur["frame"], game_id))
    return transitions


def load_checkpoint(device):
    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_moe.json").read_text())
    meta = json.loads((CHECKPOINT_DIR / "moe_training_meta.json").read_text())
    num_experts = meta.get("num_experts", 8)
    feature_channels = meta.get("feature_channels", 64)
    encoder = CNNEncoder(out_channels=feature_channels).to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_moe.pt", map_location=device))
    encoder.eval()
    predictor = MoEPredictor(
        num_games=len(game_vocab), num_experts=num_experts,
        feature_channels=feature_channels, expert_hidden=feature_channels,
    ).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "moe_predictor.pt", map_location=device))
    predictor.eval()
    return encoder, predictor, game_vocab


@torch.no_grad()
def mean_info_gain(frames, game_ids, encoder, predictor, game_vocab, device):
    random.seed(0)
    idxs = list(range(len(frames)))
    random.shuffle(idxs)
    idxs = idxs[:SAMPLE_N]

    values = []
    for i in idxs:
        frame, game_id = frames[i], game_ids[i]
        tensor = arc3_frame_to_tensor(frame)
        x = torch.from_numpy(tensor).unsqueeze(0).to(device)
        feat = encoder(x)  # (1, C, H, W)
        game_idx = torch.full((1,), game_vocab.get(game_id, 0), dtype=torch.long, device=device)
        xy = torch.zeros((1, 2), dtype=torch.float32, device=device)

        per_action_ig = []
        for action_id in range(NUM_CANDIDATE_ACTIONS):
            action_t = torch.full((1,), action_id, dtype=torch.long, device=device)
            expert_preds = predictor.predict_all_experts(feat, action_t, xy, game_idx)  # (1, K, C, H, W)
            ig = info_gain(expert_preds.squeeze(0)).item()
            per_action_ig.append(ig)
        values.append(sum(per_action_ig) / len(per_action_ig))

    values_t = torch.tensor(values)
    return values_t.mean().item(), values_t.std().item(), len(values)


def main():
    device = get_device()
    print(f"device: {device}")

    print("loading verified corpus...")
    transitions = load_verified_transitions()
    print(f"{len(transitions)} frames loaded")

    held_out_frames, held_out_games = [], []
    trained_frames, trained_games = [], []
    for frame, game_id in transitions:
        short = game_id.split("-")[0]
        if short in HELD_OUT_GAMES:
            held_out_frames.append(frame)
            held_out_games.append(game_id)
        else:
            trained_frames.append(frame)
            trained_games.append(game_id)

    print(f"held-out-game frames available: {len(held_out_frames)}")
    print(f"trained-game frames available: {len(trained_frames)}")

    print("loading checkpoint...")
    encoder, predictor, game_vocab = load_checkpoint(device)

    print(f"computing InfoGain on {SAMPLE_N} held-out-game states x {NUM_CANDIDATE_ACTIONS} actions each...")
    ho_mean, ho_std, ho_n = mean_info_gain(held_out_frames, held_out_games, encoder, predictor, game_vocab, device)
    print(f"held-out games:  mean InfoGain = {ho_mean:.6e}  (std {ho_std:.6e}, n={ho_n})")

    print(f"computing InfoGain on {SAMPLE_N} trained-game states x {NUM_CANDIDATE_ACTIONS} actions each...")
    tr_mean, tr_std, tr_n = mean_info_gain(trained_frames, trained_games, encoder, predictor, game_vocab, device)
    print(f"trained games:   mean InfoGain = {tr_mean:.6e}  (std {tr_std:.6e}, n={tr_n})")

    ratio = ho_mean / tr_mean if tr_mean else float("nan")
    print(f"\nheld-out / trained InfoGain ratio: {ratio:.3f}")
    if ratio < 0.3:
        print("VERDICT: InfoGain collapses on held-out games -- the hypothesis bundle's")
        print("exploration signal is much weaker exactly where it's needed most.")
    elif ratio > 0.7:
        print("VERDICT: InfoGain is roughly comparable on held-out vs trained games --")
        print("expert disagreement survives even though predictive accuracy doesn't.")
    else:
        print("VERDICT: partial collapse -- InfoGain is weaker but not gone on held-out games.")


if __name__ == "__main__":
    main()
