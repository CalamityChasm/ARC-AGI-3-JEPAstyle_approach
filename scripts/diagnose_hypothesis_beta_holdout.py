"""Quick offline diagnostic (stage6-novelty-aware-beta): does the Stage 5
hypothesis bundle's entropy-driven confidence -- built from *observed*
per-expert error on transitions actually seen in the episode -- report
false confidence (high beta) on games the underlying MoE predictor was
never trained on?

Motivation: CLAUDE.md's Stage 6 addendum showed the *gated* MoE
prediction collapses toward "predict no change" on held-out games, while
the raw *ungated* per-expert disagreement (InfoGain) does not
(scripts/diagnose_infogain_holdout.py: ratio 0.999). If a collapsed
predictor's per-expert errors all land small and similar on a held-out
game (because the game itself doesn't change much under exploration
either, not because the experts are actually reliable there), the
Bayesian confidence-entropy update could still report low entropy /
high beta -- false confidence that hands control to the value head
exactly where it's least trustworthy. This script checks whether that's
actually happening, and gives a real distribution to pick
NOVELTY_BETA_CAP from (rather than guessing a value).

Mirrors scripts/diagnose_hypothesis_beta.py's replay methodology exactly,
but (a) uses the stage6-game-holdout checkpoint (5 games never trained on:
r11l, bp35, m0r0, tr87, ka59) instead of production, and (b) reports the
held-out-game and trained-game beta distributions separately.

Usage:
    python scripts/diagnose_hypothesis_beta_holdout.py
"""

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jepa.device import get_device
from jepa.grid import arc3_frame_to_tensor
from jepa.hypothesis_bundle import HypothesisBundle
from jepa.models import CNNEncoder, MoEPredictor

CHECKPOINT_DIR = Path(
    "C:/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/.claude/worktrees/"
    "agent-a0f09770086c096a6/checkpoints_holdout_baseline"
)
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"
HELD_OUT_GAMES = ["r11l", "bp35", "m0r0", "tr87", "ka59"]
TAU = 0.01  # matches Hypothesis.TAU


def load_episode(path: Path):
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if "state" in data and "frame" in data:
                frames.append(data)
    return frames


def replay_group(episode_files, encoder, predictor, game_vocab, device):
    betas, entropies = [], []
    for path in episode_files:
        frames = load_episode(path)
        if len(frames) < 2:
            continue
        game_id = frames[0].get("game_id", "unknown")
        game_idx = game_vocab.get(game_id, 0)
        bundle = HypothesisBundle(num_hypotheses=predictor.num_experts, tau=TAU)

        prev_feat = None
        prev_action = None
        prev_xy = None
        for d in frames:
            tensor = arc3_frame_to_tensor(d["frame"])
            x = torch.from_numpy(tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(x)

            if prev_feat is not None and prev_action is not None and prev_action != 0:
                action_t = torch.full((1,), prev_action, dtype=torch.long, device=device)
                if prev_xy is not None:
                    xy_t = torch.tensor(
                        [[prev_xy[0] / 63.0, prev_xy[1] / 63.0]], dtype=torch.float32, device=device
                    )
                else:
                    xy_t = torch.zeros((1, 2), dtype=torch.float32, device=device)
                game_t = torch.full((1,), game_idx, dtype=torch.long, device=device)
                with torch.no_grad():
                    expert_preds = predictor.predict_all_experts(prev_feat, action_t, xy_t, game_t)[0]
                    errors = (expert_preds - feat[0].unsqueeze(0)).pow(2).mean(dim=(1, 2, 3)).cpu()
                bundle.update(errors)
                betas.append(bundle.beta())
                entropies.append(bundle.entropy() / bundle.max_entropy())

            action_input = d.get("action_input", {})
            prev_action = action_input.get("id")
            prev_xy = None
            data_field = action_input.get("data")
            if isinstance(data_field, dict) and "x" in data_field:
                prev_xy = (data_field["x"], data_field["y"])
            prev_feat = feat

    return betas, entropies


def report(label: str, betas: list, entropies: list) -> None:
    if not betas:
        print(f"{label}: no transitions processed.")
        return
    b = torch.tensor(betas)
    print(f"{label}: n={len(betas)}")
    print(f"  beta: mean={b.mean():.4f} std={b.std():.4f} min={b.min():.4f} max={b.max():.4f}")
    for q in (0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
        print(f"  fraction beta > {q:.2f}: {(b > q).float().mean():.3f}")


def main() -> None:
    device = get_device()
    print(f"device: {device}")

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

    all_files = sorted(RECORDINGS_DIR.glob("*.random.80.*.recording.jsonl"))
    held_out_files = [f for f in all_files if f.name.split("-")[0] in HELD_OUT_GAMES]
    trained_files = [f for f in all_files if f.name.split("-")[0] not in HELD_OUT_GAMES][:20]

    print(f"held-out episode files: {len(held_out_files)}")
    print(f"trained episode files (sampled): {len(trained_files)}")

    ho_betas, ho_entropies = replay_group(held_out_files, encoder, predictor, game_vocab, device)
    tr_betas, tr_entropies = replay_group(trained_files, encoder, predictor, game_vocab, device)

    print()
    report("HELD-OUT games (never trained on)", ho_betas, ho_entropies)
    print()
    report("TRAINED games (in game_vocab)", tr_betas, tr_entropies)


if __name__ == "__main__":
    main()
