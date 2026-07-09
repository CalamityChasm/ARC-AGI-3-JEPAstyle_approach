"""Quick offline diagnostic: replay a handful of real local-recording episodes
through the Stage 5 hypothesis bundle's confidence-update mechanism (without
actually playing games) and report the resulting beta (explore/exploit blend)
distribution -- checks whether the Bayesian confidence ever actually
concentrates enough during a real episode for the value head to matter, or
whether beta stays low (InfoGain-dominated) the whole time.
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

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"


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


def main() -> None:
    device = get_device()
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_moe.pt", map_location=device))
    encoder.eval()

    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_moe.json").read_text())
    predictor = MoEPredictor(num_games=max(len(game_vocab), 1), num_experts=8).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "moe_predictor.pt", map_location=device))
    predictor.eval()

    episode_files = sorted(RECORDINGS_DIR.glob("*.random.*.recording.jsonl"))[:20]
    betas = []
    entropies = []

    for path in episode_files:
        frames = load_episode(path)
        if len(frames) < 2:
            continue
        game_id = frames[0].get("game_id", "unknown")
        game_idx = game_vocab.get(game_id, 0)
        bundle = HypothesisBundle(num_hypotheses=predictor.num_experts, tau=0.01)

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
                    xy_t = torch.tensor([[prev_xy[0] / 63.0, prev_xy[1] / 63.0]], dtype=torch.float32, device=device)
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

    if not betas:
        print("No transitions processed.")
        return

    betas_t = torch.tensor(betas)
    print(f"processed {len(betas)} transitions across {len(episode_files)} episodes")
    print(f"beta: mean={betas_t.mean():.4f} std={betas_t.std():.4f} min={betas_t.min():.4f} max={betas_t.max():.4f}")
    print(f"  fraction beta > 0.1: {(betas_t > 0.1).float().mean():.3f}")
    print(f"  fraction beta > 0.3: {(betas_t > 0.3).float().mean():.3f}")
    print(f"  fraction beta > 0.5: {(betas_t > 0.5).float().mean():.3f}")


if __name__ == "__main__":
    main()
