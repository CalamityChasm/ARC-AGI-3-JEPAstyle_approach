"""Stage 6 test-time adaptation AGENT follow-up: measure real per-turn
latency of jepa/test_time_adapter.py's TestTimeAdapter, using the actual
production checkpoint (checkpoints/) and the actual hypothesis_agent.py
integration's chosen operating point (K=5, STEPS=8, LR=5e-5), not a
synthetic microbenchmark -- so the number reported is what a real Kaggle
episode would actually pay, not a best-case estimate.

Replays real recorded transitions through TestTimeAdapter.observe() in
order, exactly matching how ARC-AGI-3-Agents/agents/templates/
hypothesis_agent.py calls it once per turn from _update_hypotheses, and
reports:
  - mean/max latency of a plain observe() call that does NOT trigger an
    adaptation step (the common case, (K-1)/K of all turns)
  - mean/max latency of an observe() call that DOES trigger one (1/K of
    turns, K adaptation events x N_STEPS gradient steps each)
  - effective mean added latency per turn (blends both), for direct
    comparison against a real per-action Kaggle turn budget

Usage: python scripts/measure_tta_latency.py
"""

import json
import statistics
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from jepa.data.trajectories import _load_frame_lines  # noqa: E402
from jepa.device import get_device  # noqa: E402
from jepa.models import CNNEncoder, MoEPredictor  # noqa: E402
from jepa.test_time_adapter import TestTimeAdapter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"


def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    game_vocab = json.loads((CHECKPOINT_DIR / "game_vocab_moe.json").read_text())
    encoder = CNNEncoder().to(device)
    encoder.load_state_dict(torch.load(CHECKPOINT_DIR / "encoder_moe.pt", map_location=device))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    predictor = MoEPredictor(num_games=len(game_vocab), num_experts=8).to(device)
    predictor.load_state_dict(torch.load(CHECKPOINT_DIR / "moe_predictor.pt", map_location=device))
    predictor.eval()

    game_idx = game_vocab.get("r11l", 0)
    adapter = TestTimeAdapter(
        predictor, encoder, device, game_idx=game_idx, k=5, n_steps=8, lr=5e-5,
    )

    # Real recorded transitions for one game, in order -- exactly what
    # choose_action's per-turn call sees during real play.
    files = sorted(RECORDINGS_DIR.glob("r11l-*.recording.jsonl"))
    files = [f for f in files if ".random.80." in f.name]
    assert files, "no r11l recordings found -- restore the 150-file corpus first"
    frames = _load_frame_lines(files[0])

    plain_latencies = []
    adapt_latencies = []

    for i in range(len(frames) - 1):
        cur, nxt = frames[i], frames[i + 1]
        action = nxt["action_input"]
        action_id = action["id"]
        xy_data = action.get("data", {}) or {}
        xy = (xy_data.get("x", 0), xy_data.get("y", 0))

        n_before = adapter.n_adapt_events
        t0 = time.perf_counter()
        adapter.observe(cur["frame"], action_id, xy, nxt["frame"])
        dt = time.perf_counter() - t0

        if adapter.n_adapt_events > n_before:
            adapt_latencies.append(dt)
        else:
            plain_latencies.append(dt)

    print(f"\nTotal observed transitions: {len(frames) - 1}")
    print(f"Adaptation events fired: {len(adapt_latencies)}")
    print(f"\nPlain observe() (no adapt step), n={len(plain_latencies)}:")
    print(f"  mean={statistics.mean(plain_latencies)*1000:.3f}ms  "
          f"max={max(plain_latencies)*1000:.3f}ms  "
          f"median={statistics.median(plain_latencies)*1000:.3f}ms")
    print(f"\nAdapt-triggering observe() (K=5, 8 AdamW steps), n={len(adapt_latencies)}:")
    if adapt_latencies:
        print(f"  mean={statistics.mean(adapt_latencies)*1000:.3f}ms  "
              f"max={max(adapt_latencies)*1000:.3f}ms  "
              f"median={statistics.median(adapt_latencies)*1000:.3f}ms")

    n_plain, n_adapt = len(plain_latencies), len(adapt_latencies)
    n_total = n_plain + n_adapt
    mean_plain = statistics.mean(plain_latencies) if plain_latencies else 0.0
    mean_adapt = statistics.mean(adapt_latencies) if adapt_latencies else 0.0
    effective_mean = (n_plain * mean_plain + n_adapt * mean_adapt) / n_total
    print(f"\nEffective mean added latency per turn (blended over K=5 cadence): "
          f"{effective_mean*1000:.3f}ms")
    print("(This is ADDED latency on top of the existing per-turn Q-scoring "
          "forward passes in _choose_action_inner -- not total turn time.)")


if __name__ == "__main__":
    main()
