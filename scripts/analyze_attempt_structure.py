"""Directly inspects a bp35/ka59 recording to characterize what happens
within a single "attempt" (a GAME_OVER-to-GAME_OVER, or RESET-to-RESET,
span) -- not just the already-established fact that these games cap
actions per attempt, but WHAT drives that cap and whether anything on the
board visibly signals it (a countdown, a moving hazard, etc.), and
whether anything about board state correlates with when GAME_OVER fires.

Usage:
    python scripts/analyze_attempt_structure.py --game bp35 --agent recurrentsearchtta
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIRS = [
    REPO_ROOT / "ARC-AGI-3-Agents" / "recordings",
    Path("E:/jepa_overflow/recordings"),
]


def find_recording(game: str, agent: str | None) -> Path:
    candidates = []
    for d in RECORDING_DIRS:
        if not d.exists():
            continue
        for f in d.glob(f"{game}-*.recording.jsonl"):
            if agent and f".{agent}." not in f.name:
                continue
            candidates.append(f)
    if not candidates:
        raise FileNotFoundError(f"No recording found for game={game!r} agent={agent!r}")
    return max(candidates, key=lambda f: f.stat().st_mtime)


def load_steps(path: Path) -> list[dict]:
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            d = rec["data"]
            frame = d["frame"][0]
            action = d.get("action_input") or {}
            action_id = action.get("id", 0)
            action_data = action.get("data") or {}
            steps.append({
                "frame": frame,
                "state": d.get("state"),
                "levels_completed": d.get("levels_completed", 0),
                "action_id": action_id,
                "x": action_data.get("x"),
                "y": action_data.get("y"),
            })
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--agent", default=None)
    args = parser.parse_args()

    path = find_recording(args.game, args.agent)
    steps = load_steps(path)
    print(f"Loaded {path.name}: {len(steps)} steps")

    # 1. Segment into attempts: an "attempt" is the span between two
    # consecutive RESET actions (action_id==0) -- the harness's own
    # attempt boundary, independent of the recording's episode guid.
    attempt_starts = [i for i, s in enumerate(steps) if s["action_id"] == 0]
    attempt_starts.append(len(steps))
    print(f"\nRESET actions found at step indices: {attempt_starts[:-1]}")
    lengths = [attempt_starts[i + 1] - attempt_starts[i] for i in range(len(attempt_starts) - 1)]
    print(f"Attempt lengths (steps between consecutive RESETs): {lengths}")

    # 2. State sequence around each GAME_OVER transition.
    print("\nState transitions (only when state changes):")
    prev_state = None
    for i, s in enumerate(steps):
        if s["state"] != prev_state:
            print(f"  step {i:4d}: state -> {s['state']}  levels_completed={s['levels_completed']}")
            prev_state = s["state"]

    # 3. Look for a "counter" region: cells whose value takes on many
    # distinct values across the sequence but changes monotonically
    # within each attempt (classic countdown-timer signature), vs cells
    # that are just noisy/static.
    import numpy as np

    frames = np.array([s["frame"] for s in steps])  # (T, 64, 64)
    n_distinct = np.array([[len(set(frames[:, r, c].tolist())) for c in range(64)] for r in range(64)])
    print(f"\nCells with the most distinct values across the whole run (top 15):")
    flat_idx = np.argsort(n_distinct.flatten())[::-1][:15]
    for idx in flat_idx:
        r, c = idx // 64, idx % 64
        vals = frames[:, r, c]
        print(f"  (r={r:2d}, c={c:2d}): {n_distinct[r, c]} distinct values, "
              f"first 10 values over time: {vals[:10].tolist()}")

    # 4. Within the FIRST attempt specifically, check the last 10 frames
    # before its terminal state for anything visually distinctive.
    first_len = lengths[0] if lengths else len(steps)
    print(f"\nFirst attempt: {first_len} steps. Frame diff count per step (# cells changed vs previous):")
    diffs = []
    for i in range(1, min(first_len, len(steps))):
        d = int((frames[i] != frames[i - 1]).sum())
        diffs.append(d)
    print(f"  {diffs}")

    print(f"\nTotal levels_completed ever observed (max): {max(s['levels_completed'] for s in steps)}")


if __name__ == "__main__":
    main()
