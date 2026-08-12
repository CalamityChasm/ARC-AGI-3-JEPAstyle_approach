"""Tracks a specific-color sprite's (row, col) centroid across every step
of a recording and prints it alongside the action taken, to directly
determine the action-id -> movement-direction mapping (e.g. does
action_id=1 reliably move the sprite up?).

Usage:
    python scripts/track_sprite_and_actions.py --game bp35 --agent recurrentsearchtta --color 6
"""

import argparse
import json
from pathlib import Path

import numpy as np

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
            steps.append({
                "frame": np.array(frame),
                "state": d.get("state"),
                "action_id": action.get("id", 0),
                "available_actions": d.get("available_actions", []),
            })
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--color", type=int, required=True, help="Color id of the sprite to track.")
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()

    path = find_recording(args.game, args.agent)
    steps = load_steps(path)
    print(f"Loaded {path.name}: {len(steps)} steps")
    print(f"available_actions at step 0: {steps[0]['available_actions']}")

    prev_pos = None
    for i, s in enumerate(steps[: args.max_steps]):
        mask = s["frame"] == args.color
        if mask.any():
            rs, cs = np.where(mask)
            pos = (float(rs.mean()), float(cs.mean()))
        else:
            pos = None
        delta = None
        if pos is not None and prev_pos is not None:
            delta = (round(pos[0] - prev_pos[0], 1), round(pos[1] - prev_pos[1], 1))
        print(f"  step {i:3d}: action={s['action_id']} state={s['state']:12s} pos={pos} delta={delta}")
        if pos is not None:
            prev_pos = pos


if __name__ == "__main__":
    main()
