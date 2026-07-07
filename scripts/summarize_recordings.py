"""Summarize Stage 0 trajectory recordings into a per-game/agent baseline table.

Reads every ARC-AGI-3-Agents/recordings/*.recording.jsonl file (produced by
the framework's Recorder) and prints final state, levels completed, and
action count per (game, agent) run.

Usage: python scripts/summarize_recordings.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = REPO_ROOT / "ARC-AGI-3-Agents" / "recordings"


def summarize_file(path: Path) -> dict:
    game_id = None
    agent = path.name.split(".")[1] if "." in path.name else "unknown"
    state = "UNKNOWN"
    levels_completed = 0
    action_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            data = event.get("data", {})
            if "state" in data and "frame" in data:
                # a FrameData dump
                game_id = data.get("game_id") or game_id
                state = data.get("state", state)
                levels_completed = data.get("levels_completed", levels_completed)
                if data.get("action_input", {}).get("id") is not None:
                    action_count += 1

    return {
        "file": path.name,
        "game_id": game_id or "unknown",
        "agent": agent,
        "final_state": state,
        "levels_completed": levels_completed,
        "actions": action_count,
    }


def main() -> None:
    if not RECORDINGS_DIR.exists():
        print(f"No recordings directory found at {RECORDINGS_DIR}")
        return

    rows = [
        summarize_file(p)
        for p in sorted(RECORDINGS_DIR.glob("*.recording.jsonl"))
    ]

    if not rows:
        print("No recordings found.")
        return

    header = f"{'game_id':<20} {'agent':<12} {'final_state':<14} {'levels':<8} {'actions':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['game_id']:<20} {r['agent']:<12} {r['final_state']:<14} "
            f"{r['levels_completed']:<8} {r['actions']:<8}"
        )


if __name__ == "__main__":
    main()
