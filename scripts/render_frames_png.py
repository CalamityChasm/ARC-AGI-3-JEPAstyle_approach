"""Renders specific steps of a recording as PNG images (one file per
requested step index, plus a diff-highlighted version vs. the previous
step) -- for direct visual inspection via the Read tool, since the
existing HTML visualizer needs a live browser and file:// access wasn't
available in this environment.

Usage:
    python scripts/render_frames_png.py --game bp35 --agent recurrentsearchtta --steps 25 45 48 49 50 51 --out-dir logs/frames_bp35
"""

import argparse
import json
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDING_DIRS = [
    REPO_ROOT / "ARC-AGI-3-Agents" / "recordings",
    Path("E:/jepa_overflow/recordings"),
]

PALETTE = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37),
    (255, 255, 255), (177, 13, 201), (57, 204, 204), (1, 255, 112), (139, 69, 19), (255, 182, 193),
]
CELL = 8


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
                "frame": frame,
                "state": d.get("state"),
                "action_id": action.get("id", 0),
                "levels_completed": d.get("levels_completed", 0),
            })
    return steps


def render(frame: list, prev_frame: list | None, out_path: Path) -> None:
    img = Image.new("RGB", (64 * CELL, 64 * CELL))
    px = img.load()
    for r in range(64):
        for c in range(64):
            v = frame[r][c]
            color = PALETTE[v] if 0 <= v < len(PALETTE) else (255, 0, 255)
            if prev_frame is not None and prev_frame[r][c] != v:
                # Tint changed cells with a red border effect (brighten).
                color = tuple(min(255, ch + 60) for ch in color)
            for dr in range(CELL):
                for dc in range(CELL):
                    px[c * CELL + dc, r * CELL + dr] = color
    img.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True)
    parser.add_argument("--agent", default=None)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    path = find_recording(args.game, args.agent)
    steps = load_steps(path)
    print(f"Loaded {path.name}: {len(steps)} steps")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in args.steps:
        if i < 0 or i >= len(steps):
            print(f"  step {i}: out of range, skipping")
            continue
        prev = steps[i - 1]["frame"] if i > 0 else None
        out_path = args.out_dir / f"step_{i:04d}_state-{steps[i]['state']}_action-{steps[i]['action_id']}.png"
        render(steps[i]["frame"], prev, out_path)
        print(f"  step {i}: state={steps[i]['state']} action={steps[i]['action_id']} -> {out_path}")


if __name__ == "__main__":
    main()
