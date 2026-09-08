"""Builds a pretraining corpus for jepa/click_effect_model.py from real
GraphExplorerAgent recordings -- same replay logic and data source as
scripts/diagnose_state_similarity.py (which found the underlying premise
holds in real data), reshaped into a labeled (patch, segment_features) ->
(frame_changed, win) dataset instead of similarity-grouping diagnostics.

Usage:
    python scripts/harvest_click_effect_data.py --src data/graph_explorer_harvest --out data/click_effect_dataset.npz
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ARC-AGI-3-Agents"))

from agents.templates.graph_explorer_agent import FrameProcessor  # noqa: E402
from jepa.click_effect_features import extract_patch, extract_segment_features  # noqa: E402

ACTION6_ID = 6


def _load_lines(path: Path) -> list[dict]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            data = event.get("data", {})
            if "frame" in data and data["frame"]:
                lines.append(data)
    return lines


def _action_key(data: dict) -> tuple[int | None, tuple[int, int] | None]:
    action_input = data.get("action_input") or {}
    action_id = action_input.get("id")
    xy_data = action_input.get("data") or {}
    if action_id == ACTION6_ID:
        return action_id, (xy_data.get("x", 0), xy_data.get("y", 0))
    return action_id, None


def _frame_array(data: dict) -> np.ndarray:
    return np.array(data["frame"], dtype=np.uint8)[-1]


def process_game(game: str, paths: list[Path], processor: FrameProcessor) -> dict:
    patches, seg_feats, changed_labels, win_labels, games = [], [], [], [], []
    for path in paths:
        lines = _load_lines(path)
        for i in range(len(lines) - 1):
            cur, nxt = lines[i], lines[i + 1]
            action_id, xy = _action_key(nxt)
            if action_id != ACTION6_ID or xy is None:
                continue
            frame_t = _frame_array(cur)
            frame_t1 = _frame_array(nxt)
            x, y = xy
            if not (0 <= y < frame_t.shape[0] and 0 <= x < frame_t.shape[1]):
                continue

            patch = extract_patch(frame_t, x, y)
            try:
                segmented, segments = processor.segment_frame(frame_t)
                seg_f = extract_segment_features(segments, segmented, x, y)
            except Exception:
                seg_f = np.zeros(5, dtype=np.float32)

            levels_delta = nxt.get("levels_completed", cur.get("levels_completed", 0)) - cur.get("levels_completed", 0)
            frame_changed = not np.array_equal(frame_t, frame_t1)

            patches.append(patch)
            seg_feats.append(seg_f)
            changed_labels.append(float(frame_changed))
            win_labels.append(float(levels_delta > 0))
            games.append(game)

    return {
        "patches": np.stack(patches) if patches else np.zeros((0, 7, 7), dtype=np.uint8),
        "seg_feats": np.stack(seg_feats) if seg_feats else np.zeros((0, 5), dtype=np.float32),
        "changed": np.array(changed_labels, dtype=np.float32),
        "win": np.array(win_labels, dtype=np.float32),
        "games": np.array(games),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    by_game: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(src.glob("*.recording.jsonl")):
        game_prefix = path.name.split("-")[0]
        by_game[game_prefix].append(path)

    processor = FrameProcessor()
    all_patches, all_segs, all_changed, all_win, all_games = [], [], [], [], []
    for game, paths in sorted(by_game.items()):
        result = process_game(game, paths, processor)
        n = len(result["changed"])
        if n == 0:
            print(f"[{game}] 0 click transitions, skipping")
            continue
        pos_rate = result["changed"].mean()
        win_rate = result["win"].mean()
        print(f"[{game}] n={n} frame_changed_rate={pos_rate:.3f} win_rate={win_rate:.4f}")
        all_patches.append(result["patches"])
        all_segs.append(result["seg_feats"])
        all_changed.append(result["changed"])
        all_win.append(result["win"])
        all_games.append(result["games"])

    patches = np.concatenate(all_patches)
    seg_feats = np.concatenate(all_segs)
    changed = np.concatenate(all_changed)
    win = np.concatenate(all_win)
    games = np.concatenate(all_games)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, patches=patches, seg_feats=seg_feats, changed=changed, win=win, games=games)
    print(f"\nwrote {len(changed)} rows across {len(set(games.tolist()))} games to {out}")
    print(f"overall frame_changed_rate={changed.mean():.3f} win_rate={win.mean():.4f}")


if __name__ == "__main__":
    main()
