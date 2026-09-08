"""Replays *.recording.jsonl files (real GraphExplorerAgent play) into a
per-decision-point training table for a learned "how many hops to a known
win from this exact state" model -- the cheap test of whether a purpose-
trained structural signal can usefully bias GraphExplorer's own travel-mode
routing (see CLAUDE.md's GraphExplorerAgent section for the design
discussion this follows up on, and stage6_rollout_compounding_error.md /
stage6_graph_lookahead.md for why this is scoped as a structural-graph
model rather than a latent-frame predictor).

Two passes per game, merged across every recording file for that game
(RESET returns to the same exact level-start state, so different episodes
of the same game share one graph -- same assumption jepa/memory.py's
TransitionGraph and graph_explorer_agent.py's transition_memory already
make):

  1. Causal replay: for each decision point, record CAUSAL structural
     features (jepa/graph_features.py: StructuralGraphTracker) -- the
     graph as it existed strictly BEFORE this decision's outcome was
     known, matching exactly what a live agent's tie_break_fn could see.
     Only non-RESET transitions are tracked (RESET isn't a graph_explorer
     edge at all -- see graph_features.py's own note).
  2. Hindsight labeling: once the full per-game graph is built, reverse-
     BFS from every win edge gives every state its true hops-to-win
     (jepa/graph_features.py: hops_to_win_labels). States with no
     reachable win in the explored graph are dropped, not sentinel-filled
     -- same honest-labeling discipline as Stage 5's value head training.

Usage:
    python scripts/harvest_graph_win_distance_data.py --src ARC-AGI-3-Agents/recordings --dst data/graph_win_distance_dataset.csv
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ARC-AGI-3-Agents"))

from jepa.graph_features import FEATURE_NAMES, StructuralGraphTracker, hops_to_win_labels  # noqa: E402
from agents.templates.graph_explorer_agent import FrameProcessor  # noqa: E402
import numpy as np  # noqa: E402

RESET_ACTION_ID = 0


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
                continue  # tolerate a truncated final line (disk-full crash etc.)
            data = event.get("data", {})
            if "frame" in data and data["frame"]:
                lines.append(data)
    return lines


def _action_key(data: dict) -> tuple[int, tuple[int, int] | None]:
    action_input = data.get("action_input") or {}
    action_id = action_input.get("id")
    xy_data = action_input.get("data") or {}
    if action_id == 6:
        return action_id, (xy_data.get("x", 0), xy_data.get("y", 0))
    return action_id, None


def _masked_hash(frame_np: np.ndarray, processor: FrameProcessor, level_up: bool, mask_holder: list) -> str:
    """Mirrors GraphExplorerAgent._choose_action_inner's own status-bar
    masking pipeline (lines ~633-662 of graph_explorer_agent.py) so
    offline-computed state keys match what the live agent would compute
    -- required for feature/label consistency with live inference. The
    live agent only recomputes the status-bar mask when `self.level_up`
    is True (episode start or right after a level-completion), reusing it
    for every frame in between; `mask_holder` is a 1-element list used as
    a mutable cell so this function can update the caller's cached mask
    without needing a class.
    """
    frame_np = frame_np.copy()
    if level_up or mask_holder[0] is None:
        segmented, segments = processor.segment_frame(frame_np)
        _bars, status_bar_mask = processor.identify_status_bars(segmented, segments)
        mask_holder[0] = status_bar_mask
    frame_np[mask_holder[0]] = processor.status_bar_color
    frame_np[frame_np == processor.status_bar_color] = 0
    return processor.hash_frame(frame_np)


def process_game(paths: list[Path], processor: FrameProcessor) -> list[dict]:
    tracker = StructuralGraphTracker()
    decisions: list[dict] = []  # rows missing only the hindsight label

    for path in paths:
        lines = _load_lines(path)
        last_levels = 0
        mask_holder = [None]  # recomputed at file start (mirrors episode start's level_up=True)
        for i in range(len(lines) - 1):
            cur = lines[i]
            nxt = lines[i + 1]
            action_id, xy = _action_key(nxt)
            if action_id is None or action_id == RESET_ACTION_ID:
                last_levels = nxt.get("levels_completed", last_levels)
                continue

            frame_np = np.array(cur["frame"], dtype=np.uint8)[-1]  # matches _choose_action_inner's own "take the last one"
            level_up_here = (i == 0)  # approximation: exact per-level recompute needs live level_up tracking
            state = _masked_hash(frame_np, processor, level_up_here, mask_holder)
            tracker.observe_state(state)

            available = cur.get("available_actions") or []
            feats = tracker.features(
                state,
                num_available_actions=len(available),
                action6_available=6 in available,
            )
            decisions.append({"state": state, **feats})

            next_frame_np = np.array(nxt["frame"], dtype=np.uint8)[-1]
            levels_completed = nxt.get("levels_completed", last_levels)
            levels_delta = levels_completed - last_levels
            next_level_up = levels_delta > 0
            next_state = _masked_hash(next_frame_np, processor, next_level_up, mask_holder)
            last_levels = levels_completed

            tracker.record_edge(state, action_id, xy, next_state, levels_delta)

    labels = hops_to_win_labels(tracker.out_edges)
    rows = []
    for d in decisions:
        label = labels.get(d["state"])
        if label is None:
            continue
        rows.append({**d, "hops_to_win": label})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()

    src = Path(args.src)
    by_game: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(src.glob("*.recording.jsonl")):
        game_prefix = path.name.split("-")[0]
        by_game[game_prefix].append(path)

    processor = FrameProcessor()
    all_rows = []
    for game, paths in sorted(by_game.items()):
        rows = process_game(paths, processor)
        for r in rows:
            r["game"] = game
        all_rows.extend(rows)
        n_labeled_states = len({r["state"] for r in rows})
        print(f"[{game}] {len(paths)} files -> {len(rows)} labeled decisions, "
              f"{n_labeled_states} distinct labeled states, "
              f"min/max hops_to_win={min((r['hops_to_win'] for r in rows), default='-')}/"
              f"{max((r['hops_to_win'] for r in rows), default='-')}")

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["game", "state", *FEATURE_NAMES, "hops_to_win"]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\nwrote {len(all_rows)} rows across {len(by_game)} games to {dst}")


if __name__ == "__main__":
    main()
