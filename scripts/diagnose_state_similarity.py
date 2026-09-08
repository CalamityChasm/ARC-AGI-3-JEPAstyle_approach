"""Diagnostic (read-only, no training, no agent changes): do genuinely
different exact-graph states ever share a *local* feature -- the small
neighborhood right around a click point, or the shape/color/size of the
object clicked on, independent of where it sits on the board -- and when
they do, does the action's outcome actually match?

Motivated by a design discussion: jepa/memory.py's TransitionGraph and
graph_explorer_core.py's GraphExplorer both key everything on the EXACT
full-frame hash, so two states that a human would call "basically the
same local situation, just elsewhere on the board" are, to those
structures, two unrelated dictionary entries with no connection at all.
Global frame-level symmetry was considered and deliberately rejected
(a puzzle could use symmetry as its actual mechanic, not decoration --
pruning on it risks silently ignoring the winning move). This checks a
narrower, safer claim instead: LOCAL similarity *within one game's own
graph*, not a cross-game or global-symmetry assumption.

Two checks, both restricted to ACTION6 (click) transitions and both
requiring the two matched instances to come from genuinely DIFFERENT
overall exact states (not two visits to the literal same state, which
the existing exact graph already handles trivially):

  A. Local-patch match: the raw pixel neighborhood around the click point
     is byte-identical in two different states. Does the action's effect
     (did anything change, and does the local neighborhood evolve the
     same way afterward) also match?
  B. Same-object-elsewhere match: FrameProcessor.segment_frame() identifies
     the clicked segment; group by a POSITION-INVARIANT signature (color,
     area, is_rectangle, bounding-box width/height) so the same kind of
     object clicked at a different board position still matches. Same
     outcome-consistency check.

This is read-only diagnosis of whether the premise holds on real data --
it does not implement any exploitation mechanism. If match rates are
near-zero, or matches exist but outcomes are inconsistent, that's a real,
useful negative result, not a failure of the script.

Usage:
    python scripts/diagnose_state_similarity.py --src data/graph_explorer_harvest --out data/state_similarity_report.json
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ARC-AGI-3-Agents"))

from agents.templates.graph_explorer_agent import FrameProcessor  # noqa: E402

PATCH_RADIUS = 3  # 7x7 window
RESET_ACTION_ID = 0
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


def _exact_hash(frame_np: np.ndarray) -> str:
    return hashlib.blake2b(frame_np.tobytes(), digest_size=12).hexdigest()


def _patch(frame_np: np.ndarray, x: int, y: int, radius: int = PATCH_RADIUS) -> tuple:
    h, w = frame_np.shape
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    sub = frame_np[y0:y1, x0:x1]
    # Pad to a fixed shape so edge-of-board patches don't spuriously match
    # interior ones just because numpy slicing clipped them to the same
    # smaller size -- pad with a sentinel (255) never present in a real
    # 0-15 ARC frame.
    padded = np.full((2 * radius + 1, 2 * radius + 1), 255, dtype=np.uint8)
    padded[: sub.shape[0], : sub.shape[1]] = sub
    # Also record where within the padded window the true top-left corner
    # landed, so an edge-clipped patch (asymmetric window) never collides
    # with an interior patch that just happens to share the same pixels
    # in the unclipped region.
    offset = (y - y0, x - x0)
    return (padded.tobytes(), offset)


def _segment_signature(segments: list[dict], x: int, y: int, segmented_frame: np.ndarray) -> tuple | None:
    seg_id = int(segmented_frame[y, x])
    if seg_id < 0 or seg_id >= len(segments):
        return None
    seg = segments[seg_id]
    x1, y1, x2, y2 = seg["bounding_box"]
    width, height = x2 - x1 + 1, y2 - y1 + 1
    return (seg["color"], seg["area"], seg["is_rectangle"], width, height)


def process_game(game: str, paths: list[Path], processor: FrameProcessor) -> dict:
    patch_groups: dict[tuple, list[dict]] = defaultdict(list)
    sig_groups: dict[tuple, list[dict]] = defaultdict(list)
    total_click_transitions = 0

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

            total_click_transitions += 1
            state_hash = _exact_hash(frame_t)
            levels_delta = nxt.get("levels_completed", cur.get("levels_completed", 0)) - cur.get("levels_completed", 0)
            frame_changed = not np.array_equal(frame_t, frame_t1)
            patch_before = _patch(frame_t, x, y)
            patch_after = _patch(frame_t1, x, y)

            row = {
                "game": game, "state_hash": state_hash, "levels_delta": levels_delta,
                "frame_changed": frame_changed, "patch_after": patch_after,
            }
            patch_groups[patch_before].append(row)

            try:
                segmented, segments = processor.segment_frame(frame_t)
                sig = _segment_signature(segments, x, y, segmented)
            except Exception:
                sig = None
            if sig is not None:
                sig_groups[sig].append(row)

    def summarize(groups: dict[tuple, list[dict]]) -> dict:
        qualifying = []
        for key, rows in groups.items():
            distinct_states = {r["state_hash"] for r in rows}
            if len(distinct_states) < 2:
                continue  # only same-state repeats -- not the novel case
            qualifying.append(rows)

        n_groups = len(qualifying)
        n_matched_transitions = sum(len(rows) for rows in qualifying)
        changed_consistent = sum(1 for rows in qualifying if len({r["frame_changed"] for r in rows}) == 1)
        patch_after_consistent = sum(1 for rows in qualifying if len({r["patch_after"] for r in rows}) == 1)
        any_win_delta = sum(1 for rows in qualifying if any(r["levels_delta"] > 0 for r in rows))
        win_delta_consistent = 0
        for rows in qualifying:
            deltas = {r["levels_delta"] > 0 for r in rows}
            if len(deltas) == 1:
                win_delta_consistent += 1

        examples = []
        for rows in sorted(qualifying, key=len, reverse=True)[:3]:
            examples.append({
                "n_members": len(rows),
                "distinct_states": len({r["state_hash"] for r in rows}),
                "frame_changed_values": [r["frame_changed"] for r in rows],
                "levels_delta_values": [r["levels_delta"] for r in rows],
            })

        return {
            "n_groups_with_cross_state_match": n_groups,
            "n_transitions_in_such_groups": n_matched_transitions,
            "frame_changed_consistency_rate": changed_consistent / n_groups if n_groups else None,
            "local_patch_after_consistency_rate": patch_after_consistent / n_groups if n_groups else None,
            "groups_containing_a_win": any_win_delta,
            "win_occurrence_consistency_rate": win_delta_consistent / n_groups if n_groups else None,
            "example_groups": examples,
        }

    return {
        "total_click_transitions": total_click_transitions,
        "patch_match": summarize(patch_groups),
        "same_object_elsewhere_match": summarize(sig_groups),
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
    report = {}
    for game, paths in sorted(by_game.items()):
        result = process_game(game, paths, processor)
        report[game] = result
        pm, sm = result["patch_match"], result["same_object_elsewhere_match"]
        print(
            f"[{game}] clicks={result['total_click_transitions']:5d}  "
            f"patch: groups={pm['n_groups_with_cross_state_match']:3d} "
            f"changed_consist={pm['frame_changed_consistency_rate']}  "
            f"same_object: groups={sm['n_groups_with_cross_state_match']:3d} "
            f"changed_consist={sm['frame_changed_consistency_rate']}"
        )

    total_clicks = sum(r["total_click_transitions"] for r in report.values())
    total_patch_groups = sum(r["patch_match"]["n_groups_with_cross_state_match"] for r in report.values())
    total_sig_groups = sum(r["same_object_elsewhere_match"]["n_groups_with_cross_state_match"] for r in report.values())
    print(f"\n=== pooled ===")
    print(f"total click transitions examined: {total_clicks}")
    print(f"total cross-state patch-match groups found: {total_patch_groups}")
    print(f"total cross-state same-object-elsewhere-match groups found: {total_sig_groups}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote full report to {out}")


if __name__ == "__main__":
    main()
