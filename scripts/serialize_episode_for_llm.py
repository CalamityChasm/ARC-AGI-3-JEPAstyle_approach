"""Stage 7 prototype helper: serialize a recorded ARC-AGI-3 episode into a
compact, LLM-readable text format (full first grid + per-step diffs), for
testing whether an LLM can form a sensible hypothesis about a game's
mechanics from a short opening sequence.

Not part of the live agent -- a standalone research/prototyping script.
Reads recordings directly from the main (non-worktree) checkout's
ARC-AGI-3-Agents/recordings/ dir, since that dir is gitignored and not
present in this isolated worktree.

Usage:
    python scripts/serialize_episode_for_llm.py <game_prefix> [--steps N] [--recordings-dir DIR]

Example:
    python scripts/serialize_episode_for_llm.py bp35 --steps 20
"""
import argparse
import glob
import json
import os

ACTION_NAMES = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}

DEFAULT_RECORDINGS_DIR = (
    r"C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\ARC-AGI-3-Agents\recordings"
)


def find_recording(game_prefix: str, recordings_dir: str, agent_filter: str = "random") -> str:
    pattern = os.path.join(recordings_dir, f"{game_prefix}*.{agent_filter}.*.recording.jsonl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fall back to any agent
        pattern = os.path.join(recordings_dir, f"{game_prefix}*.recording.jsonl")
        matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No recording found for prefix {game_prefix!r} in {recordings_dir}")
    return matches[0]


def load_frames(path: str, steps: int):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > steps:
                break
            records.append(json.loads(line))
    return records


def grid_to_ascii(grid):
    """grid: list of rows, each a list of ints 0-15. Render as space-joined hex-ish digits."""
    lines = []
    for row in grid:
        lines.append("".join(f"{v:x}" for v in row))
    return "\n".join(lines)


def diff_cells(prev_grid, cur_grid):
    diffs = []
    for r in range(len(cur_grid)):
        for c in range(len(cur_grid[r])):
            if prev_grid[r][c] != cur_grid[r][c]:
                diffs.append((r, c, prev_grid[r][c], cur_grid[r][c]))
    return diffs


def serialize(records, max_diff_cells=40):
    """Build a compact text serialization: full first grid, then per-step
    action + diff summary (bounded so a busy frame doesn't blow up the
    token budget)."""
    out_lines = []
    prev_grid = None
    for i, rec in enumerate(records):
        data = rec["data"]
        frame_layers = data["frame"]
        grid = frame_layers[0]  # convention: first layer is the actual grid
        action = data.get("action_input", {}) or {}
        action_id = action.get("id", 0)
        action_name = ACTION_NAMES.get(action_id, f"?{action_id}")
        action_data = action.get("data", {}) or {}
        state = data.get("state")
        levels = data.get("levels_completed")
        avail = data.get("available_actions")

        if i == 0:
            out_lines.append(f"=== STEP 0 (initial RESET) ===")
            out_lines.append(f"state={state} levels_completed={levels} available_actions={avail}")
            out_lines.append("grid (64x64, hex digits 0-9,a-f = colors 0-15, row-major, origin top-left):")
            out_lines.append(grid_to_ascii(grid))
            prev_grid = grid
            continue

        diffs = diff_cells(prev_grid, grid) if prev_grid is not None else []
        out_lines.append("")
        out_lines.append(f"=== STEP {i}: action={action_name} data={action_data} ===")
        out_lines.append(f"state={state} levels_completed={levels}")
        if not diffs:
            out_lines.append("diff: NO CHANGE (grid identical to previous frame)")
        else:
            out_lines.append(f"diff: {len(diffs)} cell(s) changed")
            shown = diffs[:max_diff_cells]
            for (r, c, old, new) in shown:
                out_lines.append(f"  (row={r},col={c}): {old} -> {new}")
            if len(diffs) > max_diff_cells:
                out_lines.append(f"  ... and {len(diffs) - max_diff_cells} more changed cells (truncated)")
        prev_grid = grid

    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_prefix")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    ap.add_argument("--agent", default="random")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = find_recording(args.game_prefix, args.recordings_dir, args.agent)
    records = load_frames(path, args.steps)
    text = serialize(records)
    header = f"# Source recording: {os.path.basename(path)}\n# Steps serialized: {len(records)}\n\n"
    full = header + text
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(full)
        print(f"Wrote {len(full)} chars to {args.out}")
    else:
        print(full)


if __name__ == "__main__":
    main()
