"""Filters a directory of raw Hypothesis-agent recordings down to just the
transitions from episodes that actually reached a WIN, and packages them
into a curated corpus in the exact same `*.recording.jsonl` schema the rest
of the pipeline already reads (see `jepa/data/trajectories.py:
load_transitions_from_dir`), so a future training run can point at the
output directory directly with zero extra glue.

Why not just keep the raw harvest directory as-is: `Hypothesis.MAX_ACTIONS`
was bumped well above its normal 300 for this harvest specifically so a
single subprocess call gets many RESET-to-GAME_OVER attempts per game, not
just one -- most of those attempts do *not* reach WIN. Each raw recording
file is really a concatenation of many separate episode attempts back to
back (segments separated by RESET actions), and this script's job is to
find the *one* segment (if any) in each file that ends in a WIN frame and
extract just that segment -- the actual winning trajectory, not the many
losing attempts alongside it in the same file.

Usage (from repo root, inside the venv):
    python scripts/extract_winning_transitions.py \
        --src E:/jepa_overflow/winning_harvest/recordings \
        --dst E:/jepa_overflow/winning_harvest/winning_corpus
"""

import argparse
import json
from collections import Counter
from pathlib import Path

RESET_ACTION_ID = 0  # GameAction.RESET.value


def _load_lines(path: Path) -> list[dict]:
    """Returns the raw parsed JSON lines (each `{"timestamp":..., "data":...}`),
    not just the `data` payload -- extraction re-emits lines in the exact
    same wrapper shape the framework's own Recorder writes, so output files
    are byte-for-byte-compatible with an ordinary recording file."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw)
            data = event.get("data", {})
            if "frame" in data and "action_input" in data and data["frame"]:
                lines.append(event)
    return lines


def find_winning_segment(lines: list[dict]) -> list[dict] | None:
    """Returns the sub-list of lines spanning [most-recent-RESET-before-the-
    WIN, WIN] inclusive, or None if this file never reaches WIN.

    Segment boundaries: whenever a line's `action_input.id == RESET`, that
    line is the *result* of the reset (agent.py's `append_frame` is only
    called with post-action frames) and marks the start of a fresh
    attempt -- so the winning segment starts at the most recent such line
    at or before the WIN frame, not at the very start of the file (which
    would incorrectly include every earlier failed attempt too).
    """
    win_idx = None
    for i, line in enumerate(lines):
        if line["data"].get("state") == "WIN":
            win_idx = i
            break  # is_done() stops the agent loop right after WIN, so
            # this is necessarily the last line anyway -- break defensively.
    if win_idx is None:
        return None

    segment_start = 0
    for i in range(win_idx, -1, -1):
        action = line_action_id(lines[i])
        if action == RESET_ACTION_ID:
            segment_start = i
            break
    return lines[segment_start : win_idx + 1]


def line_action_id(line: dict) -> int | None:
    action_input = line["data"].get("action_input") or {}
    return action_input.get("id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WIN-episode transitions into a curated corpus.")
    parser.add_argument("--src", required=True, help="Directory of raw *.recording.jsonl harvest files.")
    parser.add_argument("--dst", required=True, help="Output directory for the curated winning-only corpus.")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    per_game_wins = Counter()
    per_game_transitions = Counter()
    total_files = 0
    total_wins = 0
    total_transitions = 0

    for path in sorted(src.glob("*.recording.jsonl")):
        total_files += 1
        lines = _load_lines(path)
        segment = find_winning_segment(lines)
        if segment is None:
            continue

        game_id = segment[-1]["data"].get("game_id", "unknown")
        game_prefix = game_id.split("-")[0]
        n_transitions = max(len(segment) - 1, 0)

        out_path = dst / path.name.replace(".recording.jsonl", ".win.recording.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for line in segment:
                f.write(json.dumps(line))
                f.write("\n")

        total_wins += 1
        total_transitions += n_transitions
        per_game_wins[game_prefix] += 1
        per_game_transitions[game_prefix] += n_transitions
        print(
            f"[win] {path.name} -> {out_path.name} "
            f"(game={game_id}, {len(segment)} frames, {n_transitions} transitions)"
        )

    print("\n=== summary ===")
    print(f"source files scanned: {total_files}")
    print(f"winning episodes found: {total_wins}")
    print(f"total winning transitions: {total_transitions}")
    print("\nper-game breakdown (wins, transitions):")
    for game in sorted(per_game_wins):
        print(f"  {game}: {per_game_wins[game]} wins, {per_game_transitions[game]} transitions")

    summary = {
        "source_files_scanned": total_files,
        "winning_episodes_found": total_wins,
        "total_winning_transitions": total_transitions,
        "per_game_wins": dict(per_game_wins),
        "per_game_transitions": dict(per_game_transitions),
    }
    (dst / "_harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {dst / '_harvest_summary.json'}")


if __name__ == "__main__":
    main()
