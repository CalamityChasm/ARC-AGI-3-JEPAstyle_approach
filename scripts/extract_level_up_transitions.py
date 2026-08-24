"""Filters a directory of raw Hypothesis-agent recordings down to segments
that end in a real `levels_completed` increase, and packages them into a
curated corpus in the exact same `*.recording.jsonl` schema the rest of the
pipeline already reads (see `jepa/data/trajectories.py:
load_transitions_from_dir`).

Why this exists instead of extract_winning_transitions.py: that script
looked for `state == "WIN"` (the whole game, every level, complete) --
across the full harvest (49,474 frames, 25 games, MAX_ACTIONS=2000/game)
that state never occurs once. Direct inspection of the raw recordings
found why: `win_levels` is a *static* per-game field (e.g. r11l's is
always 6 -- the total level count for that game), not a progress counter,
and the actual progress counter (`levels_completed`) does increase during
play but never once reaches its `win_levels` ceiling in this harvest.
`levels_completed` is also session-cumulative, not reset by a RESET
action (confirmed directly: r11l goes 0->1 at line 57 and then stays at 1
across the remaining 110 in-file resets) -- so "did something good" here
means a `levels_completed` increase, which is exactly the signal this
project's own value-head training (`jepa/data/value_targets.py`,
Stage 5's `NONZERO_THRESHOLD`) has always been built around, not full-WIN.

Usage (from repo root, inside the venv):
    python scripts/extract_level_up_transitions.py \
        --src E:/jepa_overflow/winning_harvest/recordings \
        --dst E:/jepa_overflow/winning_harvest/levelup_corpus
"""

import argparse
import json
from collections import Counter
from pathlib import Path

RESET_ACTION_ID = 0  # GameAction.RESET.value


def _load_lines(path: Path) -> list[dict]:
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


def line_action_id(line: dict) -> int | None:
    action_input = line["data"].get("action_input") or {}
    return action_input.get("id")


def find_level_up_segments(lines: list[dict]) -> list[list[dict]]:
    """Returns one sub-list per levels_completed increase event: [most-
    recent-RESET-before-the-increase, increase-frame] inclusive. Multiple
    increases in one file each get their own (possibly overlapping)
    segment -- standard for mining several successful sub-trajectories out
    of one long multi-attempt recording."""
    segments = []
    prev_level = None
    last_reset_idx = 0
    for i, line in enumerate(lines):
        if line_action_id(line) == RESET_ACTION_ID:
            last_reset_idx = i
        lv = line["data"].get("levels_completed")
        if prev_level is not None and isinstance(lv, int) and lv > prev_level:
            segments.append(lines[last_reset_idx : i + 1])
        prev_level = lv
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract levels_completed-increase segments into a curated corpus.")
    parser.add_argument("--src", required=True, help="Directory of raw *.recording.jsonl harvest files.")
    parser.add_argument("--dst", required=True, help="Output directory for the curated corpus.")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    per_game_events = Counter()
    per_game_transitions = Counter()
    total_files = 0
    total_events = 0
    total_transitions = 0

    for path in sorted(src.glob("*.recording.jsonl")):
        total_files += 1
        lines = _load_lines(path)
        segments = find_level_up_segments(lines)
        if not segments:
            continue

        game_id = lines[0]["data"].get("game_id", "unknown") if lines else "unknown"
        game_prefix = game_id.split("-")[0]

        for seg_idx, segment in enumerate(segments):
            n_transitions = max(len(segment) - 1, 0)
            out_path = dst / path.name.replace(
                ".recording.jsonl", f".levelup{seg_idx}.recording.jsonl"
            )
            with open(out_path, "w", encoding="utf-8") as f:
                for line in segment:
                    f.write(json.dumps(line))
                    f.write("\n")

            total_events += 1
            total_transitions += n_transitions
            per_game_events[game_prefix] += 1
            per_game_transitions[game_prefix] += n_transitions
            print(
                f"[level-up] {path.name} seg{seg_idx} -> {out_path.name} "
                f"(game={game_id}, {len(segment)} frames, {n_transitions} transitions)"
            )

    print("\n=== summary ===")
    print(f"source files scanned: {total_files}")
    print(f"level-up events found: {total_events}")
    print(f"total level-up transitions: {total_transitions}")
    print("\nper-game breakdown (events, transitions):")
    for game in sorted(per_game_events):
        print(f"  {game}: {per_game_events[game]} events, {per_game_transitions[game]} transitions")

    summary = {
        "source_files_scanned": total_files,
        "level_up_events_found": total_events,
        "total_level_up_transitions": total_transitions,
        "per_game_events": dict(per_game_events),
        "per_game_transitions": dict(per_game_transitions),
    }
    (dst / "_harvest_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {dst / '_harvest_summary.json'}")


if __name__ == "__main__":
    main()
