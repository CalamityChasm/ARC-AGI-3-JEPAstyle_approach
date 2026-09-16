"""How many analysis turns does a level that DOES clear take?

This is how a stall threshold gets chosen without fitting it to the outcome.
Thuitanium set theirs from exactly this distribution ("cleared levels take
median 10 turns, 84% <= 20; the stalled level burns median 32"). Re-derive it on
our own chassis rather than inheriting their number, because our turn budget per
game is set by our own latency, not theirs.

A turn is a distinct `analysis_step`; the unit comes from the transcript headers
(one per LLM call, so duplicates are collapsed) intersected with the level each
step was on, from `artifacts/*_events.jsonl` (type == "action" rows only).

Usage:
    venv/Scripts/python.exe scripts/measure_turns_to_clear.py <run-dir> [...] \
        --json <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simulate_restart_trigger import actions_by_step, turns_from_transcript


def level_spans(run_dir: Path) -> list[dict[str, Any]]:
    """One record per (game, level) the run occupied, with turns spent on it."""
    out = []
    for tpath in sorted((run_dir / "transcripts").glob("*_p0.txt")):
        game = tpath.name[: -len("_p0.txt")]
        ev = run_dir / "artifacts" / f"{game}_p0_events.jsonl"
        if not ev.exists():
            continue
        turns = turns_from_transcript(tpath)
        acts, lvl_at = actions_by_step(ev)
        cur = None
        seq = []
        for s in turns:
            if s in lvl_at:
                cur = lvl_at[s]
            seq.append(cur)

        # Group consecutive turns by level. A level ends (is "cleared") when the
        # next group has a higher level; the final group never cleared.
        groups: list[tuple[int | None, list[int]]] = []
        for s, lv in zip(turns, seq):
            if groups and groups[-1][0] == lv:
                groups[-1][1].append(s)
            else:
                groups.append((lv, [s]))
        for i, (lv, steps) in enumerate(groups):
            if lv is None:
                continue
            cleared = i < len(groups) - 1 and (groups[i + 1][0] or 0) > lv
            out.append(
                {
                    "game_id": game.split("-")[0],
                    "level": lv,
                    "turns": len(steps),
                    "actions": sum(acts.get(s, 0) for s in steps),
                    "cleared": bool(cleared),
                    "final": i == len(groups) - 1,
                }
            )
    return out


def pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        run_dir = Path(path or spec)
        spans = level_spans(run_dir)
        out[name] = spans

        cleared = [s["turns"] for s in spans if s["cleared"]]
        stalled = [s["turns"] for s in spans if s["final"]]
        print(f"\n=== {name} ===")
        print(f"levels cleared: {len(cleared)}   final (never cleared) levels: {len(stalled)}")
        print(
            f"turns to clear      : median {pct(cleared, 0.5)}  p75 {pct(cleared, 0.75)} "
            f" p84 {pct(cleared, 0.84)}  p90 {pct(cleared, 0.90)}  max {max(cleared or [0])}"
        )
        print(
            f"turns on the stall  : median {pct(stalled, 0.5)}  p25 {pct(stalled, 0.25)} "
            f" max {max(stalled or [0])}"
        )
        print()
        print(
            f"{'T':>4}{'cleared >= T':>14}{'(% of cleared)':>16}"
            f"{'stalls >= T':>13}{'(% of stalls)':>15}{'stall turns past T':>20}"
        )
        for t in range(6, 27, 2):
            c = sum(1 for x in cleared if x >= t)
            s = sum(1 for x in stalled if x >= t)
            past = sum(x - t + 1 for x in stalled if x >= t)
            print(
                f"{t:4d}{c:14d}{100 * c / max(len(cleared), 1):15.0f}%"
                f"{s:13d}{100 * s / max(len(stalled), 1):14.0f}%{past:20d}"
            )

        print("\nturns spent on each never-cleared level, by game:")
        rows = sorted(
            [s for s in spans if s["final"]], key=lambda s: -s["turns"]
        )
        line = "  " + "  ".join(
            f"{s['game_id']}:L{s['level']}={s['turns']}t/{s['actions']}a" for s in rows
        )
        for i in range(0, len(line), 110):
            print(line[i : i + 110])

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
