"""How much accumulated knowledge does each wipe actually destroy?

The wipe only costs something if the six fields were *populated* when it fired.
This reads the run's own transcripts, which carry the rendered block
``_summarized_knowledge_lines`` injects into every user prompt:

    Working world model carried from earlier turns:
    - World model: ...
    - Goal model: ...
    ...
    - Revise any item above immediately if ...

Pairing each turn's block with the harness's per-turn ``[ANALYZER STATUS]``
record lets us find the turns that follow an in-level game over and measure the
block's size before and after. A wipe that lands on an already-empty block costs
nothing; a wipe that lands on 900 characters of accumulated mechanics costs
those 900 characters.

Usage:
    venv/Scripts/python.exe scripts/measure_wipe_cost.py <run-dir> [<run-dir> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

BLOCK_START = "Working world model carried from earlier turns:"
BLOCK_END = "- Revise any item above immediately"
LABELS = ("World model", "Goal model", "Action model", "Recent findings", "Open questions", "Plan")
SURVIVOR = "Cross-level notes"


# The transcript delimits every analyzer turn with this header. It is the only
# reliable anchor: `_summarized_knowledge_lines` returns [] when all seven
# fields are empty, so the knowledge block is *absent* on those turns and
# counting blocks positionally silently misaligns after the first empty one.
TURN_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \|", re.MULTILINE)


def _parse_block(chunk: str) -> dict[str, str]:
    start = chunk.find(BLOCK_START)
    if start < 0:
        return {}
    end = chunk.find(BLOCK_END, start)
    body = chunk[start + len(BLOCK_START) : end if end >= 0 else len(chunk)]
    fields: dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"^- ([A-Za-z -]+): (.*)$", line.strip())
        if m and m.group(1) in (*LABELS, SURVIVOR):
            fields[m.group(1)] = m.group(2)
    return fields


def blocks(transcript: str) -> dict[int, dict[str, str]]:
    """Knowledge block per analysis_step; an absent block is an empty one."""
    marks = list(TURN_RE.finditer(transcript))
    out: dict[int, dict[str, str]] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(transcript)
        out[int(m.group(1))] = _parse_block(transcript[m.start() : end])
    return out


def _game_over_steps(events_path: Path) -> list[int]:
    """analysis_step of every real in-level game over (type=='action' rows)."""
    steps: list[int] = []
    prev_level: int | None = None
    pending: dict[int, dict[str, bool]] = {}
    with events_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "action":
                continue
            step = rec.get("analysis_step")
            lvl = rec.get("level")
            tx = prev_level is not None and lvl is not None and int(lvl) != int(prev_level)
            if lvl is not None:
                prev_level = int(lvl)
            if step is None:
                continue
            slot = pending.setdefault(int(step), {"go": False, "tx": False})
            slot["go"] |= rec.get("state") == "GAME_OVER"
            slot["tx"] |= tx
    for step, flags in sorted(pending.items()):
        if flags["go"] and not flags["tx"]:
            steps.append(step)
    return steps


def analyse(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "benchmark.json").open(encoding="utf-8") as fh:
        runs = json.load(fh)["game_runs"]

    destroyed: list[int] = []
    zero_cost = 0
    per_game = []
    for run in sorted(runs, key=lambda r: r["game_id"]):
        gid = run["game_id"]
        ev = run_dir / "artifacts" / f"{gid}_p0_events.jsonl"
        tr = run_dir / "transcripts" / f"{gid}_p0.txt"
        if not ev.exists() or not tr.exists():
            continue
        steps = _game_over_steps(ev)
        if not steps:
            continue
        bs = blocks(tr.read_text(encoding="utf-8", errors="replace"))
        rows = []
        for step in steps:
            # The block shown on turn N reflects the knowledge state *entering*
            # turn N, before the wipe that turn's own game over triggers. The
            # cost of that wipe is therefore what stood on turn N.
            if step not in bs:
                continue
            before = bs[step]
            after = bs.get(step + 1, {})
            size_before = sum(len(before.get(k, "")) for k in LABELS)
            size_after = sum(len(after.get(k, "")) for k in LABELS)
            rows.append({"step": step, "chars_before": size_before, "chars_after": size_after})
            destroyed.append(size_before)
            if size_before == 0:
                zero_cost += 1
        if rows:
            per_game.append({"game_id": gid, "wipes": len(rows), "rows": rows})

    return {
        "run_dir": str(run_dir),
        "in_level_game_over_wipes": len(destroyed),
        "wipes_on_an_empty_world_model": zero_cost,
        "chars_destroyed_total": sum(destroyed),
        "chars_destroyed_median": int(statistics.median(destroyed)) if destroyed else 0,
        "chars_destroyed_max": max(destroyed) if destroyed else 0,
        "per_game": per_game,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results = {}
    for run in args.runs:
        r = analyse(run)
        results[run.name] = r
        print(
            f"{run.name:<12} in-level-GO wipes={r['in_level_game_over_wipes']:>3} "
            f"on-empty={r['wipes_on_an_empty_world_model']:>3} "
            f"chars destroyed: total={r['chars_destroyed_total']:>6} "
            f"median={r['chars_destroyed_median']:>5} max={r['chars_destroyed_max']:>5}"
        )
        for g in r["per_game"]:
            sizes = ", ".join(str(row["chars_before"]) for row in g["rows"])
            print(f"    {g['game_id']:<16} {g['wipes']:>2} wipes, chars at wipe: {sizes}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
