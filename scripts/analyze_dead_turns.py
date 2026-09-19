"""Where the wall clock goes: LLM calls that execute no action, and the terminal
dead streaks that end most games.

Each `[ANALYZER STATUS]` block in a transcript is one LLM call and reports
`step_executed:` and a `message:`. A call that yields on `turn_time_budget`
without executing leaves the solver to retry the *same* `analysis_step`, so a
game whose model stops acting burns the rest of its 7,920 s clock on retries
that cannot move the board.

Counted, not sampled: every call has exactly one status block.

Usage:
    venv/Scripts/python.exe scripts/analyze_dead_turns.py <run-dir> [...] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

HEADER_RE = re.compile(
    r"^--- analysis_step=(?P<step>\d+) \| action=(?P<action>\d+) \| (?P<time>\d\d:\d\d:\d\d) \|"
)
EXEC_RE = re.compile(r"^step_executed: (True|False)")
MSG_RE = re.compile(r"^message: (.*)$")


def parse_transcript(path: Path) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEADER_RE.match(line)
        if m:
            cur = {
                "step": int(m["step"]),
                "action": int(m["action"]),
                "time": m["time"],
                "executed": None,
                "message": None,
            }
            calls.append(cur)
            continue
        if cur is None:
            continue
        m = EXEC_RE.match(line)
        if m and cur["executed"] is None:
            cur["executed"] = m.group(1) == "True"
            continue
        m = MSG_RE.match(line)
        if m and cur["message"] is None:
            cur["message"] = m.group(1)
    return calls


def _secs(calls: list[dict[str, Any]]) -> list[float]:
    """Wall-clock seconds of each call start, relative to the first, un-wrapping midnight."""
    out, prev, off = [], None, 0.0
    for c in calls:
        t = datetime.strptime(c["time"], "%H:%M:%S")
        if prev is not None and t < prev:
            off += 86400.0
        prev = t
        out.append((t - datetime.strptime(calls[0]["time"], "%H:%M:%S")).total_seconds() + off)
    return out


def analyse_game(path: Path, wall: float | None = None) -> dict[str, Any]:
    calls = parse_transcript(path)
    assert calls, f"no calls in {path}"
    secs = _secs(calls)
    n = len(calls)
    executed = sum(1 for c in calls if c["executed"])
    dead = n - executed

    # Terminal dead streak: the run of trailing calls that executed nothing.
    tail = 0
    for c in reversed(calls):
        if c["executed"]:
            break
        tail += 1
    # Seconds from the start of the first call in the streak to the end of the game.
    if tail:
        t_streak_start = secs[n - tail]
        t_end = wall if wall is not None else secs[-1]
        # secs are relative to the first call, which starts a little after t=0.
        tail_secs = max(0.0, t_end - t_streak_start)
    else:
        tail_secs = 0.0

    # The longest interior dead streak (not at the end).
    longest, cur = 0, 0
    for c in calls:
        if c["executed"]:
            longest = max(longest, cur)
            cur = 0
        else:
            cur += 1
    longest = max(longest, cur)

    budgets = sum(1 for c in calls if c["message"] and "turn_time_budget" in c["message"])

    # Wall clock attributed to each turn: from its own header to the next one
    # (and, for the last turn, to the end of the game's clock).
    end = wall if wall is not None else secs[-1]
    durs = [b - a for a, b in zip(secs, secs[1:])] + [max(0.0, end - secs[-1])]
    dead_s = sum(d for c, d in zip(calls, durs) if not c["executed"])
    exec_s = sum(d for c, d in zip(calls, durs) if c["executed"])

    return {
        "dead_seconds": dead_s,
        "exec_seconds": exec_s,
        "median_dead_turn_s": statistics.median(
            [d for c, d in zip(calls, durs) if not c["executed"]] or [0.0]
        ),
        "median_exec_turn_s": statistics.median(
            [d for c, d in zip(calls, durs) if c["executed"]] or [0.0]
        ),
        "game": path.name.split("-")[0],
        "calls": n,
        "executed": executed,
        "dead": dead,
        "dead_frac": dead / n,
        "distinct_steps": len({c["step"] for c in calls}),
        "max_step": max(c["step"] for c in calls),
        "tail_dead_calls": tail,
        "tail_dead_seconds": tail_secs,
        "longest_dead_streak": longest,
        "yield_turn_time_budget": budgets,
        "last_action_num": max(c["action"] for c in calls),
        "median_call_gap_s": statistics.median(
            [b - a for a, b in zip(secs, secs[1:])] or [0.0]
        ),
    }


def analyse(run_dir: Path) -> dict[str, Any]:
    bench = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    walls = {g["game_id"].split("-")[0]: g["final_wallclock_seconds"] for g in bench["game_runs"]}
    info = {
        g["game_id"].split("-")[0]: (
            g["final_score"],
            g["levels_completed"],
            g["number_of_levels"],
            sum(g["actions_per_level"]),
        )
        for g in bench["game_runs"]
    }
    rows = []
    for p in sorted((run_dir / "transcripts").glob("*_p0.txt")):
        g = p.name.split("-")[0]
        r = analyse_game(p, walls.get(g))
        r["score"], r["levels"], r["n_levels"], r["actions"] = info[g]
        r["tail_dead_frac_of_clock"] = r["tail_dead_seconds"] / walls[g]
        rows.append(r)

    tot_calls = sum(r["calls"] for r in rows)
    tot_dead = sum(r["dead"] for r in rows)
    tot_tail = sum(r["tail_dead_seconds"] for r in rows)
    tot_wall = sum(walls.values())
    return {
        "run": str(run_dir),
        "calls": tot_calls,
        "executed": tot_calls - tot_dead,
        "dead": tot_dead,
        "dead_frac": tot_dead / tot_calls,
        "tail_dead_calls": sum(r["tail_dead_calls"] for r in rows),
        "tail_dead_seconds": tot_tail,
        "wall_seconds": tot_wall,
        "tail_dead_frac_of_clock": tot_tail / tot_wall,
        "dead_seconds": sum(r["dead_seconds"] for r in rows),
        "dead_frac_of_clock": sum(r["dead_seconds"] for r in rows) / tot_wall,
        "games": rows,
    }


def report(res: dict[str, Any]) -> None:
    print(f"\n=== {res['run']}")
    print(
        f"LLM calls {res['calls']}  executed {res['executed']}  dead {res['dead']} "
        f"({res['dead_frac']:.1%})"
    )
    print(
        f"wall clock in turns that executed nothing: {res['dead_seconds']:.0f}s of "
        f"{res['wall_seconds']:.0f}s ({res['dead_frac_of_clock']:.1%})"
    )
    print(
        f"terminal dead streaks: {res['tail_dead_calls']} calls, "
        f"{res['tail_dead_seconds']:.0f}s of {res['wall_seconds']:.0f}s "
        f"({res['tail_dead_frac_of_clock']:.1%} of the whole run's clock)"
    )
    print(
        f"\n{'game':<6}{'score':>7}{'lvl':>6}{'calls':>7}{'exec':>6}{'dead':>6}"
        f"{'steps':>7}{'tail':>6}{'tail_s':>8}{'tail%':>7}{'longest':>8}"
    )
    for r in sorted(res["games"], key=lambda r: -r["tail_dead_seconds"]):
        print(
            f"{r['game']:<6}{r['score']:>7.2f}{str(r['levels'])+'/'+str(r['n_levels']):>6}"
            f"{r['calls']:>7}{r['executed']:>6}{r['dead']:>6}{r['distinct_steps']:>7}"
            f"{r['tail_dead_calls']:>6}{r['tail_dead_seconds']:>8.0f}"
            f"{r['tail_dead_frac_of_clock']:>6.0%}{r['longest_dead_streak']:>8}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).name, name
        res = analyse(Path(path))
        report(res)
        out[name] = res
    if args.json:
        args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
