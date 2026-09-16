"""Is a stall a *fixation*? Measure exact action repetition, and when it starts.

A restart-at-stall only pays if the turns it discards were going to be wasted.
The cheapest evidence for that is repetition: an action identical to one already
taken on the same level, at the same level, is by construction telling the agent
nothing it has not already seen (the environment is deterministic).

For each game this reports the exact-repeat fraction overall, and split at the
point where the last level change happened -- i.e. before vs. inside the final
stall. A pathology that is flat across that boundary is not a stall pathology.

Usage:
    venv/Scripts/python.exe scripts/measure_stall_repetition.py <run-dir> [...] \
        --json <out.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_actions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as fh:
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
            rows.append(
                {
                    "level": rec.get("level"),
                    "act": rec.get("action_display") or rec.get("action_name"),
                    "step": rec.get("analysis_step"),
                    "game_over": bool(rec.get("game_over")),
                    "changed": bool(rec.get("board_changed")),
                }
            )
    return rows


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    # Index of the first action after the last level change.
    last_change = 0
    for i in range(1, len(rows)):
        if rows[i]["level"] != rows[i - 1]["level"]:
            last_change = i

    seen: dict[tuple, int] = {}
    repeat_flags: list[bool] = []
    for r in rows:
        key = (r["level"], r["act"])
        repeat_flags.append(key in seen)
        seen[key] = seen.get(key, 0) + 1

    def frac(lo: int, hi: int) -> tuple[int, int]:
        seg = repeat_flags[lo:hi]
        return sum(seg), len(seg)

    pre_r, pre_n = frac(0, last_change)
    post_r, post_n = frac(last_change, len(rows))
    all_r, all_n = frac(0, len(rows))

    # The most-repeated single action inside the final stall.
    tail: dict[tuple, int] = {}
    for r in rows[last_change:]:
        tail[(r["level"], r["act"])] = tail.get((r["level"], r["act"]), 0) + 1
    top = max(tail.items(), key=lambda kv: kv[1]) if tail else ((None, None), 0)

    return {
        "n_actions": all_n,
        "repeat_all": all_r,
        "repeat_frac_all": all_r / max(all_n, 1),
        "pre_stall_n": pre_n,
        "pre_stall_repeat_frac": (pre_r / pre_n) if pre_n else None,
        "stall_n": post_n,
        "stall_repeat_frac": (post_r / post_n) if post_n else None,
        "stall_distinct_actions": len(tail),
        "stall_top_action": str(top[0][1]),
        "stall_top_count": top[1],
        "stall_top_share": top[1] / max(post_n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        run_dir = Path(path or spec)
        per = {}
        print(f"\n=== {name} ===")
        hdr = (
            f"{'game':6}{'n':>6}{'rep%':>6}{'preN':>6}{'pre%':>6}"
            f"{'stalN':>7}{'stal%':>7}{'distinct':>9}{'topN':>6}{'top%':>6}  top action"
        )
        print(hdr)
        print("-" * (len(hdr) + 10))
        tot_stall_n = tot_stall_rep = 0
        for ev in sorted((run_dir / "artifacts").glob("*_p0_events.jsonl")):
            gid = ev.name.split("-")[0]
            s = stats(load_actions(ev))
            if not s:
                continue
            per[gid] = s
            tot_stall_n += s["stall_n"]
            tot_stall_rep += int(round(s["stall_repeat_frac"] * s["stall_n"]))
            pf = f"{100 * s['pre_stall_repeat_frac']:.0f}" if s["pre_stall_repeat_frac"] is not None else "-"
            sf = f"{100 * s['stall_repeat_frac']:.0f}" if s["stall_repeat_frac"] is not None else "-"
            print(
                f"{gid:6}{s['n_actions']:6d}{100 * s['repeat_frac_all']:6.0f}"
                f"{s['pre_stall_n']:6d}{pf:>6}"
                f"{s['stall_n']:7d}{sf:>7}{s['stall_distinct_actions']:9d}"
                f"{s['stall_top_count']:6d}{100 * s['stall_top_share']:5.0f}%"
                f"  {s['stall_top_action']}"
            )
        out[name] = per
        print("-" * (len(hdr) + 10))
        print(
            f"actions inside the final stall: {tot_stall_n}; "
            f"exact repeats among them: {tot_stall_rep} "
            f"({100 * tot_stall_rep / max(tot_stall_n, 1):.1f}%)"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
