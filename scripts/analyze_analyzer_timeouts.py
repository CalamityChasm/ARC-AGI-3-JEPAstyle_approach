"""Quantify analyzer read-timeout loss in a Duck/TAAF Kaggle run.

Inputs: the unzipped `kaggle kernels output` directory of a Duck run, which must
contain `transcripts/*.txt`, `artifacts/*_events.jsonl` and `score.json`.

Usage:
    python scripts/analyze_analyzer_timeouts.py <output_dir> [--json out.json]

Everything it prints is derived from primary run artifacts, not from the
kernel's (stderr-buffered, end-of-run-flushed) console log.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter

HDR = re.compile(
    r"--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---"
)
ERR = re.compile(r"request_error: .*?read timeout=([0-9.]+)")
STATUS = re.compile(r"ANALYZER STATUS")


def _hhmmss(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def scan_transcripts(root: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "transcripts", "*.txt"))):
        game = os.path.basename(path).split("_p0")[0]
        text = open(path, encoding="utf-8", errors="replace").read()
        heads = list(HDR.finditer(text))
        turn_times = [_hhmmss(m.group(3)) for m in heads]
        errs = list(ERR.finditer(text))
        statuses = len(STATUS.findall(text))
        row = {
            "game": game,
            "turns": len(heads),
            "analyzer_status_records": statuses,
            "timeouts": len(errs),
            "first_turn_s": turn_times[0] if turn_times else None,
            "last_turn_s": turn_times[-1] if turn_times else None,
        }
        if errs:
            e = errs[0]
            before = [m for m in heads if m.start() < e.start()]
            last = before[-1]
            issued = _hhmmss(last.group(3))
            row.update(
                {
                    "timeout_s": float(e.group(1)),
                    "failed_step": int(last.group(1)),
                    "failed_at_action": int(last.group(2)),
                    "issued_s": issued,
                    "expiry_s": issued + float(e.group(1)),
                }
            )
        # median successful turn duration: consecutive header gaps
        gaps = [b - a for a, b in zip(turn_times, turn_times[1:]) if b >= a]
        row["median_turn_gap_s"] = statistics.median(gaps) if gaps else None
        rows.append(row)
    return rows


def scan_events(root: str) -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "artifacts", "*_events.jsonl"))):
        game = os.path.basename(path).split("_p0")[0]
        actions = 0
        level_up_actions = []
        game_overs = 0
        max_level = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") != "action":
                continue
            actions += 1
            max_level = max(max_level, int(d.get("level") or 0))
            if d.get("level_completed"):
                level_up_actions.append(int(d["action_num"]))
            if d.get("game_over"):
                game_overs += 1
        out[game] = {
            "actions": actions,
            "levels_completed": len(level_up_actions),
            "level_up_actions": level_up_actions,
            "game_overs": game_overs,
            "max_level": max_level,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tr = scan_transcripts(args.root)
    ev = scan_events(args.root)
    for r in tr:
        r.update(ev.get(r["game"], {}))

    total_turns = sum(r["turns"] for r in tr)
    total_timeouts = sum(r["timeouts"] for r in tr)
    total_actions = sum(r.get("actions", 0) for r in tr)
    total_levels = sum(r.get("levels_completed", 0) for r in tr)
    tos = [r["timeout_s"] for r in tr if r.get("timeout_s") is not None]
    exps = [r["expiry_s"] for r in tr if r.get("expiry_s") is not None]

    print(f"games                      : {len(tr)}")
    print(f"analyzer turns (total)     : {total_turns}")
    print(f"ANALYZER STATUS records    : {sum(r['analyzer_status_records'] for r in tr)}")
    print(f"analyzer read timeouts     : {total_timeouts}"
          f"  ({100.0 * total_timeouts / max(1, total_turns):.3f}% of turns)")
    print(f"timeouts per game (counts) : {dict(Counter(r['timeouts'] for r in tr))}")
    print(f"actions (total)            : {total_actions}")
    print(f"levels completed (total)   : {total_levels}")
    if tos:
        print()
        print("observed read-timeout values (s):")
        print(f"  n={len(tos)} min={min(tos):.2f} median={statistics.median(tos):.2f} "
              f"mean={statistics.mean(tos):.2f} max={max(tos):.2f} sum={sum(tos):.1f}")
    if exps:
        print()
        print("wall-clock instant each doomed request expired (s past midnight):")
        print(f"  min={min(exps):.2f} max={max(exps):.2f} "
              f"range={max(exps) - min(exps):.2f}s stdev={statistics.pstdev(exps):.2f}s")
        print("  -> a range of ~1s across all games means ONE shared deadline, not"
              " per-request variation.")

    print()
    hdr = (f"{'game':<16}{'turns':>6}{'acts':>6}{'lvls':>5}{'GO':>4}"
           f"{'fail@act':>9}{'timeout_s':>10}{'med_turn_s':>11}{'tail_acts':>10}")
    print(hdr)
    tail_total = 0.0
    for r in sorted(tr, key=lambda x: -(x.get("timeout_s") or 0)):
        # actions the wasted time could at best have bought, at this game's own rate
        rate = r["actions"] / max(1e-9, (r["last_turn_s"] - r["first_turn_s"])) if r["actions"] else 0.0
        tail = rate * (r.get("timeout_s") or 0.0)
        tail_total += tail
        print(f"{r['game']:<16}{r['turns']:>6}{r.get('actions', 0):>6}"
              f"{r.get('levels_completed', 0):>5}{r.get('game_overs', 0):>4}"
              f"{r.get('failed_at_action', -1):>9}{r.get('timeout_s', 0):>10.2f}"
              f"{(r.get('median_turn_gap_s') or 0):>11.1f}{tail:>10.2f}")

    print()
    print(f"upper-bound actions recoverable (sum of per-game rate x wasted s): {tail_total:.1f}"
          f"  = {100.0 * tail_total / max(1, total_actions):.2f}% of all actions")
    if total_levels:
        apl = total_actions / total_levels
        print(f"actions per level completed (pooled)  : {apl:.1f}")
        print(f"expected extra levels from those acts : {tail_total / apl:.2f}"
              f"  (across {len(tr)} games)")

    if args.json:
        json.dump(tr, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
