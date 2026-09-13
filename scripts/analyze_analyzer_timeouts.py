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

    feasibility_and_value(args.root)

    if args.json:
        json.dump(tr, open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")




# ---------------------------------------------------------------------------
# Second pass: is the lost time actually recoverable, and what is it worth?
# ---------------------------------------------------------------------------

SUMMARY = re.compile(
    r"(\w{4}-\w{8}): score=([0-9.]+), levels=([0-9.]+)/(\d+), actions=(\d+), tokens=(\d+)"
)


def _tri(n: float) -> float:
    return n * (n + 1) / 2


def feasibility_and_value(root: str) -> None:
    """For each game: could a turn have FIT in the leftover time, and if the
    resulting actions had completed one more level, what would RHAE do?

    RHAE per environment is  E = min(completion_term, efficiency_term).  The
    run's own scorer output lets us see which term binds: where the reported
    score equals the completion term exactly, completion binds and one extra
    level raises E by (k+1)/tri(n); where efficiency binds, an extra level
    cannot raise the min at all, so its marginal value is 0.
    """
    log_path = glob.glob(os.path.join(root, "*.log"))
    summary = {}
    if log_path:
        text = open(log_path[0], encoding="utf-8", errors="replace").read()
        for m in SUMMARY.finditer(text):
            summary[m.group(1)] = (
                float(m.group(2)), float(m.group(3)), int(m.group(4))
            )
    if not summary:
        print("\n(no per-game summary lines found in the kernel log; "
              "skipping the value estimate)")
        return

    tr = {r["game"]: r for r in scan_transcripts(root)}
    ev = scan_events(root)
    all_gaps: list[float] = []
    exp_actions = exp_levels = 0.0
    delta_real = delta_upper = 0.0

    print()
    print(f"{'game':<16}{'k/n':>7}{'E_now%':>8}{'C%':>7}{'binds':>7}"
          f"{'leftover':>9}{'minturn':>8}{'P(fit)':>8}{'E[acts]':>8}{'dE%':>7}")
    for game, row in sorted(tr.items()):
        path = os.path.join(root, "transcripts", f"{game}_p0.txt")
        text = open(path, encoding="utf-8", errors="replace").read()
        times = [_hhmmss(m.group(3)) for m in HDR.finditer(text)]
        gaps = [b - a for a, b in zip(times, times[1:])]
        all_gaps += gaps
        left = row.get("timeout_s") or 0.0
        p_fit = sum(1 for g in gaps if g <= left) / max(1, len(gaps))
        score, k, n = summary[game]
        comp = 100.0 * _tri(k) / _tri(n)
        binds = "compl" if abs(comp - score) < 0.01 else "effic"
        e = ev[game]
        per_turn = e["actions"] / max(1, row["turns"])
        a_real = p_fit * per_turn
        rate = e["actions"] / max(1e-9, row["last_turn_s"] - row["first_turn_s"])
        a_upper = rate * left
        apl = e["actions"] / e["levels_completed"] if e["levels_completed"] else None
        d_e = 100.0 * (k + 1) / _tri(n) if binds == "compl" else 0.0
        if apl:
            exp_levels += a_real / apl
            delta_real += (a_real / apl) * d_e / len(tr)
            delta_upper += (a_upper / apl) * d_e / len(tr)
        exp_actions += a_real
        print(f"{game:<16}{str(int(k)) + '/' + str(n):>7}{score:>8.2f}{comp:>7.2f}"
              f"{binds:>7}{left:>9.1f}{min(gaps):>8.0f}{p_fit * 100:>7.1f}%"
              f"{a_real:>8.2f}{d_e:>7.2f}")

    mean_left = statistics.mean(r["timeout_s"] for r in tr.values() if r.get("timeout_s"))
    print()
    print(f"turn duration pooled (s): n={len(all_gaps)} min={min(all_gaps):.0f} "
          f"p10={statistics.quantiles(all_gaps, n=10)[0]:.0f} "
          f"median={statistics.median(all_gaps):.0f} "
          f"p90={statistics.quantiles(all_gaps, n=10)[8]:.0f} max={max(all_gaps):.0f}")
    print(f"mean leftover = {mean_left:.1f}s; only "
          f"{100.0 * sum(1 for g in all_gaps if g <= mean_left) / len(all_gaps):.1f}%"
          f" of turns ever completed that fast")
    print(f"expected recoverable actions : {exp_actions:.1f}")
    print(f"expected recoverable levels  : {exp_levels:.2f}")
    print(f"RHAE delta, realistic        : +{delta_real:.4f} pts on the public-25 mean")
    print(f"RHAE delta, absolute upper   : +{delta_upper:.4f} pts on the public-25 mean")


if __name__ == "__main__":
    main()
