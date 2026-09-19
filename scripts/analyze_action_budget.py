"""Where do our actions actually go, and how many are sunk?

This supersedes `scripts/analyze_rhae_binding.py`'s waste figure, which was
computed from `artifacts/*_events.jsonl` by counting every row with a non-null
`action_num`. That over-counts: the event stream mirrors each `type == "action"`
row with a `type == "analysis"` row carrying the same `action_num`, and also
emits a leading `type == "initial"` row. On the nvfp4 baseline that inflated
3,633 real actions to 4,970 rows (+36.9%).

This script does not reconstruct anything. The harness prints one authoritative
`[finished]` line per game into the kernel log:

    [finished] sp80-589a99af state=gave_up level=0/6 score=0.00 actions=215
               tokens=64810 per-level=215/39,0/58,0/25,0/148,0/96,0/152

`per-level` is `ours/human` for every level of the game, `level=k/M` is levels
completed out of total, `score` is the reported per-game RHAE E_e as a
percentage. Everything needed is in that one line, and it cross-checks against
itself (the per-level "ours" values sum to `actions`) and against
`benchmark.json` / `score.json`.

Definitions used here:

  sunk actions   actions spent on levels that were never completed. With
                 `level=k/M`, levels 1..k completed and level k+1 is the one in
                 progress when the clock ran out, so `sum(ours[k:])` is sunk.
  engaged        levels with >0 of our actions.

Usage:
    venv/Scripts/python.exe scripts/analyze_action_budget.py \
        anim=<dir> baseline=<dir> --json experiments/stage7_action_budget.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FINISHED_RE = re.compile(
    r"\[finished\]\s+(?P<game>\S+)\s+state=(?P<state>\S+)\s+"
    r"level=(?P<done>\d+)/(?P<total>\d+)\s+score=(?P<score>[-\d.]+)\s+"
    r"actions=(?P<actions>\d+)\s+tokens=(?P<tokens>\d+)\s+"
    r"per-level=(?P<per_level>[\d/,]+)"
)


def find_log(run_dir: Path) -> Path:
    """The kernel log is the one *.log that is not a vLLM server log."""
    cands = [
        p
        for p in sorted(run_dir.glob("*.log"))
        if not p.name.startswith("vllm-")
    ]
    if not cands:
        raise FileNotFoundError(f"no kernel log in {run_dir}")
    return cands[0]


def parse_finished(log_path: Path) -> list[dict[str, Any]]:
    """Every [finished] line in a kernel log, newest-wins on duplicates.

    The log is a JSON-ish stream of {"data": "<line>"} records; the marker is
    matched inside the raw text so the exact container format does not matter.
    """
    out: dict[str, dict[str, Any]] = {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for m in FINISHED_RE.finditer(text):
        per_level = []
        for pair in m.group("per_level").split(","):
            ours_s, _, human_s = pair.partition("/")
            per_level.append((int(ours_s), int(human_s)))
        rec = {
            "game": m.group("game"),
            "game_id": m.group("game").split("-")[0],
            "state": m.group("state"),
            "levels_done": int(m.group("done")),
            "levels_total": int(m.group("total")),
            "score": float(m.group("score")),
            "actions": int(m.group("actions")),
            "tokens": int(m.group("tokens")),
            "per_level": per_level,
        }
        out[rec["game"]] = rec
    return sorted(out.values(), key=lambda r: r["game_id"])


def enrich(rec: dict[str, Any]) -> dict[str, Any]:
    ours = [a for a, _ in rec["per_level"]]
    human = [h for _, h in rec["per_level"]]
    k = rec["levels_done"]

    # Self-consistency: the per-level "ours" must sum to `actions`.
    rec["per_level_sum"] = sum(ours)
    rec["per_level_consistent"] = sum(ours) == rec["actions"]

    # Levels 1..k completed; k+1 onwards never completed.
    rec["sunk_actions"] = sum(ours[k:])
    rec["productive_actions"] = sum(ours[:k])
    rec["stall_level"] = k + 1 if k < len(ours) else None
    rec["stall_level_actions"] = ours[k] if k < len(ours) else 0
    rec["stall_level_human"] = human[k] if k < len(human) else 0
    rec["stall_ratio"] = (
        ours[k] / human[k] if k < len(ours) and k < len(human) and human[k] else None
    )
    rec["levels_engaged"] = sum(1 for a in ours if a > 0)
    rec["human_total"] = sum(human)
    # Efficiency on the levels we completed: S_l = min(1.15, h/a)**2
    rec["S_l"] = [
        round(min(1.15, (human[i] / ours[i]) if ours[i] else 0.0) ** 2, 4)
        for i in range(k)
    ]
    return rec


def analyse(run_dir: Path) -> dict[str, Any]:
    log = find_log(run_dir)
    recs = [enrich(r) for r in parse_finished(log)]

    res: dict[str, Any] = {"run_dir": str(run_dir), "log": log.name, "games": recs}

    # Cross-check the per-game action totals against benchmark.json, which is
    # written by a different code path.
    bench = run_dir / "benchmark.json"
    if bench.exists():
        with bench.open(encoding="utf-8") as fh:
            runs = json.load(fh)["game_runs"]
        by_id = {r["game_id"]: r for r in runs}
        mism = []
        for rec in recs:
            b = by_id.get(rec["game"]) or by_id.get(rec["game_id"])
            if b is None:
                continue
            ba = b.get("actions_taken", b.get("total_actions"))
            if ba is not None and int(ba) != rec["actions"]:
                mism.append((rec["game_id"], rec["actions"], int(ba)))
        res["benchmark_mismatches"] = mism

    # Cross-check the reported per-game score against score.json.
    sj = run_dir / "score.json"
    if sj.exists():
        with sj.open(encoding="utf-8") as fh:
            games = json.load(fh)["games"]
        mism = []
        for rec in recs:
            s = games.get(rec["game"], games.get(rec["game_id"], {})).get("score")
            if s is not None and abs(float(s) - rec["score"]) > 0.01:
                mism.append((rec["game_id"], rec["score"], float(s)))
        res["score_mismatches"] = mism

    res["totals"] = {
        "games": len(recs),
        "actions": sum(r["actions"] for r in recs),
        "sunk_actions": sum(r["sunk_actions"] for r in recs),
        "productive_actions": sum(r["productive_actions"] for r in recs),
        "levels_done": sum(r["levels_done"] for r in recs),
        "levels_total": sum(r["levels_total"] for r in recs),
        "human_total": sum(r["human_total"] for r in recs),
        "mean_score": (
            sum(r["score"] for r in recs) / len(recs) if recs else 0.0
        ),
        "per_level_consistent": all(r["per_level_consistent"] for r in recs),
    }
    t = res["totals"]
    t["sunk_fraction"] = t["sunk_actions"] / max(t["actions"], 1)
    return res


def report(name: str, res: dict[str, Any]) -> None:
    t = res["totals"]
    print(f"\n=== {name}  ({res['log']}) ===")
    hdr = (
        f"{'game':6}{'st':>4}{'lv':>7}{'score':>7}{'act':>6}{'sunk':>6}"
        f"{'sunk%':>7}{'stall':>6}{'ours/hum':>10}{'x':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(res["games"], key=lambda r: -r["sunk_actions"]):
        ratio = f"{r['stall_ratio']:.2f}" if r["stall_ratio"] else "-"
        print(
            f"{r['game_id']:6}{r['levels_done']:>4}"
            f"{str(r['levels_done']) + '/' + str(r['levels_total']):>7}"
            f"{r['score']:7.2f}{r['actions']:6d}{r['sunk_actions']:6d}"
            f"{100 * r['sunk_actions'] / max(r['actions'], 1):6.0f}%"
            f"{str(r['stall_level']):>6}"
            f"{str(r['stall_level_actions']) + '/' + str(r['stall_level_human']):>10}"
            f"{ratio:>6}"
        )
    print("-" * len(hdr))
    print(
        f"TOTAL  actions={t['actions']}  sunk={t['sunk_actions']} "
        f"({100 * t['sunk_fraction']:.1f}%)  productive={t['productive_actions']}"
    )
    print(
        f"       levels {t['levels_done']}/{t['levels_total']}  "
        f"public-25 mean {t['mean_score']:.2f}  "
        f"per-level self-consistent: {t['per_level_consistent']}"
    )
    if res.get("benchmark_mismatches"):
        print(f"       !! benchmark.json mismatches: {res['benchmark_mismatches']}")
    else:
        print("       benchmark.json action totals: MATCH")
    if res.get("score_mismatches"):
        print(f"       !! score.json mismatches: {res['score_mismatches']}")
    else:
        print("       score.json per-game scores: MATCH")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path pairs")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).name, spec
        res = analyse(Path(path))
        out[name] = res
        report(name, res)

    if len(out) == 2:
        (an, a), (bn, b) = list(out.items())
        print(f"\n=== per-game delta: {an} - {bn} ===")
        bi = {g["game_id"]: g for g in b["games"]}
        hdr = f"{'game':6}{'dScore':>9}{'dAct':>7}{'dSunk':>7}{'dLevels':>9}"
        print(hdr)
        print("-" * len(hdr))
        for g in sorted(a["games"], key=lambda g: g["game_id"]):
            o = bi.get(g["game_id"])
            if not o:
                continue
            print(
                f"{g['game_id']:6}{g['score'] - o['score']:9.2f}"
                f"{g['actions'] - o['actions']:7d}"
                f"{g['sunk_actions'] - o['sunk_actions']:7d}"
                f"{g['levels_done'] - o['levels_done']:9d}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
