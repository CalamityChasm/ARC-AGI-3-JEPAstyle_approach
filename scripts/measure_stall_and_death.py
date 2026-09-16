"""Size the two candidate mechanisms against the same run, head to head.

Restart-at-stall needs *stalls*: long runs of analysis turns on one level with
no level change. Death blacklist needs *deaths*: repeated game overs on one
level, and -- for the budget-inference half -- three or more of them whose
action totals agree.

Both are counted from `artifacts/*_events.jsonl`, filtered to `type == "action"`
rows (the stream mirrors each of those with a `type == "analysis"` row carrying
the same `action_num`, so an unfiltered count roughly doubles game overs and
inflates actions ~37%).

Usage:
    venv/Scripts/python.exe scripts/measure_stall_and_death.py <run-dir> [...] \
        --json experiments/stage7_action_budget_sizing.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_action_budget import analyse as budget_analyse


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
            rec.pop("board", None)
            rec.pop("board_ascii", None)
            rows.append(rec)
    return rows


def game_stats(path: Path) -> dict[str, Any]:
    rows = load_actions(path)
    if not rows:
        return {}

    steps = sorted({int(r["analysis_step"]) for r in rows if r.get("analysis_step") is not None})
    step_index = {s: i for i, s in enumerate(steps)}

    # Level changes, indexed by position in the acting-turn sequence.
    level_change_at: list[int] = []
    prev_level = None
    for r in rows:
        lvl = r.get("level")
        if lvl is None:
            continue
        if prev_level is not None and lvl != prev_level:
            level_change_at.append(step_index.get(int(r["analysis_step"]), 0))
        prev_level = lvl

    # Stall runs measured in ACTING turns (the unit the harness can observe
    # cheaply). The final run is the one that never ended.
    bounds = [0] + level_change_at + [len(steps)]
    runs = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    final_stall = runs[-1] if runs else 0

    # Deaths: a game_over flag on an action row. Record the level and how many
    # actions had been taken on that level since the previous death/level entry.
    deaths = []
    level_life_actions: dict[int, int] = {}
    cur_life = 0
    cur_level = rows[0].get("level")
    for r in rows:
        lvl = r.get("level")
        if lvl != cur_level:
            cur_level, cur_life = lvl, 0
        cur_life += 1
        if r.get("game_over"):
            deaths.append(
                {
                    "level": lvl,
                    "life_actions": cur_life,
                    "action_name": r.get("action_name") or r.get("action_display"),
                    "analysis_step": r.get("analysis_step"),
                }
            )
            level_life_actions.setdefault(int(lvl or 0), 0)
            cur_life = 0

    # Budget inference, Thuitanium's rule: >= 3 lives on the same level whose
    # totals agree within +-1 -> the median total is the inferred budget.
    by_level: dict[int, list[int]] = {}
    for d in deaths:
        by_level.setdefault(int(d["level"] or 0), []).append(d["life_actions"])
    inferred = {}
    for lvl, totals in by_level.items():
        if len(totals) < 3:
            continue
        s = sorted(totals)
        # any window of >=3 agreeing within +-1
        for i in range(len(s) - 2):
            window = [v for v in s if abs(v - s[i]) <= 1]
            if len(window) >= 3:
                inferred[lvl] = sorted(window)[len(window) // 2]
                break

    return {
        "acting_turns": len(steps),
        "level_changes": len(level_change_at),
        "stall_runs": runs,
        "final_stall_turns": final_stall,
        "max_stall_turns": max(runs) if runs else 0,
        "deaths": len(deaths),
        "deaths_by_level": {str(k): v for k, v in by_level.items()},
        "inferred_budgets": {str(k): v for k, v in inferred.items()},
        "death_actions": [d["action_name"] for d in deaths],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path pairs")
    ap.add_argument("--stall-threshold", type=int, default=20)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    out: dict[str, Any] = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        run_dir = Path(path or spec)
        budget = budget_analyse(run_dir)
        per_game = {}
        for g in budget["games"]:
            ev = run_dir / "artifacts" / f"{g['game']}_p0_events.jsonl"
            st = game_stats(ev) if ev.exists() else {}
            per_game[g["game_id"]] = {**g, **st}
        out[name] = per_game

        print(f"\n=== {name} ===")
        hdr = (
            f"{'game':6}{'score':>7}{'lv':>6}{'act':>6}{'sunk':>6}{'turns':>7}"
            f"{'lvchg':>6}{'stall':>7}{'maxstl':>7}{'deaths':>7}{'budg':>6}"
        )
        print(hdr)
        print("-" * len(hdr))
        tot_stallable = 0
        tot_sunk_stallable = 0
        for gid, g in sorted(per_game.items(), key=lambda kv: -kv[1]["sunk_actions"]):
            fires = g.get("final_stall_turns", 0) >= args.stall_threshold
            if fires:
                tot_stallable += 1
                tot_sunk_stallable += g["sunk_actions"]
            print(
                f"{gid:6}{g['score']:7.2f}"
                f"{str(g['levels_done']) + '/' + str(g['levels_total']):>6}"
                f"{g['actions']:6d}{g['sunk_actions']:6d}"
                f"{g.get('acting_turns', 0):7d}{g.get('level_changes', 0):6d}"
                f"{g.get('final_stall_turns', 0):7d}{g.get('max_stall_turns', 0):7d}"
                f"{g.get('deaths', 0):7d}"
                f"{len(g.get('inferred_budgets', {})):6d}"
            )
        print("-" * len(hdr))
        n = len(per_game)
        print(
            f"games whose FINAL stall >= {args.stall_threshold} acting turns: "
            f"{tot_stallable}/{n}   sunk actions in them: {tot_sunk_stallable}"
        )
        deaths = sum(g.get("deaths", 0) for g in per_game.values())
        gdeath = sum(1 for g in per_game.values() if g.get("deaths", 0) > 0)
        budg = sum(len(g.get("inferred_budgets", {})) for g in per_game.values())
        gbudg = sum(1 for g in per_game.values() if g.get("inferred_budgets"))
        print(
            f"deaths: {deaths} across {gdeath}/{n} games; "
            f"levels with an inferrable budget (>=3 lives agreeing +-1): "
            f"{budg} across {gbudg}/{n} games"
        )
        for gid, g in sorted(per_game.items()):
            if g.get("inferred_budgets"):
                print(
                    f"   {gid}: budgets {g['inferred_budgets']}  "
                    f"lives {g['deaths_by_level']}"
                )
        zero = [gid for gid, g in per_game.items() if g["score"] == 0.0]
        print(f"games scoring 0.00: {len(zero)} -> {', '.join(sorted(zero))}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
