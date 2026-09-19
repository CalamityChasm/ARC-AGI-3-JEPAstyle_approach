"""Which RHAE term actually binds for us: completion, or efficiency?

The competition metric is

    per level        S_l = min(1.15, h_l / a_l) ** 2
    per environment  E_e = min( sum_solved w_l / sum_all w_n ,
                                sum_solved w_l * S_l / sum_all w_l ),   w_l = l
    final            T   = mean over environments, as a percentage

`h_l` is the upper-median best-human action count for level l. It ships inside
every run's own `benchmark.json` as `base_actions_per_level`, and the totals
match the human baseline quoted in arXiv:2607.15439 (17,135 actions over the 183
levels of the 25 public games) -- so this is the official baseline, not a proxy.

`a_l` (our actions on level l) is recoverable from `artifacts/*_events.jsonl`,
which stamps every action with the `level` it was taken on.

The question this answers is strategic, not cosmetic. If the efficiency term
binds, the right move is to spend fewer actions per level. If the completion
term binds, efficiency is *free* and the only thing that raises score is
finishing more levels -- which costs more actions, not fewer.

Usage:
    venv/Scripts/python.exe scripts/analyze_rhae_binding.py <kernel-output-dir> \
        --json experiments/stage7_rhae_binding.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def level_actions(events_path: Path) -> dict[int, int]:
    """Count actions taken on each level from an events jsonl."""
    counts: dict[int, int] = {}
    with events_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("action_num") is None:
                continue
            lvl = rec.get("level")
            if lvl is None:
                continue
            counts[int(lvl)] = counts.get(int(lvl), 0) + 1
    return counts


def analyse(run_dir: Path) -> dict[str, Any]:
    with (run_dir / "benchmark.json").open(encoding="utf-8") as fh:
        runs = json.load(fh)["game_runs"]
    with (run_dir / "score.json").open(encoding="utf-8") as fh:
        scores = json.load(fh)["games"]

    out = []
    for run in sorted(runs, key=lambda r: r["game_id"]):
        gid = run["game_id"]
        human = run.get("base_actions_per_level") or []
        n_levels = run.get("number_of_levels") or len(human)
        ev = run_dir / "artifacts" / f"{gid}_p0_events.jsonl"
        per_level = level_actions(ev) if ev.exists() else {}

        reported = scores.get(gid, {}).get("score", 0.0) / 100.0
        weights = list(range(1, n_levels + 1))
        w_all = sum(weights)

        # A level is solved if the run ever moved past it. Levels are 1-indexed
        # in the event stream; the deepest level reached is the one in progress.
        reached = max(per_level) if per_level else 1
        solved = [lvl for lvl in range(1, reached)]  # everything before the
        # level we were still on when time ran out

        term1 = sum(weights[lvl - 1] for lvl in solved) / w_all if w_all else 0.0
        eff_sum = 0.0
        s_values = []
        for lvl in solved:
            h = human[lvl - 1] if lvl - 1 < len(human) else 0
            a = per_level.get(lvl, 0)
            s = min(1.15, (h / a) if a else 0.0) ** 2
            s_values.append(s)
            eff_sum += weights[lvl - 1] * s
        term2 = eff_sum / w_all if w_all else 0.0

        out.append(
            {
                "game_id": gid,
                "n_levels": n_levels,
                "levels_solved": len(solved),
                "deepest_level": reached,
                "human_actions_total": sum(human),
                "our_actions_total": sum(per_level.values()),
                "actions_on_unsolved_level": per_level.get(reached, 0),
                "term1_completion": term1,
                "term2_efficiency": term2,
                "binding": "completion" if term1 <= term2 else "efficiency",
                "model_E": min(term1, term2),
                "reported_E": reported,
                "S_l_min": min(s_values) if s_values else None,
                "S_l_capped_count": sum(1 for s in s_values if s >= 1.15**2 - 1e-9),
                "per_level_actions": per_level,
                "human_per_level": human,
            }
        )
    return {"games": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    res = analyse(args.run_dir)
    games = res["games"]

    hdr = (
        f"{'game':16}{'lv':>3}{'solv':>5}{'hum':>7}{'ours':>6}{'waste':>7}"
        f"{'term1':>8}{'term2':>8}{'binds':>12}{'E':>7}{'rep':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for g in games:
        print(
            f"{g['game_id'][:15]:16}{g['n_levels']:3d}{g['levels_solved']:5d}"
            f"{g['human_actions_total']:7d}{g['our_actions_total']:6d}"
            f"{g['actions_on_unsolved_level']:7d}"
            f"{g['term1_completion']:8.3f}{g['term2_efficiency']:8.3f}"
            f"{g['binding']:>12}{g['model_E'] * 100:7.2f}{g['reported_E'] * 100:7.2f}"
        )

    waste = sum(g["actions_on_unsolved_level"] for g in games)
    total = sum(g["our_actions_total"] for g in games)
    solved = sum(g["levels_solved"] for g in games)
    all_lv = sum(g["n_levels"] for g in games)

    # `term1` is exactly computable and `reported_E` is ground truth, so the
    # honest binding test compares those two. `term2` as reconstructed here is
    # systematically pessimistic (level attribution over-counts animation
    # frames), so it is not used to decide the binding.
    eps = 5e-4
    completion_bound = [g for g in games if g["reported_E"] >= g["term1_completion"] - eps]
    efficiency_bound = [g for g in games if g["reported_E"] < g["term1_completion"] - eps]

    print()
    print(f"levels solved: {solved} / {all_lv} ({100 * solved / all_lv:.1f}%)")
    print(
        f"actions on the never-completed level: {waste} / {total} "
        f"({100 * waste / max(total, 1):.1f}%)"
    )
    print()
    print("binding term, judged by reported_E vs the exactly-computable term1:")
    print(
        f"  COMPLETION binds (reported == term1): {len(completion_bound)} / {len(games)}"
        f"  -> {', '.join(g['game_id'][:4] for g in completion_bound)}"
    )
    print(
        f"  EFFICIENCY binds (reported <  term1): {len(efficiency_bound)} / {len(games)}"
        f"  -> {', '.join(g['game_id'][:4] for g in efficiency_bound)}"
    )
    lost = sum(g["term1_completion"] - g["reported_E"] for g in efficiency_bound)
    print(
        f"  points lost to the efficiency cap: {100 * lost / len(games):.2f} "
        f"of a {100 * sum(g['term1_completion'] for g in games) / len(games):.2f} "
        f"completion-only ceiling (reported mean "
        f"{100 * sum(g['reported_E'] for g in games) / len(games):.2f})"
    )

    # How do our actions compare to the human baseline on the levels we engage?
    eng_ours = eng_hum = 0
    for g in games:
        for lvl, a in g["per_level_actions"].items():
            eng_ours += a
            hp = g["human_per_level"]
            if lvl - 1 < len(hp):
                eng_hum += hp[lvl - 1]
    print()
    print(
        f"on the levels we actually engaged: ours {eng_ours} vs human {eng_hum} "
        f"= {eng_ours / max(eng_hum, 1):.2f}x human"
    )
    print(
        f"across ALL levels incl. the {all_lv - solved} never reached: "
        f"ours {total} vs human {sum(g['human_actions_total'] for g in games)} "
        f"= {total / max(sum(g['human_actions_total'] for g in games), 1):.2f}x"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
