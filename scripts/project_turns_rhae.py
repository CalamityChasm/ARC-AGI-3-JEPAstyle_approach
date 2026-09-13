"""Project the RHAE score of the Duck NVFP4 public-25 run under K x more turns.

Inputs
------
* `experiments/stage7_analyzer_timeouts_data.json` - per-game turns / actions /
  levels_completed / level_up_actions, parsed from the run's transcripts and
  event logs by `scripts/analyze_analyzer_timeouts.py`.
* the run's own `score.json` - the harness's own RHAE score per game.

What it does
------------
1. Recovers each game's total level count `N` by inverting the completion term
   `sum_{l<=k} l / sum_{l<=N} l` against the reported score. Where that yields a
   near-integer `N`, **completion binds**; otherwise **efficiency binds** and
   the reported score is the efficiency term directly.
2. Models level acquisition against action count from each game's own observed
   `level_up_actions`, under two bracketing assumptions.
3. Reports the resulting mean public-25 score at several turn multipliers.

Every RHAE term follows CLAUDE.md's statement of the metric:
    S_l  = min(1.15, h_l / a_l) ** 2
    E_e  = min( sum_solved w_l / sum_all w_n ,  sum_solved w_l * S_l / sum_solved w_l )
    w_l  = l
    T    = mean over environments, as a percentage.
"""

from __future__ import annotations

import argparse
import json
import os


def tri(n: int) -> int:
    return n * (n + 1) // 2


def completion_term(k: int, n: int) -> float:
    """sum_{l<=k} l / sum_{l<=n} l."""
    if n <= 0:
        return 0.0
    return tri(k) / tri(n)


def infer_total_levels(k: int, score_pct: float, max_n: int = 40, tol: float = 1e-6):
    """Return (N, 'completion') if the score is exactly a completion term, else None."""
    if k <= 0 or score_pct <= 0:
        return None
    for n in range(max(k, 1), max_n + 1):
        if abs(completion_term(k, n) * 100.0 - score_pct) < tol * 100.0 + 1e-9:
            return n
    return None


def marginal_costs(level_up_actions: list[int]) -> list[int]:
    """Actions spent acquiring each successive level."""
    out = []
    prev = 0
    for a in level_up_actions:
        out.append(a - prev)
        prev = a
    return out


def project_levels_marginal(row: dict, mult: float, growth: float, n_total: int | None) -> int:
    """Model B: extrapolate each game's own marginal action-cost-per-level.

    `growth` is the geometric factor applied to each successive level's marginal
    action cost. growth=1.0 means every further level costs what the last
    observed one cost; growth>1 means levels get harder.

    Two hard floors keep this anchored to what was actually observed:

    * the level in progress when the wall hit is known **not** to have completed
      in the `in_progress` actions already spent on it, so its cost is floored at
      `in_progress + 1`. Without this the model hands out free levels at
      mult=1.0 (e.g. `r11l` spent 77 actions on level 2 after winning level 1 in
      6, so "the next level costs 6" is refuted by the run's own data);
    * a game that completed zero levels stays at zero - this run gives no
      observed rate to extrapolate from, and inventing one is the easiest way to
      manufacture an upside that is not there.

    With both floors, mult=1.0 reproduces the observed level counts exactly,
    which is the correctness test for the whole model.
    """
    k = row["levels_completed"]
    if k <= 0:
        return 0
    actions = row["actions"]
    budget = actions * mult
    ups = list(row["level_up_actions"])
    costs = marginal_costs(ups)
    spent = ups[-1]
    # The actions already sunk into the next, unfinished level count toward it.
    in_progress = actions - spent
    level = k
    # observed-resistance floor: this level survived `in_progress` actions
    next_cost = max(costs[-1] * growth, in_progress + 1.0)
    while True:
        if n_total is not None and level >= n_total:
            return level
        if spent + next_cost > budget:
            return level
        spent += next_cost
        level += 1
        next_cost = max(1.0, next_cost * growth)


def project_levels_linear(row: dict, mult: float, n_total: int | None) -> int:
    """Model A: levels scale linearly with actions, capped at N.

    Deliberately the crudest possible model, and an aggressive one - it assumes
    later levels cost no more than the average of the ones already won, which
    the marginal-cost table shows is usually false.
    """
    k = row["levels_completed"]
    if k <= 0:
        return 0
    out = int(k * mult)
    if n_total is not None:
        out = min(out, n_total)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeouts_json")
    ap.add_argument("score_json")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--multipliers", default="1.0,1.5,2.0,3.0",
        help="turn/action multipliers to project",
    )
    args = ap.parse_args()

    rows = json.load(open(args.timeouts_json, encoding="utf-8"))
    scores = json.load(open(args.score_json, encoding="utf-8"))["games"]

    games = []
    for r in rows:
        g = r["game"]
        sc = scores.get(g, {}).get("score")
        if sc is None:
            continue
        n = infer_total_levels(r["levels_completed"], sc)
        games.append(
            {
                **r,
                "score": sc,
                "n_total": n,
                "binds": "completion" if n is not None else "efficiency",
            }
        )

    base_mean = sum(x["score"] for x in games) / len(games)
    print("=" * 92)
    print("RHAE PROJECTION: what does K x more analyzer turns buy?")
    print("=" * 92)
    print("\nreported mean public-25 score: %.3f over %d games" % (base_mean, len(games)))

    n_comp = sum(1 for x in games if x["binds"] == "completion")
    print("games where the COMPLETION term binds : %d" % n_comp)
    print("games where the EFFICIENCY term binds : %d" % (len(games) - n_comp))
    print("games with zero levels completed      : %d"
          % sum(1 for x in games if x["levels_completed"] == 0))

    print("\n-- per game --")
    print("%-18s %6s %6s %6s %6s %7s %-11s %s"
          % ("game", "turns", "acts", "lvls", "N", "score", "binds", "marginal action cost/level"))
    for x in sorted(games, key=lambda y: -y["score"]):
        print(
            "%-18s %6d %6d %6d %6s %7.2f %-11s %s"
            % (
                x["game"], x["turns"], x["actions"], x["levels_completed"],
                x["n_total"] if x["n_total"] else "?", x["score"], x["binds"],
                marginal_costs(x["level_up_actions"]) or "-",
            )
        )

    # observed growth of marginal cost per level, for the realistic model
    ratios = []
    for x in games:
        cs = marginal_costs(x["level_up_actions"])
        for a, b in zip(cs, cs[1:]):
            if a > 0:
                ratios.append(b / a)
    ratios.sort()
    med_growth = ratios[len(ratios) // 2] if ratios else 1.0
    print("\nobserved marginal-cost growth per level: n=%d, median %.2fx, "
          "min %.2fx, max %.2fx" % (len(ratios), med_growth, ratios[0] if ratios else 0,
                                    ratios[-1] if ratios else 0))

    mults = [float(m) for m in args.multipliers.split(",")]
    models = [
        ("A linear", lambda x, m: project_levels_linear(x, m, x["n_total"])),
        ("B marg 1.00x", lambda x, m: project_levels_marginal(x, m, 1.0, x["n_total"])),
        ("B marg %.2fx" % med_growth,
         lambda x, m: project_levels_marginal(x, m, med_growth, x["n_total"])),
    ]
    print("\n-- projected mean score (UPPER BOUND: completion term only) --")
    print("   E_e = min(completion, efficiency) <= completion, so scoring every")
    print("   completion-bound game at its completion term needs no assumption")
    print("   about the unobservable h_l. Efficiency-bound games are held FLAT:")
    print("   S_l is monotone decreasing in a_l, so a slowly-won extra level can")
    print("   only drag their efficiency mean down, never raise it.")
    print()
    hdr = "%8s" % "x turns"
    for name, _ in models:
        hdr += " %16s %8s" % (name, "vs base")
    print(hdr)
    out_rows = []
    for m in mults:
        line = "%8.1f" % m
        rec = {"mult": m}
        for name, fn in models:
            tot = 0.0
            lv = 0
            for x in games:
                k2 = fn(x, m)
                lv += k2
                if x["n_total"] is None:
                    tot += x["score"]
                else:
                    tot += completion_term(k2, x["n_total"]) * 100.0
            mean_s = tot / len(games)
            rec[name] = {"mean_score": mean_s, "levels": lv}
            line += " %16.3f %7.1f%%" % (mean_s, 100.0 * (mean_s / base_mean - 1.0))
        out_rows.append(rec)
        print(line)

    base_levels = sum(x["levels_completed"] for x in games)
    print("\nlevels completed across the 25 games: base %d" % base_levels)
    for r in out_rows:
        print("  x%.1f -> %s" % (r["mult"], ", ".join(
            "%s %d" % (name, r[name]["levels"]) for name, _ in models)))
    print("\n(the x1.0 row MUST reproduce base mean %.3f / %d levels for models B;"
          " that is the model's correctness test)" % (base_mean, base_levels))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"base_mean": base_mean, "games": games,
                       "median_growth": med_growth, "projection": out_rows}, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
