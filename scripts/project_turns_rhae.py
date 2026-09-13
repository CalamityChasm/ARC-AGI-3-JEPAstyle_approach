"""Project the RHAE score of a Duck public-25 run under K x more analyzer turns.

Input: the run's own `benchmark.json`, which carries, per game, everything the
score depends on -- `number_of_levels`, `base_actions_per_level` (the human
baselines), `actions_per_level` (including the actions sunk into the level that
was still unfinished at the wall), `levels_completed` and `final_score`.

Usage:
    python scripts/project_turns_rhae.py <benchmark.json> [--json out.json]
                                         [--multipliers 1.0,1.5,2.0]

THE SCORE FORMULA IS READ FROM THE HARNESS SOURCE, NOT ASSUMED
--------------------------------------------------------------
CLAUDE.md states RHAE as

    E_e = min( sum_solved w_l / sum_all w_n ,  sum w_l*S_l / sum w_l )

and flags it "high confidence, not confirmed" (the Kaggle Evaluation page is a
JS SPA that could not be read at source). The ambiguity in that statement --
whether the efficiency term's denominator runs over SOLVED or over ALL levels --
**flips the sign of this entire investigation**, because it decides whether one
more slowly-won level raises the score or lowers it.

It is resolvable. The bundle that produced this run's `score.json` ships the
scorer: `src/tufa-arc-agi-framework/src/taaf/game.py`,
`GameRun._compute_final_score`, whose own docstring says it mirrors
`arc_agi.scorecard.EnvironmentScoreCalculator` (v0.9.8). Verbatim:

    for level_idx in range(self.number_of_levels):        # ALL levels
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < self.levels_completed
        actions  = self.actions_per_level[level_idx] ...
        baseline = self.base_actions_per_level[level_idx]
        if completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    score     = total_score / total_weights
    max_score = max_weights / total_weights * 100
    return min(score, max_score)

So the efficiency denominator is over **ALL** levels and is therefore FIXED.
Consequences that matter here, and that the CLAUDE.md form does not make
obvious:

* the efficiency term is **monotone non-decreasing** in levels solved -- an
  extra level adds `w_l * S_l >= 0` to a fixed denominator. More turns can
  never *lower* the score. (Under the solved-denominator reading it could, and
  dramatically: a first draft of this script, written to that reading, produced
  a confident -29% at 2x turns. That number was an artifact of a misread
  formula, and is recorded here so nobody re-derives it.)
* the completion cap counts only levels that actually SCORED
  (`level_score > 0`), not merely levels completed.
* `min(115, ...)` means a level solved faster than the human baseline is worth
  up to 1.15x a perfect one, which is why 15 of 25 games sit exactly on their
  completion cap.

`_compute_final_score` is reimplemented verbatim below and checked against every
game's recorded `final_score` before any projection is reported.
"""

from __future__ import annotations

import argparse
import json


def rhae_score(n_levels, base, actions, levels_completed):
    """Verbatim reimplementation of GameRun._compute_final_score."""
    if base is None or n_levels == 0:
        return 0.0
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_idx in range(n_levels):
        weight = level_idx + 1
        total_weights += weight
        completed = level_idx < levels_completed
        a = actions[level_idx] if level_idx < len(actions) else 0
        b = base[level_idx]
        level_score = min(115.0, (b / a) ** 2 * 100) if (completed and a > 0) else 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    if total_weights == 0:
        return 0.0
    return min(total_score / total_weights, max_weights / total_weights * 100)


def completion_cap(n_levels, levels_completed):
    """max_score: the weighted completion fraction, as a percentage."""
    tot = sum(range(1, n_levels + 1))
    got = sum(range(1, levels_completed + 1))
    return 100.0 * got / tot if tot else 0.0


#: The three bracketing models. All three cost a not-yet-won level as
#:
#:      a_l = max( ratio * base_l , actions_already_spent_on_it + 1 )
#:
#: Using the human baseline as the SHAPE of the cost curve is a real
#: improvement over "the next level costs what the last one cost": the
#: baselines vary 5-20x within a single game (`m0r0`: 30, 111, 203, 26, 500,
#: 237), so a flat extrapolation is badly wrong on most games.
#:
#: The `already + 1` term is the observed-resistance floor and is what makes
#: this honest at all: a level that survived N actions without completing
#: demonstrably costs more than N, whatever the ratio says. Without it `r11l`
#: -- 77 actions into a level whose baseline is 33, having won level 1 in 6 --
#: gets that level free even at mult=1.0.
#:
#: What the models differ on is `ratio`, and the difference is large enough
#: that reporting only one of them would be misleading:
#:
#:  optimistic  ratio = mean over WON levels of actions/baseline. Assumes the
#:              agent keeps the efficiency it showed on the levels it cleared.
#:              On a stuck game this is plainly too kind: it projects `r11l`
#:              (ratio 0.27, from one level won in 6 actions) straight through
#:              five further levels at ~14 actions each, while ignoring that
#:              level 2 has already absorbed 77 and not fallen.
#:  stuck-aware ratio = max(that, actions_on_the_unfinished_level / its
#:              baseline). Lets the game's CURRENT evidence of difficulty
#:              override its historical efficiency. This is the headline model.
#:  +1 only     stuck-aware costs, but at most one further level per game:
#:              finish what was in progress and stop. A deliberate floor for
#:              "more turns help", since it credits no new level the agent was
#:              not already working on.
MODELS = ("optimistic", "stuck-aware", "+1 only")


def project_game(g, mult, model="stuck-aware"):
    """Project one game's levels and score at `mult` x its observed actions.

    A game that won zero levels is projected to stay at zero under every model:
    there is no observed ratio to extrapolate from, and inventing one is the
    easiest way to manufacture upside that is not there.

    At mult=1.0 every model reproduces the observed level counts and scores
    EXACTLY. That is the correctness test.
    """
    n = g["number_of_levels"]
    base = g["base_actions_per_level"]
    acts = list(g["actions_per_level"])
    k = g["levels_completed"]
    total_actions = sum(acts)
    budget = total_actions * mult

    if k <= 0 or base is None:
        return {"levels": k, "actions": acts,
                "score": rhae_score(n, base, acts, k), "ratio": None}

    ratios = [acts[i] / base[i] for i in range(k) if base[i] > 0]
    ratio = sum(ratios) / len(ratios) if ratios else 1.0
    if model != "optimistic" and k < n and base[k] > 0:
        already = acts[k] if k < len(acts) else 0
        ratio = max(ratio, already / base[k])

    max_new = 1 if model == "+1 only" else n
    proj = list(acts) + [0] * max(0, n - len(acts))
    spent = float(total_actions)
    level = k
    while level < n and (level - k) < max_new:
        already = proj[level]
        need = max(ratio * base[level], already + 1.0)
        extra = need - already
        if spent + extra > budget:
            break
        spent += extra
        proj[level] = need
        level += 1
    return {"levels": level, "actions": proj,
            "score": rhae_score(n, base, proj, level), "ratio": ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark_json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--multipliers", default="1.0,1.25,1.5,2.0,2.55,3.0")
    args = ap.parse_args()

    bm = json.load(open(args.benchmark_json, encoding="utf-8"))
    games = bm["game_runs"]
    mults = [float(m) for m in args.multipliers.split(",")]

    print("=" * 100)
    print("RHAE PROJECTION: what does K x more analyzer turns buy?")
    print("formula reimplemented verbatim from taaf/game.py:GameRun._compute_final_score")
    print("=" * 100)

    worst = 0.0
    for g in games:
        got = rhae_score(g["number_of_levels"], g["base_actions_per_level"],
                         g["actions_per_level"], g["levels_completed"])
        worst = max(worst, abs(got - g["final_score"]))
    print("\n[check 1] reimplemented scorer vs recorded final_score: max abs error "
          "%.2e over %d games" % (worst, len(games)))
    if worst > 1e-9:
        print("          !! MISMATCH -- do not trust anything below.")

    base_mean = sum(g["final_score"] for g in games) / len(games)
    base_levels = sum(g["levels_completed"] for g in games)
    print("[check 2] mean public-25 score %.3f over %d games, %d levels"
          % (base_mean, len(games), base_levels))

    print("\n-- per game --")
    print("%-18s %3s %5s %8s %8s %-11s %s"
          % ("game", "N", "lvls", "score", "cap", "binds",
             "actions/baseline per level (last = unfinished)"))
    n_comp = 0
    for g in sorted(games, key=lambda x: -x["final_score"]):
        cap = completion_cap(g["number_of_levels"], g["levels_completed"])
        binds = "completion" if abs(g["final_score"] - cap) < 1e-9 else "efficiency"
        n_comp += binds == "completion"
        pairs = " ".join("%d/%d" % (a, b) for a, b in
                         zip(g["actions_per_level"][:g["levels_completed"] + 1],
                             g["base_actions_per_level"]))
        print("%-18s %3d %5d %8.3f %8.3f %-11s %s"
              % (g["game_id"], g["number_of_levels"], g["levels_completed"],
                 g["final_score"], cap, binds, pairs))
    print("\ncompletion binds in %d of %d games; efficiency in %d"
          % (n_comp, len(games), len(games) - n_comp))

    print("\n-- projection --")
    rows = []
    for m in mults:
        rec = {"mult": m}
        for model in MODELS:
            tot = lv = 0.0
            for g in games:
                p = project_game(g, m, model)
                tot += p["score"]
                lv += p["levels"]
            rec[model] = {"mean_score": tot / len(games), "levels": int(lv)}
        rows.append(rec)
    if abs(rows[0]["mult"] - 1.0) < 1e-9:
        ok = all(abs(rows[0][m]["mean_score"] - base_mean) < 1e-9
                 and rows[0][m]["levels"] == base_levels for m in MODELS)
        print("[check 3] mult=1.0 reproduces the observed run exactly under every "
              "model: %s (score %.6f vs %.6f, levels %d vs %d)"
              % ("YES" if ok else "NO", rows[0][MODELS[0]]["mean_score"], base_mean,
                 rows[0][MODELS[0]]["levels"], base_levels))

    hdr = "%8s" % "x turns"
    for model in MODELS:
        hdr += " %14s %8s %7s" % (model, "vs base", "levels")
    print("\n" + hdr)
    for r in rows:
        line = "%8.2f" % r["mult"]
        for model in MODELS:
            line += " %14.3f %7.1f%% %7d" % (
                r[model]["mean_score"],
                100.0 * (r[model]["mean_score"] / base_mean - 1.0),
                r[model]["levels"])
        print(line)

    print("\n  MONOTONICITY, and why it is not an assumption: the efficiency")
    print("  denominator is a sum over ALL levels and is therefore fixed, so an")
    print("  extra solved level adds w_l * S_l >= 0 to a constant denominator.")
    print("  More turns can raise the score or leave it flat. It cannot lower it.")

    probe = 2.0 if any(abs(m - 2.0) < 1e-9 for m in mults) else mults[-1]
    print("\n-- per-game change at x%.2f turns --" % probe)
    print("%-18s %8s %8s %9s %6s %6s %8s"
          % ("game", "base", "proj", "delta", "lvls", "->", "cap"))
    deltas = []
    for g in games:
        p = project_game(g, probe)
        deltas.append((p["score"] - g["final_score"], g, p))
    for d, g, p in sorted(deltas, key=lambda t: -t[0]):
        print("%-18s %8.2f %8.2f %+9.2f %6d %6d %8.2f"
              % (g["game_id"], g["final_score"], p["score"], d,
                 g["levels_completed"], p["levels"],
                 completion_cap(g["number_of_levels"], p["levels"])))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"base_mean": base_mean, "base_levels": base_levels,
                       "scorer_max_abs_error": worst, "projection": rows,
                       "per_game_at_probe": [
                           {"game": g["game_id"], "base": g["final_score"],
                            "proj": p["score"], "levels": g["levels_completed"],
                            "proj_levels": p["levels"]}
                           for _, g, p in deltas]}, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
