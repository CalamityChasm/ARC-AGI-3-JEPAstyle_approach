"""Where a Duck run loses DEPTH: deaths, the level-index action profile, and the
marginal RHAE value of one more level.

RHAE weights a level by its index (`w_l = l`), and both denominators run over
ALL levels, so score is monotone non-decreasing in levels solved. This script
answers, per game and in aggregate:

  1. how many in-level GAME_OVERs there are and how much progress each destroys
  2. how actions distribute across level indices against those indices' weights
  3. how close a stalled level got to its human baseline when the clock ran out
  4. what fraction of the wall clock is spent before a game's first completion
  5. which games are one level short of a large weighted gain

Sources, in order of authority:
  * `benchmark.json`  -- the harness's own accounting (`actions_per_level`,
    `base_actions_per_level`, `final_score`, per-action `history` with
    cumulative wallclock). This is what produces `score.json`.
  * `artifacts/*_events.jsonl` -- per-action `game_over` / `level` flags. Only
    `type == "action"` rows are counted: the stream mirrors each action with an
    `analysis` row, and counting both inflates every total (a prior figure read
    4,970 against 3,633 real actions).

Usage:
    venv/Scripts/python.exe scripts/analyze_depth.py <run-dir> [name=<run-dir> ...]
        [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_turns_rhae import rhae_score  # noqa: E402


# --------------------------------------------------------------------------
# loading


def _events_for(run_dir: Path, game_id: str) -> list[dict[str, Any]]:
    matches = sorted((run_dir / "artifacts").glob(f"{game_id}_p*_events.jsonl"))
    if not matches:
        stem = game_id.split("-")[0]
        matches = sorted((run_dir / "artifacts").glob(f"{stem}-*_events.jsonl"))
    assert matches, f"no events for {game_id} in {run_dir}"
    rows = []
    with matches[0].open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # The event stream mirrors every action with an `analysis` row. Count actions once.
    return [r for r in rows if r.get("type") == "action"]


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    bench = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    out = []
    for g in bench["game_runs"]:
        gid = g["game_id"]
        apl = list(g["actions_per_level"])
        base = list(g["base_actions_per_level"])
        hist = g["history"]
        actions = _events_for(run_dir, gid)
        assert sum(apl) == len(hist), f"{gid}: apl {sum(apl)} vs history {len(hist)}"
        # The event stream is allowed to differ by a per-game +-1 in how it
        # attributes actions to levels, but the totals must agree.
        assert sum(apl) == len(actions), f"{gid}: apl {sum(apl)} vs events {len(actions)}"
        out.append(
            {
                "game": gid.split("-")[0],
                "game_id": gid,
                "score": g["final_score"],
                "levels": g["levels_completed"],
                "n_levels": g["number_of_levels"],
                "apl": apl,
                "base": base,
                "wall": g["final_wallclock_seconds"],
                "state": g["state"],
                "history": hist,
                "actions": actions,
            }
        )
    return out


# --------------------------------------------------------------------------
# 1. deaths


def deaths(game: dict[str, Any]) -> list[dict[str, Any]]:
    """Every in-level GAME_OVER, with the length of the life it ended.

    A game over auto-RESETs into the *same* level (verified in this repo's
    event streams: the `level` field does not revert and the next action is a
    RESET). So a death destroys the whole life, not the whole level: every
    action taken since the level began or since the previous death.
    """
    out = []
    life_start = 0  # index into the action sequence
    for i, ev in enumerate(game["actions"]):
        if ev.get("level_completed"):
            life_start = i + 1
        if ev.get("game_over"):
            out.append(
                {
                    "i": i,
                    "level": ev.get("level"),
                    "life_actions": i - life_start + 1,
                    "step": ev.get("analysis_step"),
                    "t": game["history"][i]["wallclock_seconds"],
                }
            )
            life_start = i + 1
    return out


# --------------------------------------------------------------------------
# 2/3/4. level profile, stall progress, budget split


def level_of_action(game: dict[str, Any]) -> list[int]:
    """0-based level index for each action, from the harness's own accounting."""
    out = []
    for idx, n in enumerate(game["apl"]):
        out.extend([idx] * n)
    return out


def profile(game: dict[str, Any]) -> dict[str, Any]:
    apl, base = game["apl"], game["base"]
    lv = game["levels"]
    hist = game["history"]
    n_act = sum(apl)

    # The level in progress when the clock ran out.
    stall = lv  # 0-based index of the first uncompleted level
    stall_actions = apl[stall] if stall < len(apl) else 0
    stall_base = base[stall] if stall < len(base) else 0

    # Wall clock spent before the first level completed.
    if lv >= 1 and apl[0] > 0:
        t_first = hist[apl[0] - 1]["wallclock_seconds"]
    else:
        t_first = None

    # Wall clock at the last completed level.
    done_actions = sum(apl[:lv])
    t_last_clear = hist[done_actions - 1]["wallclock_seconds"] if done_actions else None

    return {
        "stall_level": stall + 1,  # 1-based, for reporting
        "stall_actions": stall_actions,
        "stall_base": stall_base,
        "stall_ratio": (stall_actions / stall_base) if stall_base else None,
        "actions_before_first_clear": apl[0] if lv >= 1 else n_act,
        "t_first_clear": t_first,
        "t_last_clear": t_last_clear,
        "wall": game["wall"],
        "frac_wall_before_first_clear": (t_first / game["wall"]) if t_first else 1.0,
        "frac_wall_after_last_clear": (
            1.0 - t_last_clear / game["wall"] if t_last_clear else 1.0
        ),
        "sunk_actions": sum(apl[lv:]),
    }


# --------------------------------------------------------------------------
# 5. marginal value of one more level


def marginal(game: dict[str, Any], assume: str = "as_played") -> float:
    """Score if this game had cleared exactly one more level.

    `as_played`: the next level is cleared at the action count already spent on
    it (so its efficiency term is what the run actually earned). Where zero
    actions were spent there, the human baseline is assumed.
    `human`: the next level is cleared at exactly the human baseline (S = 1.0).
    """
    lv, apl, base = game["levels"], list(game["apl"]), game["base"]
    if lv >= game["n_levels"]:
        return game["score"]
    if assume == "human":
        apl[lv] = base[lv]
    elif apl[lv] == 0:
        apl[lv] = base[lv]
    return rhae_score(game["n_levels"], base, apl, lv + 1)


# --------------------------------------------------------------------------


def analyse(run_dir: Path) -> dict[str, Any]:
    games = load_run(run_dir)
    rows = []
    level_actions = Counter()
    level_clears = Counter()
    for g in games:
        d = deaths(g)
        p = profile(g)
        row = {
            "game": g["game"],
            "score": g["score"],
            "levels": g["levels"],
            "n_levels": g["n_levels"],
            "actions": sum(g["apl"]),
            "apl": g["apl"],
            "base": g["base"],
            "deaths": len(d),
            "death_detail": d,
            "actions_lost_to_deaths": sum(x["life_actions"] for x in d),
            "marginal_as_played": marginal(g, "as_played"),
            "marginal_human": marginal(g, "human"),
            **p,
        }
        row["d_score_plus1"] = row["marginal_as_played"] - g["score"]
        row["d_score_plus1_human"] = row["marginal_human"] - g["score"]
        rows.append(row)
        for i, n in enumerate(g["apl"]):
            level_actions[i + 1] += n
        for i in range(g["levels"]):
            level_clears[i + 1] += 1

    mean = statistics.fmean(r["score"] for r in rows)
    total_levels = sum(r["levels"] for r in rows)
    all_levels = sum(r["n_levels"] for r in rows)
    return {
        "run": str(run_dir),
        "mean_score": mean,
        "actions": sum(r["actions"] for r in rows),
        "levels": total_levels,
        "levels_available": all_levels,
        "deaths": sum(r["deaths"] for r in rows),
        "actions_lost_to_deaths": sum(r["actions_lost_to_deaths"] for r in rows),
        "games_with_deaths": sum(1 for r in rows if r["deaths"]),
        "level_actions": dict(sorted(level_actions.items())),
        "level_clears": dict(sorted(level_clears.items())),
        "games": rows,
    }


def report(res: dict[str, Any]) -> None:
    print(f"\n=== {res['run']}")
    print(
        f"mean {res['mean_score']:.4f}  actions {res['actions']}  "
        f"levels {res['levels']}/{res['levels_available']}  "
        f"deaths {res['deaths']} in {res['games_with_deaths']} games  "
        f"actions inside lives that ended in death: {res['actions_lost_to_deaths']}"
    )

    print("\n-- actions by level index (w_l = l) --")
    la = res["level_actions"]
    tot = sum(la.values())
    print(f"{'lvl':>4} {'actions':>8} {'share':>7} {'cleared':>8} {'weight share':>13}")
    wtot = sum(i * v for i, v in ((k, 1) for k in la))  # placeholder, recomputed below
    weights = {k: k for k in la}
    wsum = sum(weights.values())
    for k in sorted(la):
        print(
            f"{k:>4} {la[k]:>8} {la[k]/tot:>6.1%} "
            f"{res['level_clears'].get(k,0):>8} {weights[k]/wsum:>12.1%}"
        )

    print("\n-- per game --")
    hdr = (
        f"{'game':<6}{'score':>7}{'lvl':>6}{'act':>6}{'deaths':>7}{'lost':>6}"
        f"{'stall':>6}{'ours/hum':>10}{'ratio':>7}{'t1st%':>7}{'+1 lvl':>8}{'+1 hum':>8}"
    )
    print(hdr)
    for r in sorted(res["games"], key=lambda r: -r["d_score_plus1_human"]):
        sr = r["stall_ratio"]
        print(
            f"{r['game']:<6}{r['score']:>7.2f}{str(r['levels'])+'/'+str(r['n_levels']):>6}"
            f"{r['actions']:>6}{r['deaths']:>7}{r['actions_lost_to_deaths']:>6}"
            f"{r['stall_level']:>6}"
            f"{str(r['stall_actions'])+'/'+str(r['stall_base']):>10}"
            f"{(f'{sr:.2f}' if sr else '-'):>7}"
            f"{r['frac_wall_before_first_clear']:>6.0%}"
            f"{r['d_score_plus1']:>8.2f}{r['d_score_plus1_human']:>8.2f}"
        )
    dp = sum(r["d_score_plus1"] for r in res["games"]) / len(res["games"])
    dh = sum(r["d_score_plus1_human"] for r in res["games"]) / len(res["games"])
    print(f"\nmean gain if EVERY game cleared one more level: +{dp:.2f} (as played) / +{dh:.2f} (at human baseline)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="[name=]<run-dir>")
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
        slim = {}
        for k, v in out.items():
            v = dict(v)
            v["games"] = [
                {kk: vv for kk, vv in g.items() if kk != "death_detail"} | {
                    "death_detail": g["death_detail"]
                }
                for g in v["games"]
            ]
            slim[k] = v
        args.json.write_text(json.dumps(slim, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
