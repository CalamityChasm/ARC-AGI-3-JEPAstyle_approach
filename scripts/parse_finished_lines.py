"""Recompute a run's headline numbers from its per-game ``[finished]`` log lines.

Why this exists
---------------
These kernel logs emit a *progressive* summary block roughly every 10 games, so
the obvious thing to grep -- the summary -- is a partial-run snapshot unless you
happen to take the last one. The per-game ``[finished]`` lines are not
progressive: there is exactly one per game, written when that game ends, and it
carries everything needed to rebuild the run::

    [finished] m0r0-492f87ba state=gave_up level=1/6 score=4.76 actions=88
               tokens=78191 per-level=29/30,59/111,0/203,0/26,0/500,0/237

``level=L/N``      L levels completed of N
``per-level=a/h``  our actions / the official human baseline, per level, in order
``score``          that game's RHAE percentage

So this is an independent recomputation of ``score.json`` / ``benchmark.json``
from a different artifact, and it is also the *cleanest* source for the
wasted-action split: the never-completed level is exactly the (L+1)-th entry,
with no need to attribute animation frames the way the events stream does.

Usage
-----
    venv/Scripts/python.exe scripts/parse_finished_lines.py <kernel-output-dir> [...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FINISHED_RE = re.compile(
    r"\[finished\]\s+(?P<gid>\S+)\s+state=(?P<state>\S+)\s+"
    r"level=(?P<done>\d+)/(?P<total>\d+)\s+score=(?P<score>[\d.]+)\s+"
    r"actions=(?P<actions>\d+)\s+tokens=(?P<tokens>\d+)\s+"
    r"per-level=(?P<per_level>[\d/,]+)"
)


def log_text(run_dir: Path) -> str:
    """The kernel log is a JSON array of {stream_name, time, data} records."""
    candidates = sorted(run_dir.glob("*.log"))
    for path in candidates:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            records = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return "".join(str(r.get("data", "")) for r in records)
    raise SystemExit(f"no *.log in {run_dir}")


def parse(run_dir: Path) -> dict[str, Any]:
    text = log_text(run_dir)

    games = []
    for m in FINISHED_RE.finditer(text):
        pairs = [p.split("/") for p in m.group("per_level").split(",")]
        ours = [int(a) for a, _ in pairs]
        human = [int(h) for _, h in pairs]
        done = int(m.group("done"))
        games.append(
            {
                "game_id": m.group("gid"),
                "state": m.group("state"),
                "levels_done": done,
                "levels_total": int(m.group("total")),
                "score": float(m.group("score")),
                "actions": int(m.group("actions")),
                "tokens": int(m.group("tokens")),
                "actions_per_level": ours,
                "human_per_level": human,
                "actions_on_solved": sum(ours[:done]),
                # The level we were still on when the clock ran out.
                "actions_on_unsolved": ours[done] if done < len(ours) else 0,
            }
        )

    n = len(games)
    actions = sum(g["actions"] for g in games)
    solved_actions = sum(g["actions_on_solved"] for g in games)
    unsolved_actions = sum(g["actions_on_unsolved"] for g in games)
    return {
        "dir": str(run_dir),
        "games": n,
        "public25_mean": sum(g["score"] for g in games) / n if n else 0.0,
        "actions": actions,
        "levels_done": sum(g["levels_done"] for g in games),
        "levels_total": sum(g["levels_total"] for g in games),
        "levels_per_game": sum(g["levels_done"] for g in games) / n if n else 0.0,
        "actions_on_solved": solved_actions,
        "actions_on_unsolved": unsolved_actions,
        "wasted_frac": unsolved_actions / actions if actions else 0.0,
        "human_actions_engaged": sum(
            sum(g["human_per_level"][: g["levels_done"] + 1]) for g in games
        ),
        "human_actions_all": sum(sum(g["human_per_level"]) for g in games),
        "states": {
            s: sum(1 for g in games if g["state"] == s) for s in {g["state"] for g in games}
        },
        "per_game": games,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--per-game", action="store_true")
    args = ap.parse_args()

    results = [parse(d) for d in args.run_dirs]
    for d, r in zip(args.run_dirs, results):
        print(f"=== {d.name} ===")
        print(f"  games                 {r['games']}")
        print(f"  public-25 mean        {r['public25_mean']:.2f}")
        print(f"  total actions         {r['actions']}")
        print(f"  levels completed      {r['levels_done']} / {r['levels_total']} "
              f"({r['levels_per_game']:.2f} per game)")
        print(f"  actions on solved     {r['actions_on_solved']}")
        print(f"  actions on unsolved   {r['actions_on_unsolved']} "
              f"({r['wasted_frac']:.1%} of all actions)")
        print(f"  human baseline (all)  {r['human_actions_all']}")
        print(f"  states                {r['states']}")
        if args.per_game:
            for g in sorted(r["per_game"], key=lambda x: x["game_id"]):
                print(
                    f"    {g['game_id'][:16]:18}{g['levels_done']:2d}/{g['levels_total']:<2d}"
                    f"  score={g['score']:6.2f}  act={g['actions']:4d}"
                    f"  solved={g['actions_on_solved']:4d}"
                    f"  stuck={g['actions_on_unsolved']:4d}"
                )

    if len(results) == 2:
        a, b = results
        print()
        print(f"{'metric':24}{'baseline':>12}{'variant':>12}{'delta':>12}")
        print("-" * 60)
        for label, key, fmt in [
            ("public-25 mean", "public25_mean", "{:.2f}"),
            ("total actions", "actions", "{:.0f}"),
            ("levels completed", "levels_done", "{:.0f}"),
            ("actions on solved", "actions_on_solved", "{:.0f}"),
            ("actions on unsolved", "actions_on_unsolved", "{:.0f}"),
            ("wasted fraction", "wasted_frac", "{:.1%}"),
        ]:
            delta = (
                f"{100 * (b[key] / a[key] - 1):+.1f}%" if a[key] else "-"
            )
            print(f"{label:24}{fmt.format(a[key]):>12}{fmt.format(b[key]):>12}{delta:>12}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
