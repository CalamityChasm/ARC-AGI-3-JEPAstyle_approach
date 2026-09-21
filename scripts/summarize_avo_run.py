"""Counted telemetry for an AVO arm, against the anim incumbent.

Every number here is a **count** read out of a run's own artifacts, not a
sampled score. `stage7_noise_floor.md` put the SE of a single public-25 mean at
+/-2.46, which is wider than any effect this project has ever measured, so the
mean is reported for comparability only and **cannot rank two arms**. Counts
can: two runs of 25 games either take different numbers of turns or they do not.

Sources, in descending authority:

* `benchmark.json` -- the harness's own per-game `actions_per_level`,
  `base_actions_per_level`, `final_score`, `state`, `final_wallclock_seconds`.
* `artifacts/*_events.jsonl` -- the `experiment` event, which is where the
  solver merges `avo_*` counters (`solver.py:507-511`).
* `transcripts/*_p0.txt` -- one `[ANALYZER STATUS]` block per `analyze()` turn,
  reporting `step_executed`. This is where the dead-turn rate comes from, and
  the parsing is `scripts/analyze_dead_turns.py`'s, imported rather than
  re-implemented so the two cannot drift.

Usage:
    venv/Scripts/python.exe scripts/summarize_avo_run.py avo=<dir> anim=<dir> \\
        [--json experiments/stage7_avo_data.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_dead_turns import parse_transcript  # noqa: E402


def _transcripts(run: Path) -> list[Path]:
    return sorted((run / "transcripts").glob("*_p0.txt"))


def load(run: Path) -> dict[str, Any]:
    bm = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
    games = {g["game_id"]: g for g in bm["game_runs"]}

    out: dict[str, Any] = {"dir": str(run), "label": bm.get("label"), "games": {}}
    for name, g in sorted(games.items()):
        apl = list(g.get("actions_per_level") or [])
        base = list(g.get("base_actions_per_level") or [])
        cleared = int(g.get("levels_completed") or 0)
        out["games"][name] = {
            "score": float(g.get("final_score") or 0.0),
            "actions": sum(apl),
            "actions_per_level": apl,
            "base_actions_per_level": base,
            "levels_cleared": cleared,
            "levels_total": int(g.get("number_of_levels") or len(base) or len(apl)),
            "state": g.get("state"),
            "wallclock_s": float(g.get("final_wallclock_seconds") or 0.0),
        }

    # --- turns, from the transcripts -------------------------------------
    calls_by_game: dict[str, list[dict[str, Any]]] = {}
    for t in _transcripts(run):
        calls_by_game[t.name.split("_p0")[0]] = parse_transcript(t)
    out["turns"] = sum(len(c) for c in calls_by_game.values())
    out["executed"] = sum(1 for c in calls_by_game.values() for x in c if x["executed"])
    out["dead"] = out["turns"] - out["executed"]

    # P(act | k consecutive dead turns before it) -- the pre-registered falsifier.
    arm_hits = arm_acts = 0
    for calls in calls_by_game.values():
        streak = 0
        for c in calls:
            if streak >= 2:
                arm_hits += 1
                arm_acts += 1 if c["executed"] else 0
            streak = 0 if c["executed"] else streak + 1
    out["armed_turns"] = arm_hits
    out["armed_converted"] = arm_acts

    # --- model calls ------------------------------------------------------
    out["model_calls"] = sum(
        t.read_text(encoding="utf-8", errors="replace").count("[MODEL RESPONSE META]")
        for t in _transcripts(run)
    )

    # --- AVO counters, from the experiment event --------------------------
    avo: dict[str, int] = {}
    for ev_path in sorted((run / "artifacts").glob("*_events.jsonl")):
        for line in ev_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"experiment"' not in line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("type") != "experiment":
                continue
            for k, v in (row.get("counters") or {}).items():
                if k.startswith("avo_"):
                    avo[k] = avo.get(k, 0) + int(v or 0)
    out["avo_counters"] = avo
    out["avo_memory_files"] = len(list(run.rglob("avo_memory.json")))
    return out


def totals(run: dict[str, Any]) -> dict[str, Any]:
    gs = run["games"]
    # Actions spent inside a level that was cleared, vs one that never was.
    done = never = 0
    by_index: dict[int, dict[str, int]] = {}
    for g in gs.values():
        apl, cleared = g["actions_per_level"], g["levels_cleared"]
        for i, n in enumerate(apl, start=1):
            slot = by_index.setdefault(i, {"actions": 0, "cleared": 0})
            slot["actions"] += n
            if i <= cleared:
                done += n
                slot["cleared"] += 1
            else:
                never += n
    return {
        "mean_score": sum(g["score"] for g in gs.values()) / max(1, len(gs)),
        "actions": sum(g["actions"] for g in gs.values()),
        "levels": sum(g["levels_cleared"] for g in gs.values()),
        "levels_available": sum(g["levels_total"] for g in gs.values()),
        "games_scoring": sum(1 for g in gs.values() if g["score"] > 0),
        "games": len(gs),
        "actions_in_cleared_levels": done,
        "actions_in_never_cleared_levels": never,
        "by_index": by_index,
        "wallclock_s": sum(g["wallclock_s"] for g in gs.values()),
    }


def report(runs: dict[str, dict[str, Any]]) -> None:
    names = list(runs)
    tots = {n: totals(r) for n, r in runs.items()}

    def row(label: str, fmt, get) -> None:
        cells = "  ".join(f"{fmt(get(n)):>14}" for n in names)
        print(f"{label:<34}{cells}")

    print("\n" + " " * 34 + "  ".join(f"{n:>14}" for n in names))
    print("-" * (34 + 16 * len(names)))
    row("turns (analyze calls)", str, lambda n: runs[n]["turns"])
    row("  executed", str, lambda n: runs[n]["executed"])
    row("  dead", str, lambda n: runs[n]["dead"])
    row("  DEAD-TURN RATE",
        lambda v: f"{v:.1%}",
        lambda n: runs[n]["dead"] / max(1, runs[n]["turns"]))
    row("turns / game", lambda v: f"{v:.1f}",
        lambda n: runs[n]["turns"] / max(1, tots[n]["games"]))
    row("model calls", str, lambda n: runs[n]["model_calls"])
    row("calls / game", lambda v: f"{v:.1f}",
        lambda n: runs[n]["model_calls"] / max(1, tots[n]["games"]))
    print()
    row("actions", str, lambda n: tots[n]["actions"])
    row("actions / game", lambda v: f"{v:.1f}",
        lambda n: tots[n]["actions"] / max(1, tots[n]["games"]))
    row("  in levels that were cleared", str,
        lambda n: tots[n]["actions_in_cleared_levels"])
    row("  in levels never cleared", str,
        lambda n: tots[n]["actions_in_never_cleared_levels"])
    row("  share never cleared", lambda v: f"{v:.1%}",
        lambda n: tots[n]["actions_in_never_cleared_levels"] / max(1, tots[n]["actions"]))
    print()
    row("levels cleared", lambda v: f"{v[0]}/{v[1]}",
        lambda n: (tots[n]["levels"], tots[n]["levels_available"]))
    row("games scoring >= 1", lambda v: f"{v[0]} of {v[1]}",
        lambda n: (tots[n]["games_scoring"], tots[n]["games"]))
    row("public-25 mean  (CANNOT RANK)", lambda v: f"{v:.2f}",
        lambda n: tots[n]["mean_score"])
    print()
    row("P(act | >=2 dead before)", lambda v: f"{v[0]}/{v[1]}={v[2]:.1%}",
        lambda n: (runs[n]["armed_converted"], runs[n]["armed_turns"],
                   runs[n]["armed_converted"] / max(1, runs[n]["armed_turns"])))
    row("avo_memory.json files", str, lambda n: runs[n]["avo_memory_files"])
    for key in sorted({k for r in runs.values() for k in r["avo_counters"]}):
        row(f"  {key}", str, lambda n, k=key: runs[n]["avo_counters"].get(k, 0))

    print("\n-- levels cleared by index (w_l = l) --")
    idx = sorted({i for n in names for i in tots[n]["by_index"]})
    print(f"{'lvl':>4}  " + "  ".join(f"{n:>22}" for n in names))
    for i in idx:
        cells = []
        for n in names:
            s = tots[n]["by_index"].get(i, {"actions": 0, "cleared": 0})
            cells.append(f"{s['cleared']:>3} cleared {s['actions']:>6} act")
        print(f"{i:>4}  " + "  ".join(f"{c:>22}" for c in cells))

    print("\n-- per game --")
    allg = sorted({g for r in runs.values() for g in r["games"]})
    print(f"{'game':<12}" + "  ".join(f"{n:>26}" for n in names))
    for g in allg:
        cells = []
        for n in names:
            d = runs[n]["games"].get(g)
            cells.append(
                "-" if d is None
                else f"{d['score']:>6.2f} lvl {d['levels_cleared']}/{d['levels_total']} act {d['actions']:>4}"
            )
        print(f"{g:<12}" + "  ".join(f"{c:>26}" for c in cells))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=<unpacked kernel output dir>")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    runs: dict[str, dict[str, Any]] = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        runs[label] = load(Path(path))
    report(runs)
    if args.json:
        args.json.write_text(
            json.dumps({"runs": runs, "totals": {n: totals(r) for n, r in runs.items()}},
                       indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
