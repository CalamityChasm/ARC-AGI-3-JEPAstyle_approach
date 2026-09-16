"""Read a Duck kernel log's public-25 result without falling into the summary trap.

The harness emits a *progressive* summary roughly every few games' worth of
progress, each ending in a `mean score:` line. On the anim run there are 14 of
them and the first reads **0.39** against a true **9.97**. Reading the first, or
grepping without `tail`, silently reports a catastrophe that did not happen.

This takes the LAST summary and independently recomputes the mean from the 25
per-game `[finished]` lines, then asserts the two agree. It also totals actions
and recovers the wipe-guard's own `WIPE_GUARD_KEPT` / `WIPE_GUARD_FINAL` lines.

Usage:
    venv/Scripts/python.exe scripts/read_duck_public25_log.py <run-dir-or-log> ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

MEAN_RE = re.compile(r"mean score:\s*([0-9.]+)")
FINISHED_RE = re.compile(
    r"\[finished\] (?P<game>\S+) state=(?P<state>\S+) level=(?P<lvl>\d+)/(?P<lvls>\d+) "
    r"score=(?P<score>[0-9.]+) actions=(?P<actions>\d+) tokens=(?P<tokens>\d+) "
    r"per-level=(?P<per_level>[0-9/,]+)"
)
KEPT_RE = re.compile(r"WIPE_GUARD_KEPT n=(\d+)")
FINAL_RE = re.compile(r"WIPE_GUARD_FINAL kept=(\d+) wiped=(\d+) noop=(\d+) errors=(\d+)")
INSTALLED_RE = re.compile(r"WIPE_GUARD_INSTALLED [^\\\"]*")
ERROR_RE = re.compile(r"WIPE_GUARD_(?:UPSTREAM_)?ERROR [^\\\"]*")


def _log_path(target: Path) -> Path:
    if target.is_file():
        return target
    logs = sorted(target.glob("*.log"))
    assert logs, f"no .log in {target}"
    return logs[0]


def read(target: Path) -> dict[str, Any]:
    text = _log_path(target).read_text(encoding="utf-8", errors="replace")

    means = MEAN_RE.findall(text)
    games = [m.groupdict() for m in FINISHED_RE.finditer(text)]
    # A game can appear once per pass; public-25 is one pass, so de-dup on id.
    by_game = {g["game"]: g for g in games}

    recomputed = sum(float(g["score"]) for g in by_game.values()) / len(by_game) if by_game else 0.0
    reported = float(means[-1]) if means else 0.0

    actions = sum(int(g["actions"]) for g in by_game.values())
    levels = sum(int(g["lvl"]) for g in by_game.values())
    levels_total = sum(int(g["lvls"]) for g in by_game.values())

    # Actions on solved vs never-completed levels, from per-level=a/h,a/h,...
    solved_actions = unsolved_actions = 0
    for g in by_game.values():
        pairs = [p.split("/") for p in g["per_level"].split(",")]
        reached = int(g["lvl"])  # levels 1..reached are solved; reached+1 is in progress
        for i, (a, _h) in enumerate(pairs, start=1):
            if i <= reached:
                solved_actions += int(a)
            else:
                unsolved_actions += int(a)

    kept = [int(m) for m in KEPT_RE.findall(text)]
    final = FINAL_RE.findall(text)

    return {
        "log": str(_log_path(target)),
        "summaries_emitted": len(means),
        "first_summary_mean": float(means[0]) if means else None,
        "public25_mean": reported,
        "public25_mean_recomputed": round(recomputed, 4),
        "games_finished": len(by_game),
        "actions": actions,
        "levels_solved": levels,
        "levels_total": levels_total,
        "actions_solved_levels": solved_actions,
        "actions_never_completed_levels": unsolved_actions,
        "wasted_action_fraction": unsolved_actions / actions if actions else 0.0,
        "wipe_guard_kept_lines": len(kept),
        "wipe_guard_kept_max_n": max(kept) if kept else 0,
        "wipe_guard_final": final[-1] if final else None,
        "wipe_guard_installed": bool(INSTALLED_RE.search(text)),
        "wipe_guard_errors": len(ERROR_RE.findall(text)),
        "per_game": {k: {kk: v[kk] for kk in ("state", "lvl", "lvls", "score", "actions")} for k, v in sorted(by_game.items())},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", type=Path)
    args = ap.parse_args()
    for t in args.targets:
        r = read(t)
        assert r["games_finished"] == 25, f"{t}: only {r['games_finished']} games finished"
        assert abs(r["public25_mean"] - r["public25_mean_recomputed"]) < 0.02, (
            f"{t}: last summary {r['public25_mean']} disagrees with the per-game "
            f"recomputation {r['public25_mean_recomputed']}"
        )
        print(f"=== {t}")
        for key in (
            "public25_mean", "public25_mean_recomputed", "summaries_emitted", "first_summary_mean",
            "actions", "levels_solved", "levels_total", "actions_solved_levels",
            "actions_never_completed_levels", "wasted_action_fraction",
            "wipe_guard_installed", "wipe_guard_kept_lines", "wipe_guard_kept_max_n",
            "wipe_guard_final", "wipe_guard_errors",
        ):
            print(f"  {key:<32} {r[key]}")


if __name__ == "__main__":
    main()
