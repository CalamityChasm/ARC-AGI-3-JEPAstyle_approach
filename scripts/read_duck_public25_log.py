"""Read a Duck kernel log's public-25 result without falling into the summary trap.

The harness emits a *progressive* summary roughly every few games' worth of
progress, each ending in a `mean score:` line. On the anim run there are 14 of
them and the first reads **0.39** against a true **9.97**. Reading the first, or
grepping without `tail`, silently reports a catastrophe that did not happen.

This takes the LAST summary and independently recomputes the mean from the 25
per-game `[finished]` lines, then asserts the two agree. It also totals actions
and recovers the treatment arms' own marker lines (`WIPE_GUARD_*`, `RESTART_STALL_*`).

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
INSTALLED_RE = re.compile(r"WIPE_GUARD_INSTALLED [^\\\"\n]*")
ERROR_RE = re.compile(r"WIPE_GUARD_(?:UPSTREAM_)?ERROR [^\\\"\n]*")
RS_FIRED_RE = re.compile(
    r"RESTART_STALL_FIRED n=(?P<n>\d+) level=(?P<level>\S+) turns=(?P<turns>\d+) "
    r"nth_on_level=(?P<nth>\d+) seed=(?P<seed>\d+) kept_cross_level_notes=(?P<kept>\d+)c "
    r"session=(?P<session>\S+)"
)
RS_FINAL_RE = re.compile(
    r"RESTART_STALL_FINAL fired=(?P<fired>\d+) turns_discarded=(?P<turns>\d+) "
    r"capped=(?P<capped>\d+) no_step=(?P<no_step>\d+) errors=(?P<errors>\d+) "
    r"games=(?P<games>\d+) seed_now=(?P<seed>\d+)"
)
RS_INSTALLED_RE = re.compile(r"RESTART_STALL_INSTALLED [^\\\"\n]*")
RS_ERROR_RE = re.compile(r"RESTART_STALL_ERROR [^\\\"\n]*")
CF_INSTALLED_RE = re.compile(r"COMMIT_FLOOR_INSTALLED [^\\\"\n]*")
CF_FIRED_RE = re.compile(
    r"COMMIT_FLOOR_FIRED n=(?P<n>\d+) consec=(?P<consec>\d+) escalated=(?P<esc>\d+) "
    r"session=(?P<session>\S+)"
)
CF_RESULT_RE = re.compile(
    r"COMMIT_FLOOR_RESULT consec=(?P<consec>\d+) executed=(?P<executed>\d+) session=(?P<session>\S+)"
)
CF_FINAL_RE = re.compile(
    r"COMMIT_FLOOR_FINAL turns=(?P<turns>\d+) dead=(?P<dead>\d+) fired=(?P<fired>\d+) "
    r"escalated=(?P<escalated>\d+) converted=(?P<converted>\d+) unconverted=(?P<unconverted>\d+) "
    r"errors=(?P<errors>\d+)"
)
CF_ERROR_RE = re.compile(r"COMMIT_FLOOR_ERROR [^\\\"\n]*")
NT_INSTALLED_RE = re.compile(r"NO_THINKING_INSTALLED [^\\\"\n]*")


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

    # Everything the restart arm prints during its own synthetic probe happens
    # BEFORE the install banner. Slice there, so a build whose probe is not
    # silenced cannot inflate the firing count. (Kernel v1 of
    # arc3-duck-nvfp4-anim-rs was such a build: 3 phantom FIRED + 1 phantom
    # ERROR. The counters in RESTART_STALL_FINAL were always correct, because
    # the probe zeroes them; the LINES were not.)
    rs_banner = RS_INSTALLED_RE.search(text)
    rs_text = text[rs_banner.end():] if rs_banner else text
    rs_probe_fired = len(RS_FIRED_RE.findall(text)) - len(RS_FIRED_RE.findall(rs_text))
    rs_fired = [m.groupdict() for m in RS_FIRED_RE.finditer(rs_text)]
    rs_final = [m.groupdict() for m in RS_FINAL_RE.finditer(rs_text)]
    rs_sessions = sorted({f["session"] for f in rs_fired})

    # The commit floor's startup probe drives the real wrappers, so it emits
    # phantom FIRED/RESULT lines BEFORE the install banner (7 of each, by
    # construction -- see scripts/commit_floor_cell.py: _cf_probe). Its counters
    # are zeroed afterwards, so COMMIT_FLOOR_FINAL is always right; the LINES
    # are not. Slice at the banner, exactly as for the restart arm.
    cf_banner = CF_INSTALLED_RE.search(text)
    cf_text = text[cf_banner.end():] if cf_banner else text
    cf_probe_fired = len(CF_FIRED_RE.findall(text)) - len(CF_FIRED_RE.findall(cf_text))
    cf_fired = [m.groupdict() for m in CF_FIRED_RE.finditer(cf_text)]
    cf_results = [m.groupdict() for m in CF_RESULT_RE.finditer(cf_text)]
    cf_final = [m.groupdict() for m in CF_FINAL_RE.finditer(cf_text)]
    cf_converted = sum(1 for r in cf_results if r["executed"] == "1")

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
        "restart_stall_installed": bool(RS_INSTALLED_RE.search(text)),
        "restart_stall_fired_lines": len(rs_fired),
        "restart_stall_sessions": len(rs_sessions),
        "restart_stall_turns_discarded": sum(int(f["turns"]) for f in rs_fired),
        "restart_stall_levels": sorted({f["level"] for f in rs_fired}),
        "restart_stall_notes_kept_chars": sum(int(f["kept"]) for f in rs_fired),
        "restart_stall_final": rs_final[-1] if rs_final else None,
        "restart_stall_errors": len(RS_ERROR_RE.findall(rs_text)),
        "restart_stall_probe_lines_ignored": rs_probe_fired,
        "commit_floor_installed": bool(cf_banner),
        "commit_floor_fired_lines": len(cf_fired),
        "commit_floor_escalated_lines": sum(1 for f in cf_fired if f["esc"] == "1"),
        "commit_floor_result_lines": len(cf_results),
        "commit_floor_converted": cf_converted,
        "commit_floor_conversion_rate": (cf_converted / len(cf_results)) if cf_results else None,
        "commit_floor_sessions": len({f["session"] for f in cf_fired}),
        "commit_floor_final": cf_final[-1] if cf_final else None,
        "commit_floor_errors": len(CF_ERROR_RE.findall(cf_text)),
        "commit_floor_probe_lines_ignored": cf_probe_fired,
        "no_thinking_installed": bool(NT_INSTALLED_RE.search(text)),
        "no_thinking_banner": (NT_INSTALLED_RE.search(text).group(0) if NT_INSTALLED_RE.search(text) else None),
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
            "commit_floor_installed", "commit_floor_fired_lines", "commit_floor_escalated_lines",
            "commit_floor_result_lines", "commit_floor_converted", "commit_floor_conversion_rate",
            "commit_floor_sessions", "commit_floor_final", "commit_floor_errors",
            "commit_floor_probe_lines_ignored",
            "no_thinking_installed", "no_thinking_banner",
            "restart_stall_installed", "restart_stall_fired_lines",
            "restart_stall_sessions", "restart_stall_turns_discarded",
            "restart_stall_levels", "restart_stall_notes_kept_chars",
            "restart_stall_final", "restart_stall_errors",
            "restart_stall_probe_lines_ignored",
        ):
            print(f"  {key:<32} {r[key]}")


if __name__ == "__main__":
    main()
