"""Stage 6: run the Solver agent (systematic, policy-free search) across
all local games, one game at a time, with a bounded per-game action budget
and wall-clock timeout -- see experiments/stage6_search_harvest.md for the
full writeup this feeds.

Each game runs as its own subprocess (SOLVER_MAX_ACTIONS env var controls
the budget, matching hypothesis_agent.py's HYPOTHESIS_MAX_ACTIONS pattern)
so a single misbehaving game can't take the whole sweep down -- a timeout
or crash on one game is logged and skipped, not fatal to the rest. Disk
free space is checked before every game (this project has hit a real
disk-full crash before from exactly this kind of recording-heavy work --
see CLAUDE.md's Gotchas) and the sweep stops early (rather than crashing
mid-write) if it drops below a safety floor.

Usage:
    python scripts/harvest_solver.py --max-actions 4000 --timeout 600
    python scripts/harvest_solver.py --games bp35 vc33 --max-actions 2000
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
ENV_FILES_DIR = AGENTS_DIR / "environment_files"
LOG_DIR = REPO_ROOT / "logs" / "harvest_solver"
MIN_FREE_GB = 5.0


def _free_gb(path: Path) -> float:
    free_bytes = ctypes.c_ulonglong(0)
    ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(str(path)), None, None, ctypes.pointer(free_bytes)
    )
    return free_bytes.value / (1024**3)


def _all_games() -> list:
    return sorted(p.name for p in ENV_FILES_DIR.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest a search-based corpus with the Solver agent.")
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=900, help="Per-game wall-clock cap, seconds.")
    parser.add_argument("--games", nargs="*", default=None, help="Subset of game ids (default: all local games).")
    args = parser.parse_args()

    games = args.games or _all_games()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SOLVER_MAX_ACTIONS"] = str(args.max_actions)

    results = []
    for i, game in enumerate(games, 1):
        free = _free_gb(Path(REPO_ROOT.anchor))
        if free < MIN_FREE_GB:
            print(f"[{i}/{len(games)}] STOPPING: only {free:.1f}GB free (floor {MIN_FREE_GB}GB) before {game}")
            break
        print(f"[{i}/{len(games)}] {game}  (free disk: {free:.1f}GB)")

        t0 = time.time()
        log_path = LOG_DIR / f"{game}.log"
        try:
            proc = subprocess.run(
                [sys.executable, "main.py", f"--agent=solver", f"--game={game}"],
                cwd=AGENTS_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            elapsed = time.time() - t0
            summary_line = next(
                (l for l in proc.stdout.splitlines() + proc.stderr.splitlines() if "solver agent: summary" in l),
                None,
            )
            results.append((game, elapsed, "ok", summary_line))
            print(f"    done in {elapsed:.1f}s -- {summary_line or '(no summary line found)'}")
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - t0
            out = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            log_path.write_text(out + "\n" + err, encoding="utf-8")
            results.append((game, elapsed, "TIMEOUT", None))
            print(f"    TIMED OUT after {elapsed:.1f}s -- moving on to the next game")
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.time() - t0
            results.append((game, elapsed, f"ERROR: {exc}", None))
            print(f"    ERROR: {exc}")

    print("\n=== harvest summary ===")
    for game, elapsed, status, summary in results:
        print(f"{game:<10} {elapsed:>7.1f}s  {status}")
        if summary:
            print(f"    {summary}")


if __name__ == "__main__":
    main()
