"""Harvest real ARC-3 gameplay transitions from episodes that reach a WIN,
using the current best agent (Hypothesis, with MAX_ACTIONS temporarily
bumped -- see hypothesis_agent.py's own comment on that bump) as the
harvesting policy.

Runs one game at a time as a fresh subprocess (mirrors
`stage6-search-harvest`'s `scripts/harvest_solver.py` pattern) so one
misbehaving/slow game can't take the whole campaign down, with a
per-game wall-clock timeout and a disk-free-space check on both C: and
E: before each game -- this project has repeatedly hit full-disk crises
from exactly this kind of bulk recording generation (see CLAUDE.md's
Gotchas section).

Usage (from repo root, inside the venv):
    python scripts/harvest_wins.py --games r11l sp80 cn04 --timeout 1200
    python scripts/harvest_wins.py --all --timeout 1200   # careful: long
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
# Shared venv lives at the main checkout's fixed path, not per-worktree
# (worktrees under .claude/worktrees/ don't get their own venv/).
VENV_PYTHON = Path(
    r"C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\venv\Scripts\python.exe"
)

ALL_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]

MIN_FREE_GB = 5.0  # abort before starting a game if either drive drops below this


def free_gb(drive: str) -> float:
    total, used, free = shutil.disk_usage(drive)
    return free / (1024 ** 3)


def check_disk() -> None:
    c_free = free_gb("C:\\")
    e_free = free_gb("E:\\")
    print(f"[disk] C: {c_free:.2f}GB free, E: {e_free:.2f}GB free", flush=True)
    if c_free < MIN_FREE_GB or e_free < MIN_FREE_GB:
        print(
            f"[disk] ABORTING -- free space below {MIN_FREE_GB}GB threshold "
            f"(C: {c_free:.2f}GB, E: {e_free:.2f}GB)",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest WIN-episode recordings with Hypothesis.")
    parser.add_argument("--games", nargs="*", default=None, help="Game id prefixes to run.")
    parser.add_argument("--all", action="store_true", help="Run all 25 local games.")
    parser.add_argument("--timeout", type=int, default=1200, help="Per-game wall-clock timeout (seconds).")
    args = parser.parse_args()

    if args.all:
        games = ALL_GAMES
    elif args.games:
        games = args.games
    else:
        parser.error("Pass --games <id...> or --all")

    for game in games:
        check_disk()
        print(f"\n=== [{time.strftime('%H:%M:%S')}] harvesting {game} ===", flush=True)
        cmd = [str(VENV_PYTHON), "main.py", "--agent=hypothesis", f"--game={game}"]
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=AGENTS_DIR, timeout=args.timeout,
                capture_output=True, text=True,
            )
            elapsed = time.time() - t0
            tail = "\n".join(result.stdout.splitlines()[-5:]) if result.stdout else ""
            print(f"[{game}] exit={result.returncode} elapsed={elapsed:.1f}s\n{tail}", flush=True)
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"[{game}] TIMED OUT after {elapsed:.1f}s (limit {args.timeout}s)", flush=True)

    check_disk()
    print("\n=== harvest leg complete ===", flush=True)


if __name__ == "__main__":
    main()
