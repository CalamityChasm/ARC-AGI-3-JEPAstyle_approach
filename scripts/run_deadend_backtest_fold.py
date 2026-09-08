"""Runs one fold of a matched before/after agent-level backtest for the
tried_actions dead-end fix (experiments/stage6_transition_graph_health.md):
for --repeats independent passes, plays all 25 local games once each with
HYPOTHESIS_DEADEND_FILTER=0 (pre-fix baseline) and once each with =1 (the
fix), at the normal production MAX_ACTIONS=300 budget. Recordings for each
condition land in a separate directory so compare_agents.py-style analysis
never has to distinguish them by anything other than which folder they're in.

One game at a time, in a fresh subprocess (mirrors scripts/harvest_wins.py's
own pattern) so one slow/misbehaving game can't take the whole fold down,
with a disk-free-space check before each game.

Usage (from repo root, inside the venv):
    python scripts/run_deadend_backtest_fold.py --repeats 6 --fold-label fold1
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

# shutil.move (not Path.rename) -- the target dirs are on E:, source is on
# C:, and Path.rename/os.rename can't cross drives on Windows.

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"

ALL_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]

BEFORE_DIR = Path("E:/jepa_overflow/deadend_backtest/before")
AFTER_DIR = Path("E:/jepa_overflow/deadend_backtest/after")
MIN_FREE_GB = 5.0


def free_gb(drive: str) -> float:
    total, used, free = shutil.disk_usage(drive)
    return free / (1024 ** 3)


def check_disk() -> None:
    c_free = free_gb("C:\\")
    e_free = free_gb("E:\\")
    if c_free < MIN_FREE_GB or e_free < MIN_FREE_GB:
        print(f"[disk] ABORTING -- C: {c_free:.2f}GB, E: {e_free:.2f}GB", file=sys.stderr)
        sys.exit(1)


DEFAULT_RECORDINGS_DIR = AGENTS_DIR / "recordings"


def run_one(game: str, recordings_dir: Path, deadend_filter: str, timeout: int) -> float:
    import os
    # NOTE: main.py does `load_dotenv(dotenv_path=".env", override=True)`,
    # which silently clobbers a RECORDINGS_DIR passed via subprocess env
    # with whatever's in .env (confirmed the hard way -- fold 1's first
    # attempt dumped all 300 files into the default directory instead of
    # being split by condition). Don't rely on the env var for this; let
    # the run land in the default location, then move the one new file it
    # produced into the target directory ourselves.
    env = {**os.environ, "HYPOTHESIS_DEADEND_FILTER": deadend_filter}
    cmd = [str(VENV_PYTHON), "main.py", "--agent=hypothesis", f"--game={game}"]
    before = set(DEFAULT_RECORDINGS_DIR.glob(f"{game}-*.hypothesis.*.recording.jsonl"))
    t0 = time.time()
    try:
        subprocess.run(cmd, cwd=AGENTS_DIR, timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        pass
    elapsed = time.time() - t0
    after = set(DEFAULT_RECORDINGS_DIR.glob(f"{game}-*.hypothesis.*.recording.jsonl"))
    new_files = after - before
    recordings_dir.mkdir(parents=True, exist_ok=True)
    for f in new_files:
        shutil.move(str(f), str(recordings_dir / f.name))
    if len(new_files) != 1:
        print(f"  [warn] expected 1 new recording for {game}, found {len(new_files)}", flush=True)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--fold-label", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    BEFORE_DIR.mkdir(parents=True, exist_ok=True)
    AFTER_DIR.mkdir(parents=True, exist_ok=True)

    total_games = args.repeats * len(ALL_GAMES) * 2
    done = 0
    t_start = time.time()

    for r in range(args.repeats):
        for game in ALL_GAMES:
            check_disk()
            elapsed = run_one(game, BEFORE_DIR, "0", args.timeout)
            done += 1
            print(f"[{args.fold_label}] repeat {r+1}/{args.repeats} {game} BEFORE done={done}/{total_games} ({elapsed:.1f}s) elapsed_total={time.time()-t_start:.0f}s", flush=True)

            check_disk()
            elapsed = run_one(game, AFTER_DIR, "1", args.timeout)
            done += 1
            print(f"[{args.fold_label}] repeat {r+1}/{args.repeats} {game} AFTER  done={done}/{total_games} ({elapsed:.1f}s) elapsed_total={time.time()-t_start:.0f}s", flush=True)

    print(f"\n=== {args.fold_label} complete: {total_games} runs in {time.time()-t_start:.0f}s ===")


if __name__ == "__main__":
    main()
