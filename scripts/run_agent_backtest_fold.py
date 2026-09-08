"""Generalized version of scripts/run_deadend_backtest_fold.py -- runs one
fold of a matched before/after agent-level backtest for ANY Hypothesis
env-var toggle (DEADEND_FILTER, LOOKAHEAD_MAX_DEPTH, FORCE_BETA, ...),
not just the dead-end fix specifically. Same methodology: for --repeats
independent passes, plays all 25 local games once with the env var set to
--before-value and once with --after-value, at the normal production
MAX_ACTIONS=300 budget, recordings split into separate directories.

One game at a time in a fresh subprocess, disk-checked, same pattern as
scripts/harvest_wins.py and scripts/run_deadend_backtest_fold.py.

Usage (from repo root, inside the venv):
    python scripts/run_agent_backtest_fold.py --repeats 6 --fold-label fold1 \
        --env-var HYPOTHESIS_LOOKAHEAD_MAX_DEPTH --before-value 1 --after-value 8 \
        --out-dir E:/jepa_overflow/lookahead_backtest
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
VENV_PYTHON = REPO_ROOT / "venv" / "Scripts" / "python.exe"
DEFAULT_RECORDINGS_DIR = AGENTS_DIR / "recordings"

ALL_GAMES = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
    "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
    "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30",
]

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


def run_one(game: str, recordings_dir: Path, env_var: str, value: str, timeout: int) -> float:
    # See run_deadend_backtest_fold.py's own comment: main.py's
    # load_dotenv(override=True) clobbers a RECORDINGS_DIR passed via
    # subprocess env, so land runs in the default dir and move the new
    # file ourselves rather than relying on the env var for routing.
    env = {**os.environ, env_var: value}
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
    parser.add_argument("--env-var", required=True)
    parser.add_argument("--before-value", required=True)
    parser.add_argument("--after-value", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    before_dir = args.out_dir / "before"
    after_dir = args.out_dir / "after"
    before_dir.mkdir(parents=True, exist_ok=True)
    after_dir.mkdir(parents=True, exist_ok=True)

    total_games = args.repeats * len(ALL_GAMES) * 2
    done = 0
    t_start = time.time()

    for r in range(args.repeats):
        for game in ALL_GAMES:
            check_disk()
            elapsed = run_one(game, before_dir, args.env_var, args.before_value, args.timeout)
            done += 1
            print(f"[{args.fold_label}] repeat {r+1}/{args.repeats} {game} BEFORE done={done}/{total_games} ({elapsed:.1f}s) elapsed_total={time.time()-t_start:.0f}s", flush=True)

            check_disk()
            elapsed = run_one(game, after_dir, args.env_var, args.after_value, args.timeout)
            done += 1
            print(f"[{args.fold_label}] repeat {r+1}/{args.repeats} {game} AFTER  done={done}/{total_games} ({elapsed:.1f}s) elapsed_total={time.time()-t_start:.0f}s", flush=True)

    print(f"\n=== {args.fold_label} complete: {total_games} runs in {time.time()-t_start:.0f}s ===")


if __name__ == "__main__":
    main()
