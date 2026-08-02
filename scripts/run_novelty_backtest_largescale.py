"""Large-sample (n=30/condition) rerun of the held-out-games half of
experiments/stage6_novelty_aware_beta.md's agent-level backtest.

Identical protocol to scripts/run_novelty_backtest.sh's held-out section
(same checkpoint, same 5 held-out games, same MAX_ACTIONS=300 default,
same run_scorecard.py tooling) -- just a larger N to get real statistical
power on top of the original n=8 result. Trained-games sanity check is
not repeated here (that half of the original experiment already showed
the cap is structurally inert on in-vocab games by direct inspection of
game_vocab_moe.json, not something a larger sample changes).

Runs sequentially, in-process (not via a shell loop), and writes a
completion marker file so an external poller can detect when the whole
sweep is done without needing to inspect process state directly.

Usage: python scripts/run_novelty_backtest_largescale.py [--n 30]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
SCORECARDS_DIR = REPO_ROOT / "logs" / "scorecards"
LOG_PATH = REPO_ROOT / "logs" / "novelty_largescale_progress.log"
DONE_MARKER = REPO_ROOT / "logs" / "novelty_largescale_DONE"

HELDOUT_GAMES = "r11l,bp35,m0r0,tr87,ka59"
PY = sys.executable


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_one(cap: str, label: str) -> None:
    env = os.environ.copy()
    env["HYPOTHESIS_NOVELTY_BETA_CAP"] = cap
    cmd = [PY, "scripts/run_scorecard.py", "--agent", "hypothesis",
           "--label", label, "--game", HELDOUT_GAMES]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
    log(f"{label} (cap={cap}) done, rc={result.returncode}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DONE_MARKER.exists():
        DONE_MARKER.unlink()

    log(f"=== starting large-scale novelty backtest, n={args.n} per condition ===")

    for i in range(1, args.n + 1):
        run_one("0.15", f"ls_holdout_cap_on_r{i}")

    for i in range(1, args.n + 1):
        run_one("1.0", f"ls_holdout_cap_off_r{i}")

    log("=== ALL RUNS COMPLETE ===")
    DONE_MARKER.write_text("done\n")


if __name__ == "__main__":
    main()
