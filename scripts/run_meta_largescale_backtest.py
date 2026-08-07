"""Large-sample (n=25-30/condition) agent-level backtest comparing the
Reptile high-dose meta-learning checkpoint against the plain (normally
trained) baseline checkpoint, both under test-time adaptation, on the 5
held-out games -- see experiments/stage6_meta_learning.md's n=8
preliminary backtest for context (levels-completed came out exactly tied
with the already-published baseline+TTA numbers) and
experiments/stage6_novelty_aware_beta.md's Part 3 for why n=8 is not
trusted on this project's own established standard (a clean n=8 win
there completely evaporated at n=30).

Both conditions run HYPOTHESIS_TEST_TIME_ADAPT=1 -- the question here is
NOT "does TTA help" (already established), it's "does meta-training for
post-adaptation performance beat plain TTA on a normally-trained
checkpoint." Swaps the 4 files `hypothesis_agent.py`'s _init_models
reads (encoder_moe.pt, game_vocab_moe.json, moe_predictor.pt,
value_head.pt) into checkpoints/ for each condition, backing up and
restoring the real production files around the whole run so this never
leaves the production checkpoint directory in a modified state.

Usage: python scripts/run_meta_largescale_backtest.py [--n 30]
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
SCORECARDS_DIR = REPO_ROOT / "logs" / "scorecards"
LOG_PATH = REPO_ROOT / "logs" / "meta_largescale_progress.log"
DONE_MARKER = REPO_ROOT / "logs" / "meta_largescale_DONE"
BACKUP_DIR = REPO_ROOT / "logs" / "_prod_checkpoint_backup_meta_largescale"

HELDOUT_GAMES = "r11l,bp35,m0r0,tr87,ka59"
SWAP_FILES = ["encoder_moe.pt", "game_vocab_moe.json", "moe_predictor.pt", "value_head.pt"]
PY = sys.executable

CONDITIONS = {
    "metahd": REPO_ROOT / "checkpoints_meta_fold1_highdose",
    "baseline": REPO_ROOT / "checkpoints_holdout_baseline",
}


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def backup_production() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for name in SWAP_FILES:
        src = CHECKPOINTS_DIR / name
        if src.exists():
            shutil.copy2(src, BACKUP_DIR / name)
    log(f"backed up production checkpoint files to {BACKUP_DIR}")


def restore_production() -> None:
    for name in SWAP_FILES:
        src = BACKUP_DIR / name
        if src.exists():
            shutil.copy2(src, CHECKPOINTS_DIR / name)
    log("restored production checkpoint files")


def swap_in(condition_dir: Path) -> None:
    for name in SWAP_FILES:
        src = condition_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing {src} -- cannot swap in this condition")
        shutil.copy2(src, CHECKPOINTS_DIR / name)


def run_one(label: str) -> None:
    env = os.environ.copy()
    env["HYPOTHESIS_TEST_TIME_ADAPT"] = "1"
    cmd = [PY, "scripts/run_scorecard.py", "--agent", "hypothesis",
           "--label", label, "--game", HELDOUT_GAMES]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-6:])
    log(f"{label} done, rc={result.returncode}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DONE_MARKER.exists():
        DONE_MARKER.unlink()

    log(f"=== starting meta-largescale backtest, n={args.n} per condition ===")
    backup_production()

    try:
        for cond_name, cond_dir in CONDITIONS.items():
            log(f"--- swapping in condition '{cond_name}' from {cond_dir} ---")
            swap_in(cond_dir)
            for i in range(1, args.n + 1):
                run_one(f"meta_ls_{cond_name}_r{i}")
    finally:
        restore_production()

    log("=== ALL RUNS COMPLETE ===")
    DONE_MARKER.write_text("done\n")


if __name__ == "__main__":
    main()
