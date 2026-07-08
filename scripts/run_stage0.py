"""Play every local ARC-AGI-3 game with a given agent and log trajectories
via the framework's built-in Recorder. Originally a Stage 0 harness script;
also used to run/evaluate the Stage 2 curiosity agent against the same
25-game baseline.

Usage (run from repo root, inside the venv):
    python scripts/run_stage0.py --agent random
    python scripts/run_stage0.py --agent pressonce --game ls20
    python scripts/run_stage0.py --agent curiosity
    python scripts/run_stage0.py --agent memory
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Stage 0 agent locally.")
    parser.add_argument(
        "--agent",
        required=True,
        choices=["random", "pressonce", "curiosity", "memory"],
        help="Agent to run.",
    )
    parser.add_argument(
        "--game", default=None, help="Game id prefix (default: all local games)."
    )
    args = parser.parse_args()

    cmd = [sys.executable, "main.py", f"--agent={args.agent}"]
    if args.game:
        cmd.append(f"--game={args.game}")

    subprocess.run(cmd, cwd=AGENTS_DIR, check=True)


if __name__ == "__main__":
    main()
