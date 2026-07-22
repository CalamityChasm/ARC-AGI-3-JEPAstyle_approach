"""Run an agent across all local games and capture the framework's own
FINAL SCORECARD REPORT -- which, in offline mode, already computes the
real Kaggle scoring formula (rules.md: per-level
min(baseline_actions / agent_actions, 1.0) ** 2, weighted by level index,
averaged per game then across games) using real human baseline action
counts from `level_baseline_actions`. This is ground truth, not an
approximation -- prefer it over any hand-built efficiency proxy.

Saves the extracted scorecard JSON to logs/scorecards/<label>.json
(gitignored -- these are local experiment artifacts, not committed code)
and prints the top-level score/completion summary.

Usage: python scripts/run_scorecard.py --agent hypothesis --label baseline_300
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "ARC-AGI-3-Agents"
SCORECARDS_DIR = REPO_ROOT / "logs" / "scorecards"


def extract_scorecard(log_text: str) -> dict:
    marker = "--- FINAL SCORECARD REPORT ---"
    idx = log_text.rfind(marker)
    if idx == -1:
        raise ValueError("No 'FINAL SCORECARD REPORT' found in output")
    tail = log_text[idx + len(marker):]
    # First '{' starts the JSON block; find its matching closing brace by
    # depth-counting rather than assuming a fixed line count.
    start = tail.index("{")
    depth = 0
    for i, ch in enumerate(tail[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        raise ValueError("Unbalanced braces in scorecard JSON block")
    json_str = tail[start:end]
    # Strip leading log-line prefixes ("2026-... | INFO | ") from continuation lines.
    json_str = re.sub(r"\n\S+ \| INFO \| ", "\n", json_str)
    return json.loads(json_str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an agent and capture its real scorecard.")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--label", required=True, help="Name for the saved scorecard file.")
    parser.add_argument("--game", default=None)
    args = parser.parse_args()

    cmd = [sys.executable, "main.py", f"--agent={args.agent}"]
    if args.game:
        cmd.append(f"--game={args.game}")

    result = subprocess.run(cmd, cwd=AGENTS_DIR, capture_output=True, text=True)
    log_text = result.stdout + "\n" + result.stderr

    card = extract_scorecard(log_text)

    SCORECARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SCORECARDS_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(card, indent=2))

    print(f"score:                    {card.get('score')}")
    print(f"total_environments:       {card.get('total_environments')}")
    print(f"total_environments_completed: {card.get('total_environments_completed')}")
    print(f"total_levels_completed:  {card.get('total_levels_completed')}")
    print(f"total_levels:            {card.get('total_levels')}")
    print(f"total_actions:           {card.get('total_actions')}")
    print(f"saved to:                {out_path}")


if __name__ == "__main__":
    main()
