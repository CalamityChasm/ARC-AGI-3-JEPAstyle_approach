#!/bin/bash
set -uo pipefail
cd "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\.claude\worktrees\agent-aeab967f722885c82"
PYTHON="C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\venv\Scripts\python.exe"
GAMES="r11l,bp35,m0r0,tr87,ka59"
N=25
LOG="logs/scorecards/_tta_confirmatory_progress.log"
mkdir -p logs/scorecards
echo "sweep started $(date)" > "$LOG"

run_condition() {
  local label=$1 max_actions=$2 tta=$3
  export HYPOTHESIS_MAX_ACTIONS="$max_actions"
  export HYPOTHESIS_TEST_TIME_ADAPT="$tta"
  for i in $(seq 1 $N); do
    echo "$label r$i starting $(date)" >> "$LOG"
    "$PYTHON" scripts/run_scorecard.py --agent hypothesis --label "${label}_r${i}" --game "$GAMES" >> "$LOG" 2>&1
    echo "$label r$i done $(date)" >> "$LOG"
  done
}

run_condition baseline300_ttaoff 300 0
run_condition budget900_ttaoff 900 0
run_condition baseline300_ttaon 300 1
run_condition budget900_ttaon 900 1

echo "sweep finished $(date)" >> "$LOG"
touch logs/scorecards/_tta_confirmatory_DONE
