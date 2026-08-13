#!/bin/bash
# Stage6 novelty-aware-beta agent-level backtest driver.
# Runs run_scorecard.py repeatedly for both cap conditions, on both the
# held-out (novel) games and a trained-games sanity subset, using the
# checkpoints_holdout_baseline checkpoint already swapped into checkpoints/.
set -e
cd "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\.claude\worktrees\agent-a48054c6388b01f72"
PY="C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\venv\Scripts\python.exe"
HELDOUT_GAMES="r11l,bp35,m0r0,tr87,ka59"
TRAINED_GAMES="ft09,sp80,cd82,lp85,vc33,tn36"

echo "=== held-out games: cap ON (0.15) ==="
for i in 1 2 3 4 5 6 7 8; do
  HYPOTHESIS_NOVELTY_BETA_CAP=0.15 "$PY" scripts/run_scorecard.py --agent hypothesis \
    --label "holdout_cap_on_r${i}" --game "$HELDOUT_GAMES"
done

echo "=== held-out games: cap OFF (1.0, no-op == today's default) ==="
for i in 1 2 3 4 5 6 7 8; do
  HYPOTHESIS_NOVELTY_BETA_CAP=1.0 "$PY" scripts/run_scorecard.py --agent hypothesis \
    --label "holdout_cap_off_r${i}" --game "$HELDOUT_GAMES"
done

echo "=== trained games sanity check: cap ON (0.15) ==="
for i in 1 2 3 4 5; do
  HYPOTHESIS_NOVELTY_BETA_CAP=0.15 "$PY" scripts/run_scorecard.py --agent hypothesis \
    --label "trained_cap_on_r${i}" --game "$TRAINED_GAMES"
done

echo "=== trained games sanity check: cap OFF (1.0) ==="
for i in 1 2 3 4 5; do
  HYPOTHESIS_NOVELTY_BETA_CAP=1.0 "$PY" scripts/run_scorecard.py --agent hypothesis \
    --label "trained_cap_off_r${i}" --game "$TRAINED_GAMES"
done

echo "=== ALL RUNS COMPLETE ==="
