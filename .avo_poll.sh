#!/bin/bash
# Poll the AVO arm's free kernel until it leaves QUEUED/RUNNING.
K=calamitychasm/arc3-duck-nvfp4-avo
PY=/c/Users/desktop-06/Cal/ARC-AGI-3-JEPAstyle_approach/venv/Scripts/kaggle.exe
for i in $(seq 1 700); do
  S=$(PYTHONUTF8=1 "$PY" kernels status "$K" 2>&1 | tail -1)
  echo "$(date -u +%H:%M:%S) $S"
  case "$S" in
    *QUEUED*|*RUNNING*) sleep 180 ;;
    *) echo "TERMINAL: $S"; exit 0 ;;
  esac
done
echo "poll window exhausted"
