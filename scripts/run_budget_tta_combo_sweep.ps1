# Driver for stage6-budget-tta-combo: runs the two NEW conditions
# (MAX_ACTIONS=900 x TTA off, MAX_ACTIONS=900 x TTA on) at n=8 each on
# the 5 held-out games. The other two conditions (MAX_ACTIONS=300 x TTA
# off/on) are reused directly from experiments/stage6_test_time_adaptation_agent.md
# since this worktree uses the byte-identical checkpoints_holdout_baseline
# checkpoint and the same run_scorecard.py protocol.
#
# Run detached: this script itself blocks until done and writes a DONE
# marker; the caller launches it via Start-Process so it survives beyond
# one tool-call turn.

$ErrorActionPreference = "Continue"
$root = "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\.claude\worktrees\agent-aeab967f722885c82"
Set-Location $root
$python = "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\venv\Scripts\python.exe"
$games = "r11l,bp35,m0r0,tr87,ka59"
$logFile = "$root\logs\scorecards\_sweep_progress.log"

"sweep started $(Get-Date)" | Out-File -FilePath $logFile -Encoding utf8

# Condition A: MAX_ACTIONS=900, TTA off
$env:HYPOTHESIS_MAX_ACTIONS = "900"
$env:HYPOTHESIS_TEST_TIME_ADAPT = "0"
Remove-Item Env:\HYPOTHESIS_TEST_TIME_ADAPT -ErrorAction SilentlyContinue
for ($i = 1; $i -le 8; $i++) {
    "budget900_tta_off r$i starting $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
    & $python scripts\run_scorecard.py --agent hypothesis --label "budget900_tta_off_r$i" --game $games *>> $logFile
    "budget900_tta_off r$i done $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
}

# Condition B: MAX_ACTIONS=900, TTA on
$env:HYPOTHESIS_MAX_ACTIONS = "900"
$env:HYPOTHESIS_TEST_TIME_ADAPT = "1"
for ($i = 1; $i -le 8; $i++) {
    "budget900_tta_on r$i starting $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
    & $python scripts\run_scorecard.py --agent hypothesis --label "budget900_tta_on_r$i" --game $games *>> $logFile
    "budget900_tta_on r$i done $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
}

"sweep finished $(Get-Date)" | Out-File -FilePath $logFile -Append -Encoding utf8
New-Item -ItemType File -Path "$root\logs\scorecards\_sweep_DONE" -Force | Out-Null
