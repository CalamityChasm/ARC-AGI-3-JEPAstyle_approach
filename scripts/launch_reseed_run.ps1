param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][int]$Seed,
    [switch]$AblateGameId
)

$wt = "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\.claude\worktrees\agent-aef6e0853e41a354a"
$py = "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\venv\Scripts\python.exe"
$encoder = "C:\Users\desktop-06\Cal\ARC-AGI-3-JEPAstyle_approach\checkpoints\encoder.pt"
$logDir = "$wt\logs\gameid_reseed"
$out = "$wt\checkpoints_reseed\$Name"
$log = "$logDir\$Name.log"

$ablateArg = ""
if ($AblateGameId) { $ablateArg = "--ablate-game-id" }

$cmdline = "`$env:JEPA_NUM_WORKERS=0; & '$py' -u -m jepa.train_moe_predictor --pretrain-epochs 20 --epochs 60 --num-experts 8 --external-per-game 2000 --exclude-games r11l,bp35,m0r0,tr87,ka59 --contrast-weight 0.0 $ablateArg --checkpoint-every 5 --encoder '$encoder' --out '$out' --seed $Seed *>> '$log'"
$p = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile","-Command",$cmdline) -WorkingDirectory $wt -WindowStyle Hidden -PassThru
Write-Output "Launched $Name (seed=$Seed, ablate=$AblateGameId) as PID $($p.Id), logging to $log"
