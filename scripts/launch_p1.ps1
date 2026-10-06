# Launch scripts/run_p1.py for the given seeds as ONE process at below-normal
# priority, so the laptop stays usable while it runs (two parallel processes at
# 98 % GPU load froze the desktop on 2026-10-05). Output: logs/p1_<tag>.log.
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch_p1.ps1 -Seeds 42 -Tag seed42
#   powershell -ExecutionPolicy Bypass -File scripts\launch_p1.ps1 -Seeds "1,2" -Tag seeds1-2 -Stages "teacher,nrms,student"
# Seeds and stages are comma-separated strings: with `powershell -File` an
# argument such as `-Seeds 1,2` arrives as the text "1,2", and binding it to
# [int[]] produced the single seed 12 on 2026-10-06.
param(
    [string]$Seeds = "42,1,2",
    [string]$Stages = "teacher,nrms,student",
    [string]$Tag = "run"
)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $root
$log = Join-Path $root "logs\p1_$Tag.log"
$seedList = @($Seeds -split '[,\s]+' | Where-Object { $_ -ne "" })
$stageList = @($Stages -split '[,\s]+' | Where-Object { $_ -ne "" })
$args = @("-u", "-m", "scripts.run_p1") + $stageList + @("--seeds") + $seedList
$p = Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") -ArgumentList $args `
        -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru
$p.PriorityClass = "BelowNormal"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') started pid $($p.Id) seeds=$($seedList -join ' ') stages=$($stageList -join ' ') log=$log"
$p.WaitForExit()
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit code $($p.ExitCode)"
