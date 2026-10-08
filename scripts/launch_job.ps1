# Run one experiment module as ONE process at below-normal priority, with an
# nvidia-smi sample every 30 s next to its log. Only one training process runs on
# this laptop at a time (two at 98 % GPU load froze it on 2026-10-05).
#
#   powershell -ExecutionPolicy Bypass -File scripts\launch_job.ps1 -Module scripts.run_matrix -Tag matrix_1008 -ArgString "--seeds 42 1 2"
#
# Output: logs/<Tag>.log (+ .err) and logs/<Tag>_gpu.csv. Arguments go in one string:
# with `powershell -File` a list such as `-Seeds 1,2` arrives as text (see launch_p1.ps1).
param(
    [Parameter(Mandatory = $true)][string]$Module,
    [string]$ArgString = "",
    [string]$Tag = "run"
)
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $root
$log = Join-Path $root "logs\$Tag.log"
$gpu = Join-Path $root "logs\${Tag}_gpu.csv"
$pyArgs = @("-u", "-m", $Module) + @($ArgString -split '\s+' | Where-Object { $_ -ne "" })
$p = Start-Process -FilePath (Join-Path $root ".venv\Scripts\python.exe") -ArgumentList $pyArgs `
        -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru
$p.PriorityClass = "BelowNormal"
Start-Process -FilePath powershell.exe -WindowStyle Hidden -ArgumentList `
    "-NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\gpu_sampler.ps1`" -Out `"$gpu`" -WatchPid $($p.Id)"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') started pid $($p.Id): $Module $ArgString -> $log"
$p.WaitForExit()
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Module exit code $($p.ExitCode)"
