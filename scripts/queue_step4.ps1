# Step 4 cuts in while the step-3 matrix waits (one training process at a time on this laptop).
# Before starting it, create artifacts\runs\matrix\PAUSE: the matrix run then stops before its
# next cell. This queue waits for process $WaitPid (the paused matrix run, or a step-4 job already
# running), then:
#   1. the two µNAS encoders, English clicks (skipped where finished)
#   2. the hand-designed comparators in the fork's terms, English clicks: 64-5-384 and the grid's
#      best within each budget (64-2-384 for the H7, 32-5-384 for the F401)
#   3. the matrix, after removing the pause file
#   4. the two µNAS encoders, mixed-language clicks
# Logs: logs\<tag>.log, GPU samples logs\<tag>_gpu.csv.
#
#   powershell -ExecutionPolicy Bypass -File scripts\queue_step4.ps1 -WaitPid 17624
param([int]$WaitPid = 0)
$root = Split-Path -Parent $PSScriptRoot
$launch = Join-Path $PSScriptRoot "launch_job.ps1"
if ($WaitPid) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') process $WaitPid ended"
}
function Run-Job([string]$Module, [string]$Tag, [string]$ArgString) {
    $stamp = Get-Date -Format "MMddHHmm"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launch -Module $Module -Tag "${Tag}_$stamp" -ArgString $ArgString
}
Run-Job scripts.run_unas_full "unas_full_en" "--protocols distill_ft_en"
Run-Job scripts.run_unas_full "unas_full_hand_en" "--models hand_64-5-384 hand_64-2-384 hand_32-5-384 --protocols distill_ft_en"
Remove-Item (Join-Path $root "artifacts\runs\matrix\PAUSE") -ErrorAction SilentlyContinue
Run-Job scripts.run_matrix "matrix" "--seeds 42 1 2"
Run-Job scripts.run_unas_full "unas_full_mixed" "--protocols distill_ft_mixed"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queue done"
