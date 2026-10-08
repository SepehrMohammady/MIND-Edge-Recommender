# Step 4 cuts in while the step-3 matrix waits (one training process at a time on this laptop).
# Before starting it, create artifacts\runs\matrix\PAUSE: the matrix run then stops before its
# next cell. This queue waits for that process to exit, trains the two µNAS encoders on English
# clicks, removes the pause file and resumes the matrix, then runs the mixed-language protocol
# of the µNAS encoders. Logs: logs\<tag>.log, GPU samples logs\<tag>_gpu.csv.
#
#   powershell -ExecutionPolicy Bypass -File scripts\queue_step4.ps1 -WaitPid 17624
param([int]$WaitPid = 0)
$root = Split-Path -Parent $PSScriptRoot
$launch = Join-Path $PSScriptRoot "launch_job.ps1"
if ($WaitPid) {
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 20 }
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') process $WaitPid ended"
}
$stamp = Get-Date -Format "MMddHHmm"
& powershell -NoProfile -ExecutionPolicy Bypass -File $launch -Module scripts.run_unas_full `
    -Tag "unas_full_en_$stamp" -ArgString "--protocols distill_ft_en"
Remove-Item (Join-Path $root "artifacts\runs\matrix\PAUSE") -ErrorAction SilentlyContinue
$stamp = Get-Date -Format "MMddHHmm"
& powershell -NoProfile -ExecutionPolicy Bypass -File $launch -Module scripts.run_matrix `
    -Tag "matrix_$stamp" -ArgString "--seeds 42 1 2"
$stamp = Get-Date -Format "MMddHHmm"
& powershell -NoProfile -ExecutionPolicy Bypass -File $launch -Module scripts.run_unas_full `
    -Tag "unas_full_mixed_$stamp" -ArgString "--protocols distill_ft_mixed"
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') queue done"
