# Append one nvidia-smi sample (time, GPU %, memory MiB, power W, temperature C) to a CSV
# every $Every seconds while process $WatchPid is alive. Each sample is written and the file
# closed at once (nvidia-smi's own "-l -f" buffers its file on Windows until it exits).
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_sampler.ps1 -Out logs\job_gpu.csv -WatchPid 1234
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][int]$WatchPid,
    [int]$Every = 30
)
while (Get-Process -Id $WatchPid -ErrorAction SilentlyContinue) {
    $line = & nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
    Add-Content -Path $Out -Value $line -Encoding utf8
    Start-Sleep -Seconds $Every
}
