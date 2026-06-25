# Probe the TensorFlow + Larq + Larq-Compute-Engine toolchain in a SEPARATE venv
# (kept apart from the torch venv to avoid numpy/dep conflicts). Reveals whether
# the binary deployment path is installable on this Windows / Python 3.13 box.
$ErrorActionPreference = "Continue"
Set-Location -Path (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv-tf\Scripts\python.exe"

if (-not (Test-Path $py)) { python -m venv .venv-tf }
& $py -m pip install --upgrade pip
Write-Output "==> installing tensorflow"
& $py -m pip install "tensorflow"
Write-Output "==> installing larq"
& $py -m pip install larq
Write-Output "==> installing larq-compute-engine (may be Linux-only)"
& $py -m pip install larq-compute-engine
Write-Output "==> versions"
& $py -c "import tensorflow as tf; print('TF', tf.__version__)"
& $py -c "import larq; print('larq', larq.__version__)"
& $py -c "import larq_compute_engine as lce; print('LCE OK')"
Write-Output "TF_SETUP_DONE"
