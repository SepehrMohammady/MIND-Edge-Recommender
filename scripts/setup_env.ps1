# Sets up the project venv, downloads the datasets, then installs the heavy
# ML stack. Order matters: data lands on disk BEFORE the long torch install,
# so a mid-install internet drop never blocks dataset acquisition.
#
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path $PSScriptRoot -Parent)

$venvPy = ".\.venv\Scripts\python.exe"

Write-Output "==> [1/5] Creating venv (.venv)"
if (-not (Test-Path $venvPy)) { python -m venv .venv }
& $venvPy -m pip install --upgrade pip

Write-Output "==> [2/5] Installing lightweight data deps"
& $venvPy -m pip install -r requirements-core.txt

Write-Output "==> [3/5] Downloading MIND + xMIND (all 14 langs)"
& $venvPy -m src.download

Write-Output "==> [4/5] Installing PyTorch (cu130, Blackwell sm_120)"
& $venvPy -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

Write-Output "==> [5/5] Installing remaining ML stack + Jupyter kernel"
& $venvPy -m pip install -r requirements.txt
& $venvPy -m ipykernel install --user --name mind-edge-recommender --display-name "MIND-Edge-Recommender (.venv)"

Write-Output "==> Freezing exact versions -> requirements-lock.txt"
& $venvPy -m pip freeze | Out-File -Encoding utf8 requirements-lock.txt

Write-Output ""
Write-Output "SETUP COMPLETE"
& $venvPy -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cap', torch.cuda.get_device_capability() if torch.cuda.is_available() else None)"
