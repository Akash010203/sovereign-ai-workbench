# SovereignAI - Windows 11 setup script (Phase 1)
#
# Run this from PowerShell inside the project root folder:
#     .\scripts\setup_windows.ps1
#
# What it does:
#   1. Creates a Python virtual environment (.venv)
#   2. Activates it for the rest of this script
#   3. Upgrades pip
#   4. Installs project dependencies from requirements.txt
#   5. Runs the GPU verification script

Write-Host "== SovereignAI Windows setup ==" -ForegroundColor Cyan

python -m venv .venv

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Failed to create .venv. Is Python 3.10+ installed and on PATH?" -ForegroundColor Red
    exit 1
}

. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "== Verifying GPU / PyTorch installation ==" -ForegroundColor Cyan
python scripts\verify_gpu.py

Write-Host ""
Write-Host "Setup complete. In new terminals, re-activate the environment with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"
