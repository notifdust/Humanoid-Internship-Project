# Phase C deps: PyTorch CUDA + LeRobot (requires Python 3.12, not 3.14)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTHONIOENCODING = "utf-8"

$py312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py312)) { throw "Python 3.12 not found. Install: winget install Python.Python.3.12" }

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  & $py312 -m venv .venv
}

.\.venv\Scripts\python.exe -c "import sys; assert sys.version_info[:2] == (3,12), sys.version"
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
