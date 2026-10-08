$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python.exe -m src.train.train_bc_baseline @args
