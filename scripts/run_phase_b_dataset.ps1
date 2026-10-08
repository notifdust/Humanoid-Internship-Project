# Phase B finish: overlays + NPZ episodes
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\Activate.ps1
python -m src.viz.make_overlays --clips 01,16,30,34,41
python -m src.dataset.build_episodes
Write-Host "Phase B dataset+viz done."
