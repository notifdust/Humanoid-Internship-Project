# Phase A: 4K → 1080p → MediaPipe hands
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

if (-not (Test-Path .venv)) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install -q -r requirements.txt
python -m src.preprocess.resize_videos
python -m src.hands.extract_mediapipe
Write-Host "Phase A complete. See docs/GUIDELINE.md"
