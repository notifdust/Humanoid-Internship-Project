$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
Set-Location $PSScriptRoot\..
.\.venv\Scripts\python.exe -m src.viz.make_demo_reel --out outputs/demo/challenge_demo.mp4
Write-Host "Demo ready: outputs/demo/challenge_demo.mp4"
