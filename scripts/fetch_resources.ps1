# Download open-access papers and the free Sutton & Barto PDF into resources/.
# Modern Robotics: open the official preprint yourself (no redistribution) — see resources/books/README.md

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PaperDir = Join-Path $Root "resources\papers\pdf"
$BookDir = Join-Path $Root "resources\books"
New-Item -ItemType Directory -Force -Path $PaperDir, $BookDir | Out-Null

$Papers = [ordered]@{
  "smolvla_2506.01844.pdf"           = "https://arxiv.org/pdf/2506.01844"
  "libero_2312.08467.pdf"            = "https://arxiv.org/pdf/2312.08467"
  "egovla_2507.12440.pdf"            = "https://arxiv.org/pdf/2507.12440"
  "egomimic_2410.24221.pdf"          = "https://arxiv.org/pdf/2410.24221"
  "openvla_2406.09246.pdf"           = "https://arxiv.org/pdf/2406.09246"
  "hamer_2312.05251.pdf"             = "https://arxiv.org/pdf/2312.05251"
  "irasim_2406.14540.pdf"            = "https://arxiv.org/pdf/2406.14540"
  "ivideogpt_2405.15223.pdf"         = "https://arxiv.org/pdf/2405.15223"
  "uwm_2504.02792.pdf"               = "https://arxiv.org/pdf/2504.02792"
  "world_models_ha_1803.10122.pdf"   = "https://arxiv.org/pdf/1803.10122"
  "dreamerv3_2301.04104.pdf"         = "https://arxiv.org/pdf/2301.04104"
  "flip_2412.08261.pdf"              = "https://arxiv.org/pdf/2412.08261"
  "octo_2405.12213.pdf"              = "https://arxiv.org/pdf/2405.12213"
  "diffusion_policy_2303.04137.pdf"  = "https://arxiv.org/pdf/2303.04137"
  "rt2_2307.15818.pdf"               = "https://arxiv.org/pdf/2307.15818"
  "pi0_2410.24164.pdf"               = "https://arxiv.org/pdf/2410.24164"
  "gwm_2508.17600.pdf"               = "https://arxiv.org/pdf/2508.17600"
}

foreach ($Name in $Papers.Keys) {
  $Dest = Join-Path $PaperDir $Name
  if (Test-Path $Dest) {
    Write-Host "Skip (exists): $Name"
    continue
  }
  Write-Host "Fetching $Name ..."
  Invoke-WebRequest -Uri $Papers[$Name] -OutFile $Dest -UseBasicParsing -TimeoutSec 180
}

$Book = Join-Path $BookDir "SuttonBarto_RL_2ndEd.pdf"
if (-not (Test-Path $Book)) {
  Write-Host "Fetching Sutton & Barto RL 2nd ed. ..."
  Invoke-WebRequest -Uri "http://incompleteideas.net/book/RLbook2020.pdf" -OutFile $Book -UseBasicParsing -TimeoutSec 300
} else {
  Write-Host "Skip (exists): SuttonBarto_RL_2ndEd.pdf"
}

Write-Host "Done. See resources/papers/READING_LIST.md and resources/books/README.md"
