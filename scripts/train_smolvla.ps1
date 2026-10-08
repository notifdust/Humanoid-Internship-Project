# Fine-tune SmolVLA on ego-retarget dataset (RTX 4060-safe defaults).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$OutDir = if ($args.Count -gt 0) { $args[0] } else { "outputs/train/smolvla_pick_place_mug" }
$Steps = if ($args.Count -gt 1) { $args[1] } else { "1000" }
$LogDir = "outputs/logs"
$Log = Join-Path $LogDir "smolvla_train.log"
New-Item -ItemType Directory -Force -Path $OutDir, $LogDir | Out-Null

Write-Host "Logging to $Log"
Write-Host "Watch: Get-Content $Log -Wait -Tail 40"

& .\.venv\Scripts\lerobot-train.exe `
  --policy.path=lerobot/smolvla_base `
  --policy.device=cuda `
  --policy.push_to_hub=false `
  --policy.empty_cameras=2 `
  --policy.train_expert_only=true `
  --policy.freeze_vision_encoder=true `
  --dataset.repo_id=local/pick_place_mug_smolvla `
  --dataset.root=data/processed/lerobot_smolvla_mug `
  --output_dir=$OutDir `
  --job_name=smolvla_ego_mug `
  --batch_size=1 `
  --steps=$Steps `
  --save_freq=500 `
  --log_freq=20 `
  --num_workers=0 `
  --wandb.enable=false `
  2>&1 | Tee-Object -FilePath $Log
