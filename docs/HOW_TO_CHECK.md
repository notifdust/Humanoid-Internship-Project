# How to check work in progress

## Quick looks (no code)

| What | Where |
|------|--------|
| Plan / checklist | [`docs/GUIDELINE.md`](GUIDELINE.md) |
| Wrist-tracking overlays | `outputs/overlays/*.mp4` |
| BC baseline predictions | `outputs/eval/bc_rollout_41.mp4` |
| SmolVLA ego overlay (held-out clip 41) | `outputs/eval/smolvla_rollout_41.mp4` + `.json` |
| MuJoCo EE sim (GT actions) | `outputs/eval/mujoco_ee_gt_41.mp4` |
| MuJoCo EE sim (SmolVLA actions) | `outputs/eval/mujoco_ee_smolvla_41.mp4` |
| LIBERO Panda open-loop (WSL) | `outputs/eval/libero_openloop_smolvla_41.mp4` |
| LIBERO closed-loop (ego-ft) | `outputs/eval/libero_closedloop_ego_ft.mp4` |
| Side-by-side ego\|pred\|sim | `outputs/eval/side_by_side_41.mp4` |
| Ablation MSE table | `outputs/eval/ablation_mse.json` |
| Closed-loop research notes | [`RESEARCH_CLOSEDLOOP.md`](RESEARCH_CLOSEDLOOP.md) |
| WM future-frame strips | `outputs/wm/future_frame_strip_*.gif` (+ `.mp4`) |
| WM train curve | `outputs/wm/future_frame/history.json` |
| Retarget path plots | `outputs/viz/retarget/*_retarget.png` |
| LIBERO open-loop multi-seed | `outputs/eval/libero_openloop_multiseed.json` |
| BC training curve | `outputs/checkpoints/bc_baseline/history.json` |
| SmolVLA training curve | `outputs/train/smolvla_custom/history.json` |
| SmolVLA checkpoints | `outputs/train/smolvla_custom/checkpoint-*` |
| Packed episodes | `data/processed/episodes/*.npz` |
| LeRobot / SmolVLA dataset | `data/processed/lerobot_smolvla_mug/` |

## Live training

```powershell
Get-Content outputs\train\smolvla_custom\train.log -Wait -Tail 40
```

## GPU activity

```powershell
nvidia-smi -l 2
```

You should see Python using the 4060 when training/eval is active.

## Re-run evals

```powershell
.\.venv\Scripts\python.exe -m src.eval.rollout_smolvla_overlay
.\.venv\Scripts\python.exe -m src.eval.rollout_mujoco_ee --actions outputs\eval\smolvla_rollout_41_actions.npy --out outputs\eval\mujoco_ee_smolvla_41.mp4
wsl -e bash /mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/scripts/wsl_run_libero_rollout.sh
wsl -e bash /mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/scripts/wsl_run_libero_closedloop.sh
.\.venv\Scripts\python.exe -m src.eval.ablation_mse
.\.venv\Scripts\python.exe -m src.viz.make_side_by_side
.\.venv\Scripts\python.exe -m src.viz.plot_retarget
wsl -e bash /mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/scripts/wsl_run_libero_multiseed.sh
```
