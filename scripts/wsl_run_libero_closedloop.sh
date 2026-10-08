#!/usr/bin/env bash
# Closed-loop LIBERO with ego-finetuned SmolVLA (domain transfer).
set -euo pipefail
ROOT="/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project"
# Prefer project WSL venv (has lerobot); fall back to home venv.
if [[ -f "${ROOT}/.venv_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv_wsl_libero/bin/activate"
elif [[ -f "$HOME/hip_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/hip_wsl_libero/bin/activate"
else
  echo "No WSL LIBERO venv. Run scripts/wsl_install_libero.sh first." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONIOENCODING=utf-8
cd "$ROOT"
python -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m src.eval.rollout_libero_closedloop \
  --mode ego-ft \
  --ckpt outputs/train/smolvla_custom/checkpoint-2000 \
  --out outputs/eval/libero_closedloop_ego_ft.mp4 \
  --task-suite libero_object \
  --task-id 0 \
  --max-steps 200 \
  --scale 5.0
