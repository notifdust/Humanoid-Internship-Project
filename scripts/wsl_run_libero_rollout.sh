#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project"
if [[ -f "$HOME/hip_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/hip_wsl_libero/bin/activate"
elif [[ -f "${ROOT}/.venv_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv_wsl_libero/bin/activate"
else
  echo "No WSL LIBERO venv found. Run scripts/wsl_install_libero.sh first." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$ROOT"
python -m src.eval.rollout_libero_openloop \
  --actions outputs/eval/smolvla_rollout_41_actions.npy \
  --out outputs/eval/libero_openloop_smolvla_41.mp4 \
  --task-suite libero_object \
  --task-id 0 \
  --scale 5.0
