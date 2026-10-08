#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project"
if [[ -f "${ROOT}/.venv_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv_wsl_libero/bin/activate"
elif [[ -f "$HOME/hip_wsl_libero/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/hip_wsl_libero/bin/activate"
else
  echo "No WSL LIBERO venv." >&2
  exit 1
fi
export PYTHONPATH="${ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$ROOT"
python -m src.eval.libero_openloop_multiseed \
  --actions outputs/eval/smolvla_rollout_41_actions.npy \
  --out outputs/eval/libero_openloop_multiseed.json \
  --task-suite libero_object \
  --task-id 0 \
  --seeds 0,1,2,3,4 \
  --scale 5.0
