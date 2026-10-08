#!/usr/bin/env bash
# Install LIBERO into a Linux-home venv (avoids OneDrive /mnt/c pip bugs).
set -euo pipefail

VENV="${HOME}/hip_wsl_libero"
ROOT="/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project"
LIBERO="${ROOT}/third_party/LIBERO"

rm -rf "${VENV}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -U pip wheel

python -m pip install "numpy<2.1" "mujoco==3.1.1" "robosuite==1.4.1" \
  pyyaml bddl easydict cloudpickle "gym==0.25.2" future matplotlib einops \
  opencv-python-headless imageio imageio-ffmpeg tqdm
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Pre-write LIBERO config to avoid interactive prompt on first import.
mkdir -p "${HOME}/.libero"
python - <<'PY'
import os, yaml
root = "/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/third_party/LIBERO/libero/libero"
cfg = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "datasets": os.path.join(root, "../datasets"),
    "assets": os.path.join(root, "assets"),
}
with open(os.path.expanduser("~/.libero/config.yaml"), "w") as f:
    yaml.dump(cfg, f)
PY

cd "${LIBERO}"
python -m pip install -e . --no-deps

export PYTHONPATH="${LIBERO}:${PYTHONPATH:-}"
python - <<'PY'
import mujoco
import robosuite
from libero.libero import benchmark

print("mujoco", mujoco.__version__)
print("robosuite", robosuite.__version__)
print("libero suites:", list(benchmark.get_benchmark_dict().keys()))
print("LIBERO_OK")
print("VENV", __import__("sys").prefix)
PY
