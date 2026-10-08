#!/usr/bin/env bash
# Install LIBERO + robosuite inside WSL for Panda sim eval.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venv_wsl_libero"
LIBERO_DIR="${ROOT}/third_party/LIBERO"

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

mkdir -p "${ROOT}/third_party"
if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi

python -m pip install "torch" --index-url https://download.pytorch.org/whl/cu124
# mujoco 3.15 breaks robosuite 1.4 joint addressing; pin 3.1.1
python -m pip install "mujoco==3.1.1" imageio imageio-ffmpeg opencv-python-headless numpy tqdm
python -m pip install "robosuite==1.4.1"
cd "$LIBERO_DIR"
python -m pip install -e . --config-settings editable_mode=compat
python -m pip install pyyaml bddl easydict cloudpickle future gym matplotlib
# Match Windows train stack (v2.1 datasets / our checkpoints). lerobot>=0.6 needs v3 datasets.
python -m pip install "lerobot==0.3.2" transformers num2words
python "${ROOT}/scripts/wsl_init_libero_config.py"
python "${ROOT}/scripts/wsl_verify_libero.py"
