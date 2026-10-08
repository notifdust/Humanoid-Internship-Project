#!/usr/bin/env bash
set -euo pipefail
source "$HOME/hip_wsl_libero/bin/activate"
python /mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/scripts/wsl_fix_libero_path.py
# Prefer PYTHONPATH over broken editable finder on /mnt/c
export PYTHONPATH="/mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/third_party/LIBERO:${PYTHONPATH:-}"
python /mnt/c/Users/gabri/OneDrive/Progetti/Humanoid-Internship-Project/scripts/wsl_verify_libero.py
