#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"

echo "[1/5] Updating apt metadata"
sudo apt update

echo "[2/5] Installing system packages"
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  espeak-ng \
  alsa-utils \
  libatlas-base-dev

echo "[3/5] Creating virtual environment"
python3 -m venv "${VENV_DIR}"

echo "[4/5] Upgrading pip toolchain"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo "[5/5] Installing Python dependencies"
pip install -r "${REPO_DIR}/requirements.txt"

cat <<'EOF'

Setup completed.

Next steps:
1. source .venv/bin/activate
2. python main.py

If torch or mediapipe fails to install on your Raspberry Pi OS image,
install the matching wheels manually and rerun:
pip install -r requirements.txt
EOF
