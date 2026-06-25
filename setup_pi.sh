#!/usr/bin/env bash
# ============================================================
# Setup script for Driver Drowsiness Monitor - Raspberry Pi Kiosk
# ============================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/.venv"
SERVICE_NAME="drowsiness-monitor"

echo "[1/6] Updating apt metadata"
sudo apt update

echo "[2/6] Installing system packages"
sudo apt install -y \
  python3 \
  python3-venv \
  python3-pip \
  python3-opencv \
  espeak-ng \
  alsa-utils \
  libatlas-base-dev \
  libhdf5-dev \
  libhdf5-serial-dev \
  libjasper-dev \
  libqtgui4 \
  libqt4-test

echo "[3/6] Creating virtual environment"
python3 -m venv "${VENV_DIR}"

echo "[4/6] Upgrading pip toolchain"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

echo "[5/6] Installing Python dependencies"
pip install -r "${REPO_DIR}/requirements.txt"

echo "[6/6] Installing systemd service"
SERVICE_SRC="${REPO_DIR}/drowsiness-monitor.service.example"
SERVICE_DST="/etc/systemd/system/${SERVICE_NAME}.service"
if [ -f "$SERVICE_SRC" ]; then
  sudo cp "$SERVICE_SRC" "$SERVICE_DST"
  sudo systemctl daemon-reload
  sudo systemctl enable "${SERVICE_NAME}.service"
  echo "  Service installed and enabled: ${SERVICE_NAME}"
  echo "  Start now with: sudo systemctl start ${SERVICE_NAME}"
else
  echo "  WARNING: ${SERVICE_SRC} not found, skipping service install"
fi

cat <<'EOF'

======================== Setup Complete ========================

Hardware Pinout (BCM):
  GPIO 2  (pin 3)  - Calibration LED
  GPIO 3  (pin 5)  - Inference LED
  GPIO 14 (pin 8)  - Restart button (pull-up, active low)
  GPIO 15 (pin 10) - Pause button (pull-up, active low)

Quick start:
  cd driver-drowsiness-monitor
  source .venv/bin/activate
  python run_kiosk.py

Auto-start via systemd:
  sudo systemctl start drowsiness-monitor
  sudo systemctl status drowsiness-monitor

Video analysis mode:
  python run_hybrid.py --video <path> --output runs/demo

===============================================================
EOF
