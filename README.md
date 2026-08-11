# Driver Drowsiness Monitor for Raspberry Pi 4

Real-time driver monitoring kiosk for Raspberry Pi 4 and Linux/WSL development. The system uses a webcam, MediaPipe facial landmarks, head-pose estimation, an LSTM drowsiness classifier, phone detection, GPIO controls, and Vietnamese WAV voice alerts.

The main production entry point is `run_kiosk.py`.

## Features

- 800x480 kiosk UI for embedded screens.
- Webcam-based face landmark tracking.
- Startup calibration with strict stability checks.
- Drowsiness scoring from normalized facial features and the LSTM classifier.
- Yawning detection from calibrated MAR.
- Distraction detection from calibrated head pose.
- Phone-use detection with YOLO/ONNX object detection.
- Optional debug UI with normalized EAR, MAR, PUC, MoE, head pose, and LSTM output.
- GPIO LEDs and buttons for Raspberry Pi deployment.
- Pygame-based audio worker using local WAV assets only.
- Session reports and optional video recording.

## Repository Layout

- `run_kiosk.py`: main kiosk entry point.
- `hybrid_system/`: hybrid detection, calibration, UI, display, audio, and reporting pipeline.
- `hybrid_system/configs/default.yaml`: default kiosk configuration.
- `assets/audio/`: bundled WAV prompts and alert sounds used at runtime.
- `models/`: MediaPipe, head-pose, and drowsiness models.
- `tools/`: local utility scripts, including audio tests and asset preparation.
- `tests/`: focused tests for the current pipeline.
- `setup_pi.sh`: Raspberry Pi setup helper.

## Requirements

Recommended:

- Raspberry Pi OS 64-bit or Ubuntu/WSL for development.
- Python 3.10 or 3.11.
- USB camera exposed as `/dev/video0` or another V4L2 device.
- Speaker supported by ALSA/PulseAudio.

## Installation

Clone branch `clean-main`:

```bash
git clone --branch clean-main --single-branch \
  https://github.com/TueNguyen2006/driver-drowsiness-monitor-rpi4.git

cd driver-drowsiness-monitor-rpi4
```
Install system packages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-opencv alsa-utils libatlas-base-dev
```

Install Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Or use the Raspberry Pi setup script:

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

## Running Kiosk Mode

From the repo root:

```bash
source .venv/bin/activate
python3 run_kiosk.py
```

Common options:

```bash
python3 run_kiosk.py --camera-index 0
python3 run_kiosk.py --camera-fourcc MJPG
python3 run_kiosk.py --debug-ui
python3 run_kiosk.py --windowed
python3 run_kiosk.py --no-phone
python3 run_kiosk.py --no-display
python3 run_kiosk.py --threads 1
python3 run_kiosk.py --write-video
```

For WSL from Windows PowerShell:

```powershell
wsl -d Ubuntu-22.04 --cd /home/tuenguyen/driver-drowsiness-monitor-rpi4
source .venv/bin/activate
python3 run_kiosk.py --camera-index 0
```

If the camera is attached from Windows to WSL with `usbipd`, run the attach step from Windows first:

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl Ubuntu-22.04 --busid <BUSID>
```

Then verify inside WSL:

```bash
ls -l /dev/video*
```

## Audio

Audio no longer uses `espeak-ng`, `ffplay`, temporary WAV files, or subprocess playback.

Runtime audio uses only WAV files from:

```text
assets/audio/
```

The audio implementation:

- initializes `pygame.mixer` once when kiosk starts;
- preloads WAV files into memory;
- uses one worker thread and one voice queue;
- uses separate mixer channels for voice and alert beep;
- avoids overlapping voice prompts;
- applies cooldowns so repeated detector states do not spam speech;
- logs a warning and continues if a WAV file is missing;
- shuts down the worker, channels, and mixer when kiosk exits.

Test audio without a real audio device:

```bash
SDL_AUDIODRIVER=dummy .venv/bin/python -m unittest tests.test_hardware_audio
```

Play a sample prompt:

```bash
.venv/bin/python tools/play_audio_sample.py "Hệ thống đã sẵn sàng.wav"
```

## Calibration

At startup, kiosk plays the calibration preparation prompt first. Calibration starts only after that prompt finishes.

Calibration requires stable face samples:

- `ear_std_max: 0.01`
- `mar_std_max: 0.02`
- `pose_std_max: 0.08`
- `frame_count: 100`

If calibration times out, the kiosk falls back safely and continues into inference instead of freezing the display.

## Alert Conditions

Default trigger values are configured in `hybrid_system/configs/default.yaml`.

- `drowsy`: LSTM drowsiness classification held for `36` frames.
- `yawning`: calibrated MAR threshold held for `40` frames.
- `distracted`: calibrated head-pose deviation held for `12` frames.
- `phone_use`: phone detection held after object detector confirmation.
- `face_lost`: no visible face for `50` consecutive frames.
- `eyes_closed`: kept as a visual/internal signal and does not directly trigger TTS.

Voice alert priority:

```text
phone_use > face_lost > drowsy > distracted > yawning
```

## Display

The kiosk UI targets an 800x480 screen. Camera frames are scaled to cover the screen and center-cropped if the camera aspect ratio does not match 800x480. Landmarks use the same transform as the displayed frame so overlays stay aligned.

`run_kiosk.py` starts fullscreen by default. Use `--windowed` only for desktop debugging when you want the normal window frame and title bar.

Available display backends depend on installed packages and config. The default is `auto`, which tries SDL2 fullscreen first and falls back only if needed:

```bash
python3 run_kiosk.py --display-backend auto
python3 run_kiosk.py --display-backend opencv
```

`ffplay` may exist as an optional video display backend only. It is not used for audio.

## Raspberry Pi Performance Benchmark

The production default limits OpenCV, PyTorch, OpenMP, and BLAS to one thread per library. This avoids nested thread pools oversubscribing the four Pi 4 cores. Override it only after measuring:

```bash
python3 run_kiosk.py --threads 1
```

Video recording is disabled by default. `--write-video` enables it through the latest-frame async output worker, so encoding does not block the inference loop.

Run the five-stage benchmark matrix on the Pi with a representative recording:

```bash
python3 benchmarks/benchmark_pi_pipeline.py \
  --source kiosk_recordings/sample.mp4 \
  --frames 300 \
  --threads 1
```

To include physical display latency, add `--display`. Results are written to `benchmark_results/pi_pipeline_*.json` with camera, MediaPipe, feature extraction, head-pose, LSTM, YOLO, render/UI, display, video writer, CPU, RAM, load, and temperature measurements.

Benchmark YOLO input sizes against the current 640 output:

```bash
python3 benchmarks/benchmark_yolo_phone.py \
  --source phone_use_validation.mp4 \
  --frames 100 \
  --threads 1 \
  --sizes 320,416,640 \
  --include-ultralytics
```

The bundled ONNX model has a fixed 640 input. The benchmark marks incompatible 320/416 attempts as failed instead of silently accepting empty detections. Do not change the production input size until a correctly exported model passes the phone-presence and bounding-box consistency gate on phone-positive footage.

Optionally compare OpenCV DNN with ONNX Runtime using exactly the same input tensors and verify raw output consistency:

```bash
pip install onnxruntime
python3 benchmarks/benchmark_onnx_runtimes.py \
  --source phone_use_validation.mp4 \
  --frames 50 \
  --threads 1
```

ONNX Runtime is deliberately not a production dependency. NCNN, TFLite, and INT8 are not selected because the repository currently has no equivalent exported/quantized model and validation artifact for them.

## Raspberry Pi Hardware

Default GPIO mapping:

- Calibration LED: BCM `2`, physical pin `3`.
- Inference LED: BCM `3`, physical pin `5`.
- Restart calibration button: BCM `14`, physical pin `8`.
- Pause/resume button: BCM `15`, physical pin `10`.

Recommended wiring:

- Use a `220-330 ohm` resistor for each LED.
- Buttons connect GPIO to GND and use internal pull-up.

Notes:

- GPIO14/GPIO15 are UART pins by default. Disable serial console if using them as buttons.
- GPIO2/GPIO3 are I2C pins by default. Use different pins if I2C is needed.

## Systemd Autostart

The repository includes `drowsiness-monitor.service.example`.

Example commands:

```bash
sudo cp drowsiness-monitor.service.example /etc/systemd/system/drowsiness-monitor.service
sudo systemctl daemon-reload
sudo systemctl enable drowsiness-monitor.service
sudo systemctl start drowsiness-monitor.service
sudo systemctl status drowsiness-monitor.service
```

## Troubleshooting

Camera not opening:

```bash
ls -l /dev/video*
v4l2-ctl --list-devices
python3 run_kiosk.py --camera-index 0 --camera-fourcc MJPG
```

Audio not playing:

```bash
aplay -l
speaker-test -t sine -f 1000 -c 2
.venv/bin/python tools/play_audio_sample.py "Hệ thống đã sẵn sàng.wav"
```

OpenCV Qt font warnings such as `QFontDatabase: Cannot find font directory` are usually harmless. They come from OpenCV's Qt window backend and do not indicate a camera or detection failure.

## Tests

Run focused checks:

```bash
SDL_AUDIODRIVER=dummy .venv/bin/python -m unittest tests.test_hardware_audio
.venv/bin/python -m py_compile run_kiosk.py hybrid_system/config.py hybrid_system/kiosk.py hybrid_system/ui.py
```

Run benchmarks:

```bash
.venv/bin/python bench_pipeline.py
.venv/bin/python bench_async_output.py
```

## Acknowledgements

This project uses MediaPipe for face landmarks, Ultralytics/YOLO-style object detection for phone use, and a head-pose estimator based on the landmark ordering used by Mostafa Nafie's Head-Pose-Estimation project.
