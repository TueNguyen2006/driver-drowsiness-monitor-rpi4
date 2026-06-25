# Driver Drowsiness Monitor for Raspberry Pi 4

Real-time drowsiness detection system using USB camera, running on Raspberry Pi 4.

This project implements a comprehensive driver drowsiness monitoring system that can detect when drivers are becoming drowsy or distracted based on facial features, head pose and mouth movements. The system includes hardware integration with LEDs, buttons and audio alerts for real-time feedback.

## Key Features

- Face calibration before inference
- Detection of `normal` or `drowsy` states based on eye, mouth, and head angle features  
- Hardware control with 2 GPIO buttons:
  - Button `restart calibration`
  - Button `pause/resume inference`
- Hardware alerts:
  - Status LEDs
  - Voice notifications in Vietnamese
  - Alarm buzzer when drowsiness is detected

## Hardware Connections

Current setup uses:
- Calibration LED: physical pin `3`, GPIO BCM `2`
- Inference LED: physical pin `5`, GPIO BCM `3` 
- Restart button: physical pin `8`, GPIO BCM `14`
- Pause/Resume button: physical pin `10`, GPIO BCM `15`
- Camera: USB
- Speaker/Audio: 3.5mm jack

Recommended wiring:
- Each LED through a `220-330 ohm` resistor
- Buttons connected to GPIO and GND (uses internal pull-up, so pressed state is LOW)

**Notes:**
- `GPIO14` and `GPIO15` are by default used for UART. If using these pins as buttons, disable serial console on Raspberry Pi OS.
- `GPIO2` and `GPIO3` are by default used for I2C. If not using I2C, you can use them as LEDs.

## System Requirements

Recommended:
- Raspberry Pi OS 64-bit
- Python `3.10` or `3.11`

Important:
- `mediapipe` and `torch` on Raspberry Pi depend on wheels compatible with the OS and Python version.
- Use Raspberry Pi OS 64-bit to reduce installation issues.

## Clone and Setup

```bash
git clone https://github.com/TueNguyen2006/driver-drowsiness-monitor-rpi4.git
cd driver-drowsiness-monitor-rpi4
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
sudo apt update
sudo apt install -y espeak-ng alsa-utils libatlas-base-dev
pip install -r requirements.txt
```

Or run the quick setup script:

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

If `pip install -r requirements.txt` fails with `torch` or `mediapipe`:
- Keep other packages as they are
- Install the correct wheel compatible with Pi OS and Python version 
- Then run the program again

## Key Files

- `main.py`: main real-time pipeline
- `hardware.py`: GPIO LEDs, buttons, speaker, buzzer controls  
- `config.py`: camera, GPIO, audio configurations
- `models/`: head pose and drowsiness detection models
- `Arduino/`: legacy Arduino code (no longer required when running on Pi)

## Running the Program

```bash
source .venv/bin/activate
python main.py
```

When running:
- `calibrate`: GPIO LED `2` turns on
- `infer`: GPIO LED `3` turns on  
- `infer_paused`: both LEDs turn on

Audio feedback:
- Press restart button: "Running calibration process"
- Calibration complete: "Calibration process completed" 
- Start inference: "Have a safe journey"
- When drowsiness is stable: audio alarm will trigger

## Quick Configuration with Environment Variables

You can override default configurations:

```bash
export CAMERA_INDEX=0
export LED_CALIBRATION_PIN=2
export LED_INFERENCE_PIN=3
export BUTTON_RESTART_PIN=14
export BUTTON_PAUSE_PIN=15
export TTS_VOICE=vi
export TTS_RATE=150
```

Other available variables:
- `GPIO_DEBOUNCE_MS`
- `ALERT_BEEP_FREQUENCY`
- `ALERT_BEEP_DURATION` 
- `ALERT_BEEP_INTERVAL`
- `ALERT_HOLD_SECONDS`
- `ALERT_STABLE_SECONDS`

## Disable Serial Console to Use GPIO14/GPIO15

Run:

```bash
sudo raspi-config
```

Navigate to:
- `Interface Options` → `Serial Port`  
- Choose `No` for login shell over serial
- Choose `No` or `Yes` for serial hardware depending on your needs

Then reboot:

```bash
sudo reboot
```

## Auto-start After Boot

The repo includes a sample service file `drowsiness-monitor.service.example`.

You can use systemd. Example service configuration:

```ini
[Unit]
Description=Drowsiness Monitor
After=network.target sound.target

[Service]
User=pi
WorkingDirectory=/home/pi/New
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/pi/New/.venv/bin/python /home/pi/New/main.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
```

Save to:
```bash
sudo nano /etc/systemd/system/drowsiness-monitor.service
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable drowsiness-monitor.service
sudo systemctl start drowsiness-monitor.service
sudo systemctl status drowsiness-monitor.service
```

## Deployment Notes

- USB camera must be recognized by the OS as `/dev/video0` or appropriate camera index.
- If 3.5mm speaker doesn't play audio, check audio output:
```bash
aplay -l
speaker-test -t sine -f 1000 -c 2
```

To select 3.5mm output:
```bash
amixer cset numid=3 1
```

## Future Development

- Add systemd files directly to repository 
- Log to file instead of just displaying OpenCV window
- Add watchdog for camera/audio devices to auto-recover when devices are unplugged

## Acknowledgements

This system uses MediaPipe for face detection and landmark estimation, along with custom deep learning models for drowsiness detection based on head pose analysis.
