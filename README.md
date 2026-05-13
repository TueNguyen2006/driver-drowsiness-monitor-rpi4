# Driver Drowsiness Monitor for Raspberry Pi 4

Hệ thống phát hiện buồn ngủ theo thời gian thực bằng camera USB, chạy trên Raspberry Pi 4.

Tính năng chính:
- Hiệu chuẩn khuôn mặt trước khi chạy suy luận.
- Phát hiện trạng thái `normal` hoặc `drowsy` từ đặc trưng mắt, miệng và góc đầu.
- Điều khiển bằng 2 nút GPIO:
  - Nút `restart calibration`
  - Nút `pause/resume inference`
- Cảnh báo phần cứng:
  - Đèn báo trạng thái
  - Thông báo bằng giọng nói tiếng Việt
  - Còi cảnh báo khi phát hiện buồn ngủ

## 1. Phần cứng

Kết nối đang dùng:
- LED calibration: chân vật lý `3`, GPIO BCM `2`
- LED inference: chân vật lý `5`, GPIO BCM `3`
- Nút restart: chân vật lý `8`, GPIO BCM `14`
- Nút pause/resume: chân vật lý `10`, GPIO BCM `15`
- Camera: USB
- Loa / mạch âm thanh: jack `3.5mm`

Khuyến nghị đấu dây:
- Mỗi LED đi qua điện trở `220-330 ohm`.
- Nút nối từ GPIO xuống `GND`.
- Code dùng `pull-up` nội bộ, nên trạng thái nhấn là mức `LOW`.

Lưu ý:
- `GPIO14` và `GPIO15` mặc định liên quan đến UART. Nếu dùng làm nút, nên tắt serial console trên Raspberry Pi OS.
- `GPIO2` và `GPIO3` mặc định liên quan đến I2C. Nếu bạn không dùng I2C thì có thể dùng làm LED như hiện tại.

## 2. Yêu cầu hệ điều hành

Khuyến nghị:
- Raspberry Pi OS 64-bit
- Python `3.10` hoặc `3.11`

Quan trọng:
- `mediapipe` và `torch` trên Raspberry Pi phụ thuộc vào wheel tương thích với hệ điều hành và phiên bản Python.
- Nên dùng bản Raspberry Pi OS 64-bit để giảm rủi ro lỗi cài đặt.

## 3. Clone và setup

```bash
git clone <YOUR_REPO_URL>
cd New
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
sudo apt update
sudo apt install -y espeak-ng alsa-utils libatlas-base-dev
pip install -r requirements.txt
```

Hoặc chạy script cài nhanh:

```bash
chmod +x setup_pi.sh
./setup_pi.sh
```

Nếu `pip install -r requirements.txt` lỗi ở `torch` hoặc `mediapipe`:
- Giữ nguyên các package còn lại.
- Cài đúng wheel tương thích với Pi OS và Python đang dùng.
- Sau đó chạy lại chương trình.

## 4. File quan trọng

- `main.py`: luồng chính realtime
- `hardware.py`: GPIO đèn, nút, thoại, còi
- `config.py`: cấu hình camera, GPIO, âm thanh
- `models/`: model head pose và model drowsiness
- `Arduino/`: phần cũ dùng Arduino, không còn bắt buộc khi chạy trên Pi

## 5. Chạy chương trình

```bash
source .venv/bin/activate
python main.py
```

Khi chạy:
- `calibrate`: LED GPIO `2` sáng
- `infer`: LED GPIO `3` sáng
- `infer_paused`: cả 2 LED cùng sáng

Âm thanh:
- Bấm nút chạy lại calibration: "Chạy quá trình hiệu chuẩn"
- Hoàn tất calibration: "Hoàn thành quá trình hiệu chuẩn"
- Bắt đầu infer: "Chúc bạn có một chuyến đi vui vẻ"
- Khi phát hiện buồn ngủ ổn định: phát còi cảnh báo

## 6. Cấu hình nhanh bằng biến môi trường

Bạn có thể override cấu hình mặc định:

```bash
export CAMERA_INDEX=0
export LED_CALIBRATION_PIN=2
export LED_INFERENCE_PIN=3
export BUTTON_RESTART_PIN=14
export BUTTON_PAUSE_PIN=15
export TTS_VOICE=vi
export TTS_RATE=150
```

Một số biến khác:
- `GPIO_DEBOUNCE_MS`
- `ALERT_BEEP_FREQUENCY`
- `ALERT_BEEP_DURATION`
- `ALERT_BEEP_INTERVAL`
- `ALERT_HOLD_SECONDS`
- `ALERT_STABLE_SECONDS`

## 7. Tắt serial console để dùng GPIO14/GPIO15

Chạy:

```bash
sudo raspi-config
```

Vào:
- `Interface Options`
- `Serial Port`
- Chọn `No` cho login shell over serial
- Chọn `No` hoặc `Yes` cho serial hardware tùy bạn có còn cần UART hay không

Sau đó reboot:

```bash
sudo reboot
```

## 8. Tự chạy cùng hệ thống sau khi boot

Repo đã có sẵn file mẫu `drowsiness-monitor.service.example`.

Bạn có thể dùng `systemd`. Ví dụ service:

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

Lưu vào:

```bash
sudo nano /etc/systemd/system/drowsiness-monitor.service
```

Kích hoạt:

```bash
sudo systemctl daemon-reload
sudo systemctl enable drowsiness-monitor.service
sudo systemctl start drowsiness-monitor.service
sudo systemctl status drowsiness-monitor.service
```

## 9. Ghi chú triển khai

- Camera USB phải được hệ điều hành nhận là `/dev/video0` hoặc camera index phù hợp.
- Nếu loa 3.5mm không phát tiếng, kiểm tra output audio bằng:

```bash
aplay -l
speaker-test -t sine -f 1000 -c 2
```

- Nếu cần chọn output 3.5mm:

```bash
amixer cset numid=3 1
```

## 10. Hướng phát triển tiếp

- Thêm `systemd` file trực tiếp vào repo.
- Ghi log ra file thay vì chỉ hiển thị OpenCV window.
- Thêm watchdog cho camera/audio để tự phục hồi nếu thiết bị bị rút ra.
