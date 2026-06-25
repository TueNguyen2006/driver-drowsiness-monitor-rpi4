import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_HEAD_POSE = os.path.join(BASE_DIR, "models", "model.pkl")
MODEL_FATIGUE = os.path.join(BASE_DIR, "models", "clf_lstm.pth")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

LED_CALIBRATION_PIN = int(os.getenv("LED_CALIBRATION_PIN", "2"))
LED_INFERENCE_PIN = int(os.getenv("LED_INFERENCE_PIN", "3"))
BUTTON_RESTART_PIN = int(os.getenv("BUTTON_RESTART_PIN", "14"))
BUTTON_PAUSE_PIN = int(os.getenv("BUTTON_PAUSE_PIN", "15"))
GPIO_DEBOUNCE_MS = int(os.getenv("GPIO_DEBOUNCE_MS", "250"))

TTS_VOICE = os.getenv("TTS_VOICE", "vi")
TTS_RATE = int(os.getenv("TTS_RATE", "150"))
ALERT_BEEP_FREQUENCY = int(os.getenv("ALERT_BEEP_FREQUENCY", "1850"))
ALERT_BEEP_DURATION = float(os.getenv("ALERT_BEEP_DURATION", "0.45"))
ALERT_BEEP_INTERVAL = float(os.getenv("ALERT_BEEP_INTERVAL", "1.75"))
ALERT_HOLD_SECONDS = float(os.getenv("ALERT_HOLD_SECONDS", "2.5"))
ALERT_STABLE_SECONDS = float(os.getenv("ALERT_STABLE_SECONDS", "1.2"))

ARDUINO_PORT = os.getenv("ARDUINO_PORT")
ARDUINO_BAUDRATE = int(os.getenv("ARDUINO_BAUDRATE", "9600"))
ARDUINO_RETRY_DELAY = float(os.getenv("ARDUINO_RETRY_DELAY", "2"))
