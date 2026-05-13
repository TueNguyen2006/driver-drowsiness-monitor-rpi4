import math
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import wave

import config as cfg
import state

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


class HardwareController:
    def __init__(self):
        self.gpio_available = GPIO is not None
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.status_thread = None
        self.button_thread = None
        self.audio_lock = threading.Lock()
        self.alert_beep_path = None
        self.last_restart_press = 0.0
        self.last_pause_press = 0.0
        self.last_announced_text = {}
        self._speech_cmd = self._resolve_speech_command()
        self._audio_player_cmd = self._resolve_audio_player()

    def start(self):
        if self.gpio_available:
            self._setup_gpio()

        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()

        self.status_thread = threading.Thread(target=self._status_worker, daemon=True)
        self.status_thread.start()

        if self.gpio_available:
            self.button_thread = threading.Thread(target=self._button_worker, daemon=True)
            self.button_thread.start()

    def cleanup(self):
        self.stop_event.set()

        if self.gpio_available:
            try:
                GPIO.output(cfg.LED_CALIBRATION_PIN, GPIO.LOW)
                GPIO.output(cfg.LED_INFERENCE_PIN, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass

        if self.alert_beep_path and os.path.exists(self.alert_beep_path):
            try:
                os.remove(self.alert_beep_path)
            except OSError:
                pass

    def announce_recalibration_requested(self):
        self.speak_async("Chạy quá trình hiệu chuẩn")

    def announce_calibration_complete(self):
        self.speak_async("Hoàn thành quá trình hiệu chuẩn")

    def announce_infer_started(self):
        self.speak_async("Chúc bạn có một chuyến đi vui vẻ")

    def speak_async(self, text, dedupe_window=0.0):
        now = time.monotonic()
        last_spoken = self.last_announced_text.get(text, 0.0)
        if dedupe_window and now - last_spoken < dedupe_window:
            return

        self.last_announced_text[text] = now
        self.audio_queue.put(("speak", text))

    def queue_alert_beep(self):
        try:
            self.audio_queue.put_nowait(("beep", None))
        except queue.Full:
            pass

    def _setup_gpio(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(cfg.LED_CALIBRATION_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(cfg.LED_INFERENCE_PIN, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(cfg.BUTTON_RESTART_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(cfg.BUTTON_PAUSE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _button_worker(self):
        last_restart_state = GPIO.input(cfg.BUTTON_RESTART_PIN)
        last_pause_state = GPIO.input(cfg.BUTTON_PAUSE_PIN)
        debounce_s = cfg.GPIO_DEBOUNCE_MS / 1000.0

        while not self.stop_event.is_set():
            now = time.monotonic()

            restart_state = GPIO.input(cfg.BUTTON_RESTART_PIN)
            if restart_state == GPIO.LOW and last_restart_state == GPIO.HIGH:
                if now - self.last_restart_press >= debounce_s:
                    self._handle_button_command("RESTART_INFERENCE")
                    self.last_restart_press = now
            last_restart_state = restart_state

            pause_state = GPIO.input(cfg.BUTTON_PAUSE_PIN)
            if pause_state == GPIO.LOW and last_pause_state == GPIO.HIGH:
                if now - self.last_pause_press >= debounce_s:
                    self._handle_button_command("TOGGLE_INFER_PAUSED")
                    self.last_pause_press = now
            last_pause_state = pause_state

            time.sleep(0.02)

    def _handle_button_command(self, command):
        if command == "RESTART_INFERENCE":
            if state.mode in {"infer", "infer_paused"}:
                self.announce_recalibration_requested()
                state.request_recalibrate = True
            return

        if command == "TOGGLE_INFER_PAUSED":
            if state.mode in {"infer", "infer_paused"}:
                state.running_inference = not state.running_inference

    def _status_worker(self):
        next_alert_beep_at = 0.0

        while not self.stop_event.is_set():
            self._sync_leds()

            if state.alert:
                now = time.monotonic()
                if now >= next_alert_beep_at:
                    self.queue_alert_beep()
                    next_alert_beep_at = now + cfg.ALERT_BEEP_INTERVAL
            else:
                next_alert_beep_at = 0.0

            time.sleep(0.05)

    def _sync_leds(self):
        calibration_led_on = state.mode in {"calibrate", "infer_paused"}
        infer_led_on = state.mode in {"infer", "infer_paused"}

        if not self.gpio_available:
            return

        GPIO.output(cfg.LED_CALIBRATION_PIN, GPIO.HIGH if calibration_led_on else GPIO.LOW)
        GPIO.output(cfg.LED_INFERENCE_PIN, GPIO.HIGH if infer_led_on else GPIO.LOW)

    def _audio_worker(self):
        while not self.stop_event.is_set():
            try:
                event_type, payload = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if event_type == "speak":
                    self._speak_blocking(payload)
                elif event_type == "beep":
                    self._play_alert_beep()
            finally:
                self.audio_queue.task_done()

    def _resolve_speech_command(self):
        for command in ("espeak-ng", "espeak"):
            command_path = shutil.which(command)
            if command_path:
                return command_path
        return None

    def _resolve_audio_player(self):
        for command in ("aplay", "paplay", "ffplay"):
            command_path = shutil.which(command)
            if command_path:
                return command_path
        return None

    def _speak_blocking(self, text):
        if not self._speech_cmd:
            return

        command = [self._speech_cmd, "-s", str(cfg.TTS_RATE)]
        if cfg.TTS_VOICE:
            command.extend(["-v", cfg.TTS_VOICE])
        command.append(text)

        with self.audio_lock:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _play_alert_beep(self):
        if not self._audio_player_cmd:
            return

        beep_path = self._ensure_alert_beep_file()
        if beep_path is None:
            return

        player_name = os.path.basename(self._audio_player_cmd)
        if player_name == "aplay":
            command = [self._audio_player_cmd, "-q", beep_path]
        elif player_name == "paplay":
            command = [self._audio_player_cmd, beep_path]
        else:
            command = [
                self._audio_player_cmd,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                beep_path,
            ]

        with self.audio_lock:
            subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _ensure_alert_beep_file(self):
        if self.alert_beep_path and os.path.exists(self.alert_beep_path):
            return self.alert_beep_path

        sample_rate = 22050
        amplitude = 16000
        beep_duration = max(0.1, cfg.ALERT_BEEP_DURATION / 2.0)
        silence_duration = 0.08
        sequence = [(cfg.ALERT_BEEP_FREQUENCY, beep_duration), (0, silence_duration), (cfg.ALERT_BEEP_FREQUENCY, beep_duration)]

        fd, path = tempfile.mkstemp(prefix="drowsy_alert_", suffix=".wav")
        os.close(fd)

        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            for frequency, duration in sequence:
                frame_count = int(sample_rate * duration)
                for frame_index in range(frame_count):
                    if frequency == 0:
                        sample = 0
                    else:
                        sample = int(
                            amplitude
                            * math.sin(2.0 * math.pi * frequency * frame_index / sample_rate)
                        )
                    wav_file.writeframesraw(struct.pack("<h", sample))

        self.alert_beep_path = path
        return self.alert_beep_path
