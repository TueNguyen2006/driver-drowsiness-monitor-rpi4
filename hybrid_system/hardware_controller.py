from __future__ import annotations

import math
import os
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

GPIO = None

if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
    except (ImportError, RuntimeError):
        GPIO = None


class HardwareController:
    __slots__ = (
        "gpio_available", "stop_event", "audio_queue", "audio_thread",
        "status_thread", "button_thread", "audio_lock", "alert_beep_path",
        "last_restart_press", "last_pause_press", "last_announced_text",
        "_speech_cmd", "_audio_player_cmd", "cfg",
        "_cal_led_on", "_inf_led_on", "_alert_active",
        "_on_restart", "_on_pause_toggle",
    )

    def __init__(self, config) -> None:
        self.cfg = config.hardware
        self.gpio_available = GPIO is not None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue = queue.Queue()
        self.audio_thread: threading.Thread | None = None
        self.status_thread: threading.Thread | None = None
        self.button_thread: threading.Thread | None = None
        self.audio_lock = threading.Lock()
        self.alert_beep_path: str | None = None
        self.last_restart_press = 0.0
        self.last_pause_press = 0.0
        self.last_announced_text: dict[str, float] = {}
        self._speech_cmd = self._resolve_speech_command()
        self._audio_player_cmd = self._resolve_audio_player()
        self._cal_led_on = False
        self._inf_led_on = False
        self._alert_active = False
        self._on_restart = self._noop
        self._on_pause_toggle = self._noop

    def     start(self) -> None:
        if self.gpio_available:
            self._setup_gpio()
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        self.status_thread = threading.Thread(target=self._status_worker, daemon=True)
        self.status_thread.start()
        if self.gpio_available:
            self.button_thread = threading.Thread(target=self._button_worker, daemon=True)
            self.button_thread.start()

    def test_audio(self) -> None:
        """Play a short test tone to verify audio output."""
        self.audio_queue.put(("beep", None))

    def cleanup(self) -> None:
        self.stop_event.set()
        if self.gpio_available:
            try:
                GPIO.output(self.cfg.led_calibration_pin, GPIO.LOW)
                GPIO.output(self.cfg.led_inference_pin, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass
        if self.alert_beep_path and os.path.exists(self.alert_beep_path):
            try:
                os.remove(self.alert_beep_path)
            except OSError:
                pass

    def announce_calibration_start(self) -> None:
        self.speak_async("Đang hiệu chuẩn, vui lòng nhìn về phía trước")

    def announce_calibration_complete(self) -> None:
        self.speak_async("Hoàn thành hiệu chuẩn")

    def announce_infer_started(self) -> None:
        self.speak_async("Hệ thống đã sẵn sàng")

    def speak_async(self, text: str, dedupe_window: float = 0.0) -> None:
        now = time.monotonic()
        last_spoken = self.last_announced_text.get(text, 0.0)
        if dedupe_window and now - last_spoken < dedupe_window:
            return
        self.last_announced_text[text] = now
        self.audio_queue.put(("speak", text))

    def queue_alert_beep(self) -> None:
        try:
            self.audio_queue.put_nowait(("beep", None))
        except queue.Full:
            pass

    def _setup_gpio(self) -> None:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.cfg.led_calibration_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.cfg.led_inference_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.cfg.button_restart_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.cfg.button_pause_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def _button_worker(self) -> None:
        last_restart_state = GPIO.input(self.cfg.button_restart_pin)
        last_pause_state = GPIO.input(self.cfg.button_pause_pin)
        debounce_s = 0.25

        while not self.stop_event.is_set():
            now = time.monotonic()
            restart_state = GPIO.input(self.cfg.button_restart_pin)
            if restart_state == GPIO.LOW and last_restart_state == GPIO.HIGH:
                if now - self.last_restart_press >= debounce_s:
                    self.last_restart_press = now
                    self._on_restart()
            last_restart_state = restart_state

            pause_state = GPIO.input(self.cfg.button_pause_pin)
            if pause_state == GPIO.LOW and last_pause_state == GPIO.HIGH:
                if now - self.last_pause_press >= debounce_s:
                    self.last_pause_press = now
                    self._on_pause_toggle()
            last_pause_state = pause_state
            time.sleep(0.02)

    @staticmethod
    def _noop() -> None:
        pass

    def set_restart_callback(self, cb: callable) -> None:
        self._on_restart = cb

    def set_pause_callback(self, cb: callable) -> None:
        self._on_pause_toggle = cb

    def _status_worker(self) -> None:
        next_alert_beep_at = 0.0
        led_cal = self.cfg.led_calibration_pin
        led_inf = self.cfg.led_inference_pin
        while not self.stop_event.is_set():
            now = time.monotonic()
            cal_on = self._cal_led_on
            inf_on = self._inf_led_on
            if self.gpio_available:
                GPIO.output(led_cal, GPIO.HIGH if cal_on else GPIO.LOW)
                GPIO.output(led_inf, GPIO.HIGH if inf_on else GPIO.LOW)
            if self._alert_active:
                if now >= next_alert_beep_at:
                    self.queue_alert_beep()
                    next_alert_beep_at = now + self.cfg.beep_interval
            else:
                next_alert_beep_at = 0.0
            time.sleep(0.05)

    def set_leds(self, calibration: bool, inference: bool) -> None:
        self._cal_led_on = calibration
        self._inf_led_on = inference

    def set_alert(self, active: bool) -> None:
        self._alert_active = active

    def _audio_worker(self) -> None:
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

    def _resolve_speech_command(self) -> tuple[str, bool] | None:
        for cmd in ("espeak-ng", "espeak"):
            path = shutil.which(cmd)
            if path:
                out = subprocess.run([path, "--voices"], capture_output=True, text=True, timeout=5)
                has_vi = "vi" in out.stdout or "Vietnamese" in out.stdout
                return (path, has_vi)
        for cmd in self._windows_powershell_candidates() + ("pwsh", "powershell"):
            path = shutil.which(cmd)
            if path:
                return (path, False)
        return None

    def _resolve_audio_player(self) -> str | None:
        for cmd in ("ffplay", "paplay", "aplay"):
            path = shutil.which(cmd)
            if path:
                return path
        return None

    def _speak_blocking(self, text: str) -> None:
        if self._speech_cmd is None:
            return
        speech_path = self._speech_cmd[0]
        has_vi = len(self._speech_cmd) > 1 and bool(self._speech_cmd[1])
        backend = "powershell" if os.path.basename(speech_path).lower().startswith(("powershell", "pwsh")) else "espeak"
        voice = "vi" if has_vi else "en"
        alsa_dev = getattr(self.cfg, "alsa_device", "")
        with self.audio_lock:
            if backend == "powershell":
                self._powershell_speak(speech_path, text)
            elif self._audio_player_cmd and os.path.basename(self._audio_player_cmd) == "aplay":
                dev_flag = ["-D", alsa_dev] if alsa_dev else []
                speak = subprocess.Popen(
                    [speech_path, "-s", "150", "-a", "200", "-v", voice, "--stdout", text],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [self._audio_player_cmd, "-q", *dev_flag],
                    stdin=speak.stdout, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if speak.stdout:
                    speak.stdout.close()
                speak.wait()
            elif self._audio_player_cmd and os.path.basename(self._audio_player_cmd) == "ffplay":
                speak = subprocess.Popen(
                    [speech_path, "-s", "150", "-a", "200", "-v", voice, "--stdout", text],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                )
                subprocess.run(
                    [self._audio_player_cmd, "-nodisp", "-autoexit", "-loglevel", "quiet", "-i", "pipe:0"],
                    stdin=speak.stdout, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                if speak.stdout:
                    speak.stdout.close()
                speak.wait()
            else:
                subprocess.run(
                    [speech_path, "-s", "150", "-a", "200", "-v", voice, text],
                    check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

    def _play_alert_beep(self) -> None:
        if self._audio_player_cmd is None:
            self._play_alert_beep_windows_fallback()
            return
        beep_path = self._ensure_alert_beep_file()
        if beep_path is None:
            return
        player = os.path.basename(self._audio_player_cmd)
        alsa_dev = getattr(self.cfg, "alsa_device", "")
        with self.audio_lock:
            if player == "aplay":
                dev_flag = ["-D", alsa_dev] if alsa_dev else []
                subprocess.run(
                    [self._audio_player_cmd, "-q", *dev_flag, beep_path],
                    check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif player == "ffplay":
                subprocess.run(
                    [self._audio_player_cmd, "-nodisp", "-autoexit", "-loglevel", "quiet", beep_path],
                    check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif player == "paplay":
                subprocess.run(
                    [self._audio_player_cmd, beep_path],
                    check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(
                    [self._audio_player_cmd, "-nodisp", "-autoexit",
                     "-loglevel", "quiet", beep_path],
                    check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )

    def _play_alert_beep_windows_fallback(self, beep_path: str | None = None) -> None:
        ps = None
        for cmd in self._windows_powershell_candidates() + ("pwsh", "powershell"):
            ps = shutil.which(cmd)
            if ps:
                break
        if not ps:
            return
        if beep_path is None:
            beep_path = self._ensure_alert_beep_file()
        if beep_path is None:
            return
        try:
            win_path = subprocess.run(
                ["wslpath", "-w", beep_path],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            win_path = beep_path
        script = (
            f"$p = New-Object System.Media.SoundPlayer '{win_path}'; "
            f"$p.PlaySync();"
        )
        subprocess.run(
            [ps, "-NoProfile", "-Command", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    @staticmethod
    def _windows_powershell_candidates() -> tuple[str, ...]:
        candidates = ["powershell.exe"]
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        if windir:
            candidates.append(os.path.join(windir, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"))

        system_drive = os.environ.get("SystemDrive", "C:").rstrip(":\\/")
        drive_mount = os.path.join(os.sep, "mnt", system_drive.lower())
        candidates.append(os.path.join(drive_mount, "Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe"))
        return tuple(dict.fromkeys(candidates))

    def _powershell_speak(self, ps_path: str, text: str) -> None:
        safe_text = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{safe_text}');"
        )
        subprocess.run(
            [ps_path, "-NoProfile", "-Command", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _ensure_alert_beep_file(self) -> str | None:
        if self.alert_beep_path and os.path.exists(self.alert_beep_path):
            return self.alert_beep_path
        sample_rate = 22050
        amplitude = 24000
        beep_dur = max(0.1, self.cfg.beep_duration / 2.0)
        silence = 0.08
        freq = self.cfg.beep_frequency
        seq = [(freq, beep_dur), (0, silence), (freq, beep_dur)]
        fd, path = tempfile.mkstemp(prefix="drowsy_alert_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for freq_hz, dur in seq:
                frames = int(sample_rate * dur)
                for i in range(frames):
                    if freq_hz == 0:
                        s = 0
                    else:
                        s = int(amplitude * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
                    wf.writeframesraw(struct.pack("<h", s))
        self.alert_beep_path = path
        return path
