from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from typing import Any

from .config import PROJECT_ROOT

GPIO = None

if sys.platform == "linux":
    try:
        import RPi.GPIO as GPIO
    except (ImportError, RuntimeError):
        GPIO = None


log = logging.getLogger(__name__)


class HardwareController:
    __slots__ = (
        "gpio_available", "stop_event", "audio_queue", "audio_thread",
        "status_thread", "button_thread", "cfg", "audio_dir",
        "last_restart_press", "last_pause_press", "last_announced_text",
        "_cal_led_on", "_inf_led_on", "_alert_active",
        "_on_restart", "_on_pause_toggle", "_pygame", "_mixer_ready",
        "_sound_cache", "_missing_audio", "_voice_channel", "_alert_channel",
    )

    VOICE_CHANNEL = 0
    ALERT_CHANNEL = 1
    ALERT_BEEP_FILE = "alert_beep.wav"
    MIXER_FREQUENCY = 48000
    MIXER_SIZE = -16
    MIXER_CHANNELS = 2
    MIXER_BUFFER = 16384
    PULSE_LATENCY_MSEC = "120"
    VOICE_VOLUME = 0.50
    ALERT_VOLUME = 0.75
    CALIBRATION_START_PROMPT = "Hãy thả lỏng, thư giãn khuôn mặt mặt để chuẩn bị hiệu chuẩn"
    MAX_AUDIO_BACKLOG = 2

    def __init__(self, config) -> None:
        self.cfg = config.hardware
        self.gpio_available = GPIO is not None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self.audio_thread: threading.Thread | None = None
        self.status_thread: threading.Thread | None = None
        self.button_thread: threading.Thread | None = None
        self.audio_dir = PROJECT_ROOT / "assets" / "audio"
        self.last_restart_press = 0.0
        self.last_pause_press = 0.0
        self.last_announced_text: dict[str, float] = {}
        self._cal_led_on = False
        self._inf_led_on = False
        self._alert_active = False
        self._on_restart = self._noop
        self._on_pause_toggle = self._noop
        self._pygame: Any | None = None
        self._mixer_ready = False
        self._sound_cache: dict[str, Any] = {}
        self._missing_audio: set[str] = set()
        self._voice_channel: Any | None = None
        self._alert_channel: Any | None = None

    def start(self) -> None:
        if self.gpio_available:
            self._setup_gpio()
        self._init_audio()
        self.audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        self.status_thread = threading.Thread(target=self._status_worker, daemon=True)
        self.status_thread.start()
        if self.gpio_available:
            self.button_thread = threading.Thread(target=self._button_worker, daemon=True)
            self.button_thread.start()

    def test_audio(self) -> None:
        """Play a short test sound to verify audio output."""
        self.queue_alert_beep()

    def cleanup(self) -> None:
        self.stop_event.set()
        try:
            self.audio_queue.put_nowait(("stop", None))
        except queue.Full:
            pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=1.0)
        if self.button_thread and self.button_thread.is_alive():
            self.button_thread.join(timeout=1.0)
        if self.gpio_available:
            try:
                GPIO.output(self.cfg.led_calibration_pin, GPIO.LOW)
                GPIO.output(self.cfg.led_inference_pin, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass
        self._shutdown_audio()

    def announce_calibration_start(self, wait: bool = False) -> None:
        if wait:
            self.speak_blocking(self.CALIBRATION_START_PROMPT)
        else:
            self.speak_async(self.CALIBRATION_START_PROMPT)

    def announce_calibration_complete(self) -> None:
        self.speak_async("Hệ thống đã sẵn sàng")

    def announce_infer_started(self) -> None:
        self.speak_async("Chúc bạn có một chuyến đi vui vẻ")

    def speak(self, text: str, cooldown: float = 0.0) -> None:
        self.queue_speech(text, cooldown=cooldown)

    def speak_async(self, text: str, dedupe_window: float = 0.0) -> None:
        self.queue_speech(text, cooldown=dedupe_window)

    def speak_blocking(self, text: str, cooldown: float = 0.0) -> None:
        queued = self.queue_speech(text, cooldown=cooldown)
        if queued:
            self.audio_queue.join()

    def queue_speech(self, text: str, cooldown: float = 0.0) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        now = time.monotonic()
        last_spoken = self.last_announced_text.get(text, 0.0)
        if cooldown and now - last_spoken < cooldown:
            return False
        if self.audio_queue.qsize() >= self.MAX_AUDIO_BACKLOG:
            log.debug("Skipping voice because audio queue is busy: %s", text)
            return False
        self.last_announced_text[text] = now
        self.audio_queue.put(("speak", text))
        return True

    def queue_alert_beep(self) -> None:
        if self.audio_queue.qsize() >= self.MAX_AUDIO_BACKLOG:
            return
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

    def _init_audio(self) -> None:
        if self._mixer_ready:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            if os.environ.get("PULSE_SERVER"):
                os.environ.setdefault("PULSE_LATENCY_MSEC", self.PULSE_LATENCY_MSEC)
            if os.environ.get("PULSE_SERVER") and not os.environ.get("SDL_AUDIODRIVER"):
                os.environ["SDL_AUDIODRIVER"] = "pulseaudio"
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.pre_init(
                    frequency=self.MIXER_FREQUENCY,
                    size=self.MIXER_SIZE,
                    channels=self.MIXER_CHANNELS,
                    buffer=self.MIXER_BUFFER,
                    allowedchanges=0,
                )
                pygame.mixer.init(
                    frequency=self.MIXER_FREQUENCY,
                    size=self.MIXER_SIZE,
                    channels=self.MIXER_CHANNELS,
                    buffer=self.MIXER_BUFFER,
                    allowedchanges=0,
                )
            mixer_spec = pygame.mixer.get_init()
            if mixer_spec != (self.MIXER_FREQUENCY, self.MIXER_SIZE, self.MIXER_CHANNELS):
                log.warning("Unexpected pygame mixer format: %s", mixer_spec)
            pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
            self._pygame = pygame
            self._voice_channel = pygame.mixer.Channel(self.VOICE_CHANNEL)
            self._alert_channel = pygame.mixer.Channel(self.ALERT_CHANNEL)
            self._mixer_ready = True
            self._preload_audio()
        except Exception as exc:
            self._mixer_ready = False
            log.warning("Audio disabled: pygame.mixer init failed: %s", exc)

    def _preload_audio(self) -> None:
        if not self._mixer_ready or self._pygame is None:
            return
        if not self.audio_dir.exists():
            log.warning("Audio directory not found: %s", self.audio_dir)
            return
        for path in sorted(self.audio_dir.glob("*.wav")):
            try:
                sound = self._pygame.mixer.Sound(str(path))
            except Exception as exc:
                log.warning("Cannot load audio file %s: %s", path, exc)
                continue
            self._sound_cache[path.name] = sound
            self._sound_cache[path.stem] = sound
        log.info("Preloaded %d audio files from %s", len(self._sound_cache) // 2, self.audio_dir)

    def _audio_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                event_type, payload = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if event_type == "stop":
                    return
                if event_type == "speak" and payload:
                    self._play_voice(payload)
                elif event_type == "beep":
                    self._play_alert_beep()
            finally:
                self.audio_queue.task_done()

    def _play_voice(self, text: str) -> None:
        sound = self._sound_for_text(text)
        if sound is None or self._voice_channel is None:
            return
        while self._voice_channel.get_busy() and not self.stop_event.is_set():
            time.sleep(0.02)
        if self.stop_event.is_set():
            return
        self._voice_channel.set_volume(self.VOICE_VOLUME)
        self._voice_channel.play(sound)
        while self._voice_channel.get_busy() and not self.stop_event.is_set():
            time.sleep(0.02)

    def _play_alert_beep(self) -> None:
        sound = self._sound_for_file(self.ALERT_BEEP_FILE)
        if sound is None or self._alert_channel is None:
            return
        if not self._alert_channel.get_busy():
            self._alert_channel.set_volume(self.ALERT_VOLUME)
            self._alert_channel.play(sound)

    def _sound_for_text(self, text: str) -> Any | None:
        filename = f"{text}.wav"
        return self._sound_for_file(filename)

    def _sound_for_file(self, filename: str) -> Any | None:
        if not self._mixer_ready:
            return None
        sound = self._sound_cache.get(filename)
        if sound is not None:
            return sound
        if filename not in self._missing_audio:
            self._missing_audio.add(filename)
            log.warning("Audio file missing, skipping playback: %s", self.audio_dir / filename)
        return None

    def _shutdown_audio(self) -> None:
        if self._pygame is None:
            return
        try:
            if self._voice_channel is not None:
                self._voice_channel.stop()
            if self._alert_channel is not None:
                self._alert_channel.stop()
            self._pygame.mixer.quit()
        except Exception:
            pass
        finally:
            self._mixer_ready = False
            self._voice_channel = None
            self._alert_channel = None
