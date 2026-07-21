from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import config as cfg
import state

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    GPIO = None


log = logging.getLogger(__name__)


class HardwareController:
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

    def __init__(self):
        self.gpio_available = GPIO is not None
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.status_thread = None
        self.button_thread = None
        self.last_restart_press = 0.0
        self.last_pause_press = 0.0
        self.last_announced_text = {}
        self.audio_dir = Path(__file__).resolve().parent / "assets" / "audio"
        self._pygame: Any | None = None
        self._mixer_ready = False
        self._sound_cache: dict[str, Any] = {}
        self._missing_audio: set[str] = set()
        self._voice_channel: Any | None = None
        self._alert_channel: Any | None = None

    def start(self):
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

    def cleanup(self):
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
                GPIO.output(cfg.LED_CALIBRATION_PIN, GPIO.LOW)
                GPIO.output(cfg.LED_INFERENCE_PIN, GPIO.LOW)
                GPIO.cleanup()
            except Exception:
                pass

        self._shutdown_audio()

    def announce_recalibration_requested(self):
        self.speak_async("Vui lòng nhìn về phía trước")

    def announce_calibration_complete(self):
        self.speak_async("Hệ thống đã sẵn sàng")

    def announce_infer_started(self):
        self.speak_async("Chúc bạn có một chuyến đi vui vẻ")

    def speak(self, text, cooldown=0.0):
        self.queue_speech(text, cooldown=cooldown)

    def speak_async(self, text, dedupe_window=0.0):
        self.queue_speech(text, cooldown=dedupe_window)

    def queue_speech(self, text, cooldown=0.0):
        text = (text or "").strip()
        if not text:
            return

        now = time.monotonic()
        last_spoken = self.last_announced_text.get(text, 0.0)
        if cooldown and now - last_spoken < cooldown:
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

    def _init_audio(self):
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

    def _preload_audio(self):
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

    def _audio_worker(self):
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

    def _play_voice(self, text):
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

    def _play_alert_beep(self):
        sound = self._sound_for_file(self.ALERT_BEEP_FILE)
        if sound is None or self._alert_channel is None:
            return
        if not self._alert_channel.get_busy():
            self._alert_channel.set_volume(self.ALERT_VOLUME)
            self._alert_channel.play(sound)

    def _sound_for_text(self, text):
        return self._sound_for_file(f"{text}.wav")

    def _sound_for_file(self, filename):
        if not self._mixer_ready:
            return None
        sound = self._sound_cache.get(filename)
        if sound is not None:
            return sound
        if filename not in self._missing_audio:
            self._missing_audio.add(filename)
            log.warning("Audio file missing, skipping playback: %s", self.audio_dir / filename)
        return None

    def _shutdown_audio(self):
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
