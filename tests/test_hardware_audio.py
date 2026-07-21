from __future__ import annotations

import os
import unittest
import wave
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from hybrid_system.config import load_config
from hybrid_system.hardware_controller import HardwareController


class HardwareAudioTests(unittest.TestCase):
    def tearDown(self) -> None:
        # pygame.mixer is process-global; ensure one test cannot leak audio state
        # into the next one.
        try:
            import pygame

            pygame.mixer.quit()
        except Exception:
            pass

    def test_pygame_mixer_preloads_repo_audio(self) -> None:
        controller = HardwareController(load_config())
        controller._init_audio()
        try:
            self.assertTrue(controller._mixer_ready)
            self.assertEqual(controller._pygame.mixer.get_init(), (48000, -16, 2))
            self.assertIn("Hệ thống đã sẵn sàng.wav", controller._sound_cache)
            self.assertIn("Hệ thống đã sẵn sàng", controller._sound_cache)
        finally:
            controller.cleanup()

    def test_voice_queue_cooldown_dedupes_same_sentence(self) -> None:
        controller = HardwareController(load_config())
        controller.queue_speech("Hệ thống đã sẵn sàng", cooldown=5.0)
        controller.queue_speech("Hệ thống đã sẵn sàng", cooldown=5.0)
        try:
            self.assertEqual(controller.audio_queue.qsize(), 1)
        finally:
            controller.cleanup()

    def test_missing_audio_logs_warning_without_raising(self) -> None:
        controller = HardwareController(load_config())
        controller._init_audio()
        try:
            with self.assertLogs("hybrid_system.hardware_controller", level="WARNING") as logs:
                self.assertIsNone(controller._sound_for_file("missing-test-audio.wav"))
            self.assertIn("Audio file missing", "\n".join(logs.output))
        finally:
            controller.cleanup()

    def test_legacy_controller_uses_repo_audio_cache(self) -> None:
        from hardware import HardwareController as LegacyHardwareController

        controller = LegacyHardwareController()
        controller._init_audio()
        try:
            self.assertTrue(controller._mixer_ready)
            self.assertEqual(controller._pygame.mixer.get_init(), (48000, -16, 2))
            self.assertIn("Hệ thống đã sẵn sàng.wav", controller._sound_cache)
        finally:
            controller.cleanup()

    def test_repo_wav_assets_match_mixer_format(self) -> None:
        audio_dir = Path("assets/audio")
        wavs = sorted(audio_dir.glob("*.wav"))
        self.assertTrue(wavs, "Expected WAV files under assets/audio")
        for wav_path in wavs:
            with self.subTest(wav=wav_path.name):
                with wave.open(str(wav_path), "rb") as wav:
                    self.assertEqual(wav.getframerate(), 48000)
                    self.assertEqual(wav.getsampwidth(), 2)
                    self.assertEqual(wav.getnchannels(), 2)


if __name__ == "__main__":
    unittest.main()
