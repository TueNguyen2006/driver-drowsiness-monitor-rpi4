from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Play one repo WAV through pygame.mixer")
    parser.add_argument(
        "name",
        nargs="?",
        default="Hệ thống đã sẵn sàng",
        help="Audio stem or file name under assets/audio",
    )
    parser.add_argument("--driver", default="", help="SDL_AUDIODRIVER override, e.g. pulseaudio")
    parser.add_argument("--mode", choices=["sound", "music"], default="sound")
    parser.add_argument("--volume", type=float, default=0.50)
    parser.add_argument("--buffer", type=int, default=16384)
    parser.add_argument("--latency-msec", type=int, default=120)
    args = parser.parse_args()

    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    if os.environ.get("PULSE_SERVER"):
        os.environ.setdefault("PULSE_LATENCY_MSEC", str(args.latency_msec))
    if args.driver:
        os.environ["SDL_AUDIODRIVER"] = args.driver
    elif os.environ.get("PULSE_SERVER"):
        os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

    import pygame

    root = Path(__file__).resolve().parents[1]
    audio_dir = root / "assets" / "audio"
    filename = args.name if args.name.endswith(".wav") else f"{args.name}.wav"
    wav_path = audio_dir / filename
    if not wav_path.exists():
        raise SystemExit(f"Audio file not found: {wav_path}")

    pygame.mixer.pre_init(frequency=48000, size=-16, channels=2, buffer=args.buffer, allowedchanges=0)
    pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=args.buffer, allowedchanges=0)
    print("mixer:", pygame.mixer.get_init())
    print("driver:", os.environ.get("SDL_AUDIODRIVER", "default"))
    print("pulse_latency_msec:", os.environ.get("PULSE_LATENCY_MSEC", "default"))
    print("mode:", args.mode)
    print("volume:", args.volume)
    print("buffer:", args.buffer)
    print("file:", wav_path)

    if args.mode == "music":
        pygame.mixer.music.set_volume(args.volume)
        pygame.mixer.music.load(str(wav_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.02)
    else:
        channel = pygame.mixer.Channel(0)
        sound = pygame.mixer.Sound(str(wav_path))
        sound.set_volume(args.volume)
        channel.play(sound)
        while channel.get_busy():
            time.sleep(0.02)
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
