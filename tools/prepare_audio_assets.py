from __future__ import annotations

import argparse
import audioop
import wave
from pathlib import Path


def convert_wav_to_48k_stereo(path: Path) -> bool:
    with wave.open(str(path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        frame_count = reader.getnframes()
        data = reader.readframes(frame_count)

    changed = False
    if sample_width != 2:
        data = audioop.lin2lin(data, sample_width, 2)
        sample_width = 2
        changed = True
    if channels == 1:
        data = audioop.tostereo(data, sample_width, 1.0, 1.0)
        channels = 2
        changed = True
    elif channels != 2:
        data = audioop.tomono(data, sample_width, 0.5, 0.5)
        data = audioop.tostereo(data, sample_width, 1.0, 1.0)
        channels = 2
        changed = True
    if frame_rate != 48000:
        data, _ = audioop.ratecv(data, sample_width, channels, frame_rate, 48000, None)
        frame_rate = 48000
        changed = True

    if not changed:
        return False

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.writeframes(data)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize repo WAV assets for pygame.mixer")
    parser.add_argument("--audio-dir", default="assets/audio")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    changed = 0
    total = 0
    for path in sorted(audio_dir.glob("*.wav")):
        total += 1
        if convert_wav_to_48k_stereo(path):
            changed += 1
            print(f"converted: {path.name}")
        else:
            print(f"ok: {path.name}")
    print(f"processed={total} changed={changed}")


if __name__ == "__main__":
    main()
