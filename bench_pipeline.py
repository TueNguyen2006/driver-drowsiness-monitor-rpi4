from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np

from hybrid_system.config import load_config
from hybrid_system.pipeline import HybridPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark HybridPipeline stage latency without opening kiosk UI."
    )
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to process.")
    parser.add_argument("--width", type=int, default=640, help="Synthetic frame width.")
    parser.add_argument("--height", type=int, default=480, help="Synthetic frame height.")
    parser.add_argument("--no-phone", action="store_true", help="Disable object/phone detection.")
    parser.add_argument(
        "--async-landmarks",
        action="store_true",
        help="Benchmark optional async landmarks mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.runtime.display = False
    cfg.runtime.write_video = False
    cfg.vision.async_landmarks = bool(args.async_landmarks)
    if args.no_phone:
        cfg.runtime.phone_enabled = False
        cfg.object_detector.enabled = False

    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    pipeline = HybridPipeline(cfg)
    totals: dict[str, list[float]] = {}

    try:
        for frame_index in range(max(1, args.frames)):
            pipeline.process_frame(frame, time.time(), frame_index)
            for key, value in pipeline.last_profile.items():
                totals.setdefault(key, []).append(value)
    finally:
        pipeline.close()

    print(f"frames={args.frames}")
    print(f"phone_enabled={cfg.runtime.phone_enabled}")
    print(f"async_landmarks={cfg.vision.async_landmarks}")
    for key in sorted(totals):
        values = totals[key]
        print(f"{key}: avg={mean(values):.2f}ms min={min(values):.2f}ms max={max(values):.2f}ms")


if __name__ == "__main__":
    main()
