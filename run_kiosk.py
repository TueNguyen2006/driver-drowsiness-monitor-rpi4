#!/usr/bin/env python3
"""
Kiosk entry point for industrial driver drowsiness monitor.
Runs full-screen with hardware integration on Raspberry Pi 4.

Usage:
  python run_kiosk.py                    # default config
  python run_kiosk.py --config config.yaml
  python run_kiosk.py --no-phone         # disable phone detection
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_system.runtime_tuning import configure_thread_environment


def main() -> None:
    parser = argparse.ArgumentParser(description="Driver Drowsiness Monitor - Kiosk Mode")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--camera-index", type=int, help="Camera index, e.g. 0 for /dev/video0")
    parser.add_argument("--camera-fourcc", type=str, help="Camera FourCC, e.g. MJPG or YUYV")
    parser.add_argument("--display-backend", type=str, help="Display backend: auto, sdl2, glfw, ffplay, opencv")
    parser.add_argument("--debug-ui", action="store_true", help="Show normalized feature/debug values on kiosk UI")
    parser.add_argument("--sync-output", action="store_true", help="Disable async render/display/write path")
    parser.add_argument("--write-video", action="store_true", help="Record kiosk output video asynchronously")
    parser.add_argument("--threads", type=int, help="OpenCV/PyTorch/OpenMP/BLAS thread limit (default: config, normally 1)")
    parser.add_argument("--windowed", action="store_true", help="Run kiosk in a normal window instead of fullscreen")
    parser.add_argument("--no-phone", action="store_true", help="Disable phone detection")
    parser.add_argument("--no-display", action="store_true", help="Run headless (logging only)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    from hybrid_system.config import load_config

    config = load_config(args.config)
    requested_threads = args.threads or int(
        os.environ.get("DROWSINESS_THREADS", str(config.runtime.inference_threads))
    )
    config.runtime.inference_threads = max(1, requested_threads)
    configure_thread_environment(config.runtime.inference_threads)

    from hybrid_system.kiosk import IndustrialKiosk
    from hybrid_system.runtime_tuning import configure_runtime_threads

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("kiosk.log"),
        ],
    )
    log = logging.getLogger("kiosk")

    if args.camera_index is not None:
        config.vision.camera_index = args.camera_index
    if args.camera_fourcc:
        config.vision.camera_fourcc = args.camera_fourcc
    if args.display_backend:
        config.runtime.display_backend = args.display_backend
    if args.debug_ui:
        config.runtime.kiosk_debug = True
    if args.sync_output:
        config.runtime.async_output = False
    if args.write_video:
        config.runtime.write_video = True
    if args.windowed:
        config.runtime.fullscreen = False
    if args.no_phone:
        config.runtime.phone_enabled = False
        config.object_detector.enabled = False
    if args.no_display:
        config.runtime.display = False

    applied_threads = configure_runtime_threads(config.runtime.inference_threads)

    log.info("Starting kiosk with config: vision=%s, phone=%s",
             config.vision.provider, config.object_detector.provider)
    log.info("Display config: backend=%s, fullscreen=%s, async_output=%s, kiosk_debug=%s",
             config.runtime.display_backend, config.runtime.fullscreen,
             config.runtime.async_output, config.runtime.kiosk_debug)
    log.info("Thread limits: requested=%d applied=%s",
             config.runtime.inference_threads, applied_threads)
    kiosk = IndustrialKiosk(config)
    kiosk.run()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
        category=UserWarning,
        module=r"google\.protobuf\.symbol_database",
    )
    main()
