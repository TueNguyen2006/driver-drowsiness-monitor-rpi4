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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybrid_system.config import load_config
from hybrid_system.kiosk import IndustrialKiosk


def main() -> None:
    parser = argparse.ArgumentParser(description="Driver Drowsiness Monitor - Kiosk Mode")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--no-phone", action="store_true", help="Disable phone detection")
    parser.add_argument("--no-display", action="store_true", help="Run headless (logging only)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("kiosk.log"),
        ],
    )
    log = logging.getLogger("kiosk")

    config = load_config(args.config)
    if args.no_phone:
        config.runtime.phone_enabled = False
        config.object_detector.enabled = False
    if args.no_display:
        config.runtime.display = False

    log.info("Starting kiosk with config: vision=%s, phone=%s",
             config.vision.provider, config.object_detector.provider)

    kiosk = IndustrialKiosk(config)
    kiosk.run()


if __name__ == "__main__":
    main()
