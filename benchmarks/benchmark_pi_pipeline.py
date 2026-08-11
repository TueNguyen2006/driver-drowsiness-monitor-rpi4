#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_system.runtime_tuning import configure_thread_environment


SCENARIOS = (
    ("inference-only", False, False, False),
    ("ui", True, False, False),
    ("video", True, True, False),
    ("yolo", False, False, True),
    ("full", True, True, True),
)


def _source_value(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def _run_scenario(
    name: str,
    *,
    source: int | str,
    frame_limit: int,
    include_ui: bool,
    include_video: bool,
    include_yolo: bool,
    display: bool,
    config_path: str | None,
    output_dir: Path,
    threads: int,
    yolo_size: int | None,
) -> dict[str, object]:
    import cv2

    from hybrid_system.async_kiosk import AsyncFrameOutput
    from hybrid_system.config import load_config
    from hybrid_system.pipeline import HybridPipeline
    from hybrid_system.profiling import ProfilingWindow
    from hybrid_system.runtime_tuning import configure_runtime_threads

    cfg = load_config(config_path)
    cfg.runtime.display = display and include_ui
    cfg.runtime.write_video = include_video
    cfg.runtime.async_output = include_ui or include_video
    cfg.runtime.phone_enabled = include_yolo
    cfg.object_detector.enabled = include_yolo
    cfg.object_detector.async_enabled = True
    if yolo_size is not None:
        cfg.object_detector.input_size = yolo_size
    configure_runtime_threads(threads)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open benchmark source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = HybridPipeline(cfg)
    pipeline.calibrator.force_defaults()
    profiler = ProfilingWindow()
    writer_path = output_dir / f"{name}.mp4" if include_video else None
    output = None
    if include_ui or include_video:
        output = AsyncFrameOutput(
            cfg,
            f"Pi benchmark: {name}",
            (800, 480),
            fps,
            writer_path=writer_path,
        )

    processed = 0
    started = perf_counter()
    try:
        while processed < frame_limit:
            camera_started = perf_counter()
            ok, frame = capture.read()
            camera_ms = (perf_counter() - camera_started) * 1000
            if not ok or frame is None:
                break
            result = pipeline.process_frame(frame, time.time(), processed)
            profile = pipeline.last_profile
            profile["camera_ms"] = camera_ms
            profiler.record(profile)
            if output is not None:
                output.submit(frame, result)
            processed += 1
    finally:
        capture.release()
        pipeline.wait_for_background(timeout=10.0)
        if output is not None:
            output.wait_until_idle(timeout=5.0)
            output_profile = output.snapshot_profile()
            output_stats = asdict(output.snapshot_stats())
            output.close()
        else:
            output_profile = {}
            output_stats = {}
        elapsed = perf_counter() - started
        background_profile = pipeline.background_profile()
        pipeline.close()

    stages = profiler.summary()
    stages.update(background_profile)
    stages.update(output_profile)
    resources = asdict(profiler.resources())
    result_data: dict[str, object] = {
        "scenario": name,
        "frames": processed,
        "elapsed_seconds": elapsed,
        "throughput_fps": processed / elapsed if elapsed > 0 else 0.0,
        "threads": threads,
        "yolo_input_size": cfg.object_detector.input_size if include_yolo else None,
        "stages": stages,
        "resources": resources,
        "output": output_stats,
    }
    return result_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Raspberry Pi kiosk pipeline")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--config", help="Optional YAML config")
    parser.add_argument("--display", action="store_true", help="Include the physical display backend")
    parser.add_argument("--yolo-size", type=int, choices=(320, 416, 640))
    parser.add_argument(
        "--scenarios",
        default=",".join(item[0] for item in SCENARIOS),
        help="Comma-separated scenario names",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmark_results")
    args = parser.parse_args()

    configure_thread_environment(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    requested = {name.strip() for name in args.scenarios.split(",") if name.strip()}
    unknown = requested.difference(item[0] for item in SCENARIOS)
    if unknown:
        parser.error(f"Unknown scenarios: {', '.join(sorted(unknown))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_value(args.source)
    results = []
    for name, ui, video, yolo in SCENARIOS:
        if name not in requested:
            continue
        logging.info("Running scenario=%s frames=%d", name, args.frames)
        result = _run_scenario(
            name,
            source=source,
            frame_limit=args.frames,
            include_ui=ui,
            include_video=video,
            include_yolo=yolo,
            display=args.display,
            config_path=args.config,
            output_dir=args.output_dir,
            threads=max(1, args.threads),
            yolo_size=args.yolo_size,
        )
        results.append(result)
        logging.info(
            "scenario=%s fps=%.2f cpu=%.1f%% rss=%.1fMB",
            name,
            result["throughput_fps"],
            result["resources"]["cpu_percent"],
            result["resources"]["rss_mb"],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"pi_pipeline_{timestamp}.json"
    output_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "platform": {
                    "system": os.uname().sysname,
                    "node": os.uname().nodename,
                    "release": os.uname().release,
                    "machine": os.uname().machine,
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
