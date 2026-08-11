#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_system.config import load_config
from hybrid_system.pipeline import HybridPipeline


def load_frames(
    video_path: Path, count: int, stride: int,
    frame_width: int, frame_height: int,
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open benchmark video: {video_path}")

    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                frames.append(frame)
            frame_index += 1
    finally:
        capture.release()

    if len(frames) < count:
        raise SystemExit(
            f"Video supplied only {len(frames)} sampled frames; expected {count}"
        )
    return frames


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def summarize(
    samples: list[dict[str, float]], face_hits: int,
    bbox_widths: list[int], bbox_heights: list[int],
) -> dict[str, object]:
    stage_names = sorted({name for sample in samples for name in sample})
    stages: dict[str, dict[str, float]] = {}
    for name in stage_names:
        values = [sample[name] for sample in samples if name in sample]
        stages[name] = {
            "mean_ms": round(statistics.fmean(values), 3),
            "p50_ms": round(percentile(values, 0.50), 3),
            "p95_ms": round(percentile(values, 0.95), 3),
            "max_ms": round(max(values), 3),
        }

    total = [sample["total_ms"] for sample in samples]
    mean_total = statistics.fmean(total)
    return {
        "frames": len(samples),
        "face_hits": face_hits,
        "face_hit_rate": round(face_hits / len(samples), 4),
        "face_bbox_mean": [
            round(statistics.fmean(bbox_widths), 2) if bbox_widths else 0.0,
            round(statistics.fmean(bbox_heights), 2) if bbox_heights else 0.0,
        ],
        "face_bbox_max": [
            max(bbox_widths, default=0),
            max(bbox_heights, default=0),
        ],
        "pipeline_fps": round(1000.0 / mean_total, 3),
        "stages": stages,
    }


def run_case(
    pipeline: HybridPipeline,
    frames: list[np.ndarray],
    warmup: int,
    measured: int,
) -> dict[str, object]:
    samples: list[dict[str, float]] = []
    face_hits = 0
    bbox_widths: list[int] = []
    bbox_heights: list[int] = []
    total_runs = warmup + measured

    for index in range(total_runs):
        frame = frames[index % len(frames)]
        result = pipeline.process_frame(frame, time.time(), index)
        if index < warmup:
            continue
        samples.append(dict(pipeline.last_profile))
        if result.debug_info.get("face_visible"):
            face_hits += 1
        if result.face_bbox is not None:
            bbox_widths.append(result.face_bbox[2])
            bbox_heights.append(result.face_bbox[3])

    return summarize(
        samples, face_hits, bbox_widths, bbox_heights
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark face/no-face pipeline latency")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--process-width", type=int, default=0)
    parser.add_argument("--process-height", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=640)
    parser.add_argument("--frame-height", type=int, default=480)
    args = parser.parse_args()

    source_frames = load_frames(
        args.video.resolve(), args.frames + args.warmup, max(1, args.stride),
        args.frame_width, args.frame_height,
    )
    blank_frames = [np.zeros_like(frame) for frame in source_frames]

    config = load_config()
    config.runtime.phone_enabled = False
    config.object_detector.enabled = False
    config.vision.async_landmarks = False
    config.vision.process_every_n_frames = 1
    config.vision.process_width = max(0, args.process_width)
    config.vision.process_height = max(0, args.process_height)

    pipeline = HybridPipeline(config)
    try:
        no_face = run_case(pipeline, blank_frames, args.warmup, args.frames)
        face = run_case(pipeline, source_frames, args.warmup, args.frames)
    finally:
        pipeline.close()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "video": str(args.video.resolve()),
        "settings": {
            "measured_frames": args.frames,
            "warmup_frames": args.warmup,
            "stride": args.stride,
            "process_width": config.vision.process_width,
            "process_height": config.vision.process_height,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "phone_enabled": False,
        },
        "no_face": no_face,
        "face": face,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
