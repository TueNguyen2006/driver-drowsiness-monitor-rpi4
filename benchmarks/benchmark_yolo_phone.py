#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_system.runtime_tuning import configure_thread_environment


def _phone_detections(observations) -> list:
    return [item for item in observations if item.label.lower() in {"cell phone", "phone", "mobile"}]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _compare(candidate: list[list], baseline: list[list]) -> dict[str, float | int | bool]:
    presence_matches = 0
    positive_baseline = 0
    matched_ious: list[float] = []
    for current, reference in zip(candidate, baseline):
        current_phones = _phone_detections(current)
        reference_phones = _phone_detections(reference)
        if bool(current_phones) == bool(reference_phones):
            presence_matches += 1
        if reference_phones:
            positive_baseline += 1
            if current_phones:
                matched_ious.append(max(
                    _iou(candidate_box.bbox, reference_box.bbox)
                    for candidate_box in current_phones
                    for reference_box in reference_phones
                ))
    frame_count = len(baseline)
    agreement = presence_matches / frame_count if frame_count else 0.0
    mean_iou = statistics.fmean(matched_ious) if matched_ious else 0.0
    conclusive = positive_baseline > 0
    return {
        "phone_presence_agreement": agreement,
        "baseline_phone_positive_frames": positive_baseline,
        "mean_best_phone_iou": mean_iou,
        "conclusive": conclusive,
        "passes_consistency_gate": bool(
            conclusive and agreement >= 0.95 and mean_iou >= 0.5
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO phone input sizes and output consistency")
    parser.add_argument("--source", required=True, help="Video file containing representative phone use")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--sizes", default="320,416,640")
    parser.add_argument("--include-ultralytics", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmark_results")
    args = parser.parse_args()

    configure_thread_environment(args.threads)
    import cv2
    import numpy as np

    from hybrid_system.objects import ObjectDetector
    from hybrid_system.runtime_tuning import configure_runtime_threads

    configure_runtime_threads(args.threads)
    sizes = [int(item) for item in args.sizes.split(",")]
    if 640 not in sizes:
        parser.error("Sizes must include the 640 baseline")

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")
    frames: list[np.ndarray] = []
    index = 0
    while len(frames) < args.frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if index % max(1, args.stride) == 0:
            frames.append(frame)
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError("No frames available for YOLO benchmark")

    variants: list[tuple[str, Path, int]] = [
        (f"opencv-dnn-onnx-{size}", PROJECT_ROOT / "yolov8n.onnx", size)
        for size in sizes
    ]
    if args.include_ultralytics:
        variants.extend(
            (f"ultralytics-pytorch-{size}", PROJECT_ROOT / "yolov8n.pt", size)
            for size in sizes
        )

    detections: dict[str, list[list]] = {}
    results: dict[str, dict[str, object]] = {}
    for name, model_path, size in variants:
        detector = ObjectDetector(
            enabled=True,
            model_path=str(model_path),
            input_size=size,
            confidence_threshold=0.25,
            iou_threshold=0.45,
        )
        detector.detect(frames[0])
        latencies = []
        outputs = []
        errors: list[str] = []
        for frame in frames:
            started = perf_counter()
            output = detector.detect(frame)
            latencies.append((perf_counter() - started) * 1000)
            outputs.append(output)
            if detector.last_error:
                errors.append(detector.last_error)
        detections[name] = outputs
        results[name] = {
            "provider": detector.provider,
            "model": str(model_path),
            "input_size": size,
            "frames": len(frames),
            "mean_ms": statistics.fmean(latencies),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": 1000.0 / statistics.fmean(latencies),
            "phone_positive_frames": sum(bool(_phone_detections(items)) for items in outputs),
            "status": "failed" if errors else "ok",
            "error": errors[0] if errors else None,
        }

    baseline = detections["opencv-dnn-onnx-640"]
    for name, output in detections.items():
        comparison = _compare(output, baseline)
        if results[name]["status"] != "ok":
            comparison["conclusive"] = False
            comparison["passes_consistency_gate"] = False
        results[name]["consistency_vs_opencv_640"] = comparison

    runtime_inventory = {
        "opencv_dnn_onnx": "benchmarked",
        "ultralytics_pytorch": "benchmarked" if args.include_ultralytics else "available; opt in with --include-ultralytics",
        "onnxruntime": "installed" if importlib.util.find_spec("onnxruntime") else "unavailable",
        "ncnn": "installed but no NCNN model in repo" if importlib.util.find_spec("ncnn") else "unavailable",
        "tflite": "installed but no TFLite model in repo" if importlib.util.find_spec("tflite_runtime") else "unavailable",
        "int8": "not benchmarked: no calibrated INT8 model or validation set in repo",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"yolo_phone_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(
        json.dumps(
            {
                "source": str(Path(args.source).resolve()),
                "sampled_frames": len(frames),
                "threads": args.threads,
                "runtime_inventory": runtime_inventory,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
