#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hybrid_system.runtime_tuning import configure_thread_environment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare OpenCV DNN and ONNX Runtime on the same YOLO tensors"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "yolov8n.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "benchmark_results")
    args = parser.parse_args()

    configure_thread_environment(args.threads)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "ONNX Runtime is optional. Install it in the benchmark environment with "
            "`pip install onnxruntime`, then rerun this command."
        ) from exc

    import cv2
    import numpy as np

    from hybrid_system.objects import _letterbox
    from hybrid_system.runtime_tuning import configure_runtime_threads

    configure_runtime_threads(args.threads)
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.source}")
    frames = []
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
        raise RuntimeError("No frames available")

    cv_net = cv2.dnn.readNetFromONNX(str(args.model))
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = args.threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort_session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = ort_session.get_inputs()[0].name

    blobs = []
    for frame in frames:
        resized, _, _, _ = _letterbox(frame, args.input_size, args.input_size)
        blobs.append(cv2.dnn.blobFromImage(
            resized,
            1 / 255.0,
            (args.input_size, args.input_size),
            swapRB=True,
            crop=False,
        ))

    # Warm both runtimes before collecting latency.
    cv_net.setInput(blobs[0])
    cv_net.forward()
    ort_session.run(None, {input_name: blobs[0]})

    cv_latencies = []
    ort_latencies = []
    max_abs_differences = []
    mean_abs_differences = []
    for blob in blobs:
        started = perf_counter()
        cv_net.setInput(blob)
        cv_output = cv_net.forward()
        cv_latencies.append((perf_counter() - started) * 1000)

        started = perf_counter()
        ort_output = ort_session.run(None, {input_name: blob})[0]
        ort_latencies.append((perf_counter() - started) * 1000)

        difference = np.abs(cv_output - ort_output)
        max_abs_differences.append(float(difference.max()))
        mean_abs_differences.append(float(difference.mean()))

    def summarize(latencies: list[float]) -> dict[str, float]:
        mean_ms = statistics.fmean(latencies)
        return {
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "fps": 1000.0 / mean_ms,
        }

    result = {
        "source": str(Path(args.source).resolve()),
        "model": str(args.model.resolve()),
        "frames": len(blobs),
        "threads": args.threads,
        "input_size": args.input_size,
        "opencv_dnn": summarize(cv_latencies),
        "onnxruntime": summarize(ort_latencies),
        "output_consistency": {
            "max_abs_difference": max(max_abs_differences),
            "mean_abs_difference": statistics.fmean(mean_abs_differences),
            "passes": max(max_abs_differences) <= 1e-3,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"onnx_runtimes_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
