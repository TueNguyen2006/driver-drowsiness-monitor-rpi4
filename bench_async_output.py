from __future__ import annotations

import argparse
import time
from statistics import mean

import numpy as np

from hybrid_system.async_kiosk import AsyncFrameOutput
from hybrid_system.config import load_config
from hybrid_system.models import DriverState, ProcessedFrame
from hybrid_system.ui import embed_kiosk_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark sync vs async kiosk output path.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--sink-delay-ms",
        type=float,
        default=8.0,
        help="Artificial display/write delay to emulate slow Pi4 output.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Input frame cadence. Use 0 to submit as fast as possible.",
    )
    return parser.parse_args()


def make_result(frame_index: int) -> ProcessedFrame:
    signals = {
        "eyes_closed": 0.0,
        "drowsy": 0.0,
        "yawning": 0.0,
        "distracted": 0.0,
        "phone_use": 0.0,
    }
    return ProcessedFrame(
        timestamp=time.time(),
        frame_index=frame_index,
        state=DriverState.ATTENTIVE,
        risk_score=0.0,
        signals=signals,
        events=[],
        latency_ms=0.0,
        landmarks=[],
        objects=[],
        head_pose=None,
    )


def render_with_delay(frame: np.ndarray, result: ProcessedFrame, delay_ms: float) -> np.ndarray:
    overlay = embed_kiosk_overlay(result, frame, landmark_mode="off")
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    return overlay


def pace_frame(started: float, source_fps: float) -> None:
    if source_fps <= 0:
        return
    period = 1.0 / source_fps
    remaining = period - (time.perf_counter() - started)
    if remaining > 0:
        time.sleep(remaining)


def run_sync(frame: np.ndarray, frames: int, delay_ms: float, source_fps: float) -> list[float]:
    timings: list[float] = []
    for idx in range(frames):
        started = time.perf_counter()
        render_with_delay(frame, make_result(idx), delay_ms)
        timings.append((time.perf_counter() - started) * 1000)
        pace_frame(started, source_fps)
    return timings


def run_async(frame: np.ndarray, frames: int, delay_ms: float, source_fps: float) -> tuple[list[float], int, int]:
    cfg = load_config()
    cfg.runtime.display = False
    cfg.runtime.write_video = False
    cfg.vision.overlay_landmark_mode = "off"

    def render_fn(src: np.ndarray, result: ProcessedFrame, _progress: float | None) -> np.ndarray:
        return render_with_delay(src, result, delay_ms)

    output = AsyncFrameOutput(
        cfg,
        "bench",
        (800, 480),
        30.0,
        writer_path=None,
        render_fn=render_fn,
    )
    timings: list[float] = []
    try:
        for idx in range(frames):
            frame_started = time.perf_counter()
            submit_started = time.perf_counter()
            output.submit(frame, make_result(idx), None)
            timings.append((time.perf_counter() - submit_started) * 1000)
            pace_frame(frame_started, source_fps)
    finally:
        output.close()
    stats = output.snapshot_stats()
    return timings, stats.rendered, stats.dropped


def summarize(name: str, values: list[float]) -> None:
    print(
        f"{name}: avg={mean(values):.3f}ms min={min(values):.3f}ms "
        f"max={max(values):.3f}ms fps_equiv={1000.0 / mean(values):.1f}"
    )


def main() -> None:
    args = parse_args()
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    sync = run_sync(frame, args.frames, args.sink_delay_ms, args.source_fps)
    async_submit, rendered, dropped = run_async(
        frame,
        args.frames,
        args.sink_delay_ms,
        args.source_fps,
    )
    summarize("sync_output_loop", sync)
    summarize("async_submit_loop", async_submit)
    print(f"async_rendered={rendered} async_dropped={dropped} submitted={args.frames}")


if __name__ == "__main__":
    main()
