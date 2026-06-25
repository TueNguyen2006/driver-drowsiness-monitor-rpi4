from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_system.config import HybridConfig, load_config
from hybrid_system.exports import export_run_artifacts
from hybrid_system.models import DetectionEvent, SessionSummary
from hybrid_system.overlay import AnnotatedVideoWriter, draw_overlay
from hybrid_system.pipeline import HybridPipeline
from hybrid_system.risk_scorer import RiskScorer
from hybrid_system.ui import embed_kiosk_overlay


def analyze_video(video_path: str, output_dir: str, config: HybridConfig,
                  kiosk_ui: bool = False) -> dict[str, Path]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pipeline = HybridPipeline(config)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if config.runtime.write_video:
        out_fps = config.runtime.output_fps or fps
        out_size = (800, 480) if kiosk_ui else (width, height)
        writer = AnnotatedVideoWriter(out_dir / "annotated.mp4", out_fps, out_size)

    processed = 0
    last_ts = 0.0
    risk_timeline: list[dict[str, float]] = []
    latencies: list[float] = []
    all_events: list[DetectionEvent] = []

    print(f"Processing: {video_path}  ({total_frames} frames, {fps:.1f} fps)")
    print("Calibrating...", end=" ", flush=True)

    cal_done = False
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps

        result = pipeline.process_frame(frame, ts, frame_idx)

        if not cal_done and not pipeline.is_calibrating:
            cal_done = True
            print("done")
        elif not cal_done and frame_idx > config.calibration.frame_count + 30:
            print("no face found, using defaults")
            pipeline.calibrator.force_defaults()
            cal_done = True
            print("done")

        processed += 1
        last_ts = ts
        risk_timeline.append({"timestamp": ts, "risk_score": result.risk_score})
        latencies.append(result.latency_ms)
        all_events.extend(result.events)

        if kiosk_ui:
            overlay = embed_kiosk_overlay(result, frame)
            if pipeline.is_calibrating:
                prog = pipeline.calibration_progress
                bar_w, bar_h, bar_x, bar_y = 200, 16, (800 - 200) // 2, 6
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 36, 38), -1)
                fill = int(bar_w * prog)
                if fill > 0:
                    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (62, 197, 124), -1)
                cv2.putText(overlay, f"CALIBRATING {prog:.0%}", (bar_x + 12, bar_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            overlay = draw_overlay(result, frame)

        if writer:
            writer.write(overlay)

        if config.runtime.display:
            cv2.imshow("Hybrid Driver Monitor", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if processed % 100 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {frame_idx}/{total_frames} ({pct:.0f}%)", flush=True)

        if config.runtime.max_frames and processed >= config.runtime.max_frames:
            break

    cap.release()
    if writer:
        writer.close()
    if config.runtime.display:
        cv2.destroyWindow("Hybrid Driver Monitor")

    all_events.extend(pipeline.get_events())

    session_id = f"{Path(video_path).stem}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    event_counts: dict[str, int] = {}
    for e in all_events:
        event_counts[e.signal] = event_counts.get(e.signal, 0) + 1

    max_unsafe = 0.0
    cur_unsafe = 0.0
    unsafe_thr = config.report.unsafe_threshold
    for p in risk_timeline:
        if p["risk_score"] >= unsafe_thr:
            cur_unsafe += 1.0 / fps
        else:
            if cur_unsafe > max_unsafe:
                max_unsafe = cur_unsafe
            cur_unsafe = 0.0
    if cur_unsafe > max_unsafe:
        max_unsafe = cur_unsafe

    scorer = RiskScorer()
    summary = scorer.summarize(
        session_id=session_id,
        source=video_path,
        duration_seconds=last_ts,
        processed_frames=processed,
        events=all_events,
        frame_scores=[(p["timestamp"], p["risk_score"]) for p in risk_timeline],
        metrics={
            "source_fps": round(fps, 2),
            "avg_latency_ms": round(float(np.mean(latencies)), 3) if latencies else 0,
            "p95_latency_ms": round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 3) if len(latencies) > 20 else 0,
            "estimated_runtime_fps": round(1000.0 / float(np.mean(latencies)), 2) if latencies and np.mean(latencies) > 0 else 0,
            "face_provider": "mediapipe",
            "object_provider": "yolo11n" if config.runtime.phone_enabled else "none",
            "timeline_stride": config.report.timeline_stride_frames,
        },
    )

    artifacts = export_run_artifacts(out_dir, events=all_events, summary=summary)

    print(f"\nResults:")
    print(f"  Frames: {processed}, Duration: {last_ts:.1f}s")
    for name, p in artifacts.items():
        print(f"  {name}: {p}")
    if writer:
        print(f"  annotated_video: {out_dir / 'annotated.mp4'}")

    return artifacts


def run_webcam(config: HybridConfig, camera_index: int = 0,
               kiosk_ui: bool = False) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    if config.runtime.max_frames:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    pipeline = HybridPipeline(config)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if config.runtime.write_video:
        out_dir = Path("webcam_recording")
        out_dir.mkdir(exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_size = (800, 480) if kiosk_ui else (width, height)
        writer = AnnotatedVideoWriter(out_dir / f"recording_{ts_str}.mp4", fps, out_size)

    display = config.runtime.display
    cv_window = "Hybrid Driver Monitor"
    if display:
        cv2.namedWindow(cv_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(cv_window, 960, 720)

    print("Webcam running. Press 'q' to quit")
    frame_idx = 0
    cal_announced = False
    last_event_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps

        result = pipeline.process_frame(frame, ts, frame_idx)

        if not cal_announced and not pipeline.is_calibrating:
            print("Calibration complete")
            cal_announced = True
        if not cal_announced and frame_idx > config.calibration.frame_count + 30:
            pipeline.calibrator.force_defaults()
            print("Calibration complete (defaults)")
            cal_announced = True

        if time.time() - last_event_time > max(1.0, config.runtime.alert_cooldown_seconds):
            for event in result.events:
                if event.severity.value in ("warning", "critical"):
                    print(f"[{event.severity.value.upper()}] {event.message} (score={event.score:.2f})")
                    last_event_time = time.time()

        if kiosk_ui:
            overlay = embed_kiosk_overlay(result, frame)
            if pipeline.is_calibrating:
                prog = pipeline.calibration_progress
                bar_w, bar_h, bar_x, bar_y = 200, 16, (800 - 200) // 2, 6
                cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 36, 38), -1)
                fill = int(bar_w * prog)
                if fill > 0:
                    cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (62, 197, 124), -1)
                cv2.putText(overlay, f"CALIBRATING {prog:.0%}", (bar_x + 12, bar_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            overlay = draw_overlay(result, frame)

        if display:
            cv2.imshow(cv_window, overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            if writer:
                writer.write(overlay)

    cap.release()
    if writer:
        writer.close()
    cv2.destroyAllWindows()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid Driver Monitor System")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--webcam", type=int, nargs="?", const=0, default=None,
                        help="Webcam index (default: 0)")
    parser.add_argument("--output", type=str, default="runs/demo", help="Output directory")
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--no-phone", action="store_true", help="Disable phone detection")
    parser.add_argument("--max-frames", type=int, help="Max frames to process")
    parser.add_argument("--skip-frames", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--kiosk-ui", action="store_true", help="Render using kiosk UI (800x480)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_display:
        config.runtime.display = False
    if args.no_phone:
        config.runtime.phone_enabled = False
        config.object_detector.enabled = False
    if args.max_frames:
        config.runtime.max_frames = args.max_frames
    if args.skip_frames > 1:
        config.vision.process_every_n_frames = args.skip_frames

    kiosk_ui = args.kiosk_ui
    if args.video:
        analyze_video(args.video, args.output, config, kiosk_ui)
    elif args.webcam is not None:
        run_webcam(config, args.webcam, kiosk_ui)
    else:
        print("Specify --video <path> or --webcam")
        sys.exit(1)


if __name__ == "__main__":
    main()
