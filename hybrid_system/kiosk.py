from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from .config import HybridConfig
from .exports import export_run_artifacts
from .hardware_controller import HardwareController
from .models import DetectionEvent, DriverState, SessionSummary
from .overlay import AnnotatedVideoWriter
from .pipeline import HybridPipeline, ThreadedCamera
from .risk_scorer import RiskScorer
from .ui import embed_kiosk_overlay

log = logging.getLogger("kiosk")


class KioskMode(str, Enum):
    INIT = "init"
    CALIBRATING = "calibrating"
    INFERRING = "inferring"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class IndustrialKiosk:
    __slots__ = (
        "cfg", "pipeline", "hardware", "mode", "_mode_lock",
        "_cap", "_writer", "_fps", "_frame_size",
        "_running", "_paused", "_restart_requested",
        "_frame_idx", "_events", "_risk_timeline",
        "_latencies", "_alert_active", "_alert_start",
        "_last_alert_time", "_cal_start_time",
        "_last_gc", "_window_name",
        "_fps_times",
    )

    PANEL_W = 320
    MARGIN = 16
    HEADER_H = 60
    FOOTER_H = 80

    def __init__(self, config: HybridConfig) -> None:
        self.cfg = config
        self.pipeline = HybridPipeline(config)
        self.hardware = HardwareController(config)
        self.mode = KioskMode.INIT
        self._mode_lock = threading.Lock()

        self._cap: cv2.VideoCapture | None = None
        self._writer: AnnotatedVideoWriter | None = None
        self._fps = 30.0
        self._frame_size = (640, 480)

        self._running = True
        self._paused = False
        self._restart_requested = False

        self._frame_idx = 0
        self._events: list[DetectionEvent] = []
        self._risk_timeline: list[dict[str, float]] = []
        self._latencies: list[float] = []
        self._alert_active = False
        self._alert_start = 0.0
        self._last_alert_time = 0.0
        self._cal_start_time = 0.0
        self._last_gc = 0.0
        self._fps_times: deque = deque(maxlen=30)

        self._window_name = "Driver Safety Kiosk"

    def run(self) -> None:
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log.info("Kiosk starting")
        self.hardware.set_restart_callback(self._request_restart)
        self.hardware.set_pause_callback(self._toggle_pause)
        self.hardware.start()

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            self._window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

        try:
            while self._running:
                with self._mode_lock:
                    mode = self.mode

                if mode == KioskMode.SHUTDOWN:
                    break

                if self._restart_requested:
                    self._handle_restart()

                if mode == KioskMode.INIT:
                    self._open_camera()
                    self._enter_calibration()
                elif mode == KioskMode.CALIBRATING:
                    self._run_calibration_loop()
                elif mode == KioskMode.INFERRING:
                    self._run_inference_loop()
                elif mode == KioskMode.PAUSED:
                    self._run_paused_loop()
                elif mode == KioskMode.ERROR:
                    self._run_error_loop()

        except KeyboardInterrupt:
            log.info("Kiosk interrupted")
        except Exception as e:
            log.exception("Kiosk error: %s", e)
            self._show_error_screen(str(e))
        finally:
            self._shutdown(start_ts)

    def _set_mode(self, mode: KioskMode) -> None:
        with self._mode_lock:
            self.mode = mode

    def _open_camera(self) -> None:
        idx = self.cfg.vision.camera_index
        self._cap = ThreadedCamera(idx)
        if not self._cap or not self._cap.isOpened():
            log.error("Cannot open camera %d", idx)
            self._set_mode(KioskMode.ERROR)
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._frame_size = (w, h)

        if self.cfg.runtime.write_video:
            out_dir = Path("kiosk_recordings")
            out_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._writer = AnnotatedVideoWriter(
                out_dir / f"session_{ts}.mp4", self._fps, self._frame_size
            )

    def _close_camera(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _enter_calibration(self) -> None:
        self._reset_buffers()
        self._set_mode(KioskMode.CALIBRATING)
        self._cal_start_time = time.time()
        self._open_camera()
        self.hardware.set_leds(calibration=True, inference=False)
        self.hardware.announce_calibration_start()
        log.info("Starting calibration")

    def _run_calibration_loop(self) -> None:
        timeout = 30.0
        last_frame_time = 0.0

        while self._running and self.mode == KioskMode.CALIBRATING:
            now = time.time()
            if now - self._cal_start_time > timeout:
                log.warning("Calibration timeout, using defaults")
                self.pipeline.calibrator.force_defaults()
                break

            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self.pipeline.process_frame(frame, now, self._frame_idx)
            self._frame_idx += 1

            self._fps_times.append(now)

            overlay = self._draw_kiosk_frame(frame, result)
            cv2.imshow(self._window_name, overlay)
            if self._writer:
                self._writer.write(overlay)

            if not self.pipeline.is_calibrating:
                self.hardware.set_leds(calibration=False, inference=True)
                self.hardware.announce_calibration_complete()
                time.sleep(0.5)
                self.hardware.announce_infer_started()
                self._set_mode(KioskMode.INFERRING)
                log.info("Calibration complete, starting inference")
                return

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

    def _run_inference_loop(self) -> None:
        self._reset_buffers()
        self._alert_active = False
        self._alert_start = 0.0
        self._last_alert_time = 0.0
        self.hardware.set_alert(False)
        self.hardware.set_leds(calibration=False, inference=True)
        frame_count = 0

        while self._running and self.mode == KioskMode.INFERRING:
            if self._paused:
                self._set_mode(KioskMode.PAUSED)
                return
            if self._restart_requested:
                return

            now = time.time()
            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self.pipeline.process_frame(frame, now, self._frame_idx)
            self._frame_idx += 1
            self._latencies.append(result.latency_ms)
            self._risk_timeline.append({"timestamp": now, "risk_score": result.risk_score})
            self._events.extend(result.events)

            self._fps_times.append(now)

            self._handle_alerts(result)

            overlay = self._draw_kiosk_frame(frame, result)
            cv2.imshow(self._window_name, overlay)
            if self._writer:
                self._writer.write(overlay)

            frame_count += 1
            if frame_count % 300 == 0:
                gc.collect()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._running = False

    def _run_paused_loop(self) -> None:
        self.hardware.set_leds(calibration=True, inference=True)
        self.hardware.set_alert(False)
        blank = np.zeros((self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8)
        while self._running and self.mode == KioskMode.PAUSED:
            if not self._paused:
                self._set_mode(KioskMode.INFERRING)
                return
            if self._restart_requested:
                return

            cv2.putText(blank, "PAUSED", (self._frame_size[0] // 3, self._frame_size[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
            cv2.putText(blank, "Press PAUSE button to resume", (self._frame_size[0] // 4, self._frame_size[1] // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(self._window_name, blank)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                self._running = False

    def _run_error_loop(self) -> None:
        self.hardware.set_leds(calibration=False, inference=False)
        blank = np.zeros((self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8)
        start = time.time()
        while self._running and self.mode == KioskMode.ERROR:
            cv2.putText(blank, "SYSTEM ERROR", (self._frame_size[0] // 4, self._frame_size[1] // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(blank, "Restarting in 10s...", (self._frame_size[0] // 4, self._frame_size[1] // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow(self._window_name, blank)
            if time.time() - start > 10:
                self._set_mode(KioskMode.INIT)
            if cv2.waitKey(200) & 0xFF == ord("q"):
                self._running = False

    def _grab_frame(self) -> np.ndarray | None:
        if self._cap is None or not self._cap.isOpened():
            self._open_camera()
            if self._cap is None:
                return None
        ret, frame = self._cap.read()
        if not ret:
            self._close_camera()
            return None
        return frame

    def _handle_alerts(self, result) -> None:
        now = time.time()
        thr = self.cfg.report.unsafe_threshold
        cooldown = self.cfg.runtime.alert_cooldown_seconds

        if result.risk_score >= thr:
            if not self._alert_active:
                self._alert_start = now
                self._alert_active = True
            elif now - self._alert_start >= self.cfg.hardware.alert_stable_seconds:
                if now - self._last_alert_time >= cooldown:
                    self.hardware.set_alert(True)
                    self._last_alert_time = now
                    for event in result.events:
                        if event.severity.value in ("warning", "critical"):
                            log.warning("ALERT: %s (score=%.2f)", event.message, event.score)
        else:
            if self._alert_active:
                if now - self._alert_start >= self.cfg.hardware.alert_hold_seconds:
                    self._alert_active = False
                    self.hardware.set_alert(False)

    def _request_restart(self) -> None:
        self._restart_requested = True

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._paused:
            log.info("Inference paused by button")

    def _handle_restart(self) -> None:
        self._restart_requested = False
        self._paused = False
        self._close_camera()
        self.pipeline.calibrator.stop()
        self.pipeline = HybridPipeline(self.cfg)
        self._frame_idx = 0
        self._events = []
        self._risk_timeline = []
        self._latencies = []
        self._fps_times.clear()
        self._enter_calibration()

    def _reset_buffers(self) -> None:
        self._frame_idx = 0
        self._events = []
        self._risk_timeline = []
        self._latencies = []
        self._fps_times.clear()

    def _get_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        window = self._fps_times[-1] - self._fps_times[0]
        if window <= 0:
            return 0.0
        return (len(self._fps_times) - 1) / window

    def _draw_kiosk_frame(self, frame: np.ndarray, result) -> np.ndarray:
        canvas = embed_kiosk_overlay(result, frame)
        if self.pipeline.is_calibrating:
            prog = self.pipeline.calibration_progress
            bar_w = 200
            bar_h = 16
            bx = (800 - bar_w) // 2
            by = 6
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (30, 36, 38), -1)
            fill = int(bar_w * prog)
            if fill > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fill, by + bar_h), (62, 197, 124), -1)
            cv2.putText(canvas, f"CALIBRATING {prog:.0%}", (bx + 12, by + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _show_error_screen(self, msg: str) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "FATAL ERROR", (160, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(blank, msg[:50], (80, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self._window_name, blank)
        cv2.waitKey(5000)

    def _shutdown(self, start_ts: str) -> None:
        log.info("Kiosk shutting down")
        self.hardware.set_alert(False)
        self.hardware.set_leds(calibration=False, inference=False)
        self.pipeline.calibrator.stop()

        if self._writer:
            self._writer.close()

        self._close_camera()

        if self._events or self._risk_timeline:
            try:
                scorer = RiskScorer()
                summary = scorer.summarize(
                    session_id=f"kiosk-{start_ts}",
                    source="kiosk",
                    duration_seconds=time.time() - self._cal_start_time if self._cal_start_time else 0,
                    processed_frames=self._frame_idx,
                    events=self._events,
                    frame_scores=[(p["timestamp"], p["risk_score"]) for p in self._risk_timeline],
                    metrics={
                        "avg_latency_ms": round(float(np.mean(self._latencies)), 3) if self._latencies else 0,
                        "estimated_fps": round(1000.0 / float(np.mean(self._latencies)), 2) if self._latencies else 0,
                        "face_provider": "mediapipe",
                    },
                )
                out_dir = Path("kiosk_reports")
                export_run_artifacts(out_dir, events=self._events, summary=summary)
                log.info("Session report saved to %s", out_dir)
            except Exception as e:
                log.error("Failed to save report: %s", e)

        self.hardware.cleanup()
        cv2.destroyAllWindows()
        log.info("Kiosk shutdown complete")
