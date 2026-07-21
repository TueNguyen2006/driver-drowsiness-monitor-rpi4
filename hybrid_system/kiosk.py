from __future__ import annotations

import gc
import logging
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import HybridConfig
from .async_kiosk import AsyncFrameOutput
from .exports import export_run_artifacts
from .display import create_display
from .hardware_controller import HardwareController
from .models import DetectionEvent, DriverState, SessionSummary
from .overlay import AnnotatedVideoWriter
from .pipeline import HybridPipeline, ThreadedCamera, fourcc_to_str
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
        "_fps_times", "_display",
        "_profile_sums", "_profile_count",
        "_output", "_writer_path",
        "_last_voice_time", "_last_voice_signal", "_signal_last_voice",
    )

    PANEL_W = 320
    MARGIN = 16
    HEADER_H = 60
    FOOTER_H = 80
    ALERT_PRIORITY = {
        "phone_use": 100,
        "face_lost": 90,
        "drowsy": 80,
        "distracted": 70,
        "yawning": 60,
    }

    def __init__(self, config: HybridConfig) -> None:
        self.cfg = config
        self.cfg.vision.process_every_n_frames = 1
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
        self._display = None
        self._output: AsyncFrameOutput | None = None
        self._writer_path: Path | None = None
        self._profile_sums: dict[str, float] = {}
        self._profile_count = 0
        self._last_voice_time = 0.0
        self._last_voice_signal = ""
        self._signal_last_voice: dict[str, float] = {}

        self._window_name = "Driver Safety Kiosk"

    def run(self) -> None:
        start_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log.info("Kiosk starting")
        self.hardware.set_restart_callback(self._request_restart)
        self.hardware.set_pause_callback(self._toggle_pause)
        self.hardware.start()
        self.hardware.speak_async("Hệ thống đã sẵn sàng", dedupe_window=0.0)
        time.sleep(0.2)

        try:
            while self._running:
                with self._mode_lock:
                    mode = self.mode

                if mode == KioskMode.SHUTDOWN:
                    break

                if self._restart_requested:
                    self._handle_restart()

                if mode == KioskMode.INIT:
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
        vision = self.cfg.vision
        idx = vision.camera_index
        self._cap = ThreadedCamera(
            idx,
            fourcc=vision.camera_fourcc,
            width=vision.camera_width,
            height=vision.camera_height,
            fps=vision.camera_fps,
        )
        if not self._cap or not self._cap.isOpened():
            log.error("Cannot open camera %d", idx)
            self._set_mode(KioskMode.ERROR)
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._frame_size = (w, h)
        log.info(
            "Camera %s opened: %dx%d %.2ffps fourcc=%s",
            idx,
            w,
            h,
            self._fps,
            fourcc_to_str(self._cap.get(cv2.CAP_PROP_FOURCC)),
        )

        if self.cfg.runtime.write_video:
            out_dir = Path("kiosk_recordings")
            out_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._writer_path = out_dir / f"session_{ts}.mp4"
            if self.cfg.runtime.async_output:
                self._writer = None
            else:
                self._writer = AnnotatedVideoWriter(self._writer_path, self._fps, (800, 480))

    def _close_camera(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    def _enter_calibration(self) -> None:
        self._reset_buffers()
        self._set_mode(KioskMode.CALIBRATING)
        self._open_camera()
        if self.mode == KioskMode.ERROR:
            return
        if self.cfg.runtime.async_output:
            self._ensure_output_worker()
        elif self.cfg.runtime.display and self._display is None:
            self._display = create_display(
                self._window_name,
                (800, 480),
                fullscreen=self.cfg.runtime.fullscreen,
                fps=self._fps,
                backend=self.cfg.runtime.display_backend,
            )
        self.hardware.set_leds(calibration=True, inference=False)
        self.hardware.announce_calibration_start(wait=True)
        self._cal_start_time = time.time()
        log.info("Starting calibration")

    def _run_calibration_loop(self) -> None:
        timeout = 30.0
        last_frame_time = 0.0

        while self._running and self.mode == KioskMode.CALIBRATING:
            now = time.time()
            if now - self._cal_start_time > timeout:
                used_samples = self.pipeline.calibrator.force_compute_or_defaults()
                if used_samples:
                    log.warning("Calibration timeout, using collected samples")
                else:
                    log.warning("Calibration timeout, using defaults")
                self.hardware.set_leds(calibration=False, inference=True)
                self.hardware.announce_calibration_complete()
                self.hardware.announce_infer_started()
                self._set_mode(KioskMode.INFERRING)
                return

            frame = self._grab_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            result = self.pipeline.process_frame(frame, now, self._frame_idx)
            self._record_profile()
            self._frame_idx += 1

            self._fps_times.append(now)

            if self.cfg.runtime.async_output:
                self._submit_output(frame, result, self.pipeline.calibration_progress)
            else:
                overlay = self._draw_kiosk_frame(frame, result)
                self._show_frame(overlay)
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

            if not self._poll_display():
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
            self._record_profile()
            self._frame_idx += 1
            self._latencies.append(result.latency_ms)
            self._risk_timeline.append({"timestamp": now, "risk_score": result.risk_score})
            self._events.extend(result.events)

            self._fps_times.append(now)

            self._handle_alerts(result)

            if self.cfg.runtime.async_output:
                self._submit_output(frame, result, None)
            else:
                overlay = self._draw_kiosk_frame(frame, result)
                self._show_frame(overlay)
                if self._writer:
                    self._writer.write(overlay)

            frame_count += 1
            if frame_count % 300 == 0:
                gc.collect()
            interval = max(0, self.cfg.runtime.profile_interval_frames)
            if interval and frame_count % interval == 0:
                self._log_profile()

            if not self._poll_display():
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
            self._show_frame(blank)
            if not self._poll_display(10):
                self._running = False

    def _run_error_loop(self) -> None:
        self.hardware.set_leds(calibration=False, inference=False)
        blank = np.zeros((self._frame_size[1], self._frame_size[0], 3), dtype=np.uint8)
        while self._running and self.mode == KioskMode.ERROR:
            cv2.putText(blank, "SYSTEM ERROR", (self._frame_size[0] // 4, self._frame_size[1] // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            cv2.putText(blank, "Check camera, then restart", (self._frame_size[0] // 5, self._frame_size[1] // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            self._show_frame(blank)
            if not self._poll_display(5):
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
        alert_events = [
            event for event in result.events
            if event.severity.value in ("warning", "critical")
        ]
        event_alert = bool(alert_events)
        alert_requested = result.risk_score >= thr or event_alert

        if alert_requested:
            if not self._alert_active:
                self._alert_start = now
                self._alert_active = True
                if alert_events:
                    self._maybe_queue_alert_voice(self._select_alert_event(alert_events), now)
                    self.hardware.queue_alert_beep()
            elif now - self._alert_start >= self.cfg.hardware.alert_stable_seconds:
                if now - self._last_alert_time >= cooldown:
                    self.hardware.set_alert(True)
                    self._last_alert_time = now
                    if alert_events:
                        selected = self._select_alert_event(alert_events)
                        self._maybe_queue_alert_voice(selected, now)
                        for event in alert_events:
                            log.warning("ALERT: %s (score=%.2f)", event.message, event.score)
        else:
            if self._alert_active:
                if now - self._alert_start >= self.cfg.hardware.alert_hold_seconds:
                    self._alert_active = False
                    self.hardware.set_alert(False)

    def _select_alert_event(self, events: list[DetectionEvent]) -> DetectionEvent:
        return max(
            events,
            key=lambda event: (
                self.ALERT_PRIORITY.get(event.signal, 0),
                1 if event.severity.value == "critical" else 0,
                event.score,
            ),
        )

    def _maybe_queue_alert_voice(self, event: DetectionEvent, now: float) -> bool:
        signal = event.signal
        priority = self.ALERT_PRIORITY.get(signal, 0)
        last_for_signal = self._signal_last_voice.get(signal, 0.0)
        repeat_cooldown = self.cfg.runtime.alert_voice_repeat_cooldown_seconds
        switch_cooldown = self.cfg.runtime.alert_voice_switch_cooldown_seconds
        global_cooldown = self.cfg.runtime.alert_voice_global_cooldown_seconds

        if last_for_signal and now - last_for_signal < repeat_cooldown:
            return False

        if self._last_voice_signal and self._last_voice_signal != signal:
            last_priority = self.ALERT_PRIORITY.get(self._last_voice_signal, 0)
            required_gap = switch_cooldown if priority > last_priority else global_cooldown
            if now - self._last_voice_time < required_gap:
                return False
        elif self._last_voice_time and now - self._last_voice_time < global_cooldown:
            return False

        if not self.hardware.queue_speech(event.message):
            return False
        self._last_voice_time = now
        self._last_voice_signal = signal
        self._signal_last_voice[signal] = now
        return True

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
        self.pipeline.close()
        if self._output is not None:
            self._output.close()
            self._output = None
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
        self._profile_sums = {}
        self._profile_count = 0
        self._last_voice_time = 0.0
        self._last_voice_signal = ""
        self._signal_last_voice = {}

    def _get_fps(self) -> float:
        if len(self._fps_times) < 2:
            return 0.0
        window = self._fps_times[-1] - self._fps_times[0]
        if window <= 0:
            return 0.0
        return (len(self._fps_times) - 1) / window

    def _draw_kiosk_frame(self, frame: np.ndarray, result) -> np.ndarray:
        canvas = embed_kiosk_overlay(
            result,
            frame,
            landmark_mode=self.cfg.vision.overlay_landmark_mode,
            debug=self.cfg.runtime.kiosk_debug,
        )
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

    def _ensure_output_worker(self) -> None:
        if self._output is not None:
            return
        self._output = AsyncFrameOutput(
            self.cfg,
            self._window_name,
            (800, 480),
            self._fps,
            writer_path=self._writer_path if self.cfg.runtime.write_video else None,
        )

    def _submit_output(
        self,
        frame: np.ndarray,
        result,
        calibration_progress: float | None,
    ) -> None:
        self._ensure_output_worker()
        if self._output is not None:
            self._output.submit(frame, result, calibration_progress)

    def _show_error_screen(self, msg: str) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "FATAL ERROR", (160, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        cv2.putText(blank, msg[:50], (80, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self._show_frame(blank)
        self._poll_display(60)

    def _shutdown(self, start_ts: str) -> None:
        log.info("Kiosk shutting down")
        self.hardware.set_alert(False)
        self.hardware.set_leds(calibration=False, inference=False)
        self.pipeline.close()

        if self._output is not None:
            self._output.close()
            self._output = None

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
        if self._display is not None:
            self._display.close()
        log.info("Kiosk shutdown complete")

    def _record_profile(self) -> None:
        profile = self.pipeline.last_profile
        if not profile:
            return
        self._profile_count += 1
        for key, value in profile.items():
            self._profile_sums[key] = self._profile_sums.get(key, 0.0) + value

    def _log_profile(self) -> None:
        if not self._profile_count:
            return
        avg = {
            key: value / self._profile_count
            for key, value in sorted(self._profile_sums.items())
        }
        log.info(
            "Profile avg over %d frames: %s",
            self._profile_count,
            ", ".join(f"{k}={v:.2f}ms" for k, v in avg.items()),
        )
        if self._output is not None:
            stats = self._output.snapshot_stats()
            rendered = max(1, stats.rendered)
            log.info(
                "Output avg over %d rendered/%d submitted frames: render=%.2fms display=%.2fms write=%.2fms dropped=%d",
                stats.rendered,
                stats.submitted,
                stats.render_ms / rendered,
                stats.display_ms / rendered,
                stats.write_ms / rendered,
                stats.dropped,
            )

    def _show_frame(self, frame: np.ndarray) -> None:
        if not self.cfg.runtime.display:
            return
        if self._display is not None:
            self._display.show(frame)
        else:
            cv2.imshow(self._window_name, frame)

    def _poll_display(self, fps: int = 60) -> bool:
        if not self.cfg.runtime.display:
            return True
        if self._output is not None:
            return self._output.is_running()
        if self._display is not None:
            return self._display.pump(fps)
        return (cv2.waitKey(max(1, int(1000 / max(1, fps)))) & 0xFF) != ord("q")
