from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .config import HybridConfig
from .display import create_display
from .models import ProcessedFrame
from .overlay import AnnotatedVideoWriter
from .ui import embed_kiosk_overlay

log = logging.getLogger("kiosk.output")


@dataclass(slots=True)
class FrameOutputItem:
    frame: np.ndarray
    result: ProcessedFrame
    calibration_progress: float | None = None


@dataclass(slots=True)
class FrameOutputStats:
    submitted: int = 0
    rendered: int = 0
    dropped: int = 0
    render_ms: float = 0.0
    display_ms: float = 0.0
    write_ms: float = 0.0


class LatestFrameSlot:
    """Single-item handoff: keep newest frame, drop stale work."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._item: FrameOutputItem | None = None
        self.submitted = 0
        self.dropped = 0

    def put(self, item: FrameOutputItem) -> None:
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self.submitted += 1
        self._event.set()

    def get(self, timeout: float = 0.1) -> FrameOutputItem | None:
        if not self._event.wait(timeout=timeout):
            return None
        self._event.clear()
        with self._lock:
            item = self._item
            self._item = None
        return item


class AsyncFrameOutput:
    """Render/display/write kiosk frames outside the inference loop."""

    def __init__(
        self,
        cfg: HybridConfig,
        title: str,
        size: tuple[int, int],
        fps: float,
        writer_path: Path | None = None,
        render_fn: Callable[[np.ndarray, ProcessedFrame, float | None], np.ndarray] | None = None,
    ) -> None:
        self._cfg = cfg
        self._title = title
        self._size = size
        self._fps = fps
        self._writer_path = writer_path
        self._render_fn = render_fn or self._render_default
        self._slot = LatestFrameSlot()
        self._stop = threading.Event()
        self._closed = threading.Event()
        self._display = None
        self._writer: AnnotatedVideoWriter | None = None
        self._stats = FrameOutputStats()
        self._thread = threading.Thread(target=self._loop, name="kiosk-output", daemon=True)
        self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        result: ProcessedFrame,
        calibration_progress: float | None = None,
    ) -> None:
        self._slot.put(FrameOutputItem(frame.copy(), result, calibration_progress))

    def is_running(self) -> bool:
        return not self._closed.is_set()

    def snapshot_stats(self) -> FrameOutputStats:
        stats = FrameOutputStats(
            submitted=self._slot.submitted,
            rendered=self._stats.rendered,
            dropped=self._slot.dropped,
            render_ms=self._stats.render_ms,
            display_ms=self._stats.display_ms,
            write_ms=self._stats.write_ms,
        )
        return stats

    def close(self) -> None:
        self._stop.set()
        self._slot._event.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        try:
            if self._cfg.runtime.display:
                self._display = create_display(
                    self._title,
                    self._size,
                    fullscreen=self._cfg.runtime.fullscreen,
                    fps=self._fps,
                    backend=self._cfg.runtime.display_backend,
                )
            if self._writer_path is not None:
                self._writer = AnnotatedVideoWriter(self._writer_path, self._fps, self._size)

            while not self._stop.is_set():
                item = self._slot.get(timeout=0.1)
                if item is None:
                    if self._display is not None and not self._display.pump(60):
                        break
                    continue

                started = perf_counter()
                overlay = self._render_fn(
                    item.frame,
                    item.result,
                    item.calibration_progress,
                )
                self._stats.render_ms += (perf_counter() - started) * 1000

                started = perf_counter()
                if self._display is not None:
                    self._display.show(overlay)
                    if not self._display.pump(60):
                        break
                self._stats.display_ms += (perf_counter() - started) * 1000

                started = perf_counter()
                if self._writer is not None:
                    self._writer.write(overlay)
                self._stats.write_ms += (perf_counter() - started) * 1000
                self._stats.rendered += 1
        except Exception as exc:
            log.exception("Async frame output failed: %s", exc)
        finally:
            if self._writer is not None:
                self._writer.close()
            if self._display is not None:
                self._display.close()
            self._closed.set()

    def _render_default(
        self,
        frame: np.ndarray,
        result: ProcessedFrame,
        calibration_progress: float | None,
    ) -> np.ndarray:
        canvas = embed_kiosk_overlay(
            result,
            frame,
            landmark_mode=self._cfg.vision.overlay_landmark_mode,
            debug=self._cfg.runtime.kiosk_debug,
        )
        if calibration_progress is not None:
            prog = max(0.0, min(1.0, calibration_progress))
            bar_w = 200
            bar_h = 16
            bx = (self._size[0] - bar_w) // 2
            by = 6
            cv2.rectangle(canvas, (bx, by), (bx + bar_w, by + bar_h), (30, 36, 38), -1)
            fill = int(bar_w * prog)
            if fill > 0:
                cv2.rectangle(canvas, (bx, by), (bx + fill, by + bar_h), (62, 197, 124), -1)
            cv2.putText(
                canvas,
                f"CALIBRATING {prog:.0%}",
                (bx + 12, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas
