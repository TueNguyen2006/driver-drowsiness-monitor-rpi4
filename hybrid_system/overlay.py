from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import DriverState, ProcessedFrame

# MediaPipe FaceMesh landmark indices for visualization
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_RIGHT_EYEBROW = [285, 295, 282, 283, 276, 300, 293, 334, 296, 336]
_MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0]
_MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
_NOSE = [1, 2, 98, 327, 5, 4, 195, 197, 6]

# EAR computation pairs (highlighted)
_EAR_LEFT = [33, 133, 160, 144, 159, 145, 158, 153]
_EAR_RIGHT = [263, 362, 387, 373, 386, 374, 385, 380]
_EAR_POINTS = set(_EAR_LEFT + _EAR_RIGHT)
_MAR_POINTS = {61, 291, 39, 181, 0, 17, 269, 405}

Array = NDArray[Any]

STATE_COLORS = {
    DriverState.ATTENTIVE: (76, 190, 118),
    DriverState.EYES_CLOSED: (36, 174, 222),
    DriverState.DROWSY: (38, 64, 230),
    DriverState.YAWNING: (36, 174, 222),
    DriverState.DISTRACTED: (42, 42, 238),
    DriverState.PHONE_USE: (42, 42, 238),
}

SIGNAL_BAR_LABELS = ["eyes_closed", "drowsy", "yawning", "distracted", "phone_use"]
SIGNAL_BAR_COLORS = {
    "eyes_closed": (36, 174, 222),
    "drowsy": (38, 64, 230),
    "yawning": (46, 167, 235),
    "distracted": (42, 42, 238),
    "phone_use": (235, 65, 55),
}


class AnnotatedVideoWriter:
    __slots__ = ("path", "writer")

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)

    def write(self, frame: Array) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def draw_overlay(
    processed: ProcessedFrame,
    frame: Array,
    *,
    draw_landmarks: bool = True,
    draw_pose_axes: bool = True,
) -> Array:
    h, w = frame.shape[:2]
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))

    _draw_panel_frame(frame, 12, 12, 328, 150)
    cv2.putText(
        frame, "AI DRIVER SAFETY", (28, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (246, 248, 248), 2, cv2.LINE_AA,
    )
    state_text = processed.state.value.replace("_", " ").upper()
    cv2.putText(
        frame, state_text, (28, 82),
        cv2.FONT_HERSHEY_SIMPLEX, 0.62, state_color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Risk {processed.risk_score:.2f}  {processed.latency_ms:.1f} ms", (28, 120),
        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (212, 220, 220), 1, cv2.LINE_AA,
    )

    if draw_landmarks and len(processed.landmarks) >= 468:
        _draw_face_landmarks(frame, processed.landmarks)

    if processed.head_pose is not None and draw_pose_axes and len(processed.landmarks) >= 468:
        pitch, yaw, roll = processed.head_pose
        nose = processed.landmarks[1]
        _draw_axes_inplace(frame, pitch, yaw, roll, int(nose[0]), int(nose[1]), size=40)

    for obj in processed.objects:
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (55, 65, 235), 2)
        cv2.putText(
            frame, str(obj["label"]), (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (55, 65, 235), 2, cv2.LINE_AA,
        )

    _draw_signal_bars(frame, processed.signals, w, h)

    for idx, event in enumerate(processed.events[:3]):
        ey = h - 90 + idx * 24
        cv2.putText(
            frame, event.message, (24, ey),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA,
        )

    return frame


def _draw_panel_frame(
    frame: Array, x: int, y: int, w: int, h: int, alpha: float = 0.72
) -> None:
    roi = frame[y : y + h, x : x + w]
    overlay = np.full(roi.shape, (18, 22, 23), dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def _draw_signal_bars(frame: Array, signals: dict[str, float], width: int, height: int) -> None:
    start_x = max(20, width - 248)
    start_y = 24
    _draw_panel_frame(frame, start_x - 12, start_y - 12, 236, 168, alpha=0.68)

    for idx, label in enumerate(SIGNAL_BAR_LABELS):
        value = min(1.0, max(0.0, signals.get(label, 0.0)))
        y = start_y + idx * 28
        cv2.putText(
            frame, label.replace("_", " "), (start_x, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (216, 222, 222), 1, cv2.LINE_AA,
        )
        x0, x1 = start_x + 106, start_x + 206
        bar_h = 12
        cv2.rectangle(frame, (x0, y), (x1, y + bar_h), (60, 66, 68), -1)
        bar_color = SIGNAL_BAR_COLORS.get(label, (62, 197, 124))
        filled = int(100 * value)
        if filled > 0:
            cv2.rectangle(frame, (x0, y), (x0 + filled, y + bar_h), bar_color, -1)


def _draw_axes_inplace(
    img: Array, pitch: float, yaw: float, roll: float,
    tx: int, ty: int, size: int = 50,
) -> None:
    yaw = -yaw
    rmat = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float64))[0]
    axes = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    axes = rmat @ axes
    pts = (axes[:2, :] * size).astype(int)
    pts[0, :] += tx
    pts[1, :] += ty
    origin = (int(pts[0, 3]), int(pts[1, 3]))
    cv2.line(img, origin, (int(pts[0, 0]), int(pts[1, 0])), (255, 0, 0), 3)
    cv2.line(img, origin, (int(pts[0, 1]), int(pts[1, 1])), (0, 255, 0), 3)
    cv2.line(img, origin, (int(pts[0, 2]), int(pts[1, 2])), (0, 0, 255), 3)


def _draw_face_landmarks(frame: Array, landmarks: list[tuple[float, float]]) -> None:
    pts_list = [(int(x), int(y)) for x, y in landmarks]

    def _draw_group(indices, color, radius=1, thickness=-1):
        for idx in indices:
            if idx < len(pts_list):
                cv2.circle(frame, pts_list[idx], radius, color, thickness)

    def _draw_contour(indices, color, thickness=1):
        for i in range(len(indices) - 1):
            if indices[i] < len(pts_list) and indices[i + 1] < len(pts_list):
                cv2.line(frame, pts_list[indices[i]], pts_list[indices[i + 1]], color, thickness)

    # Face oval (green)
    _draw_group(_FACE_OVAL, (120, 200, 120), radius=1)
    _draw_contour(_FACE_OVAL, (100, 180, 100), thickness=1)

    # Eyebrows (gray)
    _draw_group(_LEFT_EYEBROW, (150, 150, 150), radius=1)
    _draw_group(_RIGHT_EYEBROW, (150, 150, 150), radius=1)

    # Nose (warm yellow)
    _draw_contour(_NOSE, (180, 180, 100), thickness=1)
    _draw_group(_NOSE, (200, 200, 120), radius=1)

    # Eyes (cyan)
    _draw_contour(_LEFT_EYE, (120, 220, 240), thickness=1)
    _draw_contour(_RIGHT_EYE, (120, 220, 240), thickness=1)
    _draw_group(_LEFT_EYE, (160, 235, 250), radius=1)
    _draw_group(_RIGHT_EYE, (160, 235, 250), radius=1)

    # EAR computation points (small cyan dots)
    for idx in _EAR_LEFT + _EAR_RIGHT:
        if idx < len(pts_list):
            cv2.circle(frame, pts_list[idx], 2, (0, 255, 255), -1)
            cv2.circle(frame, pts_list[idx], 3, (255, 255, 255), 1)

    # Mouth (magenta)
    _draw_contour(_MOUTH_OUTER, (200, 100, 160), thickness=1)
    _draw_contour(_MOUTH_INNER, (200, 100, 160), thickness=1)
    _draw_group(_MOUTH_OUTER, (220, 120, 180), radius=1)

    # MAR computation points (small magenta dots)
    for idx in _MAR_POINTS:
        if idx < len(pts_list):
            cv2.circle(frame, pts_list[idx], 2, (255, 0, 255), -1)
            cv2.circle(frame, pts_list[idx], 3, (255, 255, 255), 1)
