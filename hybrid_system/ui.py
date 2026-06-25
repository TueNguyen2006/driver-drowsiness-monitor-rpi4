from __future__ import annotations

from typing import Any

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
_EAR_LEFT = [33, 133, 160, 144, 159, 145, 158, 153]
_EAR_RIGHT = [263, 362, 387, 373, 386, 374, 385, 380]
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

SIGNAL_COLORS = {
    "eyes_closed": (36, 174, 222),
    "drowsy": (38, 64, 230),
    "yawning": (46, 167, 235),
    "distracted": (42, 42, 238),
    "phone_use": (235, 65, 55),
}

# Kiosk 800x480 layout constants
CAM_W = 520
CAM_H = 480
PANEL_X = 528
PANEL_W = 268
PANEL_H = 480
FONT_LARGE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX


def embed_kiosk_overlay(
    processed: ProcessedFrame,
    frame: Array,
) -> Array:
    canvas = np.zeros((CAM_H, 800, 3), dtype=np.uint8)
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))

    # --- Camera area (left, 520x480) fill to edge ---
    h, w = frame.shape[:2]
    scale = max(CAM_W / w, CAM_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    crop_x = (nw - CAM_W) // 2
    crop_y = (nh - CAM_H) // 2
    canvas[:, :CAM_W] = resized[crop_y:crop_y + CAM_H, crop_x:crop_x + CAM_W]

    if len(processed.landmarks) >= 468:
        def _draw_lm(indices, color, radius=1):
            for idx in indices:
                if idx < len(processed.landmarks):
                    px = int(processed.landmarks[idx][0] * scale - crop_x)
                    py = int(processed.landmarks[idx][1] * scale - crop_y)
                    cv2.circle(canvas, (px, py), radius, color, -1)

        def _draw_lm_contour(indices, color, thickness=1):
            for i in range(len(indices) - 1):
                i1, i2 = indices[i], indices[i + 1]
                if i1 < len(processed.landmarks) and i2 < len(processed.landmarks):
                    p1 = (int(processed.landmarks[i1][0] * scale - crop_x), int(processed.landmarks[i1][1] * scale - crop_y))
                    p2 = (int(processed.landmarks[i2][0] * scale - crop_x), int(processed.landmarks[i2][1] * scale - crop_y))
                    cv2.line(canvas, p1, p2, color, thickness)

        # Face oval (green)
        _draw_lm_contour(_FACE_OVAL, (100, 180, 100), thickness=1)
        _draw_lm(_FACE_OVAL, (120, 200, 120), radius=1)
        # Eyebrows
        _draw_lm(_LEFT_EYEBROW, (150, 150, 150), radius=1)
        _draw_lm(_RIGHT_EYEBROW, (150, 150, 150), radius=1)
        # Nose
        _draw_lm_contour(_NOSE, (180, 180, 100), thickness=1)
        _draw_lm(_NOSE, (200, 200, 120), radius=1)
        # Eyes
        _draw_lm_contour(_LEFT_EYE, (120, 220, 240), thickness=1)
        _draw_lm_contour(_RIGHT_EYE, (120, 220, 240), thickness=1)
        _draw_lm(_LEFT_EYE, (160, 235, 250), radius=1)
        _draw_lm(_RIGHT_EYE, (160, 235, 250), radius=1)
        # EAR points (small cyan dots)
        for idx in _EAR_LEFT + _EAR_RIGHT:
            if idx < len(processed.landmarks):
                px = int(processed.landmarks[idx][0] * scale - crop_x)
                py = int(processed.landmarks[idx][1] * scale - crop_y)
                cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1)
                cv2.circle(canvas, (px, py), 3, (255, 255, 255), 1)
        # Mouth
        _draw_lm_contour(_MOUTH_OUTER, (200, 100, 160), thickness=1)
        _draw_lm_contour(_MOUTH_INNER, (200, 100, 160), thickness=1)
        _draw_lm(_MOUTH_OUTER, (220, 120, 180), radius=1)
        # MAR points (small magenta dots)
        for idx in _MAR_POINTS:
            if idx < len(processed.landmarks):
                px = int(processed.landmarks[idx][0] * scale - crop_x)
                py = int(processed.landmarks[idx][1] * scale - crop_y)
                cv2.circle(canvas, (px, py), 2, (255, 0, 255), -1)
                cv2.circle(canvas, (px, py), 3, (255, 255, 255), 1)

    # --- Status panel (right, 268x480) ---
    _draw_panel_bg(canvas, PANEL_X, 0, PANEL_W, PANEL_H)

    # State icon + text
    state_text = processed.state.value.replace("_", " ").upper()
    cv2.putText(canvas, state_text, (PANEL_X + 12, 52),
                FONT_LARGE, 1.1, state_color, 3, cv2.LINE_AA)

    # Risk score (big)
    cv2.putText(canvas, "RISK", (PANEL_X + 12, 100),
                FONT_SMALL, 0.55, (160, 170, 170), 1, cv2.LINE_AA)
    risk_str = f"{processed.risk_score:.0%}"
    cv2.putText(canvas, risk_str, (PANEL_X + 12, 158),
                FONT_LARGE, 1.9, (255, 255, 255), 3, cv2.LINE_AA)

    # Risk bar
    bar_x = PANEL_X + 12
    bar_y = 170
    bar_w = PANEL_W - 28
    bar_h = 14
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 66, 68), -1)
    filled = int(bar_w * min(1.0, processed.risk_score))
    if filled > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), state_color, -1)

    # Signal bars (compact but readable)
    signals = processed.signals
    sig_y = 208
    sig_labels = ["drowsy", "eyes_closed", "yawning", "distracted", "phone_use"]
    for idx, label in enumerate(sig_labels):
        y = sig_y + idx * 44
        value = min(1.0, max(0.0, signals.get(label, 0.0)))
        c = SIGNAL_COLORS.get(label, (62, 197, 124))

        cv2.putText(canvas, label.replace("_", " "), (PANEL_X + 12, y + 14),
                    FONT_SMALL, 0.45, (200, 210, 210), 1, cv2.LINE_AA)

        # Bar + percentage (text inside bar, right-aligned)
        bx = PANEL_X + 120
        bw = PANEL_W - 132
        cv2.rectangle(canvas, (bx, y + 2), (bx + bw, y + 18), (50, 56, 58), -1)
        fw2 = int(bw * value)
        if fw2 > 0:
            cv2.rectangle(canvas, (bx, y + 2), (bx + fw2, y + 18), c, -1)

        pct = f"{value:.0%}"
        (tw, _), _ = cv2.getTextSize(pct, FONT_SMALL, 0.4, 1)
        pct_x = bx + bw - tw - 4
        if pct_x < bx + 4:
            pct_x = bx + 4
        cv2.putText(canvas, pct, (pct_x, y + 16),
                    FONT_SMALL, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Events ticker (max 1)
    if processed.events:
        ev = processed.events[-1]
        ev_color = (255, 80, 80) if ev.severity.value == "critical" else (255, 200, 80)
        cv2.putText(canvas, ev.message[:32], (PANEL_X + 12, PANEL_H - 32),
                    FONT_SMALL, 0.4, ev_color, 1, cv2.LINE_AA)

    # Separator line
    cv2.line(canvas, (PANEL_X - 2, 0), (PANEL_X - 2, PANEL_H), (50, 56, 58), 2)

    return canvas


def _draw_panel_bg(canvas: Array, x: int, y: int, w: int, h: int,
                   alpha: float = 0.88) -> None:
    roi = canvas[y:y + h, x:x + w]
    overlay = np.full(roi.shape, (14, 18, 20), dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)


def draw_fullscreen_overlay(
    processed: ProcessedFrame,
    frame: Array,
    display_w: int = 800,
    display_h: int = 480,
) -> Array:
    return embed_kiosk_overlay(processed, frame)
