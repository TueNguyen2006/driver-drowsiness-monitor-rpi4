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
_LEFT_EYEBROW_UPPER = [70, 63, 105, 66, 107]
_LEFT_EYEBROW_LOWER = [46, 53, 52, 65, 55]
_RIGHT_EYEBROW_UPPER = [336, 296, 334, 293, 300]
_RIGHT_EYEBROW_LOWER = [285, 295, 282, 283, 276]
_UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
_UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
_LOWER_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
_LOWER_LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
_NOSE = [1, 2, 98, 327, 5, 4, 195, 197, 6]
_NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1, 2]
_NOSE_LEFT_SIDE = [193, 122, 196, 236, 198, 209, 49, 48]
_NOSE_RIGHT_SIDE = [417, 351, 419, 456, 420, 429, 279, 278]
_NOSE_UNDER = [48, 115, 220, 45, 4, 275, 440, 344, 278]
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

# Kiosk 800x480 layout constants.
DISPLAY_W = 800
DISPLAY_H = 480
CAM_W = DISPLAY_W
CAM_H = DISPLAY_H
PANEL_W = DISPLAY_W // 3
PANEL_H = DISPLAY_H // 2
PANEL_X = 18
PANEL_Y = DISPLAY_H - PANEL_H - 18
FONT_LARGE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX
LANDMARK_LINE_THICKNESS = 1


def embed_kiosk_overlay(
    processed: ProcessedFrame,
    frame: Array,
    landmark_mode: str = "minimal",
    debug: bool = False,
) -> Array:
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    canvas[:, :] = (9, 15, 24)
    state_color = STATE_COLORS.get(processed.state, (255, 255, 255))

    # --- Camera area (full 800x480) ---
    h, w = frame.shape[:2]
    scale = max(CAM_W / w, CAM_H / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    offset_x = (CAM_W - nw) // 2
    offset_y = (CAM_H - nh) // 2
    canvas[:, :] = (2, 8, 14)
    src_x = max(0, -offset_x)
    src_y = max(0, -offset_y)
    dst_x = max(0, offset_x)
    dst_y = max(0, offset_y)
    copy_w = min(CAM_W - dst_x, nw - src_x)
    copy_h = min(CAM_H - dst_y, nh - src_y)
    if copy_w > 0 and copy_h > 0:
        canvas[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = resized[src_y:src_y + copy_h, src_x:src_x + copy_w]

    landmark_mode = (landmark_mode or "minimal").lower()
    if landmark_mode != "off" and len(processed.landmarks) >= 468:
        _draw_professional_face_overlay(
            canvas,
            processed,
            scale,
            offset_x,
            offset_y,
            state_color,
            landmark_mode,
        )

    if debug:
        _draw_debug_overlay(canvas, processed)

    _draw_status_card(canvas, processed, state_color)

    return canvas


def _draw_status_card(canvas: Array, processed: ProcessedFrame, state_color: tuple[int, int, int]) -> None:
    # Render the compact UI at 2x then downsample. OpenCV's Hershey text is
    # raster-only, so supersampling is the simplest way to avoid jagged text.
    scale = 2
    panel_x, panel_y = PANEL_X, PANEL_Y
    panel_w, panel_h = PANEL_W, PANEL_H
    card = cv2.resize(
        canvas[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w],
        (panel_w * scale, panel_h * scale),
        interpolation=cv2.INTER_LINEAR,
    )

    def sx(v: int) -> int:
        return int(v * scale)

    state_color_s = tuple(int(v) for v in state_color)
    _draw_panel_bg(card, 0, 0, sx(panel_w), sx(panel_h), alpha=0.60, radius=sx(14))
    _draw_rounded_rect(card, 0, 0, sx(panel_w) - 1, sx(panel_h) - 1, (10, 18, 28), sx(3), radius=sx(14))
    _draw_rounded_rect(card, sx(3), sx(3), sx(panel_w - 6), sx(panel_h - 6), state_color_s, sx(1), radius=sx(11))

    # State icon + text
    state_text = processed.state.value.replace("_", " ").upper()
    cv2.putText(card, state_text, (sx(16), sx(36)),
                FONT_LARGE, 0.58 * scale, state_color_s, sx(2), cv2.LINE_AA)
    cv2.putText(card, "DRIVER MONITOR", (sx(17), sx(58)),
                FONT_SMALL, 0.34 * scale, (176, 196, 208), sx(1), cv2.LINE_AA)

    # Risk bar, no percentage text.
    bar_x = 16
    bar_y = 74
    bar_w = panel_w - 32
    bar_h = 13
    _draw_smooth_rounded_rect(card, sx(bar_x), sx(bar_y), sx(bar_w), sx(bar_h), (38, 50, 62), -1, radius=sx(6))
    filled = int(bar_w * min(1.0, processed.risk_score))
    if filled > 0:
        _draw_smooth_rounded_rect(card, sx(bar_x), sx(bar_y), sx(filled), sx(bar_h), state_color_s, -1, radius=sx(6))
    _draw_smooth_rounded_rect(card, sx(bar_x), sx(bar_y), sx(bar_w), sx(bar_h), (90, 108, 124), sx(1), radius=sx(6))

    # Signal bars (compact but readable)
    signals = processed.signals
    sig_y = 105
    sig_labels = ["drowsy", "eyes_closed", "yawning", "distracted", "phone_use"]
    for idx, label in enumerate(sig_labels):
        y = sig_y + idx * 24
        value = min(1.0, max(0.0, signals.get(label, 0.0)))
        c = SIGNAL_COLORS.get(label, (62, 197, 124))

        cv2.putText(card, label.replace("_", " "), (sx(16), sx(y + 12)),
                    FONT_SMALL, 0.34 * scale, (218, 228, 232), sx(1), cv2.LINE_AA)

        bx = 110
        bw = panel_w - 126
        bh = 11
        _draw_smooth_rounded_rect(card, sx(bx), sx(y + 3), sx(bw), sx(bh), (38, 50, 62), -1, radius=sx(5))
        fw2 = int(bw * value)
        if fw2 > 0:
            _draw_smooth_rounded_rect(card, sx(bx), sx(y + 3), sx(fw2), sx(bh), c, -1, radius=sx(5))
        _draw_smooth_rounded_rect(card, sx(bx), sx(y + 3), sx(bw), sx(bh), (88, 104, 118), sx(1), radius=sx(5))

    canvas[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = cv2.resize(
        card,
        (panel_w, panel_h),
        interpolation=cv2.INTER_AREA,
    )


def _draw_panel_bg(canvas: Array, x: int, y: int, w: int, h: int,
                   alpha: float = 0.60, radius: int = 12) -> None:
    roi = canvas[y:y + h, x:x + w]
    overlay = np.full(roi.shape, (30, 42, 58), dtype=np.uint8)
    overlay[:, :, 0] = np.linspace(42, 22, w, dtype=np.uint8)
    overlay[:, :, 1] = np.linspace(58, 34, w, dtype=np.uint8)
    overlay[:, :, 2] = np.linspace(78, 48, w, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    _draw_rounded_rect(mask, 0, 0, w, h, 255, -1, radius=radius)
    blended = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)
    roi[mask > 0] = blended[mask > 0]


def _draw_rounded_rect(
    image: Array,
    x: int,
    y: int,
    w: int,
    h: int,
    color,
    thickness: int = 1,
    radius: int = 8,
) -> None:
    if w <= 0 or h <= 0:
        return
    radius = max(0, min(radius, w // 2, h // 2))
    x2, y2 = x + w, y + h
    if thickness < 0:
        cv2.rectangle(image, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x, y + radius), (x2, y2 - radius), color, -1)
        cv2.circle(image, (x + radius, y + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y + radius), radius, color, -1)
        cv2.circle(image, (x + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
        return
    cv2.line(image, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(image, (x + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_smooth_rounded_rect(
    image: Array,
    x: int,
    y: int,
    w: int,
    h: int,
    color,
    thickness: int = 1,
    radius: int = 8,
    scale: int = 3,
) -> None:
    if w <= 0 or h <= 0:
        return
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return
    roi = image[y:y + h, x:x + w]
    large = cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
    draw_thickness = -1 if thickness < 0 else max(1, thickness * scale)
    _draw_rounded_rect(
        large,
        0,
        0,
        w * scale - 1,
        h * scale - 1,
        color,
        draw_thickness,
        radius=max(1, radius * scale),
    )
    image[y:y + h, x:x + w] = cv2.resize(large, (w, h), interpolation=cv2.INTER_AREA)


def _draw_professional_face_overlay(
    canvas: Array,
    processed: ProcessedFrame,
    scale: float,
    offset_x: int,
    offset_y: int,
    state_color: tuple[int, int, int],
    landmark_mode: str,
) -> None:
    pts = [
        (int(x * scale + offset_x), int(y * scale + offset_y))
        for x, y in processed.landmarks
    ]

    del state_color
    mesh_color = (182, 205, 218)
    subtle_color = (118, 132, 122)
    feature_color = (238, 220, 96)
    mouth_color = (190, 170, 235)
    nose_color = (176, 196, 222)

    line_t = LANDMARK_LINE_THICKNESS
    _draw_contour(canvas, pts, _FACE_OVAL, mesh_color, line_t)
    _draw_contour(canvas, pts, _LEFT_EYEBROW_UPPER, subtle_color, line_t)
    _draw_contour(canvas, pts, _LEFT_EYEBROW_LOWER, subtle_color, line_t)
    _draw_contour(canvas, pts, _RIGHT_EYEBROW_UPPER, subtle_color, line_t)
    _draw_contour(canvas, pts, _RIGHT_EYEBROW_LOWER, subtle_color, line_t)
    _draw_contour(canvas, pts, _NOSE_LEFT_SIDE, nose_color, line_t)
    _draw_contour(canvas, pts, _NOSE_RIGHT_SIDE, nose_color, line_t)
    _draw_contour(canvas, pts, _NOSE_UNDER, nose_color, line_t)
    _draw_contour(canvas, pts, _LEFT_EYE, feature_color, line_t)
    _draw_contour(canvas, pts, _RIGHT_EYE, feature_color, line_t)
    _draw_contour(canvas, pts, _UPPER_LIP_OUTER, mouth_color, line_t)
    _draw_contour(canvas, pts, _UPPER_LIP_INNER, mouth_color, line_t)
    _draw_contour(canvas, pts, _LOWER_LIP_INNER, mouth_color, line_t)
    _draw_contour(canvas, pts, _LOWER_LIP_OUTER, mouth_color, line_t)

    if landmark_mode == "full":
        full_points = set(
            _FACE_OVAL
            + _LEFT_EYE
            + _RIGHT_EYE
            + _UPPER_LIP_OUTER
            + _UPPER_LIP_INNER
            + _LOWER_LIP_OUTER
            + _LOWER_LIP_INNER
            + _NOSE_BRIDGE
            + _NOSE_LEFT_SIDE
            + _NOSE_RIGHT_SIDE
            + _NOSE_UNDER
            + _LEFT_EYEBROW_UPPER
            + _LEFT_EYEBROW_LOWER
            + _RIGHT_EYEBROW_UPPER
            + _RIGHT_EYEBROW_LOWER
        )
        for idx in full_points:
            _draw_point(canvas, pts, idx, (220, 222, 228), 1)

    for idx in _EAR_LEFT + _EAR_RIGHT:
        _draw_point(canvas, pts, idx, (255, 242, 120), 1)
    for idx in _MAR_POINTS:
        _draw_point(canvas, pts, idx, (210, 190, 245), 1)

    _draw_head_pose_axes(canvas, processed, pts)


def _draw_head_pose_axes(canvas: Array, processed: ProcessedFrame, pts: list[tuple[int, int]]) -> None:
    if len(pts) <= 1:
        return
    pose = processed.head_pose
    if not (isinstance(pose, (list, tuple)) and len(pose) >= 3):
        pose = processed.debug_info.get("head_pose") if processed.debug_info else None
    if not (isinstance(pose, (list, tuple)) and len(pose) >= 3):
        return

    try:
        pitch = float(pose[0])
        yaw = float(pose[1])
        roll = float(pose[2])
    except (TypeError, ValueError):
        return

    origin = pts[1]
    yaw = -yaw
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll], dtype=np.float64))[0]
    axes_points = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=np.float64,
    )
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * 44).astype(int)
    axes_points[0, :] += origin[0]
    axes_points[1, :] += origin[1]
    axis_origin = tuple(axes_points[:, 3].ravel())

    cv2.circle(canvas, origin, 3, (12, 18, 24), -1, cv2.LINE_AA)
    cv2.line(canvas, axis_origin, tuple(axes_points[:, 0].ravel()), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, axis_origin, tuple(axes_points[:, 1].ravel()), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, axis_origin, tuple(axes_points[:, 2].ravel()), (0, 0, 255), 2, cv2.LINE_AA)


def _draw_contour(
    canvas: Array,
    pts: list[tuple[int, int]],
    indices: list[int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    for i in range(len(indices) - 1):
        _draw_segment(canvas, pts, indices[i], indices[i + 1], color, thickness)


def _draw_segment(
    canvas: Array,
    pts: list[tuple[int, int]],
    i1: int,
    i2: int,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if i1 < len(pts) and i2 < len(pts):
        overlay = canvas.copy()
        cv2.line(overlay, pts[i1], pts[i2], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)


def _draw_point(
    canvas: Array,
    pts: list[tuple[int, int]],
    idx: int,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    if idx < len(pts):
        cv2.circle(canvas, pts[idx], radius, color, -1, cv2.LINE_AA)


def _debug_number(value: Any, precision: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _draw_debug_overlay(canvas: Array, processed: ProcessedFrame) -> None:
    info = processed.debug_info or {}
    hp = info.get("head_pose")
    if isinstance(hp, (list, tuple)) and len(hp) >= 3:
        hp_text = (
            f"P:{_debug_number(hp[0])} "
            f"Y:{_debug_number(hp[1])} "
            f"R:{_debug_number(hp[2])}"
        )
    else:
        hp_text = "P:n/a Y:n/a R:n/a"

    lstm_label = info.get("lstm_label")
    if lstm_label is None:
        lstm_text = "n/a"
    else:
        lstm_text = f"{lstm_label} cnt:{info.get('lstm_decision_count', 0)}"

    lines = [
        "DEBUG",
        f"EARn: {_debug_number(info.get('ear_smoothed', info.get('ear_norm')))}",
        f"MARn: {_debug_number(info.get('mar_smoothed', info.get('mar_norm')))}",
        f"PUCn: {_debug_number(info.get('puc_smoothed', info.get('puc_norm')))}",
        f"MoEn: {_debug_number(info.get('moe_smoothed', info.get('moe_norm')))}",
        f"Head: {hp_text}",
        f"LSTM: {lstm_text}",
    ]

    x, y = 10, 276
    w, h = 342, 190
    roi = canvas[y:y + h, x:x + w]
    overlay = np.full(roi.shape, (8, 18, 28), dtype=np.uint8)
    cv2.addWeighted(overlay, 0.72, roi, 0.28, 0, roi)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (54, 190, 210), 1)
    for idx, line in enumerate(lines):
        color = (90, 220, 245) if idx == 0 else (232, 244, 248)
        cv2.putText(canvas, line, (x + 10, y + 24 + idx * 24),
                    FONT_SMALL, 0.48, color, 1, cv2.LINE_AA)


def draw_fullscreen_overlay(
    processed: ProcessedFrame,
    frame: Array,
    display_w: int = 800,
    display_h: int = 480,
    debug: bool = False,
) -> Array:
    return embed_kiosk_overlay(processed, frame, debug=debug)
