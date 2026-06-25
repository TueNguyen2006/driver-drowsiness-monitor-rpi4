from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]

LEFT_EYE_IDXS = [[33, 133], [160, 144], [159, 145], [158, 153]]
RIGHT_EYE_IDXS = [[263, 362], [387, 373], [386, 374], [385, 380]]
MOUTH_IDXS = [[61, 291], [39, 181], [0, 17], [269, 405]]

# flake8: noqa: E201, E202
_EYE_L1 = (33, 133)
_EYE_L2 = (160, 144)
_EYE_L3 = (159, 145)
_EYE_L4 = (158, 153)
_EYE_R1 = (263, 362)
_EYE_R2 = (387, 373)
_EYE_R3 = (386, 374)
_EYE_R4 = (385, 380)
_MTH1 = (61, 291)
_MTH2 = (39, 181)
_MTH3 = (0, 17)
_MTH4 = (269, 405)

_1_3 = 1.0 / 3.0


def _dist(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _ear_for(pts: Sequence[Point], i1, i2, i3, i4) -> float:
    n1 = _dist(pts[i2[0]], pts[i2[1]])
    n2 = _dist(pts[i3[0]], pts[i3[1]])
    n3 = _dist(pts[i4[0]], pts[i4[1]])
    d = _dist(pts[i1[0]], pts[i1[1]])
    return (n1 + n2 + n3) * _1_3 / d if d != 0 else 0.0


def eye_feature(landmarks: Sequence[Point]) -> float:
    left = _ear_for(landmarks, _EYE_L1, _EYE_L2, _EYE_L3, _EYE_L4)
    right = _ear_for(landmarks, _EYE_R1, _EYE_R2, _EYE_R3, _EYE_R4)
    return (left + right) * 0.5


def mouth_aspect_ratio(landmarks: Sequence[Point]) -> float:
    return _ear_for(landmarks, _MTH1, _MTH2, _MTH3, _MTH4)


def mouth_feature(landmarks: Sequence[Point]) -> float:
    return mouth_aspect_ratio(landmarks)


def pupil_circularity(landmarks: Sequence[Point], i1, i2, i3, i4) -> float:
    perimeter = (
        _dist(landmarks[i1[0]], landmarks[i2[0]]) +
        _dist(landmarks[i2[0]], landmarks[i3[0]]) +
        _dist(landmarks[i3[0]], landmarks[i4[0]]) +
        _dist(landmarks[i4[0]], landmarks[i1[1]]) +
        _dist(landmarks[i1[1]], landmarks[i4[1]]) +
        _dist(landmarks[i4[1]], landmarks[i3[1]]) +
        _dist(landmarks[i3[1]], landmarks[i2[1]]) +
        _dist(landmarks[i2[1]], landmarks[i1[0]])
    )
    d = _dist(landmarks[i2[0]], landmarks[i4[1]]) * 0.5
    area = math.pi * (d * d)
    if perimeter == 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter * perimeter)


def pupil_feature(landmarks: Sequence[Point]) -> float:
    left = pupil_circularity(landmarks, _EYE_L1, _EYE_L2, _EYE_L3, _EYE_L4)
    right = pupil_circularity(landmarks, _EYE_R1, _EYE_R2, _EYE_R3, _EYE_R4)
    return (left + right) * 0.5


def extract_all(landmarks: Sequence[Point]) -> dict[str, float]:
    ear = eye_feature(landmarks)
    mar = mouth_feature(landmarks)
    puc = pupil_feature(landmarks)
    moe = mar / ear if ear != 0 else 0.0
    return {"ear": ear, "mar": mar, "puc": puc, "moe": moe}
