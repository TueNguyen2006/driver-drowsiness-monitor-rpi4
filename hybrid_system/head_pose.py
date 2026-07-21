from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import cv2
import numpy as np


class HeadPoseEstimator:
    __slots__ = ("model", "available")

    FOREHEAD = 10
    NOSE = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 199
    LEFT_EYE = 33
    RIGHT_EYE = 263
    # Must match the training notebook column order:
    # nose, forehead, left_eye, mouth_left, chin, right_eye, mouth_right.
    KEY_ORDER = [NOSE, FOREHEAD, LEFT_EYE, MOUTH_LEFT, CHIN, RIGHT_EYE, MOUTH_RIGHT]

    def __init__(self, model_path: str | Path) -> None:
        self.model = None
        self.available = False
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            self.available = True
        except Exception as exc:
            warnings.warn(
                f"Head pose model disabled: cannot load {model_path} ({exc})",
                RuntimeWarning,
            )

    def estimate(self, landmarks: list[tuple[float, float]]) -> tuple[float, float, float]:
        if not self.available or self.model is None:
            return 0.0, 0.0, 0.0
        if len(landmarks) < self.RIGHT_EYE + 1:
            return 0.0, 0.0, 0.0
        features = []
        for idx in self.KEY_ORDER:
            if idx < len(landmarks):
                features.append(landmarks[idx][0])
                features.append(landmarks[idx][1])
        if len(features) < 14:
            return 0.0, 0.0, 0.0
        arr = np.array(features)
        normalized = arr.copy()
        for dim_idx in (0, 1):
            for feat_idx in range(dim_idx, 14, 2):
                normalized[feat_idx] = arr[feat_idx] - arr[dim_idx]
            diff = arr[12 + dim_idx] - arr[4 + dim_idx]
            if diff != 0:
                for feat_idx in range(dim_idx, 14, 2):
                    normalized[feat_idx] /= diff
        pitch, yaw, roll = self.model.predict([normalized]).ravel()
        return float(pitch), float(yaw), float(roll)
