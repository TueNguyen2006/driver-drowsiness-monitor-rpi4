from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np


class HeadPoseEstimator:
    __slots__ = ("model",)

    FOREHEAD = 10
    NOSE = 1
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    CHIN = 199
    LEFT_EYE = 33
    RIGHT_EYE = 263
    KEY_ORDER = [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]

    def __init__(self, model_path: str | Path) -> None:
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def estimate(self, landmarks: list[tuple[float, float]]) -> tuple[float, float, float]:
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
