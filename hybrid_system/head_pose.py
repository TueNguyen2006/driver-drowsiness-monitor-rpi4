from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


FOREHEAD = 10
NOSE = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 199
LEFT_EYE = 33
RIGHT_EYE = 263

# Must match the training notebook column order.
KEY_ORDER = (NOSE, FOREHEAD, LEFT_EYE, MOUTH_LEFT, CHIN, RIGHT_EYE, MOUTH_RIGHT)


def preprocess_landmarks(
    landmarks: list[tuple[float, float]],
) -> np.ndarray | None:
    if len(landmarks) <= RIGHT_EYE:
        return None
    features = np.asarray(
        [coordinate for index in KEY_ORDER for coordinate in landmarks[index]],
        dtype=np.float64,
    )
    normalized = features.copy()
    for dimension in (0, 1):
        normalized[dimension::2] -= features[dimension]
        scale = features[12 + dimension] - features[4 + dimension]
        if scale != 0.0:
            normalized[dimension::2] /= scale
    return normalized


class HeadPoseEstimator:
    __slots__ = ("_estimators", "available")

    def __init__(self, model_path: str | Path) -> None:
        self._estimators: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        self.available = False
        try:
            with np.load(model_path, allow_pickle=False) as data:
                count = int(data["estimator_count"])
                for index in range(count):
                    self._estimators.append((
                        np.asarray(data[f"support_vectors_{index}"], dtype=np.float64),
                        np.asarray(data[f"dual_coef_{index}"], dtype=np.float64).reshape(-1),
                        float(data[f"intercept_{index}"]),
                        float(data[f"gamma_{index}"]),
                    ))
            self.available = len(self._estimators) == 3
        except Exception as exc:
            warnings.warn(
                f"Head pose model disabled: cannot load {model_path} ({exc})",
                RuntimeWarning,
            )

    def estimate(self, landmarks: list[tuple[float, float]]) -> tuple[float, float, float]:
        if not self.available:
            return 0.0, 0.0, 0.0
        features = preprocess_landmarks(landmarks)
        if features is None:
            return 0.0, 0.0, 0.0
        prediction = [self._predict_rbf(features, *estimator) for estimator in self._estimators]
        return float(prediction[0]), float(prediction[1]), float(prediction[2])

    @staticmethod
    def _predict_rbf(
        features: np.ndarray,
        support_vectors: np.ndarray,
        dual_coef: np.ndarray,
        intercept: float,
        gamma: float,
    ) -> float:
        squared_distance = np.sum((support_vectors - features) ** 2, axis=1)
        kernel = np.exp(-gamma * squared_distance)
        return float(np.dot(dual_coef, kernel) + intercept)
