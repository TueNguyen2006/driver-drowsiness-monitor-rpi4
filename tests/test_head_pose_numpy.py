from __future__ import annotations

import pickle
import unittest
import warnings
from pathlib import Path

import numpy as np

from hybrid_system.head_pose import HeadPoseEstimator, preprocess_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HeadPoseNumpyRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with (PROJECT_ROOT / "models/model.pkl").open("rb") as model_file:
                    cls.sklearn_model = pickle.load(model_file)
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f"sklearn is only required for regression testing: {exc}")
        cls.numpy_model = HeadPoseEstimator(PROJECT_ROOT / "models/head_pose_svr.npz")

    def test_predictions_match_legacy_sklearn_model(self) -> None:
        rng = np.random.default_rng(20260811)
        for _ in range(200):
            landmarks = [(0.0, 0.0)] * 468
            mutable = list(landmarks)
            for index in (1, 10, 33, 61, 199, 263, 291):
                mutable[index] = (
                    float(rng.uniform(0.25, 0.75)),
                    float(rng.uniform(0.15, 0.85)),
                )
            features = preprocess_landmarks(mutable)
            self.assertIsNotNone(features)
            expected = self.sklearn_model.predict(features.reshape(1, -1))[0]
            actual = self.numpy_model.estimate(mutable)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
