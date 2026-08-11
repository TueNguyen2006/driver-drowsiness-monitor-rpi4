#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def export_model(source: Path, destination: Path) -> None:
    with source.open("rb") as model_file:
        model = pickle.load(model_file)

    estimators = getattr(model, "estimators_", None)
    if not estimators or len(estimators) != 3:
        raise ValueError("Expected a fitted three-output MultiOutputRegressor")

    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(1, dtype=np.int64),
        "estimator_count": np.asarray(len(estimators), dtype=np.int64),
        "key_order": np.asarray([1, 10, 33, 61, 199, 263, 291], dtype=np.int64),
    }
    for index, estimator in enumerate(estimators):
        if estimator.kernel != "rbf":
            raise ValueError(f"Estimator {index} uses unsupported kernel {estimator.kernel!r}")
        arrays[f"support_vectors_{index}"] = np.asarray(
            estimator.support_vectors_, dtype=np.float64
        )
        arrays[f"dual_coef_{index}"] = np.asarray(estimator.dual_coef_, dtype=np.float64)
        arrays[f"intercept_{index}"] = np.asarray(estimator.intercept_[0], dtype=np.float64)
        arrays[f"gamma_{index}"] = np.asarray(estimator._gamma, dtype=np.float64)

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sklearn head-pose SVRs to NumPy")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "models/model.pkl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models/head_pose_svr.npz")
    args = parser.parse_args()
    export_model(args.input.resolve(), args.output.resolve())
    print(f"Exported {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
