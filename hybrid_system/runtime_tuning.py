from __future__ import annotations

import os


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def configure_thread_environment(threads: int = 1) -> None:
    """Set native thread limits before importing numerical libraries."""
    value = str(max(1, int(threads)))
    for name in THREAD_ENV_VARS:
        os.environ[name] = value


def configure_runtime_threads(threads: int = 1) -> dict[str, int]:
    """Apply thread limits exposed by OpenCV and PyTorch."""
    count = max(1, int(threads))
    applied: dict[str, int] = {}

    try:
        import cv2

        cv2.setNumThreads(count)
        applied["opencv"] = cv2.getNumThreads()
    except (ImportError, AttributeError):
        pass

    try:
        import torch

        torch.set_num_threads(count)
        applied["pytorch"] = torch.get_num_threads()
        try:
            torch.set_num_interop_threads(count)
        except RuntimeError:
            # PyTorch only permits setting inter-op threads before parallel work starts.
            pass
        applied["pytorch_interop"] = torch.get_num_interop_threads()
    except (ImportError, AttributeError):
        pass

    return applied
