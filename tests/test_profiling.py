from __future__ import annotations

import os
import unittest

from hybrid_system.profiling import ProfilingWindow
from hybrid_system.runtime_tuning import THREAD_ENV_VARS, configure_thread_environment


class ProfilingTests(unittest.TestCase):
    def test_periodic_stage_uses_its_own_sample_count(self) -> None:
        profiler = ProfilingWindow()
        profiler.record({"total_ms": 10.0})
        profiler.record({"total_ms": 20.0, "yolo_ms": 100.0})

        summary = profiler.summary()

        self.assertEqual(summary["total_ms"]["count"], 2)
        self.assertEqual(summary["total_ms"]["mean_ms"], 15.0)
        self.assertEqual(summary["yolo_ms"]["count"], 1)
        self.assertEqual(summary["yolo_ms"]["mean_ms"], 100.0)

    def test_thread_environment_is_enforced(self) -> None:
        configure_thread_environment(1)
        for name in THREAD_ENV_VARS:
            self.assertEqual(os.environ[name], "1")


if __name__ == "__main__":
    unittest.main()
