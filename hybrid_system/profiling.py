from __future__ import annotations

import os
import resource
import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cpu_percent: float
    rss_mb: float
    peak_rss_mb: float
    load_1m: float
    temperature_c: float | None


class ProfilingWindow:
    """Collect stage samples without diluting stages that run periodically."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._wall_started = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._cpu_started = usage.ru_utime + usage.ru_stime

    def record(self, values: dict[str, float]) -> None:
        with self._lock:
            for name, value in values.items():
                self._samples[name].append(float(value))

    def summary(self, *, reset: bool = False) -> dict[str, dict[str, float | int]]:
        result: dict[str, dict[str, float | int]] = {}
        with self._lock:
            samples = {name: list(values) for name, values in self._samples.items()}
            if reset:
                self._samples.clear()
        for name, values in sorted(samples.items()):
            data = np.asarray(values, dtype=np.float64)
            result[name] = {
                "count": int(data.size),
                "mean_ms": float(data.mean()),
                "p50_ms": float(np.percentile(data, 50)),
                "p95_ms": float(np.percentile(data, 95)),
                "max_ms": float(data.max()),
            }
        return result

    def resources(self) -> ResourceSnapshot:
        now = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_now = usage.ru_utime + usage.ru_stime
        wall_delta = max(now - self._wall_started, 1e-9)
        cpu_percent = 100.0 * (cpu_now - self._cpu_started) / wall_delta
        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            rss_mb=_current_rss_mb(),
            peak_rss_mb=usage.ru_maxrss / 1024.0,
            load_1m=os.getloadavg()[0],
            temperature_c=_temperature_c(),
        )

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
        self._wall_started = time.perf_counter()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self._cpu_started = usage.ru_utime + usage.ru_stime


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _temperature_c() -> float | None:
    try:
        with open(
            "/sys/class/thermal/thermal_zone0/temp", encoding="ascii"
        ) as sensor:
            return float(sensor.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None
