from __future__ import annotations

import statistics
import time

import numpy as np

from hybrid_system.display import FFplayDisplay, OpenCVDisplay


def bench_backend(name: str, display_cls, size: tuple[int, int], frames: int = 240) -> dict[str, float]:
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    start = time.perf_counter()
    disp = display_cls("bench", size, fullscreen=False, fps=30.0)
    init_ms = (time.perf_counter() - start) * 1000.0

    samples: list[float] = []
    for _ in range(frames):
        t0 = time.perf_counter()
        disp.show(frame)
        disp.pump(60)
        samples.append((time.perf_counter() - t0) * 1000.0)

    disp.close()
    return {
        "name": name,
        "init_ms": init_ms,
        "avg_ms": statistics.mean(samples),
        "p95_ms": statistics.quantiles(samples, n=20)[18],
        "fps": 1000.0 / statistics.mean(samples) if samples else 0.0,
    }


def main() -> None:
    size = (800, 480)
    results = []
    for name, cls in (("OpenCV", OpenCVDisplay), ("FFplay", FFplayDisplay)):
        try:
            results.append(bench_backend(name, cls, size))
        except Exception as exc:
            results.append({"name": name, "error": str(exc)})

    for result in results:
        if "error" in result:
            print(f"{result['name']}: ERROR: {result['error']}")
            continue
        print(
            f"{result['name']}: init={result['init_ms']:.1f}ms "
            f"avg_show={result['avg_ms']:.3f}ms p95={result['p95_ms']:.3f}ms "
            f"throughput={result['fps']:.1f} fps"
        )


if __name__ == "__main__":
    main()
