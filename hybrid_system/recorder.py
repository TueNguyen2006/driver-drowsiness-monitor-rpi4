from __future__ import annotations

from pathlib import Path


class SessionRecorder:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events: list[dict] = []
        self.frame_scores: list[tuple[float, float]] = []
        self.latencies_ms: list[float] = []

    def write_event(self, event: dict) -> None:
        self.events.append(event)

    def write_frame_score(self, timestamp: float, score: float, latency_ms: float) -> None:
        self.frame_scores.append((timestamp, score))
        self.latencies_ms.append(latency_ms)
