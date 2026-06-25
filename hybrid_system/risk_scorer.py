from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import mean

from .models import DetectionEvent, DriverState, SessionSummary

DEFAULT_SIGNAL_WEIGHTS = {
    "eyes_closed": 0.34,
    "drowsy": 0.54,
    "yawning": 0.22,
    "distracted": 0.34,
    "phone_use": 0.64,
}


class RollingWindow:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self._values: list[float] = []

    def add(self, value: float) -> float:
        self._values.append(float(value))
        while len(self._values) > self.size:
            self._values.pop(0)
        return self.mean

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)


class SignalSmoother:
    def __init__(self, window_size: int = 5) -> None:
        self._windows: dict[str, RollingWindow] = {}
        self._window_size = window_size

    def update(self, signals: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in signals.items():
            if name not in self._windows:
                self._windows[name] = RollingWindow(size=self._window_size)
            result[name] = self._windows[name].add(value)
        return result


class RiskScorer:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or DEFAULT_SIGNAL_WEIGHTS

    def score(self, signals: dict[str, float]) -> float:
        evidence = [
            _clamp(signals.get(name, 0.0)) * weight
            for name, weight in self.weights.items()
            if signals.get(name, 0.0) > 0
        ]
        fused = _noisy_or(evidence)
        fused += self._cross_signal_boost(signals)
        return round(_clamp(fused), 4)

    def _cross_signal_boost(self, signals: dict[str, float]) -> float:
        drowsy = _clamp(signals.get("drowsy", 0.0))
        eyes_closed = _clamp(signals.get("eyes_closed", 0.0))
        yawning = _clamp(signals.get("yawning", 0.0))
        distracted = _clamp(signals.get("distracted", 0.0))
        phone = _clamp(signals.get("phone_use", 0.0))
        boost = 0.0
        if drowsy >= 0.75 and eyes_closed >= 0.75:
            boost += 0.08
        if drowsy >= 0.6 and yawning >= 0.5:
            boost += 0.08
        vision_fatigue = max(drowsy, eyes_closed * 0.85, yawning * 0.55)
        visual_dist = max(distracted, phone)
        if vision_fatigue >= 0.6:
            boost += 0.05
        if visual_dist >= 0.6:
            boost += 0.05
        return boost

    def state_from_events(self, events: list[DetectionEvent], risk_score: float) -> DriverState:
        priority = [
            DriverState.PHONE_USE,
            DriverState.DROWSY,
            DriverState.EYES_CLOSED,
            DriverState.YAWNING,
            DriverState.DISTRACTED,
        ]
        active = {event.state for event in events}
        for state in priority:
            if state in active:
                return state
        if risk_score >= 0.55:
            return DriverState.DISTRACTED
        return DriverState.ATTENTIVE

    def summarize(
        self,
        session_id: str,
        source: str,
        duration_seconds: float,
        processed_frames: int,
        events: list[DetectionEvent],
        frame_scores: list[tuple[float, float]],
        metrics: dict[str, float | int | str],
    ) -> SessionSummary:
        event_counts = Counter(event.signal for event in events)
        risk_timeline = [
            {"timestamp": round(ts, 3), "risk_score": round(sc, 4)}
            for ts, sc in frame_scores
        ]
        unsafe_ts = [ts for ts, sc in frame_scores if sc >= 0.45]
        longest_unsafe = _longest_contiguous(unsafe_ts)
        signal_scores: dict[str, list[float]] = {}
        for event in events:
            signal_scores.setdefault(event.signal, []).append(event.score)
        conf_dist = {
            signal: round(mean(scores), 4) for signal, scores in sorted(signal_scores.items())
        }
        return SessionSummary(
            session_id=session_id,
            source=source,
            duration_seconds=round(duration_seconds, 3),
            processed_frames=processed_frames,
            event_counts=dict(sorted(event_counts.items())),
            risk_timeline=risk_timeline,
            longest_unsafe_interval_seconds=round(longest_unsafe, 3),
            confidence_distribution=conf_dist,
            metrics=dict(metrics),
        )


def _longest_contiguous(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    longest = 0.0
    start = prev = timestamps[0]
    for t in timestamps[1:]:
        if t - prev > 1.25:
            longest = max(longest, prev - start)
            start = t
        prev = t
    return max(longest, prev - start)


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


def _noisy_or(evidence: list[float]) -> float:
    p = 1.0
    for v in evidence:
        p *= 1.0 - _clamp(v)
    return 1.0 - p
