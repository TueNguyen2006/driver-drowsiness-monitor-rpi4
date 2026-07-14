from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationParams:
    ear_mean: float = 0.3
    ear_std: float = 0.05
    mar_mean: float = 0.3
    mar_std: float = 0.1
    puc_mean: float = 0.0
    puc_std: float = 0.1
    moe_mean: float = 0.0
    moe_std: float = 0.1
    pitch_offset: float = 0.0
    pitch_std: float = 0.05
    yaw_offset: float = 0.0
    yaw_std: float = 0.05
    roll_offset: float = 0.0
    roll_std: float = 0.05
    calibrated: bool = False

    ear_buffer: list[float] = field(default_factory=list)
    mar_buffer: list[float] = field(default_factory=list)
    puc_buffer: list[float] = field(default_factory=list)
    moe_buffer: list[float] = field(default_factory=list)
    pitch_buffer: list[float] = field(default_factory=list)
    yaw_buffer: list[float] = field(default_factory=list)
    roll_buffer: list[float] = field(default_factory=list)

    def add_sample(
        self,
        ear: float,
        mar: float,
        puc: float,
        moe: float,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> None:
        self.ear_buffer.append(ear)
        self.mar_buffer.append(mar)
        self.puc_buffer.append(puc)
        self.moe_buffer.append(moe)
        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)

    def compute(self) -> None:
        if not self.ear_buffer:
            self.calibrated = True
            return
        self.ear_mean = float(np.mean(self.ear_buffer))
        self.ear_std = float(np.std(self.ear_buffer)) or 0.01
        self.mar_mean = float(np.mean(self.mar_buffer))
        self.mar_std = float(np.std(self.mar_buffer)) or 0.01
        self.puc_mean = float(np.mean(self.puc_buffer))
        self.puc_std = float(np.std(self.puc_buffer)) or 0.01
        self.moe_mean = float(np.mean(self.moe_buffer))
        self.moe_std = float(np.std(self.moe_buffer)) or 0.01
        self.pitch_offset = float(np.mean(self.pitch_buffer))
        self.pitch_std = float(np.std(self.pitch_buffer)) or 0.01
        self.yaw_offset = float(np.mean(self.yaw_buffer))
        self.yaw_std = float(np.std(self.yaw_buffer)) or 0.01
        self.roll_offset = float(np.mean(self.roll_buffer))
        self.roll_std = float(np.std(self.roll_buffer)) or 0.01
        self.calibrated = True

    def normalize_ear(self, ear: float) -> float:
        return (ear - self.ear_mean) / self.ear_std

    def normalize_mar(self, mar: float) -> float:
        return (mar - self.mar_mean) / self.mar_std

    def normalize_puc(self, puc: float) -> float:
        return (puc - self.puc_mean) / self.puc_std

    def normalize_moe(self, moe: float) -> float:
        return (moe - self.moe_mean) / self.moe_std

    def normalize_pitch(self, pitch: float) -> float:
        return pitch - self.pitch_offset

    def normalize_yaw(self, yaw: float) -> float:
        return yaw - self.yaw_offset

    def normalize_roll(self, roll: float) -> float:
        return roll - self.roll_offset


class SmartCalibrator:
    def __init__(self, params: CalibrationParams,
                 min_samples: int = 50,
                 max_samples: int = 200,
                 min_ear: float = 0.15,
                 max_mar: float = 0.60,
                 max_abs_pitch: float = 1.25,
                 max_abs_yaw: float = 1.25,
                 max_abs_roll: float = 1.25,
                 stable_window_size: int = 10,
                 ear_std_max: float = 0.01,
                 mar_std_max: float = 0.02,
                 pose_std_max: float = 0.08) -> None:
        self._params = params
        self._min = min_samples
        self._max = max_samples
        self._good_frames = 0
        self._min_ear = min_ear
        self._max_mar = max_mar
        self._max_abs_pitch = max_abs_pitch
        self._max_abs_yaw = max_abs_yaw
        self._max_abs_roll = max_abs_roll
        self._stable_window_size = stable_window_size
        self._ear_std_max = ear_std_max
        self._mar_std_max = mar_std_max
        self._pose_std_max = pose_std_max
        self._queue: queue.Queue[tuple[float, ...]] = queue.Queue(maxsize=500)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = True
        self._thread.start()

    def push_sample(
        self, ear: float, mar: float, puc: float, moe: float,
        pitch: float, yaw: float, roll: float,
    ) -> None:
        if self._params.calibrated:
            return
        try:
            self._queue.put_nowait((ear, mar, puc, moe, pitch, yaw, roll))
        except queue.Full:
            pass

    def _loop(self) -> None:
        window: list[tuple[float, ...]] = []
        while self._running:
            try:
                sample = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            window.append(sample)
            max_win = max(self._max, self._stable_window_size * 2, 60)
            if len(window) > max_win * 2:
                window = window[-max_win:]

            ear, mar, _puc, _moe, pitch, yaw, _roll = sample

            if ear <= self._min_ear:
                continue
            if mar >= self._max_mar:
                continue
            if abs(pitch) > self._max_abs_pitch or abs(yaw) > self._max_abs_yaw or abs(_roll) > self._max_abs_roll:
                continue

            if len(window) >= self._stable_window_size:
                recent = window[-self._stable_window_size:]
                ears = np.array([s[0] for s in recent])
                mars = np.array([s[1] for s in recent])
                pitches = np.array([s[4] for s in recent])
                yaws = np.array([s[5] for s in recent])
                rolls = np.array([s[6] for s in recent])
                if np.std(ears) > self._ear_std_max:
                    continue
                if np.std(mars) > self._mar_std_max:
                    continue
                if max(np.std(pitches), np.std(yaws), np.std(rolls)) > self._pose_std_max:
                    continue

            self._params.add_sample(ear, mar, _puc, _moe, pitch, yaw, _roll)
            self._good_frames += 1
            if self._good_frames >= self._min and not self._params.calibrated:
                self._params.compute()
            elif self._good_frames >= self._max and not self._params.calibrated:
                self._params.compute()

    def force_defaults(self) -> None:
        if not self._params.calibrated:
            self._params.calibrated = True

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=1)

    @property
    def calibrated(self) -> bool:
        return self._params.calibrated

    @property
    def progress(self) -> float:
        if self._min == 0:
            return 1.0
        return min(1.0, self._good_frames / self._min)

    def normalize_ear(self, ear: float) -> float:
        return self._params.normalize_ear(ear)

    def normalize_mar(self, mar: float) -> float:
        return self._params.normalize_mar(mar)

    def normalize_puc(self, puc: float) -> float:
        return self._params.normalize_puc(puc)

    def normalize_moe(self, moe: float) -> float:
        return self._params.normalize_moe(moe)

    def normalize_pitch(self, pitch: float) -> float:
        return self._params.normalize_pitch(pitch)

    def normalize_yaw(self, yaw: float) -> float:
        return self._params.normalize_yaw(yaw)

    def normalize_roll(self, roll: float) -> float:
        return self._params.normalize_roll(roll)
