from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .calibration import CalibrationParams, SmartCalibrator
from .config import HybridConfig
from .features import extract_all
from .models import DetectionEvent, DriverState, ProcessedFrame, Severity
from .risk_scorer import RiskScorer, SignalSmoother

import threading
import time

MAX_EVENTS = 256
OBJECT_SKIP_INTERVAL = 3


def open_camera_capture(src=0) -> cv2.VideoCapture:
    candidates: list[tuple[object, int | None]] = []
    if isinstance(src, int):
        candidates.extend([
            (f"/dev/video{src}", cv2.CAP_V4L2),
            (src, cv2.CAP_V4L2),
        ])
    else:
        candidates.extend([
            (src, cv2.CAP_V4L2),
        ])

    for target, backend in candidates:
        cap = cv2.VideoCapture(target) if backend is None else cv2.VideoCapture(target, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return cv2.VideoCapture()


class ThreadedCamera:
    """Separate thread for camera capture to maximize FPS."""
    def __init__(self, src=0):
        self._cap = open_camera_capture(src)
        self._ret = False
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 2.0
        while self._frame is None and self._running and time.monotonic() < deadline:
            time.sleep(0.005)

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if frame is not None:
                with self._lock:
                    self._ret = ret
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def release(self):
        self._running = False
        self._thread.join(timeout=1)
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, val):
        self._cap.set(prop, val)


class HybridPipeline:
    def __init__(self, config: HybridConfig) -> None:
        self.cfg = config

        self._landmarker = None
        self._head_pose_model = None
        self._lstm = None
        self._object_det = None

        self._init_landmarker()
        self._init_lstm()
        self._init_objects()

        self.smoother = SignalSmoother(window_size=5)
        self.scorer = RiskScorer(config.signal_weights or None)
        calib_params = CalibrationParams()
        cal_min = max(25, config.calibration.frame_count // 2)
        cal_max = config.calibration.frame_count
        c = config.calibration
        self.calibrator = SmartCalibrator(
            calib_params,
            min_samples=cal_min,
            max_samples=cal_max,
            min_ear=c.min_ear,
            max_mar=c.max_mar,
            max_abs_pitch=c.max_abs_pitch,
            max_abs_yaw=c.max_abs_yaw,
            max_abs_roll=c.max_abs_roll,
            stable_window_size=c.stable_window_size,
            ear_std_max=c.ear_std_max,
            mar_std_max=c.mar_std_max,
            pose_std_max=c.pose_std_max,
        )

        self._missing_face_counter = 0
        self._phone_hold_counter = 0
        self._events: list[DetectionEvent] = []

        self._ear_smoothed = -1000.0
        self._mar_smoothed = -1000.0
        self._puc_smoothed = -1000.0
        self._moe_smoothed = -1000.0
        self._decay = 0.9
        self._pitch_count = 0
        self._head_dropped = 0
        self._ear_trigger_count = 0
        self._mar_trigger_count = 0
        self._yaw_distract_count = 0
        self._decision_count = 0
        self._frame_before_run = 0
        self._input_data: list[list[float]] = []
        self._lstm_label: int | None = None

        self._frame_skip_counter = 0
        self._obj_skip_counter = 0
        self._skip_n = max(1, config.vision.process_every_n_frames)
        self._last_landmarks: list[tuple[float, float]] | None = None
        self._last_features: dict[str, float] = {"ear": 0.5, "mar": 0.1, "puc": 0.5, "moe": 0.2}

    def _init_landmarker(self) -> None:
        model_path = Path(self.cfg.vision.face_landmarker_model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found: {model_path}"
            )
        from mediapipe.tasks import python
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        opts = FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(opts)
        from mediapipe import ImageFormat, Image as MpImage
        self._mp_image_cls = MpImage
        self._mp_image_format = ImageFormat

        hp = self.cfg.head_pose
        if hp.model_path and Path(hp.model_path).exists():
            from .head_pose import HeadPoseEstimator
            self._head_pose_model = HeadPoseEstimator(hp.model_path)

    def _init_lstm(self) -> None:
        path = self.cfg.lstm.model_path
        if path and Path(path).exists():
            from .lstm import LSTMClassifier
            self._lstm = LSTMClassifier(path)

    def _init_objects(self) -> None:
        od = self.cfg.object_detector
        if od.enabled and od.model_path and Path(od.model_path).exists():
            from .objects import ObjectDetector
            self._object_det = ObjectDetector(
                enabled=True,
                model_path=od.model_path,
                confidence_threshold=od.confidence_threshold,
                iou_threshold=od.iou_threshold,
                phone_labels=od.phone_labels,
            )

    def process_frame(
        self, frame: np.ndarray, timestamp: float, frame_index: int
    ) -> ProcessedFrame:
        started = perf_counter()
        self._frame_skip_counter += 1

        face_bbox = None
        all_points: list[tuple[float, float]] = []
        pitch = yaw = roll = 0.0

        skip_landmarks = self._frame_skip_counter % self._skip_n != 0
        if skip_landmarks and self._last_landmarks is not None:
            landmarks = self._last_landmarks
            h, w = frame.shape[:2]
            all_points = [(lm[0] * w, lm[1] * h) for lm in landmarks]
            ear, mar, puc, moe = (
                self._last_features["ear"], self._last_features["mar"],
                self._last_features["puc"], self._last_features["moe"],
            )
        else:
            landmarks = self._detect_landmarks(frame)
            self._last_landmarks = landmarks
            if landmarks and len(landmarks) >= 468:
                h, w = frame.shape[:2]
                xs = [lm[0] * w for lm in landmarks]
                ys = [lm[1] * h for lm in landmarks]
                x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
                x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
                face_bbox = (x1, y1, x2 - x1, y2 - y1)
                all_points = [(xs[i], ys[i]) for i in range(len(landmarks))]
                features = extract_all(all_points)
                ear, mar, puc, moe = features["ear"], features["mar"], features["puc"], features["moe"]
                self._last_features = features
            else:
                if self._last_features:
                    ear = self._last_features["ear"]
                    mar = self._last_features["mar"]
                    puc = self._last_features["puc"]
                    moe = self._last_features["moe"]
                else:
                    ear = mar = puc = moe = 0.0

        raw_signals: dict[str, float] = {
            "eyes_closed": 0.0, "drowsy": 0.0,
            "yawning": 0.0, "distracted": 0.0, "phone_use": 0.0,
        }
        new_events: list[DetectionEvent] = []

        if landmarks is None or len(landmarks) < 468:
            self._missing_face_counter += 1
            if self._missing_face_counter >= self.cfg.thresholds.missing_face_frames:
                raw_signals["face_lost"] = 1.0
                new_events.append(self._make_event(
                    timestamp, frame_index,
                    "face_lost", DriverState.NO_FACE, 1.0, Severity.WARNING,
                    "Face not visible - cannot assess driver state",
                ))
        else:
            self._missing_face_counter = 0

            if self._head_pose_model is not None:
                pitch, yaw, roll = self._head_pose_model.estimate(landmarks)

            s = self._do_inference(
                ear, mar, puc, moe, pitch, yaw, roll,
            )
            raw_signals.update(s)
            new_events.extend(self._events_from_signals(
                timestamp, frame_index,
            ))

            self.calibrator.push_sample(ear, mar, puc, moe, pitch, yaw, roll)

        if self._object_det:
            self._obj_skip_counter += 1
            if self._obj_skip_counter % OBJECT_SKIP_INTERVAL == 0:
                phone_events, phone_signal = self._process_objects(frame, timestamp, frame_index)
                if phone_signal > raw_signals.get("phone_use", 0.0):
                    raw_signals["phone_use"] = phone_signal
                    self._phone_hold_counter = self.cfg.thresholds.phone_hold_frames
                new_events.extend(phone_events)
            elif self._phone_hold_counter > 0:
                raw_signals["phone_use"] = max(raw_signals.get("phone_use", 0.0), 1.0)
                self._phone_hold_counter -= 1

        smoothed = self.smoother.update(raw_signals)

        risk_score = self.scorer.score(smoothed)
        state = self.scorer.state_from_events(new_events, risk_score)

        self._events.extend(new_events)
        if len(self._events) > MAX_EVENTS:
            self._events = self._events[-MAX_EVENTS:]

        latency_ms = (perf_counter() - started) * 1000

        return ProcessedFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            state=state,
            risk_score=risk_score,
            signals=smoothed,
            events=new_events,
            latency_ms=latency_ms,
            face_bbox=face_bbox,
            landmarks=all_points,
            objects=[],
            head_pose=(pitch, yaw, roll) if landmarks else None,
        )

    def _detect_landmarks(self, frame: np.ndarray) -> list[tuple[float, float]] | None:
        if self._landmarker is None:
            return None
        h, w = frame.shape[:2]
        pw = self.cfg.vision.process_width or w
        ph = self.cfg.vision.process_height or h
        if pw != w or ph != h:
            proc = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)
        else:
            proc = frame
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        image = self._mp_image_cls(
            image_format=self._mp_image_format.SRGB, data=rgb
        )
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        # Raw normalized coords (0..1) — process_frame converts to pixels via w/h
        return [(lm.x, lm.y) for lm in result.face_landmarks[0]]

    def _do_inference(
        self, ear: float, mar: float, puc: float, moe: float,
        pitch: float, yaw: float, roll: float,
    ) -> dict[str, float]:
        ear_n = self.calibrator.normalize_ear(ear)
        mar_n = self.calibrator.normalize_mar(mar)
        puc_n = self.calibrator.normalize_puc(puc)
        moe_n = self.calibrator.normalize_moe(moe)
        pitch_n = self.calibrator.normalize_pitch(pitch)
        yaw_n = self.calibrator.normalize_yaw(yaw)
        roll_n = self.calibrator.normalize_roll(roll)

        if self._ear_smoothed == -1000.0:
            self._ear_smoothed = ear_n
            self._mar_smoothed = mar_n
            self._puc_smoothed = puc_n
            self._moe_smoothed = moe_n
        else:
            d = self._decay
            self._ear_smoothed = self._ear_smoothed * d + (1 - d) * ear_n
            self._mar_smoothed = self._mar_smoothed * d + (1 - d) * mar_n
            self._puc_smoothed = self._puc_smoothed * d + (1 - d) * puc_n
            self._moe_smoothed = self._moe_smoothed * d + (1 - d) * moe_n

        t = self.cfg.thresholds
        pitch_thr = max(abs(t.pitch_upper), abs(t.pitch_lower))
        yaw_thr = getattr(t, "yaw_threshold", 0.25)

        if self.calibrator.calibrated:
            self._head_dropped = 1 if abs(pitch_n) > pitch_thr else 0
            self._pitch_count = self._pitch_count + 1 if self._head_dropped else 0
            self._yaw_distract_count = self._yaw_distract_count + 1 if abs(yaw_n) > yaw_thr else 0
        else:
            self._head_dropped = 0
            self._pitch_count = 0
            self._yaw_distract_count = 0

        # --- LSTM inference ---
        inp = self._input_data
        if len(inp) == 20:
            inp.pop(0)
        inp.append([self._ear_smoothed, self._mar_smoothed,
                     self._puc_smoothed, self._moe_smoothed])

        self._frame_before_run += 1
        if self._frame_before_run >= 15 and len(inp) == 20:
            self._frame_before_run = 0
            self._lstm_label = (
                self._lstm.classify(inp) if self._lstm is not None else 0
            )
            self._decision_count = (
                0 if self._lstm_label == 0 else self._decision_count + 1
            )

        # --- EAR trigger (eyes closed) ---
        ear_trigger = self._ear_smoothed < t.ear_zscore_threshold
        self._ear_trigger_count = self._ear_trigger_count + 1 if ear_trigger else 0

        # --- MAR trigger (mouth open / yawn) ---
        mar_trigger = self._mar_smoothed > t.mar_zscore_threshold
        self._mar_trigger_count = self._mar_trigger_count + 1 if mar_trigger else 0

        signals: dict[str, float] = {
            "eyes_closed": 0.0, "drowsy": 0.0, "yawning": 0.0,
            "distracted": 0.0, "phone_use": 0.0,
        }

        drowsy = any([
            self._ear_trigger_count > t.ear_trigger_frames,
            self._decision_count >= t.classification_threshold,
        ])

        if drowsy:
            signals["drowsy"] = min(1.0, self._ear_trigger_count / 40.0) if self._ear_trigger_count > 0 else 0.6
        if self._ear_trigger_count > 5:
            signals["eyes_closed"] = min(1.0, self._ear_trigger_count / 30.0)

        # --- Yawning signal ---
        if self._mar_trigger_count >= t.yawn_frames:
            signals["yawning"] = min(1.0, self._mar_trigger_count / 20.0)

        # --- Distraction signal ---
        if self.calibrator.calibrated:
            any_distracted = (
                (self._head_dropped and self._pitch_count >= t.distracted_frames)
                or self._yaw_distract_count >= t.distracted_frames
            )
            if any_distracted:
                signals["distracted"] = min(1.0, max(
                    self._pitch_count / t.distracted_frames if self._head_dropped else 0,
                    self._yaw_distract_count / t.distracted_frames,
                ) * 0.5)

        if self._lstm_label is not None and self._lstm_label == 1:
            signals["drowsy"] = max(signals["drowsy"], 0.7)

        return signals

    def _events_from_signals(
        self, timestamp: float, frame_index: int,
    ) -> list[DetectionEvent]:
        events: list[DetectionEvent] = []
        t = self.cfg.thresholds

        if self._ear_trigger_count >= t.eye_closed_frames:
            events.append(self._make_event(
                timestamp, frame_index, "eyes_closed", DriverState.EYES_CLOSED,
                min(1.0, self._ear_trigger_count / t.drowsy_frames),
                Severity.WARNING,
                "Eyes closed beyond configured threshold",
            ))
        if self._ear_trigger_count >= t.drowsy_frames:
            events.append(self._make_event(
                timestamp, frame_index, "drowsy", DriverState.DROWSY,
                min(1.0, self._ear_trigger_count / t.drowsy_frames),
                Severity.CRITICAL,
                "Sustained eye closure indicates drowsiness",
            ))
        if self._mar_trigger_count >= t.yawn_frames:
            events.append(self._make_event(
                timestamp, frame_index, "yawning", DriverState.YAWNING,
                min(1.0, self._mar_trigger_count / 20.0),
                Severity.WARNING,
                "Sustained mouth opening detected - possible yawning",
            ))
        if self.calibrator.calibrated and self._head_dropped and self._pitch_count >= t.distracted_frames:
            events.append(self._make_event(
                timestamp, frame_index, "distracted", DriverState.DISTRACTED,
                min(1.0, self._pitch_count / t.distracted_frames),
                Severity.WARNING,
                "Head dropped - possible distraction",
            ))
        if self.calibrator.calibrated and self._yaw_distract_count >= t.distracted_frames:
            events.append(self._make_event(
                timestamp, frame_index, "distracted", DriverState.DISTRACTED,
                min(1.0, self._yaw_distract_count / t.distracted_frames),
                Severity.WARNING,
                "Head turned away - driver distracted",
            ))
        if self._decision_count >= t.classification_threshold:
            events.append(self._make_event(
                timestamp, frame_index, "drowsy", DriverState.DROWSY,
                min(1.0, self._decision_count / (t.classification_threshold * 2)),
                Severity.CRITICAL,
                "LSTM model detected drowsiness pattern",
            ))
        return events

    def _process_objects(
        self, frame: np.ndarray, timestamp: float, frame_index: int,
    ) -> tuple[list[DetectionEvent], float]:
        if self._object_det is None:
            return [], 0.0
        objects = self._object_det.detect(frame)
        events: list[DetectionEvent] = []
        phone_signal = 0.0
        phone_labels = {l.lower() for l in self.cfg.object_detector.phone_labels}
        for obj in objects:
            if obj.label.lower() in phone_labels and obj.confidence >= self.cfg.thresholds.phone_confidence:
                phone_signal = max(phone_signal, obj.confidence)
                events.append(self._make_event(
                    timestamp, frame_index, "phone_use", DriverState.PHONE_USE,
                    obj.confidence, Severity.CRITICAL, "Phone use detected",
                    bbox=obj.bbox,
                    metadata={"label": obj.label, "provider": obj.provider},
                ))
        return events, phone_signal

    def _make_event(
        self, timestamp: float, frame_index: int,
        signal: str, state: DriverState, score: float,
        severity: Severity, message: str,
        bbox: tuple[int, int, int, int] | None = None,
        metadata: dict[str, float | str] | None = None,
    ) -> DetectionEvent:
        return DetectionEvent(
            timestamp=timestamp, frame_index=frame_index,
            signal=signal, state=state,
            score=round(float(score), 4),
            severity=severity, message=message,
            bbox=bbox, metadata=metadata or {},
        )

    @property
    def is_calibrating(self) -> bool:
        return not self.calibrator.calibrated

    @property
    def calibration_progress(self) -> float:
        return self.calibrator.progress

    def get_events(self) -> list[DetectionEvent]:
        return list(self._events)
