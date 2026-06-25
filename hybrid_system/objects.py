from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class ObjectObservation:
    __slots__ = ("label", "confidence", "bbox", "provider")

    def __init__(
        self,
        label: str,
        confidence: float,
        bbox: tuple[int, int, int, int],
        provider: str = "onnx",
    ) -> None:
        self.label = label
        self.confidence = confidence
        self.bbox = bbox
        self.provider = provider

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "provider": self.provider,
        }


class ObjectDetector:
    __slots__ = (
        "enabled", "confidence_threshold", "iou_threshold",
        "phone_labels", "_session", "_input_name",
        "_model", "_provider", "_frame_count",
    )

    def __init__(
        self,
        enabled: bool = False,
        model_path: str = "",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        phone_labels: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.phone_labels = phone_labels or ["cell phone", "phone", "mobile"]
        self._session = None
        self._input_name = ""
        self._model = None
        self._provider = "none"
        self._frame_count = 0

        if not enabled or not model_path:
            return

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            return

        ext = model_path_obj.suffix.lower()
        if ext == ".onnx":
            self._init_onnx(model_path_obj)
        elif ext == ".pt":
            self._init_ultralytics(model_path_obj)

    def _init_onnx(self, model_path: Path) -> None:
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._provider = "onnx"
        except Exception:
            self._session = None

    def _init_ultralytics(self, model_path: Path) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
            self._provider = "ultralytics"
        except Exception:
            self._model = None

    def detect(self, frame: np.ndarray) -> list[ObjectObservation]:
        if not self.enabled or self._provider == "none":
            return []
        if self._provider == "ultralytics" and self._model is not None:
            return self._detect_ultralytics(frame)
        return []

    def _detect_ultralytics(self, frame: np.ndarray) -> list[ObjectObservation]:
        results = self._model(frame, verbose=False, half=True)[0]
        observations: list[ObjectObservation] = []
        if results.boxes is None:
            return observations
        h, w = frame.shape[:2]
        boxes = results.boxes
        for box, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            conf_val = float(conf)
            if conf_val < self.confidence_threshold:
                continue
            label = results.names.get(int(cls_id), "unknown")
            x1, y1, x2, y2 = map(int, box.tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            observations.append(
                ObjectObservation(
                    label=label, confidence=conf_val,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                )
            )
        return observations

    @property
    def provider(self) -> str:
        return self._provider
