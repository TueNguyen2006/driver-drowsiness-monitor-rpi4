from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


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
        "input_size",
        "phone_labels", "_session", "_input_name",
        "_model", "_provider", "_frame_count",
    )

    def __init__(
        self,
        enabled: bool = False,
        model_path: str = "",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        phone_labels: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = max(160, int(input_size))
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
            self._model = cv2.dnn.readNetFromONNX(str(model_path))
            self._provider = "onnx"
        except Exception:
            self._model = None

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
        if self._provider == "onnx" and self._model is not None:
            try:
                return self._detect_onnx(frame)
            except Exception:
                return []
        try:
            if self._provider == "ultralytics" and self._model is not None:
                return self._detect_ultralytics(frame)
        except Exception:
            return []
        return []

    def _detect_onnx(self, frame: np.ndarray) -> list[ObjectObservation]:
        input_size = self.input_size
        resized, scale, pad_x, pad_y = _letterbox(frame, input_size, input_size)
        blob = cv2.dnn.blobFromImage(resized, 1 / 255.0, (input_size, input_size), swapRB=True, crop=False)
        self._model.setInput(blob)
        out = self._model.forward()
        if out.ndim != 3:
            return []
        preds = out[0].transpose(1, 0) if out.shape[1] < out.shape[2] else out[0]
        boxes: list[list[int]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        h, w = frame.shape[:2]

        for row in preds:
            cx, cy, bw, bh = row[:4].tolist()
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence_threshold:
                continue

            x1 = (cx - bw / 2 - pad_x) / scale
            y1 = (cy - bh / 2 - pad_y) / scale
            x2 = (cx + bw / 2 - pad_x) / scale
            y2 = (cy + bh / 2 - pad_y) / scale
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        if len(indices) == 0:
            return []
        if isinstance(indices, tuple):
            indices = list(indices)

        observations: list[ObjectObservation] = []
        for idx in np.array(indices).flatten():
            class_id = class_ids[int(idx)]
            label = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else "unknown"
            observations.append(
                ObjectObservation(
                    label=label,
                    confidence=float(scores[int(idx)]),
                    bbox=tuple(boxes[int(idx)]),
                    provider="onnx",
                )
            )
        return observations

    def _detect_ultralytics(self, frame: np.ndarray) -> list[ObjectObservation]:
        results = self._model.predict(
            frame,
            verbose=False,
            device="cpu",
            half=False,
            imgsz=self.input_size,
        )[0]
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
def _letterbox(image: np.ndarray, new_w: int, new_h: int) -> tuple[np.ndarray, float, float, float]:
    h, w = image.shape[:2]
    scale = min(new_w / w, new_h / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    pad_x = (new_w - resized.shape[1]) // 2
    pad_y = (new_h - resized.shape[0]) // 2
    canvas[pad_y:pad_y + resized.shape[0], pad_x:pad_x + resized.shape[1]] = resized
    return canvas, scale, float(pad_x), float(pad_y)
