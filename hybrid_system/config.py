from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ThresholdConfig:
    eye_aspect_ratio: float = 0.22
    mouth_aspect_ratio: float = 0.50
    head_offset: float = 0.42
    ear_zscore_threshold: float = -2.0
    mar_zscore_threshold: float = 30.0
    ear_trigger_frames: int = 30
    pitch_trigger_frames: int = 15
    pitch_upper: float = 0.3
    pitch_lower: float = -0.2
    phone_confidence: float = 0.45
    phone_use_frames: int = 2
    phone_hold_frames: int = 12
    missing_face_frames: int = 8
    eye_closed_frames: int = 8
    drowsy_frames: int = 36
    yawn_frames: int = 6
    distracted_frames: int = 12
    classification_threshold: int = 6
    yaw_threshold: float = 0.25


@dataclass(slots=True)
class VisionConfig:
    provider: str = "auto"
    face_landmarker_model: str = "/tmp/face_landmarker.task"
    fallback_to_haar: bool = True
    process_every_n_frames: int = 2
    draw_landmarks: bool = True
    camera_index: int = 0
    process_width: int = 0
    process_height: int = 0


@dataclass(slots=True)
class ObjectDetectorConfig:
    enabled: bool = True
    provider: str = "yolo11n"
    model_path: str = str(Path.home() / "yolo11n.onnx")
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    phone_labels: list[str] = field(default_factory=lambda: ["cell phone", "phone", "mobile"])


@dataclass(slots=True)
class CalibrationConfig:
    frame_count: int = 100
    frames_start: int = 0


@dataclass(slots=True)
class LSTMConfig:
    model_path: str = ""
    input_sequence_length: int = 20
    classification_threshold: int = 3


@dataclass(slots=True)
class HeadPoseConfig:
    model_path: str = ""
    axis_draw_size: int = 50


@dataclass(slots=True)
class HardwareConfig:
    gpio_enabled: bool = False
    led_calibration_pin: int = 2
    led_inference_pin: int = 3
    button_restart_pin: int = 14
    button_pause_pin: int = 15
    beep_frequency: int = 1850
    beep_duration: float = 0.45
    beep_interval: float = 1.75
    alert_hold_seconds: float = 2.5
    alert_stable_seconds: float = 1.2
    alsa_device: str = ""


@dataclass(slots=True)
class RuntimeConfig:
    output_fps: float | None = None
    max_frames: int | None = None
    display: bool = True
    write_video: bool = True
    phone_enabled: bool = True
    alert_cooldown_seconds: float = 2.0


@dataclass(slots=True)
class ReportConfig:
    unsafe_threshold: float = 0.45
    timeline_stride_frames: int = 3


@dataclass(slots=True)
class HybridConfig:
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    object_detector: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    signal_weights: dict[str, float] = field(default_factory=dict)


def load_config(path: str | Path | None = None) -> HybridConfig:
    config = HybridConfig()
    if path is None:
        default_path = Path(__file__).parent / "configs" / "default.yaml"
        path = default_path if default_path.exists() else None
    if path is None:
        return config

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    merged = asdict(config)
    _deep_update(merged, data)
    return _from_nested(merged)


def _from_nested(data: dict[str, Any]) -> HybridConfig:
    return HybridConfig(
        thresholds=ThresholdConfig(**data.get("thresholds", {})),
        vision=VisionConfig(**data.get("vision", {})),
        object_detector=ObjectDetectorConfig(**data.get("object_detector", {})),
        calibration=CalibrationConfig(**data.get("calibration", {})),
        lstm=LSTMConfig(**data.get("lstm", {})),
        head_pose=HeadPoseConfig(**data.get("head_pose", {})),
        hardware=HardwareConfig(**data.get("hardware", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        report=ReportConfig(**data.get("report", {})),
        signal_weights=data.get("signal_weights", {}) or {},
    )


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
