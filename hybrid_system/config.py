from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


@dataclass(slots=True)
class ThresholdConfig:
    eye_aspect_ratio: float = 0.22
    mouth_aspect_ratio: float = 0.50
    head_offset: float = 0.42
    ear_zscore_threshold: float = -2.0
    mar_zscore_threshold: float = 40.0
    ear_trigger_frames: int = 30
    pitch_trigger_frames: int = 15
    pitch_upper: float = 0.1
    pitch_lower: float = -0.1
    phone_confidence: float = 0.45
    phone_use_frames: int = 2
    phone_hold_frames: int = 12
    missing_face_frames: int = 50
    eye_closed_frames: int = 8
    drowsy_frames: int = 36
    yawn_frames: int = 40
    distracted_frames: int = 12
    classification_threshold: int = 6
    yaw_threshold: float = 0.25


@dataclass(slots=True)
class VisionConfig:
    provider: str = "auto"
    face_landmarker_model: str = "models/face_landmarker.task"
    fallback_to_haar: bool = True
    process_every_n_frames: int = 2
    draw_landmarks: bool = True
    overlay_landmark_mode: str = "minimal"
    async_landmarks: bool = False
    camera_index: int = 0
    camera_fourcc: str = "MJPG"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: float = 30.0
    process_width: int = 0
    process_height: int = 0


@dataclass(slots=True)
class ObjectDetectorConfig:
    enabled: bool = True
    provider: str = "onnx"
    model_path: str = "yolov8n.onnx"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    process_every_n_frames: int = 10
    async_enabled: bool = True
    input_size: int = 640
    phone_labels: list[str] = field(default_factory=lambda: ["cell phone", "phone", "mobile"])


@dataclass(slots=True)
class CalibrationConfig:
    frame_count: int = 100
    frames_start: int = 0
    min_ear: float = 0.15
    max_mar: float = 0.60
    max_abs_pitch: float = 1.25
    max_abs_yaw: float = 1.25
    max_abs_roll: float = 1.25
    stable_window_size: int = 10
    ear_std_max: float = 0.01
    mar_std_max: float = 0.02
    pose_std_max: float = 0.08


@dataclass(slots=True)
class LSTMConfig:
    model_path: str = "models/clf_lstm.pth"
    input_sequence_length: int = 20
    classification_threshold: int = 3


@dataclass(slots=True)
class HeadPoseConfig:
    model_path: str = "models/model.pkl"
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
    beep_interval: float = 6.0
    alert_hold_seconds: float = 2.5
    alert_stable_seconds: float = 1.2
    alsa_device: str = ""


@dataclass(slots=True)
class RuntimeConfig:
    output_fps: float | None = None
    max_frames: int | None = None
    display: bool = True
    display_backend: str = "auto"
    fullscreen: bool = True
    async_output: bool = False
    kiosk_debug: bool = False
    write_video: bool = True
    phone_enabled: bool = True
    alert_cooldown_seconds: float = 2.0
    alert_voice_global_cooldown_seconds: float = 8.0
    alert_voice_repeat_cooldown_seconds: float = 24.0
    alert_voice_switch_cooldown_seconds: float = 4.0
    profile_interval_frames: int = 300


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
        default_path = PACKAGE_DIR / "configs" / "default.yaml"
        path = default_path if default_path.exists() else None
    if path is None:
        return config

    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    merged = asdict(config)
    _deep_update(merged, data)
    _resolve_paths(merged, config_path.parent)
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


def _resolve_paths(data: dict[str, Any], base_dir: Path) -> None:
    path_fields = [
        ("vision", "face_landmarker_model"),
        ("object_detector", "model_path"),
        ("lstm", "model_path"),
        ("head_pose", "model_path"),
    ]
    for section, field in path_fields:
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            continue
        value = section_data.get(field)
        if not value:
            continue
        value_str = str(value)
        value_path = Path(value_str)
        if value_path.is_absolute():
            continue
        candidates = [
            (base_dir / value_path).resolve(),
            (PROJECT_ROOT / value_path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                section_data[field] = str(candidate)
                break
        else:
            section_data[field] = str(candidates[-1])
