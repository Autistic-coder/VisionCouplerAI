from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


APP_NAME = "CouplerGuard AI"
DEFAULT_CAMERA_INDEX = 0
FRAME_QUEUE_SIZE = 3
LOG_QUEUE_SIZE = 12
TARGET_DISPLAY_FPS = 20


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_DIR = app_base_dir()
CAMERA_CONFIG_PATH = APP_DIR / "camera_config.json"
MODEL_PATH = APP_DIR / "models" / "best.pt"
CLASSIFIER_MODEL_PATH = APP_DIR / "models" / "classifier.pt"
OUTPUTS_DIR = APP_DIR / "outputs"
CONFIDENCE_THRESHOLD = 0.65
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
DISCONNECTED_OVERRIDE_CONFIDENCE = 0.50
LARGE_DISCONNECTED_AREA_RATIO = 0.08
SAVE_EVERY_N_ALERT_FRAMES = 20

SCREENSHOT_DIR = OUTPUTS_DIR / "screenshots"
PROCESSED_VIDEO_DIR = OUTPUTS_DIR / "processed_videos"

INFERENCE_SIZES = [320, 416, 512, 640]
PROCESSING_WIDTHS = [480, 640, 960, 1280]
FRAME_SKIP_OPTIONS = [1, 2, 3, 4, 5]


@dataclass(frozen=True)
class CameraSource:
    name: str
    source: int | str


@dataclass(frozen=True)
class RuntimeSettings:
    model_path: Path = MODEL_PATH
    classifier_model_path: Path = CLASSIFIER_MODEL_PATH
    classifier_confidence_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD
    classifier_inference_size: int = 224
    use_classifier: bool = True
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    disconnected_override_confidence: float = DISCONNECTED_OVERRIDE_CONFIDENCE
    large_disconnected_area_ratio: float = LARGE_DISCONNECTED_AREA_RATIO
    iou_threshold: float = 0.45
    inference_size: int = 416
    processing_width: int = 640
    process_every_nth_frame: int = 2
    majority_vote_window: int = 7
    show_decision_box_only: bool = True
    show_all_boxes: bool = False
    show_confidence: bool = True
    save_screenshots: bool = False
    save_processed_video: bool = False
    save_every_n_alert_frames: int = SAVE_EVERY_N_ALERT_FRAMES
    target_display_fps: int = TARGET_DISPLAY_FPS

    def with_performance_mode(self) -> "RuntimeSettings":
        return replace(
            self,
            inference_size=640,
            processing_width=960,
            process_every_nth_frame=1,
            show_decision_box_only=False,
            show_all_boxes=True,
            save_screenshots=False,
            save_processed_video=False,
        )

    def with_quality_mode(self) -> "RuntimeSettings":
        return replace(
            self,
            inference_size=640,
            processing_width=960,
            process_every_nth_frame=1,
            show_decision_box_only=False,
            show_all_boxes=True,
            show_confidence=True,
        )


def _parse_capture_source(value: Any) -> int | str | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    source = value.strip()
    if not source:
        return None
    if source.isdigit():
        return int(source)
    return source


def load_ethernet_camera_sources(config_path: Path = CAMERA_CONFIG_PATH) -> list[CameraSource]:
    if not config_path.exists():
        return []

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    cameras = data.get("ethernet_cameras", [])
    if isinstance(cameras, dict):
        cameras = [cameras]
    if not isinstance(cameras, list):
        return []

    sources: list[CameraSource] = []
    for index, camera in enumerate(cameras, start=1):
        if not isinstance(camera, dict):
            continue
        if not camera.get("enabled", True):
            continue
        source = _parse_capture_source(camera.get("url") or camera.get("source"))
        if source is None:
            continue
        name = str(camera.get("name") or f"Ethernet Camera {index}").strip()
        sources.append(CameraSource(name=name, source=source))
    return sources


def ensure_runtime_dirs() -> None:
    for folder in (
        OUTPUTS_DIR,
        SCREENSHOT_DIR,
        PROCESSED_VIDEO_DIR,
        OUTPUTS_DIR / "evaluation",
        OUTPUTS_DIR / "disconnected_cases",
        OUTPUTS_DIR / "unclear_cases",
        APP_DIR / "models",
        APP_DIR / "raw_videos",
        APP_DIR / "raw_photos",
        APP_DIR / "reference_images",
    ):
        folder.mkdir(parents=True, exist_ok=True)
