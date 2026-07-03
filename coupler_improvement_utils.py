from __future__ import annotations

import csv
import hashlib
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from inspect_coupler import (
    CLASS_NAMES,
    MODEL_PATH,
    STATUS_CONNECTED,
    STATUS_DISCONNECTED,
    STATUS_NO_DETECTION,
    STATUS_POSSIBLE_DISCONNECTED,
    STATUS_UNCLEAR,
    draw_results,
    choose_best_detection,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
EXPECTED_CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}
DEFAULT_TESTING_DIR = Path("testing")
LEGACY_TESTING_DIR = Path("testinggg")


@dataclass
class VideoStats:
    video: str
    total_frames: int = 0
    processed_frames: int = 0
    no_detection_frames: int = 0
    engaged_frames: int = 0
    disengaged_frames: int = 0
    unclear_frames: int = 0
    low_confidence_frames: int = 0
    confidence_values: list[float] = field(default_factory=list)
    engaged_confidences: list[float] = field(default_factory=list)
    disengaged_confidences: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def as_row(self) -> dict[str, str | int | float]:
        average_confidence = mean(self.confidence_values)
        average_engaged = mean(self.engaged_confidences)
        average_disengaged = mean(self.disengaged_confidences)
        min_confidence = min(self.confidence_values) if self.confidence_values else 0.0
        low_percent = percent(self.low_confidence_frames, self.processed_frames)
        fps = self.processed_frames / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
        return {
            "video": self.video,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "frames_with_no_detection": self.no_detection_frames,
            "frames_classified_as_engaged": self.engaged_frames,
            "frames_classified_as_disengaged": self.disengaged_frames,
            "frames_classified_as_unclear": self.unclear_frames,
            "average_confidence": f"{average_confidence:.4f}",
            "average_engaged_confidence": f"{average_engaged:.4f}",
            "average_disengaged_confidence": f"{average_disengaged:.4f}",
            "minimum_confidence": f"{min_confidence:.4f}",
            "low_confidence_frame_percentage": f"{low_percent:.2f}",
            "approximate_fps": f"{fps:.2f}",
        }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percent(part: int | float, whole: int | float) -> float:
    return (float(part) / float(whole) * 100.0) if whole else 0.0


def resolve_testing_dir(preferred: Path = DEFAULT_TESTING_DIR) -> Path | None:
    if preferred.exists():
        return preferred
    if preferred == DEFAULT_TESTING_DIR and LEGACY_TESTING_DIR.exists():
        print("testing/ was not found. Using existing folder testinggg/ for now.")
        print("For the long-term workflow, rename testinggg/ to testing/ or add new videos to testing/.")
        return LEGACY_TESTING_DIR
    return None


def list_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_required_model(model_path: Path = MODEL_PATH) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}. Train first or place best.pt inside models/.")
    return YOLO(str(model_path))


def classify_decision(decision: dict) -> str:
    status = decision.get("status", "")
    class_id = decision.get("class_id")
    if status == STATUS_NO_DETECTION:
        return "no_detection"
    if status in {STATUS_DISCONNECTED, STATUS_POSSIBLE_DISCONNECTED} or class_id == 1:
        return "disengaged"
    if status == STATUS_CONNECTED or class_id == 0:
        return "engaged"
    return "unclear"


def update_stats(stats: VideoStats, decision: dict, low_confidence_threshold: float) -> None:
    stats.processed_frames += 1
    confidence = float(decision.get("confidence", -1.0))
    category = classify_decision(decision)

    if category == "no_detection":
        stats.no_detection_frames += 1
        stats.low_confidence_frames += 1
        return

    if confidence >= 0:
        stats.confidence_values.append(confidence)
    if confidence < low_confidence_threshold:
        stats.low_confidence_frames += 1

    if category == "engaged":
        stats.engaged_frames += 1
        if confidence >= 0:
            stats.engaged_confidences.append(confidence)
    elif category == "disengaged":
        stats.disengaged_frames += 1
        if confidence >= 0:
            stats.disengaged_confidences.append(confidence)
    else:
        stats.unclear_frames += 1


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_stem(path: Path) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in path.stem)


def save_frame(folder: Path, video_path: Path, frame_number: int, frame: np.ndarray, suffix: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    output_path = folder / f"{safe_stem(video_path)}_frame_{frame_number:06d}_{suffix}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def frame_hash(frame: np.ndarray, hash_size: int = 12) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(small.tobytes()).hexdigest()


def blur_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def edge_density(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def skin_ratio(frame: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    return float(np.count_nonzero(mask)) / float(mask.size)


def perspective_angle_score(decision: dict) -> float:
    details = decision.get("box_details", [])
    if not details:
        return 0.0
    ratios = []
    for item in details:
        area = float(item.get("area_ratio", 0.0))
        if area > 0:
            ratios.append(area)
    return max(ratios) if ratios else 0.0


def validate_yolo_label(label_path: Path) -> tuple[bool, str, list[int]]:
    if not label_path.exists():
        return False, "missing label", []
    if label_path.stat().st_size == 0:
        return False, "empty label", []

    class_ids: list[int] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            return False, f"line {line_number}: expected 5 YOLO values", class_ids
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            return False, f"line {line_number}: non-numeric YOLO value", class_ids
        if class_id not in EXPECTED_CLASS_NAMES:
            return False, f"line {line_number}: invalid class ID {class_id}", class_ids
        if any(math.isnan(value) or value < 0.0 or value > 1.0 for value in coords):
            return False, f"line {line_number}: coordinates must be normalized between 0 and 1", class_ids
        class_ids.append(class_id)

    if not class_ids:
        return False, "empty label", class_ids
    return True, "", class_ids


def copy_issue_pair(image_path: Path, label_path: Path | None, issue_dir: Path, reason: str) -> None:
    destination = issue_dir / reason.replace(" ", "_")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination / image_path.name)
    if label_path is not None and label_path.exists():
        shutil.copy2(label_path, destination / label_path.name)


def dataset_class_counts(labels_root: Path = Path("annotated_images") / "labels") -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for label_path in labels_root.rglob("*.txt"):
        ok, _, class_ids = validate_yolo_label(label_path)
        if not ok:
            continue
        for class_id in class_ids:
            if class_id in counts:
                counts[class_id] += 1
    return counts


def run_video_evaluation(
    model: YOLO,
    video_path: Path,
    annotated_video_path: Path | None,
    low_confidence_dir: Path | None,
    no_detection_dir: Path | None,
    disconnected_failure_dir: Path | None,
    low_confidence_threshold: float = 0.50,
    good_confidence_threshold: float = 0.70,
    frame_stride: int = 1,
    max_saved_per_reason: int = 80,
) -> VideoStats:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    writer = None
    if annotated_video_path is not None:
        annotated_video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    stats = VideoStats(video=video_path.name, total_frames=total_frames)
    saved_counts = {"low": 0, "none": 0, "disconnected_failure": 0}
    frame_number = 0
    start = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_number += 1
        if frame_stride > 1 and frame_number % frame_stride != 0:
            if writer is not None:
                writer.write(frame)
            continue

        result = model(frame, verbose=False)[0]
        decision = choose_best_detection(result, frame_shape=frame.shape)
        update_stats(stats, decision, low_confidence_threshold)

        confidence = float(decision.get("confidence", -1.0))
        status = decision.get("status", "")
        no_detection = status == STATUS_NO_DETECTION
        low_confidence = no_detection or confidence < low_confidence_threshold
        possible_disconnected_failure = (
            status in {STATUS_UNCLEAR, STATUS_NO_DETECTION, STATUS_POSSIBLE_DISCONNECTED}
            or (decision.get("class_id") == 1 and confidence < good_confidence_threshold)
            or float(decision.get("largest_disconnected_box_area_ratio", 0.0)) >= 0.04
            and confidence < good_confidence_threshold
        )

        if low_confidence and low_confidence_dir is not None and saved_counts["low"] < max_saved_per_reason:
            save_frame(low_confidence_dir, video_path, frame_number, frame, "low_conf")
            saved_counts["low"] += 1
        if no_detection and no_detection_dir is not None and saved_counts["none"] < max_saved_per_reason:
            save_frame(no_detection_dir, video_path, frame_number, frame, "no_detection")
            saved_counts["none"] += 1
        if (
            possible_disconnected_failure
            and disconnected_failure_dir is not None
            and saved_counts["disconnected_failure"] < max_saved_per_reason
        ):
            save_frame(disconnected_failure_dir, video_path, frame_number, frame, "possible_disconnected_failure")
            saved_counts["disconnected_failure"] += 1

        if writer is not None:
            annotated = draw_results(
                frame,
                result,
                decision,
                show_all_boxes=True,
                show_decision_box_only=False,
                show_confidence=True,
            )
            writer.write(annotated)

    stats.elapsed_seconds = time.perf_counter() - start
    capture.release()
    if writer is not None:
        writer.release()
    return stats
