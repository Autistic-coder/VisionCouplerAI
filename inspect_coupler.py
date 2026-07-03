from __future__ import annotations

import csv
import os
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from runtime_config import APP_DIR

if TYPE_CHECKING:
    from ultralytics import YOLO


MODEL_PATH = APP_DIR / "models" / "best.pt"
CLASSIFIER_MODEL_PATH = APP_DIR / "models" / "classifier.pt"
OUTPUTS_DIR = APP_DIR / "outputs"
PROCESSED_VIDEOS_DIR = OUTPUTS_DIR / "processed_videos"
PROCESSED_IMAGES_DIR = OUTPUTS_DIR / "processed_images"
DISCONNECTED_DIR = OUTPUTS_DIR / "disconnected_cases"
UNCLEAR_DIR = OUTPUTS_DIR / "unclear_cases"
LOGS_DIR = OUTPUTS_DIR / "logs"
LOG_FILE = LOGS_DIR / "inspection_log.csv"

CONFIDENCE_THRESHOLD = 0.65
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
CLASSIFIER_CROP_PADDING = 0.12
DISCONNECTED_OVERRIDE_CONFIDENCE = 0.50
LARGE_DISCONNECTED_AREA_RATIO = 0.08
SUSPICIOUS_AREA_RATIO = 0.90
SAVE_EVERY_N_ALERT_FRAMES = 20
CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}
STATUS_CONNECTED = "COUPLER CONNECTED"
STATUS_DISCONNECTED = "COUPLER DISCONNECTED"
STATUS_UNCLEAR = "UNCLEAR - MANUAL CHECK REQUIRED"
STATUS_NO_DETECTION = "NO COUPLER DETECTED"
STATUS_POSSIBLE_DISCONNECTED = "POSSIBLE DISCONNECTED - CHECK REQUIRED"
STATUS_BOX_TOO_LARGE = "UNCLEAR - BOX TOO LARGE"

GREEN = (0, 180, 0)
RED = (0, 0, 255)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)


def ensure_output_dirs() -> None:
    for folder in (
        PROCESSED_VIDEOS_DIR,
        PROCESSED_IMAGES_DIR,
        DISCONNECTED_DIR,
        UNCLEAR_DIR,
        LOGS_DIR,
        OUTPUTS_DIR / "evaluation",
        APP_DIR / "models",
        APP_DIR / "raw_videos",
        APP_DIR / "raw_photos",
    ):
        folder.mkdir(parents=True, exist_ok=True)


def load_model(model_path: Path = MODEL_PATH) -> Optional[YOLO]:
    if not model_path.exists():
        print("Trained model not found.")
        print("Please run python train_yolo.py first or place best.pt inside models/")
        return None
    from ultralytics import YOLO

    return YOLO(str(model_path))


def load_classifier_model(model_path: Path = CLASSIFIER_MODEL_PATH) -> Optional[YOLO]:
    if not model_path.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(model_path))


def append_log(
    log_path: Path,
    source: str,
    frame_number: int,
    detected_class: str,
    confidence: float,
    final_status: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["timestamp", "source", "frame_number", "detected_class", "confidence", "final_status"])
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                source,
                frame_number,
                detected_class,
                f"{confidence:.4f}" if confidence >= 0 else "",
                final_status,
            ]
        )


def write_detection_log(
    log_path: Path,
    source: str,
    frame_number: int,
    decision: dict,
    fps: float = 0.0,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "source",
                    "frame_number",
                    "final_status",
                    "final_class",
                    "best_confidence",
                    "largest_disconnected_box_area_ratio",
                    "disconnected_override_used",
                    "number_of_boxes",
                    "fps",
                ]
            )
        writer.writerow(
            [
                datetime.now().isoformat(timespec="seconds"),
                source,
                frame_number,
                decision.get("status", ""),
                decision.get("detected_class", ""),
                f"{decision.get('confidence', -1):.4f}" if decision.get("confidence", -1) >= 0 else "",
                f"{decision.get('largest_disconnected_box_area_ratio', 0):.4f}",
                bool(decision.get("disconnected_override_used", False)),
                decision.get("number_of_boxes", 0),
                f"{fps:.2f}",
            ]
        )


def status_color(status: str) -> tuple[int, int, int]:
    if status == STATUS_CONNECTED:
        return GREEN
    if status == STATUS_DISCONNECTED:
        return RED
    if status in {STATUS_UNCLEAR, STATUS_POSSIBLE_DISCONNECTED, STATUS_BOX_TOO_LARGE}:
        return YELLOW
    return ORANGE


def save_case_frame(
    frame: np.ndarray,
    source_name: str,
    status: str,
    frame_number: int,
    save_screenshots: bool = True,
    save_every_n: int = 1,
) -> None:
    if not save_screenshots:
        return
    if save_every_n > 1 and frame_number % save_every_n != 0:
        return
    if status not in {STATUS_DISCONNECTED, STATUS_UNCLEAR, STATUS_POSSIBLE_DISCONNECTED, STATUS_BOX_TOO_LARGE}:
        return

    output_dir = DISCONNECTED_DIR if status == STATUS_DISCONNECTED else UNCLEAR_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_source = Path(str(source_name)).stem or "webcam"
    case_name = "disconnected" if status == STATUS_DISCONNECTED else "unclear"
    output_path = output_dir / f"{safe_source}_frame_{frame_number:06d}_{case_name}_{timestamp}.jpg"
    cv2.imwrite(str(output_path), frame)


def _box_area_ratio(box, frame_area: float) -> float:
    x1, y1, x2, y2 = map(float, box.xyxy[0])
    return max(0.0, (x2 - x1) * (y2 - y1)) / max(frame_area, 1.0)


def choose_best_detection(
    result,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    frame_shape: tuple[int, int, int] | None = None,
    disconnected_override_confidence: float = DISCONNECTED_OVERRIDE_CONFIDENCE,
    large_disconnected_area_ratio: float = LARGE_DISCONNECTED_AREA_RATIO,
) -> dict:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "class_id": None,
            "detected_class": "",
            "confidence": -1.0,
            "status": STATUS_NO_DETECTION,
            "color": ORANGE,
            "decision_box_index": None,
            "disconnected_override_used": False,
            "largest_disconnected_box_area_ratio": 0.0,
            "number_of_boxes": 0,
            "box_details": [],
        }

    if frame_shape is not None:
        frame_area = float(frame_shape[0] * frame_shape[1])
    else:
        frame_area = 1.0

    details = []
    engaged = []
    disengaged = []
    suspicious = []
    for index, box in enumerate(boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        area_ratio = _box_area_ratio(box, frame_area)
        detail = {
            "index": index,
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"unknown_class_{class_id}"),
            "confidence": confidence,
            "area_ratio": area_ratio,
            "score": confidence + min(area_ratio, 0.50) * 0.30,
        }
        details.append(detail)
        if area_ratio > SUSPICIOUS_AREA_RATIO:
            suspicious.append(detail)
        if class_id == 0:
            engaged.append(detail)
        elif class_id == 1:
            disengaged.append(detail)

    if suspicious:
        decision = max(suspicious, key=lambda item: item["confidence"])
        return {
            "class_id": decision["class_id"],
            "detected_class": decision["class_name"],
            "confidence": decision["confidence"],
            "status": STATUS_BOX_TOO_LARGE,
            "color": YELLOW,
            "decision_box_index": decision["index"],
            "disconnected_override_used": False,
            "largest_disconnected_box_area_ratio": max((item["area_ratio"] for item in disengaged), default=0.0),
            "number_of_boxes": len(details),
            "box_details": details,
        }

    largest_disconnected = max(disengaged, key=lambda item: item["area_ratio"], default=None)
    best_disconnected = max(disengaged, key=lambda item: item["score"], default=None)
    best_engaged = max(engaged, key=lambda item: item["confidence"], default=None)

    if largest_disconnected and (
        largest_disconnected["confidence"] >= disconnected_override_confidence
        and largest_disconnected["area_ratio"] >= large_disconnected_area_ratio
    ):
        decision = largest_disconnected
        status = STATUS_DISCONNECTED
        override = True
    elif largest_disconnected and (
        0.40 <= largest_disconnected["confidence"] < disconnected_override_confidence
        and largest_disconnected["area_ratio"] >= large_disconnected_area_ratio
    ):
        decision = largest_disconnected
        status = STATUS_POSSIBLE_DISCONNECTED
        override = False
    elif best_disconnected and best_disconnected["confidence"] >= confidence_threshold:
        decision = best_disconnected
        status = STATUS_DISCONNECTED
        override = False
    elif best_engaged and best_engaged["confidence"] >= confidence_threshold:
        decision = best_engaged
        status = STATUS_CONNECTED
        override = False
    else:
        decision = max(details, key=lambda item: item["confidence"])
        status = STATUS_UNCLEAR
        override = False

    return {
        "class_id": decision["class_id"],
        "detected_class": decision["class_name"],
        "confidence": decision["confidence"],
        "status": status,
        "color": status_color(status),
        "decision_box_index": decision["index"],
        "disconnected_override_used": override,
        "largest_disconnected_box_area_ratio": largest_disconnected["area_ratio"] if largest_disconnected else 0.0,
        "number_of_boxes": len(details),
        "box_details": details,
    }


def _class_id_from_name(class_name: str) -> int | None:
    normalized = class_name.lower().replace("-", "_").replace(" ", "_")
    if "disconnected" in normalized or "disengaged" in normalized:
        return 1
    if "connected" in normalized or "engaged" in normalized:
        return 0
    return None


def _classifier_prediction(classifier_result) -> tuple[int | None, str, float]:
    probs = getattr(classifier_result, "probs", None)
    if probs is None:
        return None, "", -1.0

    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    names = getattr(classifier_result, "names", {}) or {}
    class_name = str(names.get(class_id, CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")))
    mapped_class_id = _class_id_from_name(class_name)
    return (mapped_class_id if mapped_class_id is not None else class_id), class_name, confidence


def crop_decision_region(
    frame: np.ndarray,
    result,
    decision: dict,
    padding_ratio: float = CLASSIFIER_CROP_PADDING,
) -> np.ndarray | None:
    boxes = getattr(result, "boxes", None)
    box_index = decision.get("decision_box_index")
    if boxes is None or box_index is None or box_index < 0 or box_index >= len(boxes):
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(float, boxes[box_index].xyxy[0])
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return frame[top:bottom, left:right]


def apply_classifier_decision(
    decision: dict,
    classifier_class_id: int | None,
    classifier_class_name: str,
    classifier_confidence: float,
    threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> dict:
    refined = dict(decision)
    refined.update(
        {
            "classifier_available": True,
            "classifier_class_id": classifier_class_id,
            "classifier_class": classifier_class_name,
            "classifier_confidence": classifier_confidence,
            "classifier_used": False,
            "classifier_reason": "below_threshold",
        }
    )

    if classifier_class_id not in CLASS_NAMES or classifier_confidence < threshold:
        return refined

    classifier_status = STATUS_DISCONNECTED if classifier_class_id == 1 else STATUS_CONNECTED
    classifier_class = CLASS_NAMES[classifier_class_id]
    detector_class_id = refined.get("class_id")
    detector_status = refined.get("status")

    def set_classifier_result(status: str, reason: str) -> None:
        refined.update(
            {
                "class_id": classifier_class_id,
                "detected_class": classifier_class,
                "confidence": classifier_confidence,
                "status": status,
                "color": status_color(status),
                "classifier_used": True,
                "classifier_reason": reason,
            }
        )

    if detector_class_id == classifier_class_id:
        refined["classifier_used"] = True
        refined["classifier_reason"] = "confirmed_detector"
        refined["confidence"] = max(float(refined.get("confidence", -1.0)), classifier_confidence)
        return refined

    if detector_status in {STATUS_UNCLEAR, STATUS_NO_DETECTION}:
        set_classifier_result(classifier_status, "resolved_unclear")
        return refined

    if detector_status == STATUS_POSSIBLE_DISCONNECTED and classifier_class_id == 1:
        set_classifier_result(STATUS_DISCONNECTED, "confirmed_possible_disconnected")
        return refined

    if classifier_class_id == 1:
        set_classifier_result(STATUS_POSSIBLE_DISCONNECTED, "classifier_detector_disagreement")
    else:
        refined.update(
            {
                "status": STATUS_UNCLEAR,
                "color": status_color(STATUS_UNCLEAR),
                "classifier_used": True,
                "classifier_reason": "classifier_detector_disagreement",
            }
        )
    return refined


def classify_decision_crop(
    classifier_model: YOLO | None,
    frame: np.ndarray,
    result,
    decision: dict,
    threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
    imgsz: int = 224,
    device=None,
    half: bool = False,
) -> dict:
    if classifier_model is None:
        refined = dict(decision)
        refined.update(
            {
                "classifier_available": False,
                "classifier_class_id": None,
                "classifier_class": "",
                "classifier_confidence": -1.0,
                "classifier_used": False,
                "classifier_reason": "model_missing",
            }
        )
        return refined

    crop = crop_decision_region(frame, result, decision)
    if crop is None:
        refined = dict(decision)
        refined.update(
            {
                "classifier_available": True,
                "classifier_class_id": None,
                "classifier_class": "",
                "classifier_confidence": -1.0,
                "classifier_used": False,
                "classifier_reason": "no_decision_crop",
            }
        )
        return refined

    classifier_result = classifier_model.predict(crop, imgsz=imgsz, device=device, half=half, verbose=False)[0]
    class_id, class_name, confidence = _classifier_prediction(classifier_result)
    return apply_classifier_decision(decision, class_id, class_name, confidence, threshold)


def draw_status_banner(frame: np.ndarray, status: str, confidence: float, color: tuple[int, int, int], show_confidence: bool) -> None:
    text = status
    if show_confidence and confidence >= 0:
        text = f"{text}  CONF {confidence:.2f}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 62), (20, 20, 20), -1)
    cv2.putText(frame, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)


def draw_results(
    frame: np.ndarray,
    result,
    decision: dict,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    show_confidence: bool = True,
    display_status: str | None = None,
    show_all_boxes: bool = True,
    show_decision_box_only: bool = False,
) -> np.ndarray:
    annotated = frame.copy()
    boxes = result.boxes

    if boxes is not None:
        for box_index, box in enumerate(boxes):
            if show_decision_box_only and box_index != decision.get("decision_box_index"):
                continue
            if not show_all_boxes and box_index != decision.get("decision_box_index"):
                continue
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = CLASS_NAMES.get(class_id, f"unknown_class_{class_id}")
            if confidence < confidence_threshold:
                color = YELLOW
            elif class_id == 0:
                color = GREEN
            elif class_id == 1:
                color = RED
            else:
                color = YELLOW

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_decision_box = box_index == decision.get("decision_box_index")
            thickness = 5 if is_decision_box and decision.get("status") == STATUS_DISCONNECTED else 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = class_name
            if is_decision_box and decision.get("status") == STATUS_DISCONNECTED and class_id == 1:
                label = "DISCONNECTED EVIDENCE"
            if show_confidence:
                label = f"{label} {confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(82, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    final_status = display_status or decision["status"]
    draw_status_banner(annotated, final_status, decision["confidence"], status_color(final_status), show_confidence)
    return annotated


def inspect_frame(
    model: YOLO,
    frame: np.ndarray,
    source_name: str,
    frame_number: int = 0,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    show_confidence: bool = True,
    save_screenshots: bool = True,
    log_path: Path = LOG_FILE,
    display_status: str | None = None,
    classifier_model: YOLO | None = None,
    classifier_threshold: float = CLASSIFIER_CONFIDENCE_THRESHOLD,
) -> tuple[np.ndarray, dict]:
    ensure_output_dirs()
    result = model(frame, verbose=False)[0]
    decision = choose_best_detection(result, confidence_threshold, frame.shape)
    decision = classify_decision_crop(classifier_model, frame, result, decision, classifier_threshold)
    annotated = draw_results(frame, result, decision, confidence_threshold, show_confidence, display_status)
    save_case_frame(annotated, source_name, decision["status"], frame_number, save_screenshots, SAVE_EVERY_N_ALERT_FRAMES)
    log_decision = {**decision, "status": display_status or decision["status"]}
    write_detection_log(log_path, source_name, frame_number, log_decision)
    return annotated, decision


def majority_status(statuses: deque[str]) -> str:
    if not statuses:
        return STATUS_NO_DETECTION
    priority = {
        STATUS_DISCONNECTED: 5,
        STATUS_POSSIBLE_DISCONNECTED: 4,
        STATUS_UNCLEAR: 3,
        STATUS_BOX_TOO_LARGE: 3,
        STATUS_CONNECTED: 2,
        STATUS_NO_DETECTION: 1,
    }
    counts = Counter(statuses)
    return max(counts, key=lambda status: (counts[status], priority.get(status, 0)))


def run_on_image(image_path: str = "raw_photos/test.jpg") -> None:
    ensure_output_dirs()
    model = load_model()
    if model is None:
        return
    classifier_model = load_classifier_model()

    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {path}")
        return

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}")
        return

    annotated, decision = inspect_frame(model, frame, str(path), frame_number=0, classifier_model=classifier_model)
    output_path = PROCESSED_IMAGES_DIR / f"{path.stem}_processed.jpg"
    cv2.imwrite(str(output_path), annotated)
    print(f"Decision: {decision['status']}")
    print(f"Confidence: {decision['confidence']:.4f}" if decision["confidence"] >= 0 else "Confidence: none")
    print(f"Processed image saved to: {output_path}")
    cv2.imshow("Coupler Inspection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_video(video_path: str = "raw_videos/test.mp4") -> None:
    ensure_output_dirs()
    model = load_model()
    if model is None:
        return
    classifier_model = load_classifier_model()

    path = Path(video_path)
    if not path.exists():
        print(f"Video not found: {path}")
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"Could not open video: {path}")
        return

    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = PROCESSED_VIDEOS_DIR / f"{path.stem}_processed.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_number = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        annotated, _ = inspect_frame(model, frame, str(path), frame_number=frame_number, classifier_model=classifier_model)
        writer.write(annotated)
        cv2.imshow("Coupler Inspection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_number += 1

    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved to: {output_path}")


def _open_live_camera(camera_source: int | str) -> cv2.VideoCapture:
    if isinstance(camera_source, int):
        return cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)
    if "://" in camera_source:
        if camera_source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        capture = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture
    return cv2.VideoCapture(camera_source)


def run_on_webcam(camera_index: int | str = 0, vote_window: int = 7) -> None:
    ensure_output_dirs()
    model = load_model()
    if model is None:
        return
    classifier_model = load_classifier_model()

    capture = _open_live_camera(camera_index)
    if not capture.isOpened():
        print(f"Could not open camera source {camera_index}.")
        return

    statuses: deque[str] = deque(maxlen=vote_window)
    frame_number = 0
    print("Running webcam inspection. Press q to quit.")
    while True:
        success, frame = capture.read()
        if not success:
            print("Could not read frame from webcam.")
            break

        result = model(frame, verbose=False)[0]
        decision = choose_best_detection(result, CONFIDENCE_THRESHOLD, frame.shape)
        decision = classify_decision_crop(classifier_model, frame, result, decision)
        statuses.append(decision["status"])
        voted_status = majority_status(statuses)
        annotated = draw_results(frame, result, decision, CONFIDENCE_THRESHOLD, True, voted_status)
        source_name = f"camera_{camera_index}"
        save_case_frame(annotated, source_name, voted_status, frame_number)
        append_log(LOG_FILE, source_name, frame_number, decision["detected_class"], decision["confidence"], voted_status)

        cv2.imshow("Coupler Inspection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_number += 1

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    run_on_webcam()


if __name__ == "__main__":
    main()
