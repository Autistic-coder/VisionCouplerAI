from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import cv2
from PySide6.QtWidgets import QApplication
from ultralytics import YOLO

from desktop_app import MainWindow
from inspect_coupler import choose_best_detection, draw_results
from runtime_config import RuntimeSettings


OUTPUT_ROOT = Path("outputs") / "desktop_app_acceptance"
VIDEO_DIR = OUTPUT_ROOT / "videos"
SCREENSHOT_DIR = OUTPUT_ROOT / "screenshots"
FAILURE_DIR = OUTPUT_ROOT / "failures"
REPORT_PATH = OUTPUT_ROOT / "desktop_app_video_report.csv"
MODEL_PATH = Path("models") / "best.pt"
TESTING_DIR = Path("testing")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
DISCONNECTED_CLASS_ID = 1
BOX_CONFIDENCE_THRESHOLD = 0.50
LOW_CONFIDENCE_THRESHOLD = 0.65


def resize_for_app(frame, settings: RuntimeSettings):
    if settings.processing_width <= 0 or frame.shape[1] <= settings.processing_width:
        return frame
    scale = settings.processing_width / float(frame.shape[1])
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (settings.processing_width, height), interpolation=cv2.INTER_AREA)


def disconnected_boxes(result, threshold: float) -> list[float]:
    if result.boxes is None:
        return []
    confidences: list[float] = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        if class_id == DISCONNECTED_CLASS_ID and confidence >= threshold:
            confidences.append(confidence)
    return sorted(confidences, reverse=True)


def overlay_telemetry(frame, *, frame_number: int, inference_fps: float, average_fps: float, box_count: int) -> None:
    lines = [
        f"Frame: {frame_number}",
        f"Inference FPS: {inference_fps:.1f}",
        f"Average FPS: {average_fps:.1f}",
        f"Disconnected boxes >= {BOX_CONFIDENCE_THRESHOLD:.2f}: {box_count}",
    ]
    x = 16
    y = frame.shape[0] - 104
    cv2.rectangle(frame, (8, max(0, y - 28)), (430, frame.shape[0] - 8), (20, 20, 20), -1)
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
        y += 24


def save_screenshot(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def smoke_test_app() -> dict[str, str]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    screenshot_path = SCREENSHOT_DIR / "desktop_app_loaded.png"
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(screenshot_path))
    model_status = window.model_status.text()
    fps_label = window.fps_label.text()
    window.close()
    app.processEvents()
    return {
        "app_loaded": "yes",
        "model_status": model_status,
        "fps_label": fps_label,
        "screenshot": str(screenshot_path),
    }


def process_video(model: YOLO, video_path: Path, settings: RuntimeSettings) -> dict[str, str | int | float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    output_fps = max(1.0, source_fps / max(1, settings.process_every_nth_frame))

    ok, probe = capture.read()
    if not ok or probe is None:
        capture.release()
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    probe = resize_for_app(probe, settings)
    height, width = probe.shape[:2]
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    writer_path = VIDEO_DIR / f"{video_path.stem}_app_test.mp4"
    writer = cv2.VideoWriter(str(writer_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))

    processed_frames = 0
    frames_with_any_disconnected = 0
    frames_with_two_disconnected = 0
    frames_with_one_disconnected = 0
    frames_with_zero_disconnected = 0
    low_confidence_frames = 0
    no_detection_frames = 0
    minimum_confidence = 1.0
    best_two_box: tuple[int, float, object] | None = None
    first_failure: tuple[int, str, object] | None = None
    first_low_confidence: tuple[int, float, object] | None = None
    start = time.perf_counter()
    frame_number = 0
    last_inference_fps = 0.0

    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame_number += 1
        if (frame_number - 1) % max(1, settings.process_every_nth_frame) != 0:
            continue

        frame = resize_for_app(frame, settings)
        inference_start = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=settings.inference_size,
            conf=0.05,
            iou=settings.iou_threshold,
            verbose=False,
        )[0]
        inference_elapsed = max(time.perf_counter() - inference_start, 1e-6)
        last_inference_fps = 1.0 / inference_elapsed
        processed_frames += 1

        decision = choose_best_detection(
            result,
            settings.confidence_threshold,
            frame.shape,
            settings.disconnected_override_confidence,
            settings.large_disconnected_area_ratio,
        )
        confidence = float(decision.get("confidence", -1.0))
        if confidence >= 0:
            minimum_confidence = min(minimum_confidence, confidence)
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence_frames += 1
            if first_low_confidence is None:
                first_low_confidence = (frame_number, confidence, frame.copy())
        if decision.get("number_of_boxes", 0) == 0:
            no_detection_frames += 1

        disengaged_confidences = disconnected_boxes(result, BOX_CONFIDENCE_THRESHOLD)
        box_count = len(disengaged_confidences)
        if box_count > 0:
            frames_with_any_disconnected += 1
        if box_count >= 2:
            frames_with_two_disconnected += 1
            score = sum(disengaged_confidences[:2])
            if best_two_box is None or score > best_two_box[1]:
                annotated_candidate = draw_results(
                    frame,
                    result,
                    decision,
                    settings.confidence_threshold,
                    settings.show_confidence,
                    decision["status"],
                    show_all_boxes=True,
                    show_decision_box_only=False,
                )
                average_fps = processed_frames / max(time.perf_counter() - start, 1e-6)
                overlay_telemetry(
                    annotated_candidate,
                    frame_number=frame_number,
                    inference_fps=last_inference_fps,
                    average_fps=average_fps,
                    box_count=box_count,
                )
                best_two_box = (frame_number, score, annotated_candidate)
        elif box_count == 1:
            frames_with_one_disconnected += 1
            if first_failure is None:
                first_failure = (frame_number, "only_one_disconnected_box", frame.copy())
        else:
            frames_with_zero_disconnected += 1
            if first_failure is None:
                first_failure = (frame_number, "no_disconnected_box", frame.copy())

        annotated = draw_results(
            frame,
            result,
            decision,
            settings.confidence_threshold,
            settings.show_confidence,
            decision["status"],
            show_all_boxes=settings.show_all_boxes,
            show_decision_box_only=settings.show_decision_box_only,
        )
        average_fps = processed_frames / max(time.perf_counter() - start, 1e-6)
        overlay_telemetry(
            annotated,
            frame_number=frame_number,
            inference_fps=last_inference_fps,
            average_fps=average_fps,
            box_count=box_count,
        )
        writer.write(annotated)

    elapsed = time.perf_counter() - start
    capture.release()
    writer.release()

    safe_stem = video_path.stem
    if best_two_box is not None:
        save_screenshot(
            SCREENSHOT_DIR / f"{safe_stem}_best_two_disconnected_boxes_frame_{best_two_box[0]:06d}.jpg",
            best_two_box[2],
        )
    if first_failure is not None:
        frame_no, reason, failure_frame = first_failure
        save_screenshot(FAILURE_DIR / f"{safe_stem}_frame_{frame_no:06d}_{reason}.jpg", failure_frame)
    if first_low_confidence is not None:
        frame_no, confidence, low_frame = first_low_confidence
        save_screenshot(
            FAILURE_DIR / f"{safe_stem}_frame_{frame_no:06d}_low_conf_{confidence:.2f}.jpg",
            low_frame,
        )

    return {
        "video": video_path.name,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "frames_with_any_disconnected_box": frames_with_any_disconnected,
        "frames_with_two_or_more_disconnected_boxes": frames_with_two_disconnected,
        "frames_with_one_disconnected_box": frames_with_one_disconnected,
        "frames_with_zero_disconnected_boxes": frames_with_zero_disconnected,
        "low_confidence_frames": low_confidence_frames,
        "no_detection_frames": no_detection_frames,
        "minimum_decision_confidence": f"{minimum_confidence if minimum_confidence < 1.0 else 0.0:.4f}",
        "average_processing_fps": f"{processed_frames / elapsed if elapsed > 0 else 0.0:.2f}",
        "last_inference_fps": f"{last_inference_fps:.2f}",
        "annotated_video": str(writer_path),
        "two_box_detection_pass": "yes" if frames_with_two_disconnected > 0 else "no",
    }


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)

    app_info = smoke_test_app()
    print(f"Desktop app loaded: {app_info['app_loaded']}")
    print(f"Desktop app model status: {app_info['model_status']}")
    print(f"Desktop app FPS label: {app_info['fps_label']}")
    print(f"Desktop app screenshot: {app_info['screenshot']}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    videos = sorted(path for path in TESTING_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No test videos found in {TESTING_DIR}")

    settings = RuntimeSettings().with_performance_mode()
    print(
        "Runtime settings: "
        f"model={MODEL_PATH}, imgsz={settings.inference_size}, width={settings.processing_width}, "
        f"stride={settings.process_every_nth_frame}, show_all_boxes={settings.show_all_boxes}"
    )
    model = YOLO(str(MODEL_PATH))
    rows = [process_video(model, video, settings) for video in videos]

    fieldnames = list(rows[0].keys())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Report saved to: {REPORT_PATH}")
    print(f"Annotated videos saved to: {VIDEO_DIR}")
    print(f"Screenshots saved to: {SCREENSHOT_DIR}")
    print(f"Failure examples saved to: {FAILURE_DIR}")
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
