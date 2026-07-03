from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from inspect_coupler import choose_best_detection
from runtime_config import RuntimeSettings, ensure_runtime_dirs


REPORT_PATH = Path("outputs/evaluation/stress_test_report.csv")


def find_default_video() -> Path | None:
    raw_dir = Path("raw_videos")
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"):
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def resize_frame(frame, width: int):
    if width <= 0 or frame.shape[1] <= width:
        return frame
    scale = width / float(frame.shape[1])
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def run_stress(video_path: Path, max_frames: int, settings: RuntimeSettings) -> dict:
    from ultralytics import YOLO
    import torch

    model = YOLO(str(settings.model_path))
    device = 0 if torch.cuda.is_available() else "cpu"
    half = bool(torch.cuda.is_available())
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_number = 0
    processed = 0
    skipped = 0
    errors = 0
    fps_values: list[float] = []
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_number += 1
            if max_frames and frame_number > max_frames:
                break

            if frame_number % max(1, settings.process_every_nth_frame) != 0:
                skipped += 1
                continue

            frame = resize_frame(frame, settings.processing_width)
            before = time.perf_counter()
            try:
                result = model.predict(
                    frame,
                    imgsz=settings.inference_size,
                    conf=0.05,
                    iou=settings.iou_threshold,
                    device=device,
                    half=half,
                    verbose=False,
                )[0]
                choose_best_detection(
                    result,
                    settings.confidence_threshold,
                    frame.shape,
                    settings.disconnected_override_confidence,
                    settings.large_disconnected_area_ratio,
                )
                processed += 1
                fps_values.append(1.0 / max(time.perf_counter() - before, 1e-6))
            except Exception:
                errors += 1
    finally:
        capture.release()

    elapsed = max(time.perf_counter() - start, 1e-6)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": str(video_path),
        "device": "GPU" if device != "cpu" else "CPU",
        "frames_read": frame_number,
        "processed_frames": processed,
        "skipped_frames": skipped,
        "dropped_frames": 0,
        "inference_errors": errors,
        "average_fps": sum(fps_values) / len(fps_values) if fps_values else 0.0,
        "minimum_fps": min(fps_values) if fps_values else 0.0,
        "maximum_fps": max(fps_values) if fps_values else 0.0,
        "end_to_end_fps": frame_number / elapsed,
        "memory_mb": memory_mb(),
        "inference_size": settings.inference_size,
        "processing_width": settings.processing_width,
        "process_every_nth_frame": settings.process_every_nth_frame,
    }


def save_report(row: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = REPORT_PATH.exists()
    with REPORT_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CouplerGuard AI long-run inference stress test.")
    parser.add_argument("video", nargs="?", help="Optional video path. Defaults to first file in raw_videos/.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames. 0 means full video.")
    parser.add_argument("--performance", action="store_true", help="Use performance-mode settings.")
    parser.add_argument("--quality", action="store_true", help="Use quality-mode settings.")
    args = parser.parse_args()

    ensure_runtime_dirs()
    settings = RuntimeSettings()
    if args.performance:
        settings = settings.with_performance_mode()
    if args.quality:
        settings = settings.with_quality_mode()

    if not settings.model_path.exists():
        print("FAIL: models/best.pt is missing.")
        return 1

    video = Path(args.video) if args.video else find_default_video()
    if video is None or not video.exists():
        print("FAIL: no video found. Place a video in raw_videos/ or pass a path.")
        return 1

    try:
        row = run_stress(video, args.max_frames, settings)
        save_report(row)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    print("Stress test complete")
    for key, value in row.items():
        print(f"{key}: {value}")
    print(f"Report saved to: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
