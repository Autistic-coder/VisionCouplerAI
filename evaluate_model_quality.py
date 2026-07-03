from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from coupler_improvement_utils import (
    DEFAULT_TESTING_DIR,
    EXPECTED_CLASS_NAMES,
    classify_decision,
    list_images,
    list_videos,
    load_required_model,
    mean,
    percent,
    resolve_testing_dir,
    run_video_evaluation,
    write_csv,
)
from inspect_coupler import choose_best_detection, draw_results


DATA_YAML_PATH = Path("annotated_images") / "data.yaml"
MODEL_QUALITY_REPORT = Path("outputs") / "evaluation" / "model_quality_report.csv"
TESTING_QUALITY_REPORT = Path("outputs") / "evaluation" / "testing_video_quality_report.csv"
SAMPLE_DIR = Path("outputs") / "evaluation" / "sample_predictions"


def as_float(value, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(np.nanmean(value))
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return float(np.nanmean(np.array(value, dtype=float)))
        return float(value)
    except (TypeError, ValueError):
        return default


def per_class_value(values, class_index: int) -> float:
    try:
        array = np.array(values, dtype=float)
        if array.ndim == 0:
            return float(array)
        if len(array) > class_index:
            return float(array[class_index])
    except (TypeError, ValueError):
        pass
    return 0.0


def evaluate_split(model: YOLO, split: str) -> list[dict]:
    if not DATA_YAML_PATH.exists():
        print("Cannot evaluate dataset splits: annotated_images/data.yaml is missing.")
        return []

    print(f"Evaluating {split} split")
    metrics = model.val(data=str(DATA_YAML_PATH), split=split, imgsz=640, plots=True, verbose=False)
    box = getattr(metrics, "box", None)
    rows = []
    rows.append(
        {
            "split": split,
            "class_id": "all",
            "class_name": "all",
            "precision": f"{as_float(getattr(box, 'mp', 0.0)):.4f}",
            "recall": f"{as_float(getattr(box, 'mr', 0.0)):.4f}",
            "map50": f"{as_float(getattr(box, 'map50', 0.0)):.4f}",
            "map50_95": f"{as_float(getattr(box, 'map', 0.0)):.4f}",
        }
    )

    precision_values = getattr(box, "p", [])
    recall_values = getattr(box, "r", [])
    map50_values = getattr(box, "ap50", [])
    map_values = getattr(box, "ap", [])
    for class_id, class_name in EXPECTED_CLASS_NAMES.items():
        rows.append(
            {
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "precision": f"{per_class_value(precision_values, class_id):.4f}",
                "recall": f"{per_class_value(recall_values, class_id):.4f}",
                "map50": f"{per_class_value(map50_values, class_id):.4f}",
                "map50_95": f"{per_class_value(map_values, class_id):.4f}",
            }
        )
    return rows


def save_sample_predictions(model: YOLO, max_samples_per_split: int = 20) -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        images_dir = Path("annotated_images") / "images" / split
        images = list_images(images_dir)[:max_samples_per_split]
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            result = model(frame, verbose=False)[0]
            decision = choose_best_detection(result, frame_shape=frame.shape)
            annotated = draw_results(frame, result, decision, show_all_boxes=True)
            output_path = SAMPLE_DIR / f"{split}_{image_path.stem}_prediction.jpg"
            cv2.imwrite(str(output_path), annotated)
    print(f"Sample predictions saved to: {SAMPLE_DIR}")


def summarize_testing_rows(rows: list[dict]) -> dict[str, float]:
    processed = sum(int(row["processed_frames"]) for row in rows)
    no_detection = sum(int(row["frames_with_no_detection"]) for row in rows)
    low_percent_values = [float(row["low_confidence_frame_percentage"]) for row in rows]
    confidences = [float(row["average_confidence"]) for row in rows if float(row["average_confidence"]) > 0]
    disconnected_confidences = [
        float(row["average_disengaged_confidence"]) for row in rows if float(row["average_disengaged_confidence"]) > 0
    ]
    disengaged = sum(int(row["frames_classified_as_disengaged"]) for row in rows)
    fps_values = [float(row["approximate_fps"]) for row in rows if float(row["approximate_fps"]) > 0]
    return {
        "average_testing_confidence": mean(confidences),
        "average_disconnected_confidence": mean(disconnected_confidences),
        "disconnected_detection_rate": percent(disengaged, processed),
        "no_detection_rate": percent(no_detection, processed),
        "low_confidence_rate": mean(low_percent_values),
        "fps": mean(fps_values),
    }


def evaluate_testing_quality(
    model: YOLO,
    testing_dir: Path | None = None,
    low_confidence_threshold: float = 0.50,
    good_confidence_threshold: float = 0.70,
    frame_stride: int = 2,
) -> tuple[list[dict], dict[str, float]]:
    source_dir = testing_dir or resolve_testing_dir(DEFAULT_TESTING_DIR)
    if source_dir is None:
        print("No testing folder found. Skipping testing-video quality evaluation.")
        return [], {}

    videos = list_videos(source_dir)
    rows = []
    for video_path in videos:
        print(f"Evaluating testing quality: {video_path.name}")
        stats = run_video_evaluation(
            model=model,
            video_path=video_path,
            annotated_video_path=None,
            low_confidence_dir=None,
            no_detection_dir=None,
            disconnected_failure_dir=None,
            low_confidence_threshold=low_confidence_threshold,
            good_confidence_threshold=good_confidence_threshold,
            frame_stride=frame_stride,
        )
        rows.append(stats.as_row())

    if rows:
        write_csv(
            TESTING_QUALITY_REPORT,
            [
                "video",
                "total_frames",
                "processed_frames",
                "frames_with_no_detection",
                "frames_classified_as_engaged",
                "frames_classified_as_disengaged",
                "frames_classified_as_unclear",
                "average_confidence",
                "average_engaged_confidence",
                "average_disengaged_confidence",
                "minimum_confidence",
                "low_confidence_frame_percentage",
                "approximate_fps",
            ],
            rows,
        )
    return rows, summarize_testing_rows(rows)


def evaluate_model_quality(
    testing_dir: Path | None = None,
    low_confidence_threshold: float = 0.50,
    good_confidence_threshold: float = 0.70,
    frame_stride: int = 2,
    save_samples: bool = True,
) -> dict[str, float]:
    model = load_required_model()
    split_rows = []
    for split in ("val", "test"):
        split_rows.extend(evaluate_split(model, split))

    if split_rows:
        write_csv(
            MODEL_QUALITY_REPORT,
            ["split", "class_id", "class_name", "precision", "recall", "map50", "map50_95"],
            split_rows,
        )
        print(f"Model quality report saved to: {MODEL_QUALITY_REPORT}")

    if save_samples:
        save_sample_predictions(model)

    testing_rows, testing_summary = evaluate_testing_quality(
        model,
        testing_dir=testing_dir,
        low_confidence_threshold=low_confidence_threshold,
        good_confidence_threshold=good_confidence_threshold,
        frame_stride=frame_stride,
    )
    if testing_rows:
        print(f"Testing video quality report saved to: {TESTING_QUALITY_REPORT}")

    summary = dict(testing_summary)
    for row in split_rows:
        if row["split"] == "val" and row["class_id"] == "all":
            summary["validation_map50"] = float(row["map50"])
        if row["split"] == "val" and str(row["class_id"]) == "1":
            summary["disconnected_recall"] = float(row["recall"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final model quality honestly.")
    parser.add_argument("--testing-dir", type=Path, default=None)
    parser.add_argument("--low-confidence", type=float, default=0.50)
    parser.add_argument("--good-confidence", type=float, default=0.70)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--no-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_model_quality(
        testing_dir=args.testing_dir,
        low_confidence_threshold=args.low_confidence,
        good_confidence_threshold=args.good_confidence,
        frame_stride=max(1, args.frame_stride),
        save_samples=not args.no_samples,
    )
    if summary:
        print("\nQuality summary")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
