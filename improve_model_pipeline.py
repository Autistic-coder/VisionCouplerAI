from __future__ import annotations

from pathlib import Path

from evaluate_model_quality import evaluate_model_quality
from evaluate_testing_videos import evaluate_testing_videos
from fine_tune_disconnected_improved import fine_tune_disconnected
from merge_hard_annotations import merge_hard_annotations
from mine_hard_frames import mine_hard_frames
from organize_annotated_dataset import organize_dataset
from prepare_hard_frames_for_annotation import prepare_hard_frames_for_annotation
from train_classifier import train_classifier
from train_improved_yolo import train_improved_model
from validate_dataset import validate_dataset


HARD_ANNOTATED_DIR = Path("hard_frames_annotated")
FINAL_HOLDOUT_DIR = Path("final_holdout_videos")

QUALITY_GATES = {
    "validation_map50": 0.85,
    "disconnected_recall": 0.85,
    "average_testing_confidence": 0.70,
    "low_confidence_rate_max": 25.0,
    "no_detection_rate_max": 15.0,
}


def hard_annotations_available() -> bool:
    if not HARD_ANNOTATED_DIR.exists():
        return False
    images = [path for path in HARD_ANNOTATED_DIR.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    return any(image.with_suffix(".txt").exists() and image.with_suffix(".txt").stat().st_size > 0 for image in images)


def print_manual_annotation_stop() -> None:
    print("\nHard frames have been extracted.")
    print("Please annotate the photos in hard_frames_for_annotation/images/.")
    print("Then place the annotated images and labels into hard_frames_annotated/.")
    print("Then run this pipeline again.")
    print("The pipeline will not train on unannotated hard frames.")


def evaluate_quality_gate(summary: dict[str, float]) -> bool:
    failures = []
    if summary.get("validation_map50", 0.0) < QUALITY_GATES["validation_map50"]:
        failures.append(f"validation mAP50 below {QUALITY_GATES['validation_map50']}")
    if summary.get("disconnected_recall", 0.0) < QUALITY_GATES["disconnected_recall"]:
        failures.append(f"disconnected recall below {QUALITY_GATES['disconnected_recall']}")
    if summary.get("average_testing_confidence", 0.0) < QUALITY_GATES["average_testing_confidence"]:
        failures.append(f"average testing confidence below {QUALITY_GATES['average_testing_confidence']}")
    if summary.get("low_confidence_rate", 100.0) > QUALITY_GATES["low_confidence_rate_max"]:
        failures.append(f"low-confidence frames above {QUALITY_GATES['low_confidence_rate_max']}%")
    if summary.get("no_detection_rate", 100.0) > QUALITY_GATES["no_detection_rate_max"]:
        failures.append(f"no-detection frames above {QUALITY_GATES['no_detection_rate_max']}%")

    print("\nQuality gate")
    if not failures:
        print("PASS: model meets the default minimum gates.")
        if not FINAL_HOLDOUT_DIR.exists():
            print("Create final_holdout_videos/ with videos never used for mining/training before claiming real-world readiness.")
        return True

    print("FAIL: model is not ready yet.")
    for failure in failures:
        print(f"  - {failure}")
    print("\nLikely next checks:")
    print("  - Add more disconnected annotations, especially technician-disconnection frames.")
    print("  - Add hand-occlusion examples if the technician hand appears during failure cases.")
    print("  - Add lighting and camera-angle variation if those appear in hard_frame_index.csv.")
    print("  - Verify large disconnected latch-region boxes are annotated consistently.")
    return False


def run_pipeline() -> bool:
    print("Step 1: validating current dataset")
    validate_dataset()

    print("\nStep 2: evaluating current model on testing videos")
    evaluate_testing_videos(frame_stride=2)

    print("\nStep 3: mining hard frames from testing videos")
    mine_hard_frames(frame_stride=5)
    prepare_hard_frames_for_annotation()

    if not hard_annotations_available():
        print_manual_annotation_stop()
        return False

    print("\nStep 4: merging reviewed hard annotations")
    if not merge_hard_annotations():
        print("No valid hard annotations were merged. Stopping before training.")
        return False

    print("\nStep 5: reorganizing dataset")
    if not organize_dataset():
        print("Dataset organization failed. Stopping.")
        return False

    print("\nStep 6: validating expanded dataset")
    if not validate_dataset():
        print("Expanded dataset validation failed. Stopping before training.")
        return False

    print("\nStep 7: training improved YOLO model")
    if not train_improved_model():
        print("Improved training failed.")
        return False

    print("\nStep 8: fine-tuning disconnected class carefully")
    if not fine_tune_disconnected():
        print("Disconnected fine-tuning failed.")
        return False

    print("\nStep 9: training optional crop classifier")
    if not train_classifier():
        print("Classifier training failed. The app can still run with YOLO only, but classifier accuracy boost will be unavailable.")

    print("\nStep 10: evaluating final quality")
    summary = evaluate_model_quality(frame_stride=2)
    return evaluate_quality_gate(summary)


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
