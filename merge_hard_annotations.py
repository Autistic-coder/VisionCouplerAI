from __future__ import annotations

import shutil
from pathlib import Path

from coupler_improvement_utils import (
    IMAGE_EXTENSIONS,
    copy_issue_pair,
    list_images,
    validate_yolo_label,
)


SOURCE_DIR = Path("hard_frames_annotated")
DESTINATION_DIR = Path("annotated_mixed")
ISSUE_DIR = Path("sorted_unsorted") / "hard_frame_issues"


def unique_destination(path: Path) -> Path:
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)
    candidate = DESTINATION_DIR / path.name
    if not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = DESTINATION_DIR / f"{path.stem}_hard_{counter}{path.suffix.lower()}"
        if not candidate.exists():
            return candidate
        counter += 1


def merge_hard_annotations() -> bool:
    if not SOURCE_DIR.exists():
        print("No hard_frames_annotated/ folder found.")
        print("Annotate hard frames first, then place reviewed images and labels there.")
        return False

    images = list_images(SOURCE_DIR)
    summary = {
        "total_hard_images_found": len(images),
        "valid_pairs_merged": 0,
        "missing_labels": 0,
        "empty_labels": 0,
        "invalid_labels": 0,
    }

    ISSUE_DIR.mkdir(parents=True, exist_ok=True)
    DESTINATION_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            summary["missing_labels"] += 1
            copy_issue_pair(image_path, None, ISSUE_DIR, "missing_labels")
            continue

        valid, reason, _ = validate_yolo_label(label_path)
        if not valid:
            if reason == "empty label":
                summary["empty_labels"] += 1
                issue_name = "empty_labels"
            else:
                summary["invalid_labels"] += 1
                issue_name = "invalid_labels"
                print(f"Invalid hard label: {label_path} ({reason})")
            copy_issue_pair(image_path, label_path, ISSUE_DIR, issue_name)
            continue

        image_destination = unique_destination(image_path)
        label_destination = image_destination.with_suffix(".txt")
        shutil.copy2(image_path, image_destination)
        shutil.copy2(label_path, label_destination)
        summary["valid_pairs_merged"] += 1

    print("\nHard annotation merge summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Valid hard examples copied to: {DESTINATION_DIR}")
    print(f"Problematic files copied to: {ISSUE_DIR}")
    return summary["valid_pairs_merged"] > 0


def main() -> None:
    merge_hard_annotations()


if __name__ == "__main__":
    main()
