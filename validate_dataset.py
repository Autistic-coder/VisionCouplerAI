from pathlib import Path


DATASET_DIR = Path("annotated_images")
DATA_YAML_PATH = DATASET_DIR / "data.yaml"
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}


def find_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def find_labels(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".txt")


def validate_label(label_path: Path) -> tuple[bool, list[int], list[str]]:
    class_ids: list[int] = []
    errors: list[str] = []

    if label_path.stat().st_size == 0:
        return False, class_ids, [f"{label_path}: empty label file"]

    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                errors.append(f"{label_path}:{line_number}: expected 5 YOLO values")
                continue

            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number}: non-numeric value")
                continue

            if class_id not in CLASS_NAMES:
                errors.append(f"{label_path}:{line_number}: invalid class ID {class_id}")
            else:
                class_ids.append(class_id)

            if any(value < 0.0 or value > 1.0 for value in coords):
                errors.append(f"{label_path}:{line_number}: coordinates must be between 0 and 1")

    if not class_ids and not errors:
        errors.append(f"{label_path}: empty label file")

    return not errors, class_ids, errors


def validate_split(split: str) -> tuple[bool, dict[int, int], dict[str, int]]:
    images_dir = DATASET_DIR / "images" / split
    labels_dir = DATASET_DIR / "labels" / split
    class_counts = {0: 0, 1: 0}
    issue_counts = {"missing_labels": 0, "orphan_labels": 0, "empty_labels": 0, "invalid_labels": 0}
    split_ok = True

    if not images_dir.exists():
        print(f"Missing folder: {images_dir}")
        split_ok = False
    if not labels_dir.exists():
        print(f"Missing folder: {labels_dir}")
        split_ok = False
    if not split_ok:
        return False, class_counts, issue_counts

    images = find_images(images_dir)
    labels = find_labels(labels_dir)
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    for image_path in images:
        if image_path.stem not in label_stems:
            issue_counts["missing_labels"] += 1
            split_ok = False
            print(f"Missing label for image: {image_path}")

    for label_path in labels:
        if label_path.stem not in image_stems:
            issue_counts["orphan_labels"] += 1
            split_ok = False
            print(f"Orphan label without image: {label_path}")
            continue

        label_ok, class_ids, errors = validate_label(label_path)
        if not label_ok:
            split_ok = False
            if label_path.stat().st_size == 0:
                issue_counts["empty_labels"] += 1
            else:
                issue_counts["invalid_labels"] += 1
            for error in errors:
                print(error)

        for class_id in class_ids:
            class_counts[class_id] += 1

    print(f"\n{split} split")
    print(f"  images: {len(images)}")
    print(f"  labels: {len(labels)}")
    print(f"  coupler_engaged: {class_counts[0]}")
    print(f"  coupler_disengaged: {class_counts[1]}")
    print(f"  missing labels: {issue_counts['missing_labels']}")
    print(f"  orphan labels: {issue_counts['orphan_labels']}")
    print(f"  empty labels: {issue_counts['empty_labels']}")
    print(f"  invalid labels: {issue_counts['invalid_labels']}")

    return split_ok, class_counts, issue_counts


def validate_dataset() -> bool:
    dataset_ok = True
    total_class_counts = {0: 0, 1: 0}
    total_issue_counts = {"missing_labels": 0, "orphan_labels": 0, "empty_labels": 0, "invalid_labels": 0}

    if not DATA_YAML_PATH.exists():
        print("Missing annotated_images/data.yaml")
        dataset_ok = False
    else:
        print("Found annotated_images/data.yaml")

    for split in SPLITS:
        split_ok, class_counts, issue_counts = validate_split(split)
        dataset_ok = dataset_ok and split_ok
        for class_id, count in class_counts.items():
            total_class_counts[class_id] += count
        for issue_name, count in issue_counts.items():
            total_issue_counts[issue_name] += count

    print("\nOverall class distribution")
    print(f"  coupler_engaged: {total_class_counts[0]}")
    print(f"  coupler_disengaged: {total_class_counts[1]}")
    print("Overall issues")
    for issue_name, count in total_issue_counts.items():
        print(f"  {issue_name}: {count}")

    if dataset_ok:
        print("\nDataset validation passed")
    else:
        print("\nDataset validation failed")

    return dataset_ok


def main() -> None:
    validate_dataset()


if __name__ == "__main__":
    main()
