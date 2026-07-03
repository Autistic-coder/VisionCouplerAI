import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


RANDOM_SEED = 42
PRIMARY_MIXED_DIR = Path("annotated_mixed")
FALLBACK_MIXED_DIR = Path("final annotated images")
DATASET_DIR = Path("annotated_images")
UNSORTED_DIR = Path("sorted_unsorted")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_IDS = {0, 1}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


@dataclass(frozen=True)
class ValidPair:
    image_path: Path
    label_path: Path
    first_class_id: int
    label_lines: tuple[str, ...]


def find_source_dir() -> Path | None:
    if PRIMARY_MIXED_DIR.exists():
        return PRIMARY_MIXED_DIR
    if FALLBACK_MIXED_DIR.exists():
        print(f"annotated_mixed/ not found. Using existing folder: {FALLBACK_MIXED_DIR}")
        return FALLBACK_MIXED_DIR
    return None


def ensure_clean_output_dirs() -> None:
    for split in SPLITS:
        for kind in ("images", "labels"):
            folder = DATASET_DIR / kind / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)

    for issue in ("missing_labels", "orphan_labels", "empty_labels", "invalid_labels"):
        folder = UNSORTED_DIR / issue
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    for folder in (
        Path("models"),
        Path("outputs/processed_videos"),
        Path("outputs/processed_images"),
        Path("outputs/disconnected_cases"),
        Path("outputs/unclear_cases"),
        Path("outputs/logs"),
        Path("outputs/evaluation"),
        Path("raw_videos"),
        Path("raw_photos"),
    ):
        folder.mkdir(parents=True, exist_ok=True)


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def safe_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(relative).replace("\\", "_").replace("/", "_"))


def copy_issue_file(path: Path, source_root: Path, issue_name: str) -> None:
    destination = UNSORTED_DIR / issue_name / safe_name(path, source_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def find_obj_names(label_path: Path, source_root: Path) -> Path | None:
    for folder in (label_path.parent, *label_path.parents):
        candidate = folder / "obj.names"
        if candidate.exists():
            return candidate
        if folder == source_root:
            break
    return None


def build_class_map(label_path: Path, source_root: Path) -> dict[int, int]:
    obj_names = find_obj_names(label_path, source_root)
    if obj_names is None:
        return {0: 0, 1: 1}

    mapping: dict[int, int] = {}
    names = [line.strip().lower() for line in obj_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    for source_id, name in enumerate(names):
        if name in {"connected", "coupler_engaged", "engaged"}:
            mapping[source_id] = 0
        elif name in {"disconnected", "coupler_disengaged", "disengaged"}:
            mapping[source_id] = 1
    return mapping or {0: 0, 1: 1}


def validate_label(label_path: Path, source_root: Path) -> tuple[bool, str, int | None, tuple[str, ...]]:
    if label_path.stat().st_size == 0:
        return False, "empty", None, tuple()

    class_map = build_class_map(label_path, source_root)
    first_class_id = None
    remapped_lines: list[str] = []
    with label_path.open("r", encoding="utf-8") as label_file:
        for line_number, line in enumerate(label_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 5:
                return False, f"line {line_number}: expected 5 values", None

            try:
                source_class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                return False, f"line {line_number}: non-numeric YOLO value", None, tuple()

            class_id = class_map.get(source_class_id)
            if class_id is None:
                return False, f"line {line_number}: class ID {source_class_id} not found in obj.names mapping", None, tuple()
            if class_id not in VALID_CLASS_IDS:
                return False, f"line {line_number}: invalid class ID {class_id}", None, tuple()

            if any(value < 0.0 or value > 1.0 for value in values):
                return False, f"line {line_number}: coordinates must be between 0 and 1", None, tuple()

            if first_class_id is None:
                first_class_id = class_id
            remapped_lines.append(f"{class_id} {' '.join(parts[1:])}")

    if first_class_id is None:
        return False, "empty", None, tuple()
    return True, "", first_class_id, tuple(remapped_lines)


def build_label_lookup(label_paths: list[Path]) -> dict[str, list[Path]]:
    labels_by_stem: dict[str, list[Path]] = defaultdict(list)
    for label_path in label_paths:
        labels_by_stem[label_path.stem].append(label_path)
    return labels_by_stem


def find_matching_label(image_path: Path, labels_by_stem: dict[str, list[Path]]) -> Path | None:
    adjacent_label = image_path.with_suffix(".txt")
    if adjacent_label.exists():
        return adjacent_label

    candidates = labels_by_stem.get(image_path.stem, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def split_class_items(items: list[ValidPair]) -> dict[str, list[ValidPair]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    total = len(shuffled)
    train_count = round(total * SPLIT_RATIOS["train"])
    val_count = round(total * SPLIT_RATIOS["val"])

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def copy_valid_pair(pair: ValidPair, split: str, source_root: Path, used_stems: set[str]) -> None:
    base_stem = pair.image_path.stem
    output_stem = base_stem
    if output_stem in used_stems:
        output_stem = Path(safe_name(pair.image_path.with_suffix(""), source_root)).name

    counter = 2
    unique_stem = output_stem
    while unique_stem in used_stems:
        unique_stem = f"{output_stem}_{counter}"
        counter += 1
    used_stems.add(unique_stem)

    image_destination = DATASET_DIR / "images" / split / f"{unique_stem}{pair.image_path.suffix.lower()}"
    label_destination = DATASET_DIR / "labels" / split / f"{unique_stem}.txt"
    shutil.copy2(pair.image_path, image_destination)
    label_destination.write_text("\n".join(pair.label_lines) + "\n", encoding="utf-8")


def write_data_yaml() -> None:
    data_yaml = """path: annotated_images
train: images/train
val: images/val
test: images/test

names:
  0: coupler_engaged
  1: coupler_disengaged
"""
    (DATASET_DIR / "data.yaml").write_text(data_yaml, encoding="utf-8")


def count_classes(items: list[ValidPair]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for item in items:
        counts[item.first_class_id] += 1
    return counts


def organize_dataset() -> bool:
    random.seed(RANDOM_SEED)
    source_root = find_source_dir()
    if source_root is None:
        print("No annotated source folder found.")
        print("Create annotated_mixed/ with matching image and .txt label files, then run again.")
        return False

    ensure_clean_output_dirs()

    image_paths = sorted(path for path in source_root.rglob("*") if is_image(path))
    label_paths = sorted(path for path in source_root.rglob("*.txt") if path.is_file())
    labels_by_stem = build_label_lookup(label_paths)
    image_stems = {image_path.stem for image_path in image_paths}

    valid_pairs: list[ValidPair] = []
    missing_labels = 0
    empty_labels = 0
    invalid_labels = 0
    matched_labels: set[Path] = set()

    for image_path in image_paths:
        label_path = find_matching_label(image_path, labels_by_stem)
        if label_path is None:
            missing_labels += 1
            copy_issue_file(image_path, source_root, "missing_labels")
            continue

        matched_labels.add(label_path)
        is_valid, reason, first_class_id, label_lines = validate_label(label_path, source_root)
        if not is_valid:
            issue_name = "empty_labels" if reason == "empty" else "invalid_labels"
            if issue_name == "empty_labels":
                empty_labels += 1
            else:
                invalid_labels += 1
                print(f"Invalid label: {label_path} ({reason})")
            copy_issue_file(image_path, source_root, issue_name)
            copy_issue_file(label_path, source_root, issue_name)
            continue

        valid_pairs.append(ValidPair(image_path, label_path, int(first_class_id), label_lines))

    orphan_labels = 0
    for label_path in label_paths:
        if label_path in matched_labels:
            continue
        if label_path.stem not in image_stems:
            orphan_labels += 1
            copy_issue_file(label_path, source_root, "orphan_labels")

    by_class: dict[int, list[ValidPair]] = {0: [], 1: []}
    for pair in valid_pairs:
        by_class[pair.first_class_id].append(pair)

    split_items = {split: [] for split in SPLITS}
    for class_items in by_class.values():
        class_splits = split_class_items(class_items)
        for split, items in class_splits.items():
            split_items[split].extend(items)

    used_stems: set[str] = set()
    for split in SPLITS:
        random.shuffle(split_items[split])
        for pair in split_items[split]:
            copy_valid_pair(pair, split, source_root, used_stems)

    write_data_yaml()

    print("\nDataset organization complete")
    print(f"Source folder: {source_root}")
    print(f"Total valid pairs: {len(valid_pairs)}")
    for split in SPLITS:
        counts = count_classes(split_items[split])
        print(f"{split} count: {len(split_items[split])}")
        print(f"  coupler_engaged: {counts[0]}")
        print(f"  coupler_disengaged: {counts[1]}")
    print(f"Missing labels count: {missing_labels}")
    print(f"Orphan labels count: {orphan_labels}")
    print(f"Empty labels count: {empty_labels}")
    print(f"Invalid labels count: {invalid_labels}")
    print(f"data.yaml written to: {DATASET_DIR / 'data.yaml'}")

    return len(valid_pairs) > 0


def main() -> None:
    organize_dataset()


if __name__ == "__main__":
    main()
