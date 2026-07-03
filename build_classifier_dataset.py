from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml


DATA_YAML_PATH = Path("annotated_images") / "data.yaml"
OUTPUT_DIR = Path("annotated_images_classifier")
CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _dataset_root(data: dict[str, Any], data_yaml_path: Path) -> Path:
    raw_path = data.get("path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path
    return data_yaml_path.parent


def _split_dir(root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    return split_path if split_path.is_absolute() else root / split_path


def _label_path_for_image(image_path: Path, images_dir: Path, labels_dir: Path) -> Path:
    relative = image_path.relative_to(images_dir)
    return labels_dir / relative.with_suffix(".txt")


def _labels_dir_for_images_dir(root: Path, images_dir: Path) -> Path:
    try:
        relative = images_dir.relative_to(root)
        parts = list(relative.parts)
        if parts and parts[0] == "images":
            parts[0] = "labels"
            return root.joinpath(*parts)
    except ValueError:
        pass
    return images_dir.parent.parent / "labels" / images_dir.name


def _read_yolo_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            continue
        if class_id in CLASS_NAMES:
            rows.append((class_id, x_center, y_center, width, height))
    return rows


def _crop_from_yolo(image, row: tuple[int, float, float, float, float], padding: float):
    class_id, x_center, y_center, box_width, box_height = row
    height, width = image.shape[:2]
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    if right <= left or bottom <= top:
        return class_id, None
    return class_id, image[top:bottom, left:right]


def build_classifier_dataset(
    data_yaml_path: Path = DATA_YAML_PATH,
    output_dir: Path = OUTPUT_DIR,
    padding: float = 0.12,
    min_size: int = 24,
    clean: bool = True,
) -> bool:
    if not data_yaml_path.exists():
        print(f"Missing dataset yaml: {data_yaml_path}")
        return False

    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8")) or {}
    root = _dataset_root(data, data_yaml_path)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    counts = {split: {class_name: 0 for class_name in CLASS_NAMES.values()} for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not split_value:
            continue
        images_dir = _split_dir(root, str(split_value))
        labels_dir = _labels_dir_for_images_dir(root, images_dir)
        if not images_dir.exists() or not labels_dir.exists():
            print(f"Skipping {split}: missing {images_dir} or {labels_dir}")
            continue

        for image_path in images_dir.rglob("*"):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = _label_path_for_image(image_path, images_dir, labels_dir)
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for box_index, row in enumerate(_read_yolo_rows(label_path)):
                class_id, crop = _crop_from_yolo(image, row, padding)
                if crop is None or crop.shape[0] < min_size or crop.shape[1] < min_size:
                    continue
                class_name = CLASS_NAMES[class_id]
                destination_dir = output_dir / split / class_name
                destination_dir.mkdir(parents=True, exist_ok=True)
                destination = destination_dir / f"{image_path.stem}_box_{box_index:02d}{image_path.suffix.lower()}"
                cv2.imwrite(str(destination), crop)
                counts[split][class_name] += 1

    print(f"Classifier dataset written to: {output_dir}")
    for split, split_counts in counts.items():
        print(f"{split}: " + ", ".join(f"{name}={count}" for name, count in split_counts.items()))
    return any(count for split_counts in counts.values() for count in split_counts.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a YOLO crop dataset for second-stage classification.")
    parser.add_argument("--data", type=Path, default=DATA_YAML_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--min-size", type=int, default=24)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_classifier_dataset(args.data, args.output, args.padding, args.min_size, not args.keep_existing)


if __name__ == "__main__":
    main()
