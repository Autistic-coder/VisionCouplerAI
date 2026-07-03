from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "dataset_combined_all_sources"
POOL_DIR = OUTPUT_DIR / "_combined_pool"
REPORT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DATASET_PATHS_JSON = PROJECT_ROOT / "dataset_paths.json"
RANDOM_SEED = 42
FRAME_GROUP_CHUNK_SIZE = 180

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_CLASS_IDS = {0, 1}
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}
CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}

PREFERRED_SOURCE_NAMES = [
    "new videos annotated",
    "annotated_images",
    "annotated_mixed",
    "final annotated images",
    "hard_frames_annotated",
    "dataset_prepared",
]

SKIP_SOURCE_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "models",
    "outputs",
    "runs",
    "Ultralytics",
    "raw_videos",
    "new videos",
    "new videos unannotated",
    "dataset_combined_all_sources",
    "sorted_unsorted",
}


@dataclass(frozen=True)
class CandidateSource:
    path: Path
    reason: str


@dataclass
class ValidPair:
    image_path: Path
    label_path: Path
    source_root: Path
    source_reason: str
    width: int
    height: int
    image_sha256: str
    label_sha256: str
    label_lines: list[str]
    class_ids: list[int]
    output_stem: str = ""
    pool_image_path: Path | None = None
    pool_label_path: Path | None = None

    @property
    def has_disconnected(self) -> bool:
        return 1 in self.class_ids

    @property
    def has_engaged(self) -> bool:
        return 0 in self.class_ids

    @property
    def primary_class(self) -> int:
        if self.has_disconnected:
            return 1
        if self.has_engaged:
            return 0
        return -1


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def normalized_path(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def safe_stem(text: str, max_len: int = 150) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return cleaned[:max_len] or "sample"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_paths(path: Path) -> list[Path]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    paths: list[Path] = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            candidate = Path(value)
            if candidate.exists():
                resolved = (candidate if candidate.is_dir() else candidate.parent).resolve()
                if resolved == PROJECT_ROOT.resolve():
                    return
                if resolved == (PROJECT_ROOT / "models").resolve() or (PROJECT_ROOT / "models").resolve() in resolved.parents:
                    return
                if resolved == OUTPUT_DIR.resolve() or OUTPUT_DIR.resolve() in resolved.parents:
                    return
                if resolved == REPORT_DIR.resolve() or REPORT_DIR.resolve() in resolved.parents:
                    return
                paths.append(resolved)

    walk(data)
    return paths


def discover_sources() -> list[CandidateSource]:
    sources: list[CandidateSource] = []

    for name in PREFERRED_SOURCE_NAMES:
        candidate = PROJECT_ROOT / name
        if candidate.exists() and candidate.is_dir():
            sources.append(CandidateSource(candidate, f"preferred:{name}"))

    for candidate in load_json_paths(DATASET_PATHS_JSON):
        if candidate.exists() and candidate.is_dir():
            sources.append(CandidateSource(candidate, "dataset_paths.json"))

    known_external = [
        Path(r"C:\Users\Vaibhav\Desktop\coupler_dataset_storage"),
        Path(r"C:\Users\Vaibhav\Desktop\unannotated_images_backup"),
    ]
    for candidate in known_external:
        if candidate.exists() and candidate.is_dir():
            sources.append(CandidateSource(candidate, f"known_external:{candidate.name}"))

    for child in PROJECT_ROOT.iterdir():
        if not child.is_dir() or child.name in SKIP_SOURCE_NAMES:
            continue
        sources.append(CandidateSource(child, f"project_scan:{child.name}"))

    deduped: dict[str, CandidateSource] = {}
    for source in sources:
        resolved = Path(normalized_path(source.path))
        if resolved == PROJECT_ROOT.resolve():
            continue
        if resolved == OUTPUT_DIR.resolve() or OUTPUT_DIR.resolve() in resolved.parents:
            continue
        if resolved == (PROJECT_ROOT / "models").resolve() or (PROJECT_ROOT / "models").resolve() in resolved.parents:
            continue
        if resolved == REPORT_DIR.resolve() or REPORT_DIR.resolve() in resolved.parents:
            continue
        key = str(resolved)
        deduped.setdefault(key, source)
    return list(deduped.values())


def find_obj_names(label_path: Path, source_root: Path) -> Path | None:
    current = label_path.parent
    while True:
        candidate = current / "obj.names"
        if candidate.exists():
            return candidate
        if current == source_root or current == current.parent:
            break
        current = current.parent
    return None


def build_class_map(label_path: Path, source_root: Path) -> dict[int, int]:
    obj_names = find_obj_names(label_path, source_root)
    if obj_names is None:
        return {0: 0, 1: 1}

    names = [line.strip().lower() for line in obj_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping: dict[int, int] = {}
    for source_id, name in enumerate(names):
        normalized = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        if normalized in {"connected", "coupler_engaged", "engaged"}:
            mapping[source_id] = 0
        elif normalized in {"disconnected", "coupler_disengaged", "disengaged"}:
            mapping[source_id] = 1
    return mapping or {0: 0, 1: 1}


def labels_by_stem(source_root: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = defaultdict(list)
    for label_path in source_root.rglob("*.txt"):
        if label_path.name.lower() in {"train.txt", "test.txt", "valid.txt", "val.txt", "obj.names"}:
            continue
        lookup[label_path.stem].append(label_path)
    return lookup


def image_to_dataset_label(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" not in lowered:
        return None
    index = lowered.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def find_label(image_path: Path, lookup: dict[str, list[Path]]) -> Path | None:
    adjacent = image_path.with_suffix(".txt")
    if adjacent.exists():
        return adjacent

    dataset_label = image_to_dataset_label(image_path)
    if dataset_label is not None and dataset_label.exists():
        return dataset_label

    candidates = lookup.get(image_path.stem, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def validate_label(label_path: Path, source_root: Path) -> tuple[bool, str, list[str], list[int]]:
    class_map = build_class_map(label_path, source_root)
    remapped_lines: list[str] = []
    class_ids: list[int] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            return False, f"line {line_number}: expected 5 YOLO values", [], []
        try:
            source_class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            return False, f"line {line_number}: non-numeric value", [], []

        class_id = class_map.get(source_class_id)
        if class_id not in VALID_CLASS_IDS:
            return False, f"line {line_number}: invalid class id {source_class_id}", [], []
        if any(math.isnan(value) or value < 0.0 or value > 1.0 for value in coords):
            return False, f"line {line_number}: coordinates outside 0..1", [], []
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            return False, f"line {line_number}: non-positive width/height", [], []

        class_ids.append(class_id)
        remapped_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}")

    return True, "", remapped_lines, class_ids


def read_image_size(image_path: Path) -> tuple[int, int] | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    height, width = image.shape[:2]
    return width, height


def collect_valid_pairs(sources: list[CandidateSource]) -> tuple[list[ValidPair], list[dict]]:
    valid_pairs: list[ValidPair] = []
    issues: list[dict] = []
    seen_image_hashes: set[str] = set()
    seen_name_size: set[tuple[str, int, int]] = set()

    for source in sources:
        label_lookup = labels_by_stem(source.path)
        images = sorted(path for path in source.path.rglob("*") if is_image(path))
        for image_path in images:
            label_path = find_label(image_path, label_lookup)
            if label_path is None:
                continue

            size = read_image_size(image_path)
            if size is None:
                issues.append(issue_row(source, image_path, label_path, "unreadable_image"))
                continue
            width, height = size

            ok, reason, label_lines, class_ids = validate_label(label_path, source.path)
            if not ok:
                issues.append(issue_row(source, image_path, label_path, f"invalid_label:{reason}"))
                continue

            image_hash = file_sha256(image_path)
            label_hash = hashlib.sha256(("\n".join(label_lines) + "\n").encode("utf-8")).hexdigest()
            name_size_key = (image_path.name.lower(), width, height)
            if image_hash in seen_image_hashes:
                issues.append(issue_row(source, image_path, label_path, "duplicate_image_hash"))
                continue
            if name_size_key in seen_name_size:
                issues.append(issue_row(source, image_path, label_path, "duplicate_filename_and_size"))
                continue

            seen_image_hashes.add(image_hash)
            seen_name_size.add(name_size_key)
            valid_pairs.append(
                ValidPair(
                    image_path=image_path,
                    label_path=label_path,
                    source_root=source.path,
                    source_reason=source.reason,
                    width=width,
                    height=height,
                    image_sha256=image_hash,
                    label_sha256=label_hash,
                    label_lines=label_lines,
                    class_ids=class_ids,
                )
            )

    return valid_pairs, issues


def issue_row(source: CandidateSource, image_path: Path, label_path: Path | None, reason: str) -> dict:
    return {
        "source_root": str(source.path),
        "source_reason": source.reason,
        "image_path": str(image_path),
        "label_path": str(label_path or ""),
        "reason": reason,
    }


def reset_output_dirs() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "images").mkdir(parents=True, exist_ok=True)
    (POOL_DIR / "labels").mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def assign_output_stems(pairs: list[ValidPair]) -> None:
    used: set[str] = set()
    for index, pair in enumerate(pairs, start=1):
        source_name = safe_stem(pair.source_root.name)
        base = safe_stem(f"{source_name}_{pair.image_path.stem}")
        stem = base
        counter = 2
        while stem in used:
            stem = f"{base}_{counter}"
            counter += 1
        used.add(stem)
        pair.output_stem = stem


def copy_to_pool(pairs: list[ValidPair]) -> None:
    for pair in pairs:
        image_dest = POOL_DIR / "images" / f"{pair.output_stem}{pair.image_path.suffix.lower()}"
        label_dest = POOL_DIR / "labels" / f"{pair.output_stem}.txt"
        shutil.copy2(pair.image_path, image_dest)
        label_dest.write_text("\n".join(pair.label_lines) + ("\n" if pair.label_lines else ""), encoding="utf-8")
        pair.pool_image_path = image_dest
        pair.pool_label_path = label_dest


def frame_group_key(pair: ValidPair) -> str:
    stem = pair.image_path.stem
    match = re.match(r"(.+?)_frame_(\d+)$", stem, flags=re.IGNORECASE)
    if match:
        frame_number = int(match.group(2))
        chunk_id = frame_number // FRAME_GROUP_CHUNK_SIZE
        return f"{normalized_path(pair.source_root)}::{match.group(1).lower()}::chunk_{chunk_id:04d}"
    return f"{normalized_path(pair.source_root)}::{pair.image_path.parent.name.lower()}::{stem.lower()}"


def split_groups(pairs: list[ValidPair]) -> dict[str, list[ValidPair]]:
    groups: dict[str, list[ValidPair]] = defaultdict(list)
    for pair in pairs:
        groups[frame_group_key(pair)].append(pair)

    group_items = list(groups.values())
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(group_items)
    group_items.sort(key=lambda items: (not any(pair.has_disconnected for pair in items), -len(items)))

    total_images = len(pairs)
    total_disconnected = sum(1 for pair in pairs if pair.has_disconnected)
    total_engaged = sum(1 for pair in pairs if pair.has_engaged)
    targets = {
        split: {
            "images": total_images * SPLIT_RATIOS[split],
            "disconnected": total_disconnected * SPLIT_RATIOS[split],
            "engaged": total_engaged * SPLIT_RATIOS[split],
        }
        for split in SPLITS
    }

    split_items: dict[str, list[ValidPair]] = {split: [] for split in SPLITS}
    current = {split: {"images": 0, "disconnected": 0, "engaged": 0} for split in SPLITS}

    def objective(projected_current: dict[str, dict[str, int]]) -> float:
        total = 0.0
        for split in SPLITS:
            image_error = abs(projected_current[split]["images"] - targets[split]["images"]) / max(total_images, 1)
            disconnected_error = abs(
                projected_current[split]["disconnected"] - targets[split]["disconnected"]
            ) / max(total_disconnected, 1)
            engaged_error = abs(projected_current[split]["engaged"] - targets[split]["engaged"]) / max(total_engaged, 1)
            overfill = max(projected_current[split]["images"] - targets[split]["images"], 0.0) / max(total_images, 1)
            total += image_error * 4.0 + disconnected_error * 2.4 + engaged_error * 2.0 + overfill * 1.2
        return total

    for group in group_items:
        metrics = {
            "images": len(group),
            "disconnected": sum(1 for pair in group if pair.has_disconnected),
            "engaged": sum(1 for pair in group if pair.has_engaged),
        }

        def score(split: str) -> float:
            projected_current = {
                name: dict(values)
                for name, values in current.items()
            }
            for key, value in metrics.items():
                projected_current[split][key] += value
            return objective(projected_current)

        best_split = min(SPLITS, key=score)
        split_items[best_split].extend(group)
        for key, value in metrics.items():
            current[best_split][key] += value

    return split_items


def copy_split_items(split_items: dict[str, list[ValidPair]]) -> None:
    for split, pairs in split_items.items():
        for pair in pairs:
            assert pair.pool_image_path is not None
            assert pair.pool_label_path is not None
            image_dest = OUTPUT_DIR / "images" / split / pair.pool_image_path.name
            label_dest = OUTPUT_DIR / "labels" / split / pair.pool_label_path.name
            shutil.copy2(pair.pool_image_path, image_dest)
            shutil.copy2(pair.pool_label_path, label_dest)


def yolo_to_xyxy(line: str, width: int, height: int) -> tuple[int, float, float, float, float]:
    parts = line.split()
    class_id = int(parts[0])
    x_center, y_center, box_width, box_height = [float(value) for value in parts[1:]]
    x1 = (x_center - box_width / 2.0) * width
    y1 = (y_center - box_height / 2.0) * height
    x2 = (x_center + box_width / 2.0) * width
    y2 = (y_center + box_height / 2.0) * height
    return class_id, x1, y1, x2, y2


def xyxy_to_yolo(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str | None:
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return None
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def transform_boxes(label_lines: list[str], matrix: np.ndarray, width: int, height: int, perspective: bool) -> list[str]:
    transformed: list[str] = []
    for line in label_lines:
        class_id, x1, y1, x2, y2 = yolo_to_xyxy(line, width, height)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        if perspective:
            points = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        else:
            ones = np.ones((corners.shape[0], 1), dtype=np.float32)
            points = np.hstack([corners, ones]) @ matrix.T
        new_line = xyxy_to_yolo(
            class_id,
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
            width,
            height,
        )
        if new_line is not None:
            transformed.append(new_line)
    return transformed


def augment_disconnected_train(split_items: dict[str, list[ValidPair]], variants_per_image: int) -> int:
    if variants_per_image <= 0:
        return 0
    rng = random.Random(RANDOM_SEED + 17)
    augmented_count = 0
    train_disconnected = [pair for pair in split_items["train"] if pair.has_disconnected and pair.label_lines]

    for pair in train_disconnected:
        source_image = OUTPUT_DIR / "images" / "train" / f"{pair.output_stem}{pair.image_path.suffix.lower()}"
        image = cv2.imread(str(source_image))
        if image is None:
            continue
        height, width = image.shape[:2]
        for variant in range(variants_per_image):
            augmented = image.copy()

            angle = rng.uniform(-4.0, 4.0)
            scale = rng.uniform(0.98, 1.02)
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
            matrix[0, 2] += rng.uniform(-0.025, 0.025) * width
            matrix[1, 2] += rng.uniform(-0.025, 0.025) * height
            augmented = cv2.warpAffine(augmented, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
            new_lines = transform_boxes(pair.label_lines, matrix, width, height, perspective=False)
            if not new_lines:
                continue

            alpha = rng.uniform(0.88, 1.14)
            beta = rng.uniform(-16, 18)
            augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

            if rng.random() < 0.35:
                shadow = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), width, dtype=np.float32)
                shadow = np.tile(shadow, (height, 1))
                augmented = np.clip(augmented.astype(np.float32) * shadow[:, :, None], 0, 255).astype(np.uint8)

            if rng.random() < 0.35:
                augmented = cv2.GaussianBlur(augmented, (3, 3), sigmaX=rng.uniform(0.2, 0.7))

            if rng.random() < 0.45:
                noise = rng.normalvariate(0, 1)
                noise_image = np.random.default_rng(rng.randint(1, 999999)).normal(noise, 5.5, augmented.shape)
                augmented = np.clip(augmented.astype(np.float32) + noise_image, 0, 255).astype(np.uint8)

            aug_stem = f"{pair.output_stem}_aug_disc_{variant + 1}"
            image_dest = OUTPUT_DIR / "images" / "train" / f"{aug_stem}.jpg"
            label_dest = OUTPUT_DIR / "labels" / "train" / f"{aug_stem}.txt"
            cv2.imwrite(str(image_dest), augmented)
            label_dest.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            augmented_count += 1

    return augmented_count


def disconnected_report_rows(split_items: dict[str, list[ValidPair]]) -> list[dict]:
    rows: list[dict] = []
    for split, pairs in split_items.items():
        for pair in pairs:
            disconnected_areas: list[float] = []
            for line in pair.label_lines:
                parts = line.split()
                if int(parts[0]) == 1:
                    disconnected_areas.append(float(parts[3]) * float(parts[4]))
            if not disconnected_areas:
                continue

            has_two = len(disconnected_areas) >= 2
            has_large = any(area >= 0.08 for area in disconnected_areas)
            has_small = any(area <= 0.035 for area in disconnected_areas)
            warnings: list[str] = []
            if not has_two:
                warnings.append("only_1_disconnected_box")
            if not has_large:
                warnings.append("large_hanging_wire_box_may_be_missing")
            if not has_small:
                warnings.append("small_coupler_mouth_box_may_be_missing")

            rows.append(
                {
                    "split": split,
                    "image_path": str(pair.image_path),
                    "combined_image": str(OUTPUT_DIR / "images" / split / f"{pair.output_stem}{pair.image_path.suffix.lower()}"),
                    "source_dataset_folder": str(pair.source_root),
                    "disconnected_box_count": len(disconnected_areas),
                    "has_2_disconnected_boxes": has_two,
                    "largest_disconnected_area": f"{max(disconnected_areas):.6f}",
                    "smallest_disconnected_area": f"{min(disconnected_areas):.6f}",
                    "warning": ";".join(warnings),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(dataset_dir: Path) -> Path:
    data_yaml = dataset_dir / "data.yaml"
    yaml_text = f"""path: {dataset_dir.as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: {CLASS_NAMES[0]}
  1: {CLASS_NAMES[1]}
"""
    data_yaml.write_text(yaml_text, encoding="utf-8")
    return data_yaml


def count_split(split_items: dict[str, list[ValidPair]], augmented_count: int) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for split, pairs in split_items.items():
        image_count = len(pairs) + (augmented_count if split == "train" else 0)
        class_images = Counter(pair.primary_class for pair in pairs)
        box_counts = Counter()
        for pair in pairs:
            box_counts.update(pair.class_ids)
        summary[split] = {
            "images": image_count,
            "background_images": class_images[-1],
            "engaged_images": class_images[0],
            "disconnected_images": class_images[1],
            "engaged_boxes": box_counts[0],
            "disconnected_boxes": box_counts[1],
        }
    return summary


def write_manifest(
    sources: list[CandidateSource],
    split_items: dict[str, list[ValidPair]],
    issues: list[dict],
    augmented_count: int,
    data_yaml: Path,
) -> None:
    source_rows = []
    for source in sources:
        source_pairs = [pair for pairs in split_items.values() for pair in pairs if normalized_path(pair.source_root) == normalized_path(source.path)]
        source_rows.append(
            {
                "path": str(source.path),
                "resolved_path": normalized_path(source.path),
                "reason": source.reason,
                "valid_pairs_used": len(source_pairs),
            }
        )
    used_source_rows = [row for row in source_rows if row["valid_pairs_used"] > 0]

    manifest = {
        "dataset_dir": str(OUTPUT_DIR),
        "combined_pool": str(POOL_DIR),
        "data_yaml": str(data_yaml),
        "sources": used_source_rows,
        "scanned_sources": source_rows,
        "split_summary": count_split(split_items, augmented_count),
        "augmented_disconnected_train_images": augmented_count,
        "reports": {
            "source_manifest": str(REPORT_DIR / "combined_all_sources_manifest.json"),
            "skipped_pairs": str(REPORT_DIR / "combined_all_sources_skipped_pairs.csv"),
            "disconnected_annotation_report": str(REPORT_DIR / "disconnected_annotation_report_all_sources.csv"),
        },
    }
    (REPORT_DIR / "combined_all_sources_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    path_data = {
        "original_dataset_source_folders_found": used_source_rows,
        "scanned_dataset_source_folders": source_rows,
        "combined_dataset_path": str(OUTPUT_DIR),
        "prepared_train_val_test_data_yaml_path": str(data_yaml),
        "relocated_desktop_storage_path": str(Path(r"C:\Users\Vaibhav\Desktop\coupler_dataset_storage")),
        "final_model_path": str(PROJECT_ROOT / "models" / "best_retrained_all_data_disconnected_focus.pt"),
    }
    DATASET_PATHS_JSON.write_text(json.dumps(path_data, indent=2), encoding="utf-8")

    if issues:
        write_csv(
            REPORT_DIR / "combined_all_sources_skipped_pairs.csv",
            ["source_root", "source_reason", "image_path", "label_path", "reason"],
            issues,
        )


def build_dataset(variants_per_disconnected_train_image: int) -> bool:
    random.seed(RANDOM_SEED)
    sources = discover_sources()
    if not sources:
        print("No candidate annotated dataset folders found.")
        return False

    print("Candidate sources:")
    for source in sources:
        print(f"  - {source.path} ({source.reason})")

    pairs, issues = collect_valid_pairs(sources)
    if not pairs:
        print("No valid image/YOLO-label pairs were found.")
        return False

    reset_output_dirs()
    assign_output_stems(pairs)
    copy_to_pool(pairs)
    split_items = split_groups(pairs)
    copy_split_items(split_items)
    augmented_count = augment_disconnected_train(split_items, variants_per_disconnected_train_image)
    data_yaml = write_data_yaml(OUTPUT_DIR)

    report_rows = disconnected_report_rows(split_items)
    write_csv(
        REPORT_DIR / "disconnected_annotation_report_all_sources.csv",
        [
            "split",
            "image_path",
            "combined_image",
            "source_dataset_folder",
            "disconnected_box_count",
            "has_2_disconnected_boxes",
            "largest_disconnected_area",
            "smallest_disconnected_area",
            "warning",
        ],
        report_rows,
    )

    write_manifest(sources, split_items, issues, augmented_count, data_yaml)

    summary = count_split(split_items, augmented_count)
    print("\nCombined dataset complete")
    print(f"Valid unique pairs used before augmentation: {len(pairs)}")
    print(f"Disconnected train augmentations created: {augmented_count}")
    for split, counts in summary.items():
        print(f"{split}: {counts}")
    print(f"data.yaml: {data_yaml}")
    print(f"Disconnected annotation report: {REPORT_DIR / 'disconnected_annotation_report_all_sources.csv'}")
    print(f"Manifest: {REPORT_DIR / 'combined_all_sources_manifest.json'}")
    print(f"Path references: {DATASET_PATHS_JSON}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one deduplicated YOLO dataset from all available annotated sources.")
    parser.add_argument(
        "--augment-disconnected-train",
        type=int,
        default=1,
        help="Physical augmentation variants to create for each disconnected training image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ok = build_dataset(max(0, args.augment_disconnected_train))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
