import csv
from pathlib import Path


LABEL_ROOT = Path("annotated_images") / "labels"
OUTPUT_PATH = Path("outputs") / "evaluation" / "dataset_box_report.csv"
SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "coupler_engaged", 1: "coupler_disengaged"}


def read_boxes(label_path: Path) -> list[dict]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        boxes.append(
            {
                "label": str(label_path),
                "line": line_number,
                "class_id": class_id,
                "area": width * height,
            }
        )
    return boxes


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    class_box_counts = {0: 0, 1: 0}
    class_image_sets = {0: set(), 1: set()}
    disconnected_areas = []
    huge_boxes = []

    for split in SPLITS:
        split_dir = LABEL_ROOT / split
        for label_path in sorted(split_dir.glob("*.txt")):
            boxes = read_boxes(label_path)
            image_classes = set()
            for box in boxes:
                class_id = box["class_id"]
                area = box["area"]
                if class_id in class_box_counts:
                    class_box_counts[class_id] += 1
                    image_classes.add(class_id)
                if class_id == 1:
                    disconnected_areas.append(area)
                if area > 0.90:
                    huge_boxes.append(box)
                rows.append(
                    {
                        "split": split,
                        "label_file": str(label_path),
                        "line": box["line"],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES.get(class_id, "unknown"),
                        "box_area": f"{area:.6f}",
                        "large_disconnected": class_id == 1 and area >= 0.08,
                        "very_small_disconnected": class_id == 1 and area < 0.03,
                        "covers_more_than_90_percent": area > 0.90,
                    }
                )
            for class_id in image_classes:
                class_image_sets[class_id].add(label_path)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "split",
            "label_file",
            "line",
            "class_id",
            "class_name",
            "box_area",
            "large_disconnected",
            "very_small_disconnected",
            "covers_more_than_90_percent",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_boxes = class_box_counts[0] + class_box_counts[1]
    class_1_ratio = class_box_counts[1] / total_boxes if total_boxes else 0.0
    large_disconnected = [area for area in disconnected_areas if area >= 0.08]
    tiny_disconnected = [area for area in disconnected_areas if area < 0.03]

    print("Dataset box analysis")
    print(f"Class 0 boxes: {class_box_counts[0]}")
    print(f"Class 1 boxes: {class_box_counts[1]}")
    print(f"Images containing class 0: {len(class_image_sets[0])}")
    print(f"Images containing class 1: {len(class_image_sets[1])}")
    if disconnected_areas:
        print(f"Class 1 average area: {sum(disconnected_areas) / len(disconnected_areas):.4f}")
        print(f"Class 1 min area: {min(disconnected_areas):.4f}")
        print(f"Class 1 max area: {max(disconnected_areas):.4f}")
        print(f"Class 1 boxes with area >= 0.08: {len(large_disconnected)}")
        print(f"Class 1 boxes with area < 0.03: {len(tiny_disconnected)}")

    if disconnected_areas and len(large_disconnected) < len(disconnected_areas) * 0.25:
        print("WARNING: disconnected boxes are mostly small; add or verify large disconnected-region labels.")
    if len(class_image_sets[1]) < len(class_image_sets[0]):
        print("WARNING: disconnected has fewer images than engaged.")
    if class_1_ratio < 0.35:
        print("WARNING: class 1 is less than 35% of total labeled boxes.")
    if huge_boxes:
        print(f"WARNING: {len(huge_boxes)} boxes cover more than 90% of the image.")
    print(f"CSV report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
