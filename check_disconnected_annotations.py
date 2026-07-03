import csv
from pathlib import Path


LABEL_ROOT = Path("annotated_images") / "labels"
OUTPUT_PATH = Path("outputs") / "evaluation" / "disconnected_annotation_check.csv"
SPLITS = ("train", "val", "test")


def class_1_areas(label_path: Path) -> list[float]:
    areas = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            continue
        if class_id == 1:
            areas.append(width * height)
    return areas


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_disconnected_images = 0
    at_least_two_boxes = 0
    only_one_box = 0
    with_large_box = 0
    missing_large_box = 0

    for split in SPLITS:
        for label_path in sorted((LABEL_ROOT / split).glob("*.txt")):
            areas = class_1_areas(label_path)
            if not areas:
                continue
            total_disconnected_images += 1
            has_two = len(areas) >= 2
            has_large = any(area >= 0.08 for area in areas)
            at_least_two_boxes += int(has_two)
            only_one_box += int(not has_two)
            with_large_box += int(has_large)
            missing_large_box += int(not has_large)
            rows.append(
                {
                    "split": split,
                    "label_file": str(label_path),
                    "class_1_box_count": len(areas),
                    "has_at_least_2_class_1_boxes": has_two,
                    "has_large_class_1_box_area_ge_0_08": has_large,
                    "largest_class_1_area": f"{max(areas):.6f}",
                }
            )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "split",
            "label_file",
            "class_1_box_count",
            "has_at_least_2_class_1_boxes",
            "has_large_class_1_box_area_ge_0_08",
            "largest_class_1_area",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Disconnected annotation check")
    print(f"Total disconnected images: {total_disconnected_images}")
    print(f"Disconnected images with at least 2 boxes: {at_least_two_boxes}")
    print(f"Disconnected images with only 1 box: {only_one_box}")
    print(f"Disconnected images with a large disconnected box: {with_large_box}")
    print(f"Disconnected images missing large disconnected box: {missing_large_box}")
    print(f"CSV report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
