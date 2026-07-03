from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore
except ModuleNotFoundError:
    print("ultralytics not found. Install with: python -m pip install ultralytics")
    YOLO = None  # type: ignore

from coupler_improvement_utils import dataset_class_counts


DATA_YAML_PATH = Path("annotated_images") / "data.yaml"
MODEL_OUTPUT_PATH = Path("models") / "best.pt"
PROJECT_DIR = "runs"
RUN_NAME = "improved_disconnected_finetune"


def fine_tune_disconnected(
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.0005,
    patience: int = 15,
) -> bool:
    if not MODEL_OUTPUT_PATH.exists():
        print("Cannot fine-tune: models/best.pt does not exist.")
        return False
    if not DATA_YAML_PATH.exists():
        print("Cannot fine-tune: annotated_images/data.yaml was not found.")
        return False

    counts = dataset_class_counts()
    print("Fine-tuning for disconnected robustness")
    print(f"  coupler_engaged labels: {counts[0]}")
    print(f"  coupler_disengaged labels: {counts[1]}")
    if counts[1] < counts[0] * 0.50:
        print("WARNING: disconnected examples are still much fewer than engaged examples.")

    model = YOLO(str(MODEL_OUTPUT_PATH))
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        optimizer="AdamW",
        lr0=lr0,
        cos_lr=True,
        close_mosaic=5,
        workers=2,
        degrees=3,
        translate=0.035,
        scale=0.16,
        shear=0.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.10,
        mosaic=0.25,
        mixup=0.0,
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.22,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(PROJECT_DIR) / RUN_NAME))
    trained_best_path = save_dir / "weights" / "best.pt"
    requested_best_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not trained_best_path.exists() and requested_best_path.exists():
        trained_best_path = requested_best_path
    if not trained_best_path.exists():
        print(f"Fine-tuning finished, but best.pt was not found at {trained_best_path}")
        return False

    shutil.copy2(trained_best_path, MODEL_OUTPUT_PATH)
    print(f"Fine-tuned best model copied to: {MODEL_OUTPUT_PATH}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Carefully fine-tune class 1 disconnected detection.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.0005)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fine_tune_disconnected(args.epochs, args.imgsz, args.batch, args.lr0, args.patience)


if __name__ == "__main__":
    main()
