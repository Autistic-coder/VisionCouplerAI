from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from coupler_improvement_utils import dataset_class_counts


DATA_YAML_PATH = Path("annotated_images") / "data.yaml"
MODEL_OUTPUT_PATH = Path("models") / "best.pt"
PROJECT_DIR = "runs"
RUN_NAME = "improved_coupler_yolo"


def dataset_ready() -> bool:
    if not DATA_YAML_PATH.exists():
        print("Cannot train: annotated_images/data.yaml was not found.")
        print("Run python organize_annotated_dataset.py first.")
        return False
    return True


def warn_for_class_imbalance() -> None:
    counts = dataset_class_counts()
    total = counts[0] + counts[1]
    print("Class distribution from annotated_images/labels")
    print(f"  coupler_engaged: {counts[0]}")
    print(f"  coupler_disengaged: {counts[1]}")
    if total and counts[1] / total < 0.35:
        print("WARNING: disconnected class is underrepresented.")
        print("Add more reviewed disconnected examples, especially technician-disconnection frames.")


def copy_best_model(results) -> bool:
    save_dir = Path(getattr(results, "save_dir", Path(PROJECT_DIR) / RUN_NAME))
    trained_best_path = save_dir / "weights" / "best.pt"
    requested_best_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not trained_best_path.exists() and requested_best_path.exists():
        trained_best_path = requested_best_path
    if not trained_best_path.exists():
        print(f"Training finished, but best.pt was not found at {trained_best_path}")
        return False
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_best_path, MODEL_OUTPUT_PATH)
    print(f"Best model copied to: {MODEL_OUTPUT_PATH}")
    return True


def train_improved_model(
    fine_tune: bool = False,
    model_name: str = "yolov8s.pt",
    epochs: int = 150,
    imgsz: int = 640,
    batch: int | str = 8,
    patience: int = 30,
    lr0: float = 0.001,
) -> bool:
    if not dataset_ready():
        return False
    warn_for_class_imbalance()

    start_model = MODEL_OUTPUT_PATH if fine_tune and MODEL_OUTPUT_PATH.exists() else Path(model_name)
    print("Starting improved YOLO training")
    print(f"Start model: {start_model}")
    print(f"Dataset: {DATA_YAML_PATH}")

    model = YOLO(str(start_model))
    try:
        results = model.train(
            data=str(DATA_YAML_PATH),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            optimizer="AdamW",
            lr0=lr0,
            cos_lr=True,
            close_mosaic=15,
            workers=2,
            degrees=5,
            translate=0.06,
            scale=0.25,
            shear=0.0,
            perspective=0.0005,
            flipud=0.0,
            fliplr=0.15,
            mosaic=0.6,
            mixup=0.0,
            hsv_h=0.012,
            hsv_s=0.35,
            hsv_v=0.35,
            project=PROJECT_DIR,
            name=RUN_NAME,
            exist_ok=True,
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        if "out of memory" in error_text or "cuda" in error_text:
            print("Training hit a CUDA/GPU memory issue. Retrying once with batch size 4.")
            model = YOLO(str(start_model))
            try:
                results = model.train(
                    data=str(DATA_YAML_PATH),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=4,
                    patience=patience,
                    optimizer="AdamW",
                    lr0=lr0,
                    cos_lr=True,
                    close_mosaic=15,
                    workers=2,
                    degrees=5,
                    translate=0.06,
                    scale=0.25,
                    shear=0.0,
                    perspective=0.0005,
                    flipud=0.0,
                    fliplr=0.15,
                    mosaic=0.6,
                    mixup=0.0,
                    hsv_h=0.012,
                    hsv_s=0.35,
                    hsv_v=0.35,
                    project=PROJECT_DIR,
                    name=RUN_NAME,
                    exist_ok=True,
                )
            except RuntimeError:
                print("Retry also failed. Use yolov8n.pt or lower image size/batch size.")
                raise
        else:
            raise

    return copy_best_model(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train improved YOLO model for CouplerGuard AI.")
    parser.add_argument("--fine-tune", action="store_true", help="Start from models/best.pt instead of base YOLO.")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="8")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch: int | str = "auto" if args.batch.lower() == "auto" else int(args.batch)
    train_improved_model(args.fine_tune, args.model, args.epochs, args.imgsz, batch, args.patience, args.lr0)


if __name__ == "__main__":
    main()
