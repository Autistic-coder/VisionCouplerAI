from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from build_classifier_dataset import OUTPUT_DIR, build_classifier_dataset


CLASSIFIER_OUTPUT_PATH = Path("models") / "classifier.pt"
PROJECT_DIR = "runs"
RUN_NAME = "coupler_crop_classifier"


def copy_best_model(results) -> bool:
    save_dir = Path(getattr(results, "save_dir", Path(PROJECT_DIR) / RUN_NAME))
    trained_best_path = save_dir / "weights" / "best.pt"
    requested_best_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    if not trained_best_path.exists() and requested_best_path.exists():
        trained_best_path = requested_best_path
    if not trained_best_path.exists():
        print(f"Classifier training finished, but best.pt was not found at {trained_best_path}")
        return False
    CLASSIFIER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_best_path, CLASSIFIER_OUTPUT_PATH)
    print(f"Best classifier copied to: {CLASSIFIER_OUTPUT_PATH}")
    return True


def train_classifier(
    dataset_dir: Path = OUTPUT_DIR,
    model_name: str = "yolov8n-cls.pt",
    epochs: int = 80,
    imgsz: int = 224,
    batch: int | str = 16,
    patience: int = 15,
    lr0: float = 0.001,
    rebuild_dataset: bool = True,
) -> bool:
    if rebuild_dataset and not build_classifier_dataset(output_dir=dataset_dir):
        print("Classifier crop dataset could not be built.")
        return False

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        print(f"Classifier dataset missing train/val folders under {dataset_dir}.")
        return False

    model = YOLO(model_name)
    results = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        optimizer="AdamW",
        lr0=lr0,
        cos_lr=True,
        workers=2,
        degrees=4,
        translate=0.04,
        scale=0.18,
        shear=0.0,
        perspective=0.0003,
        flipud=0.0,
        fliplr=0.10,
        hsv_h=0.008,
        hsv_s=0.25,
        hsv_v=0.25,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
    )
    return copy_best_model(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the optional coupler crop classifier.")
    parser.add_argument("--dataset", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", default="16")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--skip-dataset-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch: int | str = "auto" if args.batch.lower() == "auto" else int(args.batch)
    train_classifier(
        args.dataset,
        args.model,
        args.epochs,
        args.imgsz,
        batch,
        args.patience,
        args.lr0,
        not args.skip_dataset_build,
    )


if __name__ == "__main__":
    main()
