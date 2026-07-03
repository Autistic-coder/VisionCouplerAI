from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML = PROJECT_ROOT / "dataset_combined_all_sources" / "data.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_MODEL = MODELS_DIR / "best.pt"
OLD_BACKUP_MODEL = MODELS_DIR / "best_old_backup.pt"
FINAL_FOCUSED_MODEL = MODELS_DIR / "best_retrained_all_data_disconnected_focus.pt"
REPORT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
QUALITY_REPORT = REPORT_DIR / "retrained_all_sources_model_quality_report.csv"
DATASET_PATHS_JSON = PROJECT_ROOT / "dataset_paths.json"
RUN_NAME = "retrained_all_sources_disconnected_focus"


def as_float(value, default: float = 0.0) -> float:
    try:
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(np.nanmean(value))
        if isinstance(value, (list, tuple)):
            if not value:
                return default
            return float(np.nanmean(np.array(value, dtype=float)))
        return float(value)
    except (TypeError, ValueError):
        return default


def per_class_value(values, class_index: int) -> float:
    try:
        array = np.array(values, dtype=float)
        if array.ndim == 0:
            return float(array)
        if len(array) > class_index:
            return float(array[class_index])
    except (TypeError, ValueError):
        pass
    return 0.0


def backup_current_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not CURRENT_MODEL.exists():
        raise FileNotFoundError(f"Missing current model: {CURRENT_MODEL}")
    shutil.copy2(CURRENT_MODEL, OLD_BACKUP_MODEL)
    print(f"Old model backed up to: {OLD_BACKUP_MODEL}")


def find_trained_best(results) -> Path:
    save_dir = Path(getattr(results, "save_dir", PROJECT_ROOT / "runs" / RUN_NAME))
    candidates = [
        save_dir / "weights" / "best.pt",
        PROJECT_ROOT / "runs" / RUN_NAME / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Training finished, but no best.pt was found under {save_dir}")


def train_model(epochs: int, imgsz: int, batch: int | str, patience: int, lr0: float) -> Path:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Missing combined dataset yaml: {DATA_YAML}")
    if not CURRENT_MODEL.exists():
        raise FileNotFoundError(f"Missing starting model: {CURRENT_MODEL}")

    print("Starting all-sources disconnected-focus retraining")
    print(f"Dataset: {DATA_YAML}")
    print(f"Start model: {CURRENT_MODEL}")

    model = YOLO(str(CURRENT_MODEL))
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            optimizer="AdamW",
            lr0=lr0,
            cos_lr=True,
            close_mosaic=0,
            workers=0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            project=str(PROJECT_ROOT / "runs"),
            name=RUN_NAME,
            exist_ok=True,
        )
    except RuntimeError as error:
        error_text = str(error).lower()
        if "out of memory" not in error_text and "cuda" not in error_text:
            raise
        print("GPU memory issue detected. Retrying once with batch size 4.")
        model = YOLO(str(CURRENT_MODEL))
        results = model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=4,
            patience=patience,
            optimizer="AdamW",
            lr0=lr0,
            cos_lr=True,
            close_mosaic=0,
            workers=0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            project=str(PROJECT_ROOT / "runs"),
            name=RUN_NAME,
            exist_ok=True,
        )

    trained_best = find_trained_best(results)
    shutil.copy2(trained_best, FINAL_FOCUSED_MODEL)
    shutil.copy2(trained_best, CURRENT_MODEL)
    print(f"Focused model saved to: {FINAL_FOCUSED_MODEL}")
    print(f"App model updated at: {CURRENT_MODEL}")
    return FINAL_FOCUSED_MODEL


def evaluate_model(model_path: Path, imgsz: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    rows: list[dict[str, str | int]] = []
    for split in ("val", "test"):
        print(f"Evaluating {split} split")
        metrics = model.val(data=str(DATA_YAML), split=split, imgsz=imgsz, plots=True, verbose=False)
        box = getattr(metrics, "box", None)
        rows.append(
            {
                "split": split,
                "class_id": "all",
                "class_name": "all",
                "precision": f"{as_float(getattr(box, 'mp', 0.0)):.4f}",
                "recall": f"{as_float(getattr(box, 'mr', 0.0)):.4f}",
                "map50": f"{as_float(getattr(box, 'map50', 0.0)):.4f}",
                "map50_95": f"{as_float(getattr(box, 'map', 0.0)):.4f}",
            }
        )
        for class_id, class_name in {0: "coupler_engaged", 1: "coupler_disengaged"}.items():
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": f"{per_class_value(getattr(box, 'p', []), class_id):.4f}",
                    "recall": f"{per_class_value(getattr(box, 'r', []), class_id):.4f}",
                    "map50": f"{per_class_value(getattr(box, 'ap50', []), class_id):.4f}",
                    "map50_95": f"{per_class_value(getattr(box, 'ap', []), class_id):.4f}",
                }
            )

    with QUALITY_REPORT.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["split", "class_id", "class_name", "precision", "recall", "map50", "map50_95"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Evaluation report saved to: {QUALITY_REPORT}")


def update_dataset_paths() -> None:
    if not DATASET_PATHS_JSON.exists():
        return
    data = json.loads(DATASET_PATHS_JSON.read_text(encoding="utf-8"))
    data["final_model_path"] = str(FINAL_FOCUSED_MODEL)
    data["current_app_model_path"] = str(CURRENT_MODEL)
    data["old_model_backup_path"] = str(OLD_BACKUP_MODEL)
    data["evaluation_report_path"] = str(QUALITY_REPORT)
    DATASET_PATHS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain YOLO on all combined coupler datasets.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="8")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.0007)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch: int | str = "auto" if str(args.batch).lower() == "auto" else int(args.batch)
    backup_current_model()
    model_path = train_model(args.epochs, args.imgsz, batch, args.patience, args.lr0)
    evaluate_model(model_path, args.imgsz)
    update_dataset_paths()


if __name__ == "__main__":
    main()
