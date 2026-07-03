from __future__ import annotations

import importlib
from pathlib import Path

import cv2

from runtime_config import RuntimeSettings, ensure_runtime_dirs


REQUIRED_FOLDERS = [
    Path("models"),
    Path("raw_videos"),
    Path("raw_photos"),
    Path("outputs"),
    Path("outputs/processed_videos"),
    Path("outputs/evaluation"),
]

REQUIRED_IMPORTS = [
    "PySide6",
    "cv2",
    "numpy",
    "pandas",
    "yaml",
    "torch",
    "torchvision",
    "ultralytics",
]


def result(name: str, ok: bool, detail: str = "") -> bool:
    state = "PASS" if ok else "FAIL"
    print(f"[{state}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def check_imports() -> bool:
    ok = True
    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
            result(f"Import {module}", True)
        except Exception as exc:
            ok = False
            result(f"Import {module}", False, str(exc))
    return ok


def check_folders() -> bool:
    ensure_runtime_dirs()
    ok = True
    for folder in REQUIRED_FOLDERS:
        exists = folder.exists() and folder.is_dir()
        ok = result(f"Folder {folder}", exists) and ok
    return ok


def check_writable() -> bool:
    ok = True
    for folder in [Path("outputs"), Path("outputs/evaluation")]:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            test_file = folder / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            result(f"Writable {folder}", True)
        except Exception as exc:
            ok = False
            result(f"Writable {folder}", False, str(exc))
    return ok


def check_cuda() -> bool:
    try:
        import torch

        available = torch.cuda.is_available()
        detail = torch.cuda.get_device_name(0) if available else "CPU fallback will be used"
        return result("CUDA availability", True, detail)
    except Exception as exc:
        return result("CUDA availability", False, str(exc))


def check_model() -> bool:
    settings = RuntimeSettings()
    if not settings.model_path.exists():
        return result("Model models/best.pt", False, "Place trained model at models/best.pt")
    try:
        from ultralytics import YOLO

        YOLO(str(settings.model_path))
        return result("YOLO model load", True, str(settings.model_path))
    except Exception as exc:
        return result("YOLO model load", False, str(exc))


def check_classifier_model() -> bool:
    settings = RuntimeSettings()
    if not settings.classifier_model_path.exists():
        return result("Optional classifier models/classifier.pt", True, "Not trained yet; app will use YOLO only")
    try:
        from ultralytics import YOLO

        YOLO(str(settings.classifier_model_path))
        return result("Classifier model load", True, str(settings.classifier_model_path))
    except Exception as exc:
        return result("Classifier model load", False, str(exc))


def check_webcam() -> bool:
    capture = cv2.VideoCapture(0)
    try:
        opened = capture.isOpened()
        return result("Webcam open", opened, "index 0" if opened else "webcam unavailable or busy")
    finally:
        capture.release()


def main() -> int:
    print("CouplerGuard AI offline health check")
    checks = [
        check_folders(),
        check_imports(),
        check_cuda(),
        check_model(),
        check_classifier_model(),
        check_webcam(),
        check_writable(),
    ]
    if all(checks):
        print("Overall: PASS")
        return 0
    print("Overall: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
