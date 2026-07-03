from __future__ import annotations

import shutil
from pathlib import Path

from coupler_improvement_utils import IMAGE_EXTENSIONS, list_images


SOURCE_DIR = Path("hard_frames_for_annotation")
IMAGES_DIR = SOURCE_DIR / "images"


def prepare_hard_frames_for_annotation() -> bool:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    images = [
        path
        for path in SOURCE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    copied = 0
    for image_path in images:
        destination = IMAGES_DIR / image_path.name
        if destination.exists():
            continue
        shutil.copy2(image_path, destination)
        copied += 1

    total_images = len(list_images(IMAGES_DIR))
    print(f"Prepared annotation image folder: {IMAGES_DIR}")
    print(f"Copied new images: {copied}")
    print(f"Total images ready for annotation: {total_images}")
    return total_images > 0


def main() -> None:
    prepare_hard_frames_for_annotation()


if __name__ == "__main__":
    main()
