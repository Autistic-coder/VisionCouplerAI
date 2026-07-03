from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from coupler_improvement_utils import (
    DEFAULT_TESTING_DIR,
    STATUS_NO_DETECTION,
    STATUS_POSSIBLE_DISCONNECTED,
    STATUS_UNCLEAR,
    blur_score,
    brightness_score,
    choose_best_detection,
    edge_density,
    frame_hash,
    list_videos,
    load_required_model,
    resolve_testing_dir,
    safe_stem,
    save_frame,
    skin_ratio,
)


OUTPUT_DIR = Path("hard_frames_for_annotation") / "images"


def mine_hard_frames(
    testing_dir: Path | None = None,
    low_confidence_threshold: float = 0.50,
    good_confidence_threshold: float = 0.70,
    frame_stride: int = 5,
    max_frames_per_video: int = 120,
) -> list[dict]:
    source_dir = testing_dir or resolve_testing_dir(DEFAULT_TESTING_DIR)
    if source_dir is None:
        print("No testing videos folder found. Create testing/ and place unseen videos there.")
        return []

    videos = list_videos(source_dir)
    if not videos:
        print(f"No videos found in {source_dir}")
        return []

    model = load_required_model()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_count = 0
    seen_hashes: set[str] = set()
    results: list[dict] = []

    for video_path in videos:
        print(f"Mining hard frames from {video_path.name}")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Could not open video: {video_path}")
            continue

        saved_for_video = 0
        frame_number = 0
        brightness_baseline: list[float] = []

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_stride > 1 and frame_number % frame_stride != 0:
                continue

            result = model(frame, verbose=False)[0]
            decision = choose_best_detection(result, frame_shape=frame.shape)
            confidence = float(decision.get("confidence", -1.0))
            status = decision.get("status", "")
            current_prediction = decision.get("detected_class", "") or status

            blur = blur_score(frame)
            brightness = brightness_score(frame)
            edges = edge_density(frame)
            hand_ratio = skin_ratio(frame)
            brightness_baseline.append(brightness)
            baseline = sum(brightness_baseline) / len(brightness_baseline)

            reasons = []
            if status == STATUS_NO_DETECTION:
                reasons.append("no_detection")
            if confidence < low_confidence_threshold:
                reasons.append("low_confidence")
            if status in {STATUS_UNCLEAR, STATUS_POSSIBLE_DISCONNECTED}:
                reasons.append("possible_disconnected_failure")
            if decision.get("class_id") == 1 and confidence < good_confidence_threshold:
                reasons.append("weak_disconnected_detection")
            if float(decision.get("largest_disconnected_box_area_ratio", 0.0)) >= 0.04 and confidence < good_confidence_threshold:
                reasons.append("large_disconnected_region_not_confident")
            if blur < 80.0:
                reasons.append("motion_blur")
            if abs(brightness - baseline) > 35.0 or brightness < 65.0 or brightness > 210.0:
                reasons.append("different_lighting")
            if edges < 0.035 or edges > 0.16:
                reasons.append("different_camera_angle_or_scale")
            if hand_ratio > 0.04:
                reasons.append("technician_hand_present")
            if hand_ratio > 0.08 and confidence < good_confidence_threshold:
                reasons.append("partial_occlusion")

            if not reasons:
                continue

            digest = frame_hash(frame)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            output_path = save_frame(OUTPUT_DIR, video_path, frame_number, frame, "hard")
            normalized_name = f"{safe_stem(video_path)}_frame_{frame_number:06d}{output_path.suffix.lower()}"
            normalized_path = OUTPUT_DIR / normalized_name
            if output_path != normalized_path:
                output_path.rename(normalized_path)
                output_path = normalized_path
            results.append({
                "video": video_path.name,
                "frame_number": frame_number,
                "path": str(output_path),
                "reasons": reasons,
            })

            selected_count += 1
            saved_for_video += 1
            if saved_for_video >= max_frames_per_video:
                break

        capture.release()

    print(f"Mined {selected_count} hard frames into: {OUTPUT_DIR}")
    print("Manual review is required before any of these frames are used for training.")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard frames from testing videos for manual annotation.")
    parser.add_argument("--testing-dir", type=Path, default=None)
    parser.add_argument("--low-confidence", type=float, default=0.50)
    parser.add_argument("--good-confidence", type=float, default=0.70)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mine_hard_frames(
        testing_dir=args.testing_dir,
        low_confidence_threshold=args.low_confidence,
        good_confidence_threshold=args.good_confidence,
        frame_stride=max(1, args.frame_stride),
        max_frames_per_video=max(1, args.max_frames_per_video),
    )


if __name__ == "__main__":
    main()
