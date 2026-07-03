from __future__ import annotations

import argparse
from pathlib import Path

from coupler_improvement_utils import (
    DEFAULT_TESTING_DIR,
    list_videos,
    load_required_model,
    resolve_testing_dir,
    run_video_evaluation,
    write_csv,
)


OUTPUT_ROOT = Path("outputs") / "testing_evaluation"
REPORT_PATH = OUTPUT_ROOT / "testing_video_report.csv"


def evaluate_testing_videos(
    testing_dir: Path | None = None,
    low_confidence_threshold: float = 0.50,
    good_confidence_threshold: float = 0.70,
    frame_stride: int = 1,
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
    rows = []
    for video_path in videos:
        print(f"Evaluating {video_path.name}")
        stats = run_video_evaluation(
            model=model,
            video_path=video_path,
            annotated_video_path=OUTPUT_ROOT / "videos" / f"{video_path.stem}_evaluated.mp4",
            low_confidence_dir=OUTPUT_ROOT / "low_confidence_frames",
            no_detection_dir=OUTPUT_ROOT / "no_detection_frames",
            disconnected_failure_dir=OUTPUT_ROOT / "disconnected_failure_frames",
            low_confidence_threshold=low_confidence_threshold,
            good_confidence_threshold=good_confidence_threshold,
            frame_stride=frame_stride,
        )
        rows.append(stats.as_row())

    fieldnames = [
        "video",
        "total_frames",
        "processed_frames",
        "frames_with_no_detection",
        "frames_classified_as_engaged",
        "frames_classified_as_disengaged",
        "frames_classified_as_unclear",
        "average_confidence",
        "average_engaged_confidence",
        "average_disengaged_confidence",
        "minimum_confidence",
        "low_confidence_frame_percentage",
        "approximate_fps",
    ]
    write_csv(REPORT_PATH, fieldnames, rows)
    print(f"Testing video report saved to: {REPORT_PATH}")
    print(f"Annotated videos saved to: {OUTPUT_ROOT / 'videos'}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models/best.pt on unseen testing videos.")
    parser.add_argument("--testing-dir", type=Path, default=None, help="Folder containing unseen testing videos.")
    parser.add_argument("--low-confidence", type=float, default=0.50, help="Threshold for low-confidence frames.")
    parser.add_argument("--good-confidence", type=float, default=0.70, help="Target confidence for strong predictions.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_testing_videos(args.testing_dir, args.low_confidence, args.good_confidence, max(1, args.frame_stride))


if __name__ == "__main__":
    main()
