from pathlib import Path

import cv2


RAW_VIDEOS_DIR = Path("raw_videos")
EXTRACTED_FRAMES_DIR = Path("extracted_frames")
SAMPLE_EVERY_N_FRAMES = 5
ENABLE_BLUR_FILTER = True
ENABLE_DUPLICATE_FILTER = True
BLUR_THRESHOLD = 80.0
DUPLICATE_THRESHOLD = 5.0

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def get_video_files(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        return []
    return sorted(
        file_path
        for file_path in video_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )


def calculate_blur_score(frame) -> float:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def calculate_frame_difference(frame, previous_frame) -> float:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous_gray_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    if gray_frame.shape != previous_gray_frame.shape:
        previous_gray_frame = cv2.resize(
            previous_gray_frame,
            (gray_frame.shape[1], gray_frame.shape[0]),
        )

    difference = cv2.absdiff(gray_frame, previous_gray_frame)
    return float(difference.mean())


def extract_frames_from_video(video_path: Path, output_dir: Path, sample_every: int) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Could not open video: {video_path}")
        return {
            "total_frames_read": 0,
            "frames_checked": 0,
            "frames_saved": 0,
            "skipped_blurry": 0,
            "skipped_duplicate": 0,
        }

    frame_index = 0
    total_frames_read = 0
    frames_checked = 0
    saved_count = 0
    skipped_blurry = 0
    skipped_duplicate = 0
    previous_saved_frame = None
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        success, frame = capture.read()
        if not success:
            break

        total_frames_read += 1

        if frame_index % sample_every == 0:
            frames_checked += 1

            blur_score = calculate_blur_score(frame)
            if ENABLE_BLUR_FILTER and blur_score < BLUR_THRESHOLD:
                skipped_blurry += 1
                frame_index += 1
                continue

            if ENABLE_DUPLICATE_FILTER and previous_saved_frame is not None:
                frame_difference = calculate_frame_difference(frame, previous_saved_frame)
                if frame_difference < DUPLICATE_THRESHOLD:
                    skipped_duplicate += 1
                    frame_index += 1
                    continue

            frame_name = f"{video_path.stem}_frame_{frame_index:06d}.jpg"
            frame_path = video_output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            previous_saved_frame = frame.copy()
            saved_count += 1

        frame_index += 1

    capture.release()
    return {
        "total_frames_read": total_frames_read,
        "frames_checked": frames_checked,
        "frames_saved": saved_count,
        "skipped_blurry": skipped_blurry,
        "skipped_duplicate": skipped_duplicate,
    }


def main() -> None:
    if SAMPLE_EVERY_N_FRAMES < 1:
        print("SAMPLE_EVERY_N_FRAMES must be 1 or greater.")
        return

    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    video_files = get_video_files(RAW_VIDEOS_DIR)
    if not video_files:
        print("No videos found in raw_videos/. Add videos there and run this script again.")
        return

    total_frames_saved = 0

    for video_path in video_files:
        stats = extract_frames_from_video(
            video_path,
            EXTRACTED_FRAMES_DIR,
            SAMPLE_EVERY_N_FRAMES,
        )
        total_frames_saved += stats["frames_saved"]

        print(f"\n{video_path.name}")
        print(f"  total frames read: {stats['total_frames_read']}")
        print(f"  frames checked: {stats['frames_checked']}")
        print(f"  frames saved: {stats['frames_saved']}")
        print(f"  frames skipped because blurry: {stats['skipped_blurry']}")
        print(f"  frames skipped because duplicate: {stats['skipped_duplicate']}")

    print("\nFinal summary")
    print(f"  total videos processed: {len(video_files)}")
    print(f"  total frames saved: {total_frames_saved}")


if __name__ == "__main__":
    main()
