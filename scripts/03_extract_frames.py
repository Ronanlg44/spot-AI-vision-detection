"""Extract frames from videos at 1 FPS for YOLO training.

Videos are read from data/videos/ and frames are written to
data/frames/{split}/. Negative videos (no Spot visible) are capped at
MAX_NEGATIVE_FRAMES to avoid flooding the dataset with background samples.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


VIDEOS_DIR = Path("data/videos")
FRAMES_DIR = Path("data/frames")

# Which split each video belongs to. Edit here, not by moving files.
SPLITS: dict[str, list[str]] = {
    "train": [
        # Positive (Spot visible)
        "day2_passage1.mp4",
        "day2_passage2.mp4",
        "day2_passage3.mp4",
        "day3_passage1.mp4",
        "day3_passage2.mp4",
        "day4_passage1.mp4",
        "day4_passage3.mp4",
        "day4_passage6.mp4",
        "day4_passage7.mp4",
        "spot_warehouse.mp4",
        # Negative (no Spot)
        "day2_passage5_fake.mp4",
    ],
    "val": [
        "day2_passage4.mp4",
        "day4_passage2.mp4",
    ],
    "test": [
        "day4_passage5.mp4",
        "day4_passage8.mp4",
        "spot_launch.mp4",
        "day4_passage4_fake.mp4",
    ],
}

# Files matching any of these substrings are treated as negative samples.
NEGATIVE_MARKERS = ("fake", "empty", "negative", "sans_spot")
MAX_NEGATIVE_FRAMES = 200
FPS = 1


def extract_video(video_path: Path, output_dir: Path) -> None:
    """Extract frames from a single video at FPS into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    pattern = str(output_dir / f"{stem}_frame_%04d.jpg")

    cmd = ["ffmpeg", "-i", str(video_path), "-r", str(FPS), "-q:v", "2"]

    is_negative = any(marker in stem.lower() for marker in NEGATIVE_MARKERS)
    if is_negative:
        cmd += ["-vframes", str(MAX_NEGATIVE_FRAMES)]
        tag = "negative"
    else:
        tag = "positive"

    cmd.append(pattern)

    print(f"  [{tag}] {video_path.name}")
    subprocess.run(
        cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def main() -> None:
    if not VIDEOS_DIR.exists():
        raise SystemExit(
            f"{VIDEOS_DIR} not found. "
            f"Run scripts/02_download_darpa_videos.sh first."
        )

    for split, video_names in SPLITS.items():
        print(f"Split: {split}")
        split_dir = FRAMES_DIR / split
        for name in video_names:
            video_path = VIDEOS_DIR / name
            if not video_path.exists():
                print(f"  Warning: {name} missing, skipped.")
                continue
            extract_video(video_path, split_dir)

    print("Done.")


if __name__ == "__main__":
    main()
