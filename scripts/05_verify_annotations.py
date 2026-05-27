"""Manually review and prune Grounding DINO annotations.

Displays each annotated frame with its bounding box overlaid. The reviewer
keeps or discards the annotation with a single keystroke. Progress is
persisted to a memory file so the script can be interrupted and resumed.

Controls:
    SPACE  keep the current image
    D      delete the image and its label file
    Q      save progress and quit

Usage:
    python scripts/05_verify_annotations.py --split train
    python scripts/05_verify_annotations.py --images data/annotations_dino/train --labels data/annotations_dino/train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is not installed: pip install opencv-python")


ANNOTATIONS_ROOT = Path("data/annotations_dino")
BOX_COLOR = (0, 0, 255)  # red, BGR
BOX_THICKNESS = 3
WINDOW_NAME = "Verify annotations"


def draw_boxes(image, label_path: Path) -> None:
    """Overlay YOLO-format bounding boxes from label_path onto image in place."""
    if not label_path.exists():
        return

    height, width = image.shape[:2]
    with label_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, box_w, box_h = (float(p) for p in parts)
            x_min = int((x_center - box_w / 2) * width)
            y_min = int((y_center - box_h / 2) * height)
            x_max = int((x_center + box_w / 2) * width)
            y_max = int((y_center + box_h / 2) * height)
            cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), BOX_COLOR, BOX_THICKNESS
            )


def load_memory(memory_path: Path) -> set[str]:
    if not memory_path.exists():
        return set()
    return set(memory_path.read_text().splitlines())


def review(images_dir: Path, labels_dir: Path, memory_path: Path) -> None:
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        print(f"No images found in {images_dir}")
        return

    reviewed = load_memory(memory_path)
    remaining = [p for p in images if p.name not in reviewed]

    print(f"Total images:    {len(images)}")
    print(f"Already done:    {len(reviewed)}")
    print(f"To review:       {len(remaining)}")
    print()
    print("[SPACE] keep   [D] delete   [Q] quit")
    print()

    for image_path in remaining:
        label_path = labels_dir / image_path.with_suffix(".txt").name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Unreadable: {image_path.name}")
            continue

        draw_boxes(image, label_path)
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            print("Progress saved.")
            break

        if key == ord("d"):
            image_path.unlink()
            if label_path.exists():
                label_path.unlink()
            print(f"  deleted: {image_path.name}")
        else:
            print(f"  kept:    {image_path.name}")

        with memory_path.open("a") as f:
            f.write(image_path.name + "\n")

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        help="Review one of the default splits in data/annotations_dino/.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        help="Custom images directory (used with --labels).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Custom labels directory (used with --images).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.split is not None:
        images_dir = ANNOTATIONS_ROOT / args.split / "images"
        labels_dir = ANNOTATIONS_ROOT / args.split / "labels"
        memory_path = ANNOTATIONS_ROOT / args.split / "reviewed.txt"
    elif args.images is not None and args.labels is not None:
        images_dir = args.images
        labels_dir = args.labels
        memory_path = args.images.parent / "reviewed.txt"
    else:
        sys.exit(
            "Provide either --split or both --images and --labels. "
            "Run with -h for help."
        )

    if not images_dir.exists():
        sys.exit(f"Images directory not found: {images_dir}")

    review(images_dir, labels_dir, memory_path)


if __name__ == "__main__":
    main()
