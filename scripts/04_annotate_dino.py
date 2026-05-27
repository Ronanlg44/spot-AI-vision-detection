"""Auto-annotate Spot in extracted frames using Grounding DINO.

This is the "teacher" stage of the teacher-student distillation pipeline.
A foundation model (Grounding DINO) generates YOLO-format bounding boxes
from a text prompt, on each frame extracted by 03_extract_frames.py. The
output is then manually filtered by 05_verify_annotations.py before being
used to train YOLOv8m.

Requires a CUDA-capable GPU. Inference on CPU is technically possible but
prohibitively slow (a few hours for ~1000 images).

Dependencies: see scripts/requirements-annotate.txt.

Usage:
    python scripts/04_annotate_dino.py
    python scripts/04_annotate_dino.py --input data/frames/test --output data/annotations_dino/test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Compatibility patch ---
# Grounding DINO (via autodistill-grounding-dino) calls BertModel.get_head_mask.
# This method is no longer exposed on BertModel in recent transformers releases
# (>= 5.0); it lives on ModuleUtilsMixin, from which BertModel still inherits.
# We reattach it explicitly to keep the upstream Grounding DINO code working.
try:
    from transformers import BertModel
    from transformers.modeling_utils import ModuleUtilsMixin

    if not hasattr(BertModel, "get_head_mask"):
        BertModel.get_head_mask = ModuleUtilsMixin.get_head_mask
except ImportError:
    sys.exit("transformers is not installed. See requirements-annotate.txt.")
# --- End patch ---

try:
    from autodistill.detection import CaptionOntology
    from autodistill_grounding_dino import GroundingDINO
except ImportError:
    sys.exit(
        "autodistill-grounding-dino is not installed. "
        "See requirements-annotate.txt."
    )


DEFAULT_SPLITS = ("train", "val", "test")
FRAMES_ROOT = Path("data/frames")
ANNOTATIONS_ROOT = Path("data/annotations_dino")

# Text prompt fed to Grounding DINO, mapped to the YOLO class name.
ONTOLOGY = {"quadruped robot": "Spot"}


def annotate(input_dir: Path, output_dir: Path, model: GroundingDINO) -> None:
    """Run Grounding DINO on every .jpg in input_dir and write YOLO labels."""
    if not input_dir.exists():
        print(f"  Skipped (missing): {input_dir}")
        return

    print(f"Annotating: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model.label(
        input_folder=str(input_dir),
        extension=".jpg",
        output_folder=str(output_dir),
    )

    print(f"  -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Single input directory of .jpg frames. "
             "If omitted, processes data/frames/{train,val,test}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (required if --input is given).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input is not None and args.output is None:
        sys.exit("--output is required when --input is given.")

    print("Loading Grounding DINO...")
    model = GroundingDINO(ontology=CaptionOntology(ONTOLOGY))

    if args.input is not None:
        annotate(args.input, args.output, model)
    else:
        for split in DEFAULT_SPLITS:
            annotate(FRAMES_ROOT / split, ANNOTATIONS_ROOT / split, model)

    print("Done.")


if __name__ == "__main__":
    main()
