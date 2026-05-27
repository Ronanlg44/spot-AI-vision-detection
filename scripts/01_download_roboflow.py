"""Download the 7 Roboflow datasets used during the composite-dataset phase.

The datasets are listed in the project archive (see archive/README.md). They
are downloaded in YOLOv8 format into data/datasets/.

Requires a Roboflow account and API key.
    export ROBOFLOW_API_KEY="your_key_here"

Usage:
    python scripts/01_download_roboflow.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from roboflow import Roboflow
except ImportError:
    sys.exit(
        "Missing dependency: pip install roboflow"
    )


TARGET_DIR = Path("data/datasets")


@dataclass(frozen=True)
class DatasetSpec:
    workspace: str
    project: str
    version: int


DATASETS: list[DatasetSpec] = [
    DatasetSpec("123-c84mm", "robodog-zww0z", 1),
    DatasetSpec("berkay-kilic-dpfzs", "spot-8feoa", 1),
    DatasetSpec("cihan-emre-sahin", "spot-bgwww", 1),
    DatasetSpec("university-dnc5w", "spot-vs-atlas", 1),
    DatasetSpec("mm-5c4fp", "spot_pose", 2),
    DatasetSpec("lime-a7j9v", "underground-rh9op", 7),
    DatasetSpec(
        "akademia-grniczo-hutnicza-im-stanisawa-staszica-w-krakowie",
        "warehouse-0w7f0",
        3,
    ),
]


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY not set. Get one at "
            "https://app.roboflow.com/settings/api"
        )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(TARGET_DIR)

    rf = Roboflow(api_key=api_key)

    for spec in DATASETS:
        print(f"Downloading {spec.workspace}/{spec.project} v{spec.version}...")
        rf.workspace(spec.workspace).project(spec.project).version(
            spec.version
        ).download("yolov8")

    print(f"Done: {len(DATASETS)} datasets in {TARGET_DIR}/")


if __name__ == "__main__":
    main()
