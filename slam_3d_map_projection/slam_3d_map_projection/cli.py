# Copyright 2026 Gole Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line entry point for the offline PCD-to-Nav2 grid projector."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .configuration import load_projection_config
from .outputs import write_projection_outputs
from .pcd_io import read_binary_pcd_records, read_pcd_xyz
from .projection import project_map
from .review import write_review_figure
from .support_projection import (
    project_support_evidence,
    write_support_grid_outputs,
    write_support_review_figure,
)
from .trajectory_io import read_trajectory_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Project an optimized 3D SLAM PCD into a conservative Nav2 trinary "
            "map while retaining ramp and terrain evidence."
        )
    )
    parser.add_argument("--pcd", required=True, type=Path, help="Full localization PCD")
    parser.add_argument(
        "--trajectory", required=True, type=Path, help="Post-optimization trajectory CSV"
    )
    parser.add_argument("--config", required=True, type=Path, help="Projection YAML")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--dataset-label", default=None)
    parser.add_argument(
        "--support-evidence-pcd",
        type=Path,
        default=None,
        help="Optional slam_3d_core support-evidence PCD",
    )
    parser.add_argument("--support-evidence-radius-m", type=float, default=0.10)
    parser.add_argument("--probable-support-min-cost", type=float, default=0.35)
    parser.add_argument("--no-review", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one deterministic offline projection."""
    args = _parser().parse_args(argv)
    for input_path in (args.pcd, args.trajectory, args.config):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
    if (
        args.support_evidence_pcd is not None
        and not args.support_evidence_pcd.is_file()
    ):
        raise FileNotFoundError(args.support_evidence_pcd)
    if not args.map_name or any(character.isspace() for character in args.map_name):
        raise ValueError("map-name must be a non-empty token without whitespace")

    config = load_projection_config(args.config)
    points = read_pcd_xyz(args.pcd)
    trajectory_data = read_trajectory_data(args.trajectory)
    trajectory = trajectory_data.poses
    result = project_map(
        points,
        trajectory,
        config,
        trajectory_stamp_ns=trajectory_data.stamp_ns,
        trajectory_yaw_source=trajectory_data.yaw_source,
    )
    outputs = write_projection_outputs(
        args.output_dir,
        args.map_name,
        points,
        trajectory,
        result,
        config,
    )
    support_metrics = None
    if args.support_evidence_pcd is not None:
        support_records = read_binary_pcd_records(args.support_evidence_pcd)
        support_result = project_support_evidence(
            support_records,
            result.geometry,
            result.occupancy,
            evidence_radius_m=args.support_evidence_radius_m,
            probable_support_min_cost=args.probable_support_min_cost,
        )
        outputs.update(
            write_support_grid_outputs(args.output_dir, args.map_name, support_result)
        )
        if not args.no_review:
            support_review_path = args.output_dir / f"{args.map_name}_support_review.png"
            write_support_review_figure(
                support_review_path,
                args.dataset_label or args.map_name,
                support_result,
            )
            outputs["support_review"] = support_review_path
        support_metrics = support_result.metrics
    if not args.no_review:
        review_path = args.output_dir / f"{args.map_name}_review.png"
        write_review_figure(
            review_path,
            args.dataset_label or args.map_name,
            points,
            trajectory,
            result,
        )
        outputs["review"] = review_path
    print(
        json.dumps(
            {
                "metrics": result.metrics,
                "support_metrics": support_metrics,
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
