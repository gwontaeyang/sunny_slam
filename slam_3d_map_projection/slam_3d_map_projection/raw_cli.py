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

"""Command line entry point for paired raw-scan 3D/grid/path products."""

import argparse
import json
from pathlib import Path

from .configuration import load_projection_config
from .path_planning import (
    load_path_planning_config,
    plan_farthest_trajectory_route,
)
from .pcd_io import read_pcd_xyz
from .projection import project_map
from .raw_outputs import write_raw_visibility_outputs
from .rosbag_visibility import accumulate_bag_visibility
from .trajectory_io import read_trajectory_data
from .visibility import (
    apply_visibility_refinement,
    build_obstacle_candidates,
    load_raw_visibility_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine merged-PCD obstacle candidates with raw per-scan hit/miss "
            "evidence, then write a paired 3D map and footprint-aware path grid."
        )
    )
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--pcd", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--projection-config", required=True, type=Path)
    parser.add_argument("--raw-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--dataset-label", default="indoor raw-scan map")
    parser.add_argument(
        "--max-scans",
        type=int,
        help="Bounded development replay; omit for the full bag.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser


def main() -> None:
    """Run the complete visibility, grid, path, and artifact pipeline."""
    args = _parser().parse_args()
    projection_config = load_projection_config(args.projection_config)
    visibility_config = load_raw_visibility_config(args.raw_config)
    planning_config = load_path_planning_config(args.raw_config)
    if (
        visibility_config.height_band_edges_m[0]
        != projection_config.obstacle_min_height_m
        or visibility_config.height_band_edges_m[-1]
        != projection_config.obstacle_max_height_m
    ):
        raise ValueError(
            "raw visibility height bands must span the projection obstacle range"
        )
    if (
        planning_config.robot_footprint_length_m
        != projection_config.robot_footprint_length_m
        or planning_config.robot_footprint_width_m
        != projection_config.robot_footprint_width_m
    ):
        raise ValueError(
            "planning and projection robot footprints must be identical"
        )

    points = read_pcd_xyz(args.pcd)
    trajectory = read_trajectory_data(args.trajectory)
    projection = project_map(
        points,
        trajectory.poses,
        projection_config,
        trajectory_stamp_ns=trajectory.stamp_ns,
        trajectory_yaw_source=trajectory.yaw_source,
    )
    candidates = build_obstacle_candidates(
        points,
        projection,
        visibility_config,
    )
    evidence = accumulate_bag_visibility(
        args.bag,
        candidates,
        trajectory,
        visibility_config,
        max_scans=args.max_scans,
        progress_interval=args.progress_interval,
    )
    refinement = apply_visibility_refinement(
        projection,
        candidates,
        evidence,
        projection_config,
    )
    plan = plan_farthest_trajectory_route(
        refinement.occupancy,
        projection.geometry,
        trajectory.poses,
        projection.trajectory_yaw_rad,
        planning_config,
    )
    outputs = write_raw_visibility_outputs(
        args.output_dir,
        args.map_name,
        args.dataset_label,
        points,
        trajectory.poses,
        projection,
        candidates,
        evidence,
        refinement,
        plan,
        projection_config,
        visibility_config,
        planning_config,
    )
    print(
        json.dumps(
            {
                "raw_visibility": evidence.scan_metrics,
                "visibility_refinement": refinement.metrics,
                "path_planning": plan.metrics,
                "outputs": {name: str(path) for name, path in outputs.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
