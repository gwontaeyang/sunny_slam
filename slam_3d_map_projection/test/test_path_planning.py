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

"""Footprint-aware path generation tests."""

from pathlib import Path

import numpy as np

from slam_3d_map_projection.path_planning import (
    PathPlanningConfig,
    build_footprint_validity,
    build_turn_sweep_validity,
    densify_pose_path_for_footprint,
    exact_footprint_pose_validity,
    load_path_planning_config,
    plan_farthest_trajectory_route,
)
from slam_3d_map_projection.projection import GridClass, GridGeometry


def _geometry():
    return GridGeometry(
        origin_x_m=0.0,
        origin_y_m=0.0,
        resolution_m=0.1,
        width=100,
        height=60,
    )


def _config(**overrides):
    values = {
        "robot_footprint_length_m": 0.5,
        "robot_footprint_width_m": 0.4,
        "heading_bins": 8,
        "translation_step_m": 0.1,
        "clearance_cost_weight": 0.0,
        "goal_tolerance_m": 0.11,
        "max_expansions": 100_000,
    }
    values.update(overrides)
    return PathPlanningConfig(**values)


def test_bundled_projection_profile_has_bounded_large_map_planner():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "indoor_grid_projection.yaml"
    )
    config = load_path_planning_config(config_path)

    assert config.heading_bins == 8
    assert config.translation_step_m == 0.20
    assert config.max_expansions == 2_000_000


def test_open_map_generates_collision_free_route():
    geometry = _geometry()
    occupancy = np.full(
        (geometry.height, geometry.width),
        GridClass.FREE,
        dtype=np.int8,
    )
    x_values = np.linspace(1.0, 8.5, 80)
    trajectory = np.column_stack(
        [x_values, np.full_like(x_values, 3.0), np.zeros_like(x_values)]
    )
    yaw = np.zeros(x_values.size)

    plan = plan_farthest_trajectory_route(
        occupancy,
        geometry,
        trajectory,
        yaw,
        _config(),
    )

    assert plan.metrics["path_collision_pose_count"] == 0
    assert plan.metrics["path_length_m"] > 7.0
    assert plan.metrics["trajectory_exact_footprint_invalid_poses"] == 0
    assert plan.metrics["trajectory_lattice_footprint_invalid_poses"] == 0
    assert plan.metrics["path_exact_footprint_collision_pose_count"] == 0
    assert np.all(plan.trajectory_exact_valid)
    assert np.all(plan.trajectory_lattice_valid)
    assert np.all(plan.path_exact_valid)
    assert np.all(plan.path_swept_exact_valid)
    assert plan.metrics["path_exact_swept_collision_sample_count"] == 0
    assert plan.metrics["path_exact_swept_sample_count"] > len(plan.states)
    assert plan.metrics["trajectory_lattice_invalid_run_count"] == 0
    assert plan.metrics["selected_component_trajectory_pose_fraction"] == 1.0


def test_unknown_gap_narrower_than_footprint_is_not_a_valid_center():
    geometry = _geometry()
    occupancy = np.full(
        (geometry.height, geometry.width),
        GridClass.FREE,
        dtype=np.int8,
    )
    occupancy[:, 49:51] = GridClass.UNKNOWN
    occupancy[28:32, 49:51] = GridClass.FREE

    valid, _ = build_footprint_validity(
        occupancy,
        geometry,
        _config(robot_footprint_width_m=0.6),
    )

    assert not np.any(valid[:, 29:31, 49:51])


def test_turn_sweep_blocks_intermediate_orientation_collision():
    geometry = _geometry()
    occupancy = np.full(
        (geometry.height, geometry.width),
        GridClass.FREE,
        dtype=np.int8,
    )
    center_row = 30
    center_column = 50
    occupancy[center_row - 2, center_column - 4] = GridClass.OCCUPIED
    config = _config(
        robot_footprint_length_m=1.0,
        robot_footprint_width_m=0.2,
    )

    pose_valid, _ = build_footprint_validity(occupancy, geometry, config)
    turn_valid = build_turn_sweep_validity(occupancy, geometry, config)
    zero_heading = config.heading_bins // 2

    assert pose_valid[zero_heading, center_row, center_column]
    assert pose_valid[zero_heading + 1, center_row, center_column]
    assert not turn_valid[zero_heading, center_row, center_column]


def test_planning_config_rejects_non_quadrant_heading_count():
    try:
        _config(heading_bins=6)
    except ValueError as error:
        assert "multiple of four" in str(error)
    else:
        raise AssertionError("invalid heading lattice was accepted")


def test_exact_pose_validity_is_not_affected_by_heading_quantization():
    geometry = _geometry()
    occupancy = np.full(
        (geometry.height, geometry.width),
        GridClass.FREE,
        dtype=np.int8,
    )
    occupancy[26:34, 42] = GridClass.OCCUPIED
    poses = np.asarray(
        [
            [3.5, 3.0, np.deg2rad(22.0)],
            [4.1, 3.0, np.deg2rad(22.0)],
        ]
    )

    valid = exact_footprint_pose_validity(
        poses,
        occupancy,
        geometry,
        length_m=0.5,
        width_m=0.4,
    )

    assert valid.tolist() == [True, False]


def test_exact_pose_validity_blocks_footprint_outside_map():
    geometry = _geometry()
    occupancy = np.full(
        (geometry.height, geometry.width),
        GridClass.FREE,
        dtype=np.int8,
    )

    valid = exact_footprint_pose_validity(
        np.asarray([[0.1, 3.0, 0.0], [1.0, 3.0, 0.0]]),
        occupancy,
        geometry,
        length_m=0.5,
        width_m=0.4,
    )

    assert valid.tolist() == [False, True]


def test_path_densification_uses_short_wrapped_yaw_motion():
    dense = densify_pose_path_for_footprint(
        np.asarray(
            [
                [1.0, 1.0, np.deg2rad(170.0)],
                [1.0, 1.0, np.deg2rad(-170.0)],
            ]
        ),
        resolution_m=0.1,
        length_m=0.5,
        width_m=0.4,
    )

    assert dense.shape[0] > 2
    assert np.isclose(
        dense[-1, 2] - dense[0, 2],
        np.deg2rad(20.0),
    )
