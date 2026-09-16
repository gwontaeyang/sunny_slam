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

"""Unit tests for geometry-aware 3D-to-2D projection."""

import numpy as np

from slam_3d_map_projection.configuration import ProjectionConfig
from slam_3d_map_projection.projection import GridClass, PointClass, project_map


def _synthetic_room(with_ramp: bool = False):
    floor_points = []
    for x_value in np.arange(-2.0, 2.01, 0.1):
        for y_value in np.arange(-1.0, 1.01, 0.1):
            z_value = 0.08 * (x_value + 2.0) if with_ramp else 0.0
            floor_points.append((x_value, y_value, z_value))

    wall_points = []
    for x_value in np.arange(-2.0, 2.01, 0.1):
        for y_value in (-1.0, 1.0):
            floor_z = 0.08 * (x_value + 2.0) if with_ramp else 0.0
            for height in (0.3, 0.8, 1.4):
                wall_points.append((x_value, y_value, floor_z + height))

    ceiling_points = [
        (x_value, 0.0, 2.6)
        for x_value in np.arange(-2.0, 2.01, 0.1)
    ]
    trajectory_x = np.linspace(-1.8, 1.8, 73)
    trajectory_z = 0.08 * (trajectory_x + 2.0) if with_ramp else np.zeros_like(
        trajectory_x
    )
    trajectory = np.column_stack(
        [trajectory_x, np.zeros_like(trajectory_x), trajectory_z]
    )
    points = np.asarray(floor_points + wall_points + ceiling_points, dtype=np.float64)
    return points, trajectory


def _config(**overrides):
    values = {
        "resolution_m": 0.1,
        "bounds_quantile_low": 0.0,
        "bounds_quantile_high": 1.0,
        "initial_padding_m": 0.3,
        "output_padding_m": 0.2,
        "floor_support_radius_m": 0.18,
        "floor_smoothing_sigma_m": 0.16,
        "robot_footprint_length_m": 0.40,
        "robot_footprint_width_m": 0.40,
        "trajectory_footprint_margin_m": 0.0,
        "obstacle_keep_distance_to_free_m": 0.4,
        "obstacle_closing_radius_m": 0.0,
        "obstacle_dilation_radius_m": 0.05,
        "ramp_min_slope_deg": 2.0,
        "ramp_max_slope_deg": 10.0,
        "ramp_min_area_m2": 0.1,
    }
    values.update(overrides)
    return ProjectionConfig.from_mapping(values)


def _cell(result, x_value, y_value):
    columns, rows = result.geometry.world_to_grid(
        np.asarray([[x_value, y_value]], dtype=np.float64)
    )
    return int(rows[0]), int(columns[0])


def test_room_floor_is_free_and_walls_are_occupied():
    points, trajectory = _synthetic_room()
    result = project_map(points, trajectory, _config())

    center = _cell(result, 0.0, 0.0)
    upper_wall = _cell(result, 0.0, 1.0)
    outside = _cell(result, 0.0, 1.25)
    assert result.occupancy[center] == GridClass.FREE
    assert result.occupancy[upper_wall] == GridClass.OCCUPIED
    assert result.occupancy[outside] == GridClass.UNKNOWN
    assert result.metrics["trajectory"]["occupied_cells_after_clear"] == 0


def test_enclosed_unknown_floor_hole_can_be_completed_without_opening_exterior():
    points, trajectory = _synthetic_room()
    side_walls = np.asarray(
        [
            (x_value, y_value, height)
            for x_value in (-2.0, 2.0)
            for y_value in np.arange(-1.0, 1.01, 0.1)
            for height in (0.3, 0.8, 1.4)
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, side_walls])
    trajectory = trajectory[np.abs(trajectory[:, 0]) <= 1.5]
    floor_hole = (
        np.isclose(points[:, 2], 0.0)
        & (np.abs(points[:, 0]) <= 0.35)
        & (points[:, 1] >= 0.35)
        & (points[:, 1] <= 0.85)
    )
    points = points[~floor_hole]

    conservative = project_map(
        points,
        trajectory,
        _config(obstacle_closing_radius_m=0.15),
    )
    completed = project_map(
        points,
        trajectory,
        _config(
            obstacle_closing_radius_m=0.15,
            fill_enclosed_unknown_as_free=True,
        ),
    )

    hole = _cell(completed, 0.0, 0.60)
    exterior = _cell(completed, 0.0, 1.25)
    assert conservative.occupancy[hole] == GridClass.UNKNOWN
    assert completed.occupancy[hole] == GridClass.FREE
    assert completed.enclosed_unknown_filled_mask[hole]
    assert completed.occupancy[exterior] == GridClass.UNKNOWN
    assert not completed.enclosed_unknown_filled_mask[exterior]
    assert completed.metrics["interior_free_completion"]["filled_cells"] > 0


def test_ramp_remains_free_and_is_labeled_separately():
    points, trajectory = _synthetic_room(with_ramp=True)
    result = project_map(points, trajectory, _config())

    center = _cell(result, 0.0, 0.0)
    assert result.occupancy[center] == GridClass.FREE
    assert result.ramp_mask[center]
    assert result.metrics["grid"]["ramp_area_m2"] > 0.0


def test_high_overhang_does_not_turn_floor_into_obstacle():
    points, trajectory = _synthetic_room()
    result = project_map(points, trajectory, _config(obstacle_max_height_m=2.0))

    center = _cell(result, 0.0, 0.0)
    assert result.occupancy[center] == GridClass.FREE
    assert result.metrics["point_class_counts"]["high_overhang"] > 0


def test_disjoint_pcd_and_trajectory_frames_are_rejected():
    points, trajectory = _synthetic_room()
    trajectory[:, 0] += 100.0

    try:
        project_map(
            points,
            trajectory,
            _config(max_floor_reference_distance_m=2.0),
        )
    except ValueError as error:
        assert "same map frame" in str(error)
    else:
        raise AssertionError("disjoint PCD and trajectory frames were accepted")


def test_invalid_overlapping_height_bands_are_rejected():
    try:
        ProjectionConfig(floor_max_offset_m=0.2, obstacle_min_height_m=0.1)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping floor and obstacle bands were accepted")


def test_obstacle_evidence_masks_are_disjoint_and_explain_occupied_cells():
    points, trajectory = _synthetic_room()
    result = project_map(
        points,
        trajectory,
        _config(
            obstacle_closing_radius_m=0.15,
            obstacle_dilation_radius_m=0.10,
        ),
    )

    masks = (
        result.obstacle_endpoint_mask,
        result.obstacle_bridge_mask,
        result.obstacle_dilation_mask,
    )
    for first_index, first in enumerate(masks):
        for second in masks[first_index + 1:]:
            assert not np.any(first & second)

    explained = np.logical_or.reduce(masks)
    final_expected = explained & ~result.obstacle_cleared_by_trajectory_mask
    np.testing.assert_array_equal(
        result.occupancy == GridClass.OCCUPIED,
        final_expected,
    )
    evidence = result.metrics["obstacle_evidence"]
    assert evidence["final_endpoint_cells"] > 0
    assert evidence["final_derived_cells"] > 0
    assert 0.0 < evidence["final_derived_fraction"] < 1.0


def test_oriented_footprint_clears_obstacle_at_front_corner():
    points, trajectory_xyz = _synthetic_room()
    corner_obstacles = np.asarray(
        [
            (x_value, 0.30, height)
            for x_value in (-1.0, -0.9, -0.8)
            for height in (0.3, 0.8)
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, corner_obstacles])
    trajectory = np.column_stack(
        [trajectory_xyz, np.zeros(trajectory_xyz.shape[0], dtype=np.float64)]
    )
    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_length_m=0.60,
            robot_footprint_width_m=0.70,
            obstacle_closing_radius_m=0.0,
            obstacle_dilation_radius_m=0.0,
        ),
    )

    assert result.metrics["trajectory"]["poses_in_collision_before_clear"] > 0
    assert result.metrics["trajectory"]["poses_in_collision_after_clear"] == 0
    assert np.count_nonzero(result.obstacle_cleared_by_trajectory_mask) > 0


def test_floor_class_has_priority_at_shared_height_boundary():
    points, trajectory = _synthetic_room()
    result = project_map(
        points,
        trajectory,
        _config(
            floor_max_offset_m=0.0,
            obstacle_min_height_m=0.0,
            obstacle_closing_radius_m=0.0,
            obstacle_dilation_radius_m=0.0,
        ),
    )

    floor_z_points = np.isclose(result.relative_point_height_m, 0.0)
    assert np.any(floor_z_points)
    assert np.all(result.point_classes[floor_z_points] == PointClass.FLOOR)


def test_obstacle_without_local_floor_support_remains_unreferenced():
    points, trajectory = _synthetic_room()
    unsupported = np.asarray(
        [
            (6.0, -0.05, 0.5),
            (6.0, 0.00, 0.5),
            (6.0, 0.05, 0.5),
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, unsupported])

    result = project_map(points, trajectory, _config())

    assert np.all(result.point_classes[-len(unsupported):] == PointClass.UNREFERENCED)
    columns, rows = result.geometry.world_to_grid(unsupported[:, :2])
    in_output = (
        (columns >= 0)
        & (columns < result.geometry.width)
        & (rows >= 0)
        & (rows < result.geometry.height)
    )
    if np.any(in_output):
        assert np.all(
            result.occupancy[rows[in_output], columns[in_output]] == GridClass.UNKNOWN
        )


def test_discontinuous_trajectory_does_not_clear_wall_between_poses():
    points, _ = _synthetic_room()
    center_obstacle = np.asarray(
        [
            (-0.05, 0.0, 0.4),
            (0.00, 0.0, 0.4),
            (0.05, 0.0, 0.4),
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, center_obstacle])
    trajectory = np.asarray(
        [[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_length_m=0.2,
            robot_footprint_width_m=0.2,
            trajectory_max_segment_xy_m=0.25,
            obstacle_closing_radius_m=0.0,
            obstacle_dilation_radius_m=0.0,
        ),
    )

    center = _cell(result, 0.0, 0.0)
    assert result.occupancy[center] == GridClass.OCCUPIED
    assert not result.trajectory_mask[center]
    assert result.metrics["trajectory"]["invalid_union_segments"] == 1


def test_timestamp_gap_does_not_clear_obstacle_between_nearby_poses():
    points, _ = _synthetic_room()
    center_obstacle = np.asarray(
        [
            (-0.05, 0.0, 0.4),
            (0.00, 0.0, 0.4),
            (0.05, 0.0, 0.4),
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, center_obstacle])
    trajectory = np.asarray(
        [[-0.20, 0.0, 0.0, 0.0], [0.20, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )

    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_length_m=0.1,
            robot_footprint_width_m=0.1,
            trajectory_max_segment_xy_m=0.5,
            trajectory_max_segment_dt_s=0.5,
            obstacle_closing_radius_m=0.0,
            obstacle_dilation_radius_m=0.0,
        ),
        trajectory_stamp_ns=np.asarray([0, 1_000_000_000], dtype=np.int64),
    )

    center = _cell(result, 0.0, 0.0)
    assert result.occupancy[center] == GridClass.OCCUPIED
    assert result.metrics["trajectory"]["time_discontinuity_segments"] == 1


def test_supported_below_floor_returns_make_untraversed_cell_unknown():
    points, trajectory = _synthetic_room()
    below_floor_points = np.asarray(
        [
            (0.45, 0.50, -0.45),
            (0.50, 0.50, -0.50),
            (0.55, 0.50, -0.55),
        ],
        dtype=np.float64,
    )
    points = np.vstack([points, below_floor_points])

    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_width_m=0.2,
            below_floor_min_support_points=2,
            fill_enclosed_unknown_as_free=True,
        ),
    )

    cell = _cell(result, 0.50, 0.50)
    assert result.below_floor_endpoint_mask[cell]
    assert result.terrain_hazard_mask[cell]
    assert result.occupancy[cell] == GridClass.UNKNOWN
    assert not result.enclosed_unknown_filled_mask[cell]


def test_excessive_slope_is_unknown_outside_driven_footprint():
    points, trajectory = _synthetic_room(with_ramp=True)

    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_width_m=0.2,
            max_traversable_floor_slope_deg=4.0,
            max_traversable_floor_roughness_m=1.0,
        ),
    )

    cell = _cell(result, 0.0, 0.50)
    assert result.excessive_slope_mask[cell]
    assert result.occupancy[cell] == GridClass.UNKNOWN


def test_smooth_ramp_plane_is_not_mislabeled_as_rough():
    points, trajectory = _synthetic_room(with_ramp=True)

    result = project_map(
        points,
        trajectory,
        _config(
            robot_footprint_width_m=0.2,
            max_traversable_floor_slope_deg=20.0,
            max_traversable_floor_roughness_m=0.005,
        ),
    )

    cell = _cell(result, 0.0, 0.50)
    assert result.floor_slope_deg[cell] > 4.0
    assert result.floor_roughness_m[cell] < 0.005
    assert not result.excessive_roughness_mask[cell]
    assert result.occupancy[cell] == GridClass.FREE
    assert (
        result.metrics["terrain_hazards"]["roughness_definition"]
        == "sqrt_local_pca_minimum_eigenvalue_m"
    )


def test_stacked_trajectory_levels_are_rejected_for_single_grid():
    points, _ = _synthetic_room()
    trajectory = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.01, 1.0]],
        dtype=np.float64,
    )

    try:
        project_map(points, trajectory, _config())
    except ValueError as error:
        assert "multiple z levels" in str(error)
    else:
        raise AssertionError("stacked trajectory levels were projected into one grid")
