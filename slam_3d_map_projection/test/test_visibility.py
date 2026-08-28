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

"""Tests for height-banded raw scan visibility evidence."""

from dataclasses import replace

import numpy as np

from slam_3d_map_projection.configuration import ProjectionConfig
from slam_3d_map_projection.projection import GridClass, project_map
from slam_3d_map_projection.trajectory_io import TrajectoryData
from slam_3d_map_projection.visibility import (
    EvidenceDecision,
    ObstacleCandidates,
    RawVisibilityConfig,
    VisibilityAccumulator,
    VisibilityEvidence,
    apply_visibility_refinement,
    build_obstacle_candidates,
)


def _identity_trajectory():
    return TrajectoryData(
        poses=np.asarray(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
        orientation_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        stamp_ns=np.asarray([1_000_000_000, 2_000_000_000], dtype=np.int64),
        frame_id="map",
        yaw_source="quaternion",
    )


def _visibility_config(**overrides):
    values = {
        "sensor_translation_xyz_m": (0.0, 0.0, 0.0),
        "sensor_rotation_rpy_deg": (0.0, 0.0, 0.0),
        "evidence_resolution_m": 0.2,
        "height_band_edges_m": (0.1, 2.0),
        "min_range_m": 0.1,
        "max_range_m": 20.0,
        "azimuth_bin_deg": 1.0,
        "azimuth_neighbor_bins": 0,
        "vertical_neighbor_rings": 0,
        "max_ring_angle_error_deg": 5.0,
        "hit_tolerance_m": 0.25,
        "miss_margin_m": 0.4,
        "occlusion_margin_m": 0.25,
        "min_confirming_hit_scans": 1,
        "min_hit_fraction_for_confirmation": 0.5,
        "min_contradicting_miss_scans": 1,
        "max_hit_fraction_for_contradiction": 0.1,
        "contradiction_miss_to_hit_ratio": 1.0,
        "max_pose_time_offset_s": 0.6,
        "max_pose_interpolation_gap_s": 2.0,
    }
    values.update(overrides)
    return RawVisibilityConfig(**values)


def test_one_scan_separates_hit_miss_occluded_and_unobserved():
    candidate_points = np.asarray(
        [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-5.0, 0.0, 0.0], [0.0, -5.0, 0.0]],
        dtype=np.float64,
    )
    candidates = ObstacleCandidates(
        points_map=candidate_points,
        band_indices=np.zeros(4, dtype=np.int16),
        coarse_rows=np.zeros(4, dtype=np.int32),
        coarse_columns=np.arange(4, dtype=np.int32),
        candidate_id_by_band=np.arange(4, dtype=np.int32).reshape(1, 1, 4),
        point_candidate_ids=np.arange(4, dtype=np.int32),
        coarse_origin_xy_m=np.zeros(2),
        coarse_resolution_m=0.2,
    )
    accumulator = VisibilityAccumulator(
        candidates,
        _identity_trajectory(),
        _visibility_config(),
    )
    points = np.asarray(
        [
            [5.1, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [-3.0, 0.0, 0.0],
            [4.0, 0.0, 1.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=np.float64,
    )
    rings = np.asarray([0, 0, 0, 1, 1], dtype=np.uint16)
    timestamps = np.full(points.shape[0], 1.5, dtype=np.float64)

    assert accumulator.observe_scan(
        points,
        rings,
        timestamps,
        1.5,
        "hesai_center",
    )
    evidence = accumulator.finish()

    np.testing.assert_array_equal(evidence.hit_scan_count, [1, 0, 0, 0])
    np.testing.assert_array_equal(evidence.miss_scan_count, [0, 1, 0, 0])
    np.testing.assert_array_equal(evidence.occluded_scan_count, [0, 0, 1, 0])
    np.testing.assert_array_equal(evidence.unobserved_scan_count, [0, 0, 0, 1])
    np.testing.assert_array_equal(
        evidence.decisions,
        [
            EvidenceDecision.CONFIRMED,
            EvidenceDecision.CONTRADICTED,
            EvidenceDecision.UNCERTAIN,
            EvidenceDecision.UNCERTAIN,
        ],
    )


def _room():
    floor = [
        (x_value, y_value, 0.0)
        for x_value in np.arange(-1.0, 1.01, 0.1)
        for y_value in np.arange(-0.6, 0.61, 0.1)
    ]
    wall = [
        (x_value, 0.6, height)
        for x_value in np.arange(-1.0, 1.01, 0.1)
        for height in (0.3, 0.8, 1.2)
    ]
    points = np.asarray(floor + wall, dtype=np.float64)
    x_values = np.linspace(-0.8, 0.8, 33)
    trajectory = np.column_stack(
        [x_values, np.zeros_like(x_values), np.zeros_like(x_values)]
    )
    return points, trajectory


def test_all_height_bands_must_be_contradicted_before_endpoint_is_removed():
    points, trajectory = _room()
    projection_config = ProjectionConfig(
        resolution_m=0.1,
        bounds_quantile_low=0.0,
        bounds_quantile_high=1.0,
        initial_padding_m=0.2,
        output_padding_m=0.2,
        floor_support_radius_m=0.2,
        floor_smoothing_sigma_m=0.15,
        robot_footprint_length_m=0.3,
        robot_footprint_width_m=0.3,
        obstacle_keep_distance_to_free_m=0.8,
        obstacle_closing_radius_m=0.0,
        obstacle_dilation_radius_m=0.0,
    )
    result = project_map(points, trajectory, projection_config)
    visibility_config = _visibility_config(
        evidence_resolution_m=0.2,
        height_band_edges_m=(0.12, 0.6, 2.0),
    )
    candidates = build_obstacle_candidates(points, result, visibility_config)
    expected_columns = np.floor(
        (candidates.points_map[:, 0] - result.geometry.origin_x_m)
        / visibility_config.evidence_resolution_m
    ).astype(np.int64)
    expected_rows = np.floor(
        (candidates.points_map[:, 1] - result.geometry.origin_y_m)
        / visibility_config.evidence_resolution_m
    ).astype(np.int64)
    np.testing.assert_array_equal(candidates.coarse_columns, expected_columns)
    np.testing.assert_array_equal(candidates.coarse_rows, expected_rows)
    np.testing.assert_array_equal(
        candidates.candidate_id_by_band[
            candidates.band_indices,
            candidates.coarse_rows,
            candidates.coarse_columns,
        ],
        np.arange(candidates.points_map.shape[0]),
    )
    decisions = np.full(
        candidates.points_map.shape[0],
        EvidenceDecision.CONTRADICTED,
        dtype=np.int8,
    )
    evidence = VisibilityEvidence(
        hit_scan_count=np.zeros(decisions.size, dtype=np.uint32),
        miss_scan_count=np.full(decisions.size, 20, dtype=np.uint32),
        occluded_scan_count=np.zeros(decisions.size, dtype=np.uint32),
        ambiguous_scan_count=np.zeros(decisions.size, dtype=np.uint32),
        unobserved_scan_count=np.zeros(decisions.size, dtype=np.uint32),
        decisions=decisions,
        scan_metrics={},
    )

    refinement = apply_visibility_refinement(
        result,
        candidates,
        evidence,
        projection_config,
    )

    assert refinement.metrics["contradicted_endpoint_cells"] > 0
    assert refinement.metrics["new_occupied_cells"] < refinement.metrics[
        "old_occupied_cells"
    ]
    assert not np.any(
        refinement.occupancy[refinement.recovered_free_mask]
        != GridClass.FREE
    )
    assert not np.any(
        refinement.retained_endpoint_mask
        & (refinement.occupancy != GridClass.OCCUPIED)
    )
    assert not np.any(
        result.trajectory_mask
        & (refinement.occupancy == GridClass.OCCUPIED)
    )


def test_visibility_config_rejects_unsorted_height_bands():
    try:
        _visibility_config(height_band_edges_m=(0.1, 0.8, 0.7, 2.0))
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("unsorted height bands were accepted")


def test_refinement_preserves_border_endpoint_and_reclears_trajectory_bridge():
    points, trajectory = _room()
    projection_config = ProjectionConfig(
        resolution_m=0.1,
        bounds_quantile_low=0.0,
        bounds_quantile_high=1.0,
        initial_padding_m=0.2,
        output_padding_m=0.2,
        floor_support_radius_m=0.2,
        floor_smoothing_sigma_m=0.15,
        robot_footprint_length_m=0.3,
        robot_footprint_width_m=0.3,
        obstacle_keep_distance_to_free_m=0.8,
        obstacle_closing_radius_m=0.1,
        obstacle_dilation_radius_m=0.0,
    )
    result = project_map(points, trajectory, projection_config)
    shape = result.occupancy.shape
    endpoint = np.zeros(shape, dtype=bool)
    middle_row = shape[0] // 2
    middle_column = shape[1] // 2
    candidate_rows = np.asarray(
        [
            0,
            middle_row - 1,
            middle_row,
            middle_row + 1,
            middle_row - 1,
            middle_row,
            middle_row + 1,
        ],
        dtype=np.int32,
    )
    candidate_columns = np.asarray(
        [
            0,
            middle_column - 1,
            middle_column - 1,
            middle_column - 1,
            middle_column + 1,
            middle_column + 1,
            middle_column + 1,
        ],
        dtype=np.int32,
    )
    endpoint[candidate_rows, candidate_columns] = True
    occupancy = np.full(shape, GridClass.UNKNOWN, dtype=np.int8)
    occupancy[endpoint] = GridClass.OCCUPIED
    trajectory_mask = np.zeros(shape, dtype=bool)
    trajectory_mask[middle_row, middle_column] = True
    result = replace(
        result,
        occupancy=occupancy,
        obstacle_endpoint_mask=endpoint,
        trajectory_mask=trajectory_mask,
    )

    candidate_grid = np.full((1, shape[0], shape[1]), -1, dtype=np.int32)
    candidate_count = candidate_rows.size
    candidate_grid[0, candidate_rows, candidate_columns] = np.arange(
        candidate_count
    )
    candidate_points = np.column_stack(
        [
            result.geometry.origin_x_m
            + (candidate_columns + 0.5) * result.geometry.resolution_m,
            result.geometry.origin_y_m
            + (candidate_rows + 0.5) * result.geometry.resolution_m,
            np.full(candidate_count, 0.5),
        ]
    )
    candidates = ObstacleCandidates(
        points_map=candidate_points,
        band_indices=np.zeros(candidate_count, dtype=np.int16),
        coarse_rows=candidate_rows,
        coarse_columns=candidate_columns,
        candidate_id_by_band=candidate_grid,
        point_candidate_ids=np.full(result.point_classes.shape, -1, dtype=np.int32),
        coarse_origin_xy_m=np.asarray(
            [result.geometry.origin_x_m, result.geometry.origin_y_m]
        ),
        coarse_resolution_m=result.geometry.resolution_m,
    )
    confirmed = np.full(
        candidate_count,
        EvidenceDecision.CONFIRMED,
        dtype=np.int8,
    )
    evidence = VisibilityEvidence(
        hit_scan_count=np.ones(candidate_count, dtype=np.uint32),
        miss_scan_count=np.zeros(candidate_count, dtype=np.uint32),
        occluded_scan_count=np.zeros(candidate_count, dtype=np.uint32),
        ambiguous_scan_count=np.zeros(candidate_count, dtype=np.uint32),
        unobserved_scan_count=np.zeros(candidate_count, dtype=np.uint32),
        decisions=confirmed,
        scan_metrics={},
    )

    refinement = apply_visibility_refinement(
        result,
        candidates,
        evidence,
        projection_config,
    )

    assert np.all(refinement.occupancy[endpoint] == GridClass.OCCUPIED)
    assert (
        refinement.occupancy[middle_row, middle_column]
        != GridClass.OCCUPIED
    )
    assert refinement.metrics["trajectory_recleared_cells"] >= 1
