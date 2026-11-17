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

"""Tests for coarse-evidence and fine-obstacle navigation rasterization."""

import math

import numpy as np
import pytest

from slam_3d_map_projection.multiresolution import refine_navigation_grid
from slam_3d_map_projection.projection import GridClass, GridGeometry


def test_fine_grid_rerasterizes_obstacles_and_fills_only_enclosed_unknown():
    geometry = GridGeometry(0.0, 0.0, 0.1, 10, 10)
    occupancy = np.full((10, 10), GridClass.UNKNOWN, dtype=np.int8)
    occupancy[2:8, 2:8] = GridClass.FREE
    occupancy[4:6, 4:6] = GridClass.UNKNOWN
    ramp = np.zeros((10, 10), dtype=bool)
    hazard = np.zeros((10, 10), dtype=bool)
    coarse_completed = np.zeros((10, 10), dtype=bool)
    obstacle_points = np.asarray(
        [[0.25, 0.25, 0.5], [0.35, 0.25, 0.5]],
        dtype=np.float64,
    )

    result = refine_navigation_grid(
        occupancy,
        geometry,
        obstacle_points,
        ramp,
        hazard,
        coarse_completed,
        output_resolution_m=0.02,
        obstacle_closing_radius_m=0.0,
        enclosure_closing_radius_m=0.0,
        fill_enclosed_unknown_as_free=True,
    )

    assert result.geometry.width == 50
    assert result.geometry.height == 50
    center_column, center_row = result.geometry.world_to_grid(
        np.asarray([[0.5, 0.5]], dtype=np.float64)
    )
    assert result.occupancy[center_row[0], center_column[0]] == GridClass.FREE
    assert result.enclosed_unknown_filled_mask[
        center_row[0], center_column[0]
    ]
    outside_column, outside_row = result.geometry.world_to_grid(
        np.asarray([[0.05, 0.05]], dtype=np.float64)
    )
    assert result.occupancy[outside_row[0], outside_column[0]] == GridClass.UNKNOWN
    obstacle_column, obstacle_row = result.geometry.world_to_grid(
        obstacle_points[:1, :2]
    )
    assert (
        result.occupancy[obstacle_row[0], obstacle_column[0]]
        == GridClass.OCCUPIED
    )
    assert result.metrics["output_resolution_m"] == 0.02
    assert (
        result.metrics["unknown_provenance"][
            "interior_nonhazard_cells_after_wall_inference"
        ]
        == 0
    )


def test_default_keeps_enclosed_and_previously_completed_floor_unknown():
    geometry = GridGeometry(0.0, 0.0, 0.1, 10, 10)
    occupancy = np.full((10, 10), GridClass.UNKNOWN, dtype=np.int8)
    occupancy[2:8, 2:8] = GridClass.FREE
    occupancy[4:6, 4:6] = GridClass.UNKNOWN
    coarse_completed = np.zeros((10, 10), dtype=bool)
    coarse_completed[2, 2] = True
    mask = np.zeros((10, 10), dtype=bool)
    points = np.asarray([[0.75, 0.75, 0.5]], dtype=np.float64)

    result = refine_navigation_grid(
        occupancy,
        geometry,
        points,
        mask,
        mask,
        coarse_completed,
        output_resolution_m=0.02,
        obstacle_closing_radius_m=0.0,
        enclosure_closing_radius_m=0.0,
    )

    center_column, center_row = result.geometry.world_to_grid(
        np.asarray([[0.5, 0.5]], dtype=np.float64)
    )
    assert result.occupancy[center_row[0], center_column[0]] == GridClass.UNKNOWN
    completed_column, completed_row = result.geometry.world_to_grid(
        np.asarray([[0.25, 0.25]], dtype=np.float64)
    )
    assert (
        result.occupancy[completed_row[0], completed_column[0]]
        == GridClass.UNKNOWN
    )
    assert not np.any(result.enclosed_unknown_filled_mask)
    assert result.metrics["fill_enclosed_unknown_as_free"] is False
    assert result.metrics["newly_completed_free_cells"] == 0

    completed_result = refine_navigation_grid(
        occupancy,
        geometry,
        points,
        mask,
        mask,
        coarse_completed,
        output_resolution_m=0.02,
        obstacle_closing_radius_m=0.0,
        enclosure_closing_radius_m=0.0,
        fill_enclosed_unknown_as_free=True,
    )
    assert np.array_equal(
        result.occupancy == GridClass.OCCUPIED,
        completed_result.occupancy == GridClass.OCCUPIED,
    )


def test_enclosure_closing_cannot_be_smaller_than_final_obstacle_closing():
    geometry = GridGeometry(0.0, 0.0, 0.1, 2, 2)
    occupancy = np.full((2, 2), GridClass.FREE, dtype=np.int8)
    mask = np.zeros((2, 2), dtype=bool)
    points = np.asarray([[0.05, 0.05, 0.5]], dtype=np.float64)

    try:
        refine_navigation_grid(
            occupancy,
            geometry,
            points,
            mask,
            mask,
            mask,
            obstacle_closing_radius_m=0.10,
            enclosure_closing_radius_m=0.08,
        )
    except ValueError as error:
        assert "cannot be smaller" in str(error)
    else:
        raise AssertionError("invalid enclosure closing radii were accepted")


@pytest.mark.parametrize("radius", [math.nan, math.inf])
def test_nonfinite_closing_radii_are_rejected(radius):
    geometry = GridGeometry(0.0, 0.0, 0.1, 2, 2)
    occupancy = np.full((2, 2), GridClass.FREE, dtype=np.int8)
    mask = np.zeros((2, 2), dtype=bool)
    points = np.asarray([[0.05, 0.05, 0.5]], dtype=np.float64)

    with pytest.raises(ValueError, match="finite"):
        refine_navigation_grid(
            occupancy,
            geometry,
            points,
            mask,
            mask,
            mask,
            obstacle_closing_radius_m=radius,
        )
