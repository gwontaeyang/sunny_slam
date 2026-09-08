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

"""File-contract tests for Nav2 map-server output."""

from pathlib import Path

import numpy as np
import yaml

from slam_3d_map_projection.configuration import ProjectionConfig
from slam_3d_map_projection.outputs import write_projection_outputs
from slam_3d_map_projection.pcd_io import read_pcd_xyz
from slam_3d_map_projection.projection import GridClass, project_map


def _read_pgm(path: Path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"P5"
        width, height = [int(value) for value in stream.readline().split()]
        assert stream.readline().strip() == b"255"
        image = np.frombuffer(stream.read(), dtype=np.uint8).reshape(height, width)
    return image


def test_binary_pcd_reader_and_nav2_output_orientation(tmp_path):
    input_pcd = tmp_path / "input.pcd"
    points = np.asarray(
        [
            (-0.5, -0.5, 0.0),
            (0.0, -0.5, 0.0),
            (0.5, -0.5, 0.0),
            (-0.5, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.5, 0.8),
        ],
        dtype="<f4",
    )
    header = (
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        "COUNT 1 1 1\nWIDTH 8\nHEIGHT 1\nPOINTS 8\nDATA binary\n"
    )
    with input_pcd.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())
    loaded = read_pcd_xyz(input_pcd)
    np.testing.assert_allclose(loaded, points)

    trajectory = np.asarray([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    config = ProjectionConfig(
        resolution_m=0.1,
        bounds_quantile_low=0.0,
        bounds_quantile_high=1.0,
        initial_padding_m=0.2,
        output_padding_m=0.1,
        floor_support_radius_m=0.18,
        floor_normal_neighbors=3,
        floor_normal_max_neighbor_distance_m=0.8,
        floor_surface_spread_min=0.0,
        robot_footprint_length_m=0.2,
        robot_footprint_width_m=0.2,
        trajectory_footprint_margin_m=0.0,
        obstacle_keep_distance_to_free_m=0.4,
        obstacle_closing_radius_m=0.0,
        obstacle_dilation_radius_m=0.0,
        ramp_min_area_m2=0.0,
    )
    result = project_map(loaded, trajectory, config)
    outputs = write_projection_outputs(
        tmp_path, "room", loaded, trajectory, result, config
    )
    image = _read_pgm(outputs["map_image"])
    assert image.shape == (result.geometry.height, result.geometry.width)
    np.testing.assert_array_equal(
        image[0],
        np.flipud(
            np.where(
                result.occupancy == GridClass.FREE,
                config.free_value,
                np.where(
                    result.occupancy == GridClass.OCCUPIED,
                    config.occupied_value,
                    config.unknown_value,
                ),
            )
        )[0],
    )
    yaml_text = outputs["map_yaml"].read_text(encoding="utf-8")
    assert "mode: trinary" in yaml_text
    assert "resolution: 0.1" in yaml_text
    map_metadata = yaml.safe_load(yaml_text)
    free_probability = (255.0 - config.free_value) / 255.0
    unknown_probability = (255.0 - config.unknown_value) / 255.0
    occupied_probability = (255.0 - config.occupied_value) / 255.0
    assert free_probability < map_metadata["free_thresh"] < unknown_probability
    assert unknown_probability < map_metadata["occupied_thresh"]
    assert map_metadata["occupied_thresh"] < occupied_probability
    assert outputs["ramp_mask_yaml"].is_file()
    assert outputs["classified_pcd"].is_file()
    with np.load(outputs["terrain_layers"]) as layers:
        assert "obstacle_endpoint_mask" in layers
        assert "obstacle_bridge_mask" in layers
        assert "obstacle_dilation_mask" in layers
        assert "obstacle_cleared_by_trajectory_mask" in layers
        assert "terrain_hazard_mask" in layers
        assert "obstacle_rejected_low_support_mask" in layers
        assert layers["trajectory_yaw_rad"].shape == (trajectory.shape[0],)
        assert layers["trajectory_yaw_source"].item() == "inferred_xy_tangent"
        assert layers["trajectory_segment_valid"].shape == (
            trajectory.shape[0] - 1,
        )
