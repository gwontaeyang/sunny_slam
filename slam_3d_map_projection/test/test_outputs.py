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

import json
from pathlib import Path
import struct
import zlib

import numpy as np
import yaml

from slam_3d_map_projection.configuration import ProjectionConfig
from slam_3d_map_projection.outputs import write_projection_outputs
from slam_3d_map_projection.path_planning import (
    PathPlan,
    PathPlanningConfig,
)
from slam_3d_map_projection.pcd_io import read_pcd_xyz
from slam_3d_map_projection.projection import GridClass, project_map


def test_cli_help_distinguishes_offline_cleanup_from_core_global_cloud():
    from slam_3d_map_projection.cli import _parser

    help_text = " ".join(_parser().format_help().split())
    assert "slam_3d_map_cleanup map_static.pcd" in help_text
    assert "not the uncleaned core global_cloud" in help_text
    assert "same corrected state as the PCD" in help_text


def _read_pgm(path: Path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"P5"
        width, height = [int(value) for value in stream.readline().split()]
        assert stream.readline().strip() == b"255"
        image = np.frombuffer(stream.read(), dtype=np.uint8).reshape(height, width)
    return image


def _read_grayscale_png(path: Path):
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    width = None
    height = None
    compressed = bytearray()
    while offset < len(payload):
        size = struct.unpack(">I", payload[offset:offset + 4])[0]
        chunk_type = payload[offset + 4:offset + 8]
        chunk_data = payload[offset + 8:offset + 8 + size]
        expected_crc = struct.unpack(
            ">I", payload[offset + 8 + size:offset + 12 + size]
        )[0]
        assert zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF == expected_crc
        offset += 12 + size
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
            assert bit_depth == 8
            assert color_type == 0
            assert compression == 0
            assert filtering == 0
            assert interlace == 0
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    assert width is not None
    assert height is not None
    decoded = zlib.decompress(compressed)
    rows = np.frombuffer(decoded, dtype=np.uint8).reshape(height, width + 1)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].copy()


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
            (0.0, 0.2, 0.0),
            (0.0, 0.2, 0.8),
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

    trajectory_x = np.linspace(-0.4, 0.4, 9)
    trajectory = np.column_stack(
        [trajectory_x, np.zeros_like(trajectory_x), np.zeros_like(trajectory_x)]
    )
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
        obstacle_min_support_points=1,
        obstacle_closing_radius_m=0.0,
        obstacle_dilation_radius_m=0.0,
        ramp_min_area_m2=0.0,
    )
    result = project_map(loaded, trajectory, config)
    planning_config = PathPlanningConfig(
        robot_footprint_length_m=0.05,
        robot_footprint_width_m=0.05,
        heading_bins=8,
        translation_step_m=0.1,
        clearance_cost_weight=0.0,
        goal_tolerance_m=0.11,
        max_expansions=10_000,
    )
    path_poses = np.column_stack([trajectory[:, :2], result.trajectory_yaw_rad])
    path_plan = PathPlan(
        states=np.zeros((trajectory.shape[0], 3), dtype=np.int32),
        poses_xy_yaw=path_poses,
        start_trajectory_index=0,
        goal_trajectory_index=trajectory.shape[0] - 1,
        expansions=0,
        trajectory_exact_valid=np.ones(trajectory.shape[0], dtype=bool),
        trajectory_lattice_valid=np.ones(trajectory.shape[0], dtype=bool),
        path_exact_valid=np.ones(trajectory.shape[0], dtype=bool),
        path_swept_poses_xy_yaw=path_poses,
        path_swept_exact_valid=np.ones(trajectory.shape[0], dtype=bool),
        metrics={"path_exact_swept_collision_sample_count": 0},
    )
    outputs = write_projection_outputs(
        tmp_path,
        "room",
        loaded,
        trajectory,
        result,
        config,
        path_plan=path_plan,
        path_planning_config=planning_config,
    )
    image = _read_pgm(outputs["map_image"])
    png_image = _read_grayscale_png(outputs["map_png"])
    assert image.shape == (result.geometry.height, result.geometry.width)
    np.testing.assert_array_equal(png_image, image)
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
    png_map_metadata = yaml.safe_load(
        outputs["map_png_yaml"].read_text(encoding="utf-8")
    )
    assert png_map_metadata["image"] == outputs["map_png"].name
    assert png_map_metadata["resolution"] == map_metadata["resolution"]
    assert png_map_metadata["origin"] == map_metadata["origin"]
    assert png_map_metadata["occupied_thresh"] == map_metadata["occupied_thresh"]
    assert png_map_metadata["free_thresh"] == map_metadata["free_thresh"]
    free_probability = (255.0 - config.free_value) / 255.0
    unknown_probability = (255.0 - config.unknown_value) / 255.0
    occupied_probability = (255.0 - config.occupied_value) / 255.0
    assert free_probability < map_metadata["free_thresh"] < unknown_probability
    assert unknown_probability < map_metadata["occupied_thresh"]
    assert map_metadata["occupied_thresh"] < occupied_probability
    assert outputs["ramp_mask_yaml"].is_file()
    assert outputs["classified_pcd"].is_file()
    planning_obstacles = read_pcd_xyz(outputs["planning_obstacles_pcd"])
    assert planning_obstacles.shape[0] > 0
    assert np.all(planning_obstacles[:, 2] > config.obstacle_min_height_m)
    assert outputs["planned_path"].is_file()
    assert outputs["path_layers"].is_file()
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["outputs"]["map_png"] == outputs["map_png"].name
    assert metrics["outputs"]["map_png_yaml"] == outputs["map_png_yaml"].name
    assert metrics["planning_obstacle_export"]["selected_points"] == (
        planning_obstacles.shape[0]
    )
    assert metrics["planning_obstacle_export"]["dynamic_object_contract"] == (
        "input PCD must already be cleaned offline by slam_3d_map_cleanup "
        "(map_static.pcd) or equivalent; core global_cloud is temporally "
        "unfiltered; cleanup provenance is not verified by projection"
    )
    assert metrics["path_planning"]["path_exact_swept_collision_sample_count"] == 0
    assert metrics["path_planning_parameters"]["heading_bins"] == 8
    with np.load(outputs["terrain_layers"]) as layers:
        assert "obstacle_endpoint_mask" in layers
        assert "obstacle_bridge_mask" in layers
        assert "obstacle_dilation_mask" in layers
        assert "obstacle_cleared_by_trajectory_mask" in layers
        assert "terrain_hazard_mask" in layers
        assert "enclosed_unknown_filled_mask" in layers
        assert "obstacle_rejected_low_support_mask" in layers
        assert layers["trajectory_yaw_rad"].shape == (trajectory.shape[0],)
        assert layers["trajectory_yaw_source"].item() == "inferred_xy_tangent"
        assert layers["trajectory_segment_valid"].shape == (
            trajectory.shape[0] - 1,
        )
