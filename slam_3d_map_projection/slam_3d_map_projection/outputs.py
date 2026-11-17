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

"""Writers for Nav2 map-server files and projector evidence."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
import struct
from typing import Dict, Optional
import zlib

import numpy as np

from .configuration import ProjectionConfig
from .path_planning import PathPlan, PathPlanningConfig
from .pcd_io import write_xyzrgb_pcd
from .projection import GridClass, GridGeometry, PointClass, ProjectionResult


_POINT_COLORS: Dict[PointClass, np.ndarray] = {
    PointClass.FLOOR: np.asarray([0, 158, 115], dtype=np.uint8),
    PointClass.BODY_OBSTACLE: np.asarray([213, 94, 0], dtype=np.uint8),
    PointClass.HIGH_OVERHANG: np.asarray([0, 114, 178], dtype=np.uint8),
    PointClass.BELOW_FLOOR: np.asarray([120, 120, 120], dtype=np.uint8),
    PointClass.UNREFERENCED: np.asarray([204, 121, 167], dtype=np.uint8),
}


def point_class_colors(classes: np.ndarray) -> np.ndarray:
    """Return stable RGB colors for classified map points."""
    colors = np.zeros((classes.shape[0], 3), dtype=np.uint8)
    for label, color in _POINT_COLORS.items():
        colors[classes == label] = color
    return colors


def _write_pgm(path: Path, image_bottom_up: np.ndarray) -> None:
    image = np.asarray(image_bottom_up, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("PGM image must be two-dimensional")
    # Nav2's map origin is the lower-left cell; PGM rows begin at the top.
    stored = np.flipud(image)
    header = f"P5\n{stored.shape[1]} {stored.shape[0]}\n255\n".encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(stored.tobytes())


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", checksum)
    )


def _write_png(path: Path, image_bottom_up: np.ndarray) -> None:
    """Write an 8-bit grayscale PNG with the same orientation as `_write_pgm`."""
    image = np.asarray(image_bottom_up, dtype=np.uint8)
    if image.ndim != 2:
        raise ValueError("PNG image must be two-dimensional")
    stored = np.flipud(image)
    height, width = stored.shape
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in stored)
    compressed = zlib.compress(scanlines)
    with path.open("wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        stream.write(_png_chunk(b"IHDR", ihdr))
        stream.write(_png_chunk(b"IDAT", compressed))
        stream.write(_png_chunk(b"IEND", b""))


def _map_thresholds(config: ProjectionConfig) -> tuple[float, float]:
    """Choose thresholds that preserve the configured trinary pixel classes."""
    free_probability = (255.0 - config.free_value) / 255.0
    unknown_probability = (255.0 - config.unknown_value) / 255.0
    occupied_probability = (255.0 - config.occupied_value) / 255.0
    free_threshold = 0.5 * (free_probability + unknown_probability)
    occupied_threshold = 0.5 * (unknown_probability + occupied_probability)
    return occupied_threshold, free_threshold


def _write_map_yaml(
    path: Path,
    image_name: str,
    geometry: GridGeometry,
    config: ProjectionConfig,
) -> None:
    occupied_threshold, free_threshold = _map_thresholds(config)
    content = (
        f"image: {image_name}\n"
        "mode: trinary\n"
        f"resolution: {geometry.resolution_m:.9g}\n"
        f"origin: [{geometry.origin_x_m:.9g}, {geometry.origin_y_m:.9g}, 0.0]\n"
        "negate: 0\n"
        f"occupied_thresh: {occupied_threshold:.9g}\n"
        f"free_thresh: {free_threshold:.9g}\n"
    )
    path.write_text(content, encoding="utf-8")


def write_trinary_map(
    output_dir: Path,
    map_name: str,
    occupancy: np.ndarray,
    geometry: GridGeometry,
    config: ProjectionConfig,
) -> Dict[str, Path]:
    """Write one occupancy array as equivalent PGM and PNG Nav2 map pairs."""
    occupancy = np.asarray(occupancy)
    expected_shape = (geometry.height, geometry.width)
    if occupancy.shape != expected_shape:
        raise ValueError(
            f"occupancy shape {occupancy.shape} does not match grid {expected_shape}"
        )
    valid_values = np.isin(
        occupancy,
        [GridClass.UNKNOWN, GridClass.FREE, GridClass.OCCUPIED],
    )
    if not np.all(valid_values):
        raise ValueError("occupancy contains values outside the trinary contract")

    output_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.full(occupancy.shape, config.unknown_value, dtype=np.uint8)
    pixels[occupancy == GridClass.FREE] = config.free_value
    pixels[occupancy == GridClass.OCCUPIED] = config.occupied_value
    image_path = output_dir / f"{map_name}.pgm"
    yaml_path = output_dir / f"{map_name}.yaml"
    png_path = output_dir / f"{map_name}.png"
    png_yaml_path = output_dir / f"{map_name}_png.yaml"
    _write_pgm(image_path, pixels)
    _write_map_yaml(yaml_path, image_path.name, geometry, config)
    _write_png(png_path, pixels)
    _write_map_yaml(png_yaml_path, png_path.name, geometry, config)
    return {
        "map_image": image_path,
        "map_yaml": yaml_path,
        "map_png": png_path,
        "map_png_yaml": png_yaml_path,
    }


def write_projection_outputs(
    output_dir: Path,
    map_name: str,
    points: np.ndarray,
    trajectory: np.ndarray,
    result: ProjectionResult,
    config: ProjectionConfig,
    path_plan: Optional[PathPlan] = None,
    path_planning_config: Optional[PathPlanningConfig] = None,
) -> Dict[str, Path]:
    """Write map-server files, filtered obstacle evidence, and an optional path."""
    if (path_plan is None) != (path_planning_config is None):
        raise ValueError(
            "path_plan and path_planning_config must either both be set or both be None"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    map_paths = write_trinary_map(
        output_dir,
        map_name,
        result.occupancy,
        result.geometry,
        config,
    )
    map_image_path = map_paths["map_image"]
    map_yaml_path = map_paths["map_yaml"]
    map_png_path = map_paths["map_png"]
    map_png_yaml_path = map_paths["map_png_yaml"]

    ramp_pixels = np.full(result.ramp_mask.shape, config.unknown_value, dtype=np.uint8)
    known = result.occupancy != GridClass.UNKNOWN
    ramp_pixels[known] = config.free_value
    ramp_pixels[result.ramp_mask] = config.occupied_value
    ramp_image_path = output_dir / f"{map_name}_ramp_mask.pgm"
    ramp_yaml_path = output_dir / f"{map_name}_ramp_mask.yaml"
    _write_pgm(ramp_image_path, ramp_pixels)
    _write_map_yaml(
        ramp_yaml_path,
        ramp_image_path.name,
        result.geometry,
        config,
    )

    layers_path = output_dir / f"{map_name}_terrain_layers.npz"
    np.savez_compressed(
        layers_path,
        occupancy=result.occupancy,
        ramp_mask=result.ramp_mask,
        floor_elevation_m=result.floor_elevation_m,
        floor_slope_deg=result.floor_slope_deg,
        floor_roughness_m=result.floor_roughness_m,
        floor_counts=result.floor_counts,
        obstacle_counts=result.obstacle_counts,
        below_floor_counts=result.below_floor_counts,
        obstacle_rejected_low_count_mask=result.obstacle_rejected_low_count_mask,
        obstacle_rejected_low_support_mask=result.obstacle_rejected_low_support_mask,
        obstacle_rejected_far_from_free_mask=(
            result.obstacle_rejected_far_from_free_mask
        ),
        obstacle_endpoint_mask=result.obstacle_endpoint_mask,
        obstacle_bridge_mask=result.obstacle_bridge_mask,
        obstacle_dilation_mask=result.obstacle_dilation_mask,
        obstacle_cleared_by_trajectory_mask=(
            result.obstacle_cleared_by_trajectory_mask
        ),
        below_floor_endpoint_mask=result.below_floor_endpoint_mask,
        excessive_slope_mask=result.excessive_slope_mask,
        excessive_roughness_mask=result.excessive_roughness_mask,
        terrain_hazard_mask=result.terrain_hazard_mask,
        terrain_hazard_cleared_by_trajectory_mask=(
            result.terrain_hazard_cleared_by_trajectory_mask
        ),
        enclosed_unknown_filled_mask=result.enclosed_unknown_filled_mask,
        trajectory_mask=result.trajectory_mask,
        trajectory_centerline_mask=result.trajectory_centerline_mask,
        origin_xy_m=np.asarray(
            [result.geometry.origin_x_m, result.geometry.origin_y_m], dtype=np.float64
        ),
        resolution_m=np.asarray(result.geometry.resolution_m, dtype=np.float64),
        trajectory_xyz=np.asarray(trajectory[:, :3], dtype=np.float64),
        trajectory_yaw_rad=np.asarray(result.trajectory_yaw_rad, dtype=np.float64),
        trajectory_yaw_source=np.asarray(result.trajectory_yaw_source),
        trajectory_segment_valid=np.asarray(
            result.trajectory_segment_valid, dtype=bool
        ),
    )

    classified_pcd_path = output_dir / f"{map_name}_classified_rgb.pcd"
    write_xyzrgb_pcd(
        classified_pcd_path,
        points,
        point_class_colors(result.point_classes),
    )

    obstacle_columns, obstacle_rows = result.geometry.world_to_grid(points[:, :2])
    obstacle_in_grid = (
        (obstacle_columns >= 0)
        & (obstacle_columns < result.geometry.width)
        & (obstacle_rows >= 0)
        & (obstacle_rows < result.geometry.height)
    )
    planning_obstacle = np.zeros(points.shape[0], dtype=bool)
    body_obstacle = result.point_classes == PointClass.BODY_OBSTACLE
    candidate = body_obstacle & obstacle_in_grid
    planning_obstacle[candidate] = (
        result.occupancy[
            obstacle_rows[candidate], obstacle_columns[candidate]
        ]
        == GridClass.OCCUPIED
    )
    planning_obstacle_pcd_path = (
        output_dir / f"{map_name}_planning_obstacles_rgb.pcd"
    )
    obstacle_colors = np.tile(
        _POINT_COLORS[PointClass.BODY_OBSTACLE],
        (int(np.count_nonzero(planning_obstacle)), 1),
    )
    write_xyzrgb_pcd(
        planning_obstacle_pcd_path,
        points[planning_obstacle],
        obstacle_colors,
    )

    path_csv_path = None
    path_layers_path = None
    if path_plan is not None:
        path_csv_path = output_dir / f"{map_name}_planned_path.csv"
        with path_csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pose_index", "x_m", "y_m", "yaw_rad"])
            for index, pose in enumerate(path_plan.poses_xy_yaw):
                writer.writerow([index, *pose.tolist()])

        path_layers_path = output_dir / f"{map_name}_path_layers.npz"
        np.savez_compressed(
            path_layers_path,
            path_states=path_plan.states,
            path_poses_xy_yaw=path_plan.poses_xy_yaw,
            path_exact_valid=path_plan.path_exact_valid,
            path_swept_poses_xy_yaw=path_plan.path_swept_poses_xy_yaw,
            path_swept_exact_valid=path_plan.path_swept_exact_valid,
            trajectory_exact_footprint_valid=path_plan.trajectory_exact_valid,
            trajectory_lattice_footprint_valid=path_plan.trajectory_lattice_valid,
        )

    metrics = dict(result.metrics)
    metrics["parameters"] = config.to_dict()
    metrics["planning_obstacle_export"] = {
        "selected_points": int(np.count_nonzero(planning_obstacle)),
        "source_class": "body_obstacle",
        "requires_final_occupied_cell": True,
        "floor_points_excluded": int(
            np.count_nonzero(result.point_classes == PointClass.FLOOR)
        ),
        "high_overhang_points_excluded": int(
            np.count_nonzero(result.point_classes == PointClass.HIGH_OVERHANG)
        ),
        "dynamic_object_contract": (
            "input PCD must already be cleaned offline by slam_3d_map_cleanup "
            "(map_static.pcd) or equivalent; core global_cloud is temporally "
            "unfiltered; cleanup provenance is not verified by projection"
        ),
    }
    metrics["outputs"] = {
        "map_image": map_image_path.name,
        "map_yaml": map_yaml_path.name,
        "map_png": map_png_path.name,
        "map_png_yaml": map_png_yaml_path.name,
        "ramp_mask_image": ramp_image_path.name,
        "ramp_mask_yaml": ramp_yaml_path.name,
        "terrain_layers": layers_path.name,
        "classified_pcd": classified_pcd_path.name,
        "planning_obstacles_pcd": planning_obstacle_pcd_path.name,
    }
    if path_plan is not None:
        metrics["path_planning"] = path_plan.metrics
        metrics["path_planning_parameters"] = asdict(path_planning_config)
        metrics["outputs"].update(
            {
                "planned_path": path_csv_path.name,
                "path_layers": path_layers_path.name,
            }
        )
    metrics_path = output_dir / f"{map_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    outputs = {
        "map_image": map_image_path,
        "map_yaml": map_yaml_path,
        "map_png": map_png_path,
        "map_png_yaml": map_png_yaml_path,
        "ramp_mask": ramp_image_path,
        "ramp_mask_yaml": ramp_yaml_path,
        "terrain_layers": layers_path,
        "classified_pcd": classified_pcd_path,
        "planning_obstacles_pcd": planning_obstacle_pcd_path,
        "metrics": metrics_path,
    }
    if path_plan is not None:
        outputs.update(
            {
                "planned_path": path_csv_path,
                "path_layers": path_layers_path,
            }
        )
    return outputs
