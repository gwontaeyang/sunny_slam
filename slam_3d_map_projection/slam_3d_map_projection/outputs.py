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

import json
from pathlib import Path
from typing import Dict

import numpy as np

from .configuration import ProjectionConfig
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
    """Write one internal occupancy array as a Nav2 trinary PGM/YAML pair."""
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
    _write_pgm(image_path, pixels)
    _write_map_yaml(yaml_path, image_path.name, geometry, config)
    return {"map_image": image_path, "map_yaml": yaml_path}


def write_projection_outputs(
    output_dir: Path,
    map_name: str,
    points: np.ndarray,
    trajectory: np.ndarray,
    result: ProjectionResult,
    config: ProjectionConfig,
) -> Dict[str, Path]:
    """Write map-server files, ramp/elevation evidence, and a classified cloud."""
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

    metrics = dict(result.metrics)
    metrics["parameters"] = config.to_dict()
    metrics["outputs"] = {
        "map_image": map_image_path.name,
        "map_yaml": map_yaml_path.name,
        "ramp_mask_image": ramp_image_path.name,
        "ramp_mask_yaml": ramp_yaml_path.name,
        "terrain_layers": layers_path.name,
        "classified_pcd": classified_pcd_path.name,
    }
    metrics_path = output_dir / f"{map_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        "map_image": map_image_path,
        "map_yaml": map_yaml_path,
        "ramp_mask": ramp_image_path,
        "ramp_mask_yaml": ramp_yaml_path,
        "terrain_layers": layers_path,
        "classified_pcd": classified_pcd_path,
        "metrics": metrics_path,
    }
