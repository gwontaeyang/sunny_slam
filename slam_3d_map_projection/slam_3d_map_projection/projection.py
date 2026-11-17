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

"""Geometry-aware projection from an optimized 3D map to a 2D navigation grid."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .configuration import ProjectionConfig


class PointClass(IntEnum):
    """Point labels retained in the classified review cloud."""

    FLOOR = 0
    BODY_OBSTACLE = 1
    HIGH_OVERHANG = 2
    BELOW_FLOOR = 3
    UNREFERENCED = 4


class GridClass(IntEnum):
    """Internal occupancy values before conversion to map-server pixels."""

    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 100


@dataclass(frozen=True)
class GridGeometry:
    """Metric geometry for arrays whose row index grows with map y."""

    origin_x_m: float
    origin_y_m: float
    resolution_m: float
    width: int
    height: int

    @property
    def max_x_m(self) -> float:
        return self.origin_x_m + self.width * self.resolution_m

    @property
    def max_y_m(self) -> float:
        return self.origin_y_m + self.height * self.resolution_m

    def world_to_grid(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert XY coordinates to integer columns and rows."""
        xy = np.asarray(xy, dtype=np.float64)
        columns = np.floor((xy[:, 0] - self.origin_x_m) / self.resolution_m)
        rows = np.floor((xy[:, 1] - self.origin_y_m) / self.resolution_m)
        return columns.astype(np.int64), rows.astype(np.int64)


@dataclass
class ProjectionResult:
    """All outputs and intermediate evidence needed for review."""

    geometry: GridGeometry
    occupancy: np.ndarray
    ramp_mask: np.ndarray
    floor_elevation_m: np.ndarray
    floor_slope_deg: np.ndarray
    floor_roughness_m: np.ndarray
    floor_counts: np.ndarray
    obstacle_counts: np.ndarray
    below_floor_counts: np.ndarray
    obstacle_rejected_low_count_mask: np.ndarray
    obstacle_rejected_low_support_mask: np.ndarray
    obstacle_rejected_far_from_free_mask: np.ndarray
    obstacle_endpoint_mask: np.ndarray
    obstacle_bridge_mask: np.ndarray
    obstacle_dilation_mask: np.ndarray
    obstacle_cleared_by_trajectory_mask: np.ndarray
    below_floor_endpoint_mask: np.ndarray
    excessive_slope_mask: np.ndarray
    excessive_roughness_mask: np.ndarray
    terrain_hazard_mask: np.ndarray
    terrain_hazard_cleared_by_trajectory_mask: np.ndarray
    enclosed_unknown_filled_mask: np.ndarray
    trajectory_mask: np.ndarray
    trajectory_centerline_mask: np.ndarray
    trajectory_yaw_rad: np.ndarray
    trajectory_segment_valid: np.ndarray
    trajectory_yaw_source: str
    point_classes: np.ndarray
    relative_point_height_m: np.ndarray
    metrics: Dict[str, object]


def _disk(radius_m: float, resolution_m: float) -> np.ndarray:
    radius_cells = int(np.ceil(radius_m / resolution_m))
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    rows, columns = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]
    radius_squared = (radius_m / resolution_m) ** 2 + 1.0e-9
    return rows * rows + columns * columns <= radius_squared


def _initial_geometry(
    points: np.ndarray,
    trajectory: np.ndarray,
    config: ProjectionConfig,
) -> GridGeometry:
    low = np.quantile(points[:, :2], config.bounds_quantile_low, axis=0)
    high = np.quantile(points[:, :2], config.bounds_quantile_high, axis=0)
    low = np.minimum(low, np.min(trajectory[:, :2], axis=0)) - config.initial_padding_m
    high = np.maximum(high, np.max(trajectory[:, :2], axis=0)) + config.initial_padding_m
    resolution = config.resolution_m
    origin = np.floor(low / resolution) * resolution
    upper = np.ceil(high / resolution) * resolution
    shape = np.ceil((upper - origin) / resolution).astype(np.int64)
    width, height = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        raise ValueError("computed grid bounds are empty")
    if width * height > config.max_grid_cells:
        raise ValueError(
            f"grid has {width * height} cells, above max_grid_cells="
            f"{config.max_grid_cells}"
        )
    return GridGeometry(origin[0], origin[1], resolution, width, height)


def _valid_grid_indices(
    geometry: GridGeometry, xy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns, rows = geometry.world_to_grid(xy)
    valid = (
        (columns >= 0)
        & (columns < geometry.width)
        & (rows >= 0)
        & (rows < geometry.height)
    )
    return columns, rows, valid


def _count_grid(
    geometry: GridGeometry,
    columns: np.ndarray,
    rows: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    indices = rows[selected] * geometry.width + columns[selected]
    counts = np.bincount(indices, minlength=geometry.width * geometry.height)
    return counts.reshape(geometry.height, geometry.width).astype(np.uint32)


def _weighted_grid(
    geometry: GridGeometry,
    columns: np.ndarray,
    rows: np.ndarray,
    selected: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    indices = rows[selected] * geometry.width + columns[selected]
    sums = np.bincount(
        indices,
        weights=values[selected],
        minlength=geometry.width * geometry.height,
    )
    return sums.reshape(geometry.height, geometry.width)


def _rasterize_trajectory(
    geometry: GridGeometry,
    trajectory_xy: np.ndarray,
    segment_valid: np.ndarray,
    radius_m: float,
) -> np.ndarray:
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    columns, rows, valid = _valid_grid_indices(geometry, trajectory_xy)
    mask[rows[valid], columns[valid]] = True
    sample_spacing = max(geometry.resolution_m * 0.5, 1.0e-3)
    for start, end, interpolate in zip(
        trajectory_xy[:-1], trajectory_xy[1:], segment_valid
    ):
        if not interpolate:
            continue
        distance = float(np.linalg.norm(end - start))
        sample_count = max(2, int(np.ceil(distance / sample_spacing)) + 1)
        samples = np.linspace(start, end, sample_count)
        columns, rows, valid = _valid_grid_indices(geometry, samples)
        mask[rows[valid], columns[valid]] = True
    return ndimage.binary_dilation(mask, structure=_disk(radius_m, geometry.resolution_m))


def _trajectory_yaw_rad(trajectory: np.ndarray) -> np.ndarray:
    """Return optimized yaw, or infer a continuous tangent for XYZ-only input."""
    if trajectory.shape[1] >= 4 and np.isfinite(trajectory[:, 3]).all():
        return trajectory[:, 3].copy()

    gradient_x = np.gradient(trajectory[:, 0])
    gradient_y = np.gradient(trajectory[:, 1])
    moving = np.hypot(gradient_x, gradient_y) > 1.0e-6
    if not np.any(moving):
        return np.zeros(trajectory.shape[0], dtype=np.float64)
    moving_indices = np.flatnonzero(moving)
    moving_yaw = np.unwrap(np.arctan2(gradient_y[moving], gradient_x[moving]))
    return np.interp(
        np.arange(trajectory.shape[0]),
        moving_indices,
        moving_yaw,
    )


def _trajectory_segment_validity(
    trajectory: np.ndarray,
    yaw_rad: np.ndarray,
    config: ProjectionConfig,
    stamp_ns: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Reject pose pairs that are too discontinuous to represent swept free space."""
    delta_xy = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    delta_z = np.abs(np.diff(trajectory[:, 2]))
    delta_yaw = np.abs(
        np.arctan2(
            np.sin(np.diff(yaw_rad)),
            np.cos(np.diff(yaw_rad)),
        )
    )
    xy_valid = delta_xy <= config.trajectory_max_segment_xy_m
    z_valid = delta_z <= config.trajectory_max_segment_z_m
    yaw_valid = np.degrees(delta_yaw) <= config.trajectory_max_segment_yaw_deg
    time_valid = np.ones(delta_xy.shape, dtype=bool)
    if stamp_ns is not None:
        stamps = np.asarray(stamp_ns)
        if stamps.ndim != 1 or stamps.shape[0] != trajectory.shape[0]:
            raise ValueError("trajectory_stamp_ns must contain one value per pose")
        if not np.issubdtype(stamps.dtype, np.integer):
            raise ValueError("trajectory_stamp_ns must contain integer nanoseconds")
        delta_time_s = np.diff(stamps.astype(np.int64)) * 1.0e-9
        time_valid = (
            (delta_time_s > 0.0)
            & (delta_time_s <= config.trajectory_max_segment_dt_s)
        )
    valid = xy_valid & z_valid & yaw_valid & time_valid
    reasons = {
        "xy_discontinuity_segments": int(np.count_nonzero(~xy_valid)),
        "z_discontinuity_segments": int(np.count_nonzero(~z_valid)),
        "yaw_discontinuity_segments": int(np.count_nonzero(~yaw_valid)),
        "time_discontinuity_segments": int(np.count_nonzero(~time_valid)),
        "invalid_union_segments": int(np.count_nonzero(~valid)),
    }
    return valid, reasons


def _single_level_trajectory_z_span(
    trajectory: np.ndarray,
    config: ProjectionConfig,
) -> float:
    """Return the largest trajectory z span aliased into one XY bin."""
    xy_bin = np.floor(trajectory[:, :2] / config.single_level_xy_bin_m).astype(
        np.int64
    )
    _, inverse = np.unique(xy_bin, axis=0, return_inverse=True)
    minimum = np.full(int(np.max(inverse)) + 1, np.inf, dtype=np.float64)
    maximum = np.full(int(np.max(inverse)) + 1, -np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, trajectory[:, 2])
    np.maximum.at(maximum, inverse, trajectory[:, 2])
    return float(np.max(maximum - minimum))


def oriented_footprint_indices(
    geometry: GridGeometry,
    center_xy: np.ndarray,
    yaw_rad: float,
    half_length_m: float,
    half_width_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cells whose centers lie inside one oriented rectangle."""
    cosine = float(np.cos(yaw_rad))
    sine = float(np.sin(yaw_rad))
    extent_x = abs(cosine) * half_length_m + abs(sine) * half_width_m
    extent_y = abs(sine) * half_length_m + abs(cosine) * half_width_m
    column_min = max(
        0,
        int(np.floor((center_xy[0] - extent_x - geometry.origin_x_m) /
                     geometry.resolution_m)),
    )
    column_max = min(
        geometry.width - 1,
        int(np.floor((center_xy[0] + extent_x - geometry.origin_x_m) /
                     geometry.resolution_m)),
    )
    row_min = max(
        0,
        int(np.floor((center_xy[1] - extent_y - geometry.origin_y_m) /
                     geometry.resolution_m)),
    )
    row_max = min(
        geometry.height - 1,
        int(np.floor((center_xy[1] + extent_y - geometry.origin_y_m) /
                     geometry.resolution_m)),
    )
    if column_min > column_max or row_min > row_max:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)

    columns = np.arange(column_min, column_max + 1, dtype=np.int64)
    rows = np.arange(row_min, row_max + 1, dtype=np.int64)
    column_grid, row_grid = np.meshgrid(columns, rows)
    world_x = (
        geometry.origin_x_m + (column_grid.astype(np.float64) + 0.5) *
        geometry.resolution_m
    )
    world_y = (
        geometry.origin_y_m + (row_grid.astype(np.float64) + 0.5) *
        geometry.resolution_m
    )
    delta_x = world_x - center_xy[0]
    delta_y = world_y - center_xy[1]
    local_x = cosine * delta_x + sine * delta_y
    local_y = -sine * delta_x + cosine * delta_y
    inside = (
        (np.abs(local_x) <= half_length_m + 1.0e-9)
        & (np.abs(local_y) <= half_width_m + 1.0e-9)
    )
    return row_grid[inside], column_grid[inside]


def _rasterize_swept_footprint(
    geometry: GridGeometry,
    trajectory: np.ndarray,
    yaw_rad: np.ndarray,
    segment_valid: np.ndarray,
    config: ProjectionConfig,
) -> np.ndarray:
    """Rasterize the full oriented footprint swept between optimized poses."""
    mask = np.zeros((geometry.height, geometry.width), dtype=bool)
    half_length = (
        0.5 * config.robot_footprint_length_m
        + config.trajectory_footprint_margin_m
    )
    half_width = (
        0.5 * config.robot_footprint_width_m
        + config.trajectory_footprint_margin_m
    )
    corner_radius = float(np.hypot(half_length, half_width))
    sample_spacing = max(geometry.resolution_m * 0.5, 1.0e-3)
    for center_xy, pose_yaw in zip(trajectory[:, :2], yaw_rad):
        rows, columns = oriented_footprint_indices(
            geometry,
            center_xy,
            pose_yaw,
            half_length,
            half_width,
        )
        mask[rows, columns] = True
    for pose_index in range(trajectory.shape[0] - 1):
        if not segment_valid[pose_index]:
            continue
        start_xy = trajectory[pose_index, :2]
        end_xy = trajectory[pose_index + 1, :2]
        yaw_start = yaw_rad[pose_index]
        yaw_delta = float(
            np.arctan2(
                np.sin(yaw_rad[pose_index + 1] - yaw_start),
                np.cos(yaw_rad[pose_index + 1] - yaw_start),
            )
        )
        translation_m = float(np.linalg.norm(end_xy - start_xy))
        rotation_arc_m = abs(yaw_delta) * corner_radius
        sample_count = max(
            2,
            int(np.ceil(max(translation_m, rotation_arc_m) / sample_spacing)) + 1,
        )
        for interpolation in np.linspace(0.0, 1.0, sample_count):
            center_xy = start_xy + interpolation * (end_xy - start_xy)
            rows, columns = oriented_footprint_indices(
                geometry,
                center_xy,
                yaw_start + interpolation * yaw_delta,
                half_length,
                half_width,
            )
            mask[rows, columns] = True
    return mask


def _footprint_collision_flags(
    geometry: GridGeometry,
    trajectory: np.ndarray,
    yaw_rad: np.ndarray,
    occupied: np.ndarray,
    config: ProjectionConfig,
) -> np.ndarray:
    """Check collision evidence independently at every recorded pose."""
    half_length = (
        0.5 * config.robot_footprint_length_m
        + config.trajectory_footprint_margin_m
    )
    half_width = (
        0.5 * config.robot_footprint_width_m
        + config.trajectory_footprint_margin_m
    )
    collision = np.zeros(trajectory.shape[0], dtype=bool)
    for pose_index, (center_xy, pose_yaw) in enumerate(
        zip(trajectory[:, :2], yaw_rad)
    ):
        rows, columns = oriented_footprint_indices(
            geometry,
            center_xy,
            pose_yaw,
            half_length,
            half_width,
        )
        collision[pose_index] = bool(np.any(occupied[rows, columns]))
    return collision


def _trajectory_ramp_region_contiguous(
    geometry: GridGeometry,
    trajectory: np.ndarray,
    config: ProjectionConfig,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Find coherent z changes along the driven XY path and rasterize their region."""
    segment_distance = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
    cumulative_distance = np.concatenate([[0.0], np.cumsum(segment_distance)])
    unique = np.concatenate([[True], np.diff(cumulative_distance) > 1.0e-6])
    cumulative_distance = cumulative_distance[unique]
    unique_trajectory = trajectory[unique]
    if cumulative_distance.size < 2 or cumulative_distance[-1] < config.ramp_min_length_m:
        return np.zeros((geometry.height, geometry.width), dtype=bool), []

    spacing = max(geometry.resolution_m, 0.05)
    sample_distance = np.arange(0.0, cumulative_distance[-1] + spacing * 0.5, spacing)
    sample_x = np.interp(sample_distance, cumulative_distance, unique_trajectory[:, 0])
    sample_y = np.interp(sample_distance, cumulative_distance, unique_trajectory[:, 1])
    sample_z = np.interp(sample_distance, cumulative_distance, unique_trajectory[:, 2])
    sigma = max(config.ramp_trajectory_smoothing_m / spacing, 0.5)
    smooth_z = ndimage.gaussian_filter1d(sample_z, sigma=sigma, mode="nearest")
    slope_deg = np.degrees(np.arctan(np.abs(np.gradient(smooth_z, spacing))))
    candidate = (
        (slope_deg >= config.ramp_min_slope_deg)
        & (slope_deg <= config.ramp_max_slope_deg)
    )
    labels, run_count = ndimage.label(candidate)
    accepted = np.zeros_like(candidate)
    accepted_runs: List[Dict[str, float]] = []
    for label_index in range(1, run_count + 1):
        indices = np.flatnonzero(labels == label_index)
        if indices.size < 2:
            continue
        length_m = sample_distance[indices[-1]] - sample_distance[indices[0]]
        height_change_m = abs(smooth_z[indices[-1]] - smooth_z[indices[0]])
        if (
            length_m >= config.ramp_min_length_m
            and height_change_m >= config.ramp_min_height_change_m
        ):
            accepted[indices] = True
            middle = indices[indices.size // 2]
            accepted_runs.append(
                {
                    "start_distance_m": float(sample_distance[indices[0]]),
                    "end_distance_m": float(sample_distance[indices[-1]]),
                    "length_m": float(length_m),
                    "height_change_m": float(height_change_m),
                    "median_slope_deg": float(np.median(slope_deg[indices])),
                    "max_slope_deg": float(np.max(slope_deg[indices])),
                    "center_x_m": float(sample_x[middle]),
                    "center_y_m": float(sample_y[middle]),
                    "center_z_m": float(smooth_z[middle]),
                }
            )

    region = np.zeros((geometry.height, geometry.width), dtype=bool)
    if not np.any(accepted):
        return region, accepted_runs
    samples_xy = np.column_stack([sample_x[accepted], sample_y[accepted]])
    columns, rows, valid = _valid_grid_indices(geometry, samples_xy)
    region[rows[valid], columns[valid]] = True
    region = ndimage.binary_dilation(
        region,
        structure=_disk(config.ramp_region_radius_m, config.resolution_m),
    )
    return region, accepted_runs


def _trajectory_ramp_region(
    geometry: GridGeometry,
    trajectory: np.ndarray,
    segment_valid: np.ndarray,
    config: ProjectionConfig,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Detect ramps independently in each contiguous trajectory section."""
    combined = np.zeros((geometry.height, geometry.width), dtype=bool)
    accepted_runs: List[Dict[str, float]] = []
    section_start = 0
    section_ends = list(np.flatnonzero(~segment_valid) + 1) + [trajectory.shape[0]]
    for section_end in section_ends:
        section = trajectory[section_start:section_end]
        if section.shape[0] >= 2:
            region, runs = _trajectory_ramp_region_contiguous(
                geometry, section, config
            )
            combined |= region
            for run in runs:
                run["trajectory_section_start_pose"] = int(section_start)
                run["trajectory_section_end_pose"] = int(section_end - 1)
            accepted_runs.extend(runs)
        section_start = section_end
    return combined, accepted_runs


def _estimate_floor_normals(
    points: np.ndarray,
    candidate: np.ndarray,
    config: ProjectionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate floor candidates and return tilt and plane-normal roughness."""
    candidate_indices = np.flatnonzero(candidate)
    valid = np.zeros(points.shape[0], dtype=bool)
    tilt_deg = np.full(points.shape[0], np.nan, dtype=np.float64)
    roughness_m = np.full(points.shape[0], np.nan, dtype=np.float64)
    if candidate_indices.size < config.floor_normal_neighbors:
        return valid, tilt_deg, roughness_m

    candidate_points = points[candidate_indices]
    tree = cKDTree(candidate_points)
    neighbor_count = min(config.floor_normal_neighbors, candidate_points.shape[0])
    distances, neighbors = tree.query(candidate_points, k=neighbor_count)
    neighborhoods = candidate_points[neighbors]
    centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbor_count
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    denominator = np.maximum(np.sum(eigenvalues, axis=1), 1.0e-12)
    planarity_ratio = eigenvalues[:, 0] / denominator
    surface_spread = eigenvalues[:, 1] / np.maximum(eigenvalues[:, 2], 1.0e-12)
    normals = eigenvectors[:, :, 0]
    local_tilt = np.degrees(
        np.arccos(np.clip(np.abs(normals[:, 2]), 0.0, 1.0))
    )
    local_valid = (
        (distances[:, -1] <= config.floor_normal_max_neighbor_distance_m)
        & (planarity_ratio <= config.floor_planarity_ratio_max)
        & (surface_spread >= config.floor_surface_spread_min)
        & (local_tilt <= config.floor_normal_max_tilt_deg)
    )
    valid[candidate_indices[local_valid]] = True
    tilt_deg[candidate_indices[local_valid]] = local_tilt[local_valid]
    roughness_m[candidate_indices[local_valid]] = np.sqrt(
        np.maximum(eigenvalues[local_valid, 0], 0.0)
    )
    return valid, tilt_deg, roughness_m


def _smooth_floor_layers(
    floor_counts: np.ndarray,
    floor_sum: np.ndarray,
    floor_tilt_sum: np.ndarray,
    floor_roughness_sum: np.ndarray,
    config: ProjectionConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed = floor_counts >= config.floor_min_points_per_cell
    if not np.any(seed):
        raise ValueError("no floor cells survived the configured floor band")

    count = floor_counts.astype(np.float64)
    sigma_cells = max(config.floor_smoothing_sigma_m / config.resolution_m, 0.5)
    smooth_count = ndimage.gaussian_filter(count, sigma=sigma_cells, mode="constant")
    smooth_sum = ndimage.gaussian_filter(floor_sum, sigma=sigma_cells, mode="constant")
    smooth_tilt_sum = ndimage.gaussian_filter(
        floor_tilt_sum, sigma=sigma_cells, mode="constant"
    )
    smooth_roughness_sum = ndimage.gaussian_filter(
        floor_roughness_sum, sigma=sigma_cells, mode="constant"
    )
    denominator = np.maximum(smooth_count, 1.0e-9)
    elevation = smooth_sum / denominator
    slope_deg = smooth_tilt_sum / denominator
    roughness = smooth_roughness_sum / denominator

    distance_cells, nearest_indices = ndimage.distance_transform_edt(
        ~seed, return_indices=True
    )
    nearest_mean = np.zeros_like(elevation)
    seed_mean = np.divide(
        floor_sum,
        np.maximum(count, 1.0),
        out=np.zeros_like(floor_sum),
        where=count > 0,
    )
    nearest_mean[:] = seed_mean[nearest_indices[0], nearest_indices[1]]
    weak = smooth_count < 1.0e-4
    elevation[weak] = nearest_mean[weak]

    support = distance_cells * config.resolution_m <= config.floor_support_radius_m
    elevation[~support] = np.nan
    roughness[~support] = np.nan
    slope_deg[~support] = np.nan
    return support, elevation, slope_deg, roughness


def _retain_trajectory_connected_free(
    candidate: np.ndarray, trajectory_mask: np.ndarray
) -> Tuple[np.ndarray, int]:
    labels, component_count = ndimage.label(candidate, structure=np.ones((3, 3)))
    touched = np.unique(labels[trajectory_mask & candidate])
    touched = touched[touched != 0]
    if touched.size == 0:
        return trajectory_mask.copy(), int(component_count)
    return np.isin(labels, touched), int(component_count)


def _remove_small_ramp_regions(
    ramp: np.ndarray, min_area_m2: float, resolution_m: float
) -> np.ndarray:
    if min_area_m2 <= 0.0 or not np.any(ramp):
        return ramp
    labels, count = ndimage.label(ramp, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel()) * resolution_m * resolution_m
    keep = np.flatnonzero(areas >= min_area_m2)
    keep = keep[keep != 0]
    if count == 0 or keep.size == 0:
        return np.zeros_like(ramp)
    return np.isin(labels, keep)


def _crop_result_arrays(
    geometry: GridGeometry,
    arrays: Dict[str, np.ndarray],
    known_mask: np.ndarray,
    padding_m: float,
) -> Tuple[GridGeometry, Dict[str, np.ndarray]]:
    rows, columns = np.nonzero(known_mask)
    if rows.size == 0:
        raise ValueError("projection produced no known grid cells")
    padding_cells = int(np.ceil(padding_m / geometry.resolution_m))
    row_min = max(0, int(rows.min()) - padding_cells)
    row_max = min(geometry.height, int(rows.max()) + padding_cells + 1)
    column_min = max(0, int(columns.min()) - padding_cells)
    column_max = min(geometry.width, int(columns.max()) + padding_cells + 1)
    cropped = {
        name: values[row_min:row_max, column_min:column_max]
        for name, values in arrays.items()
    }
    cropped_geometry = GridGeometry(
        origin_x_m=geometry.origin_x_m + column_min * geometry.resolution_m,
        origin_y_m=geometry.origin_y_m + row_min * geometry.resolution_m,
        resolution_m=geometry.resolution_m,
        width=column_max - column_min,
        height=row_max - row_min,
    )
    return cropped_geometry, cropped


def project_map(
    points: np.ndarray,
    trajectory: np.ndarray,
    config: ProjectionConfig,
    trajectory_stamp_ns: Optional[np.ndarray] = None,
    trajectory_yaw_source: Optional[str] = None,
) -> ProjectionResult:
    """Classify a 3D map and project conservative occupancy and ramp layers."""
    points = np.asarray(points, dtype=np.float64)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3) and cannot be empty")
    if (
        trajectory.ndim != 2
        or trajectory.shape[1] not in (3, 4)
        or trajectory.shape[0] < 2
    ):
        raise ValueError("trajectory must have shape (M, 3) or (M, 4), M >= 2")
    if not np.isfinite(points).all() or not np.isfinite(trajectory).all():
        raise ValueError("points and trajectory must be finite")
    trajectory_yaw = _trajectory_yaw_rad(trajectory)
    yaw_source = trajectory_yaw_source or (
        "input_orientation" if trajectory.shape[1] >= 4 else "inferred_xy_tangent"
    )
    trajectory_segment_valid, segment_rejections = _trajectory_segment_validity(
        trajectory,
        trajectory_yaw,
        config,
        trajectory_stamp_ns,
    )
    same_xy_z_span_m = _single_level_trajectory_z_span(trajectory, config)
    if same_xy_z_span_m > config.single_level_max_z_span_m:
        raise ValueError(
            "trajectory contains multiple z levels at the same XY location; "
            "a single 2D occupancy grid cannot represent this map safely "
            f"(observed span {same_xy_z_span_m:.3f} m)"
        )

    trajectory_tree = cKDTree(trajectory[:, :2])
    reference_distance, reference_index = trajectory_tree.query(points[:, :2], k=1)
    bootstrap_floor_z = trajectory[reference_index, 2]
    bootstrap_relative_height = points[:, 2] - bootstrap_floor_z
    bootstrap_referenced = reference_distance <= config.max_floor_reference_distance_m
    if not np.any(bootstrap_referenced):
        raise ValueError(
            "no PCD points are within max_floor_reference_distance_m of the "
            "optimized trajectory; verify that both inputs use the same map frame"
        )

    floor_band_candidate = (
        bootstrap_referenced
        & (bootstrap_relative_height >= config.floor_min_offset_m)
        & (bootstrap_relative_height <= config.floor_max_offset_m)
    )
    (
        floor_normal_valid,
        floor_tilt_deg,
        floor_plane_roughness_m,
    ) = _estimate_floor_normals(
        points,
        floor_band_candidate,
        config,
    )
    floor = floor_band_candidate & floor_normal_valid

    geometry = _initial_geometry(points[bootstrap_referenced], trajectory, config)
    columns, rows, in_grid = _valid_grid_indices(geometry, points[:, :2])
    floor_in_grid = floor & in_grid
    floor_counts = _count_grid(geometry, columns, rows, floor_in_grid)
    floor_sum = _weighted_grid(
        geometry, columns, rows, floor_in_grid, points[:, 2]
    )
    floor_tilt_sum = _weighted_grid(
        geometry, columns, rows, floor_in_grid, np.nan_to_num(floor_tilt_deg)
    )
    floor_roughness_sum = _weighted_grid(
        geometry,
        columns,
        rows,
        floor_in_grid,
        np.nan_to_num(floor_plane_roughness_m),
    )
    floor_support, elevation, slope_deg, roughness = _smooth_floor_layers(
        floor_counts,
        floor_sum,
        floor_tilt_sum,
        floor_roughness_sum,
        config,
    )

    local_floor_referenced = np.zeros(points.shape[0], dtype=bool)
    local_floor_referenced[in_grid] = floor_support[rows[in_grid], columns[in_grid]]
    local_floor_z = np.full(points.shape[0], np.nan, dtype=np.float64)
    local_floor_z[local_floor_referenced] = elevation[
        rows[local_floor_referenced],
        columns[local_floor_referenced],
    ]
    relative_height = points[:, 2] - local_floor_z
    obstacle = (
        local_floor_referenced
        & ~floor
        & (relative_height >= config.obstacle_min_height_m)
        & (relative_height <= config.obstacle_max_height_m)
    )
    high_overhang = (
        local_floor_referenced
        & ~floor
        & (relative_height > config.obstacle_max_height_m)
    )
    below_floor = (
        local_floor_referenced
        & ~floor
        & (relative_height < config.floor_min_offset_m)
    )
    point_classes = np.full(points.shape[0], PointClass.UNREFERENCED, dtype=np.uint8)
    point_classes[floor] = PointClass.FLOOR
    point_classes[obstacle] = PointClass.BODY_OBSTACLE
    point_classes[high_overhang] = PointClass.HIGH_OVERHANG
    point_classes[below_floor] = PointClass.BELOW_FLOOR
    obstacle_in_grid = obstacle & in_grid
    obstacle_counts = _count_grid(geometry, columns, rows, obstacle_in_grid)
    below_floor_in_grid = below_floor & in_grid
    below_floor_counts = _count_grid(
        geometry, columns, rows, below_floor_in_grid
    )

    trajectory_free = _rasterize_swept_footprint(
        geometry,
        trajectory,
        trajectory_yaw,
        trajectory_segment_valid,
        config,
    )
    trajectory_centerline = _rasterize_trajectory(
        geometry, trajectory[:, :2], trajectory_segment_valid, 0.0
    )
    trajectory_clear = trajectory_free
    trajectory_ramp_region, ramp_runs = _trajectory_ramp_region(
        geometry, trajectory, trajectory_segment_valid, config
    )
    below_floor_count_candidate = (
        below_floor_counts >= config.below_floor_min_points_per_cell
    )
    below_floor_support = ndimage.convolve(
        below_floor_counts,
        _disk(
            config.below_floor_support_radius_m,
            config.resolution_m,
        ).astype(np.uint32),
        mode="constant",
    )
    below_floor_endpoint = (
        below_floor_count_candidate
        & (below_floor_support >= config.below_floor_min_support_points)
    )
    excessive_slope = (
        floor_support
        & np.isfinite(slope_deg)
        & (slope_deg > config.max_traversable_floor_slope_deg)
    )
    excessive_roughness = (
        floor_support
        & np.isfinite(roughness)
        & (roughness > config.max_traversable_floor_roughness_m)
    )
    terrain_hazard = below_floor_endpoint | excessive_slope | excessive_roughness
    terrain_hazard_cleared_by_trajectory = terrain_hazard & trajectory_clear
    untraversed_terrain_hazard = terrain_hazard & ~trajectory_clear
    free_candidate, floor_component_count = _retain_trajectory_connected_free(
        (floor_support & ~untraversed_terrain_hazard) | trajectory_free,
        trajectory_free,
    )

    obstacle_count_candidate = (
        obstacle_counts >= config.obstacle_min_points_per_cell
    )
    obstacle_rejected_low_count = (
        (obstacle_counts > 0) & ~obstacle_count_candidate
    )
    obstacle_support = ndimage.convolve(
        obstacle_counts,
        _disk(config.obstacle_support_radius_m, config.resolution_m).astype(np.uint32),
        mode="constant",
    )
    obstacle_support_candidate = (
        obstacle_count_candidate
        & (obstacle_support >= config.obstacle_min_support_points)
    )
    obstacle_rejected_low_support = (
        obstacle_count_candidate & ~obstacle_support_candidate
    )
    distance_to_free = ndimage.distance_transform_edt(~free_candidate)
    obstacle_close_to_free = (
        distance_to_free * config.resolution_m
        <= config.obstacle_keep_distance_to_free_m
    )
    obstacle_rejected_far_from_free = (
        obstacle_support_candidate & ~obstacle_close_to_free
    )
    obstacle_endpoint = obstacle_support_candidate & obstacle_close_to_free
    obstacle_closed = obstacle_endpoint.copy()
    if config.obstacle_closing_radius_m > 0.0:
        # Closing is gap reconstruction, not sensor evidence. Preserve every
        # supported endpoint even where binary erosion would otherwise remove it.
        obstacle_closed |= ndimage.binary_closing(
            obstacle_endpoint,
            structure=_disk(
                config.obstacle_closing_radius_m,
                config.resolution_m,
            ),
        )
    obstacle_bridge = obstacle_closed & ~obstacle_endpoint
    occupied_before_clear = ndimage.binary_dilation(
        obstacle_closed,
        structure=_disk(config.obstacle_dilation_radius_m, config.resolution_m),
    )
    obstacle_dilation = occupied_before_clear & ~obstacle_closed
    footprint_collision_before_clear = _footprint_collision_flags(
        geometry,
        trajectory,
        trajectory_yaw,
        occupied_before_clear,
        config,
    )
    obstacle_cleared_by_trajectory = occupied_before_clear & trajectory_clear
    occupied = occupied_before_clear & ~trajectory_clear
    footprint_collision_after_clear = _footprint_collision_flags(
        geometry,
        trajectory,
        trajectory_yaw,
        occupied,
        config,
    )
    free = free_candidate & ~occupied
    enclosed_unknown_filled = np.zeros_like(free)
    if config.fill_enclosed_unknown_as_free:
        known_floor_or_wall = free | occupied
        enclosed_unknown = (
            ndimage.binary_fill_holes(known_floor_or_wall)
            & ~known_floor_or_wall
        )
        enclosed_unknown_filled = (
            enclosed_unknown & ~untraversed_terrain_hazard
        )
        free |= enclosed_unknown_filled

    ramp_support = ndimage.binary_erosion(
        floor_support,
        structure=_disk(config.ramp_support_margin_m, config.resolution_m),
    )
    ramp = (
        free
        & ramp_support
        & trajectory_ramp_region
        & np.isfinite(slope_deg)
        & (slope_deg >= config.ramp_min_slope_deg)
        & (slope_deg <= config.ramp_max_slope_deg)
    )
    ramp = _remove_small_ramp_regions(
        ramp, config.ramp_min_area_m2, config.resolution_m
    )

    occupancy = np.full(free.shape, GridClass.UNKNOWN, dtype=np.int8)
    occupancy[free] = GridClass.FREE
    occupancy[occupied] = GridClass.OCCUPIED
    known = free | occupied | trajectory_free | terrain_hazard
    arrays = {
        "occupancy": occupancy,
        "ramp_mask": ramp,
        "floor_elevation_m": elevation,
        "floor_slope_deg": slope_deg,
        "floor_roughness_m": roughness,
        "floor_counts": floor_counts,
        "obstacle_counts": obstacle_counts,
        "below_floor_counts": below_floor_counts,
        "obstacle_rejected_low_count_mask": obstacle_rejected_low_count,
        "obstacle_rejected_low_support_mask": obstacle_rejected_low_support,
        "obstacle_rejected_far_from_free_mask": obstacle_rejected_far_from_free,
        "obstacle_endpoint_mask": obstacle_endpoint,
        "obstacle_bridge_mask": obstacle_bridge,
        "obstacle_dilation_mask": obstacle_dilation,
        "obstacle_cleared_by_trajectory_mask": obstacle_cleared_by_trajectory,
        "below_floor_endpoint_mask": below_floor_endpoint,
        "excessive_slope_mask": excessive_slope,
        "excessive_roughness_mask": excessive_roughness,
        "terrain_hazard_mask": terrain_hazard,
        "terrain_hazard_cleared_by_trajectory_mask": (
            terrain_hazard_cleared_by_trajectory
        ),
        "enclosed_unknown_filled_mask": enclosed_unknown_filled,
        "trajectory_mask": trajectory_free,
        "trajectory_centerline_mask": trajectory_centerline,
    }
    geometry, arrays = _crop_result_arrays(
        geometry, arrays, known, config.output_padding_m
    )

    occupancy = arrays["occupancy"]
    ramp = arrays["ramp_mask"]
    trajectory_mask = arrays["trajectory_mask"]
    trajectory_centerline = arrays["trajectory_centerline_mask"]
    occupied_mask = occupancy == GridClass.OCCUPIED
    clearance = ndimage.distance_transform_edt(~occupied_mask) * geometry.resolution_m
    trajectory_clearance = clearance[trajectory_mask]
    centerline_clearance = clearance[trajectory_centerline]
    cell_area = geometry.resolution_m * geometry.resolution_m
    class_counts = {
        label.name.lower(): int(np.count_nonzero(point_classes == label))
        for label in PointClass
    }
    grid_counts = {
        "free": int(np.count_nonzero(occupancy == GridClass.FREE)),
        "occupied": int(np.count_nonzero(occupancy == GridClass.OCCUPIED)),
        "unknown": int(np.count_nonzero(occupancy == GridClass.UNKNOWN)),
        "ramp": int(np.count_nonzero(ramp)),
    }
    endpoint_final = arrays["obstacle_endpoint_mask"] & occupied_mask
    bridge_final = arrays["obstacle_bridge_mask"] & occupied_mask
    dilation_final = arrays["obstacle_dilation_mask"] & occupied_mask
    occupied_count = max(grid_counts["occupied"], 1)
    derived_final_count = int(np.count_nonzero(bridge_final | dilation_final))
    terrain_hazard_unknown = (
        arrays["terrain_hazard_mask"]
        & (occupancy == GridClass.UNKNOWN)
    )
    metrics: Dict[str, object] = {
        "input_points": int(points.shape[0]),
        "trajectory_poses": int(trajectory.shape[0]),
        "point_class_counts": class_counts,
        "floor_reference": {
            "bootstrap_source": "nearest_optimized_trajectory_z",
            "classification_source": "locally_smoothed_floor_elevation",
            "bootstrap_max_distance_m": config.max_floor_reference_distance_m,
            "trajectory_xy_distance_median": float(np.median(reference_distance)),
            "trajectory_xy_distance_p95": float(
                np.percentile(reference_distance, 95)
            ),
            "trajectory_xy_distance_max": float(np.max(reference_distance)),
            "bootstrap_referenced_points": int(
                np.count_nonzero(bootstrap_referenced)
            ),
            "local_floor_referenced_points": int(
                np.count_nonzero(local_floor_referenced)
            ),
            "local_floor_reference_fraction": float(
                np.mean(local_floor_referenced)
            ),
        },
        "grid": {
            "resolution_m": geometry.resolution_m,
            "width": geometry.width,
            "height": geometry.height,
            "origin_x_m": geometry.origin_x_m,
            "origin_y_m": geometry.origin_y_m,
            "max_x_m": geometry.max_x_m,
            "max_y_m": geometry.max_y_m,
            "cell_counts": grid_counts,
            "free_area_m2": grid_counts["free"] * cell_area,
            "occupied_area_m2": grid_counts["occupied"] * cell_area,
            "ramp_area_m2": grid_counts["ramp"] * cell_area,
            "ramp_trajectory_runs": ramp_runs,
            "floor_candidate_components": floor_component_count,
            "single_level_same_xy_max_z_span_m": same_xy_z_span_m,
        },
        "obstacle_evidence": {
            "source": "merged_pcd_endpoints",
            "contains_scan_miss_evidence": False,
            "contains_temporal_observation_count": False,
            "rejected_low_count_cells": int(
                np.count_nonzero(arrays["obstacle_rejected_low_count_mask"])
            ),
            "rejected_low_support_cells": int(
                np.count_nonzero(arrays["obstacle_rejected_low_support_mask"])
            ),
            "rejected_far_from_free_cells": int(
                np.count_nonzero(arrays["obstacle_rejected_far_from_free_mask"])
            ),
            "endpoint_cells_before_trajectory_clear": int(
                np.count_nonzero(arrays["obstacle_endpoint_mask"])
            ),
            "bridge_only_cells_before_trajectory_clear": int(
                np.count_nonzero(arrays["obstacle_bridge_mask"])
            ),
            "dilation_only_cells_before_trajectory_clear": int(
                np.count_nonzero(arrays["obstacle_dilation_mask"])
            ),
            "cleared_by_trajectory_cells": int(
                np.count_nonzero(arrays["obstacle_cleared_by_trajectory_mask"])
            ),
            "cleared_endpoint_cells": int(
                np.count_nonzero(
                    arrays["obstacle_endpoint_mask"]
                    & arrays["obstacle_cleared_by_trajectory_mask"]
                )
            ),
            "final_endpoint_cells": int(np.count_nonzero(endpoint_final)),
            "final_bridge_only_cells": int(np.count_nonzero(bridge_final)),
            "final_dilation_only_cells": int(np.count_nonzero(dilation_final)),
            "final_derived_cells": derived_final_count,
            "final_derived_fraction": derived_final_count / occupied_count,
            "final_derived_area_m2": derived_final_count * cell_area,
        },
        "terrain_hazards": {
            "policy": "unknown_unless_historically_traversed",
            "roughness_definition": "sqrt_local_pca_minimum_eigenvalue_m",
            "below_floor_endpoint_cells": int(
                np.count_nonzero(arrays["below_floor_endpoint_mask"])
            ),
            "excessive_slope_cells": int(
                np.count_nonzero(arrays["excessive_slope_mask"])
            ),
            "excessive_roughness_cells": int(
                np.count_nonzero(arrays["excessive_roughness_mask"])
            ),
            "hazard_union_cells": int(
                np.count_nonzero(arrays["terrain_hazard_mask"])
            ),
            "hazard_cells_cleared_by_trajectory": int(
                np.count_nonzero(
                    arrays["terrain_hazard_cleared_by_trajectory_mask"]
                )
            ),
            "hazard_cells_left_unknown": int(
                np.count_nonzero(terrain_hazard_unknown)
            ),
            "requires_planner_allow_unknown_false": True,
            "thresholds_are_robot_measurement_verified": False,
        },
        "interior_free_completion": {
            "enabled": config.fill_enclosed_unknown_as_free,
            "policy": "fill_holes_in_known_floor_or_wall_envelope",
            "filled_cells": int(
                np.count_nonzero(arrays["enclosed_unknown_filled_mask"])
            ),
            "filled_area_m2": float(
                np.count_nonzero(arrays["enclosed_unknown_filled_mask"])
                * cell_area
            ),
            "exterior_unknown_preserved": True,
            "untraversed_terrain_hazards_preserved_unknown": True,
        },
        "trajectory": {
            "yaw_source": yaw_source,
            "timestamp_source_available": trajectory_stamp_ns is not None,
            "segment_count": int(trajectory.shape[0] - 1),
            "interpolated_segment_count": int(
                np.count_nonzero(trajectory_segment_valid)
            ),
            **segment_rejections,
            "footprint_length_m": config.robot_footprint_length_m,
            "footprint_width_m": config.robot_footprint_width_m,
            "footprint_margin_m": config.trajectory_footprint_margin_m,
            "poses_in_collision_before_clear": int(
                np.count_nonzero(footprint_collision_before_clear)
            ),
            "pose_collision_fraction_before_clear": float(
                np.mean(footprint_collision_before_clear)
            ),
            "poses_in_collision_after_clear": int(
                np.count_nonzero(footprint_collision_after_clear)
            ),
            "collision_after_clear_is_self_consistency_only": True,
            "pose_collision_fraction_after_clear": float(
                np.mean(footprint_collision_after_clear)
            ),
            "occupied_cells_after_clear": int(
                np.count_nonzero(occupied_mask & trajectory_mask)
            ),
            "minimum_obstacle_clearance_m": (
                float(np.min(trajectory_clearance))
                if trajectory_clearance.size
                else None
            ),
            "median_obstacle_clearance_m": (
                float(np.median(trajectory_clearance))
                if trajectory_clearance.size
                else None
            ),
            "centerline_minimum_obstacle_clearance_m": (
                float(np.min(centerline_clearance))
                if centerline_clearance.size
                else None
            ),
            "centerline_median_obstacle_clearance_m": (
                float(np.median(centerline_clearance))
                if centerline_clearance.size
                else None
            ),
        },
    }
    return ProjectionResult(
        geometry=geometry,
        occupancy=occupancy,
        ramp_mask=ramp,
        floor_elevation_m=arrays["floor_elevation_m"].astype(np.float32),
        floor_slope_deg=arrays["floor_slope_deg"].astype(np.float32),
        floor_roughness_m=arrays["floor_roughness_m"].astype(np.float32),
        floor_counts=arrays["floor_counts"],
        obstacle_counts=arrays["obstacle_counts"],
        below_floor_counts=arrays["below_floor_counts"],
        obstacle_rejected_low_count_mask=arrays[
            "obstacle_rejected_low_count_mask"
        ],
        obstacle_rejected_low_support_mask=arrays[
            "obstacle_rejected_low_support_mask"
        ],
        obstacle_rejected_far_from_free_mask=arrays[
            "obstacle_rejected_far_from_free_mask"
        ],
        obstacle_endpoint_mask=arrays["obstacle_endpoint_mask"],
        obstacle_bridge_mask=arrays["obstacle_bridge_mask"],
        obstacle_dilation_mask=arrays["obstacle_dilation_mask"],
        obstacle_cleared_by_trajectory_mask=arrays[
            "obstacle_cleared_by_trajectory_mask"
        ],
        below_floor_endpoint_mask=arrays["below_floor_endpoint_mask"],
        excessive_slope_mask=arrays["excessive_slope_mask"],
        excessive_roughness_mask=arrays["excessive_roughness_mask"],
        terrain_hazard_mask=arrays["terrain_hazard_mask"],
        terrain_hazard_cleared_by_trajectory_mask=arrays[
            "terrain_hazard_cleared_by_trajectory_mask"
        ],
        enclosed_unknown_filled_mask=arrays[
            "enclosed_unknown_filled_mask"
        ],
        trajectory_mask=trajectory_mask,
        trajectory_centerline_mask=trajectory_centerline,
        trajectory_yaw_rad=trajectory_yaw,
        trajectory_segment_valid=trajectory_segment_valid,
        trajectory_yaw_source=yaw_source,
        point_classes=point_classes,
        relative_point_height_m=relative_height.astype(np.float32),
        metrics=metrics,
    )
