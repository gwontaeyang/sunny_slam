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

"""Per-scan visibility evidence for merged-PCD obstacle candidates."""

from dataclasses import dataclass, fields
from enum import IntEnum
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation, Slerp
import yaml

from .configuration import ProjectionConfig
from .projection import GridClass, PointClass, ProjectionResult
from .trajectory_io import TrajectoryData


class VisibilityState(IntEnum):
    """Mutually exclusive result of one scan observing one candidate."""

    UNOBSERVED = 0
    AMBIGUOUS = 1
    HIT = 2
    MISS = 3
    OCCLUDED = 4


class EvidenceDecision(IntEnum):
    """Persistent obstacle decision after all accepted scans."""

    UNCERTAIN = 0
    CONFIRMED = 1
    CONTRADICTED = 2


@dataclass(frozen=True)
class RawVisibilityConfig:
    """Sensor and inverse-model parameters for raw scan visibility audits."""

    topic: str = "/hesai/center/lidar_points"
    expected_frame_id: str = "hesai_center"
    sensor_translation_xyz_m: Tuple[float, float, float] = (0.36, 0.0, 1.5496)
    sensor_rotation_rpy_deg: Tuple[float, float, float] = (0.0, 0.0, 90.0)
    evidence_resolution_m: float = 0.20
    height_band_edges_m: Tuple[float, ...] = (0.12, 0.40, 0.80, 1.20, 2.0)
    min_range_m: float = 1.0
    max_range_m: float = 45.0
    azimuth_bin_deg: float = 0.5
    azimuth_neighbor_bins: int = 1
    vertical_neighbor_rings: int = 1
    max_ring_angle_error_deg: float = 1.5
    hit_tolerance_m: float = 0.30
    miss_margin_m: float = 0.45
    occlusion_margin_m: float = 0.30
    min_confirming_hit_scans: int = 2
    min_hit_fraction_for_confirmation: float = 0.20
    min_contradicting_miss_scans: int = 10
    max_hit_fraction_for_contradiction: float = 0.10
    contradiction_miss_to_hit_ratio: float = 4.0
    max_pose_time_offset_s: float = 0.060
    max_pose_interpolation_gap_s: float = 0.25
    scan_stride: int = 1
    raw_point_stride: int = 1

    def __post_init__(self) -> None:
        integer_names = {
            "azimuth_neighbor_bins",
            "vertical_neighbor_rings",
            "min_confirming_hit_scans",
            "min_contradicting_miss_scans",
            "scan_stride",
            "raw_point_stride",
        }
        tuple_names = {
            "sensor_translation_xyz_m",
            "sensor_rotation_rpy_deg",
            "height_band_edges_m",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in tuple_names or isinstance(value, str):
                continue
            if item.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{item.name} must be an integer")
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{item.name} must be a real number")
            elif not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")

        if not self.topic or not self.expected_frame_id:
            raise ValueError("topic and expected_frame_id cannot be empty")
        if len(self.sensor_translation_xyz_m) != 3:
            raise ValueError("sensor_translation_xyz_m must contain XYZ")
        if len(self.sensor_rotation_rpy_deg) != 3:
            raise ValueError("sensor_rotation_rpy_deg must contain roll, pitch, yaw")
        if len(self.height_band_edges_m) < 2:
            raise ValueError("height_band_edges_m requires at least two edges")
        numeric_tuples = (
            self.sensor_translation_xyz_m,
            self.sensor_rotation_rpy_deg,
            self.height_band_edges_m,
        )
        if not all(math.isfinite(float(value)) for values in numeric_tuples for value in values):
            raise ValueError("sensor transforms and height bands must be finite")
        edges = np.asarray(self.height_band_edges_m, dtype=np.float64)
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("height_band_edges_m must be strictly increasing")
        if self.evidence_resolution_m <= 0.0:
            raise ValueError("evidence_resolution_m must be positive")
        if not 0.0 < self.min_range_m < self.max_range_m:
            raise ValueError("raw range bounds are invalid")
        if not 0.0 < self.azimuth_bin_deg <= 10.0:
            raise ValueError("azimuth_bin_deg must be in (0, 10]")
        if self.azimuth_neighbor_bins < 0 or self.vertical_neighbor_rings < 0:
            raise ValueError("range-image neighborhoods cannot be negative")
        positive_values = (
            self.max_ring_angle_error_deg,
            self.hit_tolerance_m,
            self.miss_margin_m,
            self.occlusion_margin_m,
            self.contradiction_miss_to_hit_ratio,
            self.max_pose_time_offset_s,
            self.max_pose_interpolation_gap_s,
        )
        if any(value <= 0.0 for value in positive_values):
            raise ValueError("visibility tolerances must be positive")
        fractions = (
            self.min_hit_fraction_for_confirmation,
            self.max_hit_fraction_for_contradiction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("hit fractions must be in [0, 1]")
        if self.min_confirming_hit_scans <= 0:
            raise ValueError("min_confirming_hit_scans must be positive")
        if self.min_contradicting_miss_scans <= 0:
            raise ValueError("min_contradicting_miss_scans must be positive")
        if self.scan_stride <= 0 or self.raw_point_stride <= 0:
            raise ValueError("scan and point strides must be positive")

    @classmethod
    def from_mapping(cls, values: Dict[str, object]) -> "RawVisibilityConfig":
        """Create a validated config while rejecting misspelled keys."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown raw visibility parameters: {', '.join(unknown)}")
        converted = dict(values)
        for name in (
            "sensor_translation_xyz_m",
            "sensor_rotation_rpy_deg",
            "height_band_edges_m",
        ):
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted)


def load_raw_visibility_config(path: Path) -> RawVisibilityConfig:
    """Load the raw_visibility mapping from YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("raw_visibility"), dict
    ):
        raise ValueError("config must contain a mapping named 'raw_visibility'")
    return RawVisibilityConfig.from_mapping(document["raw_visibility"])


class PoseInterpolator:
    """Continuous SE(3) interpolation over an optimized timestamped trajectory."""

    def __init__(
        self,
        trajectory: TrajectoryData,
        max_interpolation_gap_s: float,
    ) -> None:
        if trajectory.stamp_ns is None:
            raise ValueError("raw visibility requires trajectory stamp_ns")
        if trajectory.orientation_xyzw is None:
            raise ValueError("raw visibility requires full trajectory quaternions")
        self._origin_second = int(trajectory.stamp_ns[0] // 1_000_000_000)
        self._times_s = (
            trajectory.stamp_ns.astype(np.float64) * 1.0e-9
            - self._origin_second
        )
        self._positions = np.asarray(trajectory.poses[:, :3], dtype=np.float64)
        self._rotations = Rotation.from_quat(trajectory.orientation_xyzw)
        self._slerp = Slerp(self._times_s, self._rotations)
        self._max_gap_s = float(max_interpolation_gap_s)

    def _relative_times(self, absolute_seconds: np.ndarray) -> np.ndarray:
        return np.asarray(absolute_seconds, dtype=np.float64) - self._origin_second

    def nearest_offset_s(self, absolute_second: float) -> float:
        """Return absolute offset to the nearest optimized pose."""
        query = float(absolute_second) - self._origin_second
        insertion = int(np.searchsorted(self._times_s, query))
        candidates = np.asarray(
            [max(0, insertion - 1), min(self._times_s.size - 1, insertion)],
            dtype=np.int64,
        )
        return float(np.min(np.abs(self._times_s[candidates] - query)))

    def sample(
        self, absolute_seconds: np.ndarray
    ) -> Tuple[np.ndarray, Rotation, np.ndarray]:
        """Interpolate positions and rotations; return validity per query."""
        query = self._relative_times(absolute_seconds)
        valid_range = (query >= self._times_s[0]) & (query <= self._times_s[-1])
        clipped = np.clip(query, self._times_s[0], self._times_s[-1])
        lower = np.searchsorted(self._times_s, clipped, side="right") - 1
        lower = np.clip(lower, 0, self._times_s.size - 2)
        upper = lower + 1
        span = self._times_s[upper] - self._times_s[lower]
        valid = valid_range & (span <= self._max_gap_s)
        alpha = np.divide(
            clipped - self._times_s[lower],
            span,
            out=np.zeros_like(clipped),
            where=span > 0.0,
        )
        positions = (
            self._positions[lower]
            + alpha[:, None] * (self._positions[upper] - self._positions[lower])
        )
        rotations = self._slerp(clipped)
        return positions, rotations, valid


@dataclass(frozen=True)
class ObstacleCandidates:
    """Height-banded obstacle surfaces represented on a coarse evidence grid."""

    points_map: np.ndarray
    band_indices: np.ndarray
    coarse_rows: np.ndarray
    coarse_columns: np.ndarray
    candidate_id_by_band: np.ndarray
    point_candidate_ids: np.ndarray
    coarse_origin_xy_m: np.ndarray
    coarse_resolution_m: float


@dataclass(frozen=True)
class RangeImage:
    """Nearest raw return per native ring and azimuth bin."""

    ranges_m: np.ndarray
    timestamps_s: np.ndarray
    ring_values: np.ndarray
    ring_elevations_rad: np.ndarray
    azimuth_bin_rad: float


@dataclass
class VisibilityEvidence:
    """Accumulated distinct-scan evidence for every obstacle candidate."""

    hit_scan_count: np.ndarray
    miss_scan_count: np.ndarray
    occluded_scan_count: np.ndarray
    ambiguous_scan_count: np.ndarray
    unobserved_scan_count: np.ndarray
    decisions: np.ndarray
    scan_metrics: Dict[str, object]


@dataclass(frozen=True)
class VisibilityRefinement:
    """Planning grid and provenance after applying conservative visibility rules."""

    occupancy: np.ndarray
    retained_endpoint_mask: np.ndarray
    bridge_mask: np.ndarray
    contradicted_endpoint_mask: np.ndarray
    confirmed_endpoint_mask: np.ndarray
    uncertain_endpoint_mask: np.ndarray
    recovered_free_mask: np.ndarray
    point_decisions: np.ndarray
    metrics: Dict[str, object]


def build_obstacle_candidates(
    points: np.ndarray,
    result: ProjectionResult,
    config: RawVisibilityConfig,
) -> ObstacleCandidates:
    """Aggregate final direct endpoints into coarse XY and height-band candidates."""
    geometry = result.geometry
    coarse_width = int(
        np.ceil(geometry.width * geometry.resolution_m / config.evidence_resolution_m)
    )
    coarse_height = int(
        np.ceil(geometry.height * geometry.resolution_m / config.evidence_resolution_m)
    )
    columns, rows = geometry.world_to_grid(points[:, :2])
    in_grid = (
        (columns >= 0)
        & (columns < geometry.width)
        & (rows >= 0)
        & (rows < geometry.height)
    )
    final_endpoint = result.obstacle_endpoint_mask & (
        result.occupancy == GridClass.OCCUPIED
    )
    endpoint_point = np.zeros(points.shape[0], dtype=bool)
    endpoint_point[in_grid] = final_endpoint[rows[in_grid], columns[in_grid]]
    body_point = result.point_classes == PointClass.BODY_OBSTACLE
    candidate_point = endpoint_point & body_point
    if not np.any(candidate_point):
        raise ValueError("no final body-obstacle endpoints are available for visibility")

    relative_height = result.relative_point_height_m[candidate_point]
    band_edges = np.asarray(config.height_band_edges_m, dtype=np.float64)
    band_indices = np.searchsorted(
        band_edges[1:-1], relative_height, side="right"
    )
    selected_points = points[candidate_point]
    coarse_columns = np.floor(
        (selected_points[:, 0] - geometry.origin_x_m)
        / config.evidence_resolution_m
    ).astype(np.int64)
    coarse_rows = np.floor(
        (selected_points[:, 1] - geometry.origin_y_m)
        / config.evidence_resolution_m
    ).astype(np.int64)
    valid = (
        (coarse_columns >= 0)
        & (coarse_columns < coarse_width)
        & (coarse_rows >= 0)
        & (coarse_rows < coarse_height)
        & (relative_height >= band_edges[0])
        & (relative_height <= band_edges[-1])
    )
    selected_indices = np.flatnonzero(candidate_point)[valid]
    selected_points = selected_points[valid]
    band_indices = band_indices[valid]
    coarse_columns = coarse_columns[valid]
    coarse_rows = coarse_rows[valid]
    plane_size = coarse_height * coarse_width
    keys = (
        band_indices.astype(np.int64) * plane_size
        + coarse_rows * coarse_width
        + coarse_columns
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((unique_keys.size, 3), dtype=np.float64)
    np.add.at(sums, inverse, selected_points)
    representative = sums / count[:, None]
    group_band = unique_keys // plane_size
    remainder = unique_keys % plane_size
    group_rows = remainder // coarse_width
    group_columns = remainder % coarse_width
    candidate_id_by_band = np.full(
        (band_edges.size - 1, coarse_height, coarse_width),
        -1,
        dtype=np.int32,
    )
    candidate_id_by_band[group_band, group_rows, group_columns] = np.arange(
        unique_keys.size, dtype=np.int32
    )
    point_candidate_ids = np.full(points.shape[0], -1, dtype=np.int32)
    point_candidate_ids[selected_indices] = inverse.astype(np.int32)
    return ObstacleCandidates(
        points_map=representative,
        band_indices=group_band.astype(np.int16),
        coarse_rows=group_rows.astype(np.int32),
        coarse_columns=group_columns.astype(np.int32),
        candidate_id_by_band=candidate_id_by_band,
        point_candidate_ids=point_candidate_ids,
        coarse_origin_xy_m=np.asarray(
            [geometry.origin_x_m, geometry.origin_y_m], dtype=np.float64
        ),
        coarse_resolution_m=config.evidence_resolution_m,
    )


def _ring_calibration(
    points_sensor: np.ndarray,
    rings: np.ndarray,
    min_range_m: float,
    max_range_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    ranges = np.linalg.norm(points_sensor, axis=1)
    finite = (
        np.isfinite(points_sensor).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= min_range_m)
        & (ranges <= max_range_m)
    )
    ring_values = np.unique(rings[finite].astype(np.int64))
    if ring_values.size < 2:
        raise ValueError("raw cloud does not contain enough LiDAR rings")
    horizontal = np.linalg.norm(points_sensor[finite, :2], axis=1)
    elevation = np.arctan2(points_sensor[finite, 2], horizontal)
    finite_rings = rings[finite].astype(np.int64)
    ring_elevations = np.asarray(
        [np.median(elevation[finite_rings == value]) for value in ring_values],
        dtype=np.float64,
    )
    return ring_values, ring_elevations


def build_range_image(
    points_sensor: np.ndarray,
    rings: np.ndarray,
    timestamps_s: np.ndarray,
    config: RawVisibilityConfig,
    calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> RangeImage:
    """Build a nearest-return range image without inventing no-return beams."""
    points_sensor = np.asarray(points_sensor, dtype=np.float64)
    rings = np.asarray(rings)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    if points_sensor.ndim != 2 or points_sensor.shape[1] != 3:
        raise ValueError("raw points must have shape (N, 3)")
    if rings.shape != (points_sensor.shape[0],):
        raise ValueError("ring array length does not match raw points")
    if timestamps_s.shape != (points_sensor.shape[0],):
        raise ValueError("timestamp array length does not match raw points")
    if calibration is None:
        calibration = _ring_calibration(
            points_sensor,
            rings,
            config.min_range_m,
            config.max_range_m,
        )
    ring_values, ring_elevations = calibration
    sample = slice(None, None, config.raw_point_stride)
    points_sensor = points_sensor[sample]
    rings = rings[sample].astype(np.int64, copy=False)
    timestamps_s = timestamps_s[sample]
    ranges = np.linalg.norm(points_sensor, axis=1)
    ring_rows = np.searchsorted(ring_values, rings)
    valid_ring = ring_rows < ring_values.size
    valid_ring &= ring_values[np.minimum(ring_rows, ring_values.size - 1)] == rings
    finite = (
        np.isfinite(points_sensor).all(axis=1)
        & np.isfinite(timestamps_s)
        & np.isfinite(ranges)
        & valid_ring
        & (ranges >= config.min_range_m)
        & (ranges <= config.max_range_m)
    )
    if not np.any(finite):
        raise ValueError("raw scan has no finite points inside configured range")
    points_sensor = points_sensor[finite]
    ranges = ranges[finite]
    timestamps_s = timestamps_s[finite]
    ring_rows = ring_rows[finite]
    azimuth_bin_count = int(np.ceil(360.0 / config.azimuth_bin_deg))
    azimuth_bin_rad = 2.0 * np.pi / azimuth_bin_count
    azimuth = np.mod(np.arctan2(points_sensor[:, 1], points_sensor[:, 0]), 2.0 * np.pi)
    azimuth_columns = np.floor(azimuth / azimuth_bin_rad).astype(np.int64)
    flat_indices = ring_rows * azimuth_bin_count + azimuth_columns
    cell_count = ring_values.size * azimuth_bin_count
    flat_ranges = np.full(cell_count, np.inf, dtype=np.float64)
    np.minimum.at(flat_ranges, flat_indices, ranges)
    is_minimum = ranges <= flat_ranges[flat_indices] + 1.0e-9
    flat_timestamps = np.full(cell_count, np.nan, dtype=np.float64)
    flat_timestamps[flat_indices[is_minimum]] = timestamps_s[is_minimum]
    return RangeImage(
        ranges_m=flat_ranges.reshape(ring_values.size, azimuth_bin_count),
        timestamps_s=flat_timestamps.reshape(ring_values.size, azimuth_bin_count),
        ring_values=np.asarray(ring_values, dtype=np.int64),
        ring_elevations_rad=np.asarray(ring_elevations, dtype=np.float64),
        azimuth_bin_rad=azimuth_bin_rad,
    )


def _nearest_sorted_indices(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    insertion = np.searchsorted(sorted_values, values)
    left = np.clip(insertion - 1, 0, sorted_values.size - 1)
    right = np.clip(insertion, 0, sorted_values.size - 1)
    choose_right = np.abs(sorted_values[right] - values) < np.abs(
        sorted_values[left] - values
    )
    return np.where(choose_right, right, left)


def _lookup_range_image(
    points_sensor: np.ndarray,
    image: RangeImage,
    config: RawVisibilityConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.linalg.norm(points_sensor, axis=1)
    horizontal = np.linalg.norm(points_sensor[:, :2], axis=1)
    elevation = np.arctan2(points_sensor[:, 2], horizontal)
    azimuth = np.mod(np.arctan2(points_sensor[:, 1], points_sensor[:, 0]), 2.0 * np.pi)
    sorted_rows = np.argsort(image.ring_elevations_rad)
    sorted_elevations = image.ring_elevations_rad[sorted_rows]
    nearest_position = _nearest_sorted_indices(sorted_elevations, elevation)
    nearest_error = np.abs(sorted_elevations[nearest_position] - elevation)
    azimuth_columns = np.floor(azimuth / image.azimuth_bin_rad).astype(np.int64)
    azimuth_columns %= image.ranges_m.shape[1]
    best_range = np.full(points_sensor.shape[0], np.inf, dtype=np.float64)
    best_timestamp = np.full(points_sensor.shape[0], np.nan, dtype=np.float64)
    for ring_offset in range(
        -config.vertical_neighbor_rings,
        config.vertical_neighbor_rings + 1,
    ):
        position = np.clip(
            nearest_position + ring_offset,
            0,
            sorted_rows.size - 1,
        )
        ring_rows = sorted_rows[position]
        for azimuth_offset in range(
            -config.azimuth_neighbor_bins,
            config.azimuth_neighbor_bins + 1,
        ):
            columns = (azimuth_columns + azimuth_offset) % image.ranges_m.shape[1]
            observed = image.ranges_m[ring_rows, columns]
            better = observed < best_range
            best_range[better] = observed[better]
            best_timestamp[better] = image.timestamps_s[ring_rows[better], columns[better]]
    valid = (
        np.isfinite(points_sensor).all(axis=1)
        & np.isfinite(best_range)
        & np.isfinite(best_timestamp)
        & (ranges >= config.min_range_m)
        & (ranges <= config.max_range_m)
        & (
            nearest_error
            <= np.deg2rad(config.max_ring_angle_error_deg)
        )
    )
    return best_range, best_timestamp, valid


class VisibilityAccumulator:
    """Accumulate scan-distinct hit, miss, occlusion, and ambiguity counters."""

    def __init__(
        self,
        candidates: ObstacleCandidates,
        trajectory: TrajectoryData,
        config: RawVisibilityConfig,
    ) -> None:
        self.candidates = candidates
        self.config = config
        self.pose_interpolator = PoseInterpolator(
            trajectory,
            config.max_pose_interpolation_gap_s,
        )
        self.sensor_translation = np.asarray(
            config.sensor_translation_xyz_m, dtype=np.float64
        )
        self.sensor_rotation = Rotation.from_euler(
            "xyz",
            config.sensor_rotation_rpy_deg,
            degrees=True,
        )
        candidate_count = candidates.points_map.shape[0]
        self.hit_scan_count = np.zeros(candidate_count, dtype=np.uint32)
        self.miss_scan_count = np.zeros(candidate_count, dtype=np.uint32)
        self.occluded_scan_count = np.zeros(candidate_count, dtype=np.uint32)
        self.ambiguous_scan_count = np.zeros(candidate_count, dtype=np.uint32)
        self.unobserved_scan_count = np.zeros(candidate_count, dtype=np.uint32)
        self.raw_scan_count = 0
        self.accepted_scan_count = 0
        self.rejected_pose_time_count = 0
        self.rejected_frame_count = 0
        self.rejected_empty_count = 0
        self._calibration: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def _map_to_sensor(
        self,
        points_map: np.ndarray,
        timestamps_s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        positions, rotations, valid = self.pose_interpolator.sample(timestamps_s)
        points_base = rotations.inv().apply(points_map - positions)
        points_sensor = self.sensor_rotation.inv().apply(
            points_base - self.sensor_translation
        )
        return points_sensor, valid

    def observe_scan(
        self,
        points_sensor: np.ndarray,
        rings: np.ndarray,
        timestamps_s: np.ndarray,
        header_timestamp_s: float,
        frame_id: str,
    ) -> bool:
        """Consume one raw scan and count at most one state per candidate."""
        scan_index = self.raw_scan_count
        self.raw_scan_count += 1
        if scan_index % self.config.scan_stride != 0:
            return False
        if frame_id != self.config.expected_frame_id:
            self.rejected_frame_count += 1
            return False
        if (
            self.pose_interpolator.nearest_offset_s(header_timestamp_s)
            > self.config.max_pose_time_offset_s
        ):
            self.rejected_pose_time_count += 1
            return False
        try:
            if self._calibration is None:
                self._calibration = _ring_calibration(
                    points_sensor,
                    rings,
                    self.config.min_range_m,
                    self.config.max_range_m,
                )
            image = build_range_image(
                points_sensor,
                rings,
                timestamps_s,
                self.config,
                self._calibration,
            )
        except ValueError:
            self.rejected_empty_count += 1
            return False

        candidate_count = self.candidates.points_map.shape[0]
        header_times = np.full(candidate_count, header_timestamp_s, dtype=np.float64)
        initial_points, initial_pose_valid = self._map_to_sensor(
            self.candidates.points_map,
            header_times,
        )
        _, initial_beam_times, initial_lookup_valid = _lookup_range_image(
            initial_points,
            image,
            self.config,
        )
        query_times = np.where(
            np.isfinite(initial_beam_times),
            initial_beam_times,
            header_timestamp_s,
        )
        beam_points, beam_pose_valid = self._map_to_sensor(
            self.candidates.points_map,
            query_times,
        )
        observed_range, _, lookup_valid = _lookup_range_image(
            beam_points,
            image,
            self.config,
        )
        candidate_range = np.linalg.norm(beam_points, axis=1)
        valid = (
            initial_pose_valid
            & initial_lookup_valid
            & beam_pose_valid
            & lookup_valid
        )
        delta = observed_range - candidate_range
        hit = valid & (np.abs(delta) <= self.config.hit_tolerance_m)
        miss = valid & (delta >= self.config.miss_margin_m)
        occluded = valid & (delta <= -self.config.occlusion_margin_m)
        ambiguous = valid & ~(hit | miss | occluded)
        unobserved = ~valid
        self.hit_scan_count += hit.astype(np.uint32)
        self.miss_scan_count += miss.astype(np.uint32)
        self.occluded_scan_count += occluded.astype(np.uint32)
        self.ambiguous_scan_count += ambiguous.astype(np.uint32)
        self.unobserved_scan_count += unobserved.astype(np.uint32)
        self.accepted_scan_count += 1
        return True

    def finish(self) -> VisibilityEvidence:
        """Freeze counters and derive conservative persistent decisions."""
        hit = self.hit_scan_count
        miss = self.miss_scan_count
        decisive = hit.astype(np.float64) + miss.astype(np.float64)
        hit_fraction = np.divide(
            hit,
            decisive,
            out=np.zeros_like(decisive),
            where=decisive > 0.0,
        )
        confirmed = (
            (hit >= self.config.min_confirming_hit_scans)
            & (
                hit_fraction
                >= self.config.min_hit_fraction_for_confirmation
            )
        )
        contradicted = (
            ~confirmed
            & (miss >= self.config.min_contradicting_miss_scans)
            & (
                hit_fraction
                <= self.config.max_hit_fraction_for_contradiction
            )
            & (
                miss.astype(np.float64)
                >= self.config.contradiction_miss_to_hit_ratio
                * np.maximum(hit.astype(np.float64), 1.0)
            )
        )
        decisions = np.full(
            hit.shape,
            EvidenceDecision.UNCERTAIN,
            dtype=np.int8,
        )
        decisions[confirmed] = EvidenceDecision.CONFIRMED
        decisions[contradicted] = EvidenceDecision.CONTRADICTED
        metrics = {
            "raw_scans": self.raw_scan_count,
            "accepted_scans": self.accepted_scan_count,
            "rejected_pose_time_scans": self.rejected_pose_time_count,
            "rejected_frame_scans": self.rejected_frame_count,
            "rejected_empty_scans": self.rejected_empty_count,
            "candidate_groups": int(hit.size),
            "confirmed_groups": int(np.count_nonzero(confirmed)),
            "contradicted_groups": int(np.count_nonzero(contradicted)),
            "uncertain_groups": int(
                np.count_nonzero(decisions == EvidenceDecision.UNCERTAIN)
            ),
            "hit_scan_count_median": float(np.median(hit)),
            "miss_scan_count_median": float(np.median(miss)),
        }
        return VisibilityEvidence(
            hit_scan_count=hit.copy(),
            miss_scan_count=miss.copy(),
            occluded_scan_count=self.occluded_scan_count.copy(),
            ambiguous_scan_count=self.ambiguous_scan_count.copy(),
            unobserved_scan_count=self.unobserved_scan_count.copy(),
            decisions=decisions,
            scan_metrics=metrics,
        )


def _disk(radius_m: float, resolution_m: float) -> np.ndarray:
    radius_cells = int(np.ceil(radius_m / resolution_m))
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    rows, columns = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]
    return (
        rows * rows + columns * columns
        <= (radius_m / resolution_m) ** 2 + 1.0e-9
    )


def apply_visibility_refinement(
    result: ProjectionResult,
    candidates: ObstacleCandidates,
    evidence: VisibilityEvidence,
    projection_config: ProjectionConfig,
) -> VisibilityRefinement:
    """Remove only endpoint regions contradicted in every represented height band."""
    candidate_grid = candidates.candidate_id_by_band
    present = candidate_grid >= 0
    decision_grid = np.full(candidate_grid.shape, -1, dtype=np.int8)
    decision_grid[present] = evidence.decisions[candidate_grid[present]]
    group_count = np.count_nonzero(present, axis=0)
    contradicted_count = np.count_nonzero(
        decision_grid == EvidenceDecision.CONTRADICTED,
        axis=0,
    )
    confirmed_count = np.count_nonzero(
        decision_grid == EvidenceDecision.CONFIRMED,
        axis=0,
    )
    all_contradicted = (group_count > 0) & (contradicted_count == group_count)
    any_confirmed = confirmed_count > 0

    geometry = result.geometry
    fine_rows, fine_columns = np.indices(result.occupancy.shape)
    world_x = geometry.origin_x_m + (fine_columns + 0.5) * geometry.resolution_m
    world_y = geometry.origin_y_m + (fine_rows + 0.5) * geometry.resolution_m
    coarse_columns = np.floor(
        (world_x - candidates.coarse_origin_xy_m[0])
        / candidates.coarse_resolution_m
    ).astype(np.int64)
    coarse_rows = np.floor(
        (world_y - candidates.coarse_origin_xy_m[1])
        / candidates.coarse_resolution_m
    ).astype(np.int64)
    coarse_columns = np.clip(coarse_columns, 0, all_contradicted.shape[1] - 1)
    coarse_rows = np.clip(coarse_rows, 0, all_contradicted.shape[0] - 1)
    final_endpoint = result.obstacle_endpoint_mask & (
        result.occupancy == GridClass.OCCUPIED
    )
    contradicted_endpoint = final_endpoint & all_contradicted[
        coarse_rows, coarse_columns
    ]
    confirmed_endpoint = final_endpoint & any_confirmed[
        coarse_rows, coarse_columns
    ]
    uncertain_endpoint = final_endpoint & ~(
        contradicted_endpoint | confirmed_endpoint
    )
    retained_endpoint = final_endpoint & ~contradicted_endpoint

    closed = retained_endpoint.copy()
    if projection_config.obstacle_closing_radius_m > 0.0:
        # Closing only reconstructs raster gaps; it cannot erase direct returns.
        closed |= ndimage.binary_closing(
            retained_endpoint,
            structure=_disk(
                projection_config.obstacle_closing_radius_m,
                geometry.resolution_m,
            ),
        )
    occupied_before_trajectory_clear = closed
    if projection_config.obstacle_dilation_radius_m > 0.0:
        occupied_before_trajectory_clear = ndimage.binary_dilation(
            closed,
            structure=_disk(
                projection_config.obstacle_dilation_radius_m,
                geometry.resolution_m,
            ),
        )
    trajectory_reclear = (
        occupied_before_trajectory_clear & result.trajectory_mask
    )
    occupied = occupied_before_trajectory_clear & ~result.trajectory_mask
    bridge = closed & ~retained_endpoint & ~result.trajectory_mask

    old_free = result.occupancy == GridClass.FREE
    old_occupied = result.occupancy == GridClass.OCCUPIED
    removed = old_occupied & ~occupied
    safe_floor = (
        np.isfinite(result.floor_elevation_m)
        & ~result.terrain_hazard_mask
    )
    potential_free = old_free | (removed & safe_floor)
    labels, _ = ndimage.label(potential_free, structure=np.ones((3, 3)))
    touched_labels = np.unique(labels[old_free | result.trajectory_mask])
    touched_labels = touched_labels[touched_labels != 0]
    recovered_free = removed & np.isin(labels, touched_labels)
    occupancy = np.full(result.occupancy.shape, GridClass.UNKNOWN, dtype=np.int8)
    occupancy[old_free | recovered_free] = GridClass.FREE
    occupancy[occupied] = GridClass.OCCUPIED

    point_decisions = np.full(
        result.point_classes.shape,
        -1,
        dtype=np.int8,
    )
    mapped_points = candidates.point_candidate_ids >= 0
    point_decisions[mapped_points] = evidence.decisions[
        candidates.point_candidate_ids[mapped_points]
    ]
    cell_area = geometry.resolution_m * geometry.resolution_m
    metrics = {
        "old_occupied_cells": int(np.count_nonzero(old_occupied)),
        "new_occupied_cells": int(np.count_nonzero(occupied)),
        "contradicted_endpoint_cells": int(
            np.count_nonzero(contradicted_endpoint)
        ),
        "confirmed_endpoint_cells": int(np.count_nonzero(confirmed_endpoint)),
        "uncertain_endpoint_cells": int(np.count_nonzero(uncertain_endpoint)),
        "closing_bridge_cells": int(np.count_nonzero(bridge)),
        "trajectory_recleared_cells": int(np.count_nonzero(trajectory_reclear)),
        "recovered_free_cells": int(np.count_nonzero(recovered_free)),
        "old_occupied_area_m2": float(np.count_nonzero(old_occupied) * cell_area),
        "new_occupied_area_m2": float(np.count_nonzero(occupied) * cell_area),
        "recovered_free_area_m2": float(np.count_nonzero(recovered_free) * cell_area),
        "unknown_cells": int(np.count_nonzero(occupancy == GridClass.UNKNOWN)),
        "free_cells": int(np.count_nonzero(occupancy == GridClass.FREE)),
        "occupied_cells": int(np.count_nonzero(occupancy == GridClass.OCCUPIED)),
        "contradicted_policy": (
            "free_only_with_local_floor_and_connection_to_existing_free"
        ),
    }
    return VisibilityRefinement(
        occupancy=occupancy,
        retained_endpoint_mask=retained_endpoint,
        bridge_mask=bridge,
        contradicted_endpoint_mask=contradicted_endpoint,
        confirmed_endpoint_mask=confirmed_endpoint,
        uncertain_endpoint_mask=uncertain_endpoint,
        recovered_free_mask=recovered_free,
        point_decisions=point_decisions,
        metrics=metrics,
    )
