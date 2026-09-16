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

"""Configuration validation for the offline map projector."""

from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class ProjectionConfig:
    """Parameters controlling geometric classification and 2D projection."""

    resolution_m: float = 0.05
    bounds_quantile_low: float = 0.001
    bounds_quantile_high: float = 0.999
    initial_padding_m: float = 1.0
    output_padding_m: float = 0.75
    max_grid_cells: int = 8_000_000
    floor_min_offset_m: float = -0.18
    floor_max_offset_m: float = 0.12
    obstacle_min_height_m: float = 0.12
    obstacle_max_height_m: float = 2.0
    max_floor_reference_distance_m: float = 45.0
    floor_min_points_per_cell: int = 1
    obstacle_min_points_per_cell: int = 1
    obstacle_support_radius_m: float = 0.12
    obstacle_min_support_points: int = 2
    floor_support_radius_m: float = 0.40
    floor_smoothing_sigma_m: float = 0.30
    floor_normal_neighbors: int = 12
    floor_normal_max_neighbor_distance_m: float = 0.75
    floor_normal_max_tilt_deg: float = 25.0
    floor_planarity_ratio_max: float = 0.20
    floor_surface_spread_min: float = 0.08
    ramp_min_slope_deg: float = 3.0
    ramp_max_slope_deg: float = 15.0
    ramp_min_area_m2: float = 0.50
    ramp_support_margin_m: float = 0.20
    ramp_trajectory_smoothing_m: float = 0.75
    ramp_min_length_m: float = 0.75
    ramp_min_height_change_m: float = 0.08
    ramp_region_radius_m: float = 0.80
    robot_footprint_length_m: float = 1.066
    robot_footprint_width_m: float = 0.88
    trajectory_footprint_margin_m: float = 0.0
    trajectory_max_segment_xy_m: float = 0.25
    trajectory_max_segment_z_m: float = 0.20
    trajectory_max_segment_yaw_deg: float = 45.0
    trajectory_max_segment_dt_s: float = 0.50
    obstacle_keep_distance_to_free_m: float = 0.80
    obstacle_closing_radius_m: float = 0.05
    obstacle_dilation_radius_m: float = 0.0
    below_floor_min_points_per_cell: int = 1
    below_floor_support_radius_m: float = 0.12
    below_floor_min_support_points: int = 2
    max_traversable_floor_slope_deg: float = 15.0
    max_traversable_floor_roughness_m: float = 0.08
    single_level_xy_bin_m: float = 0.25
    single_level_max_z_span_m: float = 0.75
    fill_enclosed_unknown_as_free: bool = False
    free_value: int = 254
    occupied_value: int = 0
    unknown_value: int = 205

    def __post_init__(self) -> None:
        boolean_names = {"fill_enclosed_unknown_as_free"}
        integer_names = {
            "max_grid_cells",
            "floor_min_points_per_cell",
            "obstacle_min_points_per_cell",
            "obstacle_min_support_points",
            "floor_normal_neighbors",
            "below_floor_min_points_per_cell",
            "below_floor_min_support_points",
            "free_value",
            "occupied_value",
            "unknown_value",
        }
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in boolean_names:
                if not isinstance(value, bool):
                    raise ValueError(f"{item.name} must be a boolean")
            elif item.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{item.name} must be an integer")
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{item.name} must be a real number")
            elif not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")

        if not 0.0 < self.resolution_m <= 1.0:
            raise ValueError("resolution_m must be in (0, 1]")
        if not 0.0 <= self.bounds_quantile_low < self.bounds_quantile_high <= 1.0:
            raise ValueError("bounds quantiles must satisfy 0 <= low < high <= 1")
        if self.initial_padding_m < 0.0 or self.output_padding_m < 0.0:
            raise ValueError("map padding cannot be negative")
        if self.max_grid_cells <= 0:
            raise ValueError("max_grid_cells must be positive")
        if self.floor_min_offset_m >= self.floor_max_offset_m:
            raise ValueError("floor_min_offset_m must be below floor_max_offset_m")
        if self.obstacle_min_height_m < self.floor_max_offset_m:
            raise ValueError("obstacle_min_height_m cannot overlap the floor band")
        if self.obstacle_max_height_m <= self.obstacle_min_height_m:
            raise ValueError("obstacle height bounds are invalid")
        if self.max_floor_reference_distance_m <= 0.0:
            raise ValueError("max_floor_reference_distance_m must be positive")
        if self.floor_min_points_per_cell <= 0 or self.obstacle_min_points_per_cell <= 0:
            raise ValueError("minimum point counts must be positive")
        if self.obstacle_min_support_points <= 0 or self.below_floor_min_support_points <= 0:
            raise ValueError("support point counts must be positive")
        if self.below_floor_min_points_per_cell <= 0:
            raise ValueError("below_floor_min_points_per_cell must be positive")
        if self.robot_footprint_length_m <= 0.0 or self.robot_footprint_width_m <= 0.0:
            raise ValueError("robot footprint dimensions must be positive")
        if self.floor_normal_neighbors < 3:
            raise ValueError("floor_normal_neighbors must be at least 3")
        if self.floor_normal_max_neighbor_distance_m <= 0.0:
            raise ValueError("floor normal neighbor distance must be positive")
        if not 0.0 < self.floor_normal_max_tilt_deg < 90.0:
            raise ValueError("floor_normal_max_tilt_deg must be in (0, 90)")
        if not 0.0 < self.floor_planarity_ratio_max < 1.0:
            raise ValueError("floor_planarity_ratio_max must be in (0, 1)")
        if not 0.0 <= self.floor_surface_spread_min <= 1.0:
            raise ValueError("floor_surface_spread_min must be in [0, 1]")
        radius_values = (
            self.floor_support_radius_m,
            self.obstacle_support_radius_m,
            self.floor_smoothing_sigma_m,
            self.trajectory_footprint_margin_m,
            self.obstacle_keep_distance_to_free_m,
            self.obstacle_closing_radius_m,
            self.obstacle_dilation_radius_m,
            self.ramp_support_margin_m,
            self.ramp_trajectory_smoothing_m,
            self.ramp_min_length_m,
            self.ramp_min_height_change_m,
            self.ramp_region_radius_m,
            self.below_floor_support_radius_m,
        )
        if any(value < 0.0 for value in radius_values):
            raise ValueError("radius and smoothing parameters cannot be negative")
        if not 0.0 <= self.ramp_min_slope_deg < self.ramp_max_slope_deg < 90.0:
            raise ValueError("ramp slope bounds must satisfy 0 <= min < max < 90")
        if self.ramp_min_area_m2 < 0.0:
            raise ValueError("ramp_min_area_m2 cannot be negative")
        if self.trajectory_max_segment_xy_m <= 0.0:
            raise ValueError("trajectory_max_segment_xy_m must be positive")
        if self.trajectory_max_segment_z_m <= 0.0:
            raise ValueError("trajectory_max_segment_z_m must be positive")
        if not 0.0 < self.trajectory_max_segment_yaw_deg <= 180.0:
            raise ValueError("trajectory_max_segment_yaw_deg must be in (0, 180]")
        if self.trajectory_max_segment_dt_s <= 0.0:
            raise ValueError("trajectory_max_segment_dt_s must be positive")
        if not 0.0 < self.max_traversable_floor_slope_deg < 90.0:
            raise ValueError("max_traversable_floor_slope_deg must be in (0, 90)")
        if self.max_traversable_floor_roughness_m <= 0.0:
            raise ValueError("max_traversable_floor_roughness_m must be positive")
        if self.single_level_xy_bin_m <= 0.0:
            raise ValueError("single_level_xy_bin_m must be positive")
        if self.single_level_max_z_span_m <= 0.0:
            raise ValueError("single_level_max_z_span_m must be positive")
        image_values = (self.free_value, self.occupied_value, self.unknown_value)
        if any(value < 0 or value > 255 for value in image_values):
            raise ValueError("PGM values must be in [0, 255]")
        if len(set(image_values)) != len(image_values):
            raise ValueError("free, occupied, and unknown PGM values must differ")
        if not self.occupied_value < self.unknown_value < self.free_value:
            raise ValueError(
                "PGM values must satisfy occupied_value < unknown_value < free_value"
            )

    @classmethod
    def from_mapping(cls, values: Dict[str, Any]) -> "ProjectionConfig":
        """Build a validated config while rejecting misspelled keys."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown projection parameters: {', '.join(unknown)}")
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable parameter dictionary."""
        return asdict(self)


def load_projection_config(path: Path) -> ProjectionConfig:
    """Load the projection section from a YAML file."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("projection"), dict):
        raise ValueError("config must contain a mapping named 'projection'")
    return ProjectionConfig.from_mapping(document["projection"])
