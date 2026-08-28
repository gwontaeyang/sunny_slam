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

"""Deterministic footprint-aware path generation for offline map review."""

from dataclasses import dataclass, fields
import heapq
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
import yaml

from .projection import GridClass, GridGeometry, oriented_footprint_indices


@dataclass(frozen=True)
class PathPlanningConfig:
    """Motion lattice and footprint settings for offline global paths."""

    robot_footprint_length_m: float = 1.066
    robot_footprint_width_m: float = 0.88
    heading_bins: int = 8
    translation_step_m: float = 0.10
    allow_reverse: bool = True
    reverse_cost_multiplier: float = 1.20
    turn_cost_m: float = 0.12
    clearance_cost_weight: float = 0.03
    goal_tolerance_m: float = 0.15
    max_expansions: int = 1_000_000

    def __post_init__(self) -> None:
        integer_names = {"heading_bins", "max_expansions"}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "allow_reverse":
                if not isinstance(value, bool):
                    raise ValueError("allow_reverse must be boolean")
            elif item.name in integer_names:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError(f"{item.name} must be an integer")
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{item.name} must be a real number")
            elif not math.isfinite(float(value)):
                raise ValueError(f"{item.name} must be finite")
        if self.robot_footprint_length_m <= 0.0:
            raise ValueError("robot_footprint_length_m must be positive")
        if self.robot_footprint_width_m <= 0.0:
            raise ValueError("robot_footprint_width_m must be positive")
        if self.heading_bins < 4 or self.heading_bins % 4 != 0:
            raise ValueError("heading_bins must be a multiple of four")
        if self.translation_step_m <= 0.0:
            raise ValueError("translation_step_m must be positive")
        if self.reverse_cost_multiplier < 1.0:
            raise ValueError("reverse_cost_multiplier must be at least one")
        if self.turn_cost_m < 0.0 or self.clearance_cost_weight < 0.0:
            raise ValueError("planning cost weights cannot be negative")
        if self.goal_tolerance_m < 0.0:
            raise ValueError("goal_tolerance_m cannot be negative")
        if self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")

    @classmethod
    def from_mapping(cls, values: Dict[str, object]) -> "PathPlanningConfig":
        """Create a validated planner config while rejecting unknown keys."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown path planning parameters: {', '.join(unknown)}")
        return cls(**values)


def load_path_planning_config(path: Path) -> PathPlanningConfig:
    """Load the path_planning mapping from YAML."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(
        document.get("path_planning"), dict
    ):
        raise ValueError("config must contain a mapping named 'path_planning'")
    return PathPlanningConfig.from_mapping(document["path_planning"])


@dataclass(frozen=True)
class PathPlan:
    """One collision-checked motion-lattice route and its diagnostics."""

    states: np.ndarray
    poses_xy_yaw: np.ndarray
    start_trajectory_index: int
    goal_trajectory_index: int
    expansions: int
    trajectory_exact_valid: np.ndarray
    trajectory_lattice_valid: np.ndarray
    path_exact_valid: np.ndarray
    path_swept_poses_xy_yaw: np.ndarray
    path_swept_exact_valid: np.ndarray
    metrics: Dict[str, object]


def heading_angles(heading_bins: int) -> np.ndarray:
    """Return evenly spaced wrapped headings in radians."""
    return np.linspace(-np.pi, np.pi, heading_bins, endpoint=False)


def yaw_to_heading_indices(yaw_rad: np.ndarray, heading_bins: int) -> np.ndarray:
    """Quantize wrapped yaw to the nearest heading lattice bin."""
    yaw = np.asarray(yaw_rad, dtype=np.float64)
    wrapped = np.mod(yaw + np.pi, 2.0 * np.pi)
    scaled = wrapped * heading_bins / (2.0 * np.pi)
    return np.floor(scaled + 0.5).astype(np.int64) % heading_bins


def _footprint_kernel(
    geometry: GridGeometry,
    length_m: float,
    width_m: float,
    yaw_rad: float,
) -> np.ndarray:
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m
    radius_cells = int(
        np.ceil(np.hypot(half_length, half_width) / geometry.resolution_m)
    )
    rows, columns = np.mgrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]
    delta_x = columns.astype(np.float64) * geometry.resolution_m
    delta_y = rows.astype(np.float64) * geometry.resolution_m
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    local_x = cosine * delta_x + sine * delta_y
    local_y = -sine * delta_x + cosine * delta_y
    return (
        (np.abs(local_x) <= half_length + 1.0e-9)
        & (np.abs(local_y) <= half_width + 1.0e-9)
    )


def build_footprint_validity(
    occupancy: np.ndarray,
    geometry: GridGeometry,
    config: PathPlanningConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute collision-free robot centers for every discrete heading."""
    occupancy = np.asarray(occupancy)
    if occupancy.shape != (geometry.height, geometry.width):
        raise ValueError("occupancy and geometry shapes do not match")
    blocked = occupancy != GridClass.FREE
    angles = heading_angles(config.heading_bins)
    valid = np.empty(
        (config.heading_bins, geometry.height, geometry.width),
        dtype=bool,
    )
    for index, angle in enumerate(angles):
        collision = ndimage.binary_dilation(
            blocked,
            structure=_footprint_kernel(
                geometry,
                config.robot_footprint_length_m,
                config.robot_footprint_width_m,
                angle,
            ),
            border_value=1,
        )
        valid[index] = ~collision
    clearance_m = (
        ndimage.distance_transform_edt(~blocked) * geometry.resolution_m
    )
    return valid, clearance_m


def build_turn_sweep_validity(
    occupancy: np.ndarray,
    geometry: GridGeometry,
    config: PathPlanningConfig,
) -> np.ndarray:
    """Precompute collision-free centers for each counterclockwise heading turn."""
    occupancy = np.asarray(occupancy)
    if occupancy.shape != (geometry.height, geometry.width):
        raise ValueError("occupancy and geometry shapes do not match")
    blocked = occupancy != GridClass.FREE
    angles = heading_angles(config.heading_bins)
    heading_step = 2.0 * np.pi / config.heading_bins
    corner_radius = 0.5 * float(
        np.hypot(
            config.robot_footprint_length_m,
            config.robot_footprint_width_m,
        )
    )
    interval_count = max(
        1,
        int(
            np.ceil(
                heading_step * corner_radius
                / (0.5 * geometry.resolution_m)
            )
        ),
    )
    valid = np.empty(
        (config.heading_bins, geometry.height, geometry.width),
        dtype=bool,
    )
    for index, angle in enumerate(angles):
        swept_kernel = None
        for sample_angle in np.linspace(
            angle,
            angle + heading_step,
            interval_count + 1,
        ):
            kernel = _footprint_kernel(
                geometry,
                config.robot_footprint_length_m,
                config.robot_footprint_width_m,
                sample_angle,
            )
            if swept_kernel is None:
                swept_kernel = kernel
            else:
                swept_kernel |= kernel
        collision = ndimage.binary_dilation(
            blocked,
            structure=swept_kernel,
            border_value=1,
        )
        valid[index] = ~collision
    return valid


def trajectory_pose_validity(
    trajectory: np.ndarray,
    yaw_rad: np.ndarray,
    geometry: GridGeometry,
    valid_by_heading: np.ndarray,
) -> np.ndarray:
    """Check each optimized pose with the same footprint masks used by search."""
    columns, rows = geometry.world_to_grid(trajectory[:, :2])
    headings = yaw_to_heading_indices(yaw_rad, valid_by_heading.shape[0])
    inside = (
        (columns >= 0)
        & (columns < geometry.width)
        & (rows >= 0)
        & (rows < geometry.height)
    )
    valid = np.zeros(trajectory.shape[0], dtype=bool)
    valid[inside] = valid_by_heading[
        headings[inside], rows[inside], columns[inside]
    ]
    return valid


def exact_footprint_pose_validity(
    poses_xy_yaw: np.ndarray,
    occupancy: np.ndarray,
    geometry: GridGeometry,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Check each continuous pose without heading or grid-center quantization."""
    poses = np.asarray(poses_xy_yaw, dtype=np.float64)
    occupancy = np.asarray(occupancy)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses_xy_yaw must have shape (N, 3)")
    if not np.isfinite(poses).all():
        raise ValueError("poses_xy_yaw must contain finite values")
    if occupancy.shape != (geometry.height, geometry.width):
        raise ValueError("occupancy and geometry shapes do not match")
    if length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("footprint dimensions must be positive")

    blocked = occupancy != GridClass.FREE
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m
    valid = np.zeros(poses.shape[0], dtype=bool)
    for pose_index, (x_m, y_m, yaw_rad) in enumerate(poses):
        cosine = float(np.cos(yaw_rad))
        sine = float(np.sin(yaw_rad))
        extent_x = abs(cosine) * half_length + abs(sine) * half_width
        extent_y = abs(sine) * half_length + abs(cosine) * half_width
        if (
            x_m - extent_x < geometry.origin_x_m
            or x_m + extent_x > geometry.max_x_m
            or y_m - extent_y < geometry.origin_y_m
            or y_m + extent_y > geometry.max_y_m
        ):
            continue
        rows, columns = oriented_footprint_indices(
            geometry,
            np.asarray([x_m, y_m]),
            yaw_rad,
            half_length,
            half_width,
        )
        valid[pose_index] = rows.size > 0 and not np.any(blocked[rows, columns])
    return valid


def densify_pose_path_for_footprint(
    poses_xy_yaw: np.ndarray,
    resolution_m: float,
    length_m: float,
    width_m: float,
) -> np.ndarray:
    """Sample translation and corner rotation at no more than half a grid cell."""
    poses = np.asarray(poses_xy_yaw, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0:
        raise ValueError("poses_xy_yaw must have nonempty shape (N, 3)")
    if not np.isfinite(poses).all():
        raise ValueError("poses_xy_yaw must contain finite values")
    if resolution_m <= 0.0 or length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("resolution and footprint dimensions must be positive")

    corner_radius = 0.5 * float(np.hypot(length_m, width_m))
    max_motion_m = 0.5 * resolution_m
    samples = [poses[0]]
    for start, end in zip(poses[:-1], poses[1:]):
        delta_xy = end[:2] - start[:2]
        delta_yaw = float(
            np.arctan2(
                np.sin(end[2] - start[2]),
                np.cos(end[2] - start[2]),
            )
        )
        swept_motion_m = max(
            float(np.linalg.norm(delta_xy)),
            abs(delta_yaw) * corner_radius,
        )
        interval_count = max(1, int(np.ceil(swept_motion_m / max_motion_m)))
        for fraction in np.linspace(0.0, 1.0, interval_count + 1)[1:]:
            sample = np.empty(3, dtype=np.float64)
            sample[:2] = start[:2] + fraction * delta_xy
            sample[2] = start[2] + fraction * delta_yaw
            samples.append(sample)
    return np.asarray(samples, dtype=np.float64)


def _select_farthest_trajectory_pair(
    trajectory: np.ndarray,
    yaw_rad: np.ndarray,
    geometry: GridGeometry,
    valid_by_heading: np.ndarray,
) -> Tuple[int, int]:
    pose_valid = trajectory_pose_validity(
        trajectory,
        yaw_rad,
        geometry,
        valid_by_heading,
    )
    columns, rows = geometry.world_to_grid(trajectory[:, :2])
    union_valid = np.any(valid_by_heading, axis=0)
    labels, _ = ndimage.label(union_valid, structure=np.ones((3, 3)))
    pose_labels = np.zeros(trajectory.shape[0], dtype=np.int32)
    pose_labels[pose_valid] = labels[rows[pose_valid], columns[pose_valid]]
    nonzero = pose_labels[pose_labels > 0]
    if nonzero.size < 2:
        raise RuntimeError("fewer than two trajectory poses fit the planning footprint")
    counts = np.bincount(nonzero)
    component = int(np.argmax(counts))
    indices = np.flatnonzero(pose_labels == component)
    if indices.size < 2:
        raise RuntimeError("no connected trajectory pose pair is available")
    seed = indices[0]
    first = indices[
        np.argmax(np.linalg.norm(trajectory[indices, :2] - trajectory[seed, :2], axis=1))
    ]
    second = indices[
        np.argmax(np.linalg.norm(trajectory[indices, :2] - trajectory[first, :2], axis=1))
    ]
    if first == second:
        raise RuntimeError("selected trajectory route has zero spatial extent")
    return int(first), int(second)


def _footprint_connectivity_metrics(
    trajectory: np.ndarray,
    geometry: GridGeometry,
    valid_by_heading: np.ndarray,
    trajectory_valid: np.ndarray,
    start_index: int,
) -> Dict[str, object]:
    """Summarize where footprint erosion disconnects the recorded route."""
    columns, rows = geometry.world_to_grid(trajectory[:, :2])
    union_valid = np.any(valid_by_heading, axis=0)
    labels, component_count = ndimage.label(
        union_valid,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    inside = (
        (columns >= 0)
        & (columns < geometry.width)
        & (rows >= 0)
        & (rows < geometry.height)
    )
    pose_labels = np.zeros(trajectory.shape[0], dtype=np.int32)
    pose_labels[inside] = labels[rows[inside], columns[inside]]
    start_component = int(pose_labels[start_index])
    in_start_component = pose_labels == start_component
    component_sizes = np.bincount(labels.ravel())
    nonzero_sizes = component_sizes[1:]

    invalid = ~trajectory_valid
    padded = np.concatenate([[False], invalid, [False]]).astype(np.int8)
    transitions = np.diff(padded)
    run_starts = np.flatnonzero(transitions == 1)
    run_ends = np.flatnonzero(transitions == -1)
    invalid_runs = [
        {
            "start_index": int(start),
            "end_index": int(end - 1),
            "pose_count": int(end - start),
        }
        for start, end in zip(run_starts, run_ends)
    ]
    return {
        "footprint_center_component_count": int(component_count),
        "largest_footprint_center_component_area_m2": (
            float(np.max(nonzero_sizes)) * geometry.resolution_m ** 2
            if nonzero_sizes.size
            else 0.0
        ),
        "selected_component_trajectory_pose_fraction": float(
            np.mean(in_start_component)
        ),
        "selected_component_lattice_valid_pose_fraction": float(
            np.mean(in_start_component & trajectory_valid)
        ),
        "trajectory_lattice_invalid_run_count": len(invalid_runs),
        "trajectory_lattice_invalid_runs": invalid_runs,
    }


def _motion_offsets(
    heading_index: int,
    config: PathPlanningConfig,
    resolution_m: float,
) -> Tuple[int, int]:
    step_cells = max(1, int(round(config.translation_step_m / resolution_m)))
    angle = heading_angles(config.heading_bins)[heading_index]
    delta_column = int(round(np.cos(angle) * step_cells))
    delta_row = int(round(np.sin(angle) * step_cells))
    if delta_column == 0 and delta_row == 0:
        raise RuntimeError("translation_step_m is too small for the map resolution")
    return delta_row, delta_column


def _translation_is_valid(
    row: int,
    column: int,
    delta_row: int,
    delta_column: int,
    heading: int,
    valid_by_heading: np.ndarray,
) -> bool:
    step_count = max(abs(delta_row), abs(delta_column))
    for step in range(1, step_count + 1):
        next_row = row + int(round(delta_row * step / step_count))
        next_column = column + int(round(delta_column * step / step_count))
        if (
            next_row < 0
            or next_row >= valid_by_heading.shape[1]
            or next_column < 0
            or next_column >= valid_by_heading.shape[2]
            or not valid_by_heading[heading, next_row, next_column]
        ):
            return False
    return True


def _astar(
    start: Tuple[int, int, int],
    goal_row: int,
    goal_column: int,
    valid_by_heading: np.ndarray,
    turn_sweep_valid_ccw: np.ndarray,
    clearance_m: np.ndarray,
    geometry: GridGeometry,
    config: PathPlanningConfig,
) -> Tuple[List[Tuple[int, int, int]], int]:
    goal_tolerance_cells = config.goal_tolerance_m / geometry.resolution_m

    def heuristic(row: int, column: int) -> float:
        return float(np.hypot(row - goal_row, column - goal_column)) * geometry.resolution_m

    queue = [(heuristic(start[0], start[1]), 0.0, start)]
    g_score = {start: 0.0}
    parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    closed = set()
    expansions = 0
    goal_state = None
    while queue and expansions < config.max_expansions:
        _, current_cost, state = heapq.heappop(queue)
        if state in closed or current_cost > g_score.get(state, np.inf) + 1.0e-12:
            continue
        closed.add(state)
        expansions += 1
        row, column, heading = state
        if np.hypot(row - goal_row, column - goal_column) <= goal_tolerance_cells:
            goal_state = state
            break

        for heading_delta in (-1, 1):
            next_heading = (heading + heading_delta) % config.heading_bins
            next_state = (row, column, next_heading)
            turn_interval = heading if heading_delta > 0 else next_heading
            if (
                not valid_by_heading[next_heading, row, column]
                or not turn_sweep_valid_ccw[turn_interval, row, column]
            ):
                continue
            next_cost = current_cost + config.turn_cost_m
            if next_cost < g_score.get(next_state, np.inf):
                g_score[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic(row, column),
                        next_cost,
                        next_state,
                    ),
                )

        delta_row, delta_column = _motion_offsets(
            heading,
            config,
            geometry.resolution_m,
        )
        directions = [(delta_row, delta_column, 1.0)]
        if config.allow_reverse:
            directions.append(
                (
                    -delta_row,
                    -delta_column,
                    config.reverse_cost_multiplier,
                )
            )
        for move_row, move_column, multiplier in directions:
            if not _translation_is_valid(
                row,
                column,
                move_row,
                move_column,
                heading,
                valid_by_heading,
            ):
                continue
            next_row = row + move_row
            next_column = column + move_column
            next_state = (next_row, next_column, heading)
            distance_m = float(np.hypot(move_row, move_column)) * geometry.resolution_m
            clearance_penalty = config.clearance_cost_weight / max(
                clearance_m[next_row, next_column],
                geometry.resolution_m,
            )
            next_cost = current_cost + multiplier * distance_m + clearance_penalty
            if next_cost < g_score.get(next_state, np.inf):
                g_score[next_state] = next_cost
                parent[next_state] = state
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic(next_row, next_column),
                        next_cost,
                        next_state,
                    ),
                )

    if goal_state is None:
        raise RuntimeError(
            "footprint-aware A* did not reach the goal "
            f"within {config.max_expansions} expansions"
        )
    path = [goal_state]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return path, expansions


def plan_farthest_trajectory_route(
    occupancy: np.ndarray,
    geometry: GridGeometry,
    trajectory: np.ndarray,
    trajectory_yaw_rad: np.ndarray,
    config: PathPlanningConfig,
) -> PathPlan:
    """Plan between the farthest footprint-valid trajectory poses."""
    valid_by_heading, clearance_m = build_footprint_validity(
        occupancy,
        geometry,
        config,
    )
    turn_sweep_valid_ccw = build_turn_sweep_validity(
        occupancy,
        geometry,
        config,
    )
    start_index, goal_index = _select_farthest_trajectory_pair(
        trajectory,
        trajectory_yaw_rad,
        geometry,
        valid_by_heading,
    )
    columns, rows = geometry.world_to_grid(trajectory[:, :2])
    headings = yaw_to_heading_indices(trajectory_yaw_rad, config.heading_bins)
    start = (
        int(rows[start_index]),
        int(columns[start_index]),
        int(headings[start_index]),
    )
    path, expansions = _astar(
        start,
        int(rows[goal_index]),
        int(columns[goal_index]),
        valid_by_heading,
        turn_sweep_valid_ccw,
        clearance_m,
        geometry,
        config,
    )
    states = np.asarray(path, dtype=np.int32)
    angles = heading_angles(config.heading_bins)
    poses = np.column_stack(
        [
            geometry.origin_x_m
            + (states[:, 1].astype(np.float64) + 0.5) * geometry.resolution_m,
            geometry.origin_y_m
            + (states[:, 0].astype(np.float64) + 0.5) * geometry.resolution_m,
            angles[states[:, 2]],
        ]
    )
    path_valid = valid_by_heading[
        states[:, 2],
        states[:, 0],
        states[:, 1],
    ]
    trajectory_valid = trajectory_pose_validity(
        trajectory,
        trajectory_yaw_rad,
        geometry,
        valid_by_heading,
    )
    exact_trajectory_valid = exact_footprint_pose_validity(
        np.column_stack([trajectory[:, :2], trajectory_yaw_rad]),
        occupancy,
        geometry,
        config.robot_footprint_length_m,
        config.robot_footprint_width_m,
    )
    exact_path_valid = exact_footprint_pose_validity(
        poses,
        occupancy,
        geometry,
        config.robot_footprint_length_m,
        config.robot_footprint_width_m,
    )
    path_swept_poses = densify_pose_path_for_footprint(
        poses,
        geometry.resolution_m,
        config.robot_footprint_length_m,
        config.robot_footprint_width_m,
    )
    path_swept_exact_valid = exact_footprint_pose_validity(
        path_swept_poses,
        occupancy,
        geometry,
        config.robot_footprint_length_m,
        config.robot_footprint_width_m,
    )
    swept_collision_count = int(np.count_nonzero(~path_swept_exact_valid))
    if swept_collision_count:
        raise RuntimeError(
            "motion-lattice path has "
            f"{swept_collision_count} exact swept-footprint collision samples"
        )
    path_length_m = float(
        np.sum(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1))
    )
    path_clearance = clearance_m[states[:, 0], states[:, 1]]
    connectivity_metrics = _footprint_connectivity_metrics(
        trajectory,
        geometry,
        valid_by_heading,
        trajectory_valid,
        start_index,
    )
    metrics = {
        "start_trajectory_index": start_index,
        "goal_trajectory_index": goal_index,
        "start_xy_m": trajectory[start_index, :2].tolist(),
        "goal_xy_m": trajectory[goal_index, :2].tolist(),
        "straight_line_distance_m": float(
            np.linalg.norm(
                trajectory[goal_index, :2] - trajectory[start_index, :2]
            )
        ),
        "path_length_m": path_length_m,
        "path_pose_count": int(states.shape[0]),
        "path_collision_pose_count": int(np.count_nonzero(~path_valid)),
        "path_minimum_blocked_cell_clearance_m": float(np.min(path_clearance)),
        "path_median_blocked_cell_clearance_m": float(np.median(path_clearance)),
        "trajectory_pose_count": int(trajectory.shape[0]),
        "trajectory_exact_footprint_valid_fraction": float(
            np.mean(exact_trajectory_valid)
        ),
        "trajectory_exact_footprint_invalid_poses": int(
            np.count_nonzero(~exact_trajectory_valid)
        ),
        "trajectory_lattice_footprint_valid_fraction": float(
            np.mean(trajectory_valid)
        ),
        "trajectory_lattice_footprint_invalid_poses": int(
            np.count_nonzero(~trajectory_valid)
        ),
        "path_exact_footprint_collision_pose_count": int(
            np.count_nonzero(~exact_path_valid)
        ),
        "path_exact_swept_sample_count": int(path_swept_poses.shape[0]),
        "path_exact_swept_collision_sample_count": swept_collision_count,
        "unknown_is_blocked": True,
        "occupied_is_blocked": True,
        "heading_bins": config.heading_bins,
        "turn_primitive_checks_swept_footprint": True,
        "postcheck_max_corner_motion_m": 0.5 * geometry.resolution_m,
        "expansions": expansions,
        **connectivity_metrics,
    }
    return PathPlan(
        states=states,
        poses_xy_yaw=poses,
        start_trajectory_index=start_index,
        goal_trajectory_index=goal_index,
        expansions=expansions,
        trajectory_exact_valid=exact_trajectory_valid,
        trajectory_lattice_valid=trajectory_valid,
        path_exact_valid=exact_path_valid,
        path_swept_poses_xy_yaw=path_swept_poses,
        path_swept_exact_valid=path_swept_exact_valid,
        metrics=metrics,
    )
