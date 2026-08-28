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

"""Read optimized SLAM trajectories used as local floor and free-space anchors."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TrajectoryData:
    """Optimized poses plus ordering metadata needed for safe interpolation."""

    poses: np.ndarray
    orientation_xyzw: Optional[np.ndarray]
    stamp_ns: Optional[np.ndarray]
    frame_id: Optional[str]
    yaw_source: str


def _normalized_quaternion(row: dict) -> np.ndarray:
    quaternion = np.asarray(
        [float(row[name]) for name in ("qx", "qy", "qz", "qw")],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm <= 1.0e-12:
        raise ValueError("trajectory contains an invalid quaternion")
    return quaternion / norm


def _quaternion_yaw(quaternion: np.ndarray) -> float:
    qx, qy, qz, qw = quaternion
    return float(
        np.arctan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
    )


def read_trajectory_data(path: Path) -> TrajectoryData:
    """Read, validate, and timestamp-sort one optimized trajectory CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"x_m", "y_m", "z_m"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("trajectory CSV must contain x_m, y_m, and z_m")
        has_yaw = "yaw_deg" in reader.fieldnames
        quaternion_fields = {"qx", "qy", "qz", "qw"}
        has_quaternion = quaternion_fields.issubset(reader.fieldnames)
        include_yaw = has_yaw or has_quaternion
        has_stamp = "stamp_ns" in reader.fieldnames
        has_frame = "frame_id" in reader.fieldnames
        rows = []
        stamps = []
        frames = []
        yaw_sources = []
        orientations = []
        for row in reader:
            quaternion = (
                _normalized_quaternion(row) if has_quaternion else None
            )
            if has_yaw and row["yaw_deg"] != "":
                yaw_rad = np.deg2rad(float(row["yaw_deg"]))
                yaw_sources.append("yaw_deg")
            elif quaternion is not None:
                yaw_rad = _quaternion_yaw(quaternion)
                yaw_sources.append("quaternion")
            elif has_yaw:
                raise ValueError("trajectory contains a missing yaw_deg value")
            else:
                yaw_rad = None
            position = (
                float(row["x_m"]),
                float(row["y_m"]),
                float(row["z_m"]),
            )
            rows.append(position + (yaw_rad,) if include_yaw else position)
            if quaternion is not None:
                orientations.append(quaternion)
            if has_stamp:
                if row["stamp_ns"] == "":
                    raise ValueError("trajectory contains a missing stamp_ns value")
                stamps.append(int(row["stamp_ns"]))
            if has_frame:
                frames.append(row["frame_id"].strip())
    trajectory = np.asarray(rows, dtype=np.float64)
    expected_columns = 4 if include_yaw else 3
    if (
        trajectory.ndim != 2
        or trajectory.shape[0] < 2
        or trajectory.shape[1] != expected_columns
    ):
        raise ValueError("trajectory must contain at least two XYZ poses")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory contains non-finite pose values")
    stamp_array: Optional[np.ndarray] = None
    orientation_array: Optional[np.ndarray] = None
    if has_quaternion:
        orientation_array = np.asarray(orientations, dtype=np.float64)
    if has_stamp:
        stamp_array = np.asarray(stamps, dtype=np.int64)
        order = np.argsort(stamp_array, kind="stable")
        stamp_array = stamp_array[order]
        trajectory = trajectory[order]
        if orientation_array is not None:
            orientation_array = orientation_array[order]
        if np.any(np.diff(stamp_array) <= 0):
            raise ValueError("trajectory stamp_ns values must be unique")

    frame_id: Optional[str] = None
    if has_frame:
        nonempty_frames = {frame for frame in frames if frame}
        if len(nonempty_frames) > 1:
            raise ValueError("trajectory contains multiple frame_id values")
        if nonempty_frames:
            frame_id = next(iter(nonempty_frames))

    if not include_yaw:
        yaw_source = "inferred_xy_tangent"
    elif len(set(yaw_sources)) == 1:
        yaw_source = yaw_sources[0]
    else:
        yaw_source = "mixed_yaw_and_quaternion"
    return TrajectoryData(
        poses=trajectory,
        orientation_xyzw=orientation_array,
        stamp_ns=stamp_array,
        frame_id=frame_id,
        yaw_source=yaw_source,
    )


def read_trajectory_poses(path: Path) -> np.ndarray:
    """Read XYZ and optional optimized yaw from a trajectory CSV."""
    return read_trajectory_data(path).poses


def read_trajectory_xyz(path: Path) -> np.ndarray:
    """Read only XYZ for callers that do not consume optimized orientation."""
    return read_trajectory_data(path).poses[:, :3]
