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

"""Trajectory CSV orientation contract tests."""

import numpy as np
import pytest

from slam_3d_map_projection.trajectory_io import (
    read_trajectory_data,
    read_trajectory_poses,
)


def test_xyz_only_trajectory_remains_three_columns(tmp_path):
    path = tmp_path / "xyz.csv"
    path.write_text(
        "x_m,y_m,z_m\n0.0,0.0,0.1\n1.0,0.0,0.1\n",
        encoding="utf-8",
    )

    trajectory = read_trajectory_poses(path)

    assert trajectory.shape == (2, 3)
    np.testing.assert_allclose(trajectory[:, 0], [0.0, 1.0])


def test_yaw_degrees_are_converted_to_radians(tmp_path):
    path = tmp_path / "yaw.csv"
    path.write_text(
        "x_m,y_m,z_m,yaw_deg\n0.0,0.0,0.0,0.0\n1.0,0.0,0.0,90.0\n",
        encoding="utf-8",
    )

    trajectory = read_trajectory_poses(path)

    assert trajectory.shape == (2, 4)
    np.testing.assert_allclose(trajectory[:, 3], [0.0, np.pi / 2.0])


def test_non_unit_quaternion_is_normalized(tmp_path):
    path = tmp_path / "quaternion.csv"
    path.write_text(
        "x_m,y_m,z_m,qx,qy,qz,qw\n"
        "0.0,0.0,0.0,0.0,0.0,0.0,2.0\n"
        "1.0,0.0,0.0,0.0,0.0,1.414213562,1.414213562\n",
        encoding="utf-8",
    )

    data = read_trajectory_data(path)

    np.testing.assert_allclose(data.poses[:, 3], [0.0, np.pi / 2.0], atol=1.0e-7)
    np.testing.assert_allclose(
        np.linalg.norm(data.orientation_xyzw, axis=1),
        [1.0, 1.0],
        atol=1.0e-7,
    )


def test_missing_yaw_without_quaternion_is_rejected(tmp_path):
    path = tmp_path / "missing_yaw.csv"
    path.write_text(
        "x_m,y_m,z_m,yaw_deg\n0.0,0.0,0.0,0.0\n1.0,0.0,0.0,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing yaw_deg"):
        read_trajectory_poses(path)


def test_timestamped_trajectory_is_sorted_and_preserves_metadata(tmp_path):
    path = tmp_path / "timestamped.csv"
    path.write_text(
        "stamp_ns,frame_id,x_m,y_m,z_m,yaw_deg\n"
        "300,map,3.0,0.0,0.0,30.0\n"
        "100,map,1.0,0.0,0.0,10.0\n"
        "200,map,2.0,0.0,0.0,20.0\n",
        encoding="utf-8",
    )

    data = read_trajectory_data(path)

    np.testing.assert_array_equal(data.stamp_ns, [100, 200, 300])
    np.testing.assert_allclose(data.poses[:, 0], [1.0, 2.0, 3.0])
    assert data.orientation_xyzw is None
    assert data.frame_id == "map"
    assert data.yaw_source == "yaw_deg"


def test_timestamp_sort_also_reorders_full_orientation(tmp_path):
    path = tmp_path / "timestamped_quaternion.csv"
    path.write_text(
        "stamp_ns,x_m,y_m,z_m,qx,qy,qz,qw\n"
        "200,2.0,0.0,0.0,0.0,0.0,1.0,0.0\n"
        "100,1.0,0.0,0.0,0.0,0.0,0.0,1.0\n",
        encoding="utf-8",
    )

    data = read_trajectory_data(path)

    np.testing.assert_array_equal(data.stamp_ns, [100, 200])
    np.testing.assert_allclose(data.orientation_xyzw[0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(data.orientation_xyzw[1], [0.0, 0.0, 1.0, 0.0])


def test_duplicate_timestamps_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "stamp_ns,x_m,y_m,z_m\n100,0.0,0.0,0.0\n100,1.0,0.0,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unique"):
        read_trajectory_data(path)


def test_mixed_frames_are_rejected(tmp_path):
    path = tmp_path / "mixed_frame.csv"
    path.write_text(
        "frame_id,x_m,y_m,z_m\nmap,0.0,0.0,0.0\nodom,1.0,0.0,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple frame_id"):
        read_trajectory_data(path)
