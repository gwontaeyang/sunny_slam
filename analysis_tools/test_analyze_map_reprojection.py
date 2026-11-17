import numpy as np
import pytest

import analyze_map_reprojection as analysis


def write_xyz_pcd(path, points):
    points = np.asarray(points, dtype="<f4")
    header = (
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n"
        f"COUNT 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\nPOINTS {len(points)}\nDATA binary\n"
    ).encode("ascii")
    path.write_bytes(header + points.tobytes())


def test_binary_pcd_and_map_distance_metrics(tmp_path):
    before_path = tmp_path / "before.pcd"
    after_path = tmp_path / "after.pcd"
    before = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = before + np.asarray([0.1, 0.0, 0.0])
    write_xyz_pcd(before_path, before)
    write_xyz_pcd(after_path, after)

    loaded_before = analysis.read_xyz_pcd(before_path)
    loaded_after = analysis.read_xyz_pcd(after_path)
    summary, _, _, _ = analysis.compare_maps(loaded_before, loaded_after)

    assert loaded_before.shape == (2, 3)
    assert summary["centroid_shift_m"] == pytest.approx(0.1)
    assert summary["before_to_after_nearest_m"]["median"] == pytest.approx(0.1)
    assert summary["before_points_over_threshold_ratio"]["0.05"] == pytest.approx(1.0)
