import csv

import pytest

import analyze_revisit_consistency as analysis


FIELDS = [
    "stamp_ns",
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
]


def write_trajectory(path):
    rows = [
        [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 179.0],
        [
            100_000_000_000,
            25.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.5,
            2.5,
            90.0,
        ],
        [
            400_000_000_000,
            0.1,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
            -179.0,
        ],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def test_long_separated_revisit_reports_full_pose_difference(tmp_path):
    trajectory = tmp_path / "trajectory.csv"
    write_trajectory(trajectory)

    poses = analysis.load_trajectory(trajectory)
    pairs = analysis.nearest_revisit_pairs(
        poses, min_time_separation_s=300.0, min_path_separation_m=20.0
    )
    summary = analysis.summarize(poses, pairs, (0.25, 0.5))

    assert len(pairs) == 1
    assert pairs[0]["historical_index"] == 0
    assert pairs[0]["xy_distance_m"] == pytest.approx(0.1)
    assert pairs[0]["z_difference_m"] == pytest.approx(0.3)
    assert pairs[0]["yaw_difference_deg"] == pytest.approx(2.0)
    assert pairs[0]["body_up_difference_deg"] == pytest.approx(0.0)
    assert summary["xy_gate_metrics"]["0.25"]["sample_count"] == 1
    assert summary["xy_gate_metrics"]["0.25"]["abs_z_difference_m"][
        "median"
    ] == pytest.approx(0.3)


def test_pair_manifest_keeps_same_physical_timestamp_pair(tmp_path):
    trajectory = tmp_path / "trajectory.csv"
    write_trajectory(trajectory)
    poses = analysis.load_trajectory(trajectory)
    manifest = [
        {
            "current_stamp_ns": 400_000_000_000,
            "historical_stamp_ns": 0,
        }
    ]

    pairs = analysis.evaluate_pair_manifest(
        poses, manifest, max_timestamp_delta_s=0.01
    )

    assert len(pairs) == 1
    assert pairs[0]["current_index"] == 2
    assert pairs[0]["historical_index"] == 0
    assert pairs[0]["current_stamp_offset_s"] == pytest.approx(0.0)
    assert pairs[0]["abs_z_difference_m"] == pytest.approx(0.3)


def test_pair_manifest_rejects_unmatched_timestamps(tmp_path):
    trajectory = tmp_path / "trajectory.csv"
    write_trajectory(trajectory)
    poses = analysis.load_trajectory(trajectory)
    manifest = [
        {
            "current_stamp_ns": 410_000_000_000,
            "historical_stamp_ns": 0,
        }
    ]

    assert not analysis.evaluate_pair_manifest(
        poses, manifest, max_timestamp_delta_s=0.2
    )
