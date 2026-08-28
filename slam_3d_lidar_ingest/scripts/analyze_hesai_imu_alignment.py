#!/usr/bin/env python3
"""Validate Hesai built-in IMU units and axes against wheel yaw rate."""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import matplotlib
import numpy as np
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


SENSOR_TO_BASE_QUATERNIONS = {
    "/hesai/front/lidar_imu": (
        0.9255365,
        -0.3786584,
        0.0000034,
        0.00000139,
    ),
    "/hesai/rear/lidar_imu": (
        0.3866995,
        0.9222058,
        0.00000142,
        -0.00000339,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-plot", required=True, type=Path)
    parser.add_argument("--odom-topic", default="/motor_driver_node/odom")
    parser.add_argument(
        "--imu-topics",
        nargs="+",
        default=list(SENSOR_TO_BASE_QUATERNIONS),
    )
    parser.add_argument("--imu-stride", type=int, default=20)
    parser.add_argument("--minimum-wheel-yaw-rate", type=float, default=0.015)
    return parser.parse_args()


def database_path(bag):
    candidates = sorted(bag.glob("*.db3"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one db3 in {bag}, found {len(candidates)}"
        )
    return candidates[0]


def quaternion_rotation(quaternion):
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 1.0e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    )


def topic_id(connection, topic):
    row = connection.execute(
        "SELECT id FROM topics WHERE name = ?",
        (topic,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"bag is missing topic: {topic}")
    return int(row[0])


def read_wheel(connection, topic):
    records = []
    query = (
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id = ? ORDER BY timestamp"
    )
    for record_ns, payload in connection.execute(
        query,
        (topic_id(connection, topic),),
    ):
        message = deserialize_message(bytes(payload), Odometry)
        records.append(
            (
                int(record_ns),
                float(message.twist.twist.angular.z),
            )
        )
    if len(records) < 2:
        raise RuntimeError("wheel odometry has fewer than two records")
    return np.asarray(records, dtype=float)


def read_imu(connection, topic, stride):
    rotation = quaternion_rotation(SENSOR_TO_BASE_QUATERNIONS[topic])
    records = []
    query = (
        "SELECT timestamp, data FROM messages "
        "WHERE topic_id = ? AND (id % ?) = 0 ORDER BY timestamp"
    )
    for record_ns, payload in connection.execute(
        query,
        (topic_id(connection, topic), max(1, stride)),
    ):
        message = deserialize_message(bytes(payload), Imu)
        sensor_gyro_deg_s = np.asarray(
            [
                message.angular_velocity.x,
                message.angular_velocity.y,
                message.angular_velocity.z,
            ],
            dtype=float,
        )
        base_gyro_rad_s = (
            rotation @ sensor_gyro_deg_s
        ) * (math.pi / 180.0)
        records.append((int(record_ns), *base_gyro_rad_s))
    if len(records) < 2:
        raise RuntimeError(f"IMU topic has fewer than two sampled records: {topic}")
    return np.asarray(records, dtype=float), rotation


def comparison_metrics(imu, wheel, minimum_wheel_yaw_rate):
    wheel_time = wheel[:, 0]
    wheel_yaw = wheel[:, 1]
    imu_time = imu[:, 0]
    common = (imu_time >= wheel_time[0]) & (imu_time <= wheel_time[-1])
    imu = imu[common]
    imu_time = imu[:, 0]
    matched_wheel = np.interp(imu_time, wheel_time, wheel_yaw)
    imu_yaw = imu[:, 3]
    finite = np.isfinite(matched_wheel) & np.isfinite(imu_yaw)
    moving = finite & (np.abs(matched_wheel) >= minimum_wheel_yaw_rate)
    if np.count_nonzero(moving) < 20:
        moving = finite
    wheel_selected = matched_wheel[moving]
    imu_selected = imu_yaw[moving]
    design = np.column_stack(
        (wheel_selected, np.ones_like(wheel_selected))
    )
    slope, intercept = np.linalg.lstsq(
        design,
        imu_selected,
        rcond=None,
    )[0]
    correlation = float(np.corrcoef(wheel_selected, imu_selected)[0, 1])
    residual = imu_selected - wheel_selected
    return {
        "sample_count": int(len(imu)),
        "moving_sample_count": int(np.count_nonzero(moving)),
        "pearson_correlation": correlation,
        "fit_imu_equals_slope_wheel_plus_intercept": {
            "slope": float(slope),
            "intercept_rad_s": float(intercept),
        },
        "direct_rmse_rad_s": float(np.sqrt(np.mean(residual**2))),
        "direct_p95_abs_rad_s": float(
            np.percentile(np.abs(residual), 95.0)
        ),
        "imu_yaw_rate_p95_abs_rad_s": float(
            np.percentile(np.abs(imu_selected), 95.0)
        ),
        "wheel_yaw_rate_p95_abs_rad_s": float(
            np.percentile(np.abs(wheel_selected), 95.0)
        ),
        "series": {
            "record_time_ns": imu_time.astype(np.int64),
            "imu_base_yaw_rate_rad_s": imu_yaw,
            "wheel_yaw_rate_rad_s": matched_wheel,
            "moving_mask": moving,
        },
    }


def write_plot(path, dataset, comparisons):
    figure, axes = plt.subplots(
        len(comparisons),
        2,
        figsize=(15, 5 * len(comparisons)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    figure.suptitle(f"{dataset}: Hesai built-in IMU vs wheel yaw rate")
    for row, (topic, result) in enumerate(comparisons.items()):
        series = result["series"]
        time_s = (
            series["record_time_ns"] - series["record_time_ns"][0]
        ) * 1.0e-9
        stride = max(1, int(math.ceil(len(time_s) / 5000)))
        axes[row, 0].plot(
            time_s[::stride],
            series["imu_base_yaw_rate_rad_s"][::stride],
            linewidth=0.8,
            label="Hesai IMU transformed to base",
        )
        axes[row, 0].plot(
            time_s[::stride],
            series["wheel_yaw_rate_rad_s"][::stride],
            linewidth=0.8,
            label="wheel odom twist",
        )
        axes[row, 0].set_title(topic)
        axes[row, 0].set_xlabel("bag record elapsed time [s]")
        axes[row, 0].set_ylabel("yaw rate [rad/s]")
        axes[row, 0].grid(True, linewidth=0.3)
        axes[row, 0].legend()

        moving = series["moving_mask"]
        axes[row, 1].scatter(
            series["wheel_yaw_rate_rad_s"][moving],
            series["imu_base_yaw_rate_rad_s"][moving],
            s=2.0,
            alpha=0.35,
        )
        bounds = np.asarray(axes[row, 1].get_xlim())
        axes[row, 1].plot(bounds, bounds, color="black", linewidth=0.8)
        axes[row, 1].set_title(
            "moving samples: "
            f"corr={result['pearson_correlation']:.3f}, "
            f"slope={result['fit_imu_equals_slope_wheel_plus_intercept']['slope']:.3f}"
        )
        axes[row, 1].set_xlabel("wheel yaw rate [rad/s]")
        axes[row, 1].set_ylabel("Hesai IMU yaw rate [rad/s]")
        axes[row, 1].set_aspect("equal", adjustable="box")
        axes[row, 1].grid(True, linewidth=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    database = database_path(args.bag)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    wheel = read_wheel(connection, args.odom_topic)
    comparisons = {}
    rotations = {}
    for topic in args.imu_topics:
        if topic not in SENSOR_TO_BASE_QUATERNIONS:
            raise ValueError(f"no captured extrinsic for {topic}")
        imu, rotation = read_imu(connection, topic, args.imu_stride)
        comparisons[topic] = comparison_metrics(
            imu,
            wheel,
            args.minimum_wheel_yaw_rate,
        )
        rotations[topic] = rotation.tolist()
    connection.close()

    write_plot(args.output_plot, args.bag.name, comparisons)
    serializable_comparisons = {}
    for topic, result in comparisons.items():
        serializable_comparisons[topic] = {
            key: value
            for key, value in result.items()
            if key != "series"
        }
    output = {
        "bag": str(args.bag),
        "database": str(database),
        "odom_topic": args.odom_topic,
        "imu_source_units": {
            "angular_velocity": "degree_per_second",
            "linear_acceleration": "g",
        },
        "normalization": {
            "angular_velocity_scale_to_rad_s": math.pi / 180.0,
            "linear_acceleration_scale_to_m_s2": 9.80665,
        },
        "sensor_to_base_rotations": rotations,
        "comparisons": serializable_comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
