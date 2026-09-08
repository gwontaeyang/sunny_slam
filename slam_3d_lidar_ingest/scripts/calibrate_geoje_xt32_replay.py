#!/usr/bin/env python3
"""Fit deterministic XT32 and JT128-IMU clocks to Geoje bag record time."""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2, PointField


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cloud-topic",
        default="/hesai/center/lidar_points",
    )
    parser.add_argument(
        "--imu-topic",
        default="/hesai/front/lidar_imu",
    )
    parser.add_argument("--cloud-samples", type=int, default=80)
    parser.add_argument("--imu-samples", type=int, default=400)
    parser.add_argument("--outlier-sigma", type=float, default=6.0)
    return parser.parse_args()


def database_path(bag):
    candidates = sorted(bag.glob("*.db3"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one db3 in {bag}, found {len(candidates)}"
        )
    return candidates[0]


def topic_id(connection, topic):
    row = connection.execute(
        "SELECT id FROM topics WHERE name = ?",
        (topic,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"bag is missing topic: {topic}")
    return int(row[0])


def sampled_messages(connection, topic, sample_count):
    identifier = topic_id(connection, topic)
    rows = connection.execute(
        "SELECT id, timestamp FROM messages "
        "WHERE topic_id = ? ORDER BY timestamp",
        (identifier,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"topic has no messages: {topic}")
    indices = np.unique(
        np.linspace(
            0,
            len(rows) - 1,
            min(max(2, sample_count), len(rows)),
            dtype=int,
        )
    )
    samples = []
    for index in indices:
        message_id, record_ns = rows[int(index)]
        payload = connection.execute(
            "SELECT data FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()[0]
        samples.append((int(record_ns), bytes(payload)))
    return samples


def stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def point_timestamps(message):
    field = next(
        (candidate for candidate in message.fields if candidate.name == "timestamp"),
        None,
    )
    if field is None or field.datatype != PointField.FLOAT64 or field.count != 1:
        raise RuntimeError("expected one FLOAT64 timestamp field")
    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["timestamp"],
            "formats": [byte_order + "f8"],
            "offsets": [field.offset],
            "itemsize": message.point_step,
        }
    )
    rows = []
    for row_index in range(message.height):
        start = row_index * message.row_step
        stop = start + message.width * message.point_step
        rows.append(
            np.frombuffer(
                message.data[start:stop],
                dtype=dtype,
                count=message.width,
            )["timestamp"]
        )
    return np.concatenate(rows)


def cloud_observations(samples):
    observations = []
    for record_ns, payload in samples:
        message = deserialize_message(payload, PointCloud2)
        times = point_timestamps(message)
        finite = times[np.isfinite(times)]
        if finite.size == 0:
            raise RuntimeError("point cloud has no finite timestamps")
        duration_ns = int(
            round((float(np.max(finite)) - float(np.min(finite))) * 1.0e9)
        )
        observations.append(
            {
                "record_ns": record_ns,
                "input_header_ns": stamp_ns(message),
                "target_ns": record_ns - duration_ns,
                "frame_id": message.header.frame_id,
                "scan_duration_ns": duration_ns,
            }
        )
    return observations


def imu_observations(samples):
    observations = []
    for record_ns, payload in samples:
        message = deserialize_message(payload, Imu)
        observations.append(
            {
                "record_ns": record_ns,
                "input_header_ns": stamp_ns(message),
                "target_ns": record_ns,
                "frame_id": message.header.frame_id,
            }
        )
    return observations


def robust_affine_fit(observations, outlier_sigma):
    input_ns = np.asarray(
        [item["input_header_ns"] for item in observations],
        dtype=np.int64,
    )
    target_ns = np.asarray(
        [item["target_ns"] for item in observations],
        dtype=np.int64,
    )
    if np.any(np.diff(input_ns) <= 0):
        raise RuntimeError("sampled input header timestamps are not increasing")
    anchor_input_ns = int(input_ns[0])
    target_origin_ns = int(target_ns[0])
    input_relative_s = (
        input_ns - anchor_input_ns
    ).astype(np.float64) * 1.0e-9
    target_relative_s = (
        target_ns - target_origin_ns
    ).astype(np.float64) * 1.0e-9
    mask = np.ones(len(observations), dtype=bool)
    for _ in range(8):
        design = np.column_stack(
            (input_relative_s[mask], np.ones(np.count_nonzero(mask)))
        )
        scale, intercept_s = np.linalg.lstsq(
            design,
            target_relative_s[mask],
            rcond=None,
        )[0]
        predicted_s = scale * input_relative_s + intercept_s
        residual_s = target_relative_s - predicted_s
        center = float(np.median(residual_s[mask]))
        mad = float(np.median(np.abs(residual_s[mask] - center)))
        robust_sigma = max(1.0e-6, 1.4826 * mad)
        next_mask = np.abs(residual_s - center) <= (
            max(1.0, outlier_sigma) * robust_sigma
        )
        if np.count_nonzero(next_mask) < max(2, len(observations) // 2):
            break
        if np.array_equal(mask, next_mask):
            mask = next_mask
            break
        mask = next_mask

    design = np.column_stack(
        (input_relative_s[mask], np.ones(np.count_nonzero(mask)))
    )
    scale, intercept_s = np.linalg.lstsq(
        design,
        target_relative_s[mask],
        rcond=None,
    )[0]
    initial_anchor_output_ns = int(
        round(target_origin_ns + float(intercept_s) * 1.0e9)
    )
    first_inlier_index = int(np.flatnonzero(mask)[0])
    last_inlier_index = int(np.flatnonzero(mask)[-1])
    calibrated_anchor_input_ns = int(input_ns[first_inlier_index])
    calibrated_anchor_output_ns = int(
        round(
            initial_anchor_output_ns
            + scale * (calibrated_anchor_input_ns - anchor_input_ns)
        )
    )
    predicted_ns = calibrated_anchor_output_ns + scale * (
        input_ns - calibrated_anchor_input_ns
    )
    residual_s = (target_ns - predicted_ns) * 1.0e-9
    inlier_residual_s = residual_s[mask]
    return {
        "sample_count": len(observations),
        "inlier_count": int(np.count_nonzero(mask)),
        "outlier_count": int(np.count_nonzero(~mask)),
        "time_scale": float(scale),
        "rate_error_ppm": float((scale - 1.0) * 1.0e6),
        "anchor_input_header_ns": calibrated_anchor_input_ns,
        "anchor_output_header_ns": calibrated_anchor_output_ns,
        "calibration_valid_min_input_header_ns": int(np.min(input_ns[mask])),
        "calibration_valid_max_input_header_ns": int(np.max(input_ns[mask])),
        "sampled_min_input_header_ns": int(np.min(input_ns)),
        "sampled_max_input_header_ns": int(np.max(input_ns)),
        "inlier_residual_rmse_s": float(
            np.sqrt(np.mean(inlier_residual_s**2))
        ),
        "inlier_residual_p95_abs_s": float(
            np.percentile(np.abs(inlier_residual_s), 95.0)
        ),
        "inlier_residual_max_abs_s": float(
            np.max(np.abs(inlier_residual_s))
        ),
        "all_residual_max_abs_s": float(np.max(np.abs(residual_s))),
        "first_observation": observations[0],
        "last_observation": observations[-1],
        "first_inlier_observation": observations[first_inlier_index],
        "last_inlier_observation": observations[last_inlier_index],
    }


def main():
    args = parse_args()
    database = database_path(args.bag)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    cloud = cloud_observations(
        sampled_messages(
            connection,
            args.cloud_topic,
            args.cloud_samples,
        )
    )
    imu = imu_observations(
        sampled_messages(
            connection,
            args.imu_topic,
            args.imu_samples,
        )
    )
    connection.close()
    result = {
        "bag": str(args.bag),
        "database": str(database),
        "scope": "offline_replay_salvage_only_not_ptp_calibration",
        "target_clock": "bag_record_time",
        "cloud_target": "bag_record_time_minus_observed_scan_duration",
        "imu_target": "bag_record_time",
        "cloud_topic": args.cloud_topic,
        "imu_topic": args.imu_topic,
        "cloud": robust_affine_fit(cloud, args.outlier_sigma),
        "imu": robust_affine_fit(imu, args.outlier_sigma),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
