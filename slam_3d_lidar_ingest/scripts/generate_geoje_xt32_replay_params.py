#!/usr/bin/env python3
"""Generate deterministic XT32 and Hesai IMU replay parameter files."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("clock_fit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-id", default="")
    parser.add_argument("--max-output-stamp-step-s", type=float, default=10.0)
    return parser.parse_args()


def validate_fit(name, fit):
    required = (
        "time_scale",
        "anchor_input_header_ns",
        "anchor_output_header_ns",
        "calibration_valid_min_input_header_ns",
        "calibration_valid_max_input_header_ns",
        "inlier_residual_rmse_s",
        "inlier_residual_max_abs_s",
    )
    missing = [key for key in required if key not in fit]
    if missing:
        raise RuntimeError(f"{name} clock fit is missing: {', '.join(missing)}")
    minimum = int(fit["calibration_valid_min_input_header_ns"])
    maximum = int(fit["calibration_valid_max_input_header_ns"])
    anchor = int(fit["anchor_input_header_ns"])
    if minimum <= 0 or maximum < minimum or not minimum <= anchor <= maximum:
        raise RuntimeError(f"{name} clock anchor is outside its valid range")
    scale = float(fit["time_scale"])
    if not 0.5 <= scale <= 2.0:
        raise RuntimeError(f"{name} clock scale is outside the supported range")


def fit_parameters(fit, calibration_id, maximum_step_s, output_anchor_key):
    return {
        "clock_domain": "legacy_bag_record_time_offline_only",
        "clock_policy": "calibrated_sensor_time",
        "time_offset_ns": 0,
        "time_scale": float(fit["time_scale"]),
        "anchor_input_header_ns": int(fit["anchor_input_header_ns"]),
        output_anchor_key: int(fit["anchor_output_header_ns"]),
        "calibration_valid_min_input_header_ns": int(
            fit["calibration_valid_min_input_header_ns"]
        ),
        "calibration_valid_max_input_header_ns": int(
            fit["calibration_valid_max_input_header_ns"]
        ),
        "max_output_stamp_step_s": maximum_step_s,
        "calibration_fit_rmse_ms": (
            float(fit["inlier_residual_rmse_s"]) * 1000.0
        ),
        "calibration_fit_max_abs_ms": (
            float(fit["inlier_residual_max_abs_s"]) * 1000.0
        ),
        "calibration_id": calibration_id,
    }


def pointcloud_parameters(cloud_fit, calibration_id, maximum_step_s):
    parameters = {
        "front.input_topic": "/unused/hesai/front/lidar_points",
        "front.output_topic": "/unused/slam_3d/input/hesai_front/points_timed",
        "front.expected_input_frame_id": "hesai_front",
        "front.require_expected_input_frame": True,
        "front.model": "JT128",
        "front.sensor_id": 0,
        "front.max_ring_id": 127,
        "front.min_points": 200000,
        "front.max_points": 260000,
        "front.clock_domain": "unused",
        "front.clock_policy": "sensor_header",
        "front.time_offset_ns": 0,
        "rear.input_topic": "/unused/hesai/rear/lidar_points",
        "rear.output_topic": "/unused/slam_3d/input/hesai_rear/points_timed",
        "rear.expected_input_frame_id": "hesai_rear",
        "rear.require_expected_input_frame": True,
        "rear.model": "JT128",
        "rear.sensor_id": 1,
        "rear.max_ring_id": 127,
        "rear.min_points": 200000,
        "rear.max_points": 260000,
        "rear.clock_domain": "unused",
        "rear.clock_policy": "sensor_header",
        "rear.time_offset_ns": 0,
        "center.input_topic": "/hesai/center/lidar_points",
        "center.output_topic": "/slam_3d/input/hesai_center/points_timed",
        "center.expected_input_frame_id": "hesai_center",
        "center.require_expected_input_frame": True,
        "center.model": "XT32",
        "center.sensor_id": 2,
        "center.max_ring_id": 31,
        "center.min_points": 100000,
        "center.max_points": 160000,
        "health_topic": "/slam_3d/ingest/health",
        "input_depth": 10,
        "output_depth": 5,
        "input_reliable": True,
        "output_reliable": True,
        "drop_nonfinite_points": True,
        "reject_non_increasing_output_stamp": True,
        "reject_clock_misalignment": True,
        "max_sensor_header_to_receive_delta_s": 0.25,
        "max_points": 500000,
        "max_scan_duration_s": 0.2,
        "min_finite_point_ratio": 0.9,
        "max_header_first_point_delta_s": 0.005,
        "timestamp_regression_tolerance_s": 1.0e-9,
        "require_monotonic_timestamps": True,
        "require_nonzero_duration": True,
        "require_header_matches_first_point": True,
    }
    calibrated = fit_parameters(
        cloud_fit,
        calibration_id,
        maximum_step_s,
        "anchor_output_start_ns",
    )
    parameters.update(
        {f"center.{key}": value for key, value in calibrated.items()}
    )
    return {"hesai_pointcloud_ingest": {"ros__parameters": parameters}}


def imu_parameters(imu_fit, calibration_id, maximum_step_s):
    parameters = {
        "input_topic": "/hesai/front/lidar_imu",
        "output_topic": "/slam_3d/input/hesai_front/imu",
        "health_topic": "/slam_3d/ingest/imu_health",
        "expected_input_frame_id": "hesai_front",
        "output_frame_id": "hesai_front",
        "require_expected_input_frame": True,
        "angular_velocity_scale": 0.017453292519943295,
        "linear_acceleration_scale": 9.80665,
        "input_depth": 1000,
        "output_depth": 1000,
        "input_reliable": True,
        "output_reliable": True,
        "reject_non_increasing_output_stamp": True,
        "diagnostic_period_s": 1.0,
    }
    parameters.update(
        fit_parameters(
            imu_fit,
            calibration_id,
            maximum_step_s,
            "anchor_output_header_ns",
        )
    )
    return {"hesai_imu_normalizer": {"ros__parameters": parameters}}


def write_yaml(path, value):
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if args.max_output_stamp_step_s <= 0.0:
        raise RuntimeError("max output stamp step must be positive")
    raw_fit = args.clock_fit.read_bytes()
    fit = json.loads(raw_fit)
    if fit.get("scope") != "offline_replay_salvage_only_not_ptp_calibration":
        raise RuntimeError("clock fit is not marked for offline replay salvage")
    validate_fit("cloud", fit["cloud"])
    validate_fit("imu", fit["imu"])

    fit_digest = hashlib.sha256(raw_fit).hexdigest()
    calibration_id = args.calibration_id or (
        f"geoje_xt32_front_imu_{fit_digest[:12]}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cloud_path = args.output_dir / "xt32_ingest_params.yaml"
    imu_path = args.output_dir / "front_lidar_imu_params.yaml"
    manifest_path = args.output_dir / "replay_clock_manifest.json"
    write_yaml(
        cloud_path,
        pointcloud_parameters(
            fit["cloud"],
            calibration_id,
            args.max_output_stamp_step_s,
        ),
    )
    write_yaml(
        imu_path,
        imu_parameters(
            fit["imu"],
            calibration_id,
            args.max_output_stamp_step_s,
        ),
    )
    manifest = {
        "dataset": fit["bag"],
        "scope": fit["scope"],
        "target_clock": fit["target_clock"],
        "calibration_id": calibration_id,
        "clock_fit_path": str(args.clock_fit),
        "clock_fit_sha256": fit_digest,
        "max_output_stamp_step_s": args.max_output_stamp_step_s,
        "pointcloud_params": str(cloud_path),
        "imu_params": str(imu_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
