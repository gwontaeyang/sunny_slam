import importlib.util
import json
from pathlib import Path

import yaml


CALIBRATION_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "calibrate_geoje_xt32_replay.py"
)
GENERATOR_SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "generate_geoje_xt32_replay_params.py"
)
SPEC = importlib.util.spec_from_file_location(
    "geoje_clock_calibration",
    CALIBRATION_SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "geoje_replay_parameter_generator",
    GENERATOR_SCRIPT,
)
GENERATOR = importlib.util.module_from_spec(GENERATOR_SPEC)
GENERATOR_SPEC.loader.exec_module(GENERATOR)


def test_robust_fit_reanchors_to_first_inlier():
    input_start_ns = 1_000_000_000
    output_start_ns = 20_000_000_000
    step_ns = 100_000_000
    scale = 0.9999
    observations = []
    for index in range(40):
        input_ns = input_start_ns + index * step_ns
        target_ns = int(
            round(output_start_ns + scale * (input_ns - input_start_ns))
        )
        if index == 0:
            target_ns += 2_000_000_000
        observations.append(
            {
                "input_header_ns": input_ns,
                "target_ns": target_ns,
            }
        )

    result = MODULE.robust_affine_fit(observations, outlier_sigma=6.0)

    assert result["outlier_count"] == 1
    assert result["anchor_input_header_ns"] == observations[1]["input_header_ns"]
    assert (
        result["calibration_valid_min_input_header_ns"]
        <= result["anchor_input_header_ns"]
        <= result["calibration_valid_max_input_header_ns"]
    )
    assert abs(result["time_scale"] - scale) < 1.0e-10
    assert result["inlier_residual_max_abs_s"] < 1.0e-8


def test_generated_params_preserve_clock_fit_and_units(tmp_path):
    fit = {
        "bag": "/data/geoje",
        "scope": "offline_replay_salvage_only_not_ptp_calibration",
        "target_clock": "bag_record_time",
        "cloud": {
            "time_scale": 0.9999,
            "anchor_input_header_ns": 1_000_000_000,
            "anchor_output_header_ns": 20_000_000_000,
            "calibration_valid_min_input_header_ns": 1_000_000_000,
            "calibration_valid_max_input_header_ns": 2_000_000_000,
            "inlier_residual_rmse_s": 0.004,
            "inlier_residual_max_abs_s": 0.012,
        },
        "imu": {
            "time_scale": 1.0001,
            "anchor_input_header_ns": 3_000_000_000,
            "anchor_output_header_ns": 20_100_000_000,
            "calibration_valid_min_input_header_ns": 3_000_000_000,
            "calibration_valid_max_input_header_ns": 4_000_000_000,
            "inlier_residual_rmse_s": 0.005,
            "inlier_residual_max_abs_s": 0.015,
        },
    }
    fit_path = tmp_path / "clock_fit.json"
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    cloud = GENERATOR.pointcloud_parameters(
        fit["cloud"],
        "test_calibration",
        10.0,
    )["hesai_pointcloud_ingest"]["ros__parameters"]
    imu = GENERATOR.imu_parameters(
        fit["imu"],
        "test_calibration",
        10.0,
    )["hesai_imu_normalizer"]["ros__parameters"]

    cloud_path = tmp_path / "cloud.yaml"
    imu_path = tmp_path / "imu.yaml"
    GENERATOR.write_yaml(
        cloud_path,
        {"hesai_pointcloud_ingest": {"ros__parameters": cloud}},
    )
    GENERATOR.write_yaml(
        imu_path,
        {"hesai_imu_normalizer": {"ros__parameters": imu}},
    )

    loaded_cloud = yaml.safe_load(cloud_path.read_text())[
        "hesai_pointcloud_ingest"
    ]["ros__parameters"]
    loaded_imu = yaml.safe_load(imu_path.read_text())[
        "hesai_imu_normalizer"
    ]["ros__parameters"]
    assert loaded_cloud["center.clock_policy"] == "calibrated_sensor_time"
    assert loaded_cloud["center.anchor_output_start_ns"] == 20_000_000_000
    assert loaded_cloud["center.max_output_stamp_step_s"] == 10.0
    assert loaded_imu["anchor_output_header_ns"] == 20_100_000_000
    assert loaded_imu["angular_velocity_scale"] == 0.017453292519943295
    assert loaded_imu["linear_acceleration_scale"] == 9.80665
