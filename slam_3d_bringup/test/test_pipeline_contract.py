from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def _load_yaml(relative_path):
    package_share = Path(get_package_share_directory("slam_3d_bringup"))
    with (package_share / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_large_scale_pipeline_preserves_raw_livox_contract():
    config = _load_yaml("config/large_scale_outdoor_pipeline.yaml")

    assert config["profile"] == "gole_dual_mid360_large_scale_outdoor_v0"
    assert (
        config["sensor_ingest"]["rule"]
        == "preserve_livox_custom_time_and_identity_fields_into_canonical_timed_clouds"
    )
    assert config["sensor_ingest"]["timestamp_contract"]["input_offset_time"].startswith(
        "uint32_nanoseconds"
    )
    assert config["sensor_ingest"]["dual_policy"]["front_rear_outputs"] == "separate"
    assert "kitti_xyz32" in config["sensor_ingest"]["legacy_dataset_bypass"][
        "allowed_for"
    ]
    assert "PointXYZ_only_conversion" in config["sensor_ingest"]["forbidden_before_lio"]
    measurement_sets = config["frontend_lio"]["update_model"]["measurement_sets"]
    assert measurement_sets["front_mid360"]["covariance_scale"] == 1.0
    assert measurement_sets["rear_mid360"]["covariance_scale"] == 1.0
    assert config["submap_manager"]["storage"]["descriptors"] == "scan_context_or_ring_key"
    assert "accepted_loops" in config["runtime_health_metrics"]["required"]


def test_topic_contract_keeps_navigation_bridge_explicit():
    contract = _load_yaml("config/dual_mid360_topic_contract.yaml")

    assert contract["existing_inputs"]["wheel_odom"]["topic"] == "/motor_driver_node/odom"
    assert contract["existing_inputs"]["external_imu"]["topic"] == "/imu/data"
    assert contract["mid360_raw_inputs"]["front_custom"]["preferred_type"].endswith(
        "CustomMsg"
    )
    timed = contract["mid360_raw_inputs"]["front_points_timed"]
    assert timed["required_fields"]["offset_time"].startswith("UINT32_nanoseconds")
    assert timed["header_stamp"].startswith("livox_custom_timebase_plus")
    assert contract["ingest_health"]["topic"] == "/slam_3d/ingest/health"
    assert (
        contract["ingest_health"]["commissioning_metadata"]["imu_clock_domain"]
        == "unverified_until_measured"
    )
    assert contract["internal_topics"]["frontend_odom"]["topic"] == "/slam_3d/frontend/odom"
    assert contract["internal_topics"]["backend_odom"]["topic"] == "/slam_3d/backend/odom"
    assert "exactly_one_localization_or_slam_stack" in contract["downstream_bridge_topics"][
        "nav2_tf"
    ]["rule"]
