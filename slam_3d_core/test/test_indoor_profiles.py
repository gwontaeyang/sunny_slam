from pathlib import Path

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"
PROFILES = {
    "xt32": "indoor_xt32_single_params.yaml",
    "xt32_gicp_quality": "indoor_xt32_single_gicp_quality_params.yaml",
    "dual_jt128": "indoor_dual_jt128_params.yaml",
}


def _load_profile(name):
    path = CONFIG_DIR / PROFILES[name]
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert set(document) == {"scan_to_map_icp_node"}
    return document["scan_to_map_icp_node"]["ros__parameters"]


def _flatten_keys(value, prefix=""):
    if not isinstance(value, dict):
        return []
    keys = []
    for key, child in value.items():
        qualified = f"{prefix}.{key}" if prefix else str(key)
        keys.append(qualified.lower())
        keys.extend(_flatten_keys(child, qualified))
    return keys


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_are_imu_deskewed_and_wheel_independent(profile_name):
    params = _load_profile(profile_name)

    assert params["imu_topic"] == "/imu/data"
    assert params["imu_subscription_reliable"] is True
    assert params["imu_require_expected_frame"] is True
    assert params["imu_expected_frame"] == "cv7_link"
    assert params["use_wheel_prediction"] is False
    expected_lidar_cv = profile_name == "xt32"
    assert params["use_constant_velocity_prediction"] is expected_lidar_cv
    if expected_lidar_cv:
        assert 0.0 < params["constant_velocity_translation_limit_m"] <= 0.5
        assert 0.0 < params["constant_velocity_rotation_limit_deg"] <= 15.0
    assert params["wheel_odom_topic"].startswith("/unused/")
    assert params["deskew_enabled"] is True
    assert params["deskew_wait_for_imu_watermark"] is True
    assert params["reject_cloud_on_deskew_failure"] is True
    assert params["deskew_point_time_field"] == "offset_time"
    assert params["deskew_point_time_scale"] == 0.0
    assert 0.0 < params["deskew_min_time_ratio"] <= 1.0
    assert params["deskew_max_imu_gap_s"] <= 0.05
    assert params["registration_gap_relaxation_enabled"] is True
    assert params["registration_gap_relaxation_min_gap_s"] > 0.0
    assert params["registration_gap_relaxation_max_speed_mps"] > 0.0
    assert (
        params["registration_gap_relaxation_max_translation_m"]
        > params["max_registration_correction_translation_m"]
    )
    assert 0.0 < params["registration_gap_relaxation_min_inlier_ratio"] <= 1.0
    assert params["registration_gap_relaxation_max_inlier_rmse_m"] > 0.0


def test_xt32_profile_enables_only_the_center_primary():
    params = _load_profile("xt32")

    assert params["primary_cloud_topic"] == (
        "/slam_3d/input/hesai_center/points_timed"
    )
    assert params["use_secondary_cloud"] is False
    assert params["use_tertiary_cloud"] is False
    assert params["secondary_cloud_topic"].startswith("/unused/")
    assert params["tertiary_cloud_topic"].startswith("/unused/")
    assert params["inter_lidar_motion_compensation_enabled"] is False
    assert params["registration_method"] == "voxel_surfel"
    assert params["surfel_min_observable_rank"] >= 3
    assert params["surfel_min_correspondences"] > 0


def test_xt32_gicp_quality_profile_remains_an_independent_fallback():
    params = _load_profile("xt32_gicp_quality")

    assert params["primary_cloud_topic"] == (
        "/slam_3d/input/hesai_center/points_timed"
    )
    assert params["registration_method"] == "gicp"
    assert params["use_constant_velocity_prediction"] is False


def test_dual_jt128_profile_enables_front_primary_and_rear_secondary():
    params = _load_profile("dual_jt128")

    assert params["primary_cloud_topic"] == (
        "/slam_3d/input/hesai_front/points_timed"
    )
    assert params["secondary_cloud_topic"] == (
        "/slam_3d/input/hesai_rear/points_timed"
    )
    assert params["use_secondary_cloud"] is True
    assert params["use_tertiary_cloud"] is False
    assert params["tertiary_cloud_topic"].startswith("/unused/")
    assert params["inter_lidar_motion_compensation_enabled"] is True
    assert params["inter_lidar_motion_compensation_source"] == "imu_rotation"
    assert 0.0 < params["inter_lidar_motion_compensation_max_dt_s"] <= 0.05
    assert params["inter_lidar_imu_rotation_scale"] == 1.0
    assert params["inter_lidar_gap_recovery_hypothesis_enabled"] is True
    assert params["inter_lidar_gap_recovery_hypothesis_min_gap_s"] >= 0.5
    assert 0.0 < (
        params[
            "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio"
        ]
    ) < 1.0
    assert 0.0 < params["max_secondary_stamp_delta_s"] <= 0.02
    assert params["cloud_sync_wait_timeout_s"] <= 0.10


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_define_se3_keyframe_and_overlapping_submap_contract(profile_name):
    params = _load_profile(profile_name)

    assert params["mapping_spatial_mode"] == "se3"
    assert params["local_map_vertical_radius_m"] > 0.0
    assert params["keyframe_vertical_translation_m"] > 0.0
    assert params["keyframe_rotation_3d_deg"] > 0.0
    assert params["submap_enabled"] is True
    assert params["submap_max_keyframes"] >= 10
    assert params["submap_max_travel_m"] > params["keyframe_translation_m"]
    assert 0 < params["submap_overlap_keyframes"] < params["submap_max_keyframes"]
    assert params["max_local_submaps"] >= 2


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_quarantine_persistent_state_during_imu_shock(profile_name):
    params = _load_profile(profile_name)

    assert params["imu_roughness_detection_enabled"] is True
    assert params["imu_shock_policy_enabled"] is True
    assert 0.0 < params["imu_roughness_window_s"] <= 0.2
    assert (
        params["imu_roughness_recent_event_window_s"]
        >= params["imu_shock_hold_duration_s"]
    )
    assert params["imu_roughness_angular_velocity_threshold_radps"] > 0.0
    assert params["imu_roughness_angular_accel_threshold_radps2"] > 0.0
    assert params["imu_roughness_accel_norm_delta_threshold_mps2"] > 0.0
    assert params["imu_roughness_linear_jerk_threshold_mps3"] > 0.0
    assert params["imu_shock_hold_duration_s"] > 0.0
    assert params["imu_shock_recovery_stable_scans"] >= 2
    assert params["suppress_constant_velocity_on_imu_shock"] is True
    assert params["omit_shock_poses_from_path"] is True
    assert params["quarantine_keyframes_on_shock"] is True
    assert params["quarantine_pose_on_shock"] is False
    assert params["ground_vehicle_motion_constraint_enabled"] is False
    assert params["ground_vehicle_nominal_max_body_z_step_m"] > 0.0
    assert (
        params["ground_vehicle_roughness_max_body_z_step_m"]
        == params["ground_vehicle_nominal_max_body_z_step_m"]
    )
    assert params["gravity_roll_pitch_constraint_enabled"] is False
    assert params["gravity_accel_time_constant_s"] > 0.0


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_publish_confirmed_3d_terrain_evidence(profile_name):
    params = _load_profile(profile_name)

    assert params["terrain_evidence_enabled"] is True
    assert params["terrain_evidence_topic"] == "/slam_3d/terrain/evidence"
    assert (
        params["terrain_evidence_colored_topic"]
        == "/slam_3d/terrain/evidence_colored"
    )
    assert (
        params["terrain_support_evidence_topic"]
        == "/slam_3d/terrain/support_evidence"
    )
    assert (
        params["terrain_preview_evidence_topic"]
        == "/slam_3d/terrain/preview_evidence"
    )
    assert params["terrain_azimuth_bins"] >= 60
    assert params["terrain_radial_bin_size_m"] > 0.0
    assert params["terrain_min_range_m"] < params["terrain_seed_max_range_m"]
    assert params["terrain_seed_max_range_m"] < params["terrain_max_range_m"]
    assert (
        0.0
        < params["terrain_classification_voxel_size_m"]
        <= 0.5 * params["terrain_step_min_height_m"]
    )
    assert params["terrain_max_profile_height_above_base_m"] <= 0.5
    assert 0.0 <= params["terrain_floor_quantile"] <= 1.0
    assert isinstance(params["terrain_seed_reference_height_m"], (int, float))
    assert params["terrain_seed_height_cluster_tolerance_m"] > 0.0
    assert params["terrain_seed_min_cluster_bins"] >= 3
    assert (
        params["terrain_floor_max_slope_deg"]
        <= params["terrain_ramp_min_slope_deg"]
        < params["terrain_ramp_max_slope_deg"]
        < 90.0
    )
    assert params["terrain_ramp_min_run_length_m"] >= 0.5
    assert params["terrain_ramp_plane_validation_enabled"] is True
    assert params["terrain_ramp_plane_neighborhood_radius_m"] > 0.0
    assert params["terrain_ramp_plane_min_points"] >= 3
    assert params["terrain_ramp_plane_min_length_m"] >= 0.5
    assert params["terrain_ramp_plane_min_width_m"] > 0.0
    assert params["terrain_ramp_plane_max_rmse_m"] > 0.0
    assert params["terrain_ramp_plane_max_outlier_distance_m"] >= params[
        "terrain_ramp_plane_max_rmse_m"
    ]
    assert 0.5 < params["terrain_ramp_plane_min_inlier_ratio"] <= 1.0
    assert 0.0 < params["terrain_step_min_height_m"] < params["terrain_step_max_height_m"]
    assert 0.0 < params["terrain_drop_min_height_m"] < params["terrain_drop_max_height_m"]
    assert params["terrain_staircase_enabled"] is True
    assert params["terrain_staircase_min_risers"] >= 3
    assert (
        0.0
        < params["terrain_staircase_min_riser_height_m"]
        < params["terrain_staircase_max_riser_height_m"]
    )
    assert (
        0.0
        < params["terrain_staircase_min_tread_depth_m"]
        < params["terrain_staircase_max_tread_depth_m"]
    )
    assert (
        params["terrain_staircase_max_profile_height_above_base_m"]
        > params["terrain_max_profile_height_above_base_m"]
    )
    assert params["terrain_min_confirmations"] >= 2
    assert params["terrain_confirmation_min_viewpoint_distance_m"] > 0.0
    assert params["terrain_require_gravity_alignment"] is True
    assert 0.5 < params["terrain_min_view_label_confidence"] <= 1.0
    assert 0.5 < params["terrain_min_label_confidence"] <= 1.0
    assert (
        0.5
        < params["terrain_probable_support_min_confidence"]
        <= params["terrain_min_label_confidence"]
    )
    assert params["terrain_probable_support_max_roughness_score"] > 0.0
    assert params["terrain_probable_support_require_confirmed_connectivity"] is True
    assert params["terrain_probable_support_max_anchor_distance_m"] > 0.0
    assert params["terrain_probable_support_max_neighbor_height_delta_m"] > 0.0
    assert 0.0 <= params["terrain_ramp_traversability_scale"] <= 1.0
    assert params["terrain_roughness_preview_enabled"] is True
    assert 0.0 < params["terrain_roughness_min_height_m"] < params[
        "terrain_step_min_height_m"
    ]
    assert params["terrain_roughness_max_score"] >= 1.0
    assert params["terrain_traversal_confirmation_enabled"] is True
    assert params["terrain_traversal_association_radius_m"] > 0.0
    assert (
        params["terrain_traversal_exit_radius_m"]
        >= params["terrain_traversal_association_radius_m"]
    )
    assert params["terrain_traversal_max_vertical_delta_m"] > 0.0
    assert params["terrain_traversal_max_sample_gap_s"] >= params[
        "imu_shock_hold_duration_s"
    ]
    assert params["terrain_traversal_max_swept_segment_length_m"] > 0.0
    assert params["terrain_preview_min_ahead_distance_m"] >= params[
        "terrain_traversal_association_radius_m"
    ]
    assert params["terrain_preview_max_lateral_distance_m"] >= params[
        "terrain_traversal_association_radius_m"
    ]
    assert 0.5 < params["terrain_traversal_min_label_confidence"] <= 1.0
    assert params["terrain_preview_roughness_score_gate"] > 0.0
    assert params["terrain_traversal_min_passage_samples"] >= 2
    assert params["terrain_traversal_min_passage_duration_s"] > 0.0
    assert params["gravity_roll_pitch_constraint_enabled"] is False


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_require_confirmed_submap_loops_and_robust_graph(profile_name):
    params = _load_profile(profile_name)

    assert params["place_recognition_enabled"] is True
    assert params["place_recognition_periodic_enabled"] is True
    assert params["place_recognition_use_submaps"] is True
    assert params["place_recognition_submap_min_consecutive_matches"] >= 2
    assert params["place_recognition_max_vertical_separation_m"] > 0.0
    assert params["place_recognition_submap_consistency_max_translation_m"] > 0.0
    assert params["place_recognition_submap_consistency_max_rotation_deg"] > 0.0
    assert params["place_recognition_require_pose_graph_correction"] is True
    assert params["pose_graph_correction_enabled"] is True
    assert params["pose_graph_planar_correction_enabled"] is False
    assert params["pose_graph_robust_loop_enabled"] is True
    assert params["pose_graph_loop_cauchy_scale"] > 0.0
    assert 0.0 < params["pose_graph_min_new_loop_robust_weight"] <= 1.0
    assert params["pose_graph_max_iteration_translation_step_m"] > 0.0
    assert params["pose_graph_max_iteration_rotation_step_deg"] > 0.0
    assert params["pose_graph_max_loop_residual_before"] > 0.0
    assert params["pose_graph_max_residual_after"] > 0.0
    assert 0.0 < params["pose_graph_max_residual_ratio"] <= 1.0


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_do_not_enable_dynamic_object_processing(profile_name):
    params = _load_profile(profile_name)
    forbidden_fragments = ("dynamic", "moving_object", "object_tracking")

    assert not any(
        fragment in key
        for key in _flatten_keys(params)
        for fragment in forbidden_fragments
    )


@pytest.mark.parametrize("profile_name", PROFILES)
def test_profiles_use_native_cv7_axis_convention(profile_name):
    params = _load_profile(profile_name)
    config_text = (CONFIG_DIR / PROFILES[profile_name]).read_text(encoding="utf-8")

    assert params["deskew_imu_to_base_roll_deg"] == 0.0
    assert params["deskew_imu_to_base_pitch_deg"] == 0.0
    assert params["deskew_imu_to_base_yaw_deg"] == 0.0
    assert "native XYZ" in config_text
    assert "identity base_link->cv7_link" in config_text
    assert "legacy roll=180" in config_text
    assert "live calibration" in config_text
