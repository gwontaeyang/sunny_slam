from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def _load_params():
    package_share = Path(get_package_share_directory("slam_3d_core"))
    params_path = package_share / "config" / "scan_to_map_icp_params.yaml"
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["scan_to_map_icp_node"]["ros__parameters"]


def test_default_topics_are_isolated_from_navigation_stack():
    params = _load_params()

    assert params["odom_topic"].startswith("/slam_3d/")
    assert params["path_topic"].startswith("/slam_3d/")
    assert params["local_map_topic"].startswith("/slam_3d/")
    assert params["global_map_topic"].startswith("/slam_3d/")
    assert params["odom_topic"] != "/odom"
    assert params["publish_tf"] is False


def test_inputs_match_existing_gole_sensor_contract():
    params = _load_params()

    assert params["primary_cloud_topic"] == "/slam_3d/input/mid360_front/points_debug"
    assert params["secondary_cloud_topic"] == "/slam_3d/input/mid360_rear/points_debug"
    assert params["wheel_odom_topic"] == "/motor_driver_node/odom"
    assert params["imu_topic"] == "/imu/data"
    assert params["use_secondary_cloud"] is True


def test_registration_and_map_parameters_are_bounded():
    params = _load_params()

    assert params["registration_method"] in ("gicp", "icp")
    assert params["min_scan_points"] >= 100
    assert params["scan_voxel_leaf_m"] > 0.0
    assert params["adaptive_scan_voxel_enabled"] is False
    assert 0.0 < params["adaptive_scan_voxel_leaf_m"] < params["scan_voxel_leaf_m"]
    assert params["adaptive_scan_voxel_min_points"] >= params["min_scan_points"]
    assert params["map_voxel_leaf_m"] > 0.0
    assert params["max_icp_iterations"] > 0
    assert params["icp_max_correspondence_m"] > 0.0
    assert params["fitness_score_threshold"] > 0.0
    assert params["robust_registration_enabled"] is False
    assert 0.0 < params["robust_registration_trim_ratio"] <= 1.0
    assert 0.0 < params["robust_registration_refinement_max_correspondence_m"] < params[
        "icp_max_correspondence_m"
    ]
    assert params["robust_registration_refinement_iterations"] > 0
    assert 0.0 < params["robust_registration_quality_max_correspondence_m"] <= params[
        "robust_registration_refinement_max_correspondence_m"
    ]
    assert params["use_constant_velocity_prediction"] is True
    assert params["use_scan_to_scan_prediction"] is False
    assert params["scan_to_scan_max_reference_age_s"] >= 0.0
    assert params["use_scan_to_scan_fallback"] is False
    assert params["add_scan_to_scan_fallback_keyframes"] is False
    assert params["use_motion_prior_fallback"] is False
    assert params["add_motion_prior_fallback_keyframes"] is False
    assert params["deskew_enabled"] is False
    assert params["deskew_point_time_field"] == "auto"
    assert params["deskew_point_time_scale"] >= 0.0
    assert abs(params["deskew_point_time_offset_s"]) <= params["deskew_max_scan_duration_s"]
    assert 0.0 < params["deskew_max_scan_duration_s"] <= 1.0
    assert 0.0 < params["deskew_max_imu_gap_s"] <= params["deskew_imu_buffer_duration_s"]
    assert params["imu_subscription_depth"] >= 200
    assert 0.0 <= params["imu_reorder_tolerance_s"]
    assert params["imu_timestamp_reset_threshold_s"] > params["imu_reorder_tolerance_s"]
    assert params["imu_watermark_topic"].startswith("/slam_3d/")
    assert params["cloud_subscription_reliable"] is False
    assert params["cloud_subscription_depth"] >= 1
    assert 0.0 <= params["deskew_min_time_ratio"] <= 1.0
    assert 0.0 <= params["deskew_time_bin_s"] <= params["deskew_max_scan_duration_s"]
    assert params["deskew_angular_velocity_scale"] > 0.0
    assert params["imu_rotation_prediction_enabled"] is False
    assert params["imu_rotation_prediction_max_dt_s"] > 0.0
    assert params["imu_rotation_prediction_scale"] != 0.0
    assert params["imu_roughness_detection_enabled"] is False
    assert params["imu_roughness_window_s"] > 0.0
    assert params["imu_roughness_recent_event_window_s"] >= params["imu_roughness_window_s"]
    assert params["imu_roughness_angular_velocity_threshold_radps"] > 0.0
    assert params["imu_roughness_angular_accel_threshold_radps2"] > 0.0
    assert params["imu_roughness_accel_norm_delta_threshold_mps2"] > 0.0
    assert params["imu_roughness_linear_jerk_threshold_mps3"] > 0.0
    assert params["constant_velocity_translation_limit_m"] > 0.0
    assert params["constant_velocity_rotation_limit_deg"] > 0.0
    assert params["motion_prior_fallback_max_step_m"] >= 0.0
    assert params["motion_prior_fallback_max_z_step_m"] >= 0.0
    assert params["motion_prior_fallback_max_untrusted_distance_m"] >= 0.0
    assert params["motion_prior_fallback_max_untrusted_z_m"] >= 0.0
    assert params["motion_prior_fallback_max_rotation_deg"] >= 0.0
    assert params["max_registration_correction_translation_m"] > 0.0
    assert params["max_registration_correction_rotation_deg"] > 0.0
    assert params["quarantine_keyframes_on_shock"] is False
    assert params["quarantine_pose_on_shock"] is False
    assert params["record_tracking_scan_on_pose_quarantine"] is False
    assert params["pose_quarantine_tracking_reset_interval"] == 0
    assert params["keyframe_quarantine_correction_translation_m"] > 0.0
    assert params["keyframe_quarantine_correction_rotation_deg"] > 0.0
    assert params["place_recognition_enabled"] is False
    assert params["place_recognition_periodic_enabled"] is False
    assert params["place_recognition_require_pose_graph_correction"] is False
    assert params["place_recognition_periodic_min_interval_s"] >= 0.0
    assert params["place_recognition_relocalization_min_interval_s"] >= 0.0
    assert params["place_recognition_shock_relocalization_cooldown_s"] >= 0.0
    assert params["place_recognition_descriptor_bins"] >= 8
    assert params["place_recognition_descriptor_rings"] >= 1
    assert params["place_recognition_min_keyframes"] > 0
    assert params["place_recognition_exclude_recent_keyframes"] > 0
    assert params["place_recognition_max_candidates"] > 0
    assert params["place_recognition_candidate_map_neighbors"] >= 0
    assert params["place_recognition_max_descriptor_distance"] > 0.0
    assert 0.0 <= params["place_recognition_min_inlier_ratio"] <= 1.0
    assert params["place_recognition_max_inlier_rmse_m"] > 0.0
    assert params["place_recognition_max_pose_jump_m"] > 0.0
    assert params["place_recognition_max_pose_jump_rotation_deg"] > 0.0
    assert 0.0 < params["place_recognition_max_yaw_offset_deg"] <= 180.0
    assert params["pose_graph_correction_enabled"] is False
    assert params["pose_graph_planar_correction_enabled"] is False
    assert params["pose_graph_robust_loop_enabled"] is False
    assert params["planar_pose_constraint_enabled"] is False
    assert params["planar_pose_max_z_correction_m"] >= 0.0
    assert params["gravity_roll_pitch_constraint_enabled"] is False
    assert 0.0 <= params["gravity_accel_low_pass_alpha"] <= 1.0
    assert params["gravity_accel_time_constant_s"] >= 0.0
    assert params["gravity_accel_min_norm_mps2"] > 0.0
    assert params["gravity_accel_max_norm_mps2"] >= params["gravity_accel_min_norm_mps2"]
    assert 0.0 < params["gravity_accel_max_age_s"] <= params["deskew_imu_buffer_duration_s"]
    assert params["map_z_filter_enabled"] is False
    assert params["map_z_filter_min_m"] < params["map_z_filter_max_m"]
    assert params["pose_graph_correction_min_keyframes"] > 0
    assert params["pose_graph_correction_max_translation_m"] > 0.0
    assert params["pose_graph_correction_max_rotation_deg"] > 0.0
    assert params["pose_graph_correction_min_interval_s"] >= 0.0
    assert params["pose_graph_optimization_iterations"] > 0
    assert 0.0 < params["pose_graph_optimization_step"] <= 1.0
    assert params["pose_graph_odom_translation_weight"] >= 0.0
    assert params["pose_graph_odom_rotation_weight"] >= 0.0
    assert params["pose_graph_loop_translation_weight"] >= 0.0
    assert params["pose_graph_loop_rotation_weight"] >= 0.0
    assert params["pose_graph_loop_cauchy_scale"] > 0.0
    assert params["pose_graph_min_new_loop_robust_weight"] == 0.0
    assert params["pose_graph_max_iteration_translation_step_m"] > 0.0
    assert params["pose_graph_max_iteration_rotation_step_deg"] > 0.0
    assert params["pose_graph_max_loop_residual_before"] >= 0.0
    assert params["pose_graph_max_residual_after"] >= 0.0
    assert params["pose_graph_max_residual_ratio"] >= 0.0
    assert 0.0 <= params["min_registration_inlier_ratio"] <= 1.0
    assert params["max_registration_inlier_rmse_m"] > 0.0
    assert params["max_keyframes"] >= params["max_local_keyframes"]
    assert params["max_path_poses"] > 0
