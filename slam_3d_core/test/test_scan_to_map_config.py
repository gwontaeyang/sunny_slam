import importlib.util
import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument


def _load_named_params(filename):
    package_share = Path(get_package_share_directory("slam_3d_core"))
    params_path = package_share / "config" / filename
    with params_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["scan_to_map_icp_node"]["ros__parameters"]


def _load_params():
    return _load_named_params("scan_to_map_icp_params.yaml")


def _load_launch_module():
    package_share = Path(get_package_share_directory("slam_3d_core"))
    launch_path = package_share / "launch" / "scan_to_map_icp.launch.py"
    spec = importlib.util.spec_from_file_location(
        "scan_to_map_icp_launch", launch_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_context_with_defaults(module):
    context = LaunchContext()
    declarations = [
        entity
        for entity in module.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    ]
    for declaration in declarations:
        context.launch_configurations[declaration.name] = "".join(
            substitution.perform(LaunchContext())
            for substitution in declaration.default_value
        )
    return context


def test_launch_preserves_registration_method_from_params_file_by_default():
    module = _load_launch_module()
    context = _launch_context_with_defaults(module)

    assert module._parameter_overrides(context) == {}

    context.launch_configurations["registration_method"] = "voxel_surfel"
    assert (
        module._parameter_overrides(context)["registration_method"]
        == "voxel_surfel"
    )


def test_launch_keeps_wheel_frame_contract_with_frame_overrides():
    module = _load_launch_module()
    context = _launch_context_with_defaults(module)
    context.launch_configurations["odom_frame"] = "test_odom"
    context.launch_configurations["base_frame"] = "test_base"
    context.launch_configurations[
        "pose_graph_marker_topic"
    ] = "/test/pose_graph"

    overrides = module._parameter_overrides(context)

    assert overrides["odom_frame"].perform(context) == "test_odom"
    assert overrides["base_frame"].perform(context) == "test_base"
    assert overrides["wheel_odom_expected_frame"].perform(context) == "test_odom"
    assert (
        overrides["wheel_odom_expected_child_frame"].perform(context)
        == "test_base"
    )
    assert (
        overrides["pose_graph_marker_topic"].perform(context)
        == "/test/pose_graph"
    )


def test_default_topics_are_isolated_from_navigation_stack():
    params = _load_params()

    assert params["odom_topic"].startswith("/slam_3d/")
    assert params["path_topic"].startswith("/slam_3d/")
    assert params["local_map_topic"].startswith("/slam_3d/")
    assert params["global_map_topic"].startswith("/slam_3d/")
    assert params["terrain_evidence_topic"].startswith("/slam_3d/")
    assert params["terrain_evidence_colored_topic"].startswith("/slam_3d/")
    assert params["terrain_support_evidence_topic"].startswith("/slam_3d/")
    assert params["preprocessed_scan_topic"].startswith("/slam_3d/")
    assert params["publish_preprocessed_scan"] is False
    assert (
        params["pose_graph_marker_topic"]
        == "/slam_3d/backend/pose_graph_markers"
    )
    assert params["odom_topic"] != "/odom"
    assert params["publish_tf"] is False


def test_inputs_match_existing_gole_sensor_contract():
    params = _load_params()

    assert params["primary_cloud_topic"] == "/slam_3d/input/hesai_center/points_timed"
    assert params["secondary_cloud_topic"] == "/slam_3d/input/hesai_front/points_timed"
    assert params["tertiary_cloud_topic"] == "/slam_3d/input/hesai_rear/points_timed"
    assert params["wheel_odom_topic"] == "/motor_driver_node/odom"
    assert params["imu_topic"] == "/imu/data"
    assert params["imu_require_expected_frame"] is False
    assert params["imu_expected_frame"] == ""
    assert params["use_secondary_cloud"] is True
    assert params["use_tertiary_cloud"] is True
    assert params["secondary_cloud_registration_only"] is False
    assert params["tertiary_cloud_registration_only"] is False
    assert params["cloud_sync_queue_size"] >= 3
    assert 0.0 < params["max_auxiliary_stamp_delta_s"] <= 0.05
    assert params["cloud_sync_wait_for_nearest_candidate"] is False
    assert params["cloud_sync_allow_wall_timeout_fallback"] is True
    assert 0.0 <= params["cloud_sync_wait_timeout_s"] <= 0.5
    assert 0.0 <= params["cloud_sync_reorder_window_s"] <= 0.1
    assert "cloud_sync_dropout_horizon_s" not in params
    assert 0.001 <= params["cloud_processing_period_s"] <= 0.1
    assert params["executor_thread_count"] >= 1
    assert params["initialization_warmup_duration_s"] >= 0.0


def test_registration_and_map_parameters_are_bounded():
    params = _load_params()

    assert params["registration_method"] in ("gicp", "icp", "voxel_surfel")
    assert params["min_scan_points"] >= 100
    assert params["scan_voxel_leaf_m"] > 0.0
    assert params["adaptive_scan_voxel_enabled"] is False
    assert 0.0 < params["adaptive_scan_voxel_leaf_m"] < params["scan_voxel_leaf_m"]
    assert params["adaptive_scan_voxel_min_points"] >= params["min_scan_points"]
    assert params["registration_only_cloud_voxel_leaf_m"] > 0.0
    assert params["registration_only_cloud_max_point_ratio"] > 0.0
    assert params["registration_only_cloud_max_points"] >= 0
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
    assert params["surfel_voxel_size_m"] > 0.0
    assert params["surfel_min_points_per_voxel"] >= 3
    assert 0.0 <= params["surfel_max_planarity_ratio"] < 1.0
    assert params["surfel_min_tangent_variance"] >= 0.0
    assert params["surfel_max_plane_distance_m"] > 0.0
    assert params["surfel_max_center_distance_m"] > 0.0
    assert params["surfel_min_correspondences"] > 0
    assert 0.0 < params["surfel_min_correspondence_ratio"] <= 1.0
    assert params["surfel_huber_delta_m"] > 0.0
    assert params["surfel_max_iterations"] > 0
    assert params["surfel_max_backtracking_steps"] >= 0
    assert 0.0 < params["surfel_backtracking_factor"] < 1.0
    assert params["surfel_rotation_convergence_rad"] >= 0.0
    assert params["surfel_translation_convergence_m"] >= 0.0
    assert params["surfel_relative_cost_convergence"] >= 0.0
    assert params["surfel_max_rotation_step_rad"] > 0.0
    assert params["surfel_max_translation_step_m"] > 0.0
    assert params["surfel_rotation_scale_m_per_rad"] > 0.0
    assert 0.0 < params["surfel_relative_eigen_threshold"] < 1.0
    assert 1 <= params["surfel_min_observable_rank"] <= 6
    assert params["surfel_max_quality_rmse_m"] > 0.0
    assert params["surfel_max_quality_p95_m"] >= params["surfel_max_quality_rmse_m"]
    assert params["use_constant_velocity_prediction"] is True
    assert params["use_scan_to_scan_prediction"] is False
    assert params["scan_to_scan_max_reference_age_s"] >= 0.0
    assert params["use_scan_to_scan_fallback"] is False
    assert params["add_scan_to_scan_fallback_keyframes"] is False
    assert params["use_motion_prior_fallback"] is False
    assert params["add_motion_prior_fallback_keyframes"] is False
    assert params["deskew_enabled"] is True
    assert params["deskew_wait_for_imu_watermark"] is True
    assert params["deskew_imu_wait_timeout_s"] > 0.0
    assert params["inter_lidar_motion_compensation_enabled"] is True
    assert params["inter_lidar_motion_compensation_source"] == "wheel_odom"
    assert 0.0 < params["inter_lidar_motion_compensation_max_dt_s"] <= 0.1
    assert params["inter_lidar_imu_rotation_scale"] != 0.0
    assert params["inter_lidar_gap_recovery_hypothesis_enabled"] is False
    assert params["inter_lidar_gap_recovery_hypothesis_min_gap_s"] >= 0.0
    assert 0.0 <= (
        params[
            "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio"
        ]
    ) <= 1.0
    assert params["inter_lidar_motion_failure_policy"] == "drop_auxiliary"
    assert params["reject_cloud_on_tf_failure"] is True
    assert params["reject_cloud_on_deskew_failure"] is True
    assert params["deskew_point_time_field"] == "offset_time"
    assert params["deskew_point_time_scale"] >= 0.0
    assert abs(params["deskew_point_time_offset_s"]) <= params["deskew_max_scan_duration_s"]
    assert 0.0 < params["deskew_max_scan_duration_s"] <= 1.0
    assert 0.0 < params["deskew_max_imu_gap_s"] <= params["deskew_imu_buffer_duration_s"]
    assert params["imu_subscription_depth"] >= 200
    assert params["imu_subscription_reliable"] is False
    assert 0.0 <= params["imu_reorder_tolerance_s"]
    assert params["imu_timestamp_reset_threshold_s"] > params["imu_reorder_tolerance_s"]
    assert params["imu_watermark_topic"].startswith("/slam_3d/")
    assert params["cloud_subscription_reliable"] is True
    assert params["cloud_subscription_depth"] >= 1
    assert 0.0 <= params["deskew_min_time_ratio"] <= 1.0
    assert 0.0 <= params["deskew_time_bin_s"] <= params["deskew_max_scan_duration_s"]
    assert params["deskew_angular_velocity_scale"] > 0.0
    assert params["imu_rotation_prediction_enabled"] is False
    assert params["imu_rotation_prediction_source"] in (
        "angular_velocity",
        "orientation",
    )
    assert params["imu_rotation_prediction_max_dt_s"] > 0.0
    assert params["imu_rotation_prediction_scale"] != 0.0
    assert params["imu_orientation_prediction_max_time_delta_s"] > 0.0
    assert params["imu_relative_roll_pitch_constraint_enabled"] is False
    assert params["imu_relative_attitude_max_time_delta_s"] > 0.0
    assert params["imu_relative_roll_pitch_max_correction_deg"] >= 0.0
    assert params["imu_roughness_detection_enabled"] is False
    assert params["imu_shock_policy_enabled"] is False
    assert params["imu_roughness_window_s"] > 0.0
    assert params["imu_roughness_recent_event_window_s"] >= params["imu_roughness_window_s"]
    assert params["imu_roughness_angular_velocity_threshold_radps"] > 0.0
    assert params["imu_roughness_angular_accel_threshold_radps2"] > 0.0
    assert params["imu_roughness_accel_norm_delta_threshold_mps2"] > 0.0
    assert params["imu_roughness_linear_jerk_threshold_mps3"] > 0.0
    assert params["imu_shock_hold_duration_s"] >= 0.0
    assert params["imu_shock_recovery_stable_scans"] >= 1
    assert params["suppress_constant_velocity_on_imu_shock"] is True
    assert params["omit_shock_poses_from_path"] is True
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
    assert params["record_tracking_scan_during_shock"] is False
    assert params["pose_quarantine_tracking_reset_interval"] == 0
    assert params["keyframe_quarantine_correction_translation_m"] > 0.0
    assert params["keyframe_quarantine_correction_rotation_deg"] > 0.0
    assert params["mapping_spatial_mode"] == "planar"
    assert params["keyframe_vertical_translation_m"] == 0.0
    assert params["keyframe_rotation_3d_deg"] == 0.0
    assert params["local_map_vertical_radius_m"] == 0.0
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
    assert (
        params["place_recognition_mapping_refinement_large_pose_jump_enabled"]
        is False
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_max_translation_m"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled"
        ]
        is False
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_inlier_ratio"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_"
            "verified_coarse_max_inlier_rmse_m"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_"
            "verified_coarse_min_target_coverage_ratio"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_large_pose_jump_"
            "verified_coarse_max_score_regression_ratio"
        ]
        == 0.0
    )
    assert 0.0 < params["place_recognition_max_yaw_offset_deg"] <= 180.0
    assert params["place_recognition_use_submaps"] is False
    assert params["place_recognition_submap_min_consecutive_matches"] == 1
    assert params["place_recognition_mapping_cloud_refinement_enabled"] is False
    assert (
        params["place_recognition_mapping_refinement_required_for_pose_graph"]
        is False
    )
    assert params["place_recognition_mapping_refinement_min_source_points"] > 0
    assert params["place_recognition_mapping_refinement_min_target_points"] > 0
    assert (
        params[
            "place_recognition_mapping_refinement_min_target_keyframe_observations"
        ]
        >= 2
    )
    assert (
        params["place_recognition_mapping_refinement_target_confirmation_voxel_m"]
        > 0.0
    )
    assert (
        params["place_recognition_mapping_refinement_quality_max_correspondence_m"]
        > 0.0
    )
    assert (
        0.0
        <= params["place_recognition_mapping_refinement_min_inlier_ratio"]
        <= 1.0
    )
    assert (
        0.0
        <= params["place_recognition_mapping_refinement_min_target_coverage_ratio"]
        <= 1.0
    )
    assert params["place_recognition_mapping_refinement_max_inlier_rmse_m"] > 0.0
    assert params["place_recognition_mapping_refinement_max_translation_m"] > 0.0
    assert params["place_recognition_mapping_refinement_max_rotation_deg"] > 0.0
    assert 0 <= params["place_recognition_mapping_refinement_min_observable_rank"] <= 6
    assert params["place_recognition_mapping_refinement_max_hessian_condition"] >= 0.0
    assert (
        0.0
        <= params["place_recognition_mapping_refinement_min_score_improvement_ratio"]
        <= 1.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_max_primary_score_regression_ratio"
        ]
        >= 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m"
        ]
        == 0.0
    )
    assert (
        params[
            "place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg"
        ]
        == 0.0
    )
    assert params["place_recognition_max_vertical_separation_m"] == 0.0
    assert params["place_recognition_registration_only_vertical_recovery_m"] == 0.0
    assert (
        params["place_recognition_registration_only_same_layer_xy_max_distance_m"]
        == 0.0
    )
    assert params["place_recognition_registration_only_max_pose_jump_m"] == 0.0
    assert params["place_recognition_submap_consistency_max_translation_m"] == 0.0
    assert (
        params[
            "place_recognition_registration_only_consistency_max_translation_m"
        ]
        == 0.0
    )
    assert params["place_recognition_registration_only_consistency_max_gap_s"] == 0.0
    assert params["place_recognition_submap_consistency_max_rotation_deg"] == 0.0
    assert params["pose_graph_correction_enabled"] is False
    assert params["pose_graph_planar_correction_enabled"] is False
    assert params["pose_graph_robust_loop_enabled"] is False
    assert params["pose_graph_distributed_loop_seed_enabled"] is False
    assert params["planar_pose_constraint_enabled"] is False
    assert params["planar_pose_max_z_correction_m"] >= 0.0
    assert params["ground_vehicle_motion_constraint_enabled"] is False
    assert params["ground_vehicle_nominal_max_body_z_step_m"] >= 0.0
    assert (
        params["ground_vehicle_roughness_max_body_z_step_m"]
        >= params["ground_vehicle_nominal_max_body_z_step_m"]
    )
    assert params["gravity_roll_pitch_constraint_enabled"] is False
    assert 0.0 <= params["gravity_accel_low_pass_alpha"] <= 1.0
    assert params["gravity_accel_time_constant_s"] >= 0.0
    assert params["gravity_accel_scale"] != 0.0
    assert params["gravity_accel_min_norm_mps2"] > 0.0
    assert params["gravity_accel_max_norm_mps2"] >= params["gravity_accel_min_norm_mps2"]
    assert 0.0 < params["gravity_accel_max_age_s"] <= params["deskew_imu_buffer_duration_s"]
    assert params["map_z_filter_enabled"] is False
    assert params["map_z_filter_min_m"] < params["map_z_filter_max_m"]
    assert params["terrain_evidence_enabled"] is False
    assert params["terrain_preview_evidence_topic"].startswith("/slam_3d/")
    assert params["terrain_azimuth_bins"] >= 8
    assert params["terrain_radial_bin_size_m"] > 0.0
    assert params["terrain_min_range_m"] < params["terrain_max_range_m"]
    assert params["terrain_min_range_m"] <= params["terrain_seed_max_range_m"]
    assert params["terrain_seed_max_range_m"] <= params["terrain_max_range_m"]
    assert params["terrain_classification_voxel_size_m"] > 0.0
    assert math.isfinite(params["terrain_seed_reference_height_m"])
    assert params["terrain_seed_height_cluster_tolerance_m"] > 0.0
    assert params["terrain_seed_min_cluster_bins"] >= 1
    assert params["terrain_max_profile_height_above_base_m"] >= 0.0
    assert 0.0 <= params["terrain_floor_quantile"] <= 1.0
    assert params["terrain_ground_band_m"] > 0.0
    assert params["terrain_max_adjacent_gap_m"] > 0.0
    assert (
        params["terrain_floor_max_slope_deg"]
        <= params["terrain_ramp_min_slope_deg"]
        <= params["terrain_ramp_max_slope_deg"]
    )
    assert params["terrain_ramp_min_run_bins"] >= 2
    assert params["terrain_ramp_min_run_length_m"] > 0.0
    assert params["terrain_ramp_plane_validation_enabled"] is False
    assert params["terrain_ramp_plane_neighborhood_radius_m"] > 0.0
    assert params["terrain_ramp_plane_min_points"] >= 3
    assert params["terrain_ramp_plane_min_length_m"] > 0.0
    assert params["terrain_ramp_plane_min_width_m"] > 0.0
    assert params["terrain_ramp_plane_max_rmse_m"] > 0.0
    assert params["terrain_ramp_plane_max_outlier_distance_m"] > 0.0
    assert 0.0 < params["terrain_ramp_plane_min_inlier_ratio"] <= 1.0
    assert params["terrain_step_min_height_m"] <= params["terrain_step_max_height_m"]
    assert params["terrain_drop_min_height_m"] <= params["terrain_drop_max_height_m"]
    assert params["terrain_transition_support_bins"] >= 1
    assert params["terrain_staircase_enabled"] is True
    assert params["terrain_staircase_min_risers"] >= 2
    assert (
        0.0
        < params["terrain_staircase_min_riser_height_m"]
        <= params["terrain_staircase_max_riser_height_m"]
    )
    assert (
        0.0
        < params["terrain_staircase_min_tread_depth_m"]
        <= params["terrain_staircase_max_tread_depth_m"]
    )
    assert (
        0.0
        <= params["terrain_staircase_min_slope_deg"]
        <= params["terrain_staircase_max_slope_deg"]
        < 90.0
    )
    assert (
        params["terrain_staircase_max_profile_height_above_base_m"]
        >= params["terrain_max_profile_height_above_base_m"]
    )
    assert params["terrain_evidence_voxel_size_m"] > 0.0
    assert params["terrain_min_confirmations"] >= 1
    assert params["terrain_confirmation_min_viewpoint_distance_m"] >= 0.0
    assert params["terrain_require_gravity_alignment"] is True
    assert 0.5 < params["terrain_min_label_confidence"] <= 1.0
    assert params["terrain_roughness_preview_enabled"] is False
    assert params["terrain_roughness_min_height_m"] > 0.0
    assert params["terrain_roughness_max_score"] >= 1.0
    assert params["terrain_traversal_confirmation_enabled"] is False
    assert params["terrain_traversal_association_radius_m"] > 0.0
    assert (
        params["terrain_traversal_exit_radius_m"]
        >= params["terrain_traversal_association_radius_m"]
    )
    assert params["terrain_traversal_max_vertical_delta_m"] > 0.0
    assert params["terrain_traversal_max_sample_gap_s"] >= 0.0
    assert params["terrain_traversal_max_swept_segment_length_m"] > 0.0
    assert params["terrain_preview_min_ahead_distance_m"] > 0.0
    assert params["terrain_preview_max_lateral_distance_m"] > 0.0
    assert 0.5 < params["terrain_traversal_min_label_confidence"] <= 1.0
    assert params["terrain_preview_roughness_score_gate"] > 0.0
    assert params["terrain_traversal_min_passage_samples"] >= 1
    assert params["terrain_traversal_min_passage_duration_s"] >= 0.0
    assert params["pose_graph_correction_min_keyframes"] > 0
    assert params["pose_graph_correction_max_translation_m"] > 0.0
    assert params["pose_graph_correction_max_rotation_deg"] > 0.0
    assert params["pose_graph_correction_min_interval_s"] >= 0.0
    assert params["pose_graph_minimum_interval_override_enabled"] is False
    assert params["pose_graph_duplicate_submap_pair_override_enabled"] is False
    assert params["pose_graph_duplicate_submap_pair_override_max_edges"] == 1
    assert params["pose_graph_minimum_interval_override_max_descriptor_distance"] > 0.0
    assert (
        0.0
        < params["pose_graph_minimum_interval_override_min_inlier_ratio"]
        <= 1.0
    )
    assert params["pose_graph_minimum_interval_override_max_inlier_rmse_m"] > 0.0
    assert (
        0.0
        < params["pose_graph_minimum_interval_override_min_target_coverage_ratio"]
        <= 1.0
    )
    assert params["pose_graph_minimum_interval_override_single_floor_z_enabled"] is False
    assert (
        params[
            "pose_graph_minimum_interval_override_single_floor_xy_max_distance_m"
        ]
        > 0.0
    )
    assert (
        params[
            "pose_graph_minimum_interval_override_single_floor_z_weight_multiplier"
        ]
        == 1.0
    )
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
    assert params["submap_enabled"] is False
    assert params["submap_max_keyframes"] >= 2
    assert params["submap_max_travel_m"] >= 0.0
    assert 0 <= params["submap_overlap_keyframes"] < params["submap_max_keyframes"]
    assert params["max_local_submaps"] >= 1
    assert params["max_path_poses"] > 0
    assert params["wheel_odom_buffer_duration_s"] > 0.0
    assert params["wheel_odom_buffer_max_samples"] >= 2
    assert params["wheel_odom_max_time_delta_s"] > 0.0
    assert params["wheel_odom_require_expected_frames"] is True
    assert params["wheel_odom_expected_frame"] == params["odom_frame"]
    assert params["wheel_odom_expected_child_frame"] == params["base_frame"]
    assert params["wheel_odom_timestamp_reset_threshold_s"] > 0.0
    assert (
        params["wheel_odom_max_interpolation_gap_s"]
        >= params["wheel_odom_max_time_delta_s"]
    )


def test_geoje_replay_profiles_are_explicitly_separate_from_production():
    production = _load_params()
    gicp = _load_named_params("geoje_legacy_scan_to_map_params.yaml")
    icp = _load_named_params("geoje_legacy_scan_to_map_icp_params.yaml")
    surfel = _load_named_params("geoje_legacy_scan_to_map_surfel_params.yaml")

    for params in (gicp, icp, surfel):
        assert params["use_sim_time"] is True
        assert params["use_secondary_cloud"] is True
        assert params["use_tertiary_cloud"] is True
        assert params["max_auxiliary_stamp_delta_s"] >= 0.05
        assert params["cloud_sync_reorder_window_s"] >= 0.0
        assert params["cloud_sync_wait_timeout_s"] >= 4.0
        assert params["cloud_sync_queue_size"] >= 64
        assert "cloud_sync_dropout_horizon_s" not in params
        assert params["inter_lidar_motion_failure_policy"] == "drop_auxiliary"
        assert params["inter_lidar_motion_compensation_source"] == "wheel_odom"
        assert params["inter_lidar_imu_rotation_scale"] != 0.0
        assert params["inter_lidar_gap_recovery_hypothesis_enabled"] is False
        assert params["reject_cloud_on_deskew_failure"] is False
        assert params["wheel_odom_buffer_duration_s"] >= 24.0
        assert params["wheel_odom_buffer_max_samples"] >= 1000
        assert params["wheel_odom_max_time_delta_s"] >= 1.5
        assert params["wheel_odom_max_interpolation_gap_s"] >= 1.5
        assert params["deskew_imu_buffer_duration_s"] >= 24.0
        assert params["imu_rotation_prediction_enabled"] is False
        assert params["imu_rotation_prediction_source"] == "orientation"
        assert params["imu_rotation_prediction_max_dt_s"] >= 1.5
        assert params["imu_orientation_prediction_max_time_delta_s"] <= 0.05

    assert production["wheel_odom_max_time_delta_s"] <= 0.15
    assert production["wheel_odom_max_interpolation_gap_s"] <= 0.25
    assert production["deskew_imu_buffer_duration_s"] < 24.0
    assert production["imu_rotation_prediction_enabled"] is False
    assert production["imu_rotation_prediction_source"] == "angular_velocity"

    assert gicp["registration_method"] == "gicp"
    assert gicp["scan_voxel_leaf_m"] >= 0.4
    assert gicp["map_voxel_leaf_m"] >= 0.4
    assert gicp["robust_registration_enabled"] is False
    assert icp["registration_method"] == "icp"
    assert icp["scan_voxel_leaf_m"] >= 0.4
    assert icp["map_voxel_leaf_m"] >= 0.4
    assert icp["robust_registration_enabled"] is True
    assert icp["robust_registration_refinement_iterations"] > 0
    assert surfel["registration_method"] == "voxel_surfel"
    assert 0.2 <= surfel["scan_voxel_leaf_m"] <= 0.4
    assert 0.2 <= surfel["map_voxel_leaf_m"] <= 0.4
    assert surfel["surfel_min_observable_rank"] >= 3
    assert surfel["surfel_max_iterations"] <= 20


def test_geoje_xt32_lidar_imu_pose_graph_profile_contract():
    params = _load_named_params("geoje_xt32_lidar_imu_pose_graph_params.yaml")

    assert params["use_sim_time"] is True
    assert (
        params["primary_cloud_topic"]
        == "/slam_3d/input/hesai_center/points_timed"
    )
    assert params["use_secondary_cloud"] is False
    assert params["use_tertiary_cloud"] is False
    assert params["inter_lidar_motion_compensation_enabled"] is False
    assert params["cloud_sync_queue_size"] >= 64
    assert params["map_publish_period_s"] >= 10.0

    assert params["imu_topic"] == "/slam_3d/input/hesai_front/imu"
    assert params["imu_require_expected_frame"] is True
    assert params["imu_expected_frame"] == "hesai_front"
    assert params["deskew_enabled"] is True
    assert params["deskew_wait_for_imu_watermark"] is True
    assert params["deskew_imu_wait_timeout_s"] >= 0.25
    assert params["deskew_point_time_field"] == "offset_time"
    assert params["deskew_point_time_scale"] == 0.0
    assert 0.0 < params["deskew_max_imu_gap_s"] < params[
        "deskew_max_scan_duration_s"
    ]
    assert 0.0 < params["deskew_min_time_ratio"] <= 1.0
    assert params["reject_cloud_on_deskew_failure"] is True
    imu_to_base_rpy = (
        params["deskew_imu_to_base_roll_deg"],
        params["deskew_imu_to_base_pitch_deg"],
        params["deskew_imu_to_base_yaw_deg"],
    )
    assert all(isinstance(value, (int, float)) for value in imu_to_base_rpy)
    assert any(abs(value) > 1.0e-3 for value in imu_to_base_rpy)

    assert params["registration_method"] == "voxel_surfel"
    assert params["place_recognition_enabled"] is True
    assert params["place_recognition_periodic_enabled"] is True
    assert params["place_recognition_require_pose_graph_correction"] is True
    assert params["pose_graph_correction_enabled"] is True
    assert params["pose_graph_planar_correction_enabled"] is True
    assert params["pose_graph_robust_loop_enabled"] is True
    assert params["pose_graph_loop_cauchy_scale"] > 0.0
    assert 0.0 < params["pose_graph_min_new_loop_robust_weight"] <= 1.0
    assert params["pose_graph_max_loop_residual_before"] > 0.0
    assert params["pose_graph_max_residual_after"] > 0.0
    assert 0.0 < params["pose_graph_max_residual_ratio"] <= 1.0
    assert params["pose_graph_loop_translation_weight"] > 0.0
    assert params["pose_graph_loop_rotation_weight"] > 0.0
    assert (
        params["pose_graph_marker_topic"]
        == "/slam_3d/backend/pose_graph_markers"
    )


def test_geoje_xt32_main_imu_no_wheel_profile_contract():
    params = _load_named_params(
        "geoje_xt32_main_imu_no_wheel_pose_graph_params.yaml"
    )

    assert params["use_sim_time"] is True
    assert (
        params["primary_cloud_topic"]
        == "/slam_3d/input/hesai_center/points_timed"
    )
    assert params["use_secondary_cloud"] is False
    assert params["use_tertiary_cloud"] is False
    assert params["inter_lidar_motion_compensation_enabled"] is False

    assert params["wheel_odom_topic"] == "/unused/wheel_odom_disabled"
    assert params["use_wheel_prediction"] is False
    assert params["use_constant_velocity_prediction"] is True
    assert params["use_motion_prior_fallback"] is False

    assert params["imu_topic"] == "/imu/data"
    assert params["imu_require_expected_frame"] is True
    assert params["imu_expected_frame"] == "cv7_link"
    assert params["deskew_enabled"] is True
    assert params["deskew_wait_for_imu_watermark"] is True
    assert params["reject_cloud_on_deskew_failure"] is True
    assert params["deskew_max_imu_gap_s"] == 0.03
    assert params["deskew_imu_to_base_roll_deg"] == 0.0
    assert params["deskew_imu_to_base_pitch_deg"] == 0.0
    assert params["deskew_imu_to_base_yaw_deg"] == 0.0
    assert params["deskew_angular_velocity_scale"] == 1.0
    assert params["imu_rotation_prediction_enabled"] is False
    assert params["gravity_roll_pitch_constraint_enabled"] is True
    assert params["gravity_accel_scale"] == -1.0
    assert params["gravity_accel_time_constant_s"] == 0.5
    assert params["gravity_accel_min_norm_mps2"] == 7.5
    assert params["gravity_accel_max_norm_mps2"] == 12.5
    assert params["gravity_accel_max_age_s"] == 0.03
    assert params["planar_pose_constraint_enabled"] is True
    assert params["planar_pose_max_z_correction_m"] == 0.0
    assert params["ground_vehicle_motion_constraint_enabled"] is True
    assert params["ground_vehicle_nominal_max_body_z_step_m"] == 0.0
    assert params["ground_vehicle_roughness_max_body_z_step_m"] == 0.0

    assert params["registration_method"] == "voxel_surfel"
    assert params["surfel_relative_eigen_threshold"] == 0.0003
    assert params["place_recognition_enabled"] is True
    assert params["pose_graph_correction_enabled"] is True
    assert params["pose_graph_robust_loop_enabled"] is True
