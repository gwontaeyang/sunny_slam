from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def _load_yaml(relative_path):
    package_share = Path(get_package_share_directory("slam_3d_bringup"))
    with (package_share / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_package_yaml(package_name, relative_path):
    package_share = Path(get_package_share_directory(package_name))
    with (package_share / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_temporal_static_confirmation_belongs_to_offline_cleanup():
    pipeline = _load_yaml("config/indoor_pipeline.yaml")
    cleanup = pipeline["offline_map_cleanup"]
    assert cleanup["package"] == "slam_3d_map_cleanup"
    assert cleanup["executable"] == "temporal_map_cleanup"
    assert cleanup["config"] == "config/indoor.yaml"
    assert cleanup["scope"] == "derived_offline_map_only"
    assert cleanup["source"] == "exported_keyframe_archive"
    assert cleanup["hit_unit"] == "synchronized_keyframe_not_sensor_count"
    assert cleanup["per_sensor_ray_origin_preserved"] is True
    assert cleanup["registration_and_loop_authority"] == "unchanged_mapping_cloud"
    assert cleanup["semantic_person_detection"] == "excluded"
    prior_free = cleanup["prior_free_space"]
    assert prior_free["enabled"] is True
    guard = prior_free["ground_grazing_guard"]
    assert guard["scope"] == "first_hit_prior_free_only"
    assert guard["max_sensor_vertical_drop_m"] == 0.9
    assert guard["existing_candidate_ray_miss"] == "unchanged"

    params = _load_package_yaml(cleanup["package"], cleanup["config"])
    assert params["enabled"] is True
    assert params["voxel_size_m"] > 0.0
    assert params["min_hit_observations"] >= 3
    assert params["min_observation_span_s"] > 0.0
    assert params["min_viewpoint_distance_m"] > 0.0
    assert params["ray_clearing_enabled"] is True
    assert params["ray_endpoint_margin_m"] >= params["voxel_size_m"]
    assert params["max_ray_length_m"] > 0.0
    assert params["max_rays_per_observation"] > 0
    assert params["prior_free_enabled"] is True
    assert 1 <= params["min_prior_free_observations"] <= (
        params["max_prior_free_observations"]
    )
    assert params["prior_free_max_sensor_vertical_drop_m"] == (
        guard["max_sensor_vertical_drop_m"]
    )
    assert 0.0 < params["retention_min_hit_ratio"] < (
        params["promotion_min_hit_ratio"]
    ) <= 1.0
    assert params["min_miss_observations_for_demotion"] >= 2


def test_large_scale_pipeline_matches_hesai_geometry_and_time_limits():
    config = _load_yaml("config/large_scale_outdoor_pipeline.yaml")

    assert config["profile"] == "gole_hesai_jt128_xt32_large_scale_outdoor_v1"
    observed = config["observed_geoje_contract"]
    assert observed["front_jt128"]["points"] == 230400
    assert observed["rear_jt128"]["rings"] == 128
    assert observed["center_xt32"]["points"] == 128000
    assert observed["timestamp"]["granularity"].startswith("packet_time")
    assert observed["timestamp"]["common_lidar_imu_clock"] is False
    assert observed["timestamp"]["constant_offset_sufficient"] is False

    ingest = config["sensor_ingest"]
    assert ingest["package"] == "slam_3d_lidar_ingest"
    assert "geometric_lidar_merge" in ingest["explicitly_not_owned"]
    assert ingest["geoje_legacy_time"]["policy"] == "calibrated_sensor_time"
    assert ingest["geoje_legacy_time"]["quality"].endswith("salvage_only")

    frontend = config["multi_lidar_frontend"]
    assert frontend["trigger_sensor"] == "center_xt32"
    assert frontend["synchronization"]["consume_once"] is True
    assert frontend["synchronization"]["structure"] == "bounded_per_sensor_queue"
    assert frontend["synchronization"]["decision_watermarks"] == [
        "latest_auxiliary_sensor_stamp"
    ]
    assert (
        frontend["synchronization"][
            "cross_topic_primary_progress_is_dropout_proof"
        ]
        is False
    )
    assert "lookup_sensor_to_base_tf_or_reject_sensor_cloud" in frontend[
        "preprocessing_order"
    ]
    assert "packet_time_not_channel_firing_time" in frontend["current_limits"]
    assert "publish_corrected_channel_firing_time_from_hesai_sdk" in config[
        "production_frontend_target"
    ]["required_before_claiming_tightly_coupled_lio"]


def test_indoor_pipeline_reuses_core_and_excludes_wheel_from_estimator():
    config = _load_yaml("config/indoor_pipeline.yaml")

    assert config["profile"] == "gole_hesai_indoor_lidar_imu_no_wheel_v1"
    assert config["purpose"]["sensors"]["imu"] == "/imu/data"
    assert config["purpose"]["excluded_estimator_inputs"] == ["wheel_odometry"]
    assert config["inputs"]["wheel_odom"] == "disabled"
    assert config["architecture"]["shared_core"]["package"] == "slam_3d_core"
    assert config["architecture"]["shared_core"]["environment_independent"] is True
    assert config["architecture"]["outdoor_profile_layer"]["status"] == "hold"

    workflow = config["development_workflow"]
    assert workflow["shared_core_source_of_truth"] == "slam_3d_core"
    assert workflow["shared_core_edit_policy"] == "single_owner_per_worktree"
    assert workflow["parallel_change_isolation"] == "separate_git_worktree_and_branch"
    assert "shared_core_build_and_tests" in workflow["core_integration_gate"]
    assert "indoor_profile_regression" in workflow["core_integration_gate"]
    assert "duplicated_indoor_and_outdoor_core_source_trees" in workflow["prohibited"]
    assert (
        "concurrent_uncoordinated_edits_to_shared_core_in_one_worktree"
        in workflow["prohibited"]
    )

    profiles = config["sensor_profiles"]
    assert set(profiles) == {"indoor_xt32_single", "indoor_dual_jt128"}
    assert profiles["indoor_xt32_single"]["wheel_prediction"] is False
    assert profiles["indoor_dual_jt128"]["wheel_prediction"] is False
    assert (
        profiles["indoor_dual_jt128"]["inter_lidar_motion_compensation"]
        == "imu_rotation_only"
    )

    assert config["mapping"]["spatial_mode"] == "se3"
    assert config["frontend"]["shock_policy"][
        "quarantine_persistent_map_mutation"
    ] is True
    terrain = config["mapping"]["terrain_evidence"]
    assert terrain["no_return_policy"] == "unknown_not_drop"
    assert terrain["minimum_independent_viewpoint_confirmations"] == 2
    assert terrain["downstream_projection_status"] == (
        "experimental_offline_continuous_support_cost"
    )
    assert config["outputs"]["terrain_evidence"] == "/slam_3d/terrain/evidence"
    assert config["outputs"]["terrain_evidence_colored"] == (
        "/slam_3d/terrain/evidence_colored"
    )
    assert config["outputs"]["terrain_support_evidence"] == (
        "/slam_3d/terrain/support_evidence"
    )
    support = terrain["support_surface"]
    assert support["probable_support_requires_direct_confirmed_anchor"] is True
    assert support["probable_support_recursive_propagation"] == "prohibited"
    assert support["any_hazard_vote_vetoes_support"] is True
    assert support["projection"] == "continuous_cost_separate_from_trinary_occupancy"
    assert terrain["ramp_geometry_gate"]["stage_2"] == (
        "robust_local_3d_plane_support"
    )
    assert terrain["semantic_rgb"]["floor"] == [105, 190, 105]
    assert terrain["semantic_rgb"]["ramp"] == [45, 165, 255]
    assert terrain["semantic_rgb"]["observed_drop"] == [235, 55, 55]
    assert config["outputs"]["terrain_preview_evidence"] == (
        "/slam_3d/terrain/preview_evidence"
    )
    preview = config["mapping"]["terrain_preview_memory"]
    assert preview["status"] == "observer_only"
    assert preview["minimum_ahead_distance_m"] == 0.75
    assert preview["maximum_lateral_distance_m"] == 1.00
    assert preview["pose_graph_edge_mutation"] == "prohibited_in_observer_stage"
    assert preview["unsafe_crossing_policy"] == "conflict_not_confirmation"
    assert config["mapping"]["submaps"]["overlap_required"] is True
    pose_graph = config["mapping"]["pose_graph"]
    assert pose_graph["dimensions"] == "se3"
    assert pose_graph["solver"] == "ceres_se3"
    assert pose_graph["nonlinear_method"] == "levenberg_marquardt"
    assert pose_graph["linear_solver"] == "sparse_normal_cholesky"
    assert pose_graph["robust_scale_schedule"] == [4.0, 2.0, 1.0]
    assert pose_graph["loop_outer_reweight"] == "dynamic_covariance_scaling"
    assert pose_graph["dcs_start_stage"] == 0
    assert pose_graph["information_mode"] == "configured_diagonal"
    assert pose_graph["information_residual_order"] == [
        "tx",
        "ty",
        "tz",
        "rx",
        "ry",
        "rz",
    ]
    assert pose_graph["information_source"] == (
        "configured_translation_and_rotation_weights"
    )
    experimental_information = pose_graph[
        "experimental_registration_observability"
    ]
    assert experimental_information["status"] == "opt_in_replay_only"
    assert experimental_information["mode"] == "registration_observability"
    assert experimental_information["source"] == (
        "robust_fixed_pose_point_to_plane_hessian_shape"
    )
    assert experimental_information["magnitude_policy"] == (
        "configured_diagonal_is_upper_bound"
    )
    assert experimental_information["weak_direction_policy"] == (
        "bounded_positive_floor"
    )
    mismatch_policy = experimental_information[
        "invalid_or_mismatched_linearization_policy"
    ]
    assert mismatch_policy == "configured_diagonal_fallback"
    assert experimental_information["calibrated_covariance_claim"] is False
    assert experimental_information["holdout_regression_gate_required"] is True
    assert pose_graph["commit_only_when_objective_improves"] is True
    assert "temporal_static_map" not in config["mapping"]
    archive = config["mapping"]["keyframe_archive"]
    assert archive["directory_parameter"] == "keyframe_archive_directory"
    assert archive["default_directory"] == ""
    assert archive["empty_directory_policy"] == (
        "disable_archive_scan_retention_and_export"
    )
    assert archive["voxel_leaf_parameter"] == "keyframe_archive_voxel_leaf_m"
    assert archive["default_voxel_leaf_m"] == 0.22
    assert archive["export_service"] == "~/export_keyframes"
    assert archive["default_node"] == "/scan_to_map_icp_node"
    assert archive["service_type"] == "std_srvs/srv/Trigger"
    assert archive["snapshot_scope"] == "retained_keyframes"
    assert archive["pruned_keyframes"] == "reported_in_manifest"
    assert archive["keyframe_ids"] == "stable_monotonic"
    assert archive["source"] == (
        "immutable_full_per_sensor_mapping_views_before_terrain_classification"
    )
    assert archive["terrain_classifier_source_copy"] == "separate_mutable_copy"
    assert archive["per_sensor_ray_origin_preserved"] is True
    assert config["outputs"]["global_map"] == "/slam_3d/map/global_cloud"
    assert config["outputs"]["global_map_semantics"] == (
        "temporal_unfiltered_mapping_cloud"
    )
    assert "temporal_static_map_candidates_debug" not in config["outputs"]
    assert config["frontend"]["deskew"]["required_frame"] == "cv7_link"
    assert config["frontend"]["deskew"]["imu_to_base_rpy_deg"] == [
        0.0,
        0.0,
        0.0,
    ]
    assert config["frontend"]["deskew"]["axis_convention"] == (
        "native_xyz_with_identity_base_link_to_cv7_link_rotation"
    )
    assert config["frontend"]["deskew"]["legacy_roll180_policy"] == (
        "archived_dataset_specific_replay_only"
    )
    assert "xsens_link" in config["frontend"]["deskew"]["not_transferable_to"]
    assert config["validation"]["current_dataset"] == "04_gangnam"
    assert config["validation"]["explicitly_not_replayed"] == ["wheel_odometry"]
    assert "no_wheel_subscription" in config["validation"]["quality_gates"]
    assert "dynamic_object_detection" in config["purpose"]["excluded_processing"]


def test_topic_contract_preserves_actual_point_layout_and_three_sensor_identity():
    contract = _load_yaml("config/hesai_jt128_xt32_topic_contract.yaml")

    assert contract["dataset_evidence"]["source"] == "02_samsung_geoje"
    assert contract["dataset_evidence"]["input_point_step_bytes"] == 26
    assert contract["dataset_evidence"]["input_fields"]["timestamp"] == {
        "offset": 18,
        "type": "FLOAT64",
    }
    assert (
        contract["dataset_evidence"]["timestamp_quality"]["semantic"]
        == "packet_time_not_channel_firing_time"
    )
    assert contract["hesai_raw_inputs"]["front_jt128"]["ring_range"] == [0, 127]
    assert contract["hesai_raw_inputs"]["center_xt32"]["ring_range"] == [0, 31]
    canonical = contract["canonical_outputs"]
    assert canonical["point_step_bytes"] == 24
    assert canonical["fields"]["offset_time"]["type"] == "UINT32"
    assert canonical["front"]["sensor_id"] == 0
    assert canonical["rear"]["sensor_id"] == 1
    assert canonical["center"]["sensor_id"] == 2
    assert contract["core_multi_lidar_sync"]["policy"]["consume_each_auxiliary_scan_at_most_once"]
    missing_decision = contract["core_multi_lidar_sync"]["policy"][
        "auxiliary_missing_decision"
    ]
    assert missing_decision == [
        "per_auxiliary_timestamp_watermark",
        "bounded_wall_timeout_on_stream_stall",
    ]
    assert (
        contract["core_multi_lidar_sync"]["policy"][
            "cross_topic_primary_progress_is_dropout_proof"
        ]
        is False
    )
    assert (
        contract["core_multi_lidar_sync"]["policy"][
            "motion_compensation_failure_policy"
        ]
        == "drop_auxiliary"
    )
    assert contract["clock_policies"]["production"]["note"].endswith(
        "not_proof_of_ptp_lock"
    )
    assert (
        contract["clock_policies"]["geoje_legacy_replay"]["policy"]
        == "calibrated_sensor_time"
    )


def test_hesai_driver_template_uses_one_host_udp_port_per_lidar():
    config = _load_yaml("config/hesai_jt128_xt32_driver.template.yaml")
    ports = [
        entry["driver"]["lidar_udp_type"]["udp_port"]
        for entry in config["lidar"]
    ]

    assert ports == [2368, 2369, 2370]
    assert len(set(ports)) == 3


def test_geoje_replay_uses_fixed_affine_sensor_clock_calibration():
    config = _load_package_yaml(
        "slam_3d_lidar_ingest",
        "config/geoje_legacy_bag_ingest_params.yaml",
    )["hesai_pointcloud_ingest"]["ros__parameters"]

    for sensor in ("front", "rear", "center"):
        prefix = f"{sensor}."
        assert config[prefix + "clock_policy"] == "calibrated_sensor_time"
        assert config[prefix + "clock_domain"] == "rosbag_record_time_affine_fit"
        assert 0.99 < config[prefix + "time_scale"] < 1.01
        assert config[prefix + "anchor_input_header_ns"] > 0
        assert config[prefix + "anchor_output_start_ns"] > 0
        assert (
            config[prefix + "calibration_valid_min_input_header_ns"]
            <= config[prefix + "anchor_input_header_ns"]
            <= config[prefix + "calibration_valid_max_input_header_ns"]
        )
        assert 0.0 < config[prefix + "max_output_stamp_step_s"] <= 2.0
        assert config[prefix + "calibration_fit_rmse_ms"] >= 0.0
        assert (
            config[prefix + "calibration_fit_max_abs_ms"]
            >= config[prefix + "calibration_fit_rmse_ms"]
        )
        assert config[prefix + "calibration_id"].endswith("record_time_fit_v1")
