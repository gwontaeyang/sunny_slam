from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_node_wrapper_is_thin_and_leaves_worker_ownership_to_node_class():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox_node.cpp"
    ).read_text(encoding="utf-8")

    assert "node->configure();" in source
    assert "node->loadPoseGraphByParams();" in source
    assert "rclcpp::spin(node->get_node_base_interface());" in source
    shutdown = source.index("rclcpp::shutdown();", source.index("rclcpp::spin("))
    node_shutdown = source.index("node->shutdown();", shutdown)
    node_reset = source.index("node.reset();", node_shutdown)
    assert shutdown < node_shutdown < node_reset
    assert "node->run()" not in source


def test_explicit_shutdown_breaks_upstream_node_cycles_before_main_returns():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "void shutdown() noexcept;" in header
    assert "std::atomic<bool> shutdown_started_{false};" in header
    shutdown = source.index("RoughnessAwareSlamToolbox::shutdown() noexcept")
    join_worker = source.index("scan_worker_.join();", shutdown)
    join_upstream = source.index("thread->join();", join_worker)
    clear_threads = source.index("threads_.clear();", join_upstream)
    reset_mapper = source.index("smapper_.reset();", clear_threads)
    reset_dataset = source.index("dataset_.reset();", reset_mapper)
    reset_solver = source.index("solver_.reset();", reset_dataset)
    assert join_worker < join_upstream < clear_threads
    assert clear_threads < reset_mapper < reset_dataset < reset_solver


def test_node_wrapper_keeps_upstream_stack_preflight_before_configuration():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox_node.cpp"
    ).read_text(encoding="utf-8")

    preflight = source.index('declare_parameter(\n        "stack_size_to_use"')
    construct = source.index(
        "std::make_shared<slam_2d_core::RoughnessAwareSlamToolbox>"
    )
    assert preflight < construct
    assert "getrlimit(RLIMIT_STACK" in source
    assert "setrlimit(RLIMIT_STACK" in source


def test_online_policy_observes_and_only_exact_bag_stamps_can_be_rejected():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("enable_roughness_innovation_gate", false);' in source
    assert 'declare_parameter("roughness_innovation_mode", "observe");' in source
    assert 'declare_parameter("roughness_experimental_reject_scan_stamps_ns", "");' in source
    assert "window->complete" in source
    assert "window->sustained" in source
    assert "roughnessInnovationDecision(" in source
    assert 'mode_ != "observe"' in source
    assert 'action=allow' in source
    assert 'mode_ == "reject"' not in source


def test_imu_window_uses_both_sides_of_scan_timestamp():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "scan_stamp_ns - static_cast<std::int64_t>(window_before_s_" in source
    assert "scan_stamp_ns + static_cast<std::int64_t>(window_after_s_" in source
    assert "imu_samples_.back().stamp_ns > upper_ns" in source


def test_scan_selection_and_odom_pose_are_captured_before_imu_delay():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    get_odom = source.index("pose_helper_->getOdomPose")
    should_process = source.index("shouldProcessScan(scan, pose)")
    enqueue_pending = source.index("pending_scans_.push_back")
    assert get_odom < should_process < enqueue_pending
    assert "pending.odom_pose" in source


def test_preview_uses_a_karto_equivalent_incremental_running_scan_buffer():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "preview_running_scans_[sensor_name]" in source
    assert "preview_last_scans_[sensor_name]" in source
    assert "running_scans.push_back(processed_scan)" in source
    assert "manager->GetAllScans()" not in source


def test_exact_near_chain_accounting_excludes_scans_not_added_to_graph():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    outcome_block = source.index(
        "if (exact_candidate_observer_requested && exact_candidate_batch.available)"
    )
    update = source.index(
        "updateKartoNearChainCandidateOutcomes(", outcome_block
    )
    exact_count = source.index(
        "near_chain_exact_candidate_count_.fetch_add(", update
    )
    logging = source.index("ROUGHNESS_ADAPTIVE_MEMORY", exact_count)
    block = source[outcome_block:logging]
    assert update < exact_count < logging
    assert "processed &&\n      exact_candidate_batch.eligible_chain_count" in block
    assert "(!processed || !exact_candidate_batch.outcome_parity)" not in block
    assert (
        "processed && index < exact_candidate_batch.candidates.size()"
        in source[logging:]
    )
    assert '"mode=%s scan_processed=%s' in source[logging:]
    assert "sensor_name_token.begin(), sensor_name_token.end()" in source[logging:]
    assert "sensor_name_token.c_str()" in source[logging:]


def test_near_chain_gate_is_opt_in_and_restores_the_mapper_parameter():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("roughness_enable_near_chain_gate", false);' in source
    assert 'declare_parameter("roughness_near_chain_recovery_duration_s", 2.0);' in source
    assert 'declare_parameter("roughness_near_chain_min_rough_samples", 1);' in source
    assert (
        'declare_parameter("roughness_near_chain_gated_min_response", 1.1);'
        in source
    )
    assert (
        '"roughness_disable_loop_closing_during_near_chain_gate", false'
        in source
    )
    assert "ScopedLinkMatchMinimumResponse threshold_override" in source
    assert "ScopedLoopClosingSuppression loop_closing_suppression" in source
    assert "threshold_override.restore()" in source
    assert "loop_closing_suppression.restore()" in source
    assert "ROUGHNESS_NEAR_CHAIN_DIAG" in source


def test_rotation_consistency_guard_is_observed_before_optional_edges_are_added():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("roughness_enable_rotation_consistency_guard", false);' in source
    assert "NearChainRotationConsistency near_chain_rotation_consistency_;" in header
    observe = source.index("observeKartoNearChainCandidates(")
    evaluate = source.index("near_chain_rotation_consistency_.evaluate(", observe)
    threshold = source.index("deriveDynamicResponseThreshold(", evaluate)
    process = source.index("mapper->Process(range_scan.get(), &covariance)", threshold)
    assert observe < evaluate < threshold < process
    assert "rotation_decision.intervention_reject" in source
    assert "rotation_decision.fail_open" in source


def test_roughness_place_scan_match_defaults_to_observe_and_soft_is_quality_gated():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert (
        'declare_parameter("roughness_enable_place_scan_match_observer", false);'
        in source
    )
    assert "enable_roughness_place_scan_match_observer_{false};" in header
    assert (
        'declare_parameter("roughness_place_evaluate_both_directions", false);'
        in source
    )
    assert "roughness_place_evaluate_both_directions_{false};" in header
    assert (
        'declare_parameter("roughness_place_enable_reference_chain_observer", false);'
        in source
    )
    assert "roughness_place_enable_reference_chain_observer_{false};" in header
    assert "ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT" in source
    assert "ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR" in source
    assert "ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS" in source
    assert "ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION" in source
    assert "roughness_place_reference_chain_observed_passages_" in header
    assert "quality_accepted_correlated_duplicate" in source
    assert "stable_side_reacquired_odom_clone_v3" in source
    assert "roughness_place_reference_chain_max_state_gap" in source
    assert "roughness_place_reference_chain_max_time_gap_s" in source
    assert "selectRoughnessPlaceConstraintHypothesis(" in source
    assert "ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS" in source
    assert "observeDirectionalPlaceScanMatch(" in source
    assert "mapper->GetLoopScanMatcher()" in source
    assert "mapper->GetSequentialScanMatcher()" in source
    assert source.count("observeDirectionalPlaceScanMatch(") >= 3
    assert "measurement_vs_odometry_translation_m" in source
    assert "bidirectional_translation_error_m" in source
    assert "forward_alignment_translation_error_m" in source
    assert 'declare_parameter("roughness_place_constraint_mode", "observe");' in source
    assert 'roughness_place_constraint_mode_{"observe"};' in header
    assert "roughness_place_constraint_anchor_strategy" in source
    assert 'roughness_place_constraint_anchor_strategy_{"best_single"};' in header
    ranked = source.index("rankedRoughnessPlaceCandidates(")
    quality = source.index("evaluateRoughnessPlaceConstraint(", ranked)
    observe_guard = source.index(
        'if (roughness_place_constraint_mode_ == "observe")', quality
    )
    correlation_guard = source.index(
        "correlated_passage_duplicate", observe_guard
    )
    duplicate_guard = source.index("graphAlreadyConnects(", correlation_guard)
    add_edge = source.index("graph->AddEdge(", duplicate_guard)
    correct = source.index("mapper->CorrectPoses();", add_edge)
    assert (
        ranked
        < quality
        < observe_guard
        < correlation_guard
        < duplicate_guard
        < add_edge
        < correct
    )
    assert 'constraint_reason = "observe_only";' in source
    assert 'constraint_reason = "existing_group_edge_preserved";' in source
    assert "ROUGHNESS_PLACE_CANDIDATE_DECISION" in source
    assert "ROUGHNESS_PLACE_CONSTRAINT" in source
    assert "ROUGHNESS_PLACE_CONSTRAINT_EDGE" in source


def test_roughness_place_constraint_parameters_are_validated_and_fail_open():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'roughness_place_candidate_top_k must be >= 1' in source
    assert 'roughness_place_min_state_separation must be >= 1' in source
    assert "validRoughnessPlaceConstraintThresholds(" in source
    assert 'roughness_place_constraint_mode must be \'observe\' or \'soft\'' in source
    assert "roughness_place_constraint_anchor_strategy must be" in source
    assert "roughness_place_constraint_mode=soft requires" in source
    assert "roughness_enable_place_scan_match_observer=true" in source
    assert "roughness_place_evaluate_both_directions=true requires" in source
    assert "roughness_place_reference_chain_scan_count must be >= 2" in source
    assert "validRoughnessPlaceScanChainConfig(" in source
    assert "roughness_place_enable_reference_chain_observer=true requires" in source
    assert "requires roughness_place_constraint_mode=observe" in source
    assert "observe-only until atomic graph insertion is available" in source
    assert 'constraint_reason = "missing_or_invalid_backend_evidence";' in source


def test_validated_pair_is_observe_only_and_preflights_the_diagnostic_group():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    strategy = source.index(
        'roughness_place_constraint_anchor_strategy_ == "validated_pair"'
    )
    append_pre = source.index('"pre",', strategy)
    append_post = source.index('"post",', append_pre)
    preflight = source.index("bool invalid_backend_evidence", append_post)
    graph_scan_check = source.index("!graphContainsScan", preflight)
    group_duplicate_check = source.index("graphAlreadyConnects(", graph_scan_check)
    add_edge = source.index("graph->AddEdge(", group_duplicate_check)
    insertion_end = source.index("if (!pending_constraints.empty())", add_edge)
    insertion = source[add_edge:insertion_end]

    assert strategy < append_pre < append_post < preflight
    assert preflight < graph_scan_check < group_duplicate_check < add_edge
    assert insertion.count("mapper->CorrectPoses();") == 1
    assert "constraint_group_partial" in insertion
    assert '"validated_pair_inserted"' in insertion


def test_legacy_fixed_memory_requires_an_explicit_mode():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")
    fallback_start = source.index(
        "if (roughness_constraint_mode_name_.empty())"
    )
    fallback_end = source.index(
        "else if (!parseRoughnessConstraintMode", fallback_start
    )
    fallback = source[fallback_start:fallback_end]

    assert (
        "roughness_constraint_mode_ = "
        "RoughnessConstraintMode::kInterventionOff;"
        in fallback
    )
    assert "enable_near_chain_gate_ ?" not in fallback
    assert "legacy_fixed_memory explicitly" in fallback


def test_spatial_passage_hysteresis_is_opt_in_and_refreshes_only_latest_stamp():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert (
        'declare_parameter("roughness_enable_spatial_passage_hysteresis", '
        "false);"
        in source
    )
    assert 'declare_parameter("roughness_spatial_passage_radius_m", 0.55);' in source
    assert 'declare_parameter("roughness_spatial_passage_max_age_s", 3.5);' in source
    assert (
        "roughness_enable_spatial_passage_hysteresis requires both"
        in source
    )
    assert "roughnessSpatialPassageDecision(" in source
    assert "existing.last_rough_stamp_ns = std::max(" in source
    assert "existing.x_m =" not in source
    assert "existing.y_m =" not in source
    assert "existing.rough_stamp_ns =" not in source


def test_pose_graph_spatial_anchors_are_opt_in_and_record_only_processed_scans():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert (
        'declare_parameter("roughness_spatial_memory_use_pose_graph_anchors", '
        "false);"
        in source
    )
    assert "std::vector<RoughnessPoseGraphAnchor> roughness_pose_graph_anchors_" in header
    process_scan = source.index("RoughnessAwareSlamToolbox::processAndTrackScan(")
    process = source.index("mapper->Process(range_scan.get(), &covariance)", process_scan)
    dataset_add = source.index("dataset_->Add(processed_scan);", process)
    track = source.index("trackProcessedScanLocked(processed_scan, mapper);", dataset_add)
    remember = source.index(
        "rememberPoseGraphSpatialAnchorLocked(processed_scan, stamp_ns);", track
    )
    assert process < dataset_add < track < remember


def test_pose_graph_spatial_query_uses_latest_odom_to_corrected_transform():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    predictor = source.index(
        "RoughnessAwareSlamToolbox::predictPoseGraphSpatialPoseLocked("
    )
    last_odom = source.index("found->second->GetOdometricPose()", predictor)
    last_corrected = source.index("found->second->GetCorrectedPose()", last_odom)
    transform = source.index(
        "karto::Transform odom_to_corrected(last_odom, last_corrected)",
        last_corrected,
    )
    prediction = source.index(
        "odom_to_corrected.TransformPose(current_odom)", transform
    )
    assert predictor < last_odom < last_corrected < transform < prediction


def test_pose_graph_spatial_anchors_follow_loop_updates_and_fail_open():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "mapper->GetAllProcessedScans()" in source
    assert "refreshRoughnessPoseGraphAnchors(" in source
    assert "ROUGHNESS_POSE_GRAPH_ANCHOR_PRUNE" in source
    assert "spatial_query_pose_valid = false;" in source
    assert "spatial_memory_pose_graph_prediction_failure_count_" in source


def test_queue_clear_preserves_spatial_anchors_but_mapper_reset_clears_them():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    clear_start = source.index("RoughnessAwareSlamToolbox::clearQueueCallback(")
    clear_end = source.index(
        "RoughnessAwareSlamToolbox::finalizePendingScansCallback(", clear_start
    )
    assert "roughness_pose_graph_anchors_.clear()" not in source[clear_start:clear_end]

    reset_start = source.index("RoughnessAwareSlamToolbox::resetMapperDerivedState(")
    assert "roughness_pose_graph_anchors_.clear()" in source[reset_start:]


def test_mapper_reload_invalidates_cached_scan_pointers():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "deserializePoseGraphCallback(" in header
    assert "clearQueueCallback(" in header
    assert "frontend_queue_mutex_" in header
    assert "mapper_epoch_mutex_" in header
    assert "std::deque<QueuedScan> scan_queue_" in header
    assert "WindowDecision decision;" in header
    assert "scan_decisions_" not in header
    assert "std::lock_guard<std::mutex> epoch_lock(mapper_epoch_mutex_)" in source
    assert "mapper_after != mapper_before" in source
    assert "pending_scans_.clear()" in source
    assert "scan_queue_.clear()" in source
    assert 'resetMapperDerivedState("mapper_deserialize_complete")' in source
    assert "did not replace the mapper" in source
    assert "roughness state." in source
    assert "preview_last_scans_.clear();" in source
    assert "preview_running_scans_.clear();" in source


def test_scan_worker_binds_generation_to_queue_item_before_laser_access():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "public slam_toolbox::SlamToolbox" in header
    assert "public slam_toolbox::SynchronousSlamToolbox" not in header
    assert "std::thread scan_worker_" in header
    assert "scan_worker_.join();" in source


def test_scan_worker_drain_diagnostics_include_in_flight_completion():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    worker = source.index("void RoughnessAwareSlamToolbox::runScanWorker()")
    pop = source.index("scan_queue_.pop_front();", worker)
    in_flight = source.index("scan_worker_in_flight_.fetch_add(1U);", pop)
    process = source.index("(void)processQueuedScan(*queued);", in_flight)
    completed = source.index("scan_worker_completed_.fetch_add(1U);", process)
    released = source.index("scan_worker_in_flight_.fetch_sub(1U);", completed)

    assert pop < in_flight < process < completed < released
    assert "std::atomic<std::uint64_t> scan_worker_in_flight_{0};" in header
    assert "std::atomic<std::uint64_t> scan_worker_completed_{0};" in header
    assert "scan_worker_in_flight=%llu " in source
    assert '"scan_worker_completed=%llu duplicate_imu=%llu "' in source

    process_queued = source.index(
        "RoughnessAwareSlamToolbox::processQueuedScan("
    )
    generation_check = source.index(
        "queued.decision.mapper_generation != current_generation",
        process_queued,
    )
    get_laser = source.index("getLaser(queued.scan)", process_queued)
    process_window = source.index(
        "processScanWithWindow(laser, queued.scan", process_queued
    )
    assert generation_check < get_laser < process_window


def test_lossless_replay_scan_qos_is_explicit_and_opt_in():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("roughness_scan_qos_reliable", false);' in source
    assert 'declare_parameter("roughness_scan_qos_depth", 5);' in source
    assert "rmw_qos_profile_sensor_data" in source
    assert "RMW_QOS_POLICY_RELIABILITY_RELIABLE" in source
    reset_filter = source.index("scan_filter_.reset();")
    reset_subscriber = source.index("scan_filter_sub_.reset();", reset_filter)
    recreate_subscriber = source.index(
        "std::make_unique<message_filters::Subscriber", reset_subscriber
    )
    recreate_filter = source.index(
        "std::make_unique<tf2_ros::MessageFilter", recreate_subscriber
    )
    register_callback = source.index(
        "scan_filter_->registerCallback", recreate_filter
    )
    assert (
        reset_filter
        < reset_subscriber
        < recreate_subscriber
        < recreate_filter
        < register_callback
    )


def test_lossless_replay_imu_qos_and_stamp_audit_are_explicit_and_opt_in():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("roughness_imu_qos_reliable", false);' in source
    assert 'declare_parameter("roughness_imu_qos_depth", 2000);' in source
    assert "rclcpp::SensorDataQoS().keep_last(imu_qos_depth_)" in source
    assert "imu_qos.reliable();" in source
    assert "received_imu_stamp_audit_.record(stamp_ns);" in source
    assert "++received_imus_;" in source
    assert "OrderedStampAudit received_imu_stamp_audit_;" in header
    assert "received_imu_stamp_fingerprint_fnv1a64" in source


def test_replay_can_finalize_pending_windows_only_after_input_audit():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert "finalizePendingScansCallback(" in header
    assert '"slam_toolbox/finalize_roughness_pending"' in source
    callback = source.index(
        "RoughnessAwareSlamToolbox::finalizePendingScansCallback("
    )
    decision = source.index("windowDecisionLocked(", callback)
    clear = source.index("pending_scans_.clear();", callback)
    queue = source.index("queueAcceptedScan(item.first, item.second);", callback)
    assert callback < decision < clear < queue
    assert "ROUGHNESS_PENDING_FINALIZE" in source


def test_spatial_zone_capacity_evicts_oldest_latest_observation():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    remember_zone = source.index(
        "RoughnessAwareSlamToolbox::rememberRoughZoneLocked("
    )
    eviction = source.index("std::min_element(", remember_zone)
    latest_stamp = source.index("lhs.last_rough_stamp_ns > 0", eviction)
    erase = source.index("roughness_spatial_zones_.erase(oldest)", eviction)
    assert eviction < latest_stamp < erase


def test_scan_conditioning_is_slam_owned_opt_in_and_keeps_laserscan_backend():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package = (ROOT / "package.xml").read_text(encoding="utf-8")

    assert 'declare_parameter("enable_roughness_scan_conditioning", false);' in source
    assert 'declare_parameter("roughness_scan_conditioning_mode", "observe");' in source
    assert "sensor_msgs::msg::LaserScan::ConstSharedPtr" in header
    assert "roughness_scan_conditioner" in cmake
    assert "amcl_guard" not in source + header + cmake + package
    assert "lidar_merger" not in source + header + cmake + package


def test_conditioned_scan_precedes_all_scan_matching_and_default_bypasses_it():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    process = source.index("RoughnessAwareSlamToolbox::processScanWithWindow(")
    default_path = source.index(
        "slam_toolbox::SlamToolbox::addScan(laser, scan, odom_pose)", process
    )
    condition = source.index("conditionScan(scan, window)", default_path)
    preview = source.index("previewScanMatchInnovation(", condition)
    process_scan = source.index(
        "processAndTrackScan(laser, effective_scan", condition
    )
    assert default_path < condition < process_scan < preview


def test_scan_header_tracks_the_scan_actually_admitted_to_the_backend():
    header = (
        ROOT / "include" / "slam_2d_core" / "roughness_aware_slam_toolbox.hpp"
    ).read_text(encoding="utf-8")
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    callback = source.index("RoughnessAwareSlamToolbox::laserCallback(")
    process = source.index("RoughnessAwareSlamToolbox::processScanWithWindow(")
    callback_body = source[callback:process]
    assert "scan_header = scan->header;" not in callback_body

    default_path = source.index("if (!enabled_)", process)
    default_stamp = source.index("updateAdmittedScanHeader(scan->header);", default_path)
    condition = source.index("conditionScan(scan, window)", default_stamp)
    admitted_stamp = source.index(
        "updateAdmittedScanHeader(effective_scan->header);", condition
    )
    backend = source.index("processAndTrackScan(", admitted_stamp)
    assert default_stamp < condition < admitted_stamp < backend

    assert "void updateAdmittedScanHeader(const std_msgs::msg::Header & header);" in header
    helper = source.index("RoughnessAwareSlamToolbox::updateAdmittedScanHeader(")
    helper_end = source.index(
        "RoughnessAwareSlamToolbox::processScanWithWindow(", helper
    )
    helper_body = source[helper:helper_end]
    mapper_lock = helper_body.index("scoped_lock mapper_lock(smapper_mutex_)")
    transform_lock = helper_body.index(
        "scoped_lock transform_lock(map_to_odom_mutex_)"
    )
    assignment = helper_body.index("scan_header = header;")
    assert mapper_lock < transform_lock < assignment
    assert source.count("scan_header =") == 1


def test_conditioning_reference_age_covers_processed_candidate_cadence():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert (
        'declare_parameter("roughness_scan_conditioning_stabilize_max_source_age_s", '
        "0.75);"
        in source
    )
    assert (
        'declare_parameter("roughness_scan_conditioning_residual_max_source_age_s", '
        "0.75);"
        in source
    )


def test_stable_scan_warp_uses_exact_times_and_fixed_odom_frame():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    transform = source.index("RoughnessAwareSlamToolbox::stableScanTransform(")
    current_time = source.index("current.header.stamp", transform)
    stable_time = source.index("stable.header.stamp", current_time)
    lookup = source.index("tf_->lookupTransform(", stable_time)
    fixed_frame = source.index("odom_frame_", lookup)
    assert transform < current_time < stable_time < lookup < fixed_frame


def test_scan_conditioning_state_resets_at_queue_and_mapper_epoch_boundaries():
    source = (
        ROOT / "src" / "roughness_aware_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    clear = source.index("RoughnessAwareSlamToolbox::clearQueueCallback(")
    finalize = source.index(
        "RoughnessAwareSlamToolbox::finalizePendingScansCallback(", clear
    )
    mapper_reset = source.index("RoughnessAwareSlamToolbox::resetMapperDerivedState(")
    diagnostics = source.index("RoughnessAwareSlamToolbox::logDiagnostics(")
    assert 'resetMapperDerivedState("queue_cleared")' in source[clear:finalize]
    assert "resetScanConditioningState(reason)" in source[mapper_reset:diagnostics]
