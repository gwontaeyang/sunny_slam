from pathlib import Path


def test_sync_wrapper_does_not_start_a_second_processing_thread():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "roughness_anchor_slam_toolbox_node.cpp"
    ).read_text(encoding="utf-8")

    assert "node->configure();" in source
    assert "node->loadPoseGraphByParams();" in source
    assert "rclcpp::spin(node->get_node_base_interface());" in source
    assert "node->run()" not in source
    assert "processing_thread" not in source


def test_loop_closure_diagnostics_are_opt_in_and_cover_acceptance_phases():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "roughness_anchor_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("enable_loop_closure_diagnostics", false);' in source
    assert 'phase = "coarse"' in source
    assert 'phase = "fine"' in source
    assert 'phase = "rejected"' in source
    assert "phase=begin" in source
    assert "phase=end" in source
    assert "AddListener(&loopClosureDiagnostics())" in source


def test_loop_closure_diagnostics_are_detached_while_serializing():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "roughness_anchor_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    callback = source.split(
        "bool RoughnessAnchorSlamToolbox::serializePoseGraphCallback", 1
    )[1].split(
        "bool RoughnessAnchorSlamToolbox::deserializePoseGraphCallback", 1
    )[0]

    assert "detachLoopClosureDiagnostics()" in callback
    assert "SynchronousSlamToolbox::serializePoseGraphCallback" in callback
    assert "attachLoopClosureDiagnostics()" in callback
    assert "RemoveListener(&loopClosureDiagnostics())" in source


def test_loaded_pose_graph_optimization_is_opt_in():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "roughness_anchor_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("optimize_loaded_pose_graph_once", false);' in source
    assert "if (optimize_loaded_pose_graph_once_)" in source
    assert "mapper->CorrectPoses();" in source
    assert "REVISIT_OPT_DIAG mode=loaded_once" in source


def test_revisit_constraint_filter_preserves_local_edges_and_is_opt_in():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "roughness_anchor_slam_toolbox.cpp"
    ).read_text(encoding="utf-8")

    assert 'declare_parameter("revisit_optimizer_filter_long_edges", false);' in source
    assert "state_gap < revisit_optimizer_min_state_id_separation_" in source
    assert "time_gap_s < revisit_optimizer_min_time_separation_s_" in source
    assert "covariance_gate_failed" in source
    assert "solver->RemoveConstraint(" in source
    assert "phase=constraint_filter" in source
