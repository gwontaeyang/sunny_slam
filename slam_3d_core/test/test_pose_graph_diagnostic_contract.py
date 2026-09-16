import re
from pathlib import Path


CORE_ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = CORE_ROOT / "include" / "slam_3d_core" / "scan_to_map_icp.hpp"
SOURCE_PATH = CORE_ROOT / "src" / "scan_to_map_icp.cpp"


def test_rejected_place_result_preserves_every_pose_graph_diagnostic():
    header = HEADER_PATH.read_text(encoding="utf-8")
    source = SOURCE_PATH.read_text(encoding="utf-8")

    registration_result_start = header.index("struct RegistrationResult")
    registration_result_end = header.index("struct DeskewStats")
    registration_result = header[registration_result_start:registration_result_end]
    pose_graph_fields = set(
        re.findall(
            r"\b(?:bool|int|double|std::string)\s+(pose_graph_[A-Za-z0-9_]+)\b",
            registration_result,
        )
    )

    restore_start = source.index(
        "const RegistrationResult rejected_place_result = result;"
    )
    restore_end = source.index("attach_input_stats(result);", restore_start)
    restore_block = re.sub(r"\.\s+", ".", source[restore_start:restore_end])
    copied_fields = set(
        re.findall(
            r"result\.(pose_graph_[A-Za-z0-9_]+)\s*=\s*"
            r"rejected_place_result\.\1\s*;",
            restore_block,
        )
    )

    # A rejected candidate must remain unapplied after restoring the accepted
    # frontend pose; every diagnostic explaining that rejection must survive.
    expected_fields = pose_graph_fields - {"pose_graph_correction_applied"}
    assert copied_fields == expected_fields


def test_backend_and_metric_availability_propagate_to_health():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", source)
    assert "stats.solver_backend = optimization.backend;" in normalized
    assert (
        "stats.solver_gradient_available = !optimization.stages.empty() && "
        "optimization.stages.back().gradient_available;"
    ) in normalized
    assert (
        "stats.solver_step_counts_available = !optimization.stages.empty() && "
        "std::all_of( optimization.stages.begin(), optimization.stages.end(),"
    ) in normalized
    assert "return stage.step_counts_available;" in normalized

    for field, reset in (
        ("solver_backend", '"not_run"'),
        ("solver_gradient_available", "false"),
        ("solver_step_counts_available", "false"),
    ):
        assert f"result.pose_graph_{field} = stats.{field};" in normalized
        assert f"result.pose_graph_{field} = {reset};" in normalized
        assert re.search(
            rf'add_value\(\s*"pose_graph_{field}",\s*result\.pose_graph_{field}\b',
            source,
        )
    assert 'add_value("pose_graph_compiled_solvers", pose_graph_compiled_solvers_);' in source
    assert "availablePoseGraphSolvers()" in source


def test_nonlinear_dispatch_and_status_are_not_hardcoded_to_ceres():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    start = source.index("ScanToMapIcp::PoseGraphStats ScanToMapIcp::optimizePoseGraphWithLoop(")
    end = source.index("void ScanToMapIcp::updatePathFromKeyframeCorrections(", start)
    optimizer = source[start:end]
    assert 'if (pose_graph_solver_ != "legacy_relaxation")' in optimizer
    assert "PoseGraphOptimizer(pose_graph_optimizer_config_).optimize" in optimizer
    assert 'stats.solver_backend = "legacy_relaxation";' in optimizer
    assert 'return finalize("accepted", convergence_reason);' in optimizer
    assert 'return finalize("failed", convergence_reason);' in optimizer
    assert 'return finalize("rejected", "dcs_outlier_gate");' in optimizer
    assert 'pose_graph_solver_ : stats.solver_backend' in optimizer
    assert '"ceres_' not in optimizer


def test_failed_solver_records_timing_before_early_return():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    start = source.index("const PoseGraphOptimizationResult optimization =")
    end = source.index('return finalize("failed", convergence_reason);', start)
    failure_path = source[start:end]
    assert "stats.iteration_runtime_ms = optimization.runtime_ms;" in failure_path
    assert "stats.mean_iteration_runtime_ms = optimization.runtime_ms /" in failure_path
    assert "reject_new_loop(optimization.failure_reason);" in failure_path
