import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("summarize_adaptive_roughness_replays.py")
SPEC = importlib.util.spec_from_file_location(
    "summarize_adaptive_roughness_replays",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate_row(**updates):
    row = {
        "edge_inserted": "true",
        "source_unique_id": "10",
        "target_unique_id": "11",
        "relative_x_m": "0.5",
        "relative_y_m": "0.0",
        "relative_yaw_rad": "0.0",
        "covariance_xx": "0.25",
        "covariance_xy": "0.0",
        "covariance_xyaw": "0.0",
        "covariance_yy": "1.0",
        "covariance_yyaw": "0.0",
        "covariance_yawyaw": "1.0",
    }
    row.update(updates)
    return row


def pose_graph_edge_row(**updates):
    row = {
        "edge_index": "42",
        "source_unique_id": "10",
        "target_unique_id": "11",
        "relative_x_m": "0.5",
        "relative_y_m": "0.0",
        "relative_yaw_rad": "0.0",
        "covariance_xx": "0.25",
        "covariance_yy": "1.0",
        "covariance_yawyaw": "1.0",
        "final_residual_x_m": "0.1",
        "final_residual_y_m": "-0.2",
        "final_residual_yaw_rad": "0.03",
        "final_covariance_weighted_residual_sq": "7.5",
    }
    row.update(updates)
    return row


def complete_summary(label, bag, mode, reference_label=""):
    row = {
        "label": label,
        "bag": bag,
        "mode": mode,
        "reference_label": reference_label,
        "instrumentation_valid": 1,
        "exact_edge_evidence_valid": 1,
        "runtime_parameter_match": 1,
        "imu_delivery_complete": 1,
        "scan_delivery_complete": 1,
        "queue_drain_complete": 1,
        "pose_graph_coverage_status": "complete",
        "pose_graph_node_count": 10,
        "quality_calibrated": int(mode == "adaptive_quality_aware"),
        "comparison_status": "compared",
        "fairness_status": "matched_with_explicit_experiment_difference",
        "reference_node_count": 10,
        "parity_failure_count": 0,
        "node_count_delta": 0,
        "map_occupied_changed_cell_ratio": 0.0,
        "odom_consistent_p95_delta_reference_minus_variant_m": 0.0,
        "return_revisit_p95_delta_reference_minus_variant_m": 0.0,
    }
    for field in MODULE.PAIR_PROVENANCE_FIELDS:
        row[field] = "a" * 64
    return row


def test_relative_pose_uses_source_local_frame_and_wraps_yaw():
    relative = MODULE.relative_pose(
        (1.0, 2.0, math.pi / 2.0),
        (1.0, 3.0, -3.0 * math.pi / 2.0),
    )
    assert relative == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)


def test_final_factor_residual_uses_exact_serialized_edge_evidence():
    row = candidate_row()
    edges = MODULE.final_edges_by_unique_ids([pose_graph_edge_row()])

    MODULE.add_final_factor_residual(row, {}, edges)

    assert row["final_residual_status"] == "ok"
    assert row["final_residual_source"] == "serialized_pose_graph_edge"
    assert row["final_pose_graph_edge_index"] == "42"
    assert row["final_residual_x_m"] == pytest.approx(0.1)
    assert row["final_residual_y_m"] == pytest.approx(-0.2)
    assert row["final_residual_yaw_rad"] == pytest.approx(0.03)
    assert row["final_residual_translation_m"] == pytest.approx(
        math.hypot(0.1, -0.2)
    )
    assert row["final_covariance_weighted_residual_sq"] == pytest.approx(7.5)


def test_reconstructed_final_factor_residual_is_marked_approximate():
    row = candidate_row()
    poses = {
        10: (1.0, 2.0, math.pi / 2.0),
        11: (1.0, 3.0, math.pi / 2.0),
    }

    MODULE.add_final_factor_residual(row, poses)

    assert row["final_residual_status"] == "approximate_covariance_frame"
    assert row["final_residual_source"] == (
        "reconstructed_pose_raw_scan_covariance"
    )
    assert row["final_residual_x_m"] == pytest.approx(0.5)
    assert row["final_residual_y_m"] == pytest.approx(0.0, abs=1e-12)
    assert row["final_residual_yaw_rad"] == pytest.approx(0.0)
    assert math.isnan(row["final_covariance_weighted_residual_sq"])
    assert row[
        "approximate_final_covariance_weighted_residual_sq"
    ] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "edge,status",
    (
        (None, "missing_final_edge"),
        (
            pose_graph_edge_row(relative_x_m="0.7"),
            "edge_measurement_mismatch",
        ),
        (
            pose_graph_edge_row(final_residual_x_m="nan"),
            "invalid_final_edge_residual",
        ),
    ),
)
def test_exact_final_factor_residual_requires_matching_valid_edge(edge, status):
    edges = MODULE.final_edges_by_unique_ids([] if edge is None else [edge])
    row = candidate_row()
    MODULE.add_final_factor_residual(row, {}, edges)
    assert row["final_residual_status"] == status
    assert math.isnan(row["final_covariance_weighted_residual_sq"])


def test_exact_edge_evidence_requires_matching_sidecar_hashes(tmp_path):
    pose_graph = tmp_path / "final_pose_graph.posegraph"
    edge_csv = tmp_path / "pose_graph_edges_final.csv"
    provenance = tmp_path / "pose_graph_edges_final.provenance.json"
    trajectory = tmp_path / "trajectory_pose_graph_final.csv"
    pose_graph.write_bytes(b"pose-graph")
    trajectory.write_text("stamp_s,x_m\n1.0,0.0\n", encoding="utf-8")
    with edge_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=sorted(MODULE.REQUIRED_EDGE_FIELDS),
        )
        writer.writeheader()
        writer.writerow(pose_graph_edge_row(edge_index="0"))
    provenance.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "exported",
                "trajectory_equivalent": 1,
                "edge_count": 1,
                "edge_csv_sha256": MODULE.file_sha256(edge_csv),
                "pose_graph_sha256": MODULE.file_sha256(pose_graph),
                "trajectory_sha256": MODULE.file_sha256(trajectory),
                "exporter_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )

    evidence = MODULE.exact_edge_evidence(tmp_path, MODULE.read_csv(edge_csv))
    assert evidence["valid"] is True
    assert evidence["status"] == "valid"

    edge_csv.write_text("tampered\n", encoding="utf-8")
    evidence = MODULE.exact_edge_evidence(tmp_path, MODULE.read_csv(edge_csv))
    assert evidence["valid"] is False
    assert evidence["status"] == "edge_provenance_mismatch"


@pytest.mark.parametrize(
    "updates,status",
    (
        ({"edge_inserted": "false"}, "not_inserted"),
        ({"source_unique_id": "99"}, "missing_final_pose"),
        ({"relative_x_m": "nan"}, "invalid_measurement"),
        ({"covariance_xx": "nan"}, "invalid_covariance"),
    ),
)
def test_final_factor_residual_reports_diagnostic_status(updates, status):
    row = candidate_row(**updates)
    MODULE.add_final_factor_residual(
        row,
        {10: (0.0, 0.0, 0.0), 11: (1.0, 0.0, 0.0)},
    )
    assert row["final_residual_status"] == status
    assert math.isnan(row["final_covariance_weighted_residual_sq"])


def test_bootstrap_mean_ci_is_deterministic_for_identical_seed():
    first = MODULE.bootstrap_mean_ci([1.0, 2.0, 4.0, 8.0], 1000, 42)
    second = MODULE.bootstrap_mean_ci([1.0, 2.0, 4.0, 8.0], 1000, 42)
    assert first == second
    assert first[0] == pytest.approx(3.75)
    assert first[1] <= first[0] <= first[2]


def test_paired_summary_excludes_unfair_or_uncompared_rows():
    control_a = complete_summary("control", "A", "intervention_off")
    adaptive_a = complete_summary(
        "adaptive", "A", "adaptive_quality_aware", "control"
    )
    adaptive_a["odom_consistent_p95_delta_reference_minus_variant_m"] = 0.2
    adaptive_a["return_revisit_p95_delta_reference_minus_variant_m"] = 0.1
    control_b = complete_summary("control", "B", "intervention_off")
    adaptive_b = complete_summary(
        "adaptive", "B", "adaptive_quality_aware", "control"
    )
    adaptive_b["fairness_status"] = "mismatched"
    adaptive_b["odom_consistent_p95_delta_reference_minus_variant_m"] = -99.0
    adaptive_b["return_revisit_p95_delta_reference_minus_variant_m"] = -99.0
    rows = [control_a, adaptive_a, control_b, adaptive_b]
    summary = MODULE.paired_metric_summary(rows, 100, 7)
    assert len(summary) == 2
    assert {row["dataset_count"] for row in summary} == {1}
    assert {row["mean"] for row in summary} == {0.1, 0.2}


def test_paired_summary_reports_sign_counts_and_outlier_sensitivity():
    rows = []
    for bag, value in (("A", 0.0), ("B", 0.2), ("C", -0.1), ("D", 4.0)):
        control = complete_summary("control", bag, "intervention_off")
        adaptive = complete_summary(
            "adaptive", bag, "adaptive_quality_aware", "control"
        )
        adaptive["odom_consistent_p95_delta_reference_minus_variant_m"] = value
        adaptive["return_revisit_p95_delta_reference_minus_variant_m"] = value
        rows.extend((control, adaptive))

    summary = MODULE.paired_metric_summary(rows, 100, 7)

    assert len(summary) == 2
    for row in summary:
        assert row["improved_count"] == 2
        assert row["unchanged_count"] == 1
        assert row["regressed_count"] == 1
        assert row["largest_absolute_effect_bag"] == "D"
        assert row["largest_absolute_effect"] == pytest.approx(4.0)
        assert row["mean_without_largest_absolute_effect"] == pytest.approx(
            1.0 / 30.0
        )


def test_paired_rows_accept_only_explicitly_compatible_adapter_hashes():
    control = complete_summary("control", "A", "intervention_off")
    adaptive = complete_summary(
        "adaptive", "A", "adaptive_quality_aware", "control"
    )
    control["lossless_adapter_sha256"] = "a" * 64
    adaptive["lossless_adapter_sha256"] = "b" * 64

    assert MODULE.paired_adaptive_rows([control, adaptive]) == []
    assert len(MODULE.paired_adaptive_rows(
        [control, adaptive],
        {"a" * 64, "b" * 64},
    )) == 1

    adaptive["runtime_slam_node_sha256"] = "c" * 64
    assert MODULE.paired_adaptive_rows(
        [control, adaptive],
        {"a" * 64, "b" * 64},
    ) == []


def test_stationary_single_node_replay_is_complete_evidence():
    row = complete_summary("control", "stationary", "intervention_off")
    row["pose_graph_coverage_status"] = "stationary_single_node_complete"
    assert MODULE.complete_replay_evidence(row) is True


def test_compatible_adapter_hash_validation_is_fail_closed():
    assert "a" * 64 in MODULE.compatible_lossless_adapter_hashes(["a" * 64])
    with pytest.raises(ValueError, match="invalid compatible lossless adapter"):
        MODULE.compatible_lossless_adapter_hashes(["not-a-sha256"])


def test_candidate_action_summary_counts_only_eligible_actions_and_fail_open():
    rows = [
        {
            "chain_size_accepted": "false",
            "shadow_action": "accepted",
            "fail_open": "true",
            "backend_action": "ineligible_chain",
        },
        {
            "chain_size_accepted": "true",
            "shadow_action": "accepted",
            "fail_open": "false",
            "backend_action": "pose_fused_edge_inserted",
            "baseline_response_accepted": "true",
            "applied_response_accepted": "true",
        },
        {
            "chain_size_accepted": "true",
            "shadow_action": "downweighted",
            "fail_open": "false",
            "backend_action": "pose_fused_edge_inserted",
            "baseline_response_accepted": "true",
            "applied_response_accepted": "true",
        },
        {
            "chain_size_accepted": "true",
            "shadow_action": "rejected",
            "fail_open": "false",
            "backend_action": "rejected_response",
            "baseline_response_accepted": "true",
            "applied_response_accepted": "false",
        },
    ]

    summary = MODULE.candidate_action_summary(rows)

    assert summary == {
        "eligible_candidate_count": 3,
        "ineligible_candidate_count": 1,
        "accepted_candidate_count": 1,
        "downweighted_candidate_count": 1,
        "rejected_candidate_count": 1,
        "adaptive_response_rejected_candidate_count": 1,
        "baseline_response_rejected_candidate_count": 0,
        "backend_edge_inserted_candidate_count": 2,
        "backend_duplicate_edge_candidate_count": 0,
        "backend_distance_rejected_candidate_count": 0,
        "fail_open_candidate_count": 0,
        "raw_fail_open_candidate_count": 1,
        "backend_response_rejected_count": 1,
    }


def test_paired_runtime_rows_matches_control_by_bag():
    control = complete_summary("control", "A03", "intervention_off")
    control.update(
        run_wall_s=10.0,
        queue_drain_wait_s=2.0,
        replay_completion_wall_s=12.0,
    )
    stale = complete_summary("stale_control", "A03", "intervention_off")
    stale.update(
        run_wall_s=90.0,
        queue_drain_wait_s=10.0,
        replay_completion_wall_s=100.0,
    )
    adaptive = complete_summary(
        "adaptive", "A03", "adaptive_quality_aware", "control"
    )
    adaptive.update(
        run_wall_s=10.0,
        queue_drain_wait_s=3.5,
        replay_completion_wall_s=13.5,
    )
    rows = [control, stale, adaptive]

    result = MODULE.paired_runtime_rows(rows)

    assert len(result) == 1
    assert result[0]["control_label"] == "control"
    assert result[0]["adaptive_minus_control_completion_wall_s"] == pytest.approx(1.5)


def test_comparison_reference_label_uses_variant_label_parent():
    comparison = {
        "stage31_dir": "/results/variant/control_release/bags/A03",
    }
    assert MODULE.comparison_reference_label(comparison) == "control_release"
    assert MODULE.comparison_reference_label({}) == ""


def test_pose_graph_overlay_uses_map_gauge_and_creates_point_plot(tmp_path):
    (tmp_path / "map.pgm").write_bytes(
        b"P5\n4 3\n255\n" + bytes([205, 254, 0, 254] * 3)
    )
    (tmp_path / "map.yaml").write_text(
        "image: map.pgm\n"
        "resolution: 0.5\n"
        "origin: [1.0, 2.0, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.25\n",
        encoding="utf-8",
    )
    (tmp_path / "trajectory_pose_graph_final.csv").write_text(
        "stamp_s,x_m,y_m,yaw_rad,source,state_id,unique_id,"
        "odom_x_m,odom_y_m,odom_yaw_rad\n"
        "1.0,1.2,2.2,0.0,pose_graph_final,0,0,1.2,2.2,0.0\n"
        "2.0,2.2,2.7,0.0,pose_graph_final,1,1,2.1,2.8,0.0\n",
        encoding="utf-8",
    )

    output = MODULE.plot_pose_graph_map_overlay(tmp_path)

    assert output == str(tmp_path / "map_with_pose_graph_trajectory.png")
    assert Path(output).stat().st_size > 0

    Path(output).unlink()
    separated_output = tmp_path / "analysis" / "map_overlay.png"
    output = MODULE.plot_pose_graph_map_overlay(tmp_path, separated_output)

    assert output == str(separated_output)
    assert separated_output.stat().st_size > 0
    assert not (tmp_path / "map_with_pose_graph_trajectory.png").exists()


def test_select_replay_paths_filters_exact_labels(tmp_path):
    paths = []
    for label in ("smoke", "A03_release", "A27_release"):
        path = tmp_path / "variant" / label / "bags" / "bag" / "replay_result.json"
        path.parent.mkdir(parents=True)
        path.touch()
        paths.append(path)

    selected = MODULE.select_replay_paths(
        reversed(paths),
        ["A03_release", "A27_release"],
    )

    assert [path.parents[2].name for path in selected] == [
        "A03_release",
        "A27_release",
    ]


def test_select_replay_paths_rejects_missing_requested_label(tmp_path):
    path = tmp_path / "variant" / "A03_release" / "bags" / "bag" / "replay_result.json"

    with pytest.raises(ValueError, match="H01_release"):
        MODULE.select_replay_paths([path], ["A03_release", "H01_release"])


def test_reference_chain_summary_preserves_counts_metrics_and_paths():
    replay = {
        "roughness_place_reference_chain_event_count": "4",
        "roughness_place_reference_chain_event_available_count": 3,
        "roughness_place_reference_chain_decision_accepted_count": 2,
        "roughness_place_reference_chain_decision_"
        "correlated_duplicate_count": (
            1
        ),
        "roughness_place_reference_chain_evaluation_duration_ms_p95": "42.5",
        "roughness_place_reference_chain_decisions_csv": "/tmp/decisions.csv",
        "roughness_place_constraint_edge_injected_count": 0,
        "roughness_place_graph_action_count": 0,
    }

    summary = MODULE.reference_chain_summary(replay)

    assert summary["reference_chain_event_count"] == 4
    assert summary["reference_chain_event_available_count"] == 3
    assert summary["reference_chain_decision_accepted_count"] == 2
    assert summary[
        "reference_chain_decision_correlated_duplicate_count"
    ] == 1
    assert summary[
        "reference_chain_evaluation_duration_ms_p95"
    ] == pytest.approx(42.5)
    assert summary["reference_chain_decisions_csv"] == "/tmp/decisions.csv"
    assert summary["place_constraint_edge_injected_count"] == 0
    assert math.isnan(summary["reference_chain_quality_risk_p95"])


def test_reference_chain_plot_writes_shadow_outcomes_and_latency(tmp_path):
    summary = {
        "bag": "A03",
        "label": "chain_observe",
        "reference_chain_decision_count": 4,
        "reference_chain_decision_available_count": 3,
        "reference_chain_decision_accepted_count": 2,
        "reference_chain_decision_correlated_duplicate_count": 1,
        "reference_chain_decision_fail_open_count": 1,
        "reference_chain_evaluation_duration_ms_median": 10.0,
        "reference_chain_evaluation_duration_ms_p95": 20.0,
        "reference_chain_evaluation_duration_ms_max": 25.0,
    }
    output = tmp_path / "plots" / "reference_chain.png"

    MODULE.plot_reference_chain_evidence([summary], output)

    assert output.stat().st_size > 0


def test_main_can_write_analysis_artifacts_outside_replay_root(
    tmp_path, monkeypatch
):
    replay_root = tmp_path / "replays"
    artifact_root = tmp_path / "field_control_analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--output-root",
            str(replay_root),
            "--artifact-root",
            str(artifact_root),
        ],
    )

    assert MODULE.main() == 0
    assert (
        artifact_root / "report" / "adaptive_roughness_memory_report.md"
    ).is_file()
    assert (artifact_root / "csv" / "adaptive_replay_summary.csv").is_file()
    assert (artifact_root / "plots" / "reference_chain_evidence.png").is_file()
    assert not (replay_root / "report").exists()
    report = (
        artifact_root / "report" / "adaptive_roughness_memory_report.md"
    ).read_text(encoding="utf-8")
    assert "Status: not evaluated" in report
    assert "development gate passed" not in report


def test_observe_only_report_lists_selected_run_and_point_only_overlay(tmp_path):
    row = complete_summary("A03_chain_observe", "A03", "intervention_off")
    row.update(
        {
            "status": "map_and_map_trajectory",
            "exact_candidate_count": 12,
            "eligible_candidate_count": 10,
            "scan_received_count": 100,
            "scan_effective_expected_count": 100,
            "map_with_pose_graph_trajectory_png": "/tmp/A03_point_only.png",
        }
    )
    report_path = tmp_path / "report" / "observe.md"

    MODULE.make_report(report_path, [row], [], [], [])

    report = report_path.read_text(encoding="utf-8")
    assert "### Selected replay artifacts" in report
    assert (
        "| A03 | A03_chain_observe | intervention_off | 10 | 12/10 |"
        in report
    )
    assert "## Primary map overlays" in report
    assert "/tmp/A03_point_only.png" in report


def test_decision_gate_is_derived_from_selected_targeted_replays():
    rows = []
    for prefix in ("A03", "A27", "C_Q1", "C_Q3", "H01"):
        control_label = f"control_{prefix}"
        rows.append(complete_summary(control_label, prefix, "intervention_off"))
        rows.append(complete_summary(
            f"{prefix}_final",
            prefix,
            "adaptive_quality_aware",
            control_label,
        ))
    lines = MODULE.decision_gate_lines(rows)
    assert any("gate: passed" in line for line in lines)

    rows[-1]["map_occupied_changed_cell_ratio"] = 0.01
    lines = MODULE.decision_gate_lines(rows)
    assert any("gate: failed" in line for line in lines)
    assert any("H01 exact no-op: False" in line for line in lines)


def test_decision_gate_rejects_uncalibrated_or_materially_worse_a03():
    rows = []
    for prefix in ("A03", "A27", "C_Q1", "C_Q3", "H01"):
        control_label = f"control_{prefix}"
        rows.append(complete_summary(control_label, prefix, "intervention_off"))
        rows.append(complete_summary(
            f"{prefix}_final",
            prefix,
            "adaptive_quality_aware",
            control_label,
        ))
    a03 = next(row for row in rows if row.get("label") == "A03_final")
    a03["odom_consistent_p95_delta_reference_minus_variant_m"] = -1.0
    assert any("gate: failed" in line for line in MODULE.decision_gate_lines(rows))

    a03["odom_consistent_p95_delta_reference_minus_variant_m"] = 0.0
    a03["quality_calibrated"] = 0
    assert any("gate: failed" in line for line in MODULE.decision_gate_lines(rows))
