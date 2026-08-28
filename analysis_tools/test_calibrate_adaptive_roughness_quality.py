import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("calibrate_adaptive_roughness_quality.py")
SPEC = importlib.util.spec_from_file_location(
    "calibrate_adaptive_roughness_quality", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def candidate(
    label,
    index,
    *,
    bag="",
    memory=1.0,
    response=0.2,
    residual=1.0,
):
    inverse_index = 20 - index
    return {
        "label": label,
        "bag": bag,
        "mode": "intervention_off",
        "instrumentation_valid": "1",
        "mapper_generation": "0",
        "stamp_ns": str(1_000_000 + index),
        "candidate_index": str(index),
        "source_unique_id": str(index),
        "target_unique_id": str(100 + index),
        "chain_size_accepted": "true",
        "edge_inserted": "true",
        "final_residual_status": "ok",
        "final_residual_source": "serialized_pose_graph_edge",
        "final_edge_evidence_valid": "1",
        "final_edge_exporter_sha256": "e" * 64,
        "normalized_response": str(response + 0.001 * index),
        "translation_disagreement_m": str(0.01 + 0.002 * inverse_index),
        "yaw_disagreement_rad": str(0.001 + 0.0002 * inverse_index),
        "xy_covariance_max_eigenvalue_m2": str(0.01 + 0.001 * inverse_index),
        "covariance_yawyaw": str(0.001 + 0.0001 * inverse_index),
        "memory_strength": str(memory),
        "final_covariance_weighted_residual_sq": str(residual + index),
        "baseline_threshold": "0.08",
    }


def control(label, bag="", count=20):
    return {
        "label": label,
        "bag": bag,
        "mode": "intervention_off",
        "instrumentation_valid": "1",
        "log_integrity_valid": "1",
        "runtime_parameter_match": "1",
        "imu_delivery_complete": "1",
        "scan_delivery_complete": "1",
        "queue_drain_complete": "1",
        "exact_edge_evidence_valid": "1",
        "pose_graph_coverage_status": "complete",
        "lossless_adapter_sha256": "a" * 64,
        "runtime_slam_node_sha256": "b" * 64,
        "runtime_cmake_cache_sha256": "f" * 64,
        "replay_pose_graph_exporter_sha256": "c" * 64,
        "slam_config_sha256": "d" * 64,
        "exact_edge_exporter_sha256": "e" * 64,
        "exact_candidate_count": str(count),
    }


def args(**updates):
    defaults = {
        "minimum_controls": 2,
        "minimum_active_candidates": 10,
        "minimum_active_datasets": 2,
        "minimum_selected_candidates": 1,
        "minimum_residual_enrichment": 0.0,
    }
    defaults.update(updates)
    return argparse.Namespace(**defaults)


def test_quality_bounds_and_risk_are_finite_and_bounded():
    rows = [candidate("A", index) for index in range(20)]
    bounds = MODULE.derive_quality_bounds(rows)
    risks = [MODULE.candidate_risk(row, bounds) for row in rows]

    assert MODULE.bounds_are_valid(bounds)
    assert all(0.0 <= risk <= 1.0 for risk in risks)
    assert max(risks) > 0.0


def test_calibration_is_deterministic_and_requires_paired_replay():
    summaries = [control("A"), control("B")]
    rows = [
        candidate(label, index, response=0.1 + 0.01 * index, residual=20 - index)
        for label in ("A", "B")
        for index in range(20)
    ]

    first = MODULE.calibrate(summaries, rows, args())
    second = MODULE.calibrate(summaries, rows, args())

    assert first["status"] == "paired_replay_required"
    assert first["production_ready"] is False
    assert first["parameters"] == second["parameters"]
    assert first["parameters"]["roughness_near_chain_quality_calibrated"] is True


def test_missing_active_data_fails_open():
    summaries = [control("A"), control("B")]
    rows = [
        candidate(label, index, memory=0.0)
        for label in ("A", "B")
        for index in range(20)
    ]

    result = MODULE.calibrate(summaries, rows, args())

    assert result["status"] == "insufficient_shadow_evidence"
    assert result["parameters"] == {
        "roughness_near_chain_quality_calibrated": False
    }


def test_reconstructed_residual_is_excluded_from_calibration():
    row = candidate("A", 1)
    row["final_residual_source"] = "reconstructed_pose_raw_scan_covariance"

    assert MODULE.select_candidates([row], [control("A", count=1)]) == []


def test_control_selection_rejects_intervention_run():
    bad = control("A")
    bad["mode"] = "adaptive_quality_aware"

    with pytest.raises(ValueError, match="invalid control"):
        MODULE.select_control_summaries([bad], ["A"])


def test_control_selection_accepts_distinct_bags_under_one_label():
    rows = [control("field", "A"), control("field", "B")]

    assert MODULE.select_control_summaries(rows, ["field"]) == rows


def test_control_selection_rejects_duplicate_label_and_bag():
    rows = [control("field", "A"), control("field", "A")]

    with pytest.raises(ValueError, match="control datasets are not unique"):
        MODULE.select_control_summaries(rows, ["field"])


def test_control_selection_rejects_mixed_adapter_provenance():
    rows = [control("field", "A"), control("field", "B")]
    rows[1]["lossless_adapter_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="mixed or missing control provenance"):
        MODULE.select_control_summaries(rows, ["field"])


def test_residual_diagnostic_is_normalized_per_bag_not_shared_label():
    rows = [
        candidate("field", index, bag="A", residual=float(index))
        for index in range(10)
    ]
    rows.extend(
        candidate("field", index, bag="B", residual=100.0 + index)
        for index in range(10)
    )

    flags = MODULE.high_residual_flags(rows)

    assert sum(flags[:10]) == 1
    assert sum(flags[10:]) == 1


def test_scored_candidates_identify_only_threshold_crossings():
    rows = [candidate("A", index) for index in range(3)]
    scored = MODULE.scored_candidate_rows(rows, [0.1, 0.4, 0.8], 0.4)

    assert [row["nominated_for_paired_replay"] for row in scored] == [0, 1, 1]
    assert all("high_residual_diagnostic" in row for row in scored)


def test_candidate_selection_rejects_stale_bag_and_duplicate_identity():
    summaries = [control("field", "A", count=1)]
    stale = candidate("field", 0, bag="B")
    with pytest.raises(ValueError, match="do not belong"):
        MODULE.select_candidates([stale], summaries)

    row = candidate("field", 0, bag="A")
    summaries[0]["exact_candidate_count"] = "2"
    with pytest.raises(ValueError, match="duplicate exact candidates"):
        MODULE.select_candidates([row, dict(row)], summaries)


def test_candidate_selection_rejects_exporter_provenance_mismatch():
    summary = control("field", "A", count=1)
    row = candidate("field", 0, bag="A")
    row["final_edge_exporter_sha256"] = "9" * 64
    assert MODULE.select_candidates([row], [summary]) == []
