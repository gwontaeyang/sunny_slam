import json

import pytest

import analyze_mapping_cloud_refinement as analysis


def record(stamp_ns, status, applied, before_score, after_score):
    return {
        "name": "scan_to_map_icp",
        "stamp_ns": stamp_ns,
        "values": {
            "place_recognition_mapping_refinement_attempted": "true",
            "place_recognition_mapping_refinement_applied": str(applied).lower(),
            "place_recognition_mapping_refinement_status": status,
            "place_recognition_mapping_refinement_before_score": str(before_score),
            "place_recognition_mapping_refinement_after_score": str(after_score),
            "place_recognition_mapping_refinement_before_inlier_rmse_m": "0.2",
            "place_recognition_mapping_refinement_after_inlier_rmse_m": "0.1",
            "place_recognition_mapping_refinement_before_target_coverage_ratio": "0.2",
            "place_recognition_mapping_refinement_after_target_coverage_ratio": "0.3",
            "place_recognition_mapping_refinement_min_target_coverage_ratio": "0.1",
            "place_recognition_mapping_refinement_primary_before_score": "0.8",
            "place_recognition_mapping_refinement_primary_after_score": "0.79",
            "place_recognition_mapping_refinement_correction_translation_m": "0.3",
            "place_recognition_mapping_refinement_correction_rotation_deg": "1.2",
            "place_recognition_mapping_refinement_runtime_ms": "4.5",
            "place_recognition_prediction_jump_translation_m": "2.0",
            "place_recognition_map_correction_translation_m": "2.5",
            "place_recognition_alignment_correction_translation_m": "2.2",
            "place_recognition_alignment_correction_rotation_deg": "4.0",
            "place_recognition_mapping_refinement_keyframe_quarantine_"
            "correction_translation_m": "3.5",
            "place_recognition_accepted": str(applied).lower(),
            "pose_graph_correction_applied": str(applied).lower(),
        },
    }


def test_load_and_summarize_events(tmp_path):
    health = tmp_path / "health.jsonl"
    items = [
        {"name": "scan_to_map_icp", "stamp_ns": 1_000_000_000, "values": {}},
        record(2_000_000_000, "accepted", True, 0.6, 0.4),
        record(3_000_000_000, "score_improvement", False, 0.5, 0.51),
    ]
    health.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")

    events = analysis.load_events(health)
    summary = analysis.summarize(events)

    assert [event["elapsed_s"] for event in events] == [1.0, 2.0]
    assert summary["attempted_count"] == 2
    assert summary["applied_count"] == 1
    assert summary["status_counts"] == {"accepted": 1, "score_improvement": 1}
    assert summary["applied_score_change"]["median"] == pytest.approx(-0.2)
    assert summary["applied_score_reduction"]["median"] == pytest.approx(0.2)
    assert summary["applied_rmse_improvement_m"]["median"] == pytest.approx(0.1)
    assert summary["applied_target_coverage_ratio"]["median"] == pytest.approx(0.3)
    assert events[0]["minimum_target_coverage_ratio"] == pytest.approx(0.1)
    assert events[0]["alignment_correction_translation_m"] == pytest.approx(2.2)
    assert summary["mapping_refinement_pose_graph_applied_count"] == 1
    assert summary["applied_map_correction_translation_m"]["median"] == pytest.approx(
        2.5
    )
