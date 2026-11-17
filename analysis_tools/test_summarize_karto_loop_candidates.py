from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("summarize_karto_loop_candidates.py")
SPEC = importlib.util.spec_from_file_location("summarize_karto_loop_candidates", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_key_values_keeps_chain_identity() -> None:
    line = (
        "[slam_toolbox]: KARTO_LOOP_CANDIDATE stamp_ns=123 "
        "candidate_index=2 chain_state_ids=1,2,3 coarse_response=0.42 "
        "disposition=fine_rejected"
    )

    parsed = MODULE.parse_key_values(line, MODULE.LOOP_CANDIDATE_MARKER)

    assert parsed is not None
    assert parsed["stamp_ns"] == "123"
    assert parsed["candidate_index"] == "2"
    assert parsed["chain_state_ids"] == "1,2,3"
    assert parsed["disposition"] == "fine_rejected"


def test_parse_log_separates_batches_and_candidates(tmp_path: Path) -> None:
    log = tmp_path / "run_launch.log"
    log.write_text(
        "KARTO_LOOP_CANDIDATE_BATCH stamp_ns=1 mapper_generation=0 "
        "scan_processed=true observer_available=true attempts=1 "
        "accepted_callbacks=0 actual_loop_edges=0 accepted_without_edges=0 "
        "edge_pairing_complete=true "
        "failure=none\n"
        "KARTO_LOOP_CANDIDATE stamp_ns=1 mapper_generation=0 candidate_index=0 "
        "disposition=fine_rejected actual_edge_present=false coarse_response=0.4\n",
        encoding="utf-8",
    )

    batches, candidates = MODULE.parse_log(log)

    assert len(batches) == 1
    assert len(candidates) == 1
    assert batches[0]["attempts"] == "1"
    assert candidates[0]["disposition"] == "fine_rejected"


def test_summary_requires_exact_batch_candidate_and_edge_parity() -> None:
    batches = [
        {
            "scan_processed": "true",
            "observer_available": "true",
            "attempts": "2",
            "accepted_callbacks": "1",
            "actual_loop_edges": "1",
            "accepted_without_edges": "0",
            "edge_pairing_complete": "true",
            "failure": "none",
        }
    ]
    candidates = [
        {
            "disposition": "fine_rejected",
            "actual_edge_present": "false",
            "coarse_response": "0.25",
            "fine_response": "0.2",
        },
        {
            "disposition": "accepted",
            "actual_edge_present": "true",
            "accepted_without_edge": "false",
            "coarse_response": "0.6",
            "fine_response": "0.5",
            "measurement_vs_odom_translation_m": "0.4",
            "measurement_vs_odom_yaw_rad": "3.0",
            "pre_residual_yaw_rad": "2.9",
            "post_residual_yaw_rad": "0.01",
            "post_covariance_weighted_residual_sq": "0.5",
        },
    ]

    summary = MODULE.summarize(batches, candidates, 90.0)

    assert summary["instrumentation_valid"] is True
    assert summary["candidate_count"] == 2
    assert summary["accepted_candidate_count"] == 1
    assert summary["actual_edge_count"] == 1
    assert summary["large_yaw_actual_edge_count"] == 1


def test_summary_marks_pairing_failure_invalid() -> None:
    batches = [
        {
            "scan_processed": "true",
            "observer_available": "true",
            "attempts": "1",
            "accepted_callbacks": "1",
            "actual_loop_edges": "0",
            "accepted_without_edges": "0",
            "edge_pairing_complete": "false",
            "failure": "loop_edge_pairing_incomplete",
        }
    ]
    candidates = [
        {
            "disposition": "accepted",
            "actual_edge_present": "false",
            "accepted_without_edge": "false",
            "coarse_response": "0.6",
            "fine_response": "0.5",
        }
    ]

    summary = MODULE.summarize(batches, candidates, 90.0)

    assert summary["instrumentation_valid"] is False
    assert summary["pairing_failure_batch_count"] == 1
    assert summary["failure_counts"] == {"loop_edge_pairing_incomplete": 1}


def test_graph_initialization_warmup_is_not_pairing_failure() -> None:
    batches = [
        {
            "scan_processed": "true",
            "observer_available": "false",
            "attempts": "0",
            "accepted_callbacks": "0",
            "actual_loop_edges": "0",
            "accepted_without_edges": "0",
            "edge_pairing_complete": "false",
            "failure": "graph_not_initialized",
        }
    ]

    summary = MODULE.summarize(batches, [], 90.0)

    assert summary["instrumentation_valid"] is True
    assert summary["graph_initialization_warmup_batch_count"] == 1
    assert summary["pairing_failure_batch_count"] == 0
    assert summary["observer_unavailable_batch_count"] == 0


def test_accepted_callback_without_karto_edge_is_valid() -> None:
    batches = [
        {
            "scan_processed": "true",
            "observer_available": "true",
            "attempts": "1",
            "accepted_callbacks": "1",
            "actual_loop_edges": "0",
            "accepted_without_edges": "1",
            "edge_pairing_complete": "true",
            "failure": "none",
        }
    ]
    candidates = [
        {
            "disposition": "accepted",
            "actual_edge_present": "false",
            "accepted_without_edge": "true",
            "coarse_response": "0.6",
            "fine_response": "0.5",
        }
    ]

    summary = MODULE.summarize(batches, candidates, 90.0)

    assert summary["instrumentation_valid"] is True
    assert summary["accepted_candidate_count"] == 1
    assert summary["actual_edge_count"] == 0
    assert summary["accepted_without_edge_count"] == 1


def test_deduplicate_ignores_log_location() -> None:
    rows = [
        {"stamp_ns": "1", "candidate_index": "0", "log_path": "a"},
        {"stamp_ns": "1", "candidate_index": "0", "log_path": "b"},
    ]

    unique, duplicate_count = MODULE.deduplicate(
        rows, ("stamp_ns", "candidate_index")
    )

    assert len(unique) == 1
    assert duplicate_count == 1
