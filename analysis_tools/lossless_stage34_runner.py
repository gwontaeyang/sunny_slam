#!/usr/bin/env python3
"""Run the legacy stage34 pipeline with recorded-scan drain protection.

The legacy queue-drain check observes only queues inside the SLAM frontend.
At accelerated bag rates, the ROS subscription can still contain LaserScan
callbacks when those queues become empty.  This adapter waits for the latest
ROUGHNESS_INNOVATION_SUMMARY to reach the bag's recorded scan count before it
allows the normal queue-drain and artifact capture sequence to continue.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_PIPELINE = (
    Path.home()
    / "analysis_outputs/slam_2d/02_roughness_return_resume/scripts"
    / "gole_slam_bag_pipeline.py"
)
ADAPTER_PROVENANCE_SCHEMA = 7
ADAPTER_RUNTIME_PARAMETER_CAPTURE_SCHEMA = 5
SLAM_PARAMETER_CAPTURE_RETRY_TIMEOUT_S = 45.0
POSE_GRAPH_SERIALIZATION_COMPLETION_GRACE_S = 5.0
POSE_GRAPH_SERIALIZATION_STABLE_WINDOW_S = 0.5
POSE_GRAPH_SERIALIZATION_FRESHNESS_TOLERANCE_NS = 1_000_000_000
POSE_GRAPH_RECOVERY_EXPORT_TIMEOUT_S = 60.0
RUNTIME_BUILD_FAIRNESS_FIELD = "slam_2d_runtime_build"
POSE_GRAPH_ANCHOR_PARAMETER = (
    "roughness_spatial_memory_use_pose_graph_anchors"
)
POSE_GRAPH_ANCHOR_DEFAULT = False
POSE_GRAPH_ANCHOR_FAIRNESS_FIELD = (
    "spatial_memory_use_pose_graph_anchors"
)
SCAN_CONDITIONING_FAIRNESS_FIELD = "slam_owned_scan_conditioning"
SCAN_CONDITIONING_DEFAULTS: dict[str, Any] = {
    "enable_roughness_scan_conditioning": False,
    "roughness_scan_conditioning_mode": "observe",
    "roughness_scan_conditioning_use_nearest_imu_sample": True,
    "roughness_scan_conditioning_max_scan_imu_dt_s": 0.20,
    "roughness_scan_conditioning_hold_duration_s": 0.10,
    "roughness_scan_conditioning_hard_hold_duration_s": 0.0,
    "roughness_scan_conditioning_hard_score_gate": 3.0,
    "roughness_scan_conditioning_throttle_min_publish_period_s": 0.50,
    "roughness_scan_conditioning_throttle_drop_hard": True,
    "roughness_scan_conditioning_max_continuous_drop_duration_s": 0.0,
    "roughness_scan_conditioning_stabilize_max_source_age_s": 0.75,
    "roughness_scan_conditioning_stabilize_max_event_duration_s": 0.50,
    "roughness_scan_conditioning_stabilize_min_publish_period_s": 0.10,
    "roughness_scan_conditioning_stabilize_max_translation_m": 0.20,
    "roughness_scan_conditioning_stabilize_max_yaw_rad": 0.15,
    "roughness_scan_conditioning_stabilize_max_roll_pitch_rad": 0.05,
    "roughness_scan_conditioning_stabilize_min_cached_valid_ratio": 0.20,
    "roughness_scan_conditioning_stabilize_min_output_valid_ratio": 0.15,
    "roughness_scan_conditioning_stabilize_min_output_valid_beams": 20,
    "roughness_scan_conditioning_stabilize_fallback_mode": "drop",
    "roughness_scan_conditioning_residual_max_source_age_s": 0.75,
    "roughness_scan_conditioning_residual_max_translation_m": 0.20,
    "roughness_scan_conditioning_residual_max_yaw_rad": 0.15,
    "roughness_scan_conditioning_residual_max_roll_pitch_rad": 0.05,
    "roughness_scan_conditioning_residual_threshold_m": 0.08,
    "roughness_scan_conditioning_residual_neighbor_bins": 1,
    "roughness_scan_conditioning_residual_min_cached_valid_ratio": 0.20,
    "roughness_scan_conditioning_residual_min_overlap_ratio": 0.35,
    "roughness_scan_conditioning_residual_min_overlap_beams": 100,
    "roughness_scan_conditioning_residual_min_output_valid_ratio": 0.20,
    "roughness_scan_conditioning_residual_min_output_valid_beams": 100,
    "roughness_scan_conditioning_residual_direction": "closer",
    "roughness_scan_conditioning_residual_max_removed_ratio": 0.03,
    "roughness_scan_conditioning_residual_promote_filtered_reference": False,
    "roughness_scan_conditioning_residual_fallback_mode": "observe",
    "roughness_scan_conditioning_quarantine_recovery_scan_count": 5,
    "roughness_scan_conditioning_quarantine_max_scan_gap_s": 0.75,
    "roughness_scan_conditioning_quarantine_min_stable_duration_s": 0.16,
    "roughness_scan_conditioning_quarantine_fault_after_s": 3.0,
    "roughness_scan_conditioning_quarantine_fail_open_on_imu_fault": True,
    "roughness_scan_conditioning_quarantine_min_valid_ratio": 0.20,
}
SCAN_CONDITIONING_PARAMETERS = tuple(SCAN_CONDITIONING_DEFAULTS)
ADAPTIVE_CONSTRAINT_FAIRNESS_FIELD = "adaptive_roughness_constraint_profile"
ADAPTIVE_CONSTRAINT_PARAMETERS = (
    "roughness_constraint_mode",
    "roughness_enable_near_chain_candidate_observer",
    "roughness_enable_loop_closure_candidate_observer",
    "roughness_enable_place_recognition_observer",
    "roughness_enable_place_scan_match_observer",
    "roughness_place_evaluate_both_directions",
    "roughness_place_enable_reference_chain_observer",
    "roughness_place_reference_chain_scan_count",
    "roughness_place_reference_chain_max_distance_m",
    "roughness_place_reference_chain_max_state_gap",
    "roughness_place_reference_chain_max_time_gap_s",
    "roughness_place_descriptor_bins",
    "roughness_place_max_observations_per_event",
    "roughness_place_max_events",
    "roughness_place_candidate_top_k",
    "roughness_place_min_revisit_separation_s",
    "roughness_place_min_state_separation",
    "roughness_place_constraint_mode",
    "roughness_place_constraint_anchor_strategy",
    "roughness_place_min_fine_response",
    "roughness_place_max_proposal_disagreement_translation_m",
    "roughness_place_max_proposal_disagreement_yaw_deg",
    "roughness_place_max_fine_xy_covariance_m2",
    "roughness_place_max_fine_yaw_covariance_deg2",
    "roughness_place_max_bidirectional_translation_error_m",
    "roughness_place_max_bidirectional_yaw_error_deg",
    "roughness_place_max_pair_alignment_translation_error_m",
    "roughness_place_max_pair_alignment_yaw_error_deg",
    "roughness_place_constraint_pair_covariance_scale",
    "roughness_place_constraint_group_covariance_scale",
    "roughness_place_constraint_xy_variance_floor_m2",
    "roughness_place_constraint_yaw_variance_floor_deg2",
    "roughness_near_chain_observer_min_parity_candidates",
    "roughness_adaptive_memory_trigger_score",
    "roughness_adaptive_memory_saturation_score",
    "roughness_adaptive_memory_recovery_score",
    "roughness_adaptive_memory_min_duration_s",
    "roughness_adaptive_memory_max_duration_s",
    "roughness_adaptive_memory_min_distance_m",
    "roughness_adaptive_memory_max_distance_m",
    "roughness_adaptive_memory_recovery_time_constant_s",
    "roughness_near_chain_quality_calibrated",
    "roughness_near_chain_quality_response_bad",
    "roughness_near_chain_quality_response_good",
    "roughness_near_chain_quality_translation_good_m",
    "roughness_near_chain_quality_translation_bad_m",
    "roughness_near_chain_quality_yaw_good_deg",
    "roughness_near_chain_quality_yaw_bad_deg",
    "roughness_near_chain_quality_xy_covariance_good_m2",
    "roughness_near_chain_quality_xy_covariance_bad_m2",
    "roughness_near_chain_quality_yaw_covariance_good_rad2",
    "roughness_near_chain_quality_yaw_covariance_bad_rad2",
    "roughness_adaptive_downweight_risk",
    "roughness_adaptive_reject_risk",
    "roughness_adaptive_minimum_information_scale",
    "roughness_adaptive_dynamic_threshold_max",
    "roughness_enable_rotation_consistency_guard",
    "roughness_rotation_history_size",
    "roughness_rotation_minimum_history_batches",
    "roughness_rotation_outlier_sigma",
    "roughness_rotation_extreme_yaw_sigma",
    "roughness_rotation_yaw_scale_floor_deg",
    "roughness_rotation_translation_scale_floor_m",
    "roughness_rotation_weighted_disagreement_scale_floor",
    "roughness_rotation_chain_overlap_trigger",
    "roughness_rotation_maximum_repeat_state_gap",
    "roughness_rotation_stable_recovery_batches",
)
ADAPTER_PATH = Path(__file__).resolve()
ACCEPTED_REPLAY_STATUSES = frozenset(
    {
        "map_and_map_trajectory",
        "map_and_stationary_pose_graph",
    }
)
_RECEIVED_PATTERN = re.compile(r"ROUGHNESS_INNOVATION_SUMMARY\b.*?\breceived=(\d+)")
_WORKER_DRAIN_PATTERN = re.compile(
    r"ROUGHNESS_INNOVATION_SUMMARY\b.*?\bqueued=(\d+).*?"
    r"\bscan_queue_depth=(\d+)\s+scan_worker_in_flight=(\d+)\s+"
    r"scan_worker_completed=(\d+)"
)
_WORKER_DIAGNOSTIC_FIELDS_PATTERN = re.compile(
    r"\s+scan_worker_in_flight=\d+\s+scan_worker_completed=\d+"
)
_PLAYBACK_FINISHED_PATTERN = re.compile(
    r"\[INFO\]\s+\[ros2-\d+\]: process has finished cleanly"
)
_PLAYBACK_FAILED_PATTERN = re.compile(
    r"^\[ERROR\]\s+\[ros2-\d+\]: process has died[^\n]*"
    r"\bros2\s+bag\s+play\b[^\n]*$",
    re.MULTILINE,
)
_SLAM_NODE_FAILED_PATTERN = re.compile(
    r"^\[ERROR\]\s+\[roughness_aware_slam_toolbox_node-\d+\]: "
    r"process has died[^\n]*$",
    re.MULTILINE,
)
_IMU_AUDIT_PATTERN = re.compile(
    r"ROUGHNESS_INNOVATION_SUMMARY[^\n]*\sreceived_imu=(\d+)\s+"
    r"received_imu_first_stamp_ns=(-?\d+)\s+"
    r"received_imu_last_stamp_ns=(-?\d+)\s+"
    r"received_imu_max_positive_gap_ns=(\d+)\s+"
    r"received_imu_stamp_duplicate_count=(\d+)\s+"
    r"received_imu_stamp_out_of_order_count=(\d+)\s+"
    r"received_imu_stamp_fingerprint_fnv1a64=([0-9a-fA-F]{16})"
)
_COMMAND_REPLAY_SIGNATURE_PATTERN = re.compile(
    rb"^replay_signature_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_COMMAND_FAIRNESS_SIGNATURE_PATTERN = re.compile(
    rb"^replay_fairness_contract_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_COMMAND_ROS_DOMAIN_PATTERN = re.compile(
    rb"^ROS_DOMAIN_ID=(\d+)$",
    re.MULTILINE,
)
_RECORDED_IMU_AUDIT_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_LEGACY_SIGNATURE_BY_ADAPTER_SIGNATURE: dict[str, str] = {}
_LEGACY_PAYLOAD_BY_ADAPTER_SIGNATURE: dict[str, dict[str, Any]] = {}
_POSE_GRAPH_DIAGNOSTIC_FIELDS_PATTERN = re.compile(
    r"spatial_pose_source=(?P<source>\S+)\s+"
    r"pose_graph_anchor_mode=(?P<mode>true|false)\s+"
)
_POSE_GRAPH_SUMMARY_COUNTERS = (
    "spatial_pose_graph_refreshes",
    "spatial_pose_graph_stale_removes",
    "spatial_pose_graph_invalid_removes",
    "spatial_pose_graph_prediction_failures",
    "spatial_pose_graph_anchor_inserts",
)
_SCAN_CONDITIONING_SUMMARY_COUNTERS = (
    "scan_conditioning_allow",
    "scan_conditioning_drop",
    "scan_conditioning_bridge",
    "scan_conditioning_quarantine",
    "scan_conditioning_stabilized",
    "scan_conditioning_residual_filtered",
    "scan_conditioning_fallback",
    "scan_conditioning_imu_quality_reject",
    "scan_conditioning_tf_failure",
)
_ROUGHNESS_PLACE_EVENT_COLUMNS = (
    "event_id",
    "pre_stamp_ns",
    "post_stamp_ns",
    "pre_state_id",
    "post_state_id",
    "pre_unique_id",
    "post_unique_id",
    "pre_odom_x_m",
    "pre_odom_y_m",
    "pre_odom_yaw_rad",
    "post_odom_x_m",
    "post_odom_y_m",
    "post_odom_yaw_rad",
    "rough_observations",
    "path_length_m",
    "descriptor_bins",
    "descriptor_channels",
    "descriptor_valid",
    "reference_events",
    "eligible_candidates",
    "candidate_available",
    "candidate_event_id",
    "candidate_direction",
    "candidate_score",
    "candidate_shape_score",
    "candidate_amplitude_score",
    "candidate_length_score",
    "candidate_second_score",
    "candidate_margin",
    "candidate_event_separation_s",
    "candidate_minimum_state_separation",
    "odom_center_distance_m",
    "corrected_center_distance_m",
    "corrected_post_heading_delta_deg",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_ANCHOR_MATCH_COLUMNS = (
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "candidate_direction",
    "anchor_role",
    "reference_state_id",
    "query_state_id",
    "state_id_separation",
    "descriptor_score",
    "descriptor_margin",
    "available",
    "fail_open",
    "failure",
    "forward_available",
    "forward_failure",
    "forward_coarse_response",
    "forward_fine_response",
    "forward_coarse_cov_xx",
    "forward_coarse_cov_xy",
    "forward_coarse_cov_xyaw",
    "forward_coarse_cov_yy",
    "forward_coarse_cov_yyaw",
    "forward_coarse_cov_yawyaw",
    "forward_fine_cov_xx",
    "forward_fine_cov_xy",
    "forward_fine_cov_xyaw",
    "forward_fine_cov_yy",
    "forward_fine_cov_yyaw",
    "forward_fine_cov_yawyaw",
    "forward_proposal_disagreement_translation_m",
    "forward_proposal_disagreement_yaw_rad",
    "reverse_available",
    "reverse_failure",
    "reverse_coarse_response",
    "reverse_fine_response",
    "reverse_coarse_cov_xx",
    "reverse_coarse_cov_xy",
    "reverse_coarse_cov_xyaw",
    "reverse_coarse_cov_yy",
    "reverse_coarse_cov_yyaw",
    "reverse_coarse_cov_yawyaw",
    "reverse_fine_cov_xx",
    "reverse_fine_cov_xy",
    "reverse_fine_cov_xyaw",
    "reverse_fine_cov_yy",
    "reverse_fine_cov_yyaw",
    "reverse_fine_cov_yawyaw",
    "reverse_proposal_disagreement_translation_m",
    "reverse_proposal_disagreement_yaw_rad",
    "odom_relative_x_m",
    "odom_relative_y_m",
    "odom_relative_yaw_rad",
    "matched_relative_x_m",
    "matched_relative_y_m",
    "matched_relative_yaw_rad",
    "measurement_vs_odometry_translation_m",
    "measurement_vs_odometry_yaw_rad",
    "bidirectional_translation_error_m",
    "bidirectional_yaw_error_rad",
    "directional_alignment_translation_error_m",
    "directional_alignment_yaw_error_rad",
    "branch_alignment_x_m",
    "branch_alignment_y_m",
    "branch_alignment_yaw_rad",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_PAIR_MATCH_COLUMNS = (
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "candidate_direction",
    "attempted",
    "available",
    "fail_open",
    "reference_pre_state_id",
    "query_for_reference_pre_state_id",
    "reference_post_state_id",
    "query_for_reference_post_state_id",
    "method",
    "forward_alignment_translation_error_m",
    "forward_alignment_yaw_error_rad",
    "reverse_alignment_translation_error_m",
    "reverse_alignment_yaw_error_rad",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT_COLUMNS = (
    "mapper_generation",
    "event_id",
    "available",
    "fail_open",
    "failure",
    "pre_anchor_state_id",
    "post_anchor_state_id",
    "pre_chain_count",
    "pre_chain_first_state_id",
    "pre_chain_last_state_id",
    "pre_chain_anchor_state_gap",
    "pre_chain_anchor_distance_m",
    "pre_chain_anchor_time_gap_s",
    "post_chain_count",
    "post_chain_first_state_id",
    "post_chain_last_state_id",
    "post_reacquisition_reset_count",
    "post_reacquisition_state_gap",
    "post_reacquisition_distance_m",
    "post_reacquisition_time_gap_s",
    "candidate_count",
    "method",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR_COLUMNS = (
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "hypothesis_index",
    "descriptor_direction",
    "evaluated_direction",
    "anchor_role",
    "available",
    "fail_open",
    "failure",
    "reference_base_scan_count",
    "query_base_scan_count",
    "reference_first_state_id",
    "reference_last_state_id",
    "query_first_state_id",
    "query_last_state_id",
    "reference_maximum_anchor_distance_m",
    "query_maximum_anchor_distance_m",
    "forward_coarse_response",
    "forward_fine_response",
    "reverse_coarse_response",
    "reverse_fine_response",
    "forward_fine_cov_xx",
    "forward_fine_cov_yy",
    "forward_fine_cov_yawyaw",
    "reverse_fine_cov_xx",
    "reverse_fine_cov_yy",
    "reverse_fine_cov_yawyaw",
    "bidirectional_translation_error_m",
    "bidirectional_yaw_error_rad",
    "method",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS_COLUMNS = (
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "hypothesis_index",
    "descriptor_direction",
    "evaluated_direction",
    "reference_context_available",
    "available",
    "fail_open",
    "failure",
    "quality_accepted",
    "quality_reason",
    "quality_available",
    "quality_risk",
    "minimum_fine_response",
    "forward_pair_translation_error_m",
    "forward_pair_yaw_error_rad",
    "reverse_pair_translation_error_m",
    "reverse_pair_yaw_error_rad",
    "selected",
    "selection_fail_open",
    "selection_reason",
    "method",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION_COLUMNS = (
    "mapper_generation",
    "event_id",
    "available",
    "fail_open",
    "quality_accepted",
    "reason",
    "candidate_event_id",
    "candidate_rank",
    "evaluated_direction",
    "quality_risk",
    "minimum_fine_response",
    "correlated_passage_duplicate",
    "correlated_with_reference_event_id",
    "correlated_with_query_event_id",
    "reference_intervening_state_count",
    "query_intervening_state_count",
    "evaluation_duration_ms",
    "method",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_CANDIDATE_DECISION_REQUIRED_COLUMNS = (
    "stamp_ns",
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "event_separation_s",
    "minimum_state_id_separation",
    "descriptor_score",
    "descriptor_margin",
    "quality_accepted",
    "fail_open",
    "reason",
    "selected",
    "selected_anchor_role",
    "selected_minimum_fine_response",
    "maximum_pair_alignment_translation_error_m",
    "maximum_pair_alignment_yaw_error_rad",
    "reference_pre_state_id",
    "reference_pre_unique_id",
    "query_pre_state_id",
    "query_pre_unique_id",
    "reference_post_state_id",
    "reference_post_unique_id",
    "query_post_state_id",
    "query_post_unique_id",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS_COLUMNS = (
    "stamp_ns",
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "hypothesis_index",
    "descriptor_direction",
    "evaluated_direction",
    "preferred",
    "quality_accepted",
    "fail_open",
    "reason",
    "quality_available",
    "quality_risk",
    "minimum_fine_response",
    "maximum_pair_alignment_translation_error_m",
    "maximum_pair_alignment_yaw_error_rad",
    "selected",
    "direction_overridden",
    "selection_fail_open",
    "selection_reason",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_CONSTRAINT_REQUIRED_COLUMNS = (
    "stamp_ns",
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "mode",
    "quality_accepted",
    "fail_open",
    "reason",
    "selected_anchor_role",
    "source_state_id",
    "source_unique_id",
    "target_state_id",
    "target_unique_id",
    "edge_type",
    "edge_index",
    "edge_count_before",
    "edge_count_after",
    "measurement_x_m",
    "measurement_y_m",
    "measurement_yaw_rad",
    "covariance_xx",
    "covariance_yy",
    "covariance_yawyaw",
    "final_covariance_xx",
    "final_covariance_xy",
    "final_covariance_yy",
    "final_covariance_yawyaw",
    "pre_residual_x_m",
    "pre_residual_y_m",
    "pre_residual_translation_m",
    "pre_residual_yaw_rad",
    "pre_covariance_weighted_residual_sq",
    "post_residual_x_m",
    "post_residual_y_m",
    "post_residual_translation_m",
    "post_residual_yaw_rad",
    "post_covariance_weighted_residual_sq",
    "injected",
    "duplicate",
    "correlated_passage_duplicate",
    "correlated_with_reference_event_id",
    "correlated_with_query_event_id",
    "reference_intervening_state_count",
    "query_intervening_state_count",
    "action",
    "graph_action",
)
_ROUGHNESS_PLACE_CONSTRAINT_EDGE_REQUIRED_COLUMNS = (
    "stamp_ns",
    "mapper_generation",
    "event_id",
    "candidate_event_id",
    "candidate_rank",
    "mode",
    "anchor_strategy",
    "group_size",
    "edge_group_index",
    "anchor_role",
    "source_state_id",
    "source_unique_id",
    "target_state_id",
    "target_unique_id",
    "edge_index",
    "measurement_x_m",
    "measurement_y_m",
    "measurement_yaw_rad",
    "covariance_xx",
    "covariance_xy",
    "covariance_xyaw",
    "covariance_yy",
    "covariance_yyaw",
    "covariance_yawyaw",
    "final_covariance_xx",
    "final_covariance_xy",
    "final_covariance_xyaw",
    "final_covariance_yy",
    "final_covariance_yyaw",
    "final_covariance_yawyaw",
    "pre_residual_x_m",
    "pre_residual_y_m",
    "pre_residual_translation_m",
    "pre_residual_yaw_rad",
    "pre_covariance_weighted_residual_sq",
    "post_residual_x_m",
    "post_residual_y_m",
    "post_residual_translation_m",
    "post_residual_yaw_rad",
    "post_covariance_weighted_residual_sq",
    "injected",
    "group_partial",
    "reason",
    "action",
    "graph_action",
)
_ADAPTIVE_NEAR_CHAIN_REQUIRED_COLUMNS = (
    "stamp_ns",
    "candidate_index",
    "source_unique_id",
    "target_unique_id",
    "normalized_response",
    "relative_x_m",
    "relative_y_m",
    "relative_yaw_rad",
    "sequential_response",
    "candidate_vs_sequential_translation_m",
    "candidate_vs_sequential_yaw_rad",
    "candidate_vs_odom_prediction_translation_m",
    "candidate_vs_odom_prediction_yaw_rad",
    "rotation_candidate_risky",
    "rotation_proposed_reject",
    "rotation_intervention_reject",
    "rotation_prior_chain_overlap",
    "rotation_yaw_limit_rad",
    "rotation_translation_limit_m",
    "fail_open",
    "outcome_parity",
    "backend_action",
)
_ADAPTIVE_MEMORY_REQUIRED_COLUMNS = (
    "stamp_ns",
    "scan_processed",
    "observer_available",
    "observer_failure",
    "candidate_endpoints_unique",
    "new_edge_accounting_complete",
    "unexpected_new_edges",
    "intervention_allowed",
    "baseline_threshold",
    "applied_threshold",
    "strength",
    "effective_duration_s",
    "effective_distance_m",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_workspace_root(path: Path) -> Path:
    """Resolve a nested source cwd to the colcon workspace that owns its build."""
    requested = path.expanduser().resolve()
    candidates = [requested, *requested.parents]
    for candidate in candidates:
        if (
            (candidate / "install/slam_2d_core").exists()
            and (candidate / "build/slam_2d_core").exists()
        ):
            return candidate
    return requested


def runtime_build_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Fingerprint the executable and CMake mode used by this replay."""
    raw_workspace = str(getattr(args, "workspace", "") or "")
    if not raw_workspace:
        return {"status": "workspace_not_provided"}

    requested_workspace = Path(raw_workspace).expanduser().resolve()
    workspace = resolve_workspace_root(requested_workspace)
    executable = (
        workspace
        / "install/slam_2d_core/lib/slam_2d_core"
        / "roughness_aware_slam_toolbox_node"
    )
    cache = workspace / "build/slam_2d_core/CMakeCache.txt"
    cache_values: dict[str, str] = {}
    try:
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.startswith("//") or line.startswith("#") or "=" not in line:
                continue
            name_and_type, value = line.split("=", 1)
            name = name_and_type.split(":", 1)[0]
            if name in {
                "CMAKE_BUILD_TYPE",
                "CMAKE_CXX_FLAGS",
                "CMAKE_CXX_FLAGS_RELEASE",
                "CMAKE_CXX_COMPILER",
                "CMAKE_CXX_COMPILER_VERSION",
            }:
                cache_values[name] = value
    except OSError:
        pass

    executable_resolved = executable.resolve() if executable.exists() else executable
    return {
        "status": (
            "complete" if executable.is_file() and cache.is_file() else "incomplete"
        ),
        "requested_workspace": str(requested_workspace),
        "workspace": str(workspace),
        "executable": str(executable),
        "executable_resolved": str(executable_resolved),
        "executable_sha256": _file_sha256(executable),
        "cmake_cache": str(cache),
        "cmake_cache_sha256": _file_sha256(cache),
        "cmake_build_type": cache_values.get("CMAKE_BUILD_TYPE", ""),
        "cmake_cxx_flags": cache_values.get("CMAKE_CXX_FLAGS", ""),
        "cmake_cxx_flags_release": cache_values.get(
            "CMAKE_CXX_FLAGS_RELEASE", ""
        ),
        "cmake_cxx_compiler": cache_values.get("CMAKE_CXX_COMPILER", ""),
        "cmake_cxx_compiler_version": cache_values.get(
            "CMAKE_CXX_COMPILER_VERSION", ""
        ),
    }


def normalize_pose_graph_near_chain_diagnostics(text: str) -> str:
    """Hide new optional fields from the frozen legacy stage34 parser."""
    normalized = _POSE_GRAPH_DIAGNOSTIC_FIELDS_PATTERN.sub("", text)
    return _WORKER_DIAGNOSTIC_FIELDS_PATTERN.sub("", normalized)


def pose_graph_anchor_log_fields(text: str) -> dict[str, Any]:
    diagnostic_matches = list(
        _POSE_GRAPH_DIAGNOSTIC_FIELDS_PATTERN.finditer(text)
    )
    source_counts: dict[str, int] = {}
    for match in diagnostic_matches:
        source = match.group("source")
        source_counts[source] = source_counts.get(source, 0) + 1

    fields: dict[str, Any] = {
        "roughness_spatial_memory_pose_graph_anchor_mode_true_count": sum(
            int(match.group("mode") == "true")
            for match in diagnostic_matches
        ),
        "roughness_spatial_memory_pose_graph_anchor_mode_false_count": sum(
            int(match.group("mode") == "false")
            for match in diagnostic_matches
        ),
        "roughness_spatial_memory_pose_source_counts_json": json.dumps(
            source_counts,
            sort_keys=True,
        ),
    }
    for source, count in source_counts.items():
        safe_source = re.sub(r"[^a-zA-Z0-9_]+", "_", source).strip("_")
        if safe_source:
            fields[
                f"roughness_spatial_memory_pose_source_{safe_source}_count"
            ] = count

    summary_lines = [
        line
        for line in text.splitlines()
        if "ROUGHNESS_INNOVATION_SUMMARY" in line
    ]
    for counter in _POSE_GRAPH_SUMMARY_COUNTERS:
        field_counter = counter.removeprefix("spatial_")
        values = [
            int(match.group(1))
            for line in summary_lines
            if (match := re.search(rf"\b{re.escape(counter)}=(\d+)\b", line))
        ]
        fields[f"roughness_spatial_memory_{field_counter}_summary_last"] = (
            values[-1] if values else 0
        )
        fields[f"roughness_spatial_memory_{field_counter}_summary_max"] = max(
            values,
            default=0,
        )
    return fields


def scan_conditioning_log_fields(text: str) -> dict[str, Any]:
    """Summarize per-scan conditioning decisions and final counters."""
    events: list[dict[str, str]] = []
    for line in text.splitlines():
        if "ROUGHNESS_SCAN_CONDITION " not in line:
            continue
        events.append(
            dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
        )

    action_counts: dict[str, int] = {}
    mode_counts: dict[str, int] = {}
    removed_beams: list[int] = []
    stamps: list[int] = []
    for event in events:
        action = event.get("action", "unknown")
        mode = event.get("mode", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        try:
            removed_beams.append(int(event.get("removed_beams", "0")))
        except ValueError:
            removed_beams.append(0)
        try:
            stamps.append(int(event.get("stamp_ns", "0")))
        except ValueError:
            pass

    fields: dict[str, Any] = {
        "roughness_scan_conditioning_event_count": len(events),
        "roughness_scan_conditioning_action_counts_json": json.dumps(
            action_counts,
            sort_keys=True,
        ),
        "roughness_scan_conditioning_mode_counts_json": json.dumps(
            mode_counts,
            sort_keys=True,
        ),
        "roughness_scan_conditioning_rough_active_count": sum(
            event.get("rough_active") == "true" for event in events
        ),
        "roughness_scan_conditioning_hard_active_count": sum(
            event.get("hard_active") == "true" for event in events
        ),
        "roughness_scan_conditioning_imu_unavailable_count": sum(
            event.get("imu_available") == "false" for event in events
        ),
        "roughness_scan_conditioning_recovery_active_count": sum(
            event.get("recovery_active") == "true" for event in events
        ),
        "roughness_scan_conditioning_recovery_fault_count": sum(
            event.get("recovery_fault") == "true" for event in events
        ),
        "roughness_scan_conditioning_removed_beams_total": sum(
            removed_beams
        ),
        "roughness_scan_conditioning_removed_beams_max": max(
            removed_beams,
            default=0,
        ),
        "roughness_scan_conditioning_first_stamp_ns": min(stamps, default=0),
        "roughness_scan_conditioning_last_stamp_ns": max(stamps, default=0),
    }
    for action, count in action_counts.items():
        safe_action = re.sub(r"[^a-zA-Z0-9_]+", "_", action).strip("_")
        if safe_action:
            fields[
                f"roughness_scan_conditioning_action_{safe_action}_count"
            ] = count

    summary_lines = [
        line
        for line in text.splitlines()
        if "ROUGHNESS_INNOVATION_SUMMARY" in line
    ]
    for counter in _SCAN_CONDITIONING_SUMMARY_COUNTERS:
        values = [
            int(match.group(1))
            for line in summary_lines
            if (match := re.search(rf"\b{re.escape(counter)}=(\d+)\b", line))
        ]
        fields[f"roughness_{counter}_summary_last"] = (
            values[-1] if values else 0
        )
        fields[f"roughness_{counter}_summary_max"] = max(values, default=0)
    return fields


def roughness_place_log_fields(
    text: str,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract observe-only terrain-event place candidates from a replay log."""
    events: list[dict[str, str]] = []
    for line in text.splitlines():
        if "ROUGHNESS_PLACE_EVENT " not in line:
            continue
        event = dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
        if event:
            events.append(event)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=_ROUGHNESS_PLACE_EVENT_COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {column: event.get(column, "") for column in writer.fieldnames}
                )

    def finite_values(name: str) -> list[float]:
        values: list[float] = []
        for event in events:
            try:
                value = float(event.get(name, "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    candidate_scores = finite_values("candidate_score")
    candidate_margins = finite_values("candidate_margin")
    odom_distances = finite_values("odom_center_distance_m")
    corrected_distances = finite_values("corrected_center_distance_m")
    descriptor_lengths = finite_values("path_length_m")
    direction_counts: dict[str, int] = {}
    for event in events:
        direction = event.get("candidate_direction", "none")
        direction_counts[direction] = direction_counts.get(direction, 0) + 1

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    return {
        "roughness_place_event_count": len(events),
        "roughness_place_valid_descriptor_count": sum(
            event.get("descriptor_valid") == "true" for event in events
        ),
        "roughness_place_candidate_count": sum(
            event.get("candidate_available") == "true" for event in events
        ),
        "roughness_place_graph_action_count": sum(
            event.get("graph_action") != "none" for event in events
        ),
        "roughness_place_direction_counts_json": json.dumps(
            direction_counts,
            sort_keys=True,
        ),
        "roughness_place_candidate_score_mean": mean(candidate_scores),
        "roughness_place_candidate_score_max": max(
            candidate_scores,
            default=math.nan,
        ),
        "roughness_place_candidate_margin_mean": mean(candidate_margins),
        "roughness_place_odom_center_distance_mean_m": mean(odom_distances),
        "roughness_place_corrected_center_distance_mean_m": mean(
            corrected_distances
        ),
        "roughness_place_descriptor_path_length_mean_m": mean(
            descriptor_lengths
        ),
        "roughness_place_events_csv": str(output_path) if output_path else "",
    }


def roughness_place_scan_match_log_fields(
    text: str,
    anchor_output_path: Path | None = None,
    pair_output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract odom-seeded, bidirectional roughness-place scan-match evidence."""

    def marker_events(marker: str) -> list[dict[str, str]]:
        return [
            dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
            for line in text.splitlines()
            if marker in line
        ]

    anchor_events = marker_events("ROUGHNESS_PLACE_ANCHOR_SCAN_MATCH ")
    pair_events = marker_events("ROUGHNESS_PLACE_PAIR_CONSISTENCY ")

    def write_events(
        path: Path | None,
        columns: tuple[str, ...],
        events: list[dict[str, str]],
    ) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=columns,
                extrasaction="ignore",
            )
            writer.writeheader()
            for event in events:
                writer.writerow(
                    {column: event.get(column, "") for column in columns}
                )

    write_events(
        anchor_output_path,
        _ROUGHNESS_PLACE_ANCHOR_MATCH_COLUMNS,
        anchor_events,
    )
    write_events(
        pair_output_path,
        _ROUGHNESS_PLACE_PAIR_MATCH_COLUMNS,
        pair_events,
    )

    def finite_values(events: list[dict[str, str]], name: str) -> list[float]:
        values: list[float] = []
        for event in events:
            try:
                value = float(event.get(name, "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    def percentile(values: list[float], probability: float) -> float:
        if not values:
            return math.nan
        ordered = sorted(values)
        position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    failure_counts: dict[str, int] = {}
    for event in anchor_events:
        failure = event.get("failure", "missing")
        failure_counts[failure] = failure_counts.get(failure, 0) + 1

    fields: dict[str, Any] = {
        "roughness_place_scan_match_anchor_count": len(anchor_events),
        "roughness_place_scan_match_anchor_available_count": sum(
            event.get("available") == "true" for event in anchor_events
        ),
        "roughness_place_scan_match_anchor_fail_open_count": sum(
            event.get("fail_open") == "true" for event in anchor_events
        ),
        "roughness_place_scan_match_pair_count": len(pair_events),
        "roughness_place_scan_match_pair_available_count": sum(
            event.get("available") == "true" for event in pair_events
        ),
        "roughness_place_scan_match_pair_fail_open_count": sum(
            event.get("fail_open") == "true" for event in pair_events
        ),
        "roughness_place_scan_match_graph_action_count": sum(
            event.get("graph_action") != "none" for event in anchor_events
        ) + sum(event.get("graph_action") != "none" for event in pair_events),
        "roughness_place_scan_match_failure_counts_json": json.dumps(
            failure_counts,
            sort_keys=True,
        ),
        "roughness_place_anchor_scan_matches_csv": (
            str(anchor_output_path) if anchor_output_path else ""
        ),
        "roughness_place_pair_consistency_csv": (
            str(pair_output_path) if pair_output_path else ""
        ),
    }
    for name in (
        "forward_coarse_response",
        "forward_fine_response",
        "reverse_coarse_response",
        "reverse_fine_response",
        "measurement_vs_odometry_translation_m",
        "measurement_vs_odometry_yaw_rad",
        "bidirectional_translation_error_m",
        "bidirectional_yaw_error_rad",
        "directional_alignment_translation_error_m",
        "directional_alignment_yaw_error_rad",
    ):
        values = finite_values(anchor_events, name)
        fields[f"roughness_place_scan_match_{name}_mean"] = mean(values)
        fields[f"roughness_place_scan_match_{name}_p95"] = percentile(values, 0.95)
        fields[f"roughness_place_scan_match_{name}_max"] = max(
            values,
            default=math.nan,
        )
    for name in (
        "forward_alignment_translation_error_m",
        "forward_alignment_yaw_error_rad",
        "reverse_alignment_translation_error_m",
        "reverse_alignment_yaw_error_rad",
    ):
        values = finite_values(pair_events, name)
        fields[f"roughness_place_pair_{name}_p95"] = percentile(values, 0.95)
        fields[f"roughness_place_pair_{name}_max"] = max(
            values,
            default=math.nan,
        )
    return fields


def roughness_place_reference_chain_log_fields(
    text: str,
    event_output_path: Path | None = None,
    anchor_output_path: Path | None = None,
    hypothesis_output_path: Path | None = None,
    decision_output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract delayed stable-side scan-chain shadow evidence."""

    def events_for(marker: str) -> list[dict[str, str]]:
        return [
            dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
            for line in text.splitlines()
            if marker in line
        ]

    def write_events(
        path: Path | None,
        events: list[dict[str, str]],
        required_columns: tuple[str, ...],
    ) -> str:
        if path is None:
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        discovered = {key for event in events for key in event}
        columns = list(required_columns)
        columns.extend(sorted(discovered.difference(columns)))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for event in events:
                writer.writerow({column: event.get(column, "") for column in columns})
        return str(path)

    context_events = events_for("ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT ")
    anchor_events = events_for("ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR ")
    hypotheses = events_for("ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS ")
    decisions = events_for("ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION ")
    context_csv = write_events(
        event_output_path,
        context_events,
        _ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT_COLUMNS,
    )
    anchor_csv = write_events(
        anchor_output_path,
        anchor_events,
        _ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR_COLUMNS,
    )
    hypothesis_csv = write_events(
        hypothesis_output_path,
        hypotheses,
        _ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS_COLUMNS,
    )
    decision_csv = write_events(
        decision_output_path,
        decisions,
        _ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION_COLUMNS,
    )

    boolean_fields = {
        "available",
        "fail_open",
        "reference_context_available",
        "quality_accepted",
        "quality_available",
        "selected",
        "selection_fail_open",
        "correlated_passage_duplicate",
    }
    integer_fields = {
        "mapper_generation",
        "event_id",
        "candidate_event_id",
        "candidate_rank",
        "hypothesis_index",
        "pre_anchor_state_id",
        "post_anchor_state_id",
        "pre_chain_count",
        "pre_chain_first_state_id",
        "pre_chain_last_state_id",
        "post_reacquisition_reset_count",
        "post_reacquisition_state_gap",
        "pre_chain_anchor_state_gap",
        "post_chain_count",
        "post_chain_first_state_id",
        "post_chain_last_state_id",
        "candidate_count",
        "reference_base_scan_count",
        "query_base_scan_count",
        "reference_first_state_id",
        "reference_last_state_id",
        "query_first_state_id",
        "query_last_state_id",
        "correlated_with_reference_event_id",
        "correlated_with_query_event_id",
        "reference_intervening_state_count",
        "query_intervening_state_count",
    }
    float_fields = {
        "pre_chain_anchor_distance_m",
        "pre_chain_anchor_time_gap_s",
        "post_reacquisition_distance_m",
        "post_reacquisition_time_gap_s",
        "reference_maximum_anchor_distance_m",
        "query_maximum_anchor_distance_m",
        "forward_coarse_response",
        "forward_fine_response",
        "reverse_coarse_response",
        "reverse_fine_response",
        "forward_fine_cov_xx",
        "forward_fine_cov_yy",
        "forward_fine_cov_yawyaw",
        "reverse_fine_cov_xx",
        "reverse_fine_cov_yy",
        "reverse_fine_cov_yawyaw",
        "bidirectional_translation_error_m",
        "bidirectional_yaw_error_rad",
        "quality_risk",
        "minimum_fine_response",
        "forward_pair_translation_error_m",
        "forward_pair_yaw_error_rad",
        "reverse_pair_translation_error_m",
        "reverse_pair_yaw_error_rad",
        "evaluation_duration_ms",
    }

    def schema_error(event: dict[str, str], required: tuple[str, ...]) -> bool:
        if any(field not in event for field in required):
            return True
        for field in boolean_fields.intersection(required):
            if event.get(field) not in {"true", "false"}:
                return True
        for field in integer_fields.intersection(required):
            try:
                int(event.get(field, ""))
            except ValueError:
                return True
        for field in float_fields.intersection(required):
            try:
                float(event.get(field, ""))
            except ValueError:
                return True
        return False

    def finite_values(events: list[dict[str, str]], name: str) -> list[float]:
        values: list[float] = []
        for event in events:
            try:
                value = float(event.get(name, "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    def percentile(values: list[float], probability: float) -> float:
        if not values:
            return math.nan
        ordered = sorted(values)
        position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    context_failures: dict[str, int] = {}
    anchor_failures: dict[str, int] = {}
    decision_reasons: dict[str, int] = {}
    for event in context_events:
        failure = event.get("failure", "missing")
        context_failures[failure] = context_failures.get(failure, 0) + 1
    for event in anchor_events:
        failure = event.get("failure", "missing")
        anchor_failures[failure] = anchor_failures.get(failure, 0) + 1
    for event in decisions:
        reason = event.get("reason", "missing")
        decision_reasons[reason] = decision_reasons.get(reason, 0) + 1

    fields: dict[str, Any] = {
        "roughness_place_reference_chain_event_count": len(context_events),
        "roughness_place_reference_chain_event_available_count": sum(
            event.get("available") == "true" for event in context_events
        ),
        "roughness_place_reference_chain_event_fail_open_count": sum(
            event.get("fail_open") == "true" for event in context_events
        ),
        "roughness_place_reference_chain_anchor_count": len(anchor_events),
        "roughness_place_reference_chain_anchor_available_count": sum(
            event.get("available") == "true" for event in anchor_events
        ),
        "roughness_place_reference_chain_anchor_fail_open_count": sum(
            event.get("fail_open") == "true" for event in anchor_events
        ),
        "roughness_place_reference_chain_hypothesis_count": len(hypotheses),
        "roughness_place_reference_chain_hypothesis_available_count": sum(
            event.get("available") == "true" for event in hypotheses
        ),
        "roughness_place_reference_chain_hypothesis_fail_open_count": sum(
            event.get("fail_open") == "true" for event in hypotheses
        ),
        "roughness_place_reference_chain_hypothesis_accepted_count": sum(
            event.get("quality_accepted") == "true" for event in hypotheses
        ),
        "roughness_place_reference_chain_hypothesis_selected_count": sum(
            event.get("selected") == "true" for event in hypotheses
        ),
        "roughness_place_reference_chain_selection_fail_open_count": sum(
            event.get("selection_fail_open") == "true" for event in hypotheses
        ),
        "roughness_place_reference_chain_decision_count": len(decisions),
        "roughness_place_reference_chain_decision_available_count": sum(
            event.get("available") == "true" for event in decisions
        ),
        "roughness_place_reference_chain_decision_fail_open_count": sum(
            event.get("fail_open") == "true" for event in decisions
        ),
        "roughness_place_reference_chain_decision_accepted_count": sum(
            event.get("quality_accepted") == "true" for event in decisions
        ),
        "roughness_place_reference_chain_decision_correlated_duplicate_count": sum(
            event.get("correlated_passage_duplicate") == "true"
            for event in decisions
        ),
        "roughness_place_reference_chain_missing_reference_context_count": sum(
            event.get("reference_context_available") == "false"
            for event in hypotheses
        ),
        "roughness_place_reference_chain_context_failure_counts_json": json.dumps(
            context_failures,
            sort_keys=True,
        ),
        "roughness_place_reference_chain_anchor_failure_counts_json": json.dumps(
            anchor_failures,
            sort_keys=True,
        ),
        "roughness_place_reference_chain_decision_reason_counts_json": json.dumps(
            decision_reasons,
            sort_keys=True,
        ),
        "roughness_place_reference_chain_event_schema_error_count": sum(
            schema_error(event, _ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT_COLUMNS)
            for event in context_events
        ),
        "roughness_place_reference_chain_anchor_schema_error_count": sum(
            schema_error(event, _ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR_COLUMNS)
            for event in anchor_events
        ),
        "roughness_place_reference_chain_hypothesis_schema_error_count": sum(
            schema_error(
                event,
                _ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS_COLUMNS,
            )
            for event in hypotheses
        ),
        "roughness_place_reference_chain_decision_schema_error_count": sum(
            schema_error(
                event,
                _ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION_COLUMNS,
            )
            for event in decisions
        ),
        "roughness_place_reference_chain_events_csv": context_csv,
        "roughness_place_reference_chain_anchors_csv": anchor_csv,
        "roughness_place_reference_chain_hypotheses_csv": hypothesis_csv,
        "roughness_place_reference_chain_decisions_csv": decision_csv,
    }
    for name in (
        "pre_chain_anchor_state_gap",
        "pre_chain_anchor_distance_m",
        "pre_chain_anchor_time_gap_s",
        "post_reacquisition_reset_count",
        "post_reacquisition_state_gap",
        "post_reacquisition_distance_m",
        "post_reacquisition_time_gap_s",
    ):
        values = finite_values(context_events, name)
        fields[
            f"roughness_place_reference_chain_event_{name}_median"
        ] = percentile(values, 0.50)
        fields[
            f"roughness_place_reference_chain_event_{name}_p95"
        ] = percentile(values, 0.95)
        fields[
            f"roughness_place_reference_chain_event_{name}_max"
        ] = max(values) if values else math.nan
    for name in (
        "reference_base_scan_count",
        "query_base_scan_count",
        "reference_maximum_anchor_distance_m",
        "query_maximum_anchor_distance_m",
        "forward_coarse_response",
        "forward_fine_response",
        "reverse_coarse_response",
        "reverse_fine_response",
        "bidirectional_translation_error_m",
        "bidirectional_yaw_error_rad",
    ):
        values = finite_values(anchor_events, name)
        fields[f"roughness_place_reference_chain_anchor_{name}_median"] = percentile(
            values,
            0.5,
        )
        fields[f"roughness_place_reference_chain_anchor_{name}_p95"] = percentile(
            values,
            0.95,
        )
        fields[f"roughness_place_reference_chain_anchor_{name}_max"] = max(
            values,
            default=math.nan,
        )
    for name in (
        "quality_risk",
        "minimum_fine_response",
        "forward_pair_translation_error_m",
        "forward_pair_yaw_error_rad",
        "reverse_pair_translation_error_m",
        "reverse_pair_yaw_error_rad",
    ):
        values = finite_values(hypotheses, name)
        fields[f"roughness_place_reference_chain_{name}_median"] = percentile(
            values,
            0.5,
        )
        fields[f"roughness_place_reference_chain_{name}_p95"] = percentile(
            values,
            0.95,
        )
        fields[f"roughness_place_reference_chain_{name}_max"] = max(
            values,
            default=math.nan,
        )
    decision_durations_ms = finite_values(decisions, "evaluation_duration_ms")
    fields["roughness_place_reference_chain_evaluation_duration_ms_median"] = (
        percentile(decision_durations_ms, 0.5)
    )
    fields["roughness_place_reference_chain_evaluation_duration_ms_p95"] = (
        percentile(decision_durations_ms, 0.95)
    )
    fields["roughness_place_reference_chain_evaluation_duration_ms_max"] = max(
        decision_durations_ms,
        default=math.nan,
    )
    return fields


def roughness_place_constraint_log_fields(
    text: str,
    decision_output_path: Path | None = None,
    constraint_output_path: Path | None = None,
    constraint_edge_output_path: Path | None = None,
    direction_hypothesis_output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract quality decisions and their optional pose-graph intervention."""

    def events_for(marker: str) -> list[dict[str, str]]:
        return [
            dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
            for line in text.splitlines()
            if marker in line
        ]

    def write_events(
        path: Path | None,
        events: list[dict[str, str]],
        required_columns: tuple[str, ...],
    ) -> str:
        if path is None:
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        discovered = {key for event in events for key in event}
        columns = list(required_columns)
        columns.extend(sorted(discovered.difference(columns)))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for event in events:
                writer.writerow({column: event.get(column, "") for column in columns})
        return str(path)

    decisions = events_for("ROUGHNESS_PLACE_CANDIDATE_DECISION ")
    direction_hypotheses = events_for("ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS ")
    constraints = events_for("ROUGHNESS_PLACE_CONSTRAINT ")
    constraint_edges = events_for("ROUGHNESS_PLACE_CONSTRAINT_EDGE ")
    decisions_csv = write_events(
        decision_output_path,
        decisions,
        _ROUGHNESS_PLACE_CANDIDATE_DECISION_REQUIRED_COLUMNS,
    )
    direction_hypotheses_csv = write_events(
        direction_hypothesis_output_path,
        direction_hypotheses,
        _ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS_COLUMNS,
    )
    constraints_csv = write_events(
        constraint_output_path,
        constraints,
        _ROUGHNESS_PLACE_CONSTRAINT_REQUIRED_COLUMNS,
    )
    constraint_edges_csv = write_events(
        constraint_edge_output_path,
        constraint_edges,
        _ROUGHNESS_PLACE_CONSTRAINT_EDGE_REQUIRED_COLUMNS,
    )

    boolean_fields = {
        "quality_accepted",
        "fail_open",
        "selected",
        "injected",
        "duplicate",
        "group_partial",
        "correlated_passage_duplicate",
        "preferred",
        "quality_available",
        "direction_overridden",
        "selection_fail_open",
    }
    identity_fields = (
        "mapper_generation",
        "event_id",
        "candidate_event_id",
        "candidate_rank",
    )

    def schema_error(event: dict[str, str], required: tuple[str, ...]) -> bool:
        if any(field not in event for field in required):
            return True
        for field in boolean_fields.intersection(required):
            if event.get(field) not in {"true", "false"}:
                return True
        for field in identity_fields:
            try:
                int(event.get(field, ""))
            except ValueError:
                return True
        return False

    decision_schema_errors = sum(
        schema_error(event, _ROUGHNESS_PLACE_CANDIDATE_DECISION_REQUIRED_COLUMNS)
        for event in decisions
    )
    direction_hypothesis_schema_errors = sum(
        schema_error(event, _ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS_COLUMNS)
        for event in direction_hypotheses
    )
    constraint_schema_errors = sum(
        schema_error(event, _ROUGHNESS_PLACE_CONSTRAINT_REQUIRED_COLUMNS)
        for event in constraints
    )
    constraint_edge_schema_errors = sum(
        schema_error(event, _ROUGHNESS_PLACE_CONSTRAINT_EDGE_REQUIRED_COLUMNS)
        for event in constraint_edges
    )

    def identity(event: dict[str, str]) -> tuple[str, ...]:
        return tuple(event.get(field, "") for field in identity_fields)

    decision_identities = [identity(event) for event in decisions]
    constraint_identities = [
        identity(event)
        for event in constraints
        if event.get("candidate_rank", "0") != "0"
    ]
    duplicate_identity_count = (
        len(decision_identities) - len(set(decision_identities))
        + len(constraint_identities) - len(set(constraint_identities))
    )
    decision_identity_set = set(decision_identities)
    unmatched_constraint_count = sum(
        item not in decision_identity_set for item in constraint_identities
    )

    def counts(events: list[dict[str, str]], field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for event in events:
            value = event.get(field, "missing")
            result[value] = result.get(value, 0) + 1
        return result

    def finite_values(events: list[dict[str, str]], field: str) -> list[float]:
        values: list[float] = []
        for event in events:
            try:
                value = float(event.get(field, "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    def percentile(values: list[float], probability: float) -> float:
        if not values:
            return math.nan
        ordered = sorted(values)
        position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    fields: dict[str, Any] = {
        "roughness_place_candidate_decision_count": len(decisions),
        "roughness_place_candidate_accepted_count": sum(
            event.get("quality_accepted") == "true" for event in decisions
        ),
        "roughness_place_candidate_rejected_count": sum(
            event.get("quality_accepted") == "false"
            and event.get("fail_open") == "false"
            for event in decisions
        ),
        "roughness_place_candidate_fail_open_count": sum(
            event.get("fail_open") == "true" for event in decisions
        ),
        "roughness_place_candidate_selected_count": sum(
            event.get("selected") == "true" for event in decisions
        ),
        "roughness_place_candidate_reason_counts_json": json.dumps(
            counts(decisions, "reason"), sort_keys=True
        ),
        "roughness_place_direction_hypothesis_count": len(direction_hypotheses),
        "roughness_place_direction_hypothesis_accepted_count": sum(
            event.get("quality_accepted") == "true"
            for event in direction_hypotheses
        ),
        "roughness_place_direction_hypothesis_fail_open_count": sum(
            event.get("fail_open") == "true" for event in direction_hypotheses
        ),
        "roughness_place_direction_selected_count": sum(
            event.get("selected") == "true" for event in direction_hypotheses
        ),
        "roughness_place_direction_override_count": sum(
            event.get("direction_overridden") == "true"
            for event in direction_hypotheses
        ),
        "roughness_place_direction_selection_fail_open_count": sum(
            event.get("selection_fail_open") == "true"
            for event in direction_hypotheses
            if event.get("preferred") == "true"
        ),
        "roughness_place_direction_selection_reason_counts_json": json.dumps(
            counts(
                [
                    event
                    for event in direction_hypotheses
                    if event.get("preferred") == "true"
                ],
                "selection_reason",
            ),
            sort_keys=True,
        ),
        "roughness_place_constraint_count": len(constraints),
        "roughness_place_constraint_edge_count": len(constraint_edges),
        "roughness_place_constraint_edge_injected_count": sum(
            event.get("injected") == "true" for event in constraint_edges
        ),
        "roughness_place_constraint_partial_group_count": sum(
            event.get("group_partial") == "true" for event in constraints
        ),
        "roughness_place_constraint_injected_count": sum(
            event.get("injected") == "true" for event in constraints
        ),
        "roughness_place_constraint_duplicate_count": sum(
            event.get("duplicate") == "true" for event in constraints
        ),
        "roughness_place_constraint_correlated_duplicate_count": sum(
            event.get("correlated_passage_duplicate") == "true"
            for event in constraints
        ),
        "roughness_place_constraint_fail_open_count": sum(
            event.get("fail_open") == "true" for event in constraints
        ),
        "roughness_place_constraint_reason_counts_json": json.dumps(
            counts(constraints, "reason"), sort_keys=True
        ),
        "roughness_place_constraint_action_counts_json": json.dumps(
            counts(constraints, "graph_action"), sort_keys=True
        ),
        "roughness_place_decision_constraint_unmatched_count": (
            unmatched_constraint_count
        ),
        "roughness_place_decision_duplicate_identity_count": (
            duplicate_identity_count
        ),
        "roughness_place_decision_schema_error_count": decision_schema_errors,
        "roughness_place_direction_hypothesis_schema_error_count": (
            direction_hypothesis_schema_errors
        ),
        "roughness_place_constraint_schema_error_count": constraint_schema_errors,
        "roughness_place_constraint_edge_schema_error_count": (
            constraint_edge_schema_errors
        ),
        "roughness_place_candidate_decisions_csv": decisions_csv,
        "roughness_place_direction_hypotheses_csv": direction_hypotheses_csv,
        "roughness_place_constraints_csv": constraints_csv,
        "roughness_place_constraint_edges_csv": constraint_edges_csv,
    }
    for name in (
        "selected_minimum_fine_response",
        "maximum_pair_alignment_translation_error_m",
        "maximum_pair_alignment_yaw_error_rad",
    ):
        values = finite_values(decisions, name)
        fields[f"roughness_place_candidate_{name}_median"] = percentile(values, 0.5)
        fields[f"roughness_place_candidate_{name}_p95"] = percentile(values, 0.95)
        fields[f"roughness_place_candidate_{name}_max"] = max(
            values, default=math.nan
        )
    for name in (
        "quality_risk",
        "minimum_fine_response",
        "maximum_pair_alignment_translation_error_m",
        "maximum_pair_alignment_yaw_error_rad",
    ):
        values = finite_values(direction_hypotheses, name)
        fields[f"roughness_place_direction_{name}_median"] = percentile(values, 0.5)
        fields[f"roughness_place_direction_{name}_p95"] = percentile(values, 0.95)
        fields[f"roughness_place_direction_{name}_max"] = max(
            values, default=math.nan
        )
    for name in (
        "pre_residual_translation_m",
        "pre_residual_yaw_rad",
        "pre_covariance_weighted_residual_sq",
        "post_residual_translation_m",
        "post_residual_yaw_rad",
        "post_covariance_weighted_residual_sq",
    ):
        values = finite_values(constraints, name)
        fields[f"roughness_place_constraint_{name}_median"] = percentile(values, 0.5)
        fields[f"roughness_place_constraint_{name}_p95"] = percentile(values, 0.95)
        fields[f"roughness_place_constraint_{name}_max"] = max(
            values, default=math.nan
        )
    for name in (
        "pre_residual_translation_m",
        "pre_residual_yaw_rad",
        "pre_covariance_weighted_residual_sq",
        "post_residual_translation_m",
        "post_residual_yaw_rad",
        "post_covariance_weighted_residual_sq",
    ):
        values = finite_values(constraint_edges, name)
        fields[f"roughness_place_constraint_edge_{name}_median"] = percentile(
            values, 0.5
        )
        fields[f"roughness_place_constraint_edge_{name}_p95"] = percentile(
            values, 0.95
        )
        fields[f"roughness_place_constraint_edge_{name}_max"] = max(
            values, default=math.nan
        )
    for name in (
        "reference_intervening_state_count",
        "query_intervening_state_count",
    ):
        values = finite_values(constraints, name)
        fields[f"roughness_place_constraint_{name}_median"] = percentile(
            values, 0.5
        )
        fields[f"roughness_place_constraint_{name}_max"] = max(
            values, default=math.nan
        )
    return fields


def adaptive_near_chain_log_fields(
    text: str,
    candidate_output_path: Path | None = None,
    memory_output_path: Path | None = None,
) -> dict[str, Any]:
    """Extract exact near-chain candidates and adaptive-memory snapshots."""

    def events_for(marker: str) -> list[dict[str, str]]:
        return [
            dict(re.findall(r"\b([a-zA-Z0-9_]+)=([^\s]+)", line))
            for line in text.splitlines()
            if marker in line
        ]

    def write_events(
        path: Path | None,
        events: list[dict[str, str]],
        required_columns: tuple[str, ...],
    ) -> str:
        if path is None:
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        event_columns = {key for event in events for key in event}
        columns = list(required_columns)
        columns.extend(sorted(event_columns.difference(required_columns)))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(events)
        return str(path)

    def finite_values(
        events: list[dict[str, str]],
        name: str,
    ) -> list[float]:
        values: list[float] = []
        for event in events:
            try:
                value = float(event.get(name, "nan"))
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    def percentile(values: list[float], fraction: float) -> float:
        if not values:
            return math.nan
        ordered = sorted(values)
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    candidate_events = events_for("ROUGHNESS_NEAR_CHAIN_CANDIDATE ")
    memory_events = events_for("ROUGHNESS_ADAPTIVE_MEMORY ")
    action_counts: dict[str, int] = {}
    shadow_action_counts: dict[str, int] = {}
    for event in candidate_events:
        action = event.get("backend_action", "unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        shadow = event.get("shadow_action", "unknown")
        shadow_action_counts[shadow] = shadow_action_counts.get(shadow, 0) + 1

    fields: dict[str, Any] = {
        "roughness_adaptive_memory_event_count": len(memory_events),
        "roughness_exact_near_chain_candidate_count": len(candidate_events),
        "roughness_exact_near_chain_eligible_count": sum(
            event.get("chain_size_accepted") == "true"
            for event in candidate_events
        ),
        "roughness_exact_near_chain_edge_inserted_count": sum(
            event.get("edge_inserted") == "true" for event in candidate_events
        ),
        "roughness_exact_near_chain_downweighted_count": sum(
            event.get("downweighted") == "true" for event in candidate_events
        ),
        "roughness_exact_near_chain_fail_open_count": sum(
            event.get("fail_open") == "true" for event in candidate_events
        ),
        "roughness_rotation_risky_candidate_count": sum(
            event.get("rotation_candidate_risky") == "true"
            for event in candidate_events
        ),
        "roughness_rotation_proposed_reject_count": sum(
            event.get("rotation_proposed_reject") == "true"
            for event in candidate_events
        ),
        "roughness_rotation_intervention_reject_count": sum(
            event.get("rotation_intervention_reject") == "true"
            for event in candidate_events
        ),
        "roughness_exact_near_chain_parity_failure_count": sum(
            event.get("outcome_parity") == "false"
            for event in candidate_events
        ),
        "roughness_exact_near_chain_backend_action_counts_json": json.dumps(
            action_counts,
            sort_keys=True,
        ),
        "roughness_exact_near_chain_shadow_action_counts_json": json.dumps(
            shadow_action_counts,
            sort_keys=True,
        ),
        "roughness_exact_near_chain_candidate_events_csv": write_events(
            candidate_output_path,
            candidate_events,
            _ADAPTIVE_NEAR_CHAIN_REQUIRED_COLUMNS,
        ),
        "roughness_adaptive_memory_events_csv": write_events(
            memory_output_path,
            memory_events,
            _ADAPTIVE_MEMORY_REQUIRED_COLUMNS,
        ),
    }
    distributions = (
        ("memory_strength", "strength", memory_events),
        ("event_intensity", "event_intensity", memory_events),
        ("effective_duration_s", "effective_duration_s", memory_events),
        ("effective_distance_m", "effective_distance_m", memory_events),
        ("normalized_response", "normalized_response", candidate_events),
        (
            "translation_disagreement_m",
            "translation_disagreement_m",
            candidate_events,
        ),
        ("yaw_disagreement_rad", "yaw_disagreement_rad", candidate_events),
        (
            "xy_covariance_max_eigenvalue_m2",
            "xy_covariance_max_eigenvalue_m2",
            candidate_events,
        ),
        ("quality_risk", "quality_risk", candidate_events),
        ("combined_risk", "combined_risk", candidate_events),
        (
            "rotation_prior_chain_overlap",
            "rotation_prior_chain_overlap",
            candidate_events,
        ),
        (
            "rotation_log1p_weighted_disagreement",
            "rotation_log1p_weighted_disagreement",
            candidate_events,
        ),
        (
            "rotation_yaw_limit_rad",
            "rotation_yaw_limit_rad",
            candidate_events,
        ),
        (
            "rotation_translation_limit_m",
            "rotation_translation_limit_m",
            candidate_events,
        ),
        (
            "scan_match_covariance_weighted_disagreement_sq",
            "scan_match_covariance_weighted_disagreement_sq",
            candidate_events,
        ),
        (
            "post_covariance_weighted_residual_sq",
            "post_covariance_weighted_residual_sq",
            candidate_events,
        ),
    )
    for output_field, source_field, events in distributions:
        values = finite_values(events, source_field)
        prefix = f"roughness_adaptive_{output_field}"
        fields[f"{prefix}_count"] = len(values)
        fields[f"{prefix}_mean"] = mean(values)
        fields[f"{prefix}_median"] = percentile(values, 0.5)
        fields[f"{prefix}_p95"] = percentile(values, 0.95)
        fields[f"{prefix}_max"] = max(values, default=math.nan)
    return fields


def compatible_roughness_log_fields(
    original_log_fields: Any,
    path: Path,
    events_path: Path | None = None,
    post_events_path: Path | None = None,
    near_chain_events_path: Path | None = None,
) -> dict[str, Any]:
    text = _read_log(path)
    artifact_stem = re.sub(r"_launch$", "", path.stem)
    normalized = normalize_pose_graph_near_chain_diagnostics(text)
    parser_path = path
    temporary_path: Path | None = None
    try:
        if normalized != text:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".log",
                delete=False,
            ) as stream:
                stream.write(normalized)
                temporary_path = Path(stream.name)
            parser_path = temporary_path
        fields = original_log_fields(
            parser_path,
            events_path,
            post_events_path,
            near_chain_events_path,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    fields.update(pose_graph_anchor_log_fields(text))
    fields.update(scan_conditioning_log_fields(text))
    fields.update(
        roughness_place_log_fields(
            text,
            path.parent / f"{artifact_stem}_roughness_place_events.csv",
        )
    )
    fields.update(
        roughness_place_scan_match_log_fields(
            text,
            path.parent /
            f"{artifact_stem}_roughness_place_anchor_scan_matches.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_pair_consistency.csv",
        )
    )
    fields.update(
        roughness_place_reference_chain_log_fields(
            text,
            path.parent /
            f"{artifact_stem}_roughness_place_reference_chain_events.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_reference_chain_anchors.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_reference_chain_hypotheses.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_reference_chain_decisions.csv",
        )
    )
    fields.update(
        roughness_place_constraint_log_fields(
            text,
            path.parent /
            f"{artifact_stem}_roughness_place_candidate_decisions.csv",
            path.parent / f"{artifact_stem}_roughness_place_constraints.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_constraint_edges.csv",
            path.parent /
            f"{artifact_stem}_roughness_place_direction_hypotheses.csv",
        )
    )
    fields.update(
        adaptive_near_chain_log_fields(
            text,
            path.parent /
            f"{artifact_stem}_near_chain_candidate_events.csv",
            path.parent / f"{artifact_stem}_adaptive_memory_events.csv",
        )
    )
    return fields


def configured_pose_graph_anchor_parameter(
    pipeline: ModuleType,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_config_path = str(getattr(args, "slam_config", "") or "")
    config_path = Path(raw_config_path).expanduser() if raw_config_path else None
    result: dict[str, Any] = {
        "name": POSE_GRAPH_ANCHOR_PARAMETER,
        "configured": 0,
        "default": POSE_GRAPH_ANCHOR_DEFAULT,
        "effective": POSE_GRAPH_ANCHOR_DEFAULT,
        "config_path": str(config_path.resolve()) if config_path else "",
    }
    if config_path is None:
        result["config_status"] = "not_provided"
        return result
    try:
        configured = pipeline.load_slam_toolbox_params(config_path)
    except Exception as exc:
        result["config_status"] = f"unreadable:{type(exc).__name__}"
        return result
    if not isinstance(configured, dict):
        result["config_status"] = "invalid_mapping"
        return result
    result["config_status"] = "loaded"
    if POSE_GRAPH_ANCHOR_PARAMETER in configured:
        result["configured"] = 1
        result["expected"] = configured[POSE_GRAPH_ANCHOR_PARAMETER]
        result["effective"] = configured[POSE_GRAPH_ANCHOR_PARAMETER]
    return result


def configured_scan_conditioning_parameters(
    pipeline: ModuleType,
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_config_path = str(getattr(args, "slam_config", "") or "")
    config_path = Path(raw_config_path).expanduser() if raw_config_path else None
    result: dict[str, Any] = {
        "configured_count": 0,
        "config_path": str(config_path.resolve()) if config_path else "",
        "effective": dict(SCAN_CONDITIONING_DEFAULTS),
    }
    if config_path is None:
        result["config_status"] = "not_provided"
        result["expected"] = {}
        return result
    try:
        configured = pipeline.load_slam_toolbox_params(config_path)
    except Exception as exc:
        result["config_status"] = f"unreadable:{type(exc).__name__}"
        result["expected"] = {}
        return result
    if not isinstance(configured, dict):
        result["config_status"] = "invalid_mapping"
        result["expected"] = {}
        return result

    expected = {
        name: configured[name]
        for name in SCAN_CONDITIONING_PARAMETERS
        if name in configured
    }
    effective = dict(SCAN_CONDITIONING_DEFAULTS)
    effective.update(expected)
    result.update(
        {
            "configured_count": len(expected),
            "config_status": "loaded",
            "expected": expected,
            "effective": effective,
        }
    )
    return result


def configured_adaptive_constraint_parameters(
    pipeline: ModuleType,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Return every explicitly configured adaptive-constraint parameter."""
    raw_config_path = str(getattr(args, "slam_config", "") or "")
    config_path = Path(raw_config_path).expanduser() if raw_config_path else None
    result: dict[str, Any] = {
        "configured_count": 0,
        "config_path": str(config_path.resolve()) if config_path else "",
        "expected": {},
    }
    if config_path is None:
        result["config_status"] = "not_provided"
        return result
    try:
        configured = pipeline.load_slam_toolbox_params(config_path)
    except Exception as exc:
        result["config_status"] = f"unreadable:{type(exc).__name__}"
        return result
    if not isinstance(configured, dict):
        result["config_status"] = "invalid_mapping"
        return result

    expected = {
        name: configured[name]
        for name in ADAPTIVE_CONSTRAINT_PARAMETERS
        if name in configured
    }
    result.update(
        {
            "configured_count": len(expected),
            "config_status": "loaded",
            "expected": expected,
        }
    )
    return result


def scan_conditioning_can_change_pose_graph_node_count(
    effective: dict[str, Any],
) -> bool:
    if not bool(effective.get("enable_roughness_scan_conditioning", False)):
        return False
    mode = str(effective.get("roughness_scan_conditioning_mode", "observe"))
    if mode in {"drop", "throttle", "quarantine"}:
        return True
    if mode == "stabilize":
        return str(
            effective.get(
                "roughness_scan_conditioning_stabilize_fallback_mode",
                "drop",
            )
        ) == "drop"
    if mode == "residual_filter":
        return str(
            effective.get(
                "roughness_scan_conditioning_residual_fallback_mode",
                "observe",
            )
        ) == "drop"
    return False


def relax_intentional_conditioning_node_parity_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for row in rows:
        row["pose_graph_node_parity_required"] = 0
        row["pose_graph_node_parity_complete"] = 1
        row["slam_owned_scan_conditioning_node_parity_relaxed"] = 1
        row["slam_owned_scan_conditioning_node_retention_floor"] = 0.85
    return rows


def enrich_stage34_rows_with_roughness_place_metrics(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy exact per-bag place-recognition metrics into the stage34 table."""
    for row in rows:
        variant_dir_text = str(row.get("variant_dir", "")).strip()
        row["variant_roughness_place_metrics_source"] = ""
        row["variant_roughness_place_metrics_count"] = 0
        if not variant_dir_text:
            row["variant_roughness_place_metrics_status"] = "missing_variant_dir"
            continue

        replay_path = Path(variant_dir_text) / "replay_result.json"
        row["variant_roughness_place_metrics_source"] = str(replay_path)
        if not replay_path.is_file():
            row["variant_roughness_place_metrics_status"] = "missing_replay_result"
            continue

        try:
            replay_result = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            row["variant_roughness_place_metrics_status"] = (
                f"unreadable:{type(exc).__name__}"
            )
            continue
        if not isinstance(replay_result, dict):
            row["variant_roughness_place_metrics_status"] = "invalid_mapping"
            continue

        metric_count = 0
        for name, value in replay_result.items():
            if not name.startswith("roughness_place_"):
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True, separators=(",", ":"))
            elif value is None:
                value = ""
            row[f"variant_{name}"] = value
            metric_count += 1
        row["variant_roughness_place_metrics_count"] = metric_count
        row["variant_roughness_place_metrics_status"] = "loaded"
    return rows


def augment_replay_provenance_payload(
    pipeline: ModuleType,
    original_payload: Any,
    row: dict[str, Any],
    args: argparse.Namespace,
    launch_command: str,
    capture_command: str,
) -> dict[str, Any]:
    payload = copy.deepcopy(
        original_payload(row, args, launch_command, capture_command)
    )
    legacy_signature = _payload_sha256(payload)
    pose_graph_anchor_parameter = configured_pose_graph_anchor_parameter(
        pipeline,
        args,
    )
    scan_conditioning_parameters = configured_scan_conditioning_parameters(
        pipeline,
        args,
    )
    adaptive_constraint_parameters = configured_adaptive_constraint_parameters(
        pipeline,
        args,
    )
    runtime_build = runtime_build_provenance(args)
    fairness_contract = payload.get("fairness_contract")
    if isinstance(fairness_contract, dict):
        fairness_contract[RUNTIME_BUILD_FAIRNESS_FIELD] = runtime_build
        roughness_profile = fairness_contract.get(
            "roughness_innovation_profile"
        )
        if not isinstance(roughness_profile, dict):
            roughness_profile = {}
            fairness_contract["roughness_innovation_profile"] = (
                roughness_profile
            )
        roughness_profile[POSE_GRAPH_ANCHOR_FAIRNESS_FIELD] = (
            pose_graph_anchor_parameter["effective"]
        )
        roughness_profile[SCAN_CONDITIONING_FAIRNESS_FIELD] = (
            scan_conditioning_parameters["effective"]
        )
        roughness_profile[ADAPTIVE_CONSTRAINT_FAIRNESS_FIELD] = (
            adaptive_constraint_parameters["expected"]
        )
    compatible_legacy_payload = copy.deepcopy(payload)
    payload["lossless_stage34_adapter"] = {
        "schema": ADAPTER_PROVENANCE_SCHEMA,
        "path": str(ADAPTER_PATH),
        "sha256": _file_sha256(ADAPTER_PATH),
        "runtime_parameter_capture_schema": (
            ADAPTER_RUNTIME_PARAMETER_CAPTURE_SCHEMA
        ),
        "pose_graph_anchor_parameter": (
            pose_graph_anchor_parameter
        ),
        "scan_conditioning_parameters": scan_conditioning_parameters,
        "adaptive_constraint_parameters": adaptive_constraint_parameters,
        "runtime_build": runtime_build,
    }
    adapter_signature = _payload_sha256(payload)
    _LEGACY_SIGNATURE_BY_ADAPTER_SIGNATURE[adapter_signature] = legacy_signature
    _LEGACY_PAYLOAD_BY_ADAPTER_SIGNATURE[adapter_signature] = (
        compatible_legacy_payload
    )
    return payload


def augment_slam_parameter_capture(
    pipeline: ModuleType,
    original_capture: Any,
    args: argparse.Namespace,
    bag_out: Path,
    env: dict[str, str],
    launch_proc: Any,
) -> dict[str, Any]:
    capture_started = time.monotonic()
    fields = original_capture(args, bag_out, env, launch_proc)
    initial_return_code = fields.get(
        "runtime_slam_parameter_dump_return_code",
        "",
    )
    initial_capture_failed = (
        initial_return_code != 0
        or int(fields.get("runtime_slam_parameter_count", 0) or 0) <= 0
    )
    retry_count = 0
    retry_status = "not_needed"
    if (
        initial_capture_failed
        and launch_proc.poll() is None
    ):
        retry_count = 1
        retry_status = "attempted"
        fields = original_capture(
            args,
            bag_out,
            env,
            launch_proc,
            timeout_s=SLAM_PARAMETER_CAPTURE_RETRY_TIMEOUT_S,
        )
        retry_succeeded = (
            fields.get("runtime_slam_parameter_dump_return_code") == 0
            and int(fields.get("runtime_slam_parameter_count", 0) or 0) > 0
        )
        retry_status = "succeeded" if retry_succeeded else "failed"
    fields.update(
        {
            "runtime_slam_parameter_capture_attempt_count": 1 + retry_count,
            "runtime_slam_parameter_capture_retry_count": retry_count,
            "runtime_slam_parameter_capture_retry_status": retry_status,
            "runtime_slam_parameter_capture_initial_return_code": (
                initial_return_code
            ),
            "runtime_slam_parameter_capture_retry_timeout_s": (
                SLAM_PARAMETER_CAPTURE_RETRY_TIMEOUT_S
            ),
            "runtime_slam_parameter_capture_wall_s": (
                time.monotonic() - capture_started
            ),
        }
    )
    anchor_expectation = configured_pose_graph_anchor_parameter(pipeline, args)
    scan_expectation = configured_scan_conditioning_parameters(pipeline, args)
    adaptive_expectation = configured_adaptive_constraint_parameters(
        pipeline,
        args,
    )
    raw_dump_path = str(fields.get("runtime_slam_parameter_dump", "") or "")
    dump_path = Path(raw_dump_path) if raw_dump_path else None
    dump_text = _read_log(dump_path) if dump_path is not None else ""
    try:
        parameters = pipeline.stage30_extract_node_parameters(
            dump_text,
            "/slam_toolbox",
        )
    except (AttributeError, TypeError, ValueError):
        parameters = {}
    if not isinstance(parameters, dict):
        parameters = {}

    dump_captured = (
        dump_path is not None
        and dump_path.is_file()
        and int(fields.get("runtime_slam_parameter_dump_return_code", -1)) == 0
        and bool(parameters)
    )
    fields.update(
        {
            "runtime_slam_full_parameter_dump_schema": (
                ADAPTER_RUNTIME_PARAMETER_CAPTURE_SCHEMA
            ),
            "runtime_slam_full_parameter_dump_captured": int(dump_captured),
            "runtime_slam_full_parameter_dump_count": len(parameters),
            "runtime_slam_full_parameter_dump_sha256": (
                _file_sha256(dump_path)
                if dump_path is not None and dump_path.is_file()
                else ""
            ),
            "runtime_slam_pose_graph_anchor_parameter_name": (
                POSE_GRAPH_ANCHOR_PARAMETER
            ),
            "runtime_slam_pose_graph_anchor_parameter_configured": int(
                anchor_expectation.get("configured", 0)
            ),
            "runtime_slam_scan_conditioning_parameter_configured_count": int(
                scan_expectation.get("configured_count", 0)
            ),
            "runtime_slam_adaptive_constraint_parameter_configured_count": int(
                adaptive_expectation.get("configured_count", 0)
            ),
        }
    )
    actual = parameters.get(POSE_GRAPH_ANCHOR_PARAMETER, "<missing>")
    fields[f"runtime_slam_actual_{POSE_GRAPH_ANCHOR_PARAMETER}"] = actual
    additional_mismatches: dict[str, Any] = {}

    if int(anchor_expectation.get("configured", 0)) != 1:
        fields["runtime_slam_pose_graph_anchor_parameter_status"] = (
            "not_configured"
        )
        fields["runtime_slam_pose_graph_anchor_parameter_match"] = 1
    else:
        expected_value = anchor_expectation.get("expected")
        fields["runtime_slam_pose_graph_anchor_parameter_expected"] = (
            expected_value
        )
        anchor_mismatches = pipeline.stage30_parameter_mismatches(
            {POSE_GRAPH_ANCHOR_PARAMETER: expected_value},
            parameters,
        )
        anchor_match = not anchor_mismatches
        fields["runtime_slam_pose_graph_anchor_parameter_status"] = (
            "matched" if anchor_match else "mismatched"
        )
        fields["runtime_slam_pose_graph_anchor_parameter_match"] = int(
            anchor_match
        )
        additional_mismatches.update(anchor_mismatches)

    scan_expected = scan_expectation.get("expected", {})
    if not isinstance(scan_expected, dict):
        scan_expected = {}
    scan_actual = {
        name: parameters.get(name, "<missing>") for name in scan_expected
    }
    scan_mismatches = pipeline.stage30_parameter_mismatches(
        scan_expected,
        parameters,
    ) if scan_expected else {}
    scan_match = not scan_mismatches
    fields.update(
        {
            "runtime_slam_scan_conditioning_parameter_status": (
                "not_configured"
                if not scan_expected
                else "matched" if scan_match else "mismatched"
            ),
            "runtime_slam_scan_conditioning_parameter_match": int(scan_match),
            "runtime_slam_scan_conditioning_parameter_expected_json": (
                json.dumps(scan_expected, sort_keys=True)
            ),
            "runtime_slam_scan_conditioning_parameter_actual_json": (
                json.dumps(scan_actual, sort_keys=True)
            ),
            "runtime_slam_scan_conditioning_parameter_mismatch_count": len(
                scan_mismatches
            ),
            "runtime_slam_scan_conditioning_parameter_mismatch_fields": (
                ";".join(sorted(scan_mismatches))
            ),
            "runtime_slam_scan_conditioning_parameter_mismatch_details_json": (
                json.dumps(scan_mismatches, sort_keys=True)
            ),
        }
    )
    additional_mismatches.update(scan_mismatches)
    adaptive_expected = adaptive_expectation.get("expected", {})
    if not isinstance(adaptive_expected, dict):
        adaptive_expected = {}
    adaptive_actual = {
        name: parameters.get(name, "<missing>")
        for name in adaptive_expected
    }
    adaptive_mismatches = pipeline.stage30_parameter_mismatches(
        adaptive_expected,
        parameters,
    ) if adaptive_expected else {}
    adaptive_match = not adaptive_mismatches
    fields.update(
        {
            "runtime_slam_adaptive_constraint_parameter_status": (
                "not_configured"
                if not adaptive_expected
                else "matched" if adaptive_match else "mismatched"
            ),
            "runtime_slam_adaptive_constraint_parameter_match": int(
                adaptive_match
            ),
            "runtime_slam_adaptive_constraint_parameter_expected_json": (
                json.dumps(adaptive_expected, sort_keys=True)
            ),
            "runtime_slam_adaptive_constraint_parameter_actual_json": (
                json.dumps(adaptive_actual, sort_keys=True)
            ),
            "runtime_slam_adaptive_constraint_parameter_mismatch_count": len(
                adaptive_mismatches
            ),
            "runtime_slam_adaptive_constraint_parameter_mismatch_fields": (
                ";".join(sorted(adaptive_mismatches))
            ),
            "runtime_slam_adaptive_constraint_parameter_mismatch_details_json": (
                json.dumps(adaptive_mismatches, sort_keys=True)
            ),
        }
    )
    additional_mismatches.update(adaptive_mismatches)
    if not additional_mismatches:
        return fields

    try:
        mismatch_details = json.loads(
            str(fields.get("runtime_slam_parameter_mismatch_details_json", "{}"))
        )
    except json.JSONDecodeError:
        mismatch_details = {}
    if not isinstance(mismatch_details, dict):
        mismatch_details = {}
    mismatch_details.update(additional_mismatches)
    fields.update(
        {
            "runtime_slam_parameter_status": "mismatched",
            "runtime_slam_parameter_match": 0,
            "runtime_slam_parameter_mismatch_count": len(mismatch_details),
            "runtime_slam_parameter_mismatch_fields": ";".join(
                sorted(mismatch_details)
            ),
            "runtime_slam_parameter_mismatch_details_json": json.dumps(
                mismatch_details,
                sort_keys=True,
            ),
        }
    )
    return fields


def latest_received_scan_count(text: str) -> int | None:
    matches = _RECEIVED_PATTERN.findall(text)
    return int(matches[-1]) if matches else None


def playback_finished(text: str) -> bool:
    return _PLAYBACK_FINISHED_PATTERN.search(text) is not None


def playback_failure(text: str) -> str:
    for pattern in (_PLAYBACK_FAILED_PATTERN, _SLAM_NODE_FAILED_PATTERN):
        match = pattern.search(text)
        if match is not None:
            return match.group(0).strip()
    return ""


def monitor_playback_failure(
    launch_log: Path,
    launch_proc: Any,
    stop_event: threading.Event,
    failure_state: dict[str, str],
    *,
    poll_s: float = 0.25,
) -> None:
    """Interrupt only this launch group if its rosbag child dies."""
    while not stop_event.is_set() and launch_proc.poll() is None:
        failure = playback_failure(_read_log(launch_log))
        if failure:
            failure_state["detail"] = failure
            try:
                os.killpg(int(launch_proc.pid), signal.SIGINT)
            except (OSError, ProcessLookupError, ValueError) as exc:
                failure_state["interrupt_error"] = repr(exc)
            return
        stop_event.wait(max(0.01, poll_s))


def expected_scan_count(row: dict[str, Any]) -> int | None:
    try:
        value = float(row.get("scan_laserscan_count", math.nan))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return int(value)


def latest_received_imu_audit(text: str) -> dict[str, Any] | None:
    matches = list(_IMU_AUDIT_PATTERN.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return {
        "count": int(match.group(1)),
        "first_stamp_ns": int(match.group(2)),
        "last_stamp_ns": int(match.group(3)),
        "max_positive_gap_ns": int(match.group(4)),
        "duplicate_count": int(match.group(5)),
        "out_of_order_count": int(match.group(6)),
        "fingerprint_fnv1a64": match.group(7).lower(),
    }


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_pipeline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("gole_slam_bag_pipeline_lossless", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def recorded_imu_audit(
    pipeline: ModuleType,
    row: dict[str, Any],
) -> dict[str, Any]:
    bag_path = Path(str(row.get("path", ""))).expanduser().resolve()
    imu_topic = str(row.get("imu_topic", "") or "/imu/data")
    cache_key = (str(bag_path), imu_topic)
    cached = _RECORDED_IMU_AUDIT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    reader = pipeline.reader_for_bag(bag_path)
    reader.set_filter(pipeline.rosbag2_py.StorageFilter(topics=[imu_topic]))
    imu_type = pipeline.get_message("sensor_msgs/msg/Imu")
    stamps: list[int] = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != imu_topic:
            continue
        message = pipeline.deserialize_message(data, imu_type)
        stamps.append(
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
    audit = pipeline.stage30_ordered_stamp_audit(stamps)
    audit["topic"] = imu_topic
    _RECORDED_IMU_AUDIT_CACHE[cache_key] = audit
    return audit


def saved_expected_imu_audit(bag_out: Path) -> dict[str, Any] | None:
    replay_path = bag_out / "replay_result.json"
    try:
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(replay, dict):
        return None
    count = replay.get("roughness_imu_delivery_expected_count")
    fingerprint = str(
        replay.get(
            "roughness_imu_delivery_expected_fingerprint_fnv1a64",
            "",
        )
    ).lower()
    if (
        type(count) is not int
        or count <= 0
        or re.fullmatch(r"[0-9a-f]{16}", fingerprint) is None
    ):
        return None
    return {
        "count": count,
        "fingerprint_fnv1a64": fingerprint,
        "source": "saved_replay",
    }


def saved_bag_storage_fingerprint(
    row: dict[str, Any],
    bag_out: Path,
) -> list[dict[str, Any]] | None:
    """Reuse a prior cryptographic fingerprint after strict stat validation."""
    provenance_path = bag_out / "replay_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(provenance, dict):
        return None

    bag_path = Path(str(row.get("path", ""))).expanduser().resolve()
    saved_bag_path = Path(
        str(provenance.get("bag_path", ""))
    ).expanduser().resolve()
    if not bag_path.is_dir() or bag_path != saved_bag_path:
        return None
    saved_fingerprint = provenance.get("bag_storage_fingerprint")
    if not isinstance(saved_fingerprint, list) or not saved_fingerprint:
        return None

    current_paths = {
        path.name: path
        for path in bag_path.iterdir()
        if path.is_file()
        and (path.name == "metadata.yaml" or path.suffix in {".db3", ".mcap"})
    }
    saved_by_name: dict[str, dict[str, Any]] = {}
    for item in saved_fingerprint:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        size_bytes = item.get("size_bytes")
        mtime_ns = item.get("mtime_ns")
        sha256 = str(item.get("sha256", ""))
        if (
            not isinstance(name, str)
            or not name
            or name in saved_by_name
            or type(size_bytes) is not int
            or size_bytes < 0
            or type(mtime_ns) is not int
            or mtime_ns < 0
            or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
        ):
            return None
        saved_by_name[name] = item
    if set(saved_by_name) != set(current_paths):
        return None

    for name, path in current_paths.items():
        stat = path.stat()
        saved_item = saved_by_name[name]
        if (
            stat.st_size != saved_item["size_bytes"]
            or stat.st_mtime_ns != saved_item["mtime_ns"]
        ):
            return None
        if name == "metadata.yaml" and _file_sha256(path) != saved_item["sha256"]:
            return None
    return copy.deepcopy(saved_fingerprint)


def rounded_stamp_exclusion_indices(
    recorded_stamps: list[int],
    rounded_stamps: list[int],
    *,
    tolerance_ns: int = 1_000_000,
) -> tuple[set[int], int]:
    excluded: set[int] = set()
    unmatched = 0
    for rounded_stamp in rounded_stamps:
        candidates = (
            (abs(stamp - rounded_stamp), index)
            for index, stamp in enumerate(recorded_stamps)
            if index not in excluded
        )
        best = min(candidates, default=None)
        if best is None or best[0] > tolerance_ns:
            unmatched += 1
            continue
        excluded.add(best[1])
    return excluded, unmatched


def tolerant_expected_scan_audit(
    pipeline: ModuleType,
    row: dict[str, Any],
    startup_drop_stamps_s: str,
) -> dict[str, Any]:
    try:
        recorded_stamps = list(pipeline.stage30_recorded_scan_stamps(row))
    except Exception as exc:
        return {
            "status": f"bag_scan_audit_error:{type(exc).__name__}",
            "recorded_count": "",
            "excluded_startup_stamp_count": 0,
            "unmatched_startup_stamp_count": 0,
        }
    rounded_stamps: list[int] = []
    invalid_count = 0
    for token in startup_drop_stamps_s.split(";"):
        if not token.strip():
            continue
        try:
            rounded_stamps.append(
                pipeline.stage30_decimal_stamp_seconds_to_ns(token)
            )
        except ValueError:
            invalid_count += 1
    excluded, unmatched = rounded_stamp_exclusion_indices(
        recorded_stamps, rounded_stamps
    )
    effective_stamps = [
        stamp
        for index, stamp in enumerate(recorded_stamps)
        if index not in excluded
    ]
    audit = pipeline.stage30_ordered_stamp_audit(effective_stamps)
    total_unmatched = unmatched + invalid_count
    audit.update(
        {
            "status": (
                "valid" if total_unmatched == 0 else "startup_drop_stamp_not_found"
            ),
            "recorded_count": len(recorded_stamps),
            "excluded_startup_stamp_count": len(excluded),
            "unmatched_startup_stamp_count": total_unmatched,
        }
    )
    return audit


def stationary_aware_pose_graph_coverage(
    pipeline: ModuleType,
    original_coverage: Any,
    bag_out: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fields = original_coverage(bag_out, args)
    if int(fields.get("pose_graph_coverage_complete", 0) or 0) == 1:
        return fields
    expected_motion_count = int(
        fields.get("pose_graph_coverage_expected_motion_count", 0) or 0
    )
    graph_samples = pipeline.stage31_load_samples(
        bag_out / "trajectory_pose_graph_final.csv"
    )
    if (
        expected_motion_count == 1
        and len(graph_samples) == 1
        and (bag_out / "map.png").is_file()
    ):
        fields.update(
            {
                "pose_graph_coverage_status": "stationary_single_node_complete",
                "pose_graph_coverage_complete": 1,
                "pose_graph_stationary_single_node": 1,
                "pose_graph_first_stamp_s": graph_samples[0][0],
                "pose_graph_last_stamp_s": graph_samples[0][0],
            }
        )
    return fields


def finalize_pending_scans(
    domain_id: int,
    *,
    ros_localhost_only: bool,
    timeout_s: float = 15.0,
) -> tuple[bool, str]:
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(domain_id)
    if ros_localhost_only:
        environment["ROS_LOCALHOST_ONLY"] = "1"
    try:
        completed = subprocess.run(
            [
                "ros2",
                "service",
                "call",
                "/slam_toolbox/finalize_roughness_pending",
                "slam_toolbox/srv/ClearQueue",
                "{}",
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)
    return completed.returncode == 0, completed.stdout.strip()


def _exact_success(value: Any) -> bool:
    return value is True or (
        type(value) in (int, float) and value == 1
    )


def _exact_zero(value: Any) -> bool:
    return value is False or (
        type(value) in (int, float) and value == 0
    )


def _strict_bool(value: Any) -> bool | None:
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    return None


def _nested_value(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_slam_2d_only_scope(
    row: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Keep this replay confined to the 2D SLAM LaserScan implementation."""
    if bool(getattr(args, "enable_laserscan_roughness_gate", False)):
        raise RuntimeError(
            "external LaserScan guard nodes are forbidden for 2D SLAM "
            "improvement; implement the feature inside slam_2d_core"
        )
    try:
        projection_enabled = bool(
            int(row.get("start_pointcloud_projection", 0) or 0)
        )
    except (TypeError, ValueError):
        projection_enabled = True
    if projection_enabled:
        raise RuntimeError(
            "PointCloud projection is forbidden in this 2D SLAM regression; "
            "use the recorded LaserScan directly"
        )
    if str(row.get("scan_topic", "")) != "/scan_laserscan":
        raise RuntimeError(
            "this 2D SLAM regression requires direct /scan_laserscan input"
        )


def _feature_enabled(
    saved_replay: dict[str, Any],
    direct_key: str,
    profile_name: str,
    profile_key: str,
) -> bool:
    evidence = (
        saved_replay.get(direct_key),
        _nested_value(
            saved_replay,
            "replay_fairness_contract",
            profile_name,
            profile_key,
        ),
    )
    return any(_strict_bool(value) is True for value in evidence)


def active_v5_required_artifacts(
    pipeline: ModuleType,
    saved_replay: dict[str, Any],
) -> set[str]:
    required = set(
        getattr(
            pipeline,
            "STAGE30_ARTIFACT_NAMES_V4",
            pipeline.stage30_artifact_names_for_schema(4),
        )
    )
    # stage30 always captures this dump, and semantic runtime matching cannot
    # be trusted without preserving the exact source YAML as an artifact.
    required.add("runtime_parameters/slam_toolbox.yaml")

    guard_enabled = _feature_enabled(
        saved_replay,
        "enable_laserscan_roughness_gate",
        "laserscan_guard_profile",
        "enabled",
    )
    if guard_enabled:
        required.update(
            {
                "guard_status_counts.csv",
                "guard_status_events.csv",
                "runtime_parameters/slam_laserscan_roughness_gate.yaml",
            }
        )

    innovation_enabled = _feature_enabled(
        saved_replay,
        "roughness_innovation_enabled",
        "roughness_innovation_profile",
        "enabled",
    )
    if innovation_enabled:
        required.add("roughness_innovation_events.csv")
        post_confirmation_enabled = _feature_enabled(
            saved_replay,
            "roughness_innovation_enable_post_confirmation",
            "roughness_innovation_profile",
            "enable_post_confirmation",
        )
        if post_confirmation_enabled:
            required.add("roughness_post_confirmation_events.csv")
    return required


def _positive_exact_int(value: Any) -> int | None:
    if type(value) is not int or value <= 0:
        return None
    return value


def existing_replay_semantic_rerun_reason(
    saved_replay: dict[str, Any],
    saved_log_fields: dict[str, Any],
    saved_coverage: dict[str, Any],
) -> str:
    if str(saved_replay.get("status", "")) not in ACCEPTED_REPLAY_STATUSES:
        return "existing_replay_status_not_accepted"

    if (
        not _exact_success(saved_replay.get("runtime_slam_parameter_match"))
        or str(saved_replay.get("runtime_slam_parameter_status", ""))
        != "matched"
        or not _exact_zero(
            saved_replay.get("runtime_slam_parameter_mismatch_count")
        )
        or saved_replay.get("runtime_slam_parameter_dump_return_code") != 0
        or _positive_exact_int(
            saved_replay.get("runtime_slam_parameter_count")
        )
        is None
    ):
        return "existing_replay_runtime_parameters_not_exact"

    if _exact_success(
        saved_replay.get(
            "runtime_slam_pose_graph_anchor_parameter_configured",
            0,
        )
    ) and not _exact_success(
        saved_replay.get("runtime_slam_pose_graph_anchor_parameter_match")
    ):
        return "existing_replay_pose_graph_anchor_parameter_mismatch"

    if (
        _positive_exact_int(
            saved_replay.get(
                "runtime_slam_scan_conditioning_parameter_configured_count"
            )
        )
        is not None
        and not _exact_success(
            saved_replay.get(
                "runtime_slam_scan_conditioning_parameter_match"
            )
        )
    ):
        return "existing_replay_scan_conditioning_parameter_mismatch"

    guard_enabled = _feature_enabled(
        saved_replay,
        "enable_laserscan_roughness_gate",
        "laserscan_guard_profile",
        "enabled",
    )
    if guard_enabled and (
        not _exact_success(saved_replay.get("runtime_guard_parameter_match"))
        or str(saved_replay.get("runtime_guard_parameter_status", ""))
        != "matched"
        or not _exact_zero(
            saved_replay.get("runtime_guard_parameter_mismatch_count")
        )
    ):
        return "existing_replay_guard_runtime_parameters_not_exact"

    if (
        not _exact_success(saved_log_fields.get("slam_input_complete"))
        or not _exact_success(saved_replay.get("slam_input_complete"))
        or not _exact_success(
            saved_replay.get("slam_recorded_scan_delivery_required")
        )
        or not _exact_success(
            saved_replay.get("slam_recorded_scan_delivery_complete")
        )
        or str(
            saved_replay.get("slam_recorded_scan_delivery_status", "")
        )
        != "complete"
        or not _exact_zero(
            saved_replay.get("slam_recorded_scan_delivery_count_delta")
        )
        or saved_replay.get("slam_recorded_scan_delivery_ratio") != 1.0
    ):
        return "existing_replay_scan_delivery_not_exact"

    expected_imu_count = _positive_exact_int(
        saved_replay.get("roughness_imu_delivery_expected_count")
    )
    received_imu_count = _positive_exact_int(
        saved_replay.get("roughness_imu_delivery_received_count")
    )
    expected_imu_fingerprint = str(
        saved_replay.get(
            "roughness_imu_delivery_expected_fingerprint_fnv1a64",
            "",
        )
    ).lower()
    received_imu_fingerprint = str(
        saved_replay.get("roughness_imu_received_fingerprint_fnv1a64", "")
    ).lower()
    if (
        not _exact_success(
            saved_replay.get("roughness_imu_delivery_complete")
        )
        or not _exact_success(
            saved_replay.get("roughness_imu_delivery_fingerprint_match")
        )
        or expected_imu_count is None
        or received_imu_count != expected_imu_count
        or re.fullmatch(r"[0-9a-f]{16}", expected_imu_fingerprint) is None
        or received_imu_fingerprint != expected_imu_fingerprint
    ):
        return "existing_replay_imu_delivery_not_exact"

    effective_scan_count = _positive_exact_int(
        saved_replay.get("subscription_drain_effective_expected_scan_count")
    )
    received_scan_count = _positive_exact_int(
        saved_replay.get("subscription_drain_received_scan_count")
    )
    subscription_expected_imu = _positive_exact_int(
        saved_replay.get("subscription_drain_expected_imu_count")
    )
    subscription_received_imu = _positive_exact_int(
        saved_replay.get("subscription_drain_received_imu_count")
    )
    worker_queued = _positive_exact_int(
        saved_replay.get("slam_queue_direct_post_playback_worker_queued")
    )
    worker_completed = _positive_exact_int(
        saved_replay.get("slam_queue_direct_post_playback_worker_completed")
    )
    if (
        not _exact_success(saved_replay.get("queue_drain_enabled"))
        or not _exact_success(saved_replay.get("queue_drain_complete"))
        or str(saved_replay.get("queue_drain_evidence", ""))
        != "direct_node_diagnostics"
        or not _exact_success(
            saved_replay.get("subscription_drain_complete")
        )
        or not _exact_success(
            saved_replay.get("subscription_finalize_pending_success")
        )
        or str(saved_replay.get("subscription_playback_failure", ""))
        or effective_scan_count is None
        or received_scan_count != effective_scan_count
        or subscription_expected_imu is None
        or subscription_received_imu != subscription_expected_imu
        or not _exact_success(
            saved_replay.get("slam_queue_worker_completion_verified")
        )
        or worker_queued is None
        or worker_completed != worker_queued
        or not _exact_zero(
            saved_replay.get(
                "slam_queue_direct_post_playback_worker_in_flight"
            )
        )
    ):
        return "existing_replay_queue_or_subscription_not_exact"

    if (
        not _exact_success(saved_replay.get("pose_graph_captured"))
        or not _exact_success(
            saved_replay.get("pose_graph_coverage_complete")
        )
        or not _exact_success(
            saved_coverage.get("pose_graph_coverage_complete")
        )
    ):
        return "existing_replay_pose_graph_not_exact"
    return ""


def legacy_command_rewrite_integrity_rerun_reason(
    pipeline: ModuleType,
    bag_out: Path,
    saved_replay: dict[str, Any],
    adapter_signature: str,
    saved_signature: str,
    current_fairness_signature: str,
    saved_fairness_signature: str,
) -> str:
    """Validate the command file mutation made before the legacy skip check.

    The legacy runner writes commands.txt before checking --skip-existing.  An
    adapter provenance signature therefore changes that artifact even when the
    replay itself is reusable.  Accept only the exact replay and fairness
    signature substitutions whose reconstructed bytes match both manifests.
    """
    commands_path = bag_out / "commands.txt"
    try:
        current_commands = commands_path.read_bytes()
    except OSError:
        return "existing_replay_legacy_commands_missing"

    matches = list(_COMMAND_REPLAY_SIGNATURE_PATTERN.finditer(current_commands))
    fairness_matches = list(
        _COMMAND_FAIRNESS_SIGNATURE_PATTERN.finditer(current_commands)
    )
    if (
        len(matches) != 1
        or matches[0].group(1).decode("ascii") != adapter_signature
        or len(fairness_matches) != 1
        or fairness_matches[0].group(1).decode("ascii")
        != current_fairness_signature
    ):
        return "existing_replay_legacy_commands_mismatch"
    legacy_commands = _COMMAND_REPLAY_SIGNATURE_PATTERN.sub(
        b"replay_signature_sha256=" + saved_signature.encode("ascii"),
        current_commands,
        count=1,
    )
    legacy_commands = _COMMAND_FAIRNESS_SIGNATURE_PATTERN.sub(
        b"replay_fairness_contract_sha256="
        + saved_fairness_signature.encode("ascii"),
        legacy_commands,
        count=1,
    )
    saved_domain = saved_replay.get("ros_domain_id")
    domain_matches = list(
        _COMMAND_ROS_DOMAIN_PATTERN.finditer(legacy_commands)
    )
    if saved_domain is not None:
        if (
            type(saved_domain) is not int
            or not 0 <= saved_domain <= 232
            or len(domain_matches) != 1
        ):
            return "existing_replay_legacy_commands_mismatch"
        legacy_commands = _COMMAND_ROS_DOMAIN_PATTERN.sub(
            b"ROS_DOMAIN_ID=" + str(saved_domain).encode("ascii"),
            legacy_commands,
            count=1,
        )
    elif domain_matches:
        return "existing_replay_legacy_commands_mismatch"
    current_commands_entry = {
        "size_bytes": len(current_commands),
        "sha256": hashlib.sha256(current_commands).hexdigest(),
    }
    legacy_commands_entry = {
        "size_bytes": len(legacy_commands),
        "sha256": hashlib.sha256(legacy_commands).hexdigest(),
    }

    saved_manifest = saved_replay.get("artifact_manifest")
    if (
        not isinstance(saved_manifest, dict)
        or saved_manifest.get("commands.txt") != legacy_commands_entry
    ):
        return "existing_replay_legacy_commands_mismatch"
    schema = int(saved_replay.get("replay_provenance_schema", 3) or 3)
    current_manifest = pipeline.stage30_artifact_manifest(
        bag_out,
        schema=schema,
    )
    if current_manifest.get("commands.txt") != current_commands_entry:
        return "existing_replay_artifact_manifest_mismatch"
    normalized_manifest = copy.deepcopy(current_manifest)
    normalized_manifest["commands.txt"] = legacy_commands_entry
    if saved_manifest != normalized_manifest:
        return "existing_replay_artifact_manifest_mismatch"

    integrity_path = bag_out / "replay_integrity.json"
    try:
        integrity_payload = json.loads(integrity_path.read_text(encoding="utf-8"))
    except OSError:
        return "existing_missing_replay_integrity"
    except json.JSONDecodeError:
        return "existing_invalid_replay_integrity_json"
    if not isinstance(integrity_payload, dict) or integrity_payload.get(
        "schema"
    ) != 1:
        return "existing_unsupported_replay_integrity_schema"
    saved_integrity = integrity_payload.get("files")
    if (
        not isinstance(saved_integrity, dict)
        or len(saved_integrity) != 5
        or any(
            not isinstance(item, dict)
            or not _exact_success(item.get("exists"))
            for item in saved_integrity.values()
        )
    ):
        return "existing_incomplete_replay_integrity_manifest"

    current_integrity = pipeline.stage30_replay_integrity_manifest(
        bag_out,
        saved_replay,
    )
    current_integrity_commands = current_integrity.get("commands")
    if not isinstance(current_integrity_commands, dict) or any(
        current_integrity_commands.get(key) != value
        for key, value in current_commands_entry.items()
    ):
        return "existing_replay_integrity_mismatch"
    normalized_integrity = copy.deepcopy(current_integrity)
    normalized_integrity["commands"].update(legacy_commands_entry)
    if saved_integrity != normalized_integrity:
        return "existing_replay_integrity_mismatch"

    restore_path = commands_path.with_name(
        f".{commands_path.name}.lossless_restore.{os.getpid()}"
    )
    try:
        restore_path.write_bytes(legacy_commands)
        os.replace(restore_path, commands_path)
    except OSError:
        try:
            restore_path.unlink(missing_ok=True)
        except OSError:
            pass
        return "existing_replay_legacy_commands_restore_failed"
    if _file_sha256(commands_path) != legacy_commands_entry["sha256"]:
        return "existing_replay_legacy_commands_restore_failed"
    return ""


def restore_current_commands_after_reuse_rejection(
    commands_path: Path,
    current_commands: bytes | None,
) -> None:
    """Undo a temporary legacy-command restore before a real replay.

    The legacy runner writes the command file before invoking its skip check.
    A compatibility check may temporarily restore the command bytes recorded by
    an older replay. If a later semantic check rejects that replay, the new run
    must retain the command bytes that it will actually execute.
    """
    if current_commands is None:
        return
    try:
        if commands_path.read_bytes() == current_commands:
            return
    except OSError:
        pass

    restore_path = commands_path.with_name(
        f".{commands_path.name}.lossless_current.{os.getpid()}"
    )
    try:
        restore_path.write_bytes(current_commands)
        os.replace(restore_path, commands_path)
        if commands_path.read_bytes() != current_commands:
            raise OSError("restored command bytes do not match")
    except OSError as exc:
        try:
            restore_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"failed to restore current replay commands: {commands_path}"
        ) from exc


def existing_replay_summary_rerun_reason(summary: dict[str, Any]) -> str:
    if int(summary.get("map_count", 0) or 0) <= 0:
        return "existing_summary_has_no_usable_map_or_trajectory"
    if (
        int(summary.get("trajectory_map_frame_count", 0) or 0) <= 1
        and int(summary.get("trajectory_odom_frame_count", 0) or 0) <= 1
    ):
        return "existing_summary_has_no_usable_map_or_trajectory"
    return ""


def compatible_saved_legacy_signature(
    bag_out: Path,
    saved_replay: dict[str, Any],
    current_legacy_payload: dict[str, Any],
) -> tuple[str, str]:
    """Accept legacy provenance changes proven inactive for this replay."""
    provenance_path = bag_out / "replay_provenance.json"
    try:
        saved_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except OSError:
        return "", "existing_replay_saved_provenance_missing"
    except json.JSONDecodeError:
        return "", "existing_replay_saved_provenance_invalid"
    if not isinstance(saved_payload, dict):
        return "", "existing_replay_saved_provenance_invalid"

    saved_signature = str(saved_replay.get("replay_signature_sha256", ""))
    if (
        re.fullmatch(r"[0-9a-f]{64}", saved_signature) is None
        or _payload_sha256(saved_payload) != saved_signature
    ):
        return "", "existing_replay_saved_provenance_signature_mismatch"

    saved = copy.deepcopy(saved_payload)
    current = copy.deepcopy(current_legacy_payload)
    saved_adapter = saved.pop("lossless_stage34_adapter", None)
    if saved_adapter is not None and (
        not isinstance(saved_adapter, dict)
        or saved_adapter.get("schema") != ADAPTER_PROVENANCE_SCHEMA
        or saved_adapter.get("path") != str(ADAPTER_PATH)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(saved_adapter.get("sha256", "")),
        )
        is None
    ):
        return "", "existing_replay_saved_adapter_provenance_invalid"
    saved_fairness = saved.get("fairness_contract")
    current_fairness = current.get("fairness_contract")
    if not isinstance(saved_fairness, dict) or not isinstance(
        current_fairness, dict
    ):
        return "", "existing_replay_legacy_provenance_mismatch"
    saved_backend = saved_fairness.get("backend_asset_sha256")
    current_backend = current_fairness.get("backend_asset_sha256")
    if not isinstance(saved_backend, dict) or not isinstance(
        current_backend, dict
    ):
        return "", "existing_replay_legacy_provenance_mismatch"

    if (
        saved.get("launch_command") != current.get("launch_command")
        or not saved_backend.get("slam_launch_file")
        or saved_backend.get("slam_launch_file")
        != current_backend.get("slam_launch_file")
    ):
        return "", "existing_replay_legacy_launch_asset_mismatch"

    saved_sources = saved.get("source_sha256")
    current_sources = current.get("source_sha256")
    if not isinstance(saved_sources, dict) or not isinstance(
        current_sources, dict
    ):
        return "", "existing_replay_legacy_provenance_mismatch"

    launch_source_suffix = (
        "/gole_autonomy/gole_slam/slam_2d_bringup/launch/"
        "slam_from_bag_launch.py"
    )
    launch_command = str(current.get("launch_command", ""))
    if launch_source_suffix not in launch_command:
        for sources in (saved_sources, current_sources):
            for path in list(sources):
                if str(path).endswith(launch_source_suffix):
                    sources.pop(path, None)
        saved_backend.pop("slam_from_bag_launch", None)
        current_backend.pop("slam_from_bag_launch", None)

    saved_guard_enabled = _strict_bool(
        saved_replay.get("enable_laserscan_roughness_gate")
    )
    current_guard_match = re.search(
        r"(?:^|\s)enable_laserscan_roughness_gate:=(\S+)(?:\s|$)",
        launch_command,
    )
    current_guard_enabled = (
        _strict_bool(current_guard_match.group(1))
        if current_guard_match is not None
        else None
    )
    if saved_guard_enabled is False and current_guard_enabled is False:
        guard_source_marker = "/amcl_guard/"
        for sources in (saved_sources, current_sources):
            for path in list(sources):
                if guard_source_marker in str(path):
                    sources.pop(path, None)
        for key in (
            "amcl_guard_node",
            "amcl_guard_header",
            "amcl_guard_source",
            "imu_quality_policy_source",
            "laser_scan_stabilizer_source",
        ):
            saved_backend.pop(key, None)
            current_backend.pop(key, None)

    if saved != current:
        return "", "existing_replay_legacy_provenance_mismatch"
    return saved_signature, ""


def compatible_existing_replay_rerun_reason(
    pipeline: ModuleType,
    original_reason: Any,
    bag_out: Path,
    summary: dict[str, Any],
    saved_replay: dict[str, Any],
    saved_log_fields: dict[str, Any],
    saved_coverage: dict[str, Any],
    replay_signature: str,
    require_queue_drain: bool,
) -> str:
    """Validate schema-v5 replays and only relax inactive V5 artifacts."""
    saved_manifest = saved_replay.get("artifact_manifest")
    schema = int(saved_replay.get("replay_provenance_schema", 3) or 3)
    supported_names = tuple(pipeline.stage30_artifact_names_for_schema(schema))
    if not isinstance(saved_manifest, dict) or schema < 5:
        return original_reason(
            bag_out,
            summary,
            saved_replay,
            saved_log_fields,
            saved_coverage,
            replay_signature,
            require_queue_drain,
        )

    recorded_names = set(saved_manifest)
    supported_set = set(supported_names)
    required_names = active_v5_required_artifacts(pipeline, saved_replay)
    if (
        not required_names.issubset(recorded_names)
        or not recorded_names.issubset(supported_set)
    ):
        return original_reason(
            bag_out,
            summary,
            saved_replay,
            saved_log_fields,
            saved_coverage,
            replay_signature,
            require_queue_drain,
        )

    signature_for_validation = replay_signature
    legacy_signature = _LEGACY_SIGNATURE_BY_ADAPTER_SIGNATURE.get(
        replay_signature,
        "",
    )
    current_legacy_payload = _LEGACY_PAYLOAD_BY_ADAPTER_SIGNATURE.get(
        replay_signature
    )
    saved_signature = str(
        saved_replay.get("replay_signature_sha256", "")
    )
    has_lossless_adapter_evidence = all(
        key in saved_replay
        for key in (
            "subscription_drain_complete",
            "subscription_finalize_pending_success",
            "roughness_imu_delivery_complete",
            "roughness_imu_delivery_fingerprint_match",
        )
    )
    if (
        saved_signature != replay_signature
        and legacy_signature
        and saved_signature == legacy_signature
        and has_lossless_adapter_evidence
    ):
        signature_for_validation = legacy_signature
    elif (
        saved_signature != replay_signature
        and legacy_signature
        and has_lossless_adapter_evidence
    ):
        if not isinstance(current_legacy_payload, dict):
            return "existing_replay_legacy_provenance_unavailable"
        compatible_signature, compatibility_reason = (
            compatible_saved_legacy_signature(
                bag_out,
                saved_replay,
                current_legacy_payload,
            )
        )
        if compatibility_reason:
            return compatibility_reason
        signature_for_validation = compatible_signature

    legacy_command_rewrite = (
        bool(signature_for_validation)
        and signature_for_validation == saved_signature
        and replay_signature != saved_signature
    )
    if legacy_command_rewrite:
        if not isinstance(current_legacy_payload, dict):
            return "existing_replay_legacy_provenance_unavailable"
        current_fairness_contract = current_legacy_payload.get(
            "fairness_contract"
        )
        saved_fairness_signature = str(
            saved_replay.get("replay_fairness_contract_sha256", "")
        )
        if (
            not isinstance(current_fairness_contract, dict)
            or re.fullmatch(
                r"[0-9a-f]{64}", saved_fairness_signature
            )
            is None
        ):
            return "existing_replay_fairness_signature_invalid"
        current_fairness_signature = _payload_sha256(
            current_fairness_contract
        )
        reason = legacy_command_rewrite_integrity_rerun_reason(
            pipeline,
            bag_out,
            saved_replay,
            replay_signature,
            saved_signature,
            current_fairness_signature,
            saved_fairness_signature,
        )
        if reason:
            return reason
        reason = existing_replay_summary_rerun_reason(summary)
        if reason:
            return reason
        return existing_replay_semantic_rerun_reason(
            saved_replay,
            saved_log_fields,
            saved_coverage,
        )

    commands_path = bag_out / "commands.txt"
    saved_commands_entry = (
        saved_manifest.get("commands.txt")
        if isinstance(saved_manifest, dict)
        else None
    )
    try:
        current_commands = commands_path.read_bytes()
    except OSError:
        current_commands = b""
    current_commands_entry = {
        "size_bytes": len(current_commands),
        "sha256": hashlib.sha256(current_commands).hexdigest(),
    }
    if (
        isinstance(saved_commands_entry, dict)
        and commands_path.is_file()
        and type(saved_commands_entry.get("size_bytes")) is int
        and isinstance(saved_commands_entry.get("sha256"), str)
        and _COMMAND_ROS_DOMAIN_PATTERN.search(current_commands) is not None
        and saved_commands_entry != current_commands_entry
        and saved_signature == replay_signature
    ):
        saved_fairness_signature = str(
            saved_replay.get("replay_fairness_contract_sha256", "")
        )
        if re.fullmatch(r"[0-9a-f]{64}", saved_fairness_signature) is None:
            return "existing_replay_fairness_signature_invalid"
        reason = legacy_command_rewrite_integrity_rerun_reason(
            pipeline,
            bag_out,
            saved_replay,
            replay_signature,
            saved_signature,
            saved_fairness_signature,
            saved_fairness_signature,
        )
        if reason:
            return reason
        reason = existing_replay_summary_rerun_reason(summary)
        if reason:
            return reason
        return existing_replay_semantic_rerun_reason(
            saved_replay,
            saved_log_fields,
            saved_coverage,
        )

    recorded_supported_names = tuple(
        name for name in supported_names if name in recorded_names
    )
    original_name_resolver = pipeline.stage30_artifact_names_for_schema
    pipeline.stage30_artifact_names_for_schema = (
        lambda requested_schema: recorded_supported_names
        if int(requested_schema) == schema
        else original_name_resolver(requested_schema)
    )
    try:
        reason = original_reason(
            bag_out,
            summary,
            saved_replay,
            saved_log_fields,
            saved_coverage,
            signature_for_validation,
            require_queue_drain,
        )
        if reason:
            return reason
        return existing_replay_semantic_rerun_reason(
            saved_replay,
            saved_log_fields,
            saved_coverage,
        )
    finally:
        pipeline.stage30_artifact_names_for_schema = original_name_resolver


def install_recorded_scan_drain_guard(pipeline: ModuleType) -> None:
    original_run_one = pipeline.stage30_run_one
    original_wait = pipeline.stage30_wait_for_slam_queue_drain
    original_queue_log_fields_for_args = pipeline.stage30_queue_log_fields_for_args
    original_capture_final_pose_graph = pipeline.stage30_capture_final_pose_graph
    original_log_fields = pipeline.stage30_roughness_innovation_log_fields
    original_capture_slam_parameters = pipeline.stage30_capture_slam_parameters
    original_replay_provenance_payload = (
        pipeline.stage30_replay_provenance_payload
    )
    original_pose_graph_coverage = pipeline.stage30_pose_graph_coverage_fields
    original_existing_replay_reason = (
        pipeline.stage30_existing_replay_rerun_reason
    )
    original_bag_storage_fingerprint = pipeline.stage30_bag_storage_fingerprint
    original_stage34_replay = pipeline.stage34_roughness_gate_replay
    original_stage34_comparison_status = pipeline.stage34_comparison_status
    active_expected_count: int | None = None
    active_imu_audit: dict[str, Any] | None = None
    active_row: dict[str, Any] | None = None
    active_bag_out: Path | None = None
    active_domain_id: int | None = None
    active_launch_log: Path | None = None
    active_failure_state: dict[str, str] | None = None
    active_watchdog_stop: threading.Event | None = None
    active_watchdog_thread: threading.Thread | None = None
    intentional_node_count_change_active = False

    def worker_aware_queue_log_fields(
        path: Path,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        fields = original_queue_log_fields_for_args(path, args)
        fields.update(
            {
                "slam_queue_direct_worker_summary_count": 0,
                "slam_queue_direct_post_playback_worker_summary_count": 0,
                "slam_queue_direct_latest_worker_queued": "",
                "slam_queue_direct_latest_worker_in_flight": "",
                "slam_queue_direct_latest_worker_completed": "",
                "slam_queue_direct_post_playback_worker_queued": "",
                "slam_queue_direct_post_playback_worker_in_flight": "",
                "slam_queue_direct_post_playback_worker_completed": "",
                "slam_queue_worker_completion_verified": 0,
            }
        )
        text = _read_log(path)
        lines = text.splitlines()
        playback_finish_indices = [
            index
            for index, line in enumerate(lines)
            if _PLAYBACK_FINISHED_PATTERN.search(line)
        ]
        samples: list[tuple[int, int, int, int]] = []
        for index, line in enumerate(lines):
            match = _WORKER_DRAIN_PATTERN.search(line)
            if match:
                samples.append(
                    (
                        index,
                        int(match.group(1)),
                        int(match.group(3)),
                        int(match.group(4)),
                    )
                )
        fields["slam_queue_direct_worker_summary_count"] = len(samples)
        if samples:
            _, queued, in_flight, completed = samples[-1]
            fields["slam_queue_direct_latest_worker_queued"] = queued
            fields["slam_queue_direct_latest_worker_in_flight"] = in_flight
            fields["slam_queue_direct_latest_worker_completed"] = completed
        post_playback_samples = (
            [sample for sample in samples if sample[0] > playback_finish_indices[-1]]
            if playback_finish_indices
            else []
        )
        fields["slam_queue_direct_post_playback_worker_summary_count"] = len(
            post_playback_samples
        )
        if post_playback_samples:
            _, queued, in_flight, completed = post_playback_samples[-1]
            fields["slam_queue_direct_post_playback_worker_queued"] = queued
            fields["slam_queue_direct_post_playback_worker_in_flight"] = in_flight
            fields["slam_queue_direct_post_playback_worker_completed"] = completed
            fields["slam_queue_worker_completion_verified"] = int(
                in_flight == 0 and completed == queued
            )
        return fields

    def capture_final_pose_graph_with_timeout_recovery(
        args: argparse.Namespace,
        bag_out: Path,
        env: dict[str, str],
    ) -> dict[str, Any]:
        capture_started_ns = time.time_ns()
        fields = original_capture_final_pose_graph(args, bag_out, env)
        fields.update(
            {
                "pose_graph_serialize_timeout_recovery_attempted": 0,
                "pose_graph_serialize_timeout_recovered": 0,
                "pose_graph_serialize_timeout_recovery_status": "not_needed",
                "pose_graph_serialize_completion_grace_s": (
                    POSE_GRAPH_SERIALIZATION_COMPLETION_GRACE_S
                ),
            }
        )
        if int(fields.get("pose_graph_serialize_return_code", -1)) != 124:
            return fields

        fields["pose_graph_serialize_timeout_recovery_attempted"] = 1
        stem = Path(str(fields.get("pose_graph_stem", bag_out / "final_pose_graph")))
        serialized_paths = (
            stem.with_suffix(".posegraph"),
            stem.with_suffix(".data"),
        )
        stable_sizes: tuple[int, int] | None = None
        stable_since: float | None = None
        deadline = time.monotonic() + POSE_GRAPH_SERIALIZATION_COMPLETION_GRACE_S
        while time.monotonic() <= deadline:
            try:
                stats = tuple(path.stat() for path in serialized_paths)
            except OSError:
                stable_sizes = None
                stable_since = None
            else:
                sizes = tuple(stat.st_size for stat in stats)
                fresh = all(
                    stat.st_mtime_ns
                    >= (
                        capture_started_ns
                        - POSE_GRAPH_SERIALIZATION_FRESHNESS_TOLERANCE_NS
                    )
                    for stat in stats
                )
                if fresh and all(size > 0 for size in sizes):
                    if sizes != stable_sizes:
                        stable_sizes = sizes
                        stable_since = time.monotonic()
                    elif (
                        stable_since is not None
                        and time.monotonic() - stable_since
                        >= POSE_GRAPH_SERIALIZATION_STABLE_WINDOW_S
                    ):
                        break
                else:
                    stable_sizes = None
                    stable_since = None
            time.sleep(0.1)
        else:
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "fresh_serialized_files_not_stable"
            )
            return fields

        export_command = str(fields.get("pose_graph_export_command", ""))
        if not export_command:
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "missing_export_command"
            )
            return fields

        log_path = Path(
            str(fields.get("pose_graph_export_log", bag_out / "pose_graph_export.log"))
        )
        log_lines = [
            "\n# timeout_recovery: fresh serialized files became stable; "
            "running offline exporter\n",
            f"# recovered_posegraph_size_bytes: {stable_sizes[0]}\n",
            f"# recovered_data_size_bytes: {stable_sizes[1]}\n",
        ]
        try:
            completed = subprocess.run(
                ["bash", "-lc", export_command],
                cwd=args.workspace,
                env=env,
                capture_output=True,
                text=True,
                timeout=POSE_GRAPH_RECOVERY_EXPORT_TIMEOUT_S,
                check=False,
            )
            fields["pose_graph_export_return_code"] = completed.returncode
            log_lines.extend([completed.stdout, completed.stderr])
        except subprocess.TimeoutExpired as exc:
            fields["pose_graph_export_return_code"] = 124
            fields["pose_graph_export_error"] = "timeout_recovery_export_timeout"
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "offline_export_timeout"
            )
            log_lines.extend([exc.stdout or "", exc.stderr or ""])
        except OSError as exc:
            fields["pose_graph_export_return_code"] = -1
            fields["pose_graph_export_error"] = "timeout_recovery_export_os_error"
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "offline_export_os_error"
            )
            log_lines.append(repr(exc))

        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("".join(str(line) for line in log_lines))

        trajectory_csv = Path(
            str(
                fields.get(
                    "pose_graph_trajectory_csv",
                    bag_out / "trajectory_pose_graph_final.csv",
                )
            )
        )
        rows = (
            pipeline.read_csv_rows(trajectory_csv)
            if trajectory_csv.is_file()
            else []
        )
        fields["pose_graph_trajectory_count"] = len(rows)
        recovered = (
            int(fields.get("pose_graph_export_return_code", -1)) == 0
            and len(rows) > 1
        )
        fields["pose_graph_captured"] = int(recovered)
        fields["pose_graph_serialize_timeout_recovered"] = int(recovered)
        if recovered:
            fields.pop("pose_graph_export_error", None)
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "recovered_from_fresh_stable_files"
            )
            fields["pose_graph_serialize_completion_evidence"] = (
                "fresh_nonempty_stable_posegraph_and_data_with_successful_export"
            )
        elif fields["pose_graph_serialize_timeout_recovery_status"] == "not_needed":
            fields["pose_graph_serialize_timeout_recovery_status"] = (
                "offline_export_failed"
            )
        return fields

    def conditioning_aware_stage34_comparison_status(**kwargs: Any) -> str:
        if intentional_node_count_change_active:
            kwargs["pose_graph_node_parity_complete"] = True
        return original_stage34_comparison_status(**kwargs)

    def conditioning_aware_stage34_replay(args: argparse.Namespace) -> None:
        nonlocal intentional_node_count_change_active
        profile = configured_scan_conditioning_parameters(
            pipeline,
            args,
        )["effective"]
        intentional_node_count_change_active = (
            scan_conditioning_can_change_pose_graph_node_count(profile)
        )
        try:
            original_stage34_replay(args)
            output = Path(args.output).expanduser().resolve()
            label = pipeline.sanitize_name(args.snapshot_label)
            csv_path = (
                output
                / "stage34_roughness_gate_replay"
                / f"roughness_gate_replay_{label}.csv"
            )
            if not csv_path.is_file():
                return
            rows = pipeline.read_csv_rows(csv_path)
            rows = enrich_stage34_rows_with_roughness_place_metrics(rows)
            if intentional_node_count_change_active:
                rows = relax_intentional_conditioning_node_parity_rows(rows)
            pipeline.write_csv(
                csv_path,
                rows,
            )
        finally:
            intentional_node_count_change_active = False

    def watched_capture_slam_parameters(
        args: argparse.Namespace,
        bag_out: Path,
        env: dict[str, str],
        launch_proc: Any,
    ) -> dict[str, Any]:
        nonlocal active_watchdog_thread
        fields = augment_slam_parameter_capture(
            pipeline,
            original_capture_slam_parameters,
            args,
            bag_out,
            env,
            launch_proc,
        )
        if (
            active_launch_log is not None
            and active_failure_state is not None
            and active_watchdog_stop is not None
            and active_watchdog_thread is None
            and launch_proc.poll() is None
        ):
            active_watchdog_thread = threading.Thread(
                target=monitor_playback_failure,
                args=(
                    active_launch_log,
                    launch_proc,
                    active_watchdog_stop,
                    active_failure_state,
                ),
                name="slam-rosbag-playback-watchdog",
                daemon=True,
            )
            active_watchdog_thread.start()
        return fields

    def enhanced_log_fields(
        path: Path,
        events_path: Path | None = None,
        post_events_path: Path | None = None,
        near_chain_events_path: Path | None = None,
    ) -> dict[str, Any]:
        fields = compatible_roughness_log_fields(
            original_log_fields,
            path,
            events_path,
            post_events_path,
            near_chain_events_path,
        )
        received = latest_received_imu_audit(_read_log(path))
        expected = active_imu_audit
        if expected is None:
            return fields
        fields["roughness_imu_delivery_expected_count"] = int(
            expected.get("count", 0)
        )
        fields["roughness_imu_delivery_expected_fingerprint_fnv1a64"] = str(
            expected.get("fingerprint_fnv1a64", "")
        ).lower()
        if received is None:
            fields["roughness_imu_delivery_received_count"] = 0
            fields["roughness_imu_delivery_complete"] = 0
            return fields
        for key, value in received.items():
            fields[f"roughness_imu_received_{key}"] = value
        fields["roughness_imu_delivery_received_count"] = received["count"]
        fields["roughness_imu_delivery_count_delta"] = (
            int(received["count"]) - int(expected.get("count", 0))
        )
        fingerprint_match = str(received["fingerprint_fnv1a64"]).lower() == str(
            expected.get("fingerprint_fnv1a64", "")
        ).lower()
        fields["roughness_imu_delivery_fingerprint_match"] = int(
            fingerprint_match
        )
        fields["roughness_imu_delivery_complete"] = int(
            int(received["count"]) == int(expected.get("count", 0))
            and fingerprint_match
        )
        return fields

    def guarded_wait(
        launch_log: Path,
        launch_proc: Any,
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        expected = active_expected_count
        expected_imu = active_imu_audit
        enabled = bool(getattr(args, "wait_for_slam_queue_drain", True))
        innovation_enabled = bool(
            getattr(args, "use_roughness_innovation_slam_toolbox", False)
        )
        if (
            not enabled
            or not innovation_enabled
            or expected is None
            or expected_imu is None
        ):
            return original_wait(launch_log, launch_proc, args)

        timeout_s = max(
            0.0, float(getattr(args, "queue_drain_timeout_s", 180.0))
        )
        started = time.monotonic()
        deadline = started + timeout_s
        observed: int | None = None
        observed_imu: dict[str, Any] | None = None
        startup_trim = 0
        effective_expected = expected
        reached = False
        finalize_success = False
        finalize_output = ""
        failure_detail = ""
        while True:
            text = _read_log(launch_log)
            failure_detail = playback_failure(text)
            if failure_detail:
                if active_failure_state is not None:
                    active_failure_state["detail"] = failure_detail
                break
            observed = latest_received_scan_count(text)
            observed_imu = latest_received_imu_audit(text)
            if playback_finished(text):
                queue_fields = pipeline.stage30_queue_log_fields_for_args(
                    launch_log, args
                )
                startup_trim = int(
                    queue_fields.get("slam_expected_startup_tf_drop_count", 0)
                    or 0
                )
                effective_expected = max(0, expected - startup_trim)
            if (
                playback_finished(text)
                and observed is not None
                and observed >= effective_expected
                and observed_imu is not None
                and int(observed_imu["count"]) >= int(expected_imu["count"])
                and int(
                    queue_fields.get(
                        "slam_queue_worker_completion_verified",
                        0,
                    ) or 0
                ) == 1
            ):
                reached = True
                break
            if launch_proc.poll() is not None or time.monotonic() >= deadline:
                break
            time.sleep(0.5)

        wait_s = time.monotonic() - started
        if reached and active_domain_id is not None:
            finalize_success, finalize_output = finalize_pending_scans(
                active_domain_id,
                ros_localhost_only=bool(
                    getattr(args, "ros_localhost_only", False)
                ),
            )
            reached = finalize_success
        if reached:
            result = original_wait(launch_log, launch_proc, args)
        else:
            no_extra_wait_args = copy.copy(args)
            no_extra_wait_args.queue_drain_timeout_s = 0.0
            result = original_wait(launch_log, launch_proc, no_extra_wait_args)
            result["queue_drain_complete"] = 0
            if failure_detail:
                result["queue_drain_status"] = "bag_playback_process_failed"
                result["queue_drain_evidence"] = "launch_log"
            else:
                result["queue_drain_status"] = "timeout_recorded_scan_delivery"
                result["queue_drain_evidence"] = "recorded_scan_count"
        result.update(
            {
                "subscription_drain_expected_scan_count": expected,
                "subscription_drain_effective_expected_scan_count": (
                    effective_expected
                ),
                "subscription_drain_startup_trim_count": startup_trim,
                "subscription_drain_received_scan_count": (
                    observed if observed is not None else ""
                ),
                "subscription_drain_complete": int(reached),
                "subscription_drain_wait_s": wait_s,
                "subscription_drain_expected_imu_count": int(
                    expected_imu["count"]
                ),
                "subscription_drain_received_imu_count": (
                    int(observed_imu["count"])
                    if observed_imu is not None else ""
                ),
                "subscription_finalize_pending_success": int(finalize_success),
                "subscription_finalize_pending_output": finalize_output[-2000:],
                "subscription_playback_failure": failure_detail,
            }
        )
        return result

    def guarded_run_one(
        row: dict[str, Any],
        args: argparse.Namespace,
        out_dir: Path,
        idx: int,
        total: int,
    ) -> dict[str, Any]:
        nonlocal active_expected_count, active_imu_audit, active_row
        nonlocal active_bag_out
        nonlocal active_domain_id
        nonlocal active_launch_log, active_failure_state
        nonlocal active_watchdog_stop, active_watchdog_thread
        previous = active_expected_count
        previous_imu_audit = active_imu_audit
        previous_row = active_row
        previous_bag_out = active_bag_out
        previous_domain_id = active_domain_id
        previous_launch_log = active_launch_log
        previous_failure_state = active_failure_state
        previous_watchdog_stop = active_watchdog_stop
        previous_watchdog_thread = active_watchdog_thread
        validate_slam_2d_only_scope(row, args)
        active_expected_count = expected_scan_count(row)
        active_row = row
        safe_bag = pipeline.sanitize_name(str(row["bag"]))
        bag_out = out_dir / "bags" / safe_bag
        active_bag_out = bag_out
        active_imu_audit = saved_expected_imu_audit(bag_out)
        if active_imu_audit is None:
            active_imu_audit = recorded_imu_audit(pipeline, row)
        active_domain_id = int(getattr(args, "ros_domain_base", 0)) + idx
        label = pipeline.sanitize_name(str(args.snapshot_label))
        active_launch_log = out_dir / "logs" / label / f"{safe_bag}_launch.log"
        active_failure_state = {}
        active_watchdog_stop = threading.Event()
        active_watchdog_thread = None
        try:
            result = original_run_one(row, args, out_dir, idx, total)
            failure_detail = active_failure_state.get("detail", "")
            if not failure_detail:
                failure_detail = playback_failure(_read_log(active_launch_log))
            stationary_complete = (
                result.get("pose_graph_coverage_status")
                == "stationary_single_node_complete"
                and int(result.get("map_count", 0) or 0) > 0
                and int(result.get("pose_graph_trajectory_count", 0) or 0) == 1
            )
            if failure_detail:
                result["status"] = "bag_playback_process_failed"
                result["queue_drain_complete"] = 0
                result["queue_drain_status"] = "bag_playback_process_failed"
                result["queue_drain_evidence"] = "launch_log"
                result["subscription_drain_complete"] = 0
                result["subscription_playback_failure"] = failure_detail
                interrupt_error = active_failure_state.get("interrupt_error", "")
                if interrupt_error:
                    result["subscription_playback_interrupt_error"] = (
                        interrupt_error
                    )
            elif stationary_complete:
                result["pose_graph_captured"] = 1
                result["stationary_replay"] = 1
                result["stationary_replay_reason"] = (
                    "single_expected_motion_sample_single_pose_graph_node"
                )
                result["status"] = "map_and_stationary_pose_graph"
            elif (
                bool(getattr(args, "use_roughness_innovation_slam_toolbox", False))
                and int(result.get("roughness_imu_delivery_complete", 0) or 0) != 1
            ):
                result["status"] = "incomplete_recorded_imu_delivery"
            if (
                failure_detail
                or stationary_complete
                or result.get("status") == "incomplete_recorded_imu_delivery"
            ):
                bag_out = Path(str(result.get("bag_output_dir", "")))
                if bag_out.is_dir():
                    (bag_out / "replay_result.json").write_text(
                        json.dumps(result, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    pipeline.stage30_write_replay_integrity(bag_out, result)
            return result
        finally:
            if active_watchdog_stop is not None:
                active_watchdog_stop.set()
            if active_watchdog_thread is not None:
                active_watchdog_thread.join(timeout=5.0)
            active_expected_count = previous
            active_imu_audit = previous_imu_audit
            active_row = previous_row
            active_bag_out = previous_bag_out
            active_domain_id = previous_domain_id
            active_launch_log = previous_launch_log
            active_failure_state = previous_failure_state
            active_watchdog_stop = previous_watchdog_stop
            active_watchdog_thread = previous_watchdog_thread

    pipeline.stage30_queue_log_fields_for_args = worker_aware_queue_log_fields
    pipeline.stage30_wait_for_slam_queue_drain = guarded_wait
    pipeline.stage30_capture_final_pose_graph = (
        capture_final_pose_graph_with_timeout_recovery
    )

    def guarded_bag_storage_fingerprint(
        row: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if active_bag_out is not None:
            saved = saved_bag_storage_fingerprint(row, active_bag_out)
            if saved is not None:
                return saved
        return original_bag_storage_fingerprint(row)

    pipeline.stage30_bag_storage_fingerprint = guarded_bag_storage_fingerprint
    pipeline.stage30_roughness_innovation_log_fields = enhanced_log_fields
    pipeline.stage30_capture_slam_parameters = watched_capture_slam_parameters
    pipeline.stage30_replay_provenance_payload = (
        lambda row, args, launch_command, capture_command: (
            augment_replay_provenance_payload(
                pipeline,
                original_replay_provenance_payload,
                row,
                args,
                launch_command,
                capture_command,
            )
        )
    )
    pipeline.stage30_pose_graph_coverage_fields = (
        lambda bag_out, args: stationary_aware_pose_graph_coverage(
            pipeline,
            original_pose_graph_coverage,
            bag_out,
            args,
        )
    )
    pipeline.stage30_expected_recorded_scan_audit = (
        lambda row, startup_drop_stamps_s: tolerant_expected_scan_audit(
            pipeline, row, startup_drop_stamps_s
        )
    )

    def enhanced_existing_replay_rerun_reason(
        bag_out: Path,
        summary: dict[str, Any],
        saved_replay: dict[str, Any],
        saved_log_fields: dict[str, Any],
        saved_coverage: dict[str, Any],
        replay_signature: str,
        require_queue_drain: bool,
    ) -> str:
        nonlocal active_imu_audit
        commands_path = bag_out / "commands.txt"
        try:
            current_commands = commands_path.read_bytes()
        except OSError:
            current_commands = None
        reason = compatible_existing_replay_rerun_reason(
            pipeline,
            original_existing_replay_reason,
            bag_out,
            summary,
            saved_replay,
            saved_log_fields,
            saved_coverage,
            replay_signature,
            require_queue_drain,
        )
        if reason:
            restore_current_commands_after_reuse_rejection(
                commands_path,
                current_commands,
            )
        decision = reason if reason else "reuse_existing"
        print(
            f"[lossless-skip] bag={bag_out.name} decision={decision}",
            flush=True,
        )
        if reason and os.environ.get(
            "GOLE_SLAM_SKIP_DIAGNOSTIC_ONLY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                f"diagnostic-only existing replay rejection: "
                f"bag={bag_out.name} reason={reason}"
            )
        if reason and active_row is not None:
            active_imu_audit = recorded_imu_audit(pipeline, active_row)
        return reason

    pipeline.stage30_existing_replay_rerun_reason = (
        enhanced_existing_replay_rerun_reason
    )
    pipeline.stage30_run_one = guarded_run_one
    pipeline.stage34_comparison_status = (
        conditioning_aware_stage34_comparison_status
    )
    pipeline.stage34_roughness_gate_replay = conditioning_aware_stage34_replay


def main() -> None:
    pipeline_path = Path(
        os.environ.get("GOLE_SLAM_PIPELINE", str(DEFAULT_PIPELINE))
    ).expanduser().resolve()
    pipeline = load_pipeline(pipeline_path)
    install_recorded_scan_drain_guard(pipeline)
    pipeline.main()


if __name__ == "__main__":
    main()
