#!/usr/bin/env python3
"""Summarize adaptive roughness replay evidence and exact Karto candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import yaml

plt.switch_backend("Agg")


SUMMARY_FIELDS = (
    "label",
    "bag",
    "mode",
    "reference_label",
    "quality_calibrated",
    "status",
    "log_integrity_valid",
    "instrumentation_valid",
    "lossless_adapter_sha256",
    "runtime_slam_node_sha256",
    "runtime_cmake_cache_sha256",
    "replay_pose_graph_exporter_sha256",
    "slam_config_sha256",
    "runtime_parameter_match",
    "imu_delivery_complete",
    "scan_delivery_complete",
    "queue_drain_complete",
    "pose_graph_coverage_status",
    "pose_graph_node_count",
    "pose_graph_edge_count",
    "exact_edge_evidence_valid",
    "exact_edge_evidence_status",
    "exact_edge_exporter_sha256",
    "pose_graph_expected_motion_count",
    "scan_effective_expected_count",
    "scan_received_count",
    "exact_candidate_count",
    "candidate_log_valid",
    "candidate_count_matches_replay",
    "memory_log_valid",
    "memory_count_matches_replay",
    "observer_failure_sample_count",
    "edge_accounting_failure_sample_count",
    "unsafe_intervention_sample_count",
    "eligible_candidate_count",
    "ineligible_candidate_count",
    "accepted_candidate_count",
    "downweighted_candidate_count",
    "rejected_candidate_count",
    "adaptive_response_rejected_candidate_count",
    "baseline_response_rejected_candidate_count",
    "backend_edge_inserted_candidate_count",
    "backend_duplicate_edge_candidate_count",
    "backend_distance_rejected_candidate_count",
    "fail_open_candidate_count",
    "raw_fail_open_candidate_count",
    "backend_response_rejected_count",
    "parity_failure_count",
    "observer_parity_ratio",
    "edge_inserted_count",
    "edge_missing_count",
    "memory_sample_count",
    "memory_strength_mean",
    "memory_strength_median",
    "memory_strength_p95",
    "effective_duration_mean_s",
    "effective_duration_p95_s",
    "effective_distance_mean_m",
    "effective_distance_p95_m",
    "response_mean",
    "response_median",
    "response_p95",
    "reference_chain_event_count",
    "reference_chain_event_available_count",
    "reference_chain_event_fail_open_count",
    "reference_chain_hypothesis_count",
    "reference_chain_hypothesis_available_count",
    "reference_chain_hypothesis_accepted_count",
    "reference_chain_hypothesis_fail_open_count",
    "reference_chain_decision_count",
    "reference_chain_decision_available_count",
    "reference_chain_decision_accepted_count",
    "reference_chain_decision_correlated_duplicate_count",
    "reference_chain_decision_fail_open_count",
    "reference_chain_evaluation_duration_ms_median",
    "reference_chain_evaluation_duration_ms_p95",
    "reference_chain_evaluation_duration_ms_max",
    "reference_chain_minimum_fine_response_median",
    "reference_chain_minimum_fine_response_p95",
    "reference_chain_quality_risk_median",
    "reference_chain_quality_risk_p95",
    "reference_chain_forward_pair_translation_error_m_p95",
    "reference_chain_forward_pair_yaw_error_rad_p95",
    "reference_chain_reverse_pair_translation_error_m_p95",
    "reference_chain_reverse_pair_yaw_error_rad_p95",
    "place_constraint_injected_count",
    "place_constraint_edge_injected_count",
    "place_graph_action_count",
    "final_factor_residual_count",
    "final_factor_residual_approximate_count",
    "final_factor_residual_mean",
    "final_factor_residual_p95",
    "run_wall_s",
    "queue_drain_wait_s",
    "replay_completion_wall_s",
    "comparison_status",
    "fairness_status",
    "reference_node_count",
    "node_count_delta",
    "map_occupied_boundary_f1",
    "map_occupied_changed_cell_ratio",
    "variant_vs_reference_xy_error_mean_m",
    "variant_vs_reference_xy_error_p95_m",
    "odom_consistent_p95_delta_reference_minus_variant_m",
    "return_revisit_p95_delta_reference_minus_variant_m",
    "map_png",
    "map_with_trajectory_png",
    "map_with_pose_graph_trajectory_png",
    "map_comparison_png",
    "trajectory_comparison_png",
    "candidate_csv",
    "memory_csv",
    "reference_chain_events_csv",
    "reference_chain_hypotheses_csv",
    "reference_chain_decisions_csv",
    "reference_chain_anchors_csv",
    "pose_graph_edge_csv",
    "pose_graph_edge_provenance_json",
    "replay_result_json",
)

ACCEPTED_FAIRNESS_STATUSES = {
    "matched",
    "matched_with_explicit_experiment_difference",
}

COMPLETE_POSE_GRAPH_COVERAGE_STATUSES = {
    "complete",
    "stationary_single_node_complete",
}

REQUIRED_CANDIDATE_FIELDS = {
    "stamp_ns",
    "candidate_index",
    "source_unique_id",
    "target_unique_id",
    "normalized_response",
    "relative_x_m",
    "relative_y_m",
    "relative_yaw_rad",
    "outcome_parity",
    "backend_action",
}

REQUIRED_MEMORY_FIELDS = {
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
}

REQUIRED_EDGE_FIELDS = {
    "edge_index",
    "source_unique_id",
    "target_unique_id",
    "relative_x_m",
    "relative_y_m",
    "relative_yaw_rad",
    "covariance_xx",
    "covariance_yy",
    "covariance_yawyaw",
    "final_residual_x_m",
    "final_residual_y_m",
    "final_residual_yaw_rad",
    "final_covariance_weighted_residual_sq",
}

QUANTILE_METRICS = (
    "memory_strength",
    "normalized_response",
    "translation_disagreement_m",
    "yaw_disagreement_rad",
    "xy_covariance_max_eigenvalue_m2",
    "scan_match_covariance_weighted_disagreement_sq",
    "post_covariance_weighted_residual_sq",
    "final_covariance_weighted_residual_sq",
    "approximate_final_covariance_weighted_residual_sq",
)


REFERENCE_CHAIN_SUMMARY_FIELDS = {
    "reference_chain_event_count": (
        "roughness_place_reference_chain_event_count"
    ),
    "reference_chain_event_available_count": (
        "roughness_place_reference_chain_event_available_count"
    ),
    "reference_chain_event_fail_open_count": (
        "roughness_place_reference_chain_event_fail_open_count"
    ),
    "reference_chain_hypothesis_count": (
        "roughness_place_reference_chain_hypothesis_count"
    ),
    "reference_chain_hypothesis_available_count": (
        "roughness_place_reference_chain_hypothesis_available_count"
    ),
    "reference_chain_hypothesis_accepted_count": (
        "roughness_place_reference_chain_hypothesis_accepted_count"
    ),
    "reference_chain_hypothesis_fail_open_count": (
        "roughness_place_reference_chain_hypothesis_fail_open_count"
    ),
    "reference_chain_decision_count": (
        "roughness_place_reference_chain_decision_count"
    ),
    "reference_chain_decision_available_count": (
        "roughness_place_reference_chain_decision_available_count"
    ),
    "reference_chain_decision_accepted_count": (
        "roughness_place_reference_chain_decision_accepted_count"
    ),
    "reference_chain_decision_correlated_duplicate_count": (
        "roughness_place_reference_chain_decision_correlated_duplicate_count"
    ),
    "reference_chain_decision_fail_open_count": (
        "roughness_place_reference_chain_decision_fail_open_count"
    ),
    "reference_chain_evaluation_duration_ms_median": (
        "roughness_place_reference_chain_evaluation_duration_ms_median"
    ),
    "reference_chain_evaluation_duration_ms_p95": (
        "roughness_place_reference_chain_evaluation_duration_ms_p95"
    ),
    "reference_chain_evaluation_duration_ms_max": (
        "roughness_place_reference_chain_evaluation_duration_ms_max"
    ),
    "reference_chain_minimum_fine_response_median": (
        "roughness_place_reference_chain_minimum_fine_response_median"
    ),
    "reference_chain_minimum_fine_response_p95": (
        "roughness_place_reference_chain_minimum_fine_response_p95"
    ),
    "reference_chain_quality_risk_median": (
        "roughness_place_reference_chain_quality_risk_median"
    ),
    "reference_chain_quality_risk_p95": (
        "roughness_place_reference_chain_quality_risk_p95"
    ),
    "reference_chain_forward_pair_translation_error_m_p95": (
        "roughness_place_reference_chain_forward_pair_translation_error_m_p95"
    ),
    "reference_chain_forward_pair_yaw_error_rad_p95": (
        "roughness_place_reference_chain_forward_pair_yaw_error_rad_p95"
    ),
    "reference_chain_reverse_pair_translation_error_m_p95": (
        "roughness_place_reference_chain_reverse_pair_translation_error_m_p95"
    ),
    "reference_chain_reverse_pair_yaw_error_rad_p95": (
        "roughness_place_reference_chain_reverse_pair_yaw_error_rad_p95"
    ),
    "place_constraint_injected_count": (
        "roughness_place_constraint_injected_count"
    ),
    "place_constraint_edge_injected_count": (
        "roughness_place_constraint_edge_injected_count"
    ),
    "place_graph_action_count": "roughness_place_graph_action_count",
    "reference_chain_events_csv": (
        "roughness_place_reference_chain_events_csv"
    ),
    "reference_chain_hypotheses_csv": (
        "roughness_place_reference_chain_hypotheses_csv"
    ),
    "reference_chain_decisions_csv": (
        "roughness_place_reference_chain_decisions_csv"
    ),
    "reference_chain_anchors_csv": (
        "roughness_place_reference_chain_anchors_csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help=(
            "Write CSV, plots, and the report below this directory while "
            "continuing to read replay inputs from --output-root. Defaults "
            "to --output-root for backward compatibility."
        ),
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=[],
        help=(
            "Only aggregate this replay label. Repeat for multiple labels; "
            "omitting the option preserves the all-run diagnostic view."
        ),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--calibration-json",
        type=Path,
        help=(
            "Calibration JSON used by the selected adaptive replay. When "
            "omitted, read <artifact-root>/calibration/"
            "adaptive_quality_calibration.json."
        ),
    )
    parser.add_argument(
        "--compatible-lossless-adapter-sha256",
        action="append",
        default=[],
        help=(
            "Explicitly allow a replay produced by this behavior-compatible "
            "lossless adapter hash. The current local adapter is included "
            "automatically; repeat for multiple audited versions."
        ),
    )
    parser.add_argument(
        "--max-odom-p95-regression-m",
        type=float,
        default=0.02,
        help="Maximum allowed A03/A27 odometry-consistency p95 regression.",
    )
    parser.add_argument(
        "--max-return-p95-regression-m",
        type=float,
        default=0.05,
        help="Maximum allowed A03/A27 return-revisit p95 regression.",
    )
    return parser.parse_args()


def select_replay_paths(
    replay_paths: Iterable[Path],
    include_labels: Iterable[str],
) -> list[Path]:
    paths = sorted(replay_paths)
    requested = {label.strip() for label in include_labels if label.strip()}
    if not requested:
        return paths

    selected = [path for path in paths if path.parents[2].name in requested]
    found = {path.parents[2].name for path in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(
            "requested replay labels were not found: " + ", ".join(missing)
        )
    return selected


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def csv_header(path: Path) -> set[str]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return set(next(csv.reader(stream), []))
    except (OSError, StopIteration):
        return set()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rotated_map_corners(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    origin_yaw: float,
) -> np.ndarray:
    local = np.asarray(
        (
            (0.0, 0.0),
            (width * resolution, 0.0),
            (0.0, height * resolution),
            (width * resolution, height * resolution),
        ),
        dtype=float,
    )
    cosine = math.cos(origin_yaw)
    sine = math.sin(origin_yaw)
    return np.column_stack(
        (
            origin_x + cosine * local[:, 0] - sine * local[:, 1],
            origin_y + sine * local[:, 0] + cosine * local[:, 1],
        )
    )


def plot_pose_graph_map_overlay(
    bag_dir: Path,
    output: Path | None = None,
) -> str:
    """Plot final graph and Karto odometric poses in the occupancy-map gauge."""
    map_yaml_path = bag_dir / "map.yaml"
    trajectory_path = bag_dir / "trajectory_pose_graph_final.csv"
    if not map_yaml_path.is_file() or not trajectory_path.is_file():
        return ""
    try:
        metadata = yaml.safe_load(map_yaml_path.read_text(encoding="utf-8"))
        image_path = map_yaml_path.parent / str(metadata["image"])
        image = plt.imread(image_path)
        image = np.asarray(image, dtype=float)
        if image.size and float(np.nanmax(image)) > 1.0:
            image /= 255.0
        resolution = float(metadata["resolution"])
        origin_x, origin_y, origin_yaw = map(float, metadata["origin"][:3])
        rows = read_csv(trajectory_path)
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError):
        return ""
    if image.ndim not in (2, 3) or resolution <= 0.0:
        return ""

    optimized = np.asarray(
        [
            (to_float(row.get("x_m")), to_float(row.get("y_m")))
            for row in rows
        ],
        dtype=float,
    )
    odometric = np.asarray(
        [
            (to_float(row.get("odom_x_m")), to_float(row.get("odom_y_m")))
            for row in rows
        ],
        dtype=float,
    )
    if optimized.size == 0:
        return ""
    optimized = optimized[np.all(np.isfinite(optimized), axis=1)]
    odometric = odometric[np.all(np.isfinite(odometric), axis=1)]
    if optimized.size == 0:
        return ""

    height, width = image.shape[:2]
    corners = _rotated_map_corners(
        width, height, resolution, origin_x, origin_y, origin_yaw
    )
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    transform = (
        Affine2D().rotate(origin_yaw).translate(origin_x, origin_y) + ax.transData
    )
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        extent=(0.0, width * resolution, 0.0, height * resolution),
        transform=transform,
        vmin=0.0,
        vmax=1.0,
    )

    if odometric.size:
        stride = max(1, int(math.ceil(len(odometric) / 900.0)))
        display = odometric[::stride]
        ax.scatter(
            display[:, 0],
            display[:, 1],
            s=7,
            color="#4c72b0",
            alpha=0.55,
            label="Karto odometric pose",
        )
    stride = max(1, int(math.ceil(len(optimized) / 900.0)))
    display = optimized[::stride]
    ax.scatter(
        display[:, 0],
        display[:, 1],
        s=8,
        color="#dd8452",
        alpha=0.82,
        label="optimized pose graph",
    )
    ax.scatter(
        optimized[:1, 0],
        optimized[:1, 1],
        s=34,
        color="#55a868",
        label="start",
    )
    ax.scatter(
        optimized[-1:, 0],
        optimized[-1:, 1],
        s=34,
        color="#c44e52",
        label="end",
    )
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    span = max(*(maximum - minimum), 1.0)
    margin = max(0.25, 0.02 * span)
    ax.set_xlim(minimum[0] - margin, maximum[0] + margin)
    ax.set_ylim(minimum[1] - margin, maximum[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("map x [m]")
    ax.set_ylabel("map y [m]")
    ax.set_title("Occupancy map with point-only final pose graph")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output = output or (bag_dir / "map_with_pose_graph_trajectory.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def to_int(value: Any) -> int:
    number = to_float(value)
    return int(number) if math.isfinite(number) else 0


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def reference_chain_summary(replay: dict[str, Any]) -> dict[str, Any]:
    """Project exact stable-side chain diagnostics into the dataset summary."""
    result: dict[str, Any] = {}
    for summary_field, replay_field in REFERENCE_CHAIN_SUMMARY_FIELDS.items():
        value = replay.get(replay_field)
        if summary_field.endswith("_count"):
            result[summary_field] = to_int(value)
        elif summary_field.endswith("_csv"):
            result[summary_field] = str(value or "")
        else:
            result[summary_field] = to_float(value)
    return result


def finite(values: Iterable[Any]) -> list[float]:
    result = [to_float(value) for value in values]
    return [value for value in result if math.isfinite(value)]


def percentile(values: Iterable[Any], quantile: float) -> float:
    samples = sorted(finite(values))
    if not samples:
        return math.nan
    if len(samples) == 1:
        return samples[0]
    position = min(max(quantile, 0.0), 1.0) * (len(samples) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return samples[lower]
    weight = position - lower
    return samples[lower] * (1.0 - weight) + samples[upper] * weight


def mean(values: Iterable[Any]) -> float:
    samples = finite(values)
    return statistics.fmean(samples) if samples else math.nan


def normalized_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def relative_pose(
    source: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    sx, sy, syaw = source
    tx, ty, tyaw = target
    dx = tx - sx
    dy = ty - sy
    cosine = math.cos(syaw)
    sine = math.sin(syaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        normalized_angle(tyaw - syaw),
    )


def final_pose_by_unique_id(path: Path) -> dict[int, tuple[float, float, float]]:
    poses: dict[int, tuple[float, float, float]] = {}
    for row in read_csv(path):
        unique_id = to_int(row.get("unique_id"))
        pose = (
            to_float(row.get("x_m")),
            to_float(row.get("y_m")),
            to_float(row.get("yaw_rad")),
        )
        if all(math.isfinite(value) for value in pose):
            poses[unique_id] = pose
    return poses


def final_edges_by_unique_ids(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[int, int], list[dict[str, str]]]:
    edges: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in rows:
        source_value = to_float(row.get("source_unique_id"))
        target_value = to_float(row.get("target_unique_id"))
        if not math.isfinite(source_value) or not math.isfinite(target_value):
            continue
        source_unique_id = int(source_value)
        target_unique_id = int(target_value)
        edges.setdefault((source_unique_id, target_unique_id), []).append(row)
    return edges


def exact_edge_evidence(
    bag_dir: Path,
    edge_rows: list[dict[str, str]],
) -> dict[str, Any]:
    edge_path = bag_dir / "pose_graph_edges_final.csv"
    provenance_path = bag_dir / "pose_graph_edges_final.provenance.json"
    pose_graph_path = bag_dir / "final_pose_graph.posegraph"
    trajectory_path = bag_dir / "trajectory_pose_graph_final.csv"
    provenance = read_json(provenance_path)
    result = {
        "valid": False,
        "status": "missing_edge_csv",
        "exporter_sha256": str(provenance.get("exporter_sha256", "")),
        "provenance_path": str(provenance_path) if provenance_path.is_file() else "",
    }
    if not edge_path.is_file():
        return result
    if not provenance_path.is_file():
        result["status"] = "missing_edge_provenance"
        return result
    checks = (
        provenance.get("schema_version") == 2,
        provenance.get("status") == "exported",
        to_int(provenance.get("trajectory_equivalent")) == 1,
        to_int(provenance.get("edge_count")) == len(edge_rows),
        provenance.get("edge_csv_sha256") == file_sha256(edge_path),
        provenance.get("pose_graph_sha256") == file_sha256(pose_graph_path),
        provenance.get("trajectory_sha256") == file_sha256(trajectory_path),
        len(str(provenance.get("exporter_sha256", ""))) == 64,
        REQUIRED_EDGE_FIELDS <= csv_header(edge_path),
    )
    if not all(checks):
        result["status"] = "edge_provenance_mismatch"
        return result
    result["valid"] = True
    result["status"] = "valid"
    return result


def _edge_measurement_error(
    candidate: tuple[float, float, float],
    edge: dict[str, str],
) -> float:
    edge_measurement = (
        to_float(edge.get("relative_x_m")),
        to_float(edge.get("relative_y_m")),
        to_float(edge.get("relative_yaw_rad")),
    )
    if not all(math.isfinite(value) for value in edge_measurement):
        return math.inf
    return math.hypot(
        edge_measurement[0] - candidate[0],
        edge_measurement[1] - candidate[1],
    ) + abs(normalized_angle(edge_measurement[2] - candidate[2]))


def add_final_factor_residual(
    row: dict[str, Any],
    poses: dict[int, tuple[float, float, float]],
    edges: dict[tuple[int, int], list[dict[str, str]]] | None = None,
) -> None:
    row.update(
        {
            "final_residual_x_m": math.nan,
            "final_residual_y_m": math.nan,
            "final_residual_yaw_rad": math.nan,
            "final_residual_translation_m": math.nan,
            "final_covariance_weighted_residual_sq": math.nan,
            "approximate_final_covariance_weighted_residual_sq": math.nan,
            "final_residual_status": "not_inserted",
            "final_residual_source": "none",
            "final_pose_graph_edge_index": "",
        }
    )
    if not is_true(row.get("edge_inserted")):
        return
    measured = (
        to_float(row.get("relative_x_m")),
        to_float(row.get("relative_y_m")),
        to_float(row.get("relative_yaw_rad")),
    )
    if not all(math.isfinite(value) for value in measured):
        row["final_residual_status"] = "invalid_measurement"
        return

    if edges is not None:
        key = (
            to_int(row.get("source_unique_id")),
            to_int(row.get("target_unique_id")),
        )
        matching_edges = edges.get(key, [])
        if not matching_edges:
            row["final_residual_status"] = "missing_final_edge"
            return
        edge = min(
            matching_edges,
            key=lambda value: _edge_measurement_error(measured, value),
        )
        measurement_error = _edge_measurement_error(measured, edge)
        if not math.isfinite(measurement_error) or measurement_error > 1.0e-5:
            row["final_residual_status"] = "edge_measurement_mismatch"
            return
        residual = (
            to_float(edge.get("final_residual_x_m")),
            to_float(edge.get("final_residual_y_m")),
            to_float(edge.get("final_residual_yaw_rad")),
        )
        weighted = to_float(edge.get("final_covariance_weighted_residual_sq"))
        if not all(math.isfinite(value) for value in residual) or not math.isfinite(
            weighted
        ):
            row["final_residual_status"] = "invalid_final_edge_residual"
            return
        row["final_residual_x_m"] = residual[0]
        row["final_residual_y_m"] = residual[1]
        row["final_residual_yaw_rad"] = residual[2]
        row["final_residual_translation_m"] = math.hypot(
            residual[0], residual[1]
        )
        row["final_covariance_weighted_residual_sq"] = weighted
        row["final_residual_status"] = "ok"
        row["final_residual_source"] = "serialized_pose_graph_edge"
        row["final_pose_graph_edge_index"] = edge.get("edge_index", "")
        return

    source = poses.get(to_int(row.get("source_unique_id")))
    target = poses.get(to_int(row.get("target_unique_id")))
    if source is None or target is None:
        row["final_residual_status"] = "missing_final_pose"
        return
    expected = relative_pose(source, target)
    residual = np.asarray(
        [
            expected[0] - measured[0],
            expected[1] - measured[1],
            normalized_angle(expected[2] - measured[2]),
        ],
        dtype=float,
    )
    covariance = np.asarray(
        [
            [
                to_float(row.get("covariance_xx")),
                to_float(row.get("covariance_xy")),
                to_float(row.get("covariance_xyaw")),
            ],
            [
                to_float(row.get("covariance_xy")),
                to_float(row.get("covariance_yy")),
                to_float(row.get("covariance_yyaw")),
            ],
            [
                to_float(row.get("covariance_xyaw")),
                to_float(row.get("covariance_yyaw")),
                to_float(row.get("covariance_yawyaw")),
            ],
        ],
        dtype=float,
    )
    row["final_residual_x_m"] = residual[0]
    row["final_residual_y_m"] = residual[1]
    row["final_residual_yaw_rad"] = residual[2]
    row["final_residual_translation_m"] = math.hypot(residual[0], residual[1])
    if not np.all(np.isfinite(covariance)):
        row["final_residual_status"] = "invalid_covariance"
        return
    try:
        information = np.linalg.pinv(covariance, hermitian=True)
        weighted = float(residual.T @ information @ residual)
    except np.linalg.LinAlgError:
        row["final_residual_status"] = "covariance_inverse_failed"
        return
    if not math.isfinite(weighted):
        row["final_residual_status"] = "invalid_weighted_residual"
        return
    row["approximate_final_covariance_weighted_residual_sq"] = weighted
    row["final_residual_status"] = "approximate_covariance_frame"
    row["final_residual_source"] = "reconstructed_pose_raw_scan_covariance"


def configured_mode(replay: dict[str, Any]) -> str:
    return str(configured_adaptive_parameters(replay).get(
        "roughness_constraint_mode", "unknown"
    ))


def configured_adaptive_parameters(replay: dict[str, Any]) -> dict[str, Any]:
    raw = replay.get("runtime_slam_adaptive_constraint_parameter_actual_json", "")
    try:
        values = json.loads(str(raw))
    except json.JSONDecodeError:
        values = {}
    if isinstance(values, dict):
        return values
    return {}


def comparison_row(stage: Path, label: str, bag: str) -> dict[str, str]:
    path = stage / f"roughness_gate_replay_{label}.csv"
    for row in read_csv(path):
        if row.get("bag") == bag:
            return row
    return {}


def comparison_reference_label(comparison: dict[str, str]) -> str:
    reference_dir = Path(str(comparison.get("stage31_dir", "")).strip())
    if not str(reference_dir) or reference_dir.parent.name != "bags":
        return ""
    return reference_dir.parent.parent.name


def diagnostic_csv_path(
    replay: dict[str, Any],
    replay_path: Path,
    field: str,
    suffix: str,
) -> Path:
    configured = Path(str(replay.get(field, ""))).expanduser()
    if configured.is_file():
        return configured.resolve()
    label = replay_path.parents[2].name
    bag = replay_path.parent.name
    log_dir = replay_path.parents[2] / "logs" / f"{label}_variant"
    unique = log_dir / f"{bag}_{suffix}"
    if unique.is_file():
        return unique.resolve()
    legacy = log_dir / suffix
    return legacy.resolve()


def action_counts(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field, "unknown")) for row in rows)


def candidate_action_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    eligible = [row for row in rows if is_true(row.get("chain_size_accepted"))]
    shadow_counts = action_counts(eligible, "shadow_action")
    backend_counts = action_counts(rows, "backend_action")
    return {
        "eligible_candidate_count": len(eligible),
        "ineligible_candidate_count": len(rows) - len(eligible),
        "accepted_candidate_count": shadow_counts.get("accepted", 0),
        "downweighted_candidate_count": shadow_counts.get("downweighted", 0),
        "rejected_candidate_count": shadow_counts.get("rejected", 0),
        "adaptive_response_rejected_candidate_count": sum(
            is_true(row.get("baseline_response_accepted"))
            and not is_true(row.get("applied_response_accepted"))
            for row in eligible
        ),
        "baseline_response_rejected_candidate_count": sum(
            not is_true(row.get("baseline_response_accepted"))
            for row in eligible
        ),
        "backend_edge_inserted_candidate_count": backend_counts.get(
            "pose_fused_edge_inserted", 0
        ),
        "backend_duplicate_edge_candidate_count": backend_counts.get(
            "pose_fused_duplicate_edge", 0
        ),
        "backend_distance_rejected_candidate_count": backend_counts.get(
            "pose_fused_distance_rejected", 0
        ),
        "fail_open_candidate_count": sum(
            is_true(row.get("fail_open")) for row in eligible
        ),
        "raw_fail_open_candidate_count": sum(
            is_true(row.get("fail_open")) for row in rows
        ),
        "backend_response_rejected_count": backend_counts.get(
            "rejected_response", 0
        ),
    }


def summarize_replay(
    stage: Path,
    replay_path: Path,
    pose_graph_overlay_output: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    replay = read_json(replay_path)
    label = replay_path.parents[2].name
    bag = replay_path.parent.name
    candidate_path = diagnostic_csv_path(
        replay,
        replay_path,
        "roughness_exact_near_chain_candidate_events_csv",
        "near_chain_candidate_events.csv",
    )
    memory_path = diagnostic_csv_path(
        replay,
        replay_path,
        "roughness_adaptive_memory_events_csv",
        "adaptive_memory_events.csv",
    )
    candidate_rows: list[dict[str, Any]] = read_csv(candidate_path)
    memory_rows: list[dict[str, Any]] = read_csv(memory_path)
    expected_candidate_count = to_int(
        replay.get("roughness_exact_near_chain_candidate_count")
    )
    expected_memory_count = to_int(
        replay.get("roughness_adaptive_memory_event_count")
    )
    candidate_log_valid = (
        candidate_path.is_file()
        and REQUIRED_CANDIDATE_FIELDS <= csv_header(candidate_path)
    )
    memory_log_valid = (
        memory_path.is_file()
        and REQUIRED_MEMORY_FIELDS <= csv_header(memory_path)
    )
    candidate_count_matches_replay = (
        candidate_log_valid and len(candidate_rows) == expected_candidate_count
    )
    memory_count_matches_replay = (
        memory_log_valid and len(memory_rows) == expected_memory_count
    )
    pose_graph_edge_path = replay_path.parent / "pose_graph_edges_final.csv"
    pose_graph_edge_rows = (
        read_csv(pose_graph_edge_path) if pose_graph_edge_path.is_file() else []
    )
    edge_evidence = exact_edge_evidence(
        replay_path.parent,
        pose_graph_edge_rows,
    )
    pose_graph_edges = (
        final_edges_by_unique_ids(pose_graph_edge_rows)
        if edge_evidence["valid"]
        else None
    )
    poses = final_pose_by_unique_id(
        replay_path.parent / "trajectory_pose_graph_final.csv"
    )
    for row in candidate_rows:
        row["label"] = label
        row["bag"] = bag
        row["mode"] = configured_mode(replay)
        row["final_edge_evidence_valid"] = int(edge_evidence["valid"])
        row["final_edge_exporter_sha256"] = edge_evidence["exporter_sha256"]
        add_final_factor_residual(row, poses, pose_graph_edges)
    for row in memory_rows:
        row["label"] = label
        row["bag"] = bag
        row["mode"] = configured_mode(replay)

    backend_counts = action_counts(candidate_rows, "backend_action")
    candidate_actions = candidate_action_summary(candidate_rows)
    parity_failures = sum(
        not is_true(row.get("outcome_parity")) for row in candidate_rows
    )
    edge_missing = backend_counts.get("pose_fused_edge_missing", 0)
    processed_observer_rows = [
        row for row in memory_rows if is_true(row.get("scan_processed"))
    ]
    observer_failure_sample_count = sum(
        not is_true(row.get("observer_available"))
        and index != 0
        for index, row in enumerate(processed_observer_rows)
    )
    edge_accounting_failure_sample_count = sum(
        is_true(row.get("observer_available"))
        and not is_true(row.get("new_edge_accounting_complete"))
        for row in processed_observer_rows
    )
    unsafe_intervention_sample_count = sum(
        is_true(row.get("intervention_allowed"))
        and to_float(row.get("applied_threshold"))
        > to_float(row.get("baseline_threshold")) + 1.0e-9
        and not is_true(row.get("outcome_parity"))
        for row in processed_observer_rows
    )
    log_integrity_valid = (
        candidate_count_matches_replay
        and memory_count_matches_replay
        and observer_failure_sample_count == 0
    )
    instrumentation_valid = (
        log_integrity_valid
        and parity_failures == 0
        and edge_missing == 0
        and edge_accounting_failure_sample_count == 0
    )
    for row in candidate_rows:
        row["instrumentation_valid"] = int(
            log_integrity_valid
            and is_true(row.get("outcome_parity"))
            and row.get("backend_action") != "pose_fused_edge_missing"
        )

    comparison = comparison_row(stage, label, bag)
    reference_label = comparison_reference_label(comparison)
    pose_graph_overlay = plot_pose_graph_map_overlay(
        replay_path.parent,
        pose_graph_overlay_output,
    )
    final_residuals = finite(
        row.get("final_covariance_weighted_residual_sq")
        for row in candidate_rows
        if row.get("final_residual_status") == "ok"
    )
    approximate_final_residual_count = sum(
        row.get("final_residual_status") == "approximate_covariance_frame"
        for row in candidate_rows
    )
    mode = configured_mode(replay)
    adaptive_parameters = configured_adaptive_parameters(replay)
    replay_provenance = read_json(
        replay_path.parent / "replay_provenance.json"
    )
    adapter_provenance = replay_provenance.get("lossless_stage34_adapter", {})
    if not isinstance(adapter_provenance, dict):
        adapter_provenance = {}
    fairness_contract = replay_provenance.get("fairness_contract", {})
    if not isinstance(fairness_contract, dict):
        fairness_contract = {}
    backend_assets = fairness_contract.get("backend_asset_sha256", {})
    if not isinstance(backend_assets, dict):
        backend_assets = {}
    runtime_build = fairness_contract.get("slam_2d_runtime_build", {})
    if not isinstance(runtime_build, dict):
        runtime_build = {}
    run_wall_s = to_float(replay.get("run_wall_s"))
    queue_drain_wait_s = to_float(replay.get("queue_drain_wait_s"))
    replay_completion_wall_s = (
        run_wall_s + queue_drain_wait_s
        if math.isfinite(run_wall_s) and math.isfinite(queue_drain_wait_s)
        else math.nan
    )
    summary = {
        "label": label,
        "bag": bag,
        "mode": mode,
        "reference_label": reference_label,
        "quality_calibrated": int(is_true(adaptive_parameters.get(
            "roughness_near_chain_quality_calibrated"
        ))),
        "status": replay.get("status", ""),
        "log_integrity_valid": int(log_integrity_valid),
        "instrumentation_valid": int(instrumentation_valid),
        "lossless_adapter_sha256": adapter_provenance.get("sha256", ""),
        "runtime_slam_node_sha256": backend_assets.get(
            "roughness_innovation_node", ""
        ),
        "runtime_cmake_cache_sha256": runtime_build.get(
            "cmake_cache_sha256", ""
        ),
        "replay_pose_graph_exporter_sha256": backend_assets.get(
            "pose_graph_exporter", ""
        ),
        "slam_config_sha256": backend_assets.get("slam_config", ""),
        "runtime_parameter_match": to_int(
            replay.get("runtime_slam_parameter_match")
        ),
        "imu_delivery_complete": to_int(
            replay.get("roughness_imu_delivery_complete")
        ),
        "scan_delivery_complete": to_int(
            replay.get("slam_recorded_scan_delivery_complete")
        ),
        "queue_drain_complete": to_int(replay.get("queue_drain_complete")),
        "pose_graph_coverage_status": replay.get(
            "pose_graph_coverage_status", ""
        ),
        "pose_graph_node_count": to_int(
            replay.get("pose_graph_trajectory_count")
        ),
        "pose_graph_edge_count": len(pose_graph_edge_rows),
        "exact_edge_evidence_valid": int(edge_evidence["valid"]),
        "exact_edge_evidence_status": edge_evidence["status"],
        "exact_edge_exporter_sha256": edge_evidence["exporter_sha256"],
        "pose_graph_expected_motion_count": to_int(
            replay.get("pose_graph_coverage_expected_motion_count")
        ),
        "scan_effective_expected_count": to_int(
            replay.get("slam_recorded_scan_effective_expected_count")
        ),
        "scan_received_count": to_int(
            replay.get("slam_recorded_scan_received_count")
        ),
        "exact_candidate_count": len(candidate_rows),
        "candidate_log_valid": int(candidate_log_valid),
        "candidate_count_matches_replay": int(candidate_count_matches_replay),
        "memory_log_valid": int(memory_log_valid),
        "memory_count_matches_replay": int(memory_count_matches_replay),
        "observer_failure_sample_count": observer_failure_sample_count,
        "edge_accounting_failure_sample_count": (
            edge_accounting_failure_sample_count
        ),
        "unsafe_intervention_sample_count": unsafe_intervention_sample_count,
        **candidate_actions,
        "parity_failure_count": parity_failures,
        "observer_parity_ratio": (
            1.0 - parity_failures / len(candidate_rows)
            if candidate_rows else 1.0
        ),
        "edge_inserted_count": sum(
            is_true(row.get("edge_inserted")) for row in candidate_rows
        ),
        "edge_missing_count": edge_missing,
        "memory_sample_count": len(memory_rows),
        "memory_strength_mean": mean(
            row.get("strength") for row in memory_rows
        ),
        "memory_strength_median": percentile(
            (row.get("strength") for row in memory_rows), 0.5
        ),
        "memory_strength_p95": percentile(
            (row.get("strength") for row in memory_rows), 0.95
        ),
        "effective_duration_mean_s": mean(
            row.get("effective_duration_s") for row in memory_rows
        ),
        "effective_duration_p95_s": percentile(
            (row.get("effective_duration_s") for row in memory_rows), 0.95
        ),
        "effective_distance_mean_m": mean(
            row.get("effective_distance_m") for row in memory_rows
        ),
        "effective_distance_p95_m": percentile(
            (row.get("effective_distance_m") for row in memory_rows), 0.95
        ),
        "response_mean": mean(
            row.get("normalized_response") for row in candidate_rows
        ),
        "response_median": percentile(
            (row.get("normalized_response") for row in candidate_rows), 0.5
        ),
        "response_p95": percentile(
            (row.get("normalized_response") for row in candidate_rows), 0.95
        ),
        **reference_chain_summary(replay),
        "final_factor_residual_count": len(final_residuals),
        "final_factor_residual_approximate_count": (
            approximate_final_residual_count
        ),
        "final_factor_residual_mean": mean(final_residuals),
        "final_factor_residual_p95": percentile(final_residuals, 0.95),
        "run_wall_s": run_wall_s,
        "queue_drain_wait_s": queue_drain_wait_s,
        "replay_completion_wall_s": replay_completion_wall_s,
        "comparison_status": comparison.get("status", ""),
        "fairness_status": comparison.get("replay_fairness_status", ""),
        "reference_node_count": to_int(
            comparison.get("stage31_pose_graph_node_count")
        ),
        "node_count_delta": to_int(comparison.get("pose_graph_node_count_delta")),
        "map_occupied_boundary_f1": to_float(
            comparison.get("map_occupied_boundary_f1")
        ),
        "map_occupied_changed_cell_ratio": to_float(
            comparison.get("map_occupied_changed_cell_ratio")
        ),
        "variant_vs_reference_xy_error_mean_m": to_float(
            comparison.get("variant_vs_stage31_xy_error_mean_m")
        ),
        "variant_vs_reference_xy_error_p95_m": to_float(
            comparison.get("variant_vs_stage31_xy_error_p95_m")
        ),
        "odom_consistent_p95_delta_reference_minus_variant_m": to_float(
            comparison.get(
                "odom_consistent_p95_delta_stage31_minus_variant_m"
            )
        ),
        "return_revisit_p95_delta_reference_minus_variant_m": to_float(
            comparison.get(
                "return_revisit_drift_p95_delta_stage31_minus_variant_m"
            )
        ),
        "map_png": str(replay_path.parent / "map.png"),
        "map_with_trajectory_png": str(
            replay_path.parent / "map_with_trajectory.png"
        ),
        "map_with_pose_graph_trajectory_png": pose_graph_overlay,
        "map_comparison_png": comparison.get("map_comparison_png", ""),
        "trajectory_comparison_png": comparison.get(
            "trajectory_comparison_png", ""
        ),
        "candidate_csv": str(candidate_path),
        "memory_csv": str(memory_path),
        "pose_graph_edge_csv": (
            str(pose_graph_edge_path) if pose_graph_edge_path.is_file() else ""
        ),
        "pose_graph_edge_provenance_json": edge_evidence["provenance_path"],
        "replay_result_json": str(replay_path),
    }
    return summary, candidate_rows, memory_rows


def candidate_quantiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if to_int(row.get("instrumentation_valid")) == 1]
    groups = {
        "all_exact": valid,
        "eligible": [row for row in valid if is_true(row.get("chain_size_accepted"))],
        "control_eligible": [
            row for row in valid
            if row.get("mode") in {"intervention_off", "adaptive_memory_observe"}
            and is_true(row.get("chain_size_accepted"))
        ],
        "adaptive_eligible": [
            row for row in valid
            if row.get("mode") == "adaptive_quality_aware"
            and is_true(row.get("chain_size_accepted"))
        ],
        "edge_inserted": [row for row in valid if is_true(row.get("edge_inserted"))],
        "memory_active": [
            row for row in valid if to_float(row.get("memory_strength")) > 0.0
        ],
    }
    result: list[dict[str, Any]] = []
    for group_name, group in groups.items():
        for metric in QUANTILE_METRICS:
            values = finite(row.get(metric) for row in group)
            result.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "count": len(values),
                    "p05": percentile(values, 0.05),
                    "p25": percentile(values, 0.25),
                    "median": percentile(values, 0.5),
                    "p75": percentile(values, 0.75),
                    "p95": percentile(values, 0.95),
                    "max": max(values, default=math.nan),
                }
            )
    return result


PAIR_PROVENANCE_FIELDS = (
    "lossless_adapter_sha256",
    "runtime_slam_node_sha256",
    "runtime_cmake_cache_sha256",
    "replay_pose_graph_exporter_sha256",
    "exact_edge_exporter_sha256",
)


def complete_replay_evidence(row: dict[str, Any]) -> bool:
    return (
        to_int(row.get("instrumentation_valid")) == 1
        and to_int(row.get("exact_edge_evidence_valid")) == 1
        and to_int(row.get("runtime_parameter_match")) == 1
        and to_int(row.get("imu_delivery_complete")) == 1
        and to_int(row.get("scan_delivery_complete")) == 1
        and to_int(row.get("queue_drain_complete")) == 1
        and row.get("pose_graph_coverage_status")
        in COMPLETE_POSE_GRAPH_COVERAGE_STATUSES
    )


def compatible_lossless_adapter_hashes(values: Iterable[str]) -> set[str]:
    accepted = {str(value).strip() for value in values if str(value).strip()}
    invalid = sorted(
        value for value in accepted if re.fullmatch(r"[0-9a-f]{64}", value) is None
    )
    if invalid:
        raise ValueError(
            "invalid compatible lossless adapter SHA-256: " + ", ".join(invalid)
        )
    current_adapter = Path(__file__).with_name("lossless_stage34_runner.py")
    if current_adapter.is_file():
        accepted.add(file_sha256(current_adapter))
    return accepted


def paired_adaptive_rows(
    summaries: list[dict[str, Any]],
    compatible_adapter_sha256s: Iterable[str] = (),
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    compatible_adapters = set(compatible_adapter_sha256s)
    control_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summaries:
        if row.get("mode") == "intervention_off":
            key = (str(row.get("label", "")), str(row.get("bag", "")))
            control_groups.setdefault(key, []).append(row)

    adaptive_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summaries:
        if row.get("mode") != "adaptive_quality_aware":
            continue
        key = (str(row.get("reference_label", "")), str(row.get("bag", "")))
        adaptive_groups.setdefault(key, []).append(row)

    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, adaptive_group in adaptive_groups.items():
        controls = control_groups.get(key, [])
        if len(adaptive_group) != 1 or len(controls) != 1 or not key[0] or not key[1]:
            continue
        adaptive = adaptive_group[0]
        control = controls[0]
        strict_provenance_matches = all(
            len(str(adaptive.get(field, ""))) == 64
            and adaptive.get(field) == control.get(field)
            for field in PAIR_PROVENANCE_FIELDS
            if field != "lossless_adapter_sha256"
        )
        adaptive_adapter = str(adaptive.get("lossless_adapter_sha256", ""))
        control_adapter = str(control.get("lossless_adapter_sha256", ""))
        adapter_provenance_matches = (
            len(adaptive_adapter) == 64
            and len(control_adapter) == 64
            and (
                adaptive_adapter == control_adapter
                or {
                    adaptive_adapter,
                    control_adapter,
                } <= compatible_adapters
            )
        )
        reference_nodes_match = (
            to_int(adaptive.get("reference_node_count")) > 0
            and to_int(adaptive.get("reference_node_count"))
            == to_int(control.get("pose_graph_node_count"))
        )
        if (
            complete_replay_evidence(control)
            and complete_replay_evidence(adaptive)
            and to_int(adaptive.get("quality_calibrated")) == 1
            and adaptive.get("comparison_status") == "compared"
            and adaptive.get("fairness_status") in ACCEPTED_FAIRNESS_STATUSES
            and strict_provenance_matches
            and adapter_provenance_matches
            and reference_nodes_match
        ):
            result.append((adaptive, control))
    return result


def paired_runtime_rows(
    summaries: list[dict[str, Any]],
    compatible_adapter_sha256s: Iterable[str] = (),
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for adaptive, control in paired_adaptive_rows(
        summaries,
        compatible_adapter_sha256s,
    ):
        bag = str(adaptive.get("bag", ""))
        control_completion = to_float(control.get("replay_completion_wall_s"))
        adaptive_completion = to_float(adaptive.get("replay_completion_wall_s"))
        result.append(
            {
                "bag": bag,
                "control_label": control.get("label", ""),
                "adaptive_label": adaptive.get("label", ""),
                "control_run_wall_s": control.get("run_wall_s", math.nan),
                "adaptive_run_wall_s": adaptive.get("run_wall_s", math.nan),
                "control_queue_drain_wait_s": control.get(
                    "queue_drain_wait_s", math.nan
                ),
                "adaptive_queue_drain_wait_s": adaptive.get(
                    "queue_drain_wait_s", math.nan
                ),
                "control_replay_completion_wall_s": control_completion,
                "adaptive_replay_completion_wall_s": adaptive_completion,
                "adaptive_minus_control_completion_wall_s": (
                    adaptive_completion - control_completion
                    if math.isfinite(control_completion)
                    and math.isfinite(adaptive_completion)
                    else math.nan
                ),
            }
        )
    return result


def bootstrap_mean_ci(
    values: list[float],
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    clean = finite(values)
    if not clean:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(clean, k=len(clean)))
        for _ in range(max(1, samples))
    ]
    return (
        statistics.fmean(clean),
        percentile(means, 0.025),
        percentile(means, 0.975),
    )


def paired_metric_summary(
    summaries: list[dict[str, Any]],
    samples: int,
    seed: int,
    compatible_adapter_sha256s: Iterable[str] = (),
) -> list[dict[str, Any]]:
    metrics = (
        "odom_consistent_p95_delta_reference_minus_variant_m",
        "return_revisit_p95_delta_reference_minus_variant_m",
    )
    result: list[dict[str, Any]] = []
    paired_rows = [
        adaptive
        for adaptive, _ in paired_adaptive_rows(
            summaries,
            compatible_adapter_sha256s,
        )
    ]
    for mode in sorted({str(row.get("mode")) for row in paired_rows}):
        mode_rows = [row for row in paired_rows if row.get("mode") == mode]
        for offset, metric in enumerate(metrics):
            bag_values = [
                (str(row.get("bag", "")), to_float(row.get(metric)))
                for row in mode_rows
                if math.isfinite(to_float(row.get(metric)))
            ]
            values = [value for _, value in bag_values]
            center, low, high = bootstrap_mean_ci(
                values,
                samples,
                seed + offset,
            )
            largest_bag, largest_value = max(
                bag_values,
                key=lambda item: abs(item[1]),
                default=("", math.nan),
            )
            without_largest = [
                value
                for bag, value in bag_values
                if (bag, value) != (largest_bag, largest_value)
            ]
            epsilon = 1.0e-12
            result.append(
                {
                    "mode": mode,
                    "metric": metric,
                    "dataset_count": len(values),
                    "mean": center,
                    "median": percentile(values, 0.5),
                    "p95": percentile(values, 0.95),
                    "worst": min(values, default=math.nan),
                    "bootstrap_mean_ci95_low": low,
                    "bootstrap_mean_ci95_high": high,
                    "improved_count": sum(value > epsilon for value in values),
                    "unchanged_count": sum(
                        abs(value) <= epsilon for value in values
                    ),
                    "regressed_count": sum(value < -epsilon for value in values),
                    "largest_absolute_effect_bag": largest_bag,
                    "largest_absolute_effect": largest_value,
                    "mean_without_largest_absolute_effect": (
                        statistics.fmean(without_largest)
                        if without_largest
                        else math.nan
                    ),
                }
            )
    return result


def plot_memory(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    values = (
        ("strength", "Memory strength M", "strength"),
        ("effective_duration_s", "Adaptive duration", "s"),
        ("effective_distance_m", "Adaptive radius", "m"),
    )
    for axis, (field, title, unit) in zip(axes, values):
        samples = finite(row.get(field) for row in rows)
        if samples:
            axis.hist(samples, bins=30, color="#287D8E", alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel(unit)
        axis.set_ylabel("samples")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_candidate_relationships(rows: list[dict[str, Any]], path: Path) -> None:
    valid = [
        row for row in rows
        if to_int(row.get("instrumentation_valid")) == 1
        and is_true(row.get("chain_size_accepted"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    final_residual = np.asarray(
        [
            to_float(row.get("final_covariance_weighted_residual_sq"))
            if row.get("final_residual_status") == "ok" else math.nan
            for row in valid
        ]
    )
    color = np.log1p(final_residual)
    plots = (
        (
            "normalized_response",
            "translation_disagreement_m",
            "Response",
            "Translation disagreement (m)",
        ),
        (
            "xy_covariance_max_eigenvalue_m2",
            "scan_match_covariance_weighted_disagreement_sq",
            "Max XY covariance eigenvalue (m^2)",
            "Covariance-weighted disagreement",
        ),
    )
    image = None
    for axis, (x_field, y_field, x_label, y_label) in zip(axes, plots):
        x = np.asarray([to_float(row.get(x_field)) for row in valid])
        y = np.asarray([to_float(row.get(y_field)) for row in valid])
        coordinate_mask = np.isfinite(x) & np.isfinite(y)
        exact_mask = coordinate_mask & np.isfinite(color)
        missing_mask = coordinate_mask & ~np.isfinite(color)
        if np.any(missing_mask):
            axis.scatter(
                x[missing_mask],
                y[missing_mask],
                color="#A0A0A0",
                marker="x",
                s=18,
                alpha=0.55,
                label="no exact final residual",
            )
        if np.any(exact_mask):
            image = axis.scatter(
                x[exact_mask],
                y[exact_mask],
                c=color[exact_mask],
                s=16,
                alpha=0.7,
                cmap="viridis",
                label="exact final residual",
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
        if np.any(missing_mask):
            axis.legend(fontsize=8)
    if image is not None:
        fig.colorbar(image, ax=axes, label="log(1 + final factor residual)")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.14, top=0.95, wspace=0.28)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_action_counts(summaries: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['bag']}\n{row['mode']}" for row in summaries]
    accepted = [to_int(row.get("accepted_candidate_count")) for row in summaries]
    downweighted = [
        to_int(row.get("downweighted_candidate_count")) for row in summaries
    ]
    rejected = [to_int(row.get("rejected_candidate_count")) for row in summaries]
    backend_rejected = [
        to_int(row.get("adaptive_response_rejected_candidate_count"))
        for row in summaries
    ]
    positions = np.arange(len(labels))
    width = 0.20
    figure_width = max(8.0, 2.0 * len(labels))
    fig, axis = plt.subplots(figsize=(figure_width, 4.6))
    axis.bar(
        positions - 1.5 * width,
        accepted,
        width,
        label="shadow accepted",
        color="#2A9D8F",
    )
    axis.bar(
        positions - 0.5 * width,
        downweighted,
        width,
        label="shadow downweighted",
        color="#E9C46A",
    )
    axis.bar(
        positions + 0.5 * width,
        rejected,
        width,
        label="shadow rejected",
        color="#D1495B",
    )
    axis.bar(
        positions + 1.5 * width,
        backend_rejected,
        width,
        label="backend adaptive reject",
        color="#343A40",
    )
    axis.set_ylabel("exact candidates")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_processing_time(summaries: list[dict[str, Any]], path: Path) -> None:
    labels = [f"{row['bag']}\n{row['mode']}" for row in summaries]
    run_wall = [to_float(row.get("run_wall_s")) for row in summaries]
    drain = [to_float(row.get("queue_drain_wait_s")) for row in summaries]
    positions = np.arange(len(labels))
    figure_width = max(8.0, 2.0 * len(labels))
    fig, axis = plt.subplots(figsize=(figure_width, 4.6))
    axis.bar(positions, run_wall, label="playback wall", color="#457B9D")
    axis.bar(positions, drain, bottom=run_wall, label="queue drain", color="#F4A261")
    axis.set_ylabel("wall time (s)")
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_reference_chain_evidence(
    summaries: list[dict[str, Any]], path: Path
) -> None:
    rows = [
        row
        for row in summaries
        if to_int(row.get("reference_chain_decision_count")) > 0
    ]
    figure_width = max(9.0, 2.2 * len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(figure_width, 8.2))
    if not rows:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No stable-side reference-chain decisions",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170)
        plt.close(fig)
        return

    labels = [f"{row['bag']}\n{row.get('mode', '')}" for row in rows]
    positions = np.arange(len(rows))
    accepted = [
        to_int(row.get("reference_chain_decision_accepted_count"))
        for row in rows
    ]
    duplicates = [
        to_int(row.get("reference_chain_decision_correlated_duplicate_count"))
        for row in rows
    ]
    unique = [max(0, total - duplicate) for total, duplicate in zip(
        accepted, duplicates
    )]
    available = [
        to_int(row.get("reference_chain_decision_available_count"))
        for row in rows
    ]
    rejected = [max(0, total - selected) for total, selected in zip(
        available, accepted
    )]
    fail_open = [
        to_int(row.get("reference_chain_decision_fail_open_count"))
        for row in rows
    ]
    outcome_width = 0.20
    axes[0].bar(
        positions - 1.5 * outcome_width,
        unique,
        outcome_width,
        label="accepted unique",
        color="#2A9D8F",
    )
    axes[0].bar(
        positions - 0.5 * outcome_width,
        duplicates,
        outcome_width,
        label="accepted correlated duplicate",
        color="#E9C46A",
    )
    axes[0].bar(
        positions + 0.5 * outcome_width,
        rejected,
        outcome_width,
        label="quality rejected",
        color="#D1495B",
    )
    axes[0].bar(
        positions + 1.5 * outcome_width,
        fail_open,
        outcome_width,
        label="fail-open",
        color="#8D99AE",
    )
    axes[0].set_ylabel("event-level decisions")
    axes[0].set_xticks(positions, labels, rotation=20, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    width = 0.25
    median_ms = [
        to_float(row.get("reference_chain_evaluation_duration_ms_median"))
        for row in rows
    ]
    p95_ms = [
        to_float(row.get("reference_chain_evaluation_duration_ms_p95"))
        for row in rows
    ]
    maximum_ms = [
        to_float(row.get("reference_chain_evaluation_duration_ms_max"))
        for row in rows
    ]
    axes[1].bar(
        positions - width, median_ms, width, label="median", color="#457B9D"
    )
    axes[1].bar(positions, p95_ms, width, label="p95", color="#F4A261")
    axes[1].bar(
        positions + width, maximum_ms, width, label="max", color="#6D597A"
    )
    axes[1].set_ylabel("event evaluation latency (ms)")
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def format_number(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "not available"


def decision_gate_lines(
    summaries: list[dict[str, Any]],
    max_odom_p95_regression_m: float = 0.02,
    max_return_p95_regression_m: float = 0.05,
    compatible_adapter_sha256s: Iterable[str] = (),
) -> list[str]:
    adaptive = [
        row for row in summaries
        if row.get("mode") == "adaptive_quality_aware"
    ]
    if not adaptive:
        return [
            "- Status: not evaluated.",
            "- This selected summary contains control/observe evidence only; "
            "it cannot establish an adaptive map improvement or regression.",
        ]

    valid = [
        row
        for row, _ in paired_adaptive_rows(
            summaries,
            compatible_adapter_sha256s,
        )
    ]
    parity_failures = sum(
        to_int(row.get("parity_failure_count")) for row in adaptive
    )
    graph_mismatches = sum(
        to_int(row.get("node_count_delta")) != 0 for row in adaptive
    )
    lines = [
        f"- Valid adaptive replays: {len(valid)}/{len(adaptive)}.",
        f"- Exact observer parity failures: {parity_failures}.",
        f"- Graph node-count mismatches: {graph_mismatches}.",
    ]

    target_prefixes = ("A03", "A27", "C_Q1", "C_Q3", "H01")
    selected = {
        prefix: [
            row for row in adaptive
            if str(row.get("label", "")).startswith(prefix)
        ]
        for prefix in target_prefixes
    }
    missing_or_ambiguous = [
        prefix for prefix, rows in selected.items() if len(rows) != 1
    ]
    if missing_or_ambiguous:
        lines.append(
            "- Targeted A03/A27/C-Q1/C-Q3/H01 gate: not evaluated in this "
            "selection; missing or ambiguous "
            + ", ".join(missing_or_ambiguous) + "."
        )
        return lines

    c_rows = selected["C_Q1"] + selected["C_Q3"]
    c_no_op = all(
        to_int(row.get("node_count_delta")) == 0
        and math.isclose(
            to_float(row.get("map_occupied_changed_cell_ratio")),
            0.0,
            abs_tol=1.0e-12,
        )
        for row in c_rows
    )
    h01_no_op = all(
        to_int(row.get("node_count_delta")) == 0
        and math.isclose(
            to_float(row.get("map_occupied_changed_cell_ratio")),
            0.0,
            abs_tol=1.0e-12,
        )
        for row in selected["H01"]
    )
    targeted_rows = [row for rows in selected.values() for row in rows]
    targeted_valid = all(row in valid for row in targeted_rows)
    performance_rows = selected["A03"] + selected["A27"]
    odom_guard = all(
        math.isfinite(to_float(row.get(
            "odom_consistent_p95_delta_reference_minus_variant_m"
        )))
        and to_float(row.get(
            "odom_consistent_p95_delta_reference_minus_variant_m"
        )) >= -max_odom_p95_regression_m
        for row in performance_rows
    )
    return_guard = all(
        math.isfinite(to_float(row.get(
            "return_revisit_p95_delta_reference_minus_variant_m"
        )))
        and to_float(row.get(
            "return_revisit_p95_delta_reference_minus_variant_m"
        )) >= -max_return_p95_regression_m
        for row in performance_rows
    )
    targeted_passed = (
        targeted_valid
        and parity_failures == 0
        and c_no_op
        and h01_no_op
        and odom_guard
        and return_guard
    )
    lines.extend(
        [
            "- Targeted A03/A27/C-Q1/C-Q3/H01 gate: "
            f"{'passed' if targeted_passed else 'failed'}.",
            f"- C-Q1/C-Q3 exact no-op: {c_no_op}; H01 exact no-op: {h01_no_op}.",
            "- A03/A27 odometry/return regression guards: "
            f"{odom_guard}/{return_guard} "
            f"(limits {max_odom_p95_regression_m:.3f}/"
            f"{max_return_p95_regression_m:.3f} m).",
            "- Passing this development gate is not production promotion; "
            "the field regression and unseen H02/H03 remain separate gates.",
        ]
    )
    return lines


def make_report(
    path: Path,
    summaries: list[dict[str, Any]],
    quantiles: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    calibration_json: Path | None = None,
    max_odom_p95_regression_m: float = 0.02,
    max_return_p95_regression_m: float = 0.05,
    compatible_adapter_sha256s: Iterable[str] = (),
) -> None:
    compatible_adapters = set(compatible_adapter_sha256s)
    valid_runs = sum(to_int(row.get("instrumentation_valid")) for row in summaries)
    adaptive_rows = [
        row for row in summaries
        if row.get("mode") == "adaptive_quality_aware"
    ]
    reference_chain_rows = [
        row
        for row in summaries
        if to_int(row.get("reference_chain_decision_count")) > 0
    ]
    controls = {
        (str(row.get("label")), str(row.get("bag"))): row
        for row in summaries
        if row.get("mode") == "intervention_off"
    }
    calibration = read_json(
        calibration_json
        if calibration_json is not None
        else path.parents[1]
        / "calibration"
        / "adaptive_quality_calibration.json"
    )
    parameters = calibration.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    control_completion = finite(
        row.get("control_replay_completion_wall_s") for row in runtime_rows
    )
    adaptive_completion = finite(
        row.get("adaptive_replay_completion_wall_s") for row in runtime_rows
    )
    completion_deltas = finite(
        row.get("adaptive_minus_control_completion_wall_s")
        for row in runtime_rows
    )
    lines = [
        "# Adaptive Roughness Memory Development Evidence",
        "",
        "This report is generated from exact Karto near-chain candidate logs. ",
        "Residuals are diagnostics, not harmful-factor ground truth.",
        "",
        "## Replay coverage",
        "",
        f"- Replay runs: {len(summaries)}",
        f"- Runs with valid exact-candidate accounting: {valid_runs}",
        f"- Adaptive runs with calibrated quality parameters: "
        f"{sum(to_int(row.get('quality_calibrated')) for row in adaptive_rows)}"
        f"/{len(adaptive_rows)}",
        f"- Calibration status: {calibration.get('status', 'not available')}; "
        f"production ready: {calibration.get('production_ready', False)}",
        f"- Calibrated reject risk / response cap: "
        f"{format_number(parameters.get('roughness_adaptive_reject_risk'), 6)} / "
        f"{format_number(parameters.get('roughness_adaptive_dynamic_threshold_max'), 6)}",
        "- Audited compatible lossless adapter SHA-256s: "
        + (", ".join(sorted(compatible_adapters)) if compatible_adapters else "none"),
        "- Post-confirmation remains observe-only and is not an online rejection label.",
        "",
        "### Selected replay artifacts",
        "",
        "| Bag | Label | Mode | Graph nodes | Exact/eligible | "
        "Scans received/effective | Status | Point-only map overlay |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in summaries:
        lines.append(
            "| {bag} | {label} | {mode} | {nodes} | {exact}/{eligible} | "
            "{scan_received}/{scan_expected} | {status} | `{overlay}` |".format(
                bag=row.get("bag", ""),
                label=row.get("label", ""),
                mode=row.get("mode", ""),
                nodes=row.get("pose_graph_node_count", ""),
                exact=row.get("exact_candidate_count", ""),
                eligible=row.get("eligible_candidate_count", ""),
                scan_received=row.get("scan_received_count", ""),
                scan_expected=row.get("scan_effective_expected_count", ""),
                status=row.get("status", ""),
                overlay=row.get("map_with_pose_graph_trajectory_png", ""),
            )
        )
    lines.extend(
        [
            "",
            "### Paired adaptive coverage",
            "",
            "| Bag | Graph control/adaptive | Exact/eligible | "
            "Shadow A/D/R | Backend adaptive rejects | Eligible fail-open | "
            "Parity failures | "
            "Scans received/effective | Map F1 / changed ratio | "
            "Odom p95 delta (m) | Return p95 delta (m) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in adaptive_rows:
        bag = str(row.get("bag", ""))
        reference_label = str(row.get("reference_label", ""))
        control = controls.get((reference_label, bag))
        control = control or {}
        lines.append(
            "| {bag} | {control_nodes}/{adaptive_nodes} | "
            "{exact}/{eligible} | {accepted}/{downweighted}/{rejected} | "
            "{adaptive_rejects} | {fail_open} | {parity} | "
            "{scan_received}/{scan_expected} | "
            "{map_f1} / {map_change} | "
            "{odom_delta} | {return_delta} |".format(
                bag=row.get("bag", ""),
                control_nodes=control.get("pose_graph_node_count", ""),
                adaptive_nodes=row.get("pose_graph_node_count", ""),
                exact=row.get("exact_candidate_count", ""),
                eligible=row.get("eligible_candidate_count", ""),
                accepted=row.get("accepted_candidate_count", ""),
                downweighted=row.get("downweighted_candidate_count", ""),
                rejected=row.get("rejected_candidate_count", ""),
                adaptive_rejects=row.get(
                    "adaptive_response_rejected_candidate_count", ""
                ),
                fail_open=row.get("fail_open_candidate_count", ""),
                parity=row.get("parity_failure_count", ""),
                scan_received=row.get("scan_received_count", ""),
                scan_expected=row.get("scan_effective_expected_count", ""),
                map_f1=format_number(row.get("map_occupied_boundary_f1"), 6),
                map_change=format_number(
                    row.get("map_occupied_changed_cell_ratio"), 6
                ),
                odom_delta=format_number(
                    row.get(
                        "odom_consistent_p95_delta_reference_minus_variant_m"
                    ),
                    6,
                ),
                return_delta=format_number(
                    row.get(
                        "return_revisit_p95_delta_reference_minus_variant_m"
                    ),
                    6,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Candidate evidence",
            "",
        "An exact final factor residual is read from the serialized Karto "
            "edge, including its source-local covariance. A pose-only "
            "reconstruction is marked approximate and excluded from "
            "calibration. Both are reported only for an inserted edge.",
            "",
        ]
    )
    eligible_response = next(
        (
            row for row in quantiles
            if row.get("group") == "control_eligible"
            and row.get("metric") == "normalized_response"
        ),
        {},
    )
    exact_final_residual_total = sum(
        to_int(row.get("final_factor_residual_count")) for row in summaries
    )
    approximate_final_residual_total = sum(
        to_int(row.get("final_factor_residual_approximate_count"))
        for row in summaries
    )
    lines.extend(
        [
            f"- Control eligible response count: {eligible_response.get('count', 0)}",
            "- Control eligible response median/p95: "
            f"{format_number(eligible_response.get('median'))} / "
            f"{format_number(eligible_response.get('p95'))}",
            "- Exact / approximate final factor residual totals: "
            f"{exact_final_residual_total} / {approximate_final_residual_total}",
            f"- Adaptive exact/eligible totals: "
            f"{sum(to_int(row.get('exact_candidate_count')) for row in adaptive_rows)} / "
            f"{sum(to_int(row.get('eligible_candidate_count')) for row in adaptive_rows)}",
            f"- Adaptive shadow accepted/downweighted/rejected totals: "
            f"{sum(to_int(row.get('accepted_candidate_count')) for row in adaptive_rows)} / "
            f"{sum(to_int(row.get('downweighted_candidate_count')) for row in adaptive_rows)} / "
            f"{sum(to_int(row.get('rejected_candidate_count')) for row in adaptive_rows)}",
            "- Adaptive-induced backend response rejects: "
            f"{sum(to_int(row.get('adaptive_response_rejected_candidate_count')) for row in adaptive_rows)}",
            f"- Eligible fail-open / observer parity failures: "
            f"{sum(to_int(row.get('fail_open_candidate_count')) for row in adaptive_rows)} / "
            f"{sum(to_int(row.get('parity_failure_count')) for row in adaptive_rows)}",
            "- Dynamic thresholds must be selected from these exact-candidate "
            "distributions and then validated by paired replay.",
            "",
            "## Stable-side reference-chain observer",
            "",
            "An accepted row is a shadow decision only. "
            "It does not modify the graph; correlated passages are counted "
            "separately so repeated "
            "evidence is not mistaken for independent support.",
            "",
            "| Dataset | Context available/total | "
            "Hypotheses available/total | Accepted unique/duplicate | "
            "Rejected/fail-open | p95/max latency (ms) | "
            "Graph actions / injected edges |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in reference_chain_rows:
        accepted = to_int(row.get("reference_chain_decision_accepted_count"))
        duplicates = to_int(
            row.get("reference_chain_decision_correlated_duplicate_count")
        )
        rejected = max(
            0,
            to_int(row.get("reference_chain_decision_available_count"))
            - accepted,
        )
        lines.append(
            "| {dataset} | {context_available}/{context_total} | "
            "{hypothesis_available}/{hypothesis_total} | "
            "{unique}/{duplicate} | {rejected}/{fail_open} | "
            "{p95}/{maximum} | {graph_actions}/{edges} |".format(
                dataset=f"{row.get('bag', '')} ({row.get('label', '')})",
                context_available=row.get(
                    "reference_chain_event_available_count", ""
                ),
                context_total=row.get("reference_chain_event_count", ""),
                hypothesis_available=row.get(
                    "reference_chain_hypothesis_available_count", ""
                ),
                hypothesis_total=row.get(
                    "reference_chain_hypothesis_count", ""
                ),
                unique=max(0, accepted - duplicates),
                duplicate=duplicates,
                rejected=rejected,
                fail_open=row.get(
                    "reference_chain_decision_fail_open_count", ""
                ),
                p95=format_number(
                    row.get("reference_chain_evaluation_duration_ms_p95"), 2
                ),
                maximum=format_number(
                    row.get("reference_chain_evaluation_duration_ms_max"), 2
                ),
                graph_actions=row.get("place_graph_action_count", ""),
                edges=row.get("place_constraint_edge_injected_count", ""),
            )
        )
    if not reference_chain_rows:
        lines.append(
            "| not evaluated | 0/0 | 0/0 | 0/0 | 0/0 | n/a | 0/0 |"
        )
    lines.extend(
        [
            "",
            "## Paired effects",
            "",
            "| Mode | Metric | Datasets | Mean | Median | p95 | Worst | Bootstrap 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row.get("mode") != "adaptive_quality_aware":
            continue
        lines.append(
            "| {mode} | {metric} | {count} | {mean} | {median} | {p95} | {worst} | "
            "[{low}, {high}] |".format(
                mode=row.get("mode", ""),
                metric=row.get("metric", ""),
                count=row.get("dataset_count", 0),
                mean=format_number(row.get("mean")),
                median=format_number(row.get("median")),
                p95=format_number(row.get("p95")),
                worst=format_number(row.get("worst")),
                low=format_number(row.get("bootstrap_mean_ci95_low")),
                high=format_number(row.get("bootstrap_mean_ci95_high")),
            )
        )
    lines.extend(
        [
            "",
            "### Outlier sensitivity",
            "",
            "Positive values mean improvement; zero means no measured change.",
            "",
            "| Metric | Improved / unchanged / regressed | "
            "Largest absolute effect (bag, m) | Mean without it (m) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in paired:
        if row.get("mode") != "adaptive_quality_aware":
            continue
        lines.append(
            "| {metric} | {improved}/{unchanged}/{regressed} | "
            "{bag}, {largest} | {without} |".format(
                metric=row.get("metric", ""),
                improved=row.get("improved_count", 0),
                unchanged=row.get("unchanged_count", 0),
                regressed=row.get("regressed_count", 0),
                bag=row.get("largest_absolute_effect_bag", ""),
                largest=format_number(row.get("largest_absolute_effect")),
                without=format_number(
                    row.get("mean_without_largest_absolute_effect")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Replay timing",
            "",
            "These values are end-to-end replay completion times "
            "(configured replay wall plus observed queue drain), not internal "
            "per-scan latency. Single-run scheduler and machine load affect them.",
            "",
            "| Bag | Control completion (s) | Adaptive completion (s) | Delta (s) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in runtime_rows:
        lines.append(
            "| {bag} | {control} | {adaptive} | {delta} |".format(
                bag=row.get("bag", ""),
                control=format_number(
                    row.get("control_replay_completion_wall_s"), 3
                ),
                adaptive=format_number(
                    row.get("adaptive_replay_completion_wall_s"), 3
                ),
                delta=format_number(
                    row.get("adaptive_minus_control_completion_wall_s"), 3
                ),
            )
        )
    lines.extend(
        [
            "",
            "- Control completion mean/p95: "
            f"{format_number(mean(control_completion), 3)} / "
            f"{format_number(percentile(control_completion, 0.95), 3)} s",
            "- Adaptive completion mean/p95: "
            f"{format_number(mean(adaptive_completion), 3)} / "
            f"{format_number(percentile(adaptive_completion, 0.95), 3)} s",
            "- Adaptive-minus-control completion mean/median/p95/worst: "
            f"{format_number(mean(completion_deltas), 3)} / "
            f"{format_number(percentile(completion_deltas, 0.5), 3)} / "
            f"{format_number(percentile(completion_deltas, 0.95), 3)} / "
            f"{format_number(max(completion_deltas, default=math.nan), 3)} s",
            "",
            "## Primary map overlays",
            "",
        ]
    )
    for row in summaries:
        lines.append(
            f"- {row.get('bag', '')} ({row.get('label', '')}, "
            f"{row.get('mode', '')}): "
            f"`{row.get('map_with_pose_graph_trajectory_png', '')}`"
        )
        if row.get("map_comparison_png"):
            lines.append(
                f"- {row.get('bag', '')} paired map comparison: "
                f"`{row.get('map_comparison_png', '')}`"
            )
    lines.extend(
        [
            "",
            "## Decision gate",
            "",
        ]
    )
    lines.extend(
        decision_gate_lines(
            summaries,
            max_odom_p95_regression_m,
            max_return_p95_regression_m,
            compatible_adapters,
        )
    )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if (
        not math.isfinite(args.max_odom_p95_regression_m)
        or args.max_odom_p95_regression_m < 0.0
        or not math.isfinite(args.max_return_p95_regression_m)
        or args.max_return_p95_regression_m < 0.0
    ):
        raise ValueError("regression guard limits must be finite and nonnegative")
    output_root = args.output_root.expanduser().resolve()
    compatible_adapters = compatible_lossless_adapter_hashes(
        args.compatible_lossless_adapter_sha256
    )
    artifact_root = (
        args.artifact_root.expanduser().resolve()
        if args.artifact_root is not None
        else output_root
    )
    stage = output_root / "stage34_roughness_gate_replay"
    replay_paths = select_replay_paths(
        (stage / "variant").glob("*/bags/*/replay_result.json"),
        args.include_label,
    )
    summaries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    for replay_path in replay_paths:
        pose_graph_overlay_output = None
        if artifact_root != output_root:
            pose_graph_overlay_output = (
                artifact_root
                / "map_overlays"
                / replay_path.parents[2].name
                / replay_path.parent.name
                / "map_with_pose_graph_trajectory.png"
            )
        summary, replay_candidates, replay_memory = summarize_replay(
            stage,
            replay_path,
            pose_graph_overlay_output,
        )
        summaries.append(summary)
        candidates.extend(replay_candidates)
        memory.extend(replay_memory)

    summaries.sort(key=lambda row: (str(row.get("bag")), str(row.get("label"))))
    candidates.sort(
        key=lambda row: (
            str(row.get("bag")),
            to_int(row.get("stamp_ns")),
            to_int(row.get("candidate_index")),
        )
    )
    memory.sort(
        key=lambda row: (str(row.get("bag")), to_int(row.get("stamp_ns")))
    )
    quantiles = candidate_quantiles(candidates)
    paired = paired_metric_summary(
        summaries,
        args.bootstrap_samples,
        args.seed,
        compatible_adapters,
    )
    runtime_rows = paired_runtime_rows(summaries, compatible_adapters)

    candidate_fields = sorted(
        {key for row in candidates for key in row},
        key=lambda value: (value not in {"label", "bag", "mode"}, value),
    )
    memory_fields = sorted(
        {key for row in memory for key in row},
        key=lambda value: (value not in {"label", "bag", "mode"}, value),
    )
    write_csv(
        artifact_root / "csv" / "adaptive_replay_summary.csv",
        summaries,
        SUMMARY_FIELDS,
    )
    write_csv(
        artifact_root / "csv" / "exact_near_chain_candidates.csv",
        candidates,
        candidate_fields,
    )
    write_csv(
        artifact_root / "csv" / "adaptive_memory_samples.csv",
        memory,
        memory_fields,
    )
    write_csv(
        artifact_root / "csv" / "candidate_metric_quantiles.csv",
        quantiles,
        ("group", "metric", "count", "p05", "p25", "median", "p75", "p95", "max"),
    )
    write_csv(
        artifact_root / "csv" / "paired_effect_bootstrap_ci.csv",
        paired,
        (
            "mode",
            "metric",
            "dataset_count",
            "mean",
            "median",
            "p95",
            "worst",
            "bootstrap_mean_ci95_low",
            "bootstrap_mean_ci95_high",
            "improved_count",
            "unchanged_count",
            "regressed_count",
            "largest_absolute_effect_bag",
            "largest_absolute_effect",
            "mean_without_largest_absolute_effect",
        ),
    )
    write_csv(
        artifact_root / "csv" / "paired_replay_timing.csv",
        runtime_rows,
        (
            "bag",
            "control_label",
            "adaptive_label",
            "control_run_wall_s",
            "adaptive_run_wall_s",
            "control_queue_drain_wait_s",
            "adaptive_queue_drain_wait_s",
            "control_replay_completion_wall_s",
            "adaptive_replay_completion_wall_s",
            "adaptive_minus_control_completion_wall_s",
        ),
    )
    plot_memory(memory, artifact_root / "plots" / "adaptive_memory_distributions.png")
    plot_candidate_relationships(
        candidates,
        artifact_root / "plots" / "candidate_quality_relationships.png",
    )
    plot_action_counts(
        summaries,
        artifact_root / "plots" / "candidate_action_counts.png",
    )
    plot_processing_time(
        summaries,
        artifact_root / "plots" / "replay_processing_time.png",
    )
    plot_reference_chain_evidence(
        summaries,
        artifact_root / "plots" / "reference_chain_evidence.png",
    )
    make_report(
        artifact_root / "report" / "adaptive_roughness_memory_report.md",
        summaries,
        quantiles,
        paired,
        runtime_rows,
        (
            args.calibration_json.expanduser().resolve()
            if args.calibration_json is not None
            else None
        ),
        args.max_odom_p95_regression_m,
        args.max_return_p95_regression_m,
        compatible_adapters,
    )
    print(artifact_root)
    print(f"replays={len(summaries)} candidates={len(candidates)} memory={len(memory)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
