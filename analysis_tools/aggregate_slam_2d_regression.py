#!/usr/bin/env python3
"""Aggregate baseline/post 2D SLAM replay evidence across field bags.

This tool deliberately treats return/revisit drift as a consistency proxy. It
does not claim absolute trajectory or map accuracy because no ground-truth
trajectory is available in the replay artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ACCEPTED_REPLAY_STATUSES = {
    "map_and_map_trajectory",
    "map_and_stationary_pose_graph",
}
REQUIRED_REPLAY_TRUE_FIELDS = (
    "runtime_slam_parameter_match",
    "slam_input_complete",
    "slam_recorded_scan_delivery_complete",
    "roughness_imu_delivery_complete",
    "roughness_imu_delivery_fingerprint_match",
    "queue_drain_complete",
    "pose_graph_captured",
    "pose_graph_coverage_complete",
)
USABLE_INPUT_STATUSES = {"ready", "reindexed"}
DEFAULT_REVISIT_CSV = "return_revisit_pairs_common.csv"
TRAJECTORY_RENDERING_POLICY = "points_only_no_connected_segments"
DEFAULT_TIE_BANDS_M = (0.01, 0.025, 0.05, 0.1)
PAIR_IDENTITY_COLUMN_SETS = (
    ("outbound_stamp_s", "return_stamp_s", "candidate_index"),
    ("pair_index",),
)

CSV_COLUMNS = (
    "bag",
    "source_root",
    "input_status",
    "input_usable",
    "input_duration_s",
    "input_scan_laserscan_count",
    "input_imu_count",
    "input_wheel_odom_count",
    "manifest_member",
    "unexpected_artifact",
    "unexpected_artifact_locations",
    "comparison_state",
    "replay_pair_complete",
    "proxy_available",
    "exclusion_reasons",
    "metric_kind",
    "ground_truth_available",
    "absolute_accuracy_claim_supported",
    "baseline_replay_state",
    "baseline_status",
    "baseline_failure_reasons",
    "baseline_map_present",
    "baseline_map_png",
    "baseline_stationary_replay",
    "baseline_pose_graph_trajectory_count",
    "baseline_map_width_m",
    "baseline_map_height_m",
    "baseline_map_area_m2",
    "baseline_map_known_ratio",
    "baseline_map_occupied_ratio",
    "baseline_roughness_spatial_passage_count",
    "baseline_roughness_recovery_count",
    "post_replay_state",
    "post_status",
    "post_failure_reasons",
    "post_map_present",
    "post_map_png",
    "post_stationary_replay",
    "post_pose_graph_trajectory_count",
    "post_map_width_m",
    "post_map_height_m",
    "post_map_area_m2",
    "post_map_known_ratio",
    "post_map_occupied_ratio",
    "post_roughness_spatial_passage_count",
    "post_roughness_recovery_count",
    "map_area_delta_m2",
    "map_known_ratio_delta",
    "map_occupied_ratio_delta",
    "comparison_csv_present",
    "comparison_csv",
    "comparison_csv_error",
    "comparison_csv_trusted",
    "comparison_csv_trust_status",
    "comparison_csv_trust_reasons",
    "comparison_map_artifact_present",
    "comparison_map_artifact_trusted",
    "comparison_map_artifact_trust_status",
    "comparison_map_artifact_trust_reasons",
    "comparison_map_artifact",
    "return_pair_count",
    "return_pair_skipped_row_count",
    "provided_improvement_mismatch_count",
    "pair_identity_columns",
    "pair_identity_validation_status",
    "pair_identity_invalid_row_count",
    "pair_identity_duplicate_row_count",
    "baseline_return_abs_drift_mean_m",
    "baseline_return_abs_drift_median_m",
    "baseline_return_abs_drift_p95_m",
    "post_return_abs_drift_mean_m",
    "post_return_abs_drift_median_m",
    "post_return_abs_drift_p95_m",
    "return_abs_drift_improvement_mean_m",
    "return_abs_drift_improvement_median_pair_m",
    "return_abs_drift_median_level_delta_m",
    "return_abs_drift_p95_delta_m",
    "improved_pair_count",
    "tied_pair_count",
    "regressed_pair_count",
    "improved_pair_fraction",
    "bag_outcome_by_mean_proxy",
)

TIE_BAND_CSV_COLUMNS = (
    "tie_band_m",
    "metric_kind",
    "ground_truth_available",
    "absolute_accuracy_claim_supported",
    "comparable_bag_count",
    "improved_bag_count",
    "tied_bag_count",
    "regressed_bag_count",
    "pooled_pair_count",
    "improved_pair_count",
    "tied_pair_count",
    "regressed_pair_count",
    "improved_pair_fraction",
    "tied_pair_fraction",
    "regressed_pair_fraction",
)


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _integer(value: Any) -> int | None:
    converted = _finite_float(value)
    return int(converted) if converted is not None else None


def _is_exact_one(value: Any) -> bool:
    """Accept numeric representations of one, and no other success value."""
    if isinstance(value, bool):
        return False
    converted = _finite_float(value)
    return converted == 1.0 if converted is not None else False


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for finite values."""
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(finite) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(payload, dict):
        return None, "top-level JSON value is not an object"
    return payload, ""


def _map_dimensions(payload: Mapping[str, Any]) -> tuple[float | None, float | None, float | None]:
    width = _finite_float(payload.get("map_width"))
    height = _finite_float(payload.get("map_height"))
    resolution = _finite_float(payload.get("map_resolution_m"))
    if width is None or height is None or resolution is None:
        return None, None, None
    width_m = width * resolution
    height_m = height * resolution
    return width_m, height_m, width_m * height_m


def inspect_replay(variant_root: Path, bag: str) -> dict[str, Any]:
    """Inspect one replay without trusting paths embedded in its JSON."""
    bag_dir = variant_root / "bags" / bag
    result_path = bag_dir / "replay_result.json"
    map_path = bag_dir / "map.png"
    base = {
        "replay_state": "missing_bag_directory",
        "replay_complete": 0,
        "failure_reasons": "missing_bag_directory",
        "status": "",
        "map_present": int(map_path.is_file()),
        "map_png": str(map_path) if map_path.is_file() else "",
        "stationary_replay": 0,
        "pose_graph_trajectory_count": None,
        "map_width_m": None,
        "map_height_m": None,
        "map_area_m2": None,
        "map_known_ratio": None,
        "map_occupied_ratio": None,
        "roughness_spatial_passage_count": None,
        "roughness_recovery_count": None,
        "replay_result_json": str(result_path),
    }
    if not bag_dir.is_dir():
        return base
    if not result_path.is_file():
        base.update(
            replay_state="missing_replay_result",
            failure_reasons="missing_replay_result",
        )
        return base

    payload, load_error = _load_json(result_path)
    if payload is None:
        base.update(
            replay_state="invalid_replay_result_json",
            failure_reasons=f"invalid_replay_result_json:{load_error}",
        )
        return base

    status = str(payload.get("status", ""))
    reasons: list[str] = []
    if status not in ACCEPTED_REPLAY_STATUSES:
        reasons.append(f"unaccepted_status:{status or 'missing'}")
    for field in REQUIRED_REPLAY_TRUE_FIELDS:
        if not _is_exact_one(payload.get(field)):
            reasons.append(f"{field}_not_exactly_one")
    if not map_path.is_file():
        reasons.append("missing_map")

    width_m, height_m, area_m2 = _map_dimensions(payload)
    base.update(
        replay_state="complete" if not reasons else "failed_integrity",
        replay_complete=int(not reasons),
        failure_reasons=";".join(reasons),
        status=status,
        map_present=int(map_path.is_file()),
        map_png=str(map_path) if map_path.is_file() else "",
        stationary_replay=int(_is_exact_one(payload.get("stationary_replay"))),
        pose_graph_trajectory_count=_integer(payload.get("pose_graph_trajectory_count")),
        map_width_m=width_m,
        map_height_m=height_m,
        map_area_m2=area_m2,
        map_known_ratio=_finite_float(payload.get("map_known_ratio")),
        map_occupied_ratio=_finite_float(payload.get("map_occupied_ratio")),
        roughness_spatial_passage_count=_integer(
            payload.get("roughness_near_chain_spatial_passage_count")
        ),
        roughness_recovery_count=_integer(
            payload.get("roughness_near_chain_recovery_count")
        ),
    )
    return base


def load_input_manifest(path: Path | None) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if path is None:
        return [], {}
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "name" not in reader.fieldnames:
            raise ValueError(f"input manifest has no name column: {path}")
        for raw in reader:
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            if name in rows:
                raise ValueError(f"duplicate bag in input manifest: {name}")
            status = str(raw.get("status", "")).strip()
            integrity = str(raw.get("storage_integrity_status", "")).strip()
            usable = status in USABLE_INPUT_STATUSES and integrity == "passed"
            rows[name] = {
                "source_root": str(raw.get("source_root", "")),
                "input_status": status,
                "input_usable": int(usable),
                "input_duration_s": _finite_float(raw.get("duration_s")),
                "input_scan_laserscan_count": _integer(raw.get("scan_laserscan_count")),
                "input_imu_count": _integer(raw.get("imu_count")),
                "input_wheel_odom_count": _integer(raw.get("wheel_odom_count")),
            }
            order.append(name)
    return order, rows


def find_input_manifest(*roots: Path) -> Path | None:
    """Find the nearest field-regression manifest above either variant root."""
    candidates: list[Path] = []
    for root in roots:
        for ancestor in (root, *root.parents[:5]):
            candidate = ancestor / "input_bag_manifest.csv"
            if candidate not in candidates:
                candidates.append(candidate)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _directory_names(root: Path) -> set[str]:
    bags = root / "bags"
    if not bags.is_dir():
        return set()
    return {path.name for path in bags.iterdir() if path.is_dir()}


def _comparison_names(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def _comparison_source_artifacts(
    baseline_root: Path, post_root: Path, bag: str
) -> list[Path]:
    """Return current replay artifacts that a comparison can depend on."""
    sources: list[Path] = []
    for root in (baseline_root, post_root):
        bag_dir = root / "bags" / bag
        candidates = [bag_dir / "replay_result.json", bag_dir / "map.png"]
        if bag_dir.is_dir():
            candidates.extend(sorted(bag_dir.glob("trajectory*.csv")))
        for candidate in candidates:
            if candidate.is_file() and candidate not in sources:
                sources.append(candidate)
    return sources


def assess_comparison_artifact(
    path: Path,
    source_artifacts: Sequence[Path],
    *,
    ineligible_reason: str = "",
) -> dict[str, Any]:
    """Conservatively trust an artifact only when it is newer than its sources."""
    assessment = {
        "present": int(path.is_file()),
        "trusted": 0,
        "status": "",
        "reasons": "",
    }
    if ineligible_reason:
        assessment.update(status="not_evaluated", reasons=ineligible_reason)
        return assessment
    if not path.is_file():
        assessment.update(status="missing", reasons="missing_artifact")
        return assessment
    try:
        target_mtime_ns = path.stat().st_mtime_ns
        source_mtimes = [
            (source, source.stat().st_mtime_ns) for source in source_artifacts
        ]
    except OSError as error:
        assessment.update(
            status="stat_error", reasons=f"{type(error).__name__}:{error}"
        )
        return assessment
    if not source_mtimes:
        assessment.update(status="untrusted", reasons="no_source_artifacts")
        return assessment
    newer_sources = [
        str(source) for source, mtime_ns in source_mtimes if mtime_ns > target_mtime_ns
    ]
    if newer_sources:
        assessment.update(
            status="stale",
            reasons="older_than_source:" + ",".join(newer_sources),
        )
        return assessment
    assessment.update(trusted=1, status="trusted")
    return assessment


def _select_pair_identity_columns(fieldnames: Sequence[str]) -> tuple[str, ...]:
    available = set(fieldnames)
    return next(
        (columns for columns in PAIR_IDENTITY_COLUMN_SETS if set(columns) <= available),
        (),
    )


def _pair_identity(
    row: Mapping[str, Any], columns: Sequence[str]
) -> tuple[float | int, ...] | None:
    identity: list[float | int] = []
    for column in columns:
        value = _finite_float(row.get(column))
        if value is None:
            return None
        if column.endswith("index"):
            if not value.is_integer():
                return None
            identity.append(int(value))
        else:
            identity.append(value)
    return tuple(identity)


def load_return_proxy_pairs(path: Path) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Load finite return/revisit proxy rows and recompute improvement."""
    diagnostics = {
        "csv_present": int(path.is_file()),
        "csv_error": "",
        "content_trusted": 0,
        "skipped_row_count": 0,
        "provided_improvement_mismatch_count": 0,
        "pair_identity_columns": "",
        "pair_identity_validation_status": "not_available",
        "pair_identity_invalid_row_count": 0,
        "pair_identity_duplicate_row_count": 0,
    }
    if not path.is_file():
        diagnostics["csv_error"] = "missing_comparison_csv"
        return [], diagnostics
    required = {"baseline_abs_drift_m", "post_abs_drift_m"}
    pairs: list[dict[str, float]] = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            if not fieldnames:
                diagnostics["csv_error"] = "empty_csv_without_header"
                return [], diagnostics
            missing = required - set(fieldnames)
            if missing:
                diagnostics["csv_error"] = "missing_columns:" + ",".join(sorted(missing))
                return [], diagnostics
            identity_columns = _select_pair_identity_columns(fieldnames)
            diagnostics["pair_identity_columns"] = ",".join(identity_columns)
            if identity_columns:
                diagnostics["pair_identity_validation_status"] = "valid"
            seen_identities: set[tuple[float | int, ...]] = set()
            for row in reader:
                if identity_columns:
                    identity = _pair_identity(row, identity_columns)
                    if identity is None:
                        diagnostics["pair_identity_invalid_row_count"] += 1
                        continue
                    if identity in seen_identities:
                        diagnostics["pair_identity_duplicate_row_count"] += 1
                        continue
                    seen_identities.add(identity)
                baseline = _finite_float(row.get("baseline_abs_drift_m"))
                post = _finite_float(row.get("post_abs_drift_m"))
                if baseline is None or post is None or baseline < 0.0 or post < 0.0:
                    diagnostics["skipped_row_count"] += 1
                    continue
                improvement = baseline - post
                provided = _finite_float(row.get("abs_drift_improvement_m"))
                if provided is not None and not math.isclose(
                    provided, improvement, rel_tol=1e-9, abs_tol=1e-9
                ):
                    diagnostics["provided_improvement_mismatch_count"] += 1
                pairs.append(
                    {
                        "baseline_abs_drift_m": baseline,
                        "post_abs_drift_m": post,
                        "improvement_m": improvement,
                    }
                )
    except (OSError, UnicodeError, csv.Error) as error:
        diagnostics["csv_error"] = f"{type(error).__name__}: {error}"
        return [], diagnostics
    identity_errors: list[str] = []
    if diagnostics["pair_identity_invalid_row_count"]:
        identity_errors.append(
            "invalid_pair_identity:"
            f"{diagnostics['pair_identity_invalid_row_count']}"
        )
    if diagnostics["pair_identity_duplicate_row_count"]:
        identity_errors.append(
            "duplicate_pair_identity:"
            f"{diagnostics['pair_identity_duplicate_row_count']}"
        )
    if identity_errors:
        diagnostics["pair_identity_validation_status"] = "invalid"
        diagnostics["csv_error"] = ";".join(identity_errors)
        return pairs, diagnostics
    diagnostics["content_trusted"] = 1
    if not pairs and not diagnostics["csv_error"]:
        diagnostics["csv_error"] = "no_valid_proxy_rows"
    return pairs, diagnostics


def summarize_pairs(pairs: Sequence[Mapping[str, float]], tie_epsilon_m: float) -> dict[str, Any]:
    baseline = [pair["baseline_abs_drift_m"] for pair in pairs]
    post = [pair["post_abs_drift_m"] for pair in pairs]
    improvements = [pair["improvement_m"] for pair in pairs]
    wins = sum(value > tie_epsilon_m for value in improvements)
    losses = sum(value < -tie_epsilon_m for value in improvements)
    ties = len(improvements) - wins - losses
    baseline_median = percentile(baseline, 0.5)
    post_median = percentile(post, 0.5)
    baseline_p95 = percentile(baseline, 0.95)
    post_p95 = percentile(post, 0.95)
    mean_improvement = _mean(improvements)
    if mean_improvement is None:
        outcome = ""
    elif mean_improvement > tie_epsilon_m:
        outcome = "improved"
    elif mean_improvement < -tie_epsilon_m:
        outcome = "regressed"
    else:
        outcome = "tied"
    return {
        "return_pair_count": len(pairs),
        "baseline_return_abs_drift_mean_m": _mean(baseline),
        "baseline_return_abs_drift_median_m": baseline_median,
        "baseline_return_abs_drift_p95_m": baseline_p95,
        "post_return_abs_drift_mean_m": _mean(post),
        "post_return_abs_drift_median_m": post_median,
        "post_return_abs_drift_p95_m": post_p95,
        "return_abs_drift_improvement_mean_m": mean_improvement,
        "return_abs_drift_improvement_median_pair_m": percentile(improvements, 0.5),
        "return_abs_drift_median_level_delta_m": _difference(
            baseline_median, post_median
        ),
        "return_abs_drift_p95_delta_m": _difference(baseline_p95, post_p95),
        "improved_pair_count": wins,
        "tied_pair_count": ties,
        "regressed_pair_count": losses,
        "improved_pair_fraction": wins / len(pairs) if pairs else None,
        "bag_outcome_by_mean_proxy": outcome,
    }


def _normalize_tie_bands(
    tie_epsilon_m: float, tie_bands_m: Sequence[float] | None
) -> list[float]:
    requested = DEFAULT_TIE_BANDS_M if tie_bands_m is None else tie_bands_m
    normalized: set[float] = {tie_epsilon_m}
    for raw_band in requested:
        band = _finite_float(raw_band)
        if band is None or band < 0.0:
            raise ValueError("tie bands must be finite and non-negative")
        normalized.add(band)
    return sorted(normalized)


def _classify_delta(delta_m: float, tie_band_m: float) -> str:
    if delta_m > tie_band_m:
        return "improved"
    if delta_m < -tie_band_m:
        return "regressed"
    return "tied"


def _tie_band_sweep(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, float]],
    tie_bands_m: Sequence[float],
) -> list[dict[str, Any]]:
    bag_deltas = [
        float(row["return_abs_drift_improvement_mean_m"])
        for row in rows
        if _finite_float(row.get("return_abs_drift_improvement_mean_m")) is not None
    ]
    pair_deltas = [float(pair["improvement_m"]) for pair in pairs]
    sweep: list[dict[str, Any]] = []
    for tie_band_m in tie_bands_m:
        bag_counts = Counter(_classify_delta(delta, tie_band_m) for delta in bag_deltas)
        pair_counts = Counter(_classify_delta(delta, tie_band_m) for delta in pair_deltas)
        pair_count = len(pair_deltas)
        sweep.append(
            {
                "tie_band_m": tie_band_m,
                "metric_kind": "map_return_consistency_proxy_sensitivity",
                "ground_truth_available": False,
                "absolute_accuracy_claim_supported": False,
                "comparable_bag_count": len(bag_deltas),
                "improved_bag_count": bag_counts["improved"],
                "tied_bag_count": bag_counts["tied"],
                "regressed_bag_count": bag_counts["regressed"],
                "pooled_pair_count": pair_count,
                "improved_pair_count": pair_counts["improved"],
                "tied_pair_count": pair_counts["tied"],
                "regressed_pair_count": pair_counts["regressed"],
                "improved_pair_fraction": (
                    pair_counts["improved"] / pair_count if pair_count else None
                ),
                "tied_pair_fraction": (
                    pair_counts["tied"] / pair_count if pair_count else None
                ),
                "regressed_pair_fraction": (
                    pair_counts["regressed"] / pair_count if pair_count else None
                ),
            }
        )
    return sweep


def _prefix_replay(row: dict[str, Any], prefix: str, replay: Mapping[str, Any]) -> None:
    for key in (
        "replay_state",
        "status",
        "failure_reasons",
        "map_present",
        "map_png",
        "stationary_replay",
        "pose_graph_trajectory_count",
        "map_width_m",
        "map_height_m",
        "map_area_m2",
        "map_known_ratio",
        "map_occupied_ratio",
        "roughness_spatial_passage_count",
        "roughness_recovery_count",
    ):
        row[f"{prefix}_{key}"] = replay.get(key)


def _empty_proxy_fields() -> dict[str, Any]:
    return {
        "return_pair_count": 0,
        "return_pair_skipped_row_count": 0,
        "provided_improvement_mismatch_count": 0,
        "baseline_return_abs_drift_mean_m": None,
        "baseline_return_abs_drift_median_m": None,
        "baseline_return_abs_drift_p95_m": None,
        "post_return_abs_drift_mean_m": None,
        "post_return_abs_drift_median_m": None,
        "post_return_abs_drift_p95_m": None,
        "return_abs_drift_improvement_mean_m": None,
        "return_abs_drift_improvement_median_pair_m": None,
        "return_abs_drift_median_level_delta_m": None,
        "return_abs_drift_p95_delta_m": None,
        "improved_pair_count": 0,
        "tied_pair_count": 0,
        "regressed_pair_count": 0,
        "improved_pair_fraction": None,
        "bag_outcome_by_mean_proxy": "",
    }


def _comparison_state(
    *,
    unexpected_artifact: bool,
    input_usable: bool,
    baseline_complete: bool,
    post_complete: bool,
    proxy_available: bool,
    comparison_csv_present: bool,
    comparison_csv_trusted: bool,
) -> str:
    if unexpected_artifact:
        return "unexpected_artifact"
    if not input_usable:
        return "input_unusable"
    if not baseline_complete and not post_complete:
        return "baseline_and_post_incomplete"
    if not baseline_complete:
        return "baseline_incomplete"
    if not post_complete:
        return "post_incomplete"
    if comparison_csv_present and not comparison_csv_trusted:
        return "complete_replays_untrusted_comparison"
    if not proxy_available:
        return "complete_replays_no_return_proxy"
    return "complete_with_return_proxy"


def _macro_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "baseline_return_abs_drift_mean_m",
        "baseline_return_abs_drift_median_m",
        "baseline_return_abs_drift_p95_m",
        "post_return_abs_drift_mean_m",
        "post_return_abs_drift_median_m",
        "post_return_abs_drift_p95_m",
        "return_abs_drift_improvement_mean_m",
        "return_abs_drift_improvement_median_pair_m",
        "return_abs_drift_median_level_delta_m",
        "return_abs_drift_p95_delta_m",
    )
    summary: dict[str, Any] = {"bag_count": len(rows)}
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if _finite_float(row.get(field)) is not None
        ]
        summary[f"mean_of_bag_{field}"] = _mean(values)
    return summary


def collect_regression(
    *,
    baseline_root: Path,
    post_root: Path,
    comparison_root: Path,
    input_manifest: Path | None = None,
    revisit_csv_name: str = DEFAULT_REVISIT_CSV,
    tie_epsilon_m: float = 1e-6,
    tie_bands_m: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Collect all-bag evidence in memory without writing output artifacts."""
    baseline_root = baseline_root.expanduser().resolve()
    post_root = post_root.expanduser().resolve()
    comparison_root = comparison_root.expanduser().resolve()
    if tie_epsilon_m < 0.0 or not math.isfinite(tie_epsilon_m):
        raise ValueError("tie_epsilon_m must be finite and non-negative")
    normalized_tie_bands = _normalize_tie_bands(tie_epsilon_m, tie_bands_m)

    manifest_order, manifest_rows = load_input_manifest(input_manifest)
    manifest_authoritative = input_manifest is not None
    baseline_names = _directory_names(baseline_root)
    post_names = _directory_names(post_root)
    comparison_names = _comparison_names(comparison_root)
    discovered = baseline_names | post_names | comparison_names
    extras = sorted(discovered - set(manifest_order))
    bag_names = [*manifest_order, *extras]
    rows: list[dict[str, Any]] = []
    pooled_pairs: list[dict[str, float]] = []

    for bag in bag_names:
        manifest_member = bag in manifest_rows
        unexpected_artifact = manifest_authoritative and not manifest_member
        unexpected_locations = [
            location
            for location, names in (
                ("baseline", baseline_names),
                ("post", post_names),
                ("comparison", comparison_names),
            )
            if bag in names
        ]
        manifest = manifest_rows.get(
            bag,
            {
                "source_root": "",
                "input_status": (
                    "unexpected_artifact" if unexpected_artifact else "discovered"
                ),
                "input_usable": int(not unexpected_artifact),
                "input_duration_s": None,
                "input_scan_laserscan_count": None,
                "input_imu_count": None,
                "input_wheel_odom_count": None,
            },
        )
        baseline = inspect_replay(baseline_root, bag)
        post = inspect_replay(post_root, bag)
        baseline_complete = bool(baseline["replay_complete"])
        post_complete = bool(post["replay_complete"])
        replay_pair_complete = baseline_complete and post_complete
        input_usable = bool(manifest["input_usable"])

        comparison_dir = comparison_root / bag
        comparison_csv = comparison_dir / revisit_csv_name
        map_artifact = comparison_dir / "map_comparison_side_by_side.png"
        if unexpected_artifact:
            comparison_ineligible_reason = "unexpected_artifact_not_in_manifest"
        elif not input_usable:
            comparison_ineligible_reason = "input_unusable"
        elif not replay_pair_complete:
            comparison_ineligible_reason = "incomplete_replay_pair"
        else:
            comparison_ineligible_reason = ""
        source_artifacts = _comparison_source_artifacts(
            baseline_root, post_root, bag
        )
        csv_assessment = assess_comparison_artifact(
            comparison_csv,
            source_artifacts,
            ineligible_reason=comparison_ineligible_reason,
        )
        map_assessment = assess_comparison_artifact(
            map_artifact,
            source_artifacts,
            ineligible_reason=comparison_ineligible_reason,
        )
        proxy_pairs: list[dict[str, float]] = []
        proxy_diagnostics = {
            "csv_present": int(comparison_csv.is_file()),
            "csv_error": "",
            "content_trusted": 0,
            "skipped_row_count": 0,
            "provided_improvement_mismatch_count": 0,
            "pair_identity_columns": "",
            "pair_identity_validation_status": "not_evaluated",
            "pair_identity_invalid_row_count": 0,
            "pair_identity_duplicate_row_count": 0,
        }
        if not comparison_ineligible_reason:
            proxy_pairs, proxy_diagnostics = load_return_proxy_pairs(comparison_csv)
            stationary_pair = bool(baseline.get("stationary_replay")) and bool(
                post.get("stationary_replay")
            )
            if (
                csv_assessment["trusted"]
                and stationary_pair
                and proxy_diagnostics["csv_error"] == "empty_csv_without_header"
            ):
                proxy_diagnostics["content_trusted"] = 1
                proxy_diagnostics["csv_error"] = "no_valid_proxy_rows"
            if csv_assessment["trusted"] and not proxy_diagnostics["content_trusted"]:
                csv_assessment.update(
                    trusted=0,
                    status=(
                        "invalid_pair_identity"
                        if proxy_diagnostics["pair_identity_validation_status"] == "invalid"
                        else "invalid_csv_content"
                    ),
                    reasons=proxy_diagnostics["csv_error"] or "untrusted_csv_content",
                )
        elif not input_usable:
            proxy_diagnostics["csv_error"] = (
                "not_evaluated_unexpected_artifact"
                if unexpected_artifact
                else "not_evaluated_input_unusable"
            )
        else:
            proxy_diagnostics["csv_error"] = "not_evaluated_incomplete_replay"
        comparison_csv_trusted = bool(csv_assessment["trusted"])
        proxy_available = comparison_csv_trusted and bool(proxy_pairs)

        reasons: list[str] = []
        if unexpected_artifact:
            reasons.append("unexpected_artifact_not_in_manifest")
        elif not input_usable:
            reasons.append(f"input_unusable:{manifest['input_status']}")
        if not baseline_complete:
            reasons.append(f"baseline:{baseline['failure_reasons']}")
        if not post_complete:
            reasons.append(f"post:{post['failure_reasons']}")
        if input_usable and replay_pair_complete and not proxy_available:
            if not comparison_csv_trusted:
                reasons.append(
                    "comparison_csv_untrusted:"
                    f"{csv_assessment['status']}:{csv_assessment['reasons']}"
                )
            else:
                reasons.append(f"return_proxy:{proxy_diagnostics['csv_error']}")

        row: dict[str, Any] = {
            "bag": bag,
            **manifest,
            "manifest_member": int(manifest_member),
            "unexpected_artifact": int(unexpected_artifact),
            "unexpected_artifact_locations": ",".join(unexpected_locations),
            "comparison_state": _comparison_state(
                unexpected_artifact=unexpected_artifact,
                input_usable=input_usable,
                baseline_complete=baseline_complete,
                post_complete=post_complete,
                proxy_available=proxy_available,
                comparison_csv_present=bool(csv_assessment["present"]),
                comparison_csv_trusted=comparison_csv_trusted,
            ),
            "replay_pair_complete": int(replay_pair_complete),
            "proxy_available": int(proxy_available),
            "exclusion_reasons": ";".join(reasons),
            "metric_kind": "map_return_consistency_proxy",
            "ground_truth_available": 0,
            "absolute_accuracy_claim_supported": 0,
            "comparison_csv_present": proxy_diagnostics["csv_present"],
            "comparison_csv": str(comparison_csv) if comparison_csv.is_file() else "",
            "comparison_csv_error": proxy_diagnostics["csv_error"],
            "comparison_csv_trusted": int(comparison_csv_trusted),
            "comparison_csv_trust_status": csv_assessment["status"],
            "comparison_csv_trust_reasons": csv_assessment["reasons"],
            "comparison_map_artifact_present": map_assessment["present"],
            "comparison_map_artifact_trusted": map_assessment["trusted"],
            "comparison_map_artifact_trust_status": map_assessment["status"],
            "comparison_map_artifact_trust_reasons": map_assessment["reasons"],
            "comparison_map_artifact": str(map_artifact) if map_artifact.is_file() else "",
            "pair_identity_columns": proxy_diagnostics["pair_identity_columns"],
            "pair_identity_validation_status": proxy_diagnostics[
                "pair_identity_validation_status"
            ],
            "pair_identity_invalid_row_count": proxy_diagnostics[
                "pair_identity_invalid_row_count"
            ],
            "pair_identity_duplicate_row_count": proxy_diagnostics[
                "pair_identity_duplicate_row_count"
            ],
        }
        _prefix_replay(row, "baseline", baseline)
        _prefix_replay(row, "post", post)
        row.update(
            map_area_delta_m2=_difference(post["map_area_m2"], baseline["map_area_m2"]),
            map_known_ratio_delta=_difference(
                post["map_known_ratio"], baseline["map_known_ratio"]
            ),
            map_occupied_ratio_delta=_difference(
                post["map_occupied_ratio"], baseline["map_occupied_ratio"]
            ),
        )
        row.update(_empty_proxy_fields())
        row["return_pair_skipped_row_count"] = proxy_diagnostics["skipped_row_count"]
        row["provided_improvement_mismatch_count"] = proxy_diagnostics[
            "provided_improvement_mismatch_count"
        ]
        if proxy_available:
            row.update(summarize_pairs(proxy_pairs, tie_epsilon_m))
            pooled_pairs.extend(proxy_pairs)
        rows.append(row)

    analysis_rows = [row for row in rows if not row["unexpected_artifact"]]
    comparable_rows = [row for row in analysis_rows if row["proxy_available"]]
    state_counts = Counter(str(row["comparison_state"]) for row in analysis_rows)
    reported_state_counts = Counter(str(row["comparison_state"]) for row in rows)
    outcome_counts = Counter(
        str(row["bag_outcome_by_mean_proxy"])
        for row in comparable_rows
        if row["bag_outcome_by_mean_proxy"]
    )
    incomplete = [
        str(row["bag"])
        for row in analysis_rows
        if row["input_usable"] and not row["replay_pair_complete"]
    ]
    complete_without_proxy = [
        str(row["bag"])
        for row in analysis_rows
        if row["input_usable"]
        and row["replay_pair_complete"]
        and not row["proxy_available"]
    ]
    unexpected_artifacts = [
        {
            "bag": str(row["bag"]),
            "locations": str(row["unexpected_artifact_locations"]).split(","),
        }
        for row in rows
        if row["unexpected_artifact"]
    ]
    untrusted_comparison_csv_bags = [
        {
            "bag": str(row["bag"]),
            "status": str(row["comparison_csv_trust_status"]),
            "reasons": str(row["comparison_csv_trust_reasons"]),
        }
        for row in analysis_rows
        if row["comparison_csv_present"] and not row["comparison_csv_trusted"]
    ]
    untrusted_comparison_map_bags = [
        {
            "bag": str(row["bag"]),
            "status": str(row["comparison_map_artifact_trust_status"]),
            "reasons": str(row["comparison_map_artifact_trust_reasons"]),
        }
        for row in analysis_rows
        if row["comparison_map_artifact_present"]
        and not row["comparison_map_artifact_trusted"]
    ]
    tie_band_sweep = _tie_band_sweep(
        comparable_rows, pooled_pairs, normalized_tie_bands
    )
    summary = {
        "schema_version": 2,
        "metric_contract": {
            "metric_kind": "map_return_consistency_proxy",
            "ground_truth_available": False,
            "absolute_accuracy_claim_supported": False,
            "primary_proxy_definition": (
                "For odometry-selected outbound/return revisit pairs, abs_drift_m is "
                "the absolute difference between SLAM trajectory separation and wheel-odometry "
                "separation. Lower values indicate better return consistency relative to odometry."
            ),
            "limitations": [
                "The proxy is not ground-truth ATE, RPE, or absolute map accuracy.",
                "Wheel odometry selects revisit pairs and is itself subject to drift and slip.",
                (
                    "Map dimensions and occupancy ratios are descriptive footprint metrics, "
                    "not quality scores."
                ),
                (
                    "Repeated revisit pairs within a bag are correlated; macro and micro "
                    "summaries are both reported."
                ),
            ],
            "positive_delta_interpretation": (
                "Positive baseline-minus-post proxy deltas favor post."
            ),
            "bag_outcome_rule": (
                f"Mean per-pair proxy improvement compared with a +/-{tie_epsilon_m:g} m tie band."
            ),
            "tie_band_sweep_interpretation": (
                "A sensitivity analysis of proxy outcome labels across practical tie bands; "
                "it does not add ground truth or support absolute-accuracy claims."
            ),
            "trajectory_rendering_policy": TRAJECTORY_RENDERING_POLICY,
            "trajectory_plots_generated": False,
        },
        "inputs": {
            "baseline_root": str(baseline_root),
            "post_root": str(post_root),
            "comparison_root": str(comparison_root),
            "input_manifest": str(input_manifest.resolve()) if input_manifest else "",
            "revisit_csv_name": revisit_csv_name,
            "primary_tie_epsilon_m": tie_epsilon_m,
            "tie_bands_m": normalized_tie_bands,
            "manifest_authoritative": manifest_authoritative,
        },
        "counts": {
            "all_bag_count": len(analysis_rows),
            "reported_row_count": len(rows),
            "unexpected_artifact_bag_count": len(unexpected_artifacts),
            "input_usable_bag_count": sum(
                bool(row["input_usable"]) for row in analysis_rows
            ),
            "input_unusable_bag_count": sum(
                not bool(row["input_usable"]) for row in analysis_rows
            ),
            "baseline_complete_bag_count": sum(
                row["baseline_replay_state"] == "complete" for row in analysis_rows
            ),
            "post_complete_bag_count": sum(
                row["post_replay_state"] == "complete" for row in analysis_rows
            ),
            "paired_replay_complete_bag_count": sum(
                bool(row["replay_pair_complete"]) for row in analysis_rows
            ),
            "return_proxy_bag_count": len(comparable_rows),
            "pooled_return_pair_count": len(pooled_pairs),
            "trusted_comparison_csv_bag_count": sum(
                bool(row["comparison_csv_trusted"]) for row in analysis_rows
            ),
            "trusted_comparison_map_bag_count": sum(
                bool(row["comparison_map_artifact_trusted"])
                for row in analysis_rows
            ),
            "comparison_state_counts": dict(sorted(state_counts.items())),
            "reported_comparison_state_counts": dict(
                sorted(reported_state_counts.items())
            ),
            "bag_outcome_by_mean_proxy_counts": dict(sorted(outcome_counts.items())),
        },
        "unexpected_artifacts": unexpected_artifacts,
        "untrusted_comparison_csv_bags": untrusted_comparison_csv_bags,
        "untrusted_comparison_map_bags": untrusted_comparison_map_bags,
        "incomplete_usable_bags": incomplete,
        "complete_replays_without_return_proxy": complete_without_proxy,
        "aggregate_return_consistency_proxy": {
            "macro_equal_bag_weight": _macro_summary(comparable_rows),
            "micro_equal_pair_weight": summarize_pairs(pooled_pairs, tie_epsilon_m),
        },
        "tie_band_sweep": tie_band_sweep,
    }
    return {"summary": summary, "bags": rows, "pairs": pooled_pairs}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in CSV_COLUMNS
                }
            )


def _write_tie_band_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=TIE_BAND_CSV_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key) is None else row.get(key)
                    for key in TIE_BAND_CSV_COLUMNS
                }
            )


def _report_number(value: Any, digits: int = 6) -> str:
    converted = _finite_float(value)
    return f"{converted:.{digits}f}" if converted is not None else "not available"


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    counts = summary["counts"]
    aggregate = summary["aggregate_return_consistency_proxy"]
    macro = aggregate["macro_equal_bag_weight"]
    micro = aggregate["micro_equal_pair_weight"]
    lines = [
        "# 2D SLAM Regression Aggregate",
        "",
        "This report uses a map return-consistency proxy. It is not ground-truth "
        "ATE, RPE, or absolute map accuracy, and it does not support absolute-accuracy claims.",
        "",
        "## Coverage and trust",
        "",
        f"- Analysis bags (manifest-authoritative when provided): {counts['all_bag_count']}",
        f"- Reported unexpected artifact bags: {counts['unexpected_artifact_bag_count']}",
        f"- Paired complete replays: {counts['paired_replay_complete_bag_count']}",
        f"- Trusted comparison CSV bags: {counts['trusted_comparison_csv_bag_count']}",
        f"- Bags contributing proxy metrics: {counts['return_proxy_bag_count']}",
        f"- Pooled trusted revisit pairs: {counts['pooled_return_pair_count']}",
        "",
        "Only comparisons at least as new as their baseline/post replay results, maps, "
        "and trajectory CSV sources contribute to metrics.",
        "",
        "## Artifact exclusions",
        "",
    ]
    unexpected = summary.get("unexpected_artifacts", [])
    untrusted_csv = summary.get("untrusted_comparison_csv_bags", [])
    untrusted_maps = summary.get("untrusted_comparison_map_bags", [])
    if not unexpected and not untrusted_csv and not untrusted_maps:
        lines.extend(["- None.", ""])
    else:
        for artifact in unexpected:
            lines.append(
                f"- Unexpected `{artifact['bag']}`: "
                f"locations={','.join(artifact['locations'])}"
            )
        for artifact in untrusted_csv:
            lines.append(
                f"- Untrusted CSV `{artifact['bag']}`: status={artifact['status']}; "
                f"reasons={artifact['reasons']}"
            )
        for artifact in untrusted_maps:
            lines.append(
                f"- Untrusted map `{artifact['bag']}`: status={artifact['status']}; "
                f"reasons={artifact['reasons']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Raw proxy metrics",
            "",
            "Positive baseline-minus-post deltas favor post. These continuous values are "
            "preserved independently of the tie-band sensitivity labels.",
            "",
            "| Weighting | Baseline mean (m) | Post mean (m) | Improvement (m) |",
            "|---|---:|---:|---:|",
            (
                "| Equal bag | "
                f"{_report_number(macro.get('mean_of_bag_baseline_return_abs_drift_mean_m'))} | "
                f"{_report_number(macro.get('mean_of_bag_post_return_abs_drift_mean_m'))} | "
                f"{_report_number(macro.get('mean_of_bag_return_abs_drift_improvement_mean_m'))} |"
            ),
            (
                "| Equal pair | "
                f"{_report_number(micro.get('baseline_return_abs_drift_mean_m'))} | "
                f"{_report_number(micro.get('post_return_abs_drift_mean_m'))} | "
                f"{_report_number(micro.get('return_abs_drift_improvement_mean_m'))} |"
            ),
            "",
            "## Tie-band sensitivity",
            "",
            "Outcome labels below are sensitivity checks on the same proxy, "
            "not new accuracy metrics.",
            "",
            "| Tie band (m) | Bags improved / tied / regressed | "
            "Pairs improved / tied / regressed |",
            "|---:|---:|---:|",
        ]
    )
    for sweep in summary["tie_band_sweep"]:
        lines.append(
            f"| {sweep['tie_band_m']:.6g} | "
            f"{sweep['improved_bag_count']} / {sweep['tied_bag_count']} / "
            f"{sweep['regressed_bag_count']} | "
            f"{sweep['improved_pair_count']} / {sweep['tied_pair_count']} / "
            f"{sweep['regressed_pair_count']} |"
        )
    lines.extend(
        [
            "",
            "## Rendering contract",
            "",
            f"Trajectory policy: `{TRAJECTORY_RENDERING_POLICY}`. This aggregate does not "
            "generate connected trajectory segments.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-slam-2d-aggregate")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    return mpimg, plt


def _plot_proxy_by_bag(rows: Sequence[Mapping[str, Any]], output: Path) -> bool:
    comparable = [row for row in rows if row.get("proxy_available")]
    if not comparable:
        return False
    _, plt = _matplotlib()
    width = max(11.0, min(26.0, 5.0 + 0.55 * len(comparable)))
    figure, axes = plt.subplots(2, 1, figsize=(width, 10), sharex=True)
    x_values = list(range(len(comparable)))
    labels = [str(row["bag"]).removeprefix("ty_bag_") for row in comparable]
    axes[0].scatter(
        x_values,
        [row["baseline_return_abs_drift_mean_m"] for row in comparable],
        marker="o",
        s=42,
        label="Baseline mean",
        color="#2f5597",
    )
    axes[0].scatter(
        x_values,
        [row["post_return_abs_drift_mean_m"] for row in comparable],
        marker="x",
        s=55,
        label="Post mean",
        color="#c84b31",
    )
    axes[0].set_ylabel("Return-consistency proxy (m, lower is better)")
    axes[0].set_title("Per-bag return consistency; not ground-truth accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    deltas = [row["return_abs_drift_improvement_mean_m"] for row in comparable]
    colors = ["#2a9d8f" if value >= 0.0 else "#d1495b" for value in deltas]
    axes[1].bar(x_values, deltas, color=colors)
    axes[1].axhline(0.0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Baseline - post mean proxy (m)")
    axes[1].set_xlabel("Bag")
    axes[1].set_xticks(x_values, labels, rotation=60, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_proxy_distribution(pairs: Sequence[Mapping[str, float]], output: Path) -> bool:
    if not pairs:
        return False
    _, plt = _matplotlib()
    baseline = [pair["baseline_abs_drift_m"] for pair in pairs]
    post = [pair["post_abs_drift_m"] for pair in pairs]
    figure, axis = plt.subplots(figsize=(8, 6))
    box = axis.boxplot(
        [baseline, post],
        labels=["Baseline", "Post"],
        showfliers=False,
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ("#9bb7d4", "#e6a57e")):
        patch.set_facecolor(color)
    axis.set_ylabel("Absolute return-consistency proxy (m, lower is better)")
    axis.set_title("Pooled revisit-pair distribution; not ground-truth accuracy")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_coverage(rows: Sequence[Mapping[str, Any]], output: Path) -> bool:
    if not rows:
        return False
    _, plt = _matplotlib()
    counts = Counter(str(row["comparison_state"]) for row in rows)
    labels = list(sorted(counts))
    values = [counts[label] for label in labels]
    colors = [
        "#2a9d8f" if label == "complete_with_return_proxy" else "#8d99ae"
        for label in labels
    ]
    figure, axis = plt.subplots(figsize=(11, max(4.0, 0.65 * len(labels) + 2.0)))
    positions = list(range(len(labels)))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Bag count")
    axis.set_title("All-bag replay and comparison coverage")
    for position, value in zip(positions, values):
        axis.text(value + 0.05, position, str(value), va="center")
    axis.set_xlim(0.0, max(values) * 1.15 + 0.5)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_map_contact_sheets(
    rows: Sequence[Mapping[str, Any]], output_dir: Path, page_size: int
) -> list[Path]:
    trusted = [
        row
        for row in rows
        if row.get("comparison_map_artifact_trusted")
        and Path(str(row["comparison_map_artifact"])).is_file()
    ]
    if not trusted:
        return []
    if page_size <= 0:
        raise ValueError("contact_sheet_page_size must be positive")
    mpimg, plt = _matplotlib()
    columns = 2
    outputs: list[Path] = []
    for page_index, start in enumerate(range(0, len(trusted), page_size), start=1):
        page = trusted[start:start + page_size]
        row_count = math.ceil(len(page) / columns)
        figure, axes = plt.subplots(row_count, columns, figsize=(18, 6.5 * row_count))
        axes_flat = list(getattr(axes, "flat", [axes]))
        for axis, row in zip(axes_flat, page):
            axis.imshow(mpimg.imread(str(row["comparison_map_artifact"])))
            delta = row.get("return_abs_drift_improvement_mean_m")
            delta_text = (
                f"mean proxy delta={delta:+.3f} m"
                if delta is not None
                else "proxy unavailable"
            )
            axis.set_title(f"{row['bag']} | {delta_text}", fontsize=10)
            axis.axis("off")
        for axis in axes_flat[len(page):]:
            axis.axis("off")
        figure.suptitle("Baseline / post map evidence (proxy metrics are not absolute accuracy)")
        figure.tight_layout()
        output = output_dir / f"04_map_comparison_contact_sheet_{page_index:02d}.png"
        figure.savefig(output, dpi=130, bbox_inches="tight")
        plt.close(figure)
        outputs.append(output)
    return outputs


def write_outputs(
    aggregation: Mapping[str, Any],
    output_dir: Path,
    *,
    create_plots: bool = True,
    contact_sheet_page_size: int = 12,
) -> list[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(aggregation["bags"])
    summary = dict(aggregation["summary"])
    csv_path = output_dir / "all_bags_summary.csv"
    json_path = output_dir / "all_bags_summary.json"
    tie_band_csv_path = output_dir / "tie_band_sweep.csv"
    report_path = output_dir / "aggregate_report.md"
    _write_csv(csv_path, rows)
    _write_tie_band_csv(tie_band_csv_path, summary["tie_band_sweep"])
    _write_report(report_path, summary)
    json_path.write_text(
        json.dumps({**summary, "bags": rows}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    outputs = [csv_path, json_path, tie_band_csv_path, report_path]
    if not create_plots:
        return outputs
    plot_specs = (
        (_plot_proxy_by_bag, rows, output_dir / "01_return_consistency_proxy_by_bag.png"),
        (
            _plot_proxy_distribution,
            list(aggregation["pairs"]),
            output_dir / "02_return_consistency_proxy_distribution.png",
        ),
        (_plot_coverage, rows, output_dir / "03_replay_comparison_coverage.png"),
    )
    for function, values, destination in plot_specs:
        if function(values, destination):
            outputs.append(destination)
    outputs.extend(
        _plot_map_contact_sheets(rows, output_dir, contact_sheet_page_size)
    )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--post-root", required=True, type=Path)
    parser.add_argument("--comparison-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help="Optional input_bag_manifest.csv; auto-detected above variant roots when omitted.",
    )
    parser.add_argument("--revisit-csv-name", default=DEFAULT_REVISIT_CSV)
    parser.add_argument("--tie-epsilon-m", type=float, default=1e-6)
    parser.add_argument(
        "--tie-band-m",
        dest="tie_bands_m",
        action="append",
        type=float,
        help=(
            "Repeatable practical tie band for sensitivity output. When omitted, "
            "0.01, 0.025, 0.05, and 0.1 m are used; the primary tie epsilon is always included."
        ),
    )
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fail-on-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.input_manifest
    if manifest is None:
        manifest = find_input_manifest(args.baseline_root, args.post_root)
    aggregation = collect_regression(
        baseline_root=args.baseline_root,
        post_root=args.post_root,
        comparison_root=args.comparison_root,
        input_manifest=manifest,
        revisit_csv_name=args.revisit_csv_name,
        tie_epsilon_m=args.tie_epsilon_m,
        tie_bands_m=args.tie_bands_m,
    )
    outputs = write_outputs(
        aggregation,
        args.output_dir,
        create_plots=not args.no_plots,
        contact_sheet_page_size=args.contact_sheet_page_size,
    )
    counts = aggregation["summary"]["counts"]
    print(
        "all_bags={all_bag_count} paired_complete={paired_replay_complete_bag_count} "
        "proxy_bags={return_proxy_bag_count} pooled_pairs={pooled_return_pair_count}".format(
            **counts
        )
    )
    print("outputs=" + ",".join(str(path) for path in outputs))
    if args.fail_on_incomplete and aggregation["summary"]["incomplete_usable_bags"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
