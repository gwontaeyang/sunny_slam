#!/usr/bin/env python3
"""Derive a conservative shadow calibration from exact near-chain controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


CONTROL_MODES = {"intervention_off", "adaptive_memory_observe"}
CONTROL_PROVENANCE_FIELDS = (
    "lossless_adapter_sha256",
    "runtime_slam_node_sha256",
    "runtime_cmake_cache_sha256",
    "replay_pose_graph_exporter_sha256",
    "slam_config_sha256",
    "exact_edge_exporter_sha256",
)
REQUIRED_METRICS = (
    "normalized_response",
    "translation_disagreement_m",
    "yaw_disagreement_rad",
    "xy_covariance_max_eigenvalue_m2",
    "covariance_yawyaw",
    "memory_strength",
    "final_covariance_weighted_residual_sq",
    "baseline_threshold",
)
REJECTION_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument("--minimum-controls", type=int, default=5)
    parser.add_argument("--minimum-active-candidates", type=int, default=50)
    parser.add_argument("--minimum-active-datasets", type=int, default=2)
    parser.add_argument("--minimum-selected-candidates", type=int, default=3)
    parser.add_argument("--minimum-residual-enrichment", type=float, default=1.5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.expanduser().open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def dataset_key(row: dict[str, str]) -> str:
    label = row.get("label", "").strip()
    bag = row.get("bag", "").strip()
    return f"{label}/{bag}" if bag else label


def quantile(values: Iterable[float], fraction: float) -> float:
    clean = np.asarray(
        [value for value in values if math.isfinite(value)],
        dtype=float,
    )
    if clean.size == 0:
        return math.nan
    return float(np.quantile(clean, fraction, method="linear"))


def unit_ramp(value: float, low: float, high: float) -> float:
    if not all(math.isfinite(item) for item in (value, low, high)) or high <= low:
        return math.nan
    return min(max((value - low) / (high - low), 0.0), 1.0)


def select_control_summaries(
    rows: list[dict[str, str]], labels: Iterable[str]
) -> list[dict[str, str]]:
    requested = {label.strip() for label in labels if label.strip()}
    selected = [row for row in rows if not requested or row.get("label") in requested]
    found = {row.get("label", "") for row in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError("missing control summaries: " + ", ".join(missing))
    dataset_keys = [dataset_key(row) for row in selected]
    duplicate = sorted(
        key for key in set(dataset_keys) if dataset_keys.count(key) != 1
    )
    if duplicate:
        raise ValueError("control datasets are not unique: " + ", ".join(duplicate))
    invalid = [
        dataset_key(row) for row in selected
        if row.get("mode") not in CONTROL_MODES
        or not is_true(row.get("log_integrity_valid"))
        or not is_true(row.get("runtime_parameter_match"))
        or not is_true(row.get("imu_delivery_complete"))
        or not is_true(row.get("scan_delivery_complete"))
        or not is_true(row.get("queue_drain_complete"))
        or not is_true(row.get("exact_edge_evidence_valid"))
        or row.get("pose_graph_coverage_status") != "complete"
    ]
    if invalid:
        raise ValueError("invalid control replay summaries: " + ", ".join(invalid))
    for label in {row.get("label", "") for row in selected}:
        label_rows = [row for row in selected if row.get("label", "") == label]
        for field in CONTROL_PROVENANCE_FIELDS:
            values = {row.get(field, "") for row in label_rows}
            if len(values) != 1 or any(len(value) != 64 for value in values):
                raise ValueError(
                    f"mixed or missing control provenance for {label}: {field}"
                )
    return selected


def candidate_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        dataset_key(row),
        row.get("mapper_generation", ""),
        row.get("stamp_ns", ""),
        row.get("candidate_index", ""),
        row.get("source_unique_id", ""),
        row.get("target_unique_id", ""),
    )


def select_candidates(
    rows: list[dict[str, str]], summaries: list[dict[str, str]]
) -> list[dict[str, str]]:
    summary_by_dataset = {dataset_key(row): row for row in summaries}
    labels = {row.get("label", "") for row in summaries}
    stale = sorted(
        {
            dataset_key(row)
            for row in rows
            if row.get("label") in labels
            and dataset_key(row) not in summary_by_dataset
        }
    )
    if stale:
        raise ValueError(
            "candidate rows do not belong to selected controls: "
            + ", ".join(stale)
    )
    for key, summary in summary_by_dataset.items():
        observed = sum(dataset_key(row) == key for row in rows)
        expected_value = to_float(summary.get("exact_candidate_count"))
        if not math.isfinite(expected_value) or expected_value < 0.0:
            raise ValueError(f"invalid exact candidate count for {key}")
        expected = int(expected_value)
        if observed != expected:
            raise ValueError(
                f"candidate count mismatch for {key}: {observed} != {expected}"
            )

    selected = []
    for row in rows:
        summary = summary_by_dataset.get(dataset_key(row))
        if summary is None:
            continue
        if row.get("mode") not in CONTROL_MODES:
            continue
        if not is_true(row.get("instrumentation_valid")):
            continue
        if not is_true(row.get("chain_size_accepted")):
            continue
        if not is_true(row.get("edge_inserted")):
            continue
        if row.get("final_residual_status") != "ok":
            continue
        if row.get("final_residual_source") != "serialized_pose_graph_edge":
            continue
        if not is_true(row.get("final_edge_evidence_valid")):
            continue
        if len(row.get("final_edge_exporter_sha256", "")) != 64:
            continue
        if row.get("final_edge_exporter_sha256") != summary.get(
            "exact_edge_exporter_sha256"
        ):
            continue
        if not all(math.isfinite(to_float(row.get(field))) for field in REQUIRED_METRICS):
            continue
        selected.append(row)
    identities = [candidate_identity(row) for row in selected]
    duplicate = sorted(
        identity for identity in set(identities) if identities.count(identity) != 1
    )
    if duplicate:
        raise ValueError(
            "duplicate exact candidates: "
            + ", ".join("/".join(identity) for identity in duplicate)
        )
    return selected


def derive_quality_bounds(rows: list[dict[str, str]]) -> dict[str, float]:
    response = [to_float(row["normalized_response"]) for row in rows]
    translation = [to_float(row["translation_disagreement_m"]) for row in rows]
    yaw = [abs(to_float(row["yaw_disagreement_rad"])) for row in rows]
    xy_covariance = [
        to_float(row["xy_covariance_max_eigenvalue_m2"]) for row in rows
    ]
    yaw_covariance = [to_float(row["covariance_yawyaw"]) for row in rows]
    return {
        "response_bad": quantile(response, 0.05),
        "response_good": quantile(response, 0.25),
        "translation_good_m": quantile(translation, 0.50),
        "translation_bad_m": quantile(translation, 0.95),
        "yaw_good_rad": quantile(yaw, 0.50),
        "yaw_bad_rad": quantile(yaw, 0.95),
        "xy_covariance_good_m2": quantile(xy_covariance, 0.50),
        "xy_covariance_bad_m2": quantile(xy_covariance, 0.95),
        "yaw_covariance_good_rad2": quantile(yaw_covariance, 0.50),
        "yaw_covariance_bad_rad2": quantile(yaw_covariance, 0.95),
    }


def bounds_are_valid(bounds: dict[str, float]) -> bool:
    pairs = (
        ("response_bad", "response_good"),
        ("translation_good_m", "translation_bad_m"),
        ("yaw_good_rad", "yaw_bad_rad"),
        ("xy_covariance_good_m2", "xy_covariance_bad_m2"),
        ("yaw_covariance_good_rad2", "yaw_covariance_bad_rad2"),
    )
    return all(
        math.isfinite(bounds[low])
        and math.isfinite(bounds[high])
        and bounds[high] > bounds[low]
        for low, high in pairs
    )


def candidate_risk(row: dict[str, str], bounds: dict[str, float]) -> float:
    response_risk = 1.0 - unit_ramp(
        to_float(row["normalized_response"]),
        bounds["response_bad"],
        bounds["response_good"],
    )
    disagreement_risk = max(
        unit_ramp(
            to_float(row["translation_disagreement_m"]),
            bounds["translation_good_m"],
            bounds["translation_bad_m"],
        ),
        unit_ramp(
            abs(to_float(row["yaw_disagreement_rad"])),
            bounds["yaw_good_rad"],
            bounds["yaw_bad_rad"],
        ),
    )
    covariance_risk = max(
        unit_ramp(
            to_float(row["xy_covariance_max_eigenvalue_m2"]),
            bounds["xy_covariance_good_m2"],
            bounds["xy_covariance_bad_m2"],
        ),
        unit_ramp(
            to_float(row["covariance_yawyaw"]),
            bounds["yaw_covariance_good_rad2"],
            bounds["yaw_covariance_bad_rad2"],
        ),
    )
    quality_risk = response_risk * max(disagreement_risk, covariance_risk)
    return min(max(to_float(row["memory_strength"]) * quality_risk, 0.0), 1.0)


def high_residual_flags(rows: list[dict[str, str]]) -> list[bool]:
    thresholds = {
        key: quantile(
            (
                to_float(row["final_covariance_weighted_residual_sq"])
                for row in rows if dataset_key(row) == key
            ),
            0.90,
        )
        for key in {dataset_key(row) for row in rows}
    }
    return [
        to_float(row["final_covariance_weighted_residual_sq"])
        >= thresholds[dataset_key(row)]
        for row in rows
    ]


def build_risk_sweep(
    rows: list[dict[str, str]], risks: list[float]
) -> list[dict[str, Any]]:
    flags = high_residual_flags(rows)
    base_prevalence = sum(flags) / len(flags) if flags else math.nan
    result = []
    for target_fraction in REJECTION_FRACTIONS:
        threshold = quantile(risks, 1.0 - target_fraction)
        selected = [
            index for index, risk in enumerate(risks)
            if risk > 0.0 and risk >= threshold
        ]
        selected_high = sum(flags[index] for index in selected)
        precision = selected_high / len(selected) if selected else math.nan
        enrichment = (
            precision / base_prevalence
            if selected and base_prevalence > 0.0 else math.nan
        )
        result.append(
            {
                "target_rejection_fraction": target_fraction,
                "risk_threshold": threshold,
                "selected_candidate_count": len(selected),
                "selected_dataset_count": len(
                    {dataset_key(rows[index]) for index in selected}
                ),
                "actual_rejection_fraction": (
                    len(selected) / len(rows) if rows else math.nan
                ),
                "high_residual_precision": precision,
                "high_residual_base_prevalence": base_prevalence,
                "high_residual_enrichment": enrichment,
            }
        )
    return result


def select_sweep_row(
    sweep: list[dict[str, Any]],
    minimum_candidates: int,
    minimum_datasets: int,
    minimum_enrichment: float,
) -> dict[str, Any] | None:
    for row in sweep:
        if (
            row["selected_candidate_count"] >= minimum_candidates
            and row["selected_dataset_count"] >= minimum_datasets
            and math.isfinite(row["high_residual_enrichment"])
            and row["high_residual_enrichment"] >= minimum_enrichment
        ):
            return row
    return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def scored_candidate_rows(
    rows: list[dict[str, str]],
    risks: list[float],
    selected_threshold: float,
) -> list[dict[str, Any]]:
    flags = high_residual_flags(rows)
    result = []
    for row, risk, high_residual in zip(rows, risks, flags):
        result.append(
            {
                "label": row.get("label", ""),
                "bag": row.get("bag", ""),
                "stamp_ns": row.get("stamp_ns", ""),
                "candidate_index": row.get("candidate_index", ""),
                "source_state_id": row.get("source_state_id", ""),
                "target_state_id": row.get("target_state_id", ""),
                "memory_strength": row.get("memory_strength", ""),
                "normalized_response": row.get("normalized_response", ""),
                "translation_disagreement_m": row.get(
                    "translation_disagreement_m", ""
                ),
                "yaw_disagreement_rad": row.get("yaw_disagreement_rad", ""),
                "xy_covariance_max_eigenvalue_m2": row.get(
                    "xy_covariance_max_eigenvalue_m2", ""
                ),
                "yaw_covariance_rad2": row.get("covariance_yawyaw", ""),
                "final_covariance_weighted_residual_sq": row.get(
                    "final_covariance_weighted_residual_sq", ""
                ),
                "high_residual_diagnostic": int(high_residual),
                "combined_shadow_risk": risk,
                "nominated_for_paired_replay": int(
                    math.isfinite(selected_threshold)
                    and risk > 0.0
                    and risk >= selected_threshold
                ),
            }
        )
    return result


def plot_diagnostics(
    rows: list[dict[str, str]], risks: list[float], path: Path
) -> None:
    residuals = np.asarray(
        [to_float(row["final_covariance_weighted_residual_sq"]) for row in rows]
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].hist(risks, bins=30, color="#287D8E", alpha=0.85)
    axes[0].set_xlabel("Combined shadow risk M x Q")
    axes[0].set_ylabel("Exact candidates")
    axes[1].scatter(
        risks,
        np.log1p(residuals),
        s=18,
        alpha=0.7,
        color="#D1495B",
    )
    axes[1].set_xlabel("Combined shadow risk M x Q")
    axes[1].set_ylabel("log(1 + final factor residual)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def calibrate(
    summaries: list[dict[str, str]],
    candidates: list[dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = {row.get("label", "") for row in summaries}
    eligible = select_candidates(candidates, summaries)
    active = [row for row in eligible if to_float(row["memory_strength"]) > 0.0]
    active_datasets = {dataset_key(row) for row in active}
    bounds = derive_quality_bounds(active)
    evidence_ok = (
        len(summaries) >= args.minimum_controls
        and len(active) >= args.minimum_active_candidates
        and len(active_datasets) >= args.minimum_active_datasets
        and bounds_are_valid(bounds)
    )
    risks = [candidate_risk(row, bounds) for row in active] if bounds_are_valid(bounds) else []
    sweep = build_risk_sweep(active, risks) if risks else []
    selected = select_sweep_row(
        sweep,
        args.minimum_selected_candidates,
        args.minimum_active_datasets,
        args.minimum_residual_enrichment,
    ) if evidence_ok else None

    parameters: dict[str, Any] = {
        "roughness_near_chain_quality_calibrated": selected is not None,
    }
    if selected is not None:
        downweight_threshold = quantile(risks, 0.95)
        response_cap = quantile(
            (to_float(row["normalized_response"]) for row in active), 0.95
        )
        baseline_threshold = max(
            to_float(row["baseline_threshold"]) for row in active
        )
        response_cap = min(max(response_cap, baseline_threshold), 1.0)
        parameters.update(
            {
                "roughness_near_chain_quality_response_bad": bounds["response_bad"],
                "roughness_near_chain_quality_response_good": bounds["response_good"],
                "roughness_near_chain_quality_translation_good_m": bounds["translation_good_m"],
                "roughness_near_chain_quality_translation_bad_m": bounds["translation_bad_m"],
                "roughness_near_chain_quality_yaw_good_deg": math.degrees(bounds["yaw_good_rad"]),
                "roughness_near_chain_quality_yaw_bad_deg": math.degrees(bounds["yaw_bad_rad"]),
                "roughness_near_chain_quality_xy_covariance_good_m2": bounds["xy_covariance_good_m2"],
                "roughness_near_chain_quality_xy_covariance_bad_m2": bounds["xy_covariance_bad_m2"],
                "roughness_near_chain_quality_yaw_covariance_good_rad2": bounds["yaw_covariance_good_rad2"],
                "roughness_near_chain_quality_yaw_covariance_bad_rad2": bounds["yaw_covariance_bad_rad2"],
                "roughness_adaptive_downweight_risk": min(
                    downweight_threshold, selected["risk_threshold"]
                ),
                "roughness_adaptive_reject_risk": selected["risk_threshold"],
                "roughness_adaptive_dynamic_threshold_max": response_cap,
            }
        )
    return {
        "schema_version": 1,
        "status": "paired_replay_required" if selected else "insufficient_shadow_evidence",
        "production_ready": False,
        "control_labels": sorted(labels),
        "control_datasets": sorted(dataset_key(row) for row in summaries),
        "control_count": len(summaries),
        "eligible_candidate_count": len(eligible),
        "memory_active_candidate_count": len(active),
        "memory_active_dataset_count": len(active_datasets),
        "quality_bounds": bounds,
        "risk_sweep": sweep,
        "selected_sweep": selected,
        "parameters": parameters,
        "caveats": [
            "Final graph residual is a diagnostic, not harmful-factor ground truth.",
            "This calibration may only nominate a paired development replay.",
            "Production promotion requires A03/A27/C/H01 paired maps and unseen H02/H03.",
        ],
        "_active_rows": active,
        "_risks": risks,
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    selected = result["selected_sweep"] or {}
    lines = [
        "# Adaptive Quality Shadow Calibration",
        "",
        f"- Status: {result['status']}",
        f"- Production ready: {result['production_ready']}",
        f"- Valid controls: {result['control_count']}",
        f"- Eligible exact candidates: {result['eligible_candidate_count']}",
        f"- Memory-active candidates: {result['memory_active_candidate_count']}",
        f"- Memory-active datasets: {result['memory_active_dataset_count']}",
        f"- Selected reject risk: {selected.get('risk_threshold', 'not available')}",
        f"- Selected residual enrichment: {selected.get('high_residual_enrichment', 'not available')}",
        "",
        "The residual is used only to prioritize a shadow sweep. The generated "
        "parameters are not a production claim and require paired map replay.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    summaries = select_control_summaries(
        read_csv(args.summary_csv), args.include_label
    )
    candidates = read_csv(args.candidate_csv)
    result = calibrate(summaries, candidates, args)
    active = result.pop("_active_rows")
    risks = result.pop("_risks")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adaptive_quality_calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "adaptive_quality_risk_sweep.csv", result["risk_sweep"])
    selected = result.get("selected_sweep") or {}
    write_csv(
        output_dir / "adaptive_quality_scored_candidates.csv",
        scored_candidate_rows(
            active,
            risks,
            to_float(selected.get("risk_threshold")),
        ),
    )
    write_report(output_dir / "adaptive_quality_calibration.md", result)
    if active:
        plot_diagnostics(
            active,
            risks,
            output_dir / "adaptive_quality_shadow_diagnostics.png",
        )
    print(output_dir)
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
