#!/usr/bin/env python3
"""Summarize exact Karto loop-closure observer logs without changing the graph."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt


plt.switch_backend("Agg")

LOOP_BATCH_MARKER = "KARTO_LOOP_CANDIDATE_BATCH"
LOOP_CANDIDATE_MARKER = "KARTO_LOOP_CANDIDATE"
KEY_VALUE_PATTERN = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        action="append",
        default=[],
        help="Launch log to parse. Repeat for multiple replays.",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        help="Recursively include *_launch.log files below this directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--large-yaw-audit-deg",
        type=float,
        default=90.0,
        help="Diagnostic-only yaw disagreement cutoff; it is not a rejection gate.",
    )
    return parser.parse_args()


def parse_key_values(line: str, marker: str) -> dict[str, str] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    suffix = line[marker_index + len(marker) :]
    return {
        match.group("key"): match.group("value")
        for match in KEY_VALUE_PATTERN.finditer(suffix)
    }


def parse_log(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    batches: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return batches, candidates

    for line_number, line in enumerate(lines, start=1):
        marker: str | None = None
        destination: list[dict[str, str]] | None = None
        if LOOP_BATCH_MARKER in line:
            marker = LOOP_BATCH_MARKER
            destination = batches
        elif LOOP_CANDIDATE_MARKER in line:
            marker = LOOP_CANDIDATE_MARKER
            destination = candidates
        if marker is None or destination is None:
            continue
        parsed = parse_key_values(line, marker)
        if parsed is None:
            continue
        parsed["log_path"] = str(path)
        parsed["line_number"] = str(line_number)
        destination.append(parsed)
    return batches, candidates


def as_float(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def as_int(row: Mapping[str, str], key: str) -> int | None:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def as_bool(row: Mapping[str, str], key: str) -> bool | None:
    value = row.get(key, "").lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def finite_values(rows: Iterable[Mapping[str, str]], key: str) -> list[float]:
    return [value for row in rows if math.isfinite(value := as_float(row, key))]


def quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        if len(finite) == 1:
            return finite[0]
        position = fraction * (len(finite) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return finite[lower]
        weight = position - lower
        return finite[lower] * (1.0 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": percentile(0.95),
        "max": finite[-1],
    }


def deduplicate(
    rows: Iterable[dict[str, str]],
    identity_fields: Sequence[str],
) -> tuple[list[dict[str, str]], int]:
    unique: list[dict[str, str]] = []
    identities: set[tuple[str, ...]] = set()
    duplicate_count = 0
    for row in rows:
        identity = tuple(row.get(field, "") for field in identity_fields)
        if identity in identities:
            duplicate_count += 1
            continue
        identities.add(identity)
        unique.append(row)
    return unique, duplicate_count


def summarize(
    batches: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, str]],
    large_yaw_audit_deg: float,
) -> dict[str, Any]:
    dispositions = Counter(row.get("disposition", "missing") for row in candidates)
    failures = Counter(
        row.get("failure", "missing")
        for row in batches
        if row.get("failure", "none") != "none"
    )
    actual_edges = [row for row in candidates if as_bool(row, "actual_edge_present")]
    accepted_without_edges = [
        row for row in candidates if as_bool(row, "accepted_without_edge")
    ]
    accepted = [row for row in candidates if row.get("disposition") == "accepted"]
    yaw_values_rad = finite_values(actual_edges, "measurement_vs_odom_yaw_rad")
    yaw_values_deg = [math.degrees(value) for value in yaw_values_rad]
    large_yaw_count = sum(value >= large_yaw_audit_deg for value in yaw_values_deg)

    batch_attempt_sum = sum(as_int(row, "attempts") or 0 for row in batches)
    batch_accepted_sum = sum(
        as_int(row, "accepted_callbacks") or 0 for row in batches
    )
    batch_edge_sum = sum(as_int(row, "actual_loop_edges") or 0 for row in batches)
    batch_accepted_without_edge_sum = sum(
        as_int(row, "accepted_without_edges") or 0 for row in batches
    )
    warmup_batches = sum(
        row.get("failure") == "graph_not_initialized"
        and (as_int(row, "attempts") or 0) == 0
        for row in batches
    )
    pairing_failure_batches = sum(
        as_bool(row, "edge_pairing_complete") is False
        for row in batches
        if as_bool(row, "scan_processed") is not False
        and row.get("failure") != "graph_not_initialized"
    )
    observer_unavailable_batches = sum(
        as_bool(row, "observer_available") is False
        and row.get("failure") != "graph_not_initialized"
        for row in batches
    )

    return {
        "batch_count": len(batches),
        "candidate_count": len(candidates),
        "batch_attempt_sum": batch_attempt_sum,
        "candidate_count_matches_batches": batch_attempt_sum == len(candidates),
        "accepted_candidate_count": len(accepted),
        "batch_accepted_callback_sum": batch_accepted_sum,
        "actual_edge_count": len(actual_edges),
        "batch_actual_edge_sum": batch_edge_sum,
        "accepted_without_edge_count": len(accepted_without_edges),
        "batch_accepted_without_edge_sum": batch_accepted_without_edge_sum,
        "accepted_count_matches_batches": batch_accepted_sum == len(accepted),
        "actual_edge_count_matches_batches": batch_edge_sum == len(actual_edges),
        "accepted_without_edge_count_matches_batches": (
            batch_accepted_without_edge_sum == len(accepted_without_edges)
        ),
        "pairing_failure_batch_count": pairing_failure_batches,
        "graph_initialization_warmup_batch_count": warmup_batches,
        "observer_unavailable_batch_count": observer_unavailable_batches,
        "failure_counts": dict(sorted(failures.items())),
        "disposition_counts": dict(sorted(dispositions.items())),
        "coarse_response": quantiles(finite_values(candidates, "coarse_response")),
        "fine_response": quantiles(finite_values(candidates, "fine_response")),
        "accepted_measurement_vs_odom_translation_m": quantiles(
            finite_values(actual_edges, "measurement_vs_odom_translation_m")
        ),
        "accepted_measurement_vs_odom_yaw_deg": quantiles(yaw_values_deg),
        "accepted_measurement_vs_odom_covariance_weighted_disagreement_sq": quantiles(
            finite_values(
                actual_edges,
                "measurement_vs_odom_covariance_weighted_disagreement_sq",
            )
        ),
        "candidate_memory_strength": quantiles(
            finite_values(candidates, "memory_strength")
        ),
        "accepted_pre_residual_yaw_deg": quantiles(
            [
                math.degrees(value)
                for value in finite_values(actual_edges, "pre_residual_yaw_rad")
            ]
        ),
        "accepted_post_residual_yaw_deg": quantiles(
            [
                math.degrees(value)
                for value in finite_values(actual_edges, "post_residual_yaw_rad")
            ]
        ),
        "accepted_post_covariance_weighted_residual_sq": quantiles(
            finite_values(actual_edges, "post_covariance_weighted_residual_sq")
        ),
        "large_yaw_audit_threshold_deg": large_yaw_audit_deg,
        "large_yaw_actual_edge_count": large_yaw_count,
        "instrumentation_valid": (
            batch_attempt_sum == len(candidates)
            and batch_accepted_sum == len(accepted)
            and batch_edge_sum == len(actual_edges)
            and batch_accepted_without_edge_sum == len(accepted_without_edges)
            and batch_accepted_sum
            == batch_edge_sum + batch_accepted_without_edge_sum
            and pairing_failure_batches == 0
        ),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_candidates(path: Path, candidates: Sequence[Mapping[str, str]]) -> bool:
    if not candidates:
        return False
    x_values = [
        as_int(row, "target_state_id")
        if as_int(row, "target_state_id") is not None
        else index
        for index, row in enumerate(candidates)
    ]
    coarse = [as_float(row, "coarse_response") for row in candidates]
    yaw_deg = [
        math.degrees(as_float(row, "measurement_vs_odom_yaw_rad"))
        if math.isfinite(as_float(row, "measurement_vs_odom_yaw_rad"))
        else math.nan
        for row in candidates
    ]
    colors = [
        "#d1495b" if row.get("disposition") == "accepted" else "#287271"
        for row in candidates
    ]

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].scatter(x_values, coarse, c=colors, s=20, alpha=0.85)
    axes[0].set_ylabel("Coarse response")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Exact Karto loop candidates (red: accepted callback)")
    axes[1].scatter(x_values, yaw_deg, c=colors, s=20, alpha=0.85)
    axes[1].set_xlabel("Target graph state ID")
    axes[1].set_ylabel("|measurement - odometry| yaw [deg]")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return True


def format_metric(metric: Mapping[str, Any], unit: str = "") -> str:
    if not metric.get("count"):
        return "not available"
    return (
        f"n={metric['count']}, median={metric['median']:.4g}{unit}, "
        f"p95={metric['p95']:.4g}{unit}, max={metric['max']:.4g}{unit}"
    )


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Exact Karto Loop-Candidate Audit",
        "",
        "## Integrity",
        "",
        f"- Instrumentation valid: `{str(summary['instrumentation_valid']).lower()}`",
        f"- Candidate batches: {summary['batch_count']}",
        f"- Exact attempts: {summary['candidate_count']}",
        "- Accepted callbacks / actual edges / accepted without edge: "
        f"{summary['accepted_candidate_count']} / {summary['actual_edge_count']} / "
        f"{summary['accepted_without_edge_count']}",
        f"- Pairing-failure batches: {summary['pairing_failure_batch_count']}",
        f"- Graph-initialization warm-up batches: "
        f"{summary['graph_initialization_warmup_batch_count']}",
        f"- Observer-unavailable batches: {summary['observer_unavailable_batch_count']}",
        "",
        "## Candidate Distributions",
        "",
        f"- Dispositions: `{json.dumps(summary['disposition_counts'], sort_keys=True)}`",
        f"- Coarse response: {format_metric(summary['coarse_response'])}",
        f"- Fine response: {format_metric(summary['fine_response'])}",
        "- Accepted edge odometry translation disagreement: "
        f"{format_metric(summary['accepted_measurement_vs_odom_translation_m'], ' m')}",
        "- Accepted edge odometry yaw disagreement: "
        f"{format_metric(summary['accepted_measurement_vs_odom_yaw_deg'], ' deg')}",
        "- Accepted edge covariance-weighted odometry disagreement: "
        f"{format_metric(summary['accepted_measurement_vs_odom_covariance_weighted_disagreement_sq'])}",
        f"- Candidate memory strength: {format_metric(summary['candidate_memory_strength'])}",
        "- Accepted edge post-optimization yaw residual: "
        f"{format_metric(summary['accepted_post_residual_yaw_deg'], ' deg')}",
        "",
        "## Diagnostic Flag",
        "",
        f"- Actual edges above {summary['large_yaw_audit_threshold_deg']:.3g} deg "
        f"odometry disagreement: {summary['large_yaw_actual_edge_count']}",
        "- This cutoff is descriptive only and is not used as an online rejection rule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_logs(explicit: Iterable[Path], root: Path | None) -> list[Path]:
    paths = list(explicit)
    if root is not None:
        paths.extend(root.rglob("*_launch.log"))
    return sorted({path.resolve() for path in paths if path.is_file()})


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.large_yaw_audit_deg) or args.large_yaw_audit_deg < 0.0:
        raise ValueError("--large-yaw-audit-deg must be finite and non-negative")
    logs = collect_logs(args.log, args.log_root)
    if not logs:
        raise FileNotFoundError("no launch logs found")

    batches: list[dict[str, str]] = []
    candidates: list[dict[str, str]] = []
    for log_path in logs:
        parsed_batches, parsed_candidates = parse_log(log_path)
        batches.extend(parsed_batches)
        candidates.extend(parsed_candidates)

    batches, duplicate_batches = deduplicate(
        batches, ("stamp_ns", "mapper_generation", "attempts")
    )
    candidates, duplicate_candidates = deduplicate(
        candidates,
        ("stamp_ns", "mapper_generation", "candidate_index", "disposition"),
    )
    summary = summarize(batches, candidates, args.large_yaw_audit_deg)
    summary["log_paths"] = [str(path) for path in logs]
    summary["duplicate_batch_lines_removed"] = duplicate_batches
    summary["duplicate_candidate_lines_removed"] = duplicate_candidates

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "karto_loop_candidate_batches.csv", batches)
    write_csv(output_dir / "karto_loop_candidates.csv", candidates)
    (output_dir / "karto_loop_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_candidates(output_dir / "karto_loop_candidate_audit.png", candidates)
    write_report(output_dir / "karto_loop_candidate_report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
