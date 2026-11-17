#!/usr/bin/env python3
"""Audit replay provenance, artifact integrity, and paired input parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ACCEPTED_STATUSES = {
    "map_and_map_trajectory",
    "map_and_stationary_pose_graph",
}
DOMAIN_PATTERN = re.compile(r"^ROS_DOMAIN_ID=(\d+)\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        metavar=("CONTROL_LABEL", "ADAPTIVE_LABEL"),
        default=[],
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_replays(output_root: Path, labels: Iterable[str]) -> list[Path]:
    stage = output_root / "stage34_roughness_gate_replay" / "variant"
    requested = {label.strip() for label in labels if label.strip()}
    paths = sorted(stage.glob("*/bags/*/replay_result.json"))
    if requested:
        paths = [path for path in paths if path.parents[2].name in requested]
        found = {path.parents[2].name for path in paths}
        missing = sorted(requested - found)
        if missing:
            raise ValueError("replay labels were not found: " + ", ".join(missing))
    return paths


def nested_dict(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    nested = value.get(key, {})
    return nested if isinstance(nested, dict) else {}


def parse_mode(result: dict[str, Any]) -> str:
    value = result.get(
        "runtime_slam_adaptive_constraint_parameter_actual_json", "{}"
    )
    try:
        parameters = json.loads(str(value))
    except json.JSONDecodeError:
        return ""
    if not isinstance(parameters, dict):
        return ""
    return str(parameters.get("roughness_constraint_mode", ""))


def command_domain(path: Path) -> int | None:
    try:
        match = DOMAIN_PATTERN.search(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    return int(match.group(1)) if match else None


def verify_artifact_manifest(
    bag_dir: Path, manifest: Any
) -> tuple[int, list[str]]:
    if not isinstance(manifest, dict) or not manifest:
        return 0, ["artifact_manifest_missing"]
    errors: list[str] = []
    for relative, expected in sorted(manifest.items()):
        if not isinstance(expected, dict):
            errors.append(f"{relative}:invalid_manifest_entry")
            continue
        path = bag_dir / relative
        if not path.is_file():
            errors.append(f"{relative}:missing")
            continue
        expected_size = expected.get("size_bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            errors.append(f"{relative}:size")
            continue
        expected_sha = str(expected.get("sha256", ""))
        if len(expected_sha) != 64 or file_sha256(path) != expected_sha:
            errors.append(f"{relative}:sha256")
    return int(not errors), errors


def audit_replay(path: Path) -> dict[str, Any]:
    bag_dir = path.parent
    label = path.parents[2].name
    bag = bag_dir.name
    result = read_json(path)
    provenance = read_json(bag_dir / "replay_provenance.json")
    fairness = nested_dict(provenance, "fairness_contract")
    backend = nested_dict(fairness, "backend_asset_sha256")
    runtime_build = nested_dict(fairness, "slam_2d_runtime_build")
    adapter = nested_dict(provenance, "lossless_stage34_adapter")
    artifact_valid, artifact_errors = verify_artifact_manifest(
        bag_dir, result.get("artifact_manifest")
    )
    command_ros_domain = command_domain(bag_dir / "commands.txt")
    result_ros_domain = result.get("ros_domain_id")
    domain_match = int(
        command_ros_domain is not None
        and result_ros_domain is not None
        and command_ros_domain == int(result_ros_domain)
    )
    status = str(result.get("status", ""))
    queue_complete = int(result.get("queue_drain_complete", 0) or 0)
    scan_complete = int(
        result.get("slam_recorded_scan_delivery_complete", 0) or 0
    )
    imu_complete = int(result.get("roughness_imu_delivery_complete", 0) or 0)
    graph_complete = int(
        result.get("pose_graph_coverage_status")
        in {"complete", "stationary_single_node_complete"}
    )
    parameter_match = int(result.get("runtime_slam_parameter_match", 0) or 0)
    adaptive_parameter_match = int(
        result.get("runtime_slam_adaptive_constraint_parameter_match", 0) or 0
    )
    replay_valid = int(
        status in ACCEPTED_STATUSES
        and domain_match
        and artifact_valid
        and queue_complete
        and scan_complete
        and imu_complete
        and graph_complete
        and parameter_match
        and adaptive_parameter_match
    )
    return {
        "label": label,
        "bag": bag,
        "mode": parse_mode(result),
        "status": status,
        "replay_valid": replay_valid,
        "command_ros_domain": command_ros_domain,
        "result_ros_domain": result_ros_domain,
        "domain_match": domain_match,
        "artifact_manifest_valid": artifact_valid,
        "artifact_manifest_errors": ";".join(artifact_errors),
        "queue_drain_complete": queue_complete,
        "scan_delivery_complete": scan_complete,
        "imu_delivery_complete": imu_complete,
        "graph_coverage_complete": graph_complete,
        "runtime_parameter_match": parameter_match,
        "adaptive_parameter_match": adaptive_parameter_match,
        "scan_expected_count": result.get(
            "slam_recorded_scan_effective_expected_count", ""
        ),
        "scan_received_count": result.get(
            "slam_recorded_scan_received_count", ""
        ),
        "scan_expected_fingerprint": result.get(
            "slam_recorded_scan_expected_fingerprint_fnv1a64", ""
        ),
        "scan_received_fingerprint": result.get(
            "slam_recorded_scan_received_fingerprint_fnv1a64", ""
        ),
        "scan_stamp_sequence_complete": int(
            result.get("slam_recorded_scan_stamp_sequence_complete", 0) or 0
        ),
        "pose_graph_node_count": result.get("pose_graph_node_count", ""),
        "lossless_adapter_sha256": adapter.get("sha256", ""),
        "runtime_slam_node_sha256": backend.get(
            "roughness_innovation_node", ""
        ),
        "pose_graph_exporter_sha256": backend.get("pose_graph_exporter", ""),
        "runtime_cmake_cache_sha256": runtime_build.get(
            "cmake_cache_sha256", ""
        ),
        "slam_config_sha256": backend.get("slam_config", ""),
    }


def pair_rows(
    rows: list[dict[str, Any]], pairs: Iterable[tuple[str, str]]
) -> list[dict[str, Any]]:
    by_key = {(str(row["label"]), str(row["bag"])): row for row in rows}
    paired: list[dict[str, Any]] = []
    for control_label, adaptive_label in pairs:
        bag_names = sorted(
            {bag for label, bag in by_key if label == control_label}
            | {bag for label, bag in by_key if label == adaptive_label}
        )
        for bag in bag_names:
            control = by_key.get((control_label, bag))
            adaptive = by_key.get((adaptive_label, bag))
            fields = (
                "scan_expected_count",
                "scan_received_count",
                "scan_expected_fingerprint",
                "scan_received_fingerprint",
                "scan_stamp_sequence_complete",
            )
            parity = int(
                control is not None
                and adaptive is not None
                and all(control.get(field) == adaptive.get(field) for field in fields)
            )
            paired.append(
                {
                    "control_label": control_label,
                    "adaptive_label": adaptive_label,
                    "bag": bag,
                    "control_present": int(control is not None),
                    "adaptive_present": int(adaptive is not None),
                    "scan_input_parity": parity,
                    "control_scan_received_count": (
                        control.get("scan_received_count", "") if control else ""
                    ),
                    "adaptive_scan_received_count": (
                        adaptive.get("scan_received_count", "") if adaptive else ""
                    ),
                    "control_scan_received_fingerprint": (
                        control.get("scan_received_fingerprint", "")
                        if control
                        else ""
                    ),
                    "adaptive_scan_received_fingerprint": (
                        adaptive.get("scan_received_fingerprint", "")
                        if adaptive
                        else ""
                    ),
                    "control_pose_graph_node_count": (
                        control.get("pose_graph_node_count", "") if control else ""
                    ),
                    "adaptive_pose_graph_node_count": (
                        adaptive.get("pose_graph_node_count", "") if adaptive else ""
                    ),
                }
            )
    return paired


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> dict[str, Any]:
    invalid = [row for row in rows if not row["replay_valid"]]
    parity_failures = [row for row in paired if not row["scan_input_parity"]]
    summary = {
        "replays": len(rows),
        "valid_replays": len(rows) - len(invalid),
        "invalid_replays": len(invalid),
        "paired_rows": len(paired),
        "scan_input_parity_failures": len(parity_failures),
        "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "invalid": [f"{row['label']}/{row['bag']}" for row in invalid],
        "parity_failures": [
            f"{row['control_label']} vs {row['adaptive_label']}/{row['bag']}"
            for row in parity_failures
        ],
    }
    lines = [
        "# Adaptive replay provenance audit",
        "",
        f"- Replays: {summary['replays']}",
        f"- Valid replays: {summary['valid_replays']}",
        f"- Invalid replays: {summary['invalid_replays']}",
        f"- Paired rows: {summary['paired_rows']}",
        f"- Scan-input parity failures: {summary['scan_input_parity_failures']}",
        "",
        "## Invalid replays",
        "",
    ]
    lines.extend(f"- {value}" for value in summary["invalid"])
    if not summary["invalid"]:
        lines.append("- None")
    lines.extend(["", "## Paired scan-input parity failures", ""])
    lines.extend(f"- {value}" for value in summary["parity_failures"])
    if not summary["parity_failures"]:
        lines.append("- None")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    labels = list(args.include_label)
    for pair in args.pair:
        labels.extend(pair)
    rows = [audit_replay(path) for path in selected_replays(output_root, labels)]
    paired = pair_rows(rows, [tuple(pair) for pair in args.pair])
    write_csv(artifact_root / "csv/replay_provenance_audit.csv", rows)
    write_csv(artifact_root / "csv/paired_scan_input_parity.csv", paired)
    summary = write_report(
        artifact_root / "report/replay_provenance_audit.md", rows, paired
    )
    (artifact_root / "report/replay_provenance_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return int(bool(summary["invalid_replays"]))


if __name__ == "__main__":
    raise SystemExit(main())
