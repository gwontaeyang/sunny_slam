#!/usr/bin/env python3
"""Export exact final Karto edge evidence for completed adaptive replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--include-label", action="append", default=[])
    parser.add_argument(
        "--exporter",
        type=Path,
        default=Path.home()
        / "gole_ws/install/slam_2d_core/lib/slam_2d_core/slam_pose_graph_edge_exporter",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def csv_fields(path: Path) -> set[str]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return set(next(csv.reader(stream), []))
    except OSError:
        return set()


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trajectories_equivalent(
    existing_path: Path,
    exported_path: Path,
    tolerance: float = 1.0e-12,
) -> tuple[bool, int]:
    existing = read_csv(existing_path)
    exported = read_csv(exported_path)
    if len(existing) != len(exported) or not existing:
        return False, len(exported)
    fields = (
        "stamp_s",
        "x_m",
        "y_m",
        "yaw_rad",
        "odom_x_m",
        "odom_y_m",
        "odom_yaw_rad",
    )
    for expected, actual in zip(existing, exported):
        for field in ("source", "state_id", "unique_id"):
            if expected.get(field) != actual.get(field):
                return False, len(exported)
        for field in fields:
            left = to_float(expected.get(field))
            right = to_float(actual.get(field))
            if not math.isfinite(left) or not math.isfinite(right):
                if str(expected.get(field, "")) != str(actual.get(field, "")):
                    return False, len(exported)
            elif not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
                return False, len(exported)
    return True, len(exported)


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


def export_one(
    replay_path: Path,
    exporter: Path,
    artifact_root: Path,
    force: bool,
    timeout_s: float,
) -> dict[str, Any]:
    label = replay_path.parents[2].name
    bag = replay_path.parent.name
    bag_dir = replay_path.parent
    pose_graph = bag_dir / "final_pose_graph.posegraph"
    existing_trajectory = bag_dir / "trajectory_pose_graph_final.csv"
    edge_output = bag_dir / "pose_graph_edges_final.csv"
    provenance_output = bag_dir / "pose_graph_edges_final.provenance.json"
    row: dict[str, Any] = {
        "label": label,
        "bag": bag,
        "status": "not_run",
        "pose_graph": str(pose_graph),
        "edge_csv": str(edge_output),
        "provenance_json": str(provenance_output),
        "node_count": 0,
        "edge_count": 0,
        "trajectory_equivalent": 0,
        "return_code": "",
        "exporter_sha256": file_sha256(exporter),
        "pose_graph_sha256": "",
        "edge_csv_sha256": "",
    }
    if not pose_graph.is_file() or not existing_trajectory.is_file():
        row["status"] = "missing_pose_graph_or_trajectory"
        return row
    row["pose_graph_sha256"] = file_sha256(pose_graph)
    trajectory_sha256 = file_sha256(existing_trajectory)
    existing_provenance = read_json(provenance_output)
    if (
        not force
        and edge_output.is_file()
        and REQUIRED_EDGE_FIELDS <= csv_fields(edge_output)
        and existing_provenance.get("schema_version") == 2
        and existing_provenance.get("status") == "exported"
        and existing_provenance.get("edge_csv_sha256") == file_sha256(edge_output)
        and existing_provenance.get("pose_graph_sha256") == row["pose_graph_sha256"]
        and existing_provenance.get("trajectory_sha256") == trajectory_sha256
        and existing_provenance.get("exporter_sha256") == row["exporter_sha256"]
        and int(existing_provenance.get("trajectory_equivalent", 0) or 0) == 1
    ):
        row["status"] = "skipped_valid_existing"
        row["node_count"] = len(read_csv(existing_trajectory))
        row["edge_count"] = len(read_csv(edge_output))
        row["trajectory_equivalent"] = 1
        row["edge_csv_sha256"] = file_sha256(edge_output)
        return row

    temporary_nodes = bag_dir / f".pose_graph_nodes_export.{os.getpid()}.csv"
    temporary_edges = bag_dir / f".pose_graph_edges_export.{os.getpid()}.csv"
    temporary_provenance = provenance_output.with_name(
        provenance_output.name + f".{os.getpid()}.tmp"
    )
    command = [
        str(exporter),
        str(pose_graph.with_suffix("")),
        str(temporary_nodes),
        str(temporary_edges),
    ]
    log_path = artifact_root / "logs" / label / f"{bag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        row["return_code"] = completed.returncode
        log_path.write_text(
            "command=" + json.dumps(command) + "\n" + completed.stdout,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            row["status"] = "export_failed"
            return row
        equivalent, node_count = trajectories_equivalent(
            existing_trajectory,
            temporary_nodes,
        )
        row["node_count"] = node_count
        row["trajectory_equivalent"] = int(equivalent)
        row["edge_count"] = len(read_csv(temporary_edges))
        if not equivalent:
            row["status"] = "trajectory_mismatch"
            return row
        if not REQUIRED_EDGE_FIELDS <= csv_fields(temporary_edges):
            row["status"] = "edge_schema_invalid"
            return row
        temporary_edges.replace(edge_output)
        row["edge_csv_sha256"] = file_sha256(edge_output)
        row["status"] = "exported"
        provenance = {
            "schema_version": 2,
            "status": "exported",
            "label": label,
            "bag": bag,
            "exporter": str(exporter),
            "exporter_sha256": row["exporter_sha256"],
            "postprocessor_sha256": file_sha256(Path(__file__).resolve()),
            "pose_graph": str(pose_graph),
            "pose_graph_sha256": row["pose_graph_sha256"],
            "existing_trajectory": str(existing_trajectory),
            "trajectory_sha256": trajectory_sha256,
            "trajectory_equivalent": row["trajectory_equivalent"],
            "node_count": row["node_count"],
            "edge_csv": str(edge_output),
            "edge_csv_sha256": row["edge_csv_sha256"],
            "edge_count": row["edge_count"],
        }
        temporary_provenance.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_provenance.replace(provenance_output)
        return row
    except subprocess.TimeoutExpired as error:
        row["status"] = "export_timeout"
        log_path.write_text(str(error), encoding="utf-8")
        return row
    except OSError as error:
        row["status"] = "export_os_error"
        log_path.write_text(str(error), encoding="utf-8")
        return row
    finally:
        temporary_nodes.unlink(missing_ok=True)
        temporary_edges.unlink(missing_ok=True)
        temporary_provenance.unlink(missing_ok=True)


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "label",
        "bag",
        "status",
        "node_count",
        "edge_count",
        "trajectory_equivalent",
        "return_code",
        "exporter_sha256",
        "pose_graph_sha256",
        "edge_csv_sha256",
        "pose_graph",
        "edge_csv",
        "provenance_json",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    exporter = args.exporter.expanduser().resolve()
    if not exporter.is_file():
        raise FileNotFoundError(f"pose graph exporter not found: {exporter}")
    replay_paths = selected_replays(output_root, args.include_label)
    if not replay_paths:
        raise ValueError("no replay results were found")
    rows = [
        export_one(
            path,
            exporter,
            artifact_root,
            args.force,
            args.timeout_s,
        )
        for path in replay_paths
    ]
    write_manifest(artifact_root / "pose_graph_edge_export_manifest.csv", rows)
    failed = [
        row for row in rows
        if row["status"] not in {"exported", "skipped_valid_existing"}
    ]
    print(
        json.dumps(
            {
                "replay_count": len(rows),
                "exported_count": sum(row["status"] == "exported" for row in rows),
                "skipped_count": sum(
                    row["status"] == "skipped_valid_existing" for row in rows
                ),
                "failed_count": len(failed),
                "manifest": str(
                    artifact_root / "pose_graph_edge_export_manifest.csv"
                ),
            },
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
