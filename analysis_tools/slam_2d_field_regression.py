#!/usr/bin/env python3
"""Run the field-bag 2D SLAM regression and publish a map-only quick view.

The raw bag directories are never modified. Bags without metadata are staged
with symlinked storage files and reindexed only inside the staging directory.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import csv
import errno
import hashlib
import json
import math
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml


DEFAULT_BAG_ROOTS = (
    Path.home() / "01_gs_gomdan",
    Path.home() / "03_금정",
)
DEFAULT_OUTPUT_ROOT = (
    Path.home() / "analysis_outputs/slam_2d/23_01_03_field_regression"
)
DEFAULT_QUICK_ROOT = (
    Path.home() / "analysis_outputs/slam_2d/00_latest_map_review"
)
DEFAULT_STAGING_ROOT = Path("/tmp/slam_2d_01_03_field_bags")
DEFAULT_PIPELINE = (
    Path(__file__).resolve().with_name("lossless_stage34_runner.py")
)
DEFAULT_SETUP_BASH = (
    Path.home()
    / "analysis_outputs/slam_2d/02_roughness_return_resume/scripts"
    / "gole_ws_env.bash"
)
DEFAULT_SLAM_CONFIG = (
    Path.home()
    / "gole_ws/src/gole_autonomy/gole_slam/slam_2d_bringup/config"
    / "mapper_params.yaml"
)
ACCEPTED_REPLAY_STATUSES = {
    "map_and_map_trajectory",
    "map_and_stationary_pose_graph",
}
ALLOWED_FAIRNESS_DIFFERENCE = (
    "roughness_innovation_profile.enable_spatial_passage_hysteresis"
)
INACTIVE_LASERSCAN_GUARD_ASSET_FIELDS = (
    "amcl_guard_header",
    "amcl_guard_node",
    "amcl_guard_source",
    "imu_quality_policy_source",
    "laser_scan_stabilizer_source",
)
REPLAY_ARTIFACT_NAMES_V3 = (
    "summary.json",
    "map.png",
    "map.pgm",
    "map.yaml",
    "map_with_trajectory.png",
    "trajectory_map_frame.csv",
    "trajectory_pose_graph_final.csv",
    "final_pose_graph.posegraph",
    "final_pose_graph.data",
    "replay_provenance.json",
)
REPLAY_ARTIFACT_NAMES_V4 = REPLAY_ARTIFACT_NAMES_V3 + (
    "commands.txt",
    "pose_graph_export.log",
)
REPLAY_ARTIFACT_NAMES_V5 = REPLAY_ARTIFACT_NAMES_V4 + (
    "guard_status_counts.csv",
    "guard_status_events.csv",
    "runtime_parameters/slam_laserscan_roughness_gate.yaml",
    "runtime_parameters/slam_toolbox.yaml",
    "roughness_innovation_events.csv",
    "roughness_post_confirmation_events.csv",
)
REPLAY_AUDIT_COLUMNS = (
    "variant",
    "bag",
    "audit_pass",
    "failure_reasons",
    "status",
    "map_present",
    "runtime_parameter_match",
    "slam_input_complete",
    "scan_delivery_complete",
    "imu_delivery_complete",
    "imu_fingerprint_match",
    "queue_drain_complete",
    "queue_drain_status",
    "queue_drain_evidence",
    "pose_graph_captured",
    "pose_graph_coverage_complete",
    "pose_graph_coverage_status",
    "pose_graph_trajectory_count",
    "stationary_replay",
    "trajectory_map_frame_path_length_m",
    "trajectory_odom_frame_path_length_m",
    "roughness_received_scan_count",
    "roughness_received_imu_count",
    "roughness_expected_imu_count",
    "roughness_spatial_passage_count",
    "roughness_recovery_count",
    "subscription_drain_complete",
    "subscription_finalize_pending_success",
    "replay_provenance_status",
    "replay_signature_match",
    "artifact_manifest_status",
    "replay_integrity_status",
    "fairness_contract_status",
    "fairness_mismatch_fields",
    "fairness_equivalent_fields",
    "replay_result_json",
    "map_png",
)


@dataclass
class BagRecord:
    name: str
    source_root: str
    source_path: str
    metadata_present: bool
    storage_files: int
    storage_bytes: int
    status: str
    staged_path: str = ""
    duration_s: float = math.nan
    scan_laserscan_count: int = 0
    imu_count: int = 0
    wheel_odom_count: int = 0
    storage_integrity_status: str = "not_checked"
    storage_integrity_detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status in {"ready", "reindexed"}


def _storage_files(bag_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in bag_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".db3", ".mcap"}
    )


def discover_bags(roots: Sequence[Path]) -> list[BagRecord]:
    records: list[BagRecord] = []
    seen: dict[str, Path] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"bag root does not exist: {root}")
        for bag_dir in sorted(root.iterdir()):
            if not bag_dir.is_dir() or not bag_dir.name.startswith("ty_bag_"):
                continue
            previous = seen.get(bag_dir.name)
            if previous is not None:
                raise RuntimeError(
                    f"duplicate bag name {bag_dir.name}: {previous} and {bag_dir}"
                )
            seen[bag_dir.name] = bag_dir
            storage = _storage_files(bag_dir)
            storage_bytes = sum(path.stat().st_size for path in storage)
            metadata_present = (bag_dir / "metadata.yaml").is_file()
            if metadata_present and storage_bytes > 0:
                status = "ready"
            elif storage_bytes > 0:
                status = "needs_reindex"
            elif storage:
                status = "empty_storage"
            else:
                status = "missing_storage"
            records.append(
                BagRecord(
                    name=bag_dir.name,
                    source_root=root.name,
                    source_path=str(bag_dir),
                    metadata_present=metadata_present,
                    storage_files=len(storage),
                    storage_bytes=storage_bytes,
                    status=status,
                )
            )
    return records


def _replace_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(
            f"refusing to replace non-symlink staging file: {destination}"
        )
    destination.symlink_to(source.resolve())


def _metadata_fields(metadata_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    info = payload.get("rosbag2_bagfile_information", {})
    duration = info.get("duration", {})
    duration_s = float(duration.get("nanoseconds", 0) or 0) * 1e-9
    topics = {}
    for item in info.get("topics_with_message_count", []):
        topic = item.get("topic_metadata", {}).get("name", "")
        topics[topic] = int(item.get("message_count", 0) or 0)
    return {
        "duration_s": duration_s,
        "scan_laserscan_count": topics.get("/scan_laserscan", 0),
        "imu_count": topics.get("/imu/data", 0),
        "wheel_odom_count": topics.get("/motor_driver_node/odom", 0),
    }


def prepare_staging(
    records: Sequence[BagRecord],
    staging_root: Path,
    *,
    reindex: bool = True,
) -> list[BagRecord]:
    staging_root = staging_root.expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    for record in records:
        if record.status not in {"ready", "needs_reindex"}:
            continue
        source = Path(record.source_path)
        staged = staging_root / record.name
        staged.mkdir(parents=True, exist_ok=True)
        for storage in _storage_files(source):
            _replace_symlink(storage, staged / storage.name)
        if record.status == "ready":
            _replace_symlink(source / "metadata.yaml", staged / "metadata.yaml")
        elif reindex:
            completed = subprocess.run(
                ["ros2", "bag", "reindex", str(staged)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode != 0 or not (staged / "metadata.yaml").is_file():
                record.status = "reindex_failed"
                continue
            record.status = "reindexed"
        record.staged_path = str(staged)
        metadata = staged / "metadata.yaml"
        if metadata.is_file():
            for key, value in _metadata_fields(metadata).items():
                setattr(record, key, value)
            if record.scan_laserscan_count <= 0:
                record.status = "missing_scan_laserscan"
            elif record.imu_count <= 0:
                record.status = "missing_imu"
    return list(records)


def _sqlite_quick_check(path: Path) -> tuple[bool, str]:
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        messages = [
            str(row[0]).strip()
            for row in connection.execute("PRAGMA quick_check").fetchall()
        ]
    finally:
        connection.close()
    passed = messages == ["ok"]
    detail = "ok" if passed else "; ".join(messages[:8])
    return passed, detail


def audit_storage_integrity(records: Sequence[BagRecord]) -> list[BagRecord]:
    """Exclude malformed SQLite bags without modifying their source files."""
    for record in records:
        if not record.usable:
            continue
        storage = _storage_files(Path(record.staged_path))
        sqlite_storage = [path for path in storage if path.suffix.lower() == ".db3"]
        if not sqlite_storage:
            record.storage_integrity_status = "not_applicable"
            record.storage_integrity_detail = "no_db3_storage"
            continue
        record.storage_integrity_status = "passed"
        record.storage_integrity_detail = "ok"
        for database in sqlite_storage:
            try:
                passed, detail = _sqlite_quick_check(database)
            except sqlite3.Error as exc:
                passed = False
                detail = f"{type(exc).__name__}: {exc}"
            if passed:
                continue
            record.status = "storage_corrupt"
            record.storage_integrity_status = "failed"
            record.storage_integrity_detail = f"{database.name}: {detail}"
            break
    return list(records)


def write_manifest(records: Sequence[BagRecord], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    csv_path = directory / "input_bag_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "bag_count": len(records),
        "usable_count": sum(record.usable for record in records),
        "unusable_count": sum(not record.usable for record in records),
        "duration_hours": sum(
            record.duration_s
            for record in records
            if record.usable and math.isfinite(record.duration_s)
        )
        / 3600.0,
        "records": rows,
    }
    (directory / "input_bag_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_replay_slam_config(
    source: Path,
    destination: Path,
    *,
    scan_queue_size: int,
    tf_buffer_duration_s: float,
    imu_buffer_duration_s: float = 1800.0,
    imu_qos_depth: int = 200000,
    diagnostics_period_s: float = 2.0,
    pending_timeout_s: float = 600.0,
    max_pending_scans: int = 20000,
) -> Path:
    if scan_queue_size <= 0:
        raise ValueError("replay scan queue size must be positive")
    if not math.isfinite(tf_buffer_duration_s) or tf_buffer_duration_s <= 0.0:
        raise ValueError("replay TF buffer duration must be finite and positive")
    if not math.isfinite(imu_buffer_duration_s) or imu_buffer_duration_s <= 0.0:
        raise ValueError("replay IMU buffer duration must be finite and positive")
    if imu_qos_depth <= 0:
        raise ValueError("replay IMU QoS depth must be positive")
    if not math.isfinite(diagnostics_period_s) or diagnostics_period_s <= 0.0:
        raise ValueError("replay diagnostics period must be finite and positive")
    if not math.isfinite(pending_timeout_s) or pending_timeout_s <= 0.0:
        raise ValueError("replay pending timeout must be finite and positive")
    if max_pending_scans <= 0:
        raise ValueError("replay maximum pending scans must be positive")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    try:
        parameters = payload["slam_toolbox"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid slam_toolbox parameter file: {source}") from exc
    parameters["scan_queue_size"] = int(scan_queue_size)
    parameters["tf_buffer_duration"] = float(tf_buffer_duration_s)
    parameters["roughness_imu_buffer_duration_s"] = float(
        imu_buffer_duration_s
    )
    parameters["roughness_imu_qos_reliable"] = True
    parameters["roughness_imu_qos_depth"] = int(imu_qos_depth)
    parameters["roughness_diagnostics_period_s"] = float(
        diagnostics_period_s
    )
    parameters["roughness_pending_timeout_s"] = float(pending_timeout_s)
    parameters["roughness_max_pending_scans"] = int(max_pending_scans)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return destination


def _rate_label(rate: float) -> str:
    return f"{rate:g}x".replace(".", "p")


def build_stage34_command(
    *,
    pipeline: Path,
    setup_bash: Path,
    staging_root: Path,
    output_root: Path,
    slam_config: Path,
    bag_names: Sequence[str],
    snapshot_label: str,
    rate: float,
    spatial_passage: bool,
    reference_root: Path | None,
    skip_existing: bool,
    ros_domain_base: int,
    scan_qos_depth: int,
    post_playback_wait_s: float,
    queue_drain_timeout_s: float = 600.0,
    startup_tf_drop_limit: int = 10,
) -> list[str]:
    command = [
        sys.executable,
        str(pipeline),
        "stage34-roughness-gate-replay",
        "--bag-root",
        str(staging_root),
        "--output",
        str(output_root),
        "--setup-bash",
        str(setup_bash),
        "--slam-config",
        str(slam_config),
        "--snapshot-label",
        snapshot_label,
        "--baseline-label",
        "field_01_03",
        "--max-bags",
        "0",
        "--ros-domain-base",
        str(ros_domain_base),
        "--bag-play-rate",
        f"{rate:g}",
        "--launch-start-delay-s",
        "15",
        "--post-playback-wait-s",
        f"{post_playback_wait_s:g}",
        "--queue-drain-timeout-s",
        f"{queue_drain_timeout_s:g}",
        "--slam-input-max-startup-tf-drop-count",
        str(startup_tf_drop_limit),
        "--bag-list",
        *bag_names,
        "--use-roughness-innovation-slam-toolbox",
        "--roughness-innovation-scan-qos-reliable",
        "--roughness-innovation-scan-qos-depth",
        str(scan_qos_depth),
        "--roughness-innovation-min-rough-samples",
        "1",
        "--roughness-innovation-min-rough-duration-s",
        "0",
        "--roughness-innovation-translation-gate-m",
        "0.015",
        "--roughness-innovation-yaw-gate-deg",
        "1.0",
        "--roughness-innovation-enable-near-chain-gate",
        "--roughness-innovation-enable-spatial-memory-gate",
        "--roughness-innovation-spatial-memory-radius-m",
        "0.6",
        "--roughness-innovation-spatial-memory-min-revisit-separation-s",
        "20.0",
        "--roughness-innovation-spatial-memory-min-heading-delta-deg",
        "0.0",
        "--roughness-innovation-spatial-memory-merge-radius-m",
        "0.1",
        "--roughness-innovation-spatial-memory-max-zones",
        "200",
        "--roughness-innovation-spatial-passage-radius-m",
        "0.55",
        "--roughness-innovation-spatial-passage-max-age-s",
        "3.5",
        "--roughness-innovation-near-chain-recovery-duration-s",
        "1.1",
        "--roughness-innovation-near-chain-min-rough-samples",
        "1",
        "--roughness-innovation-near-chain-gated-min-response",
        "1.1",
        "--roughness-innovation-enable-post-confirmation",
        "--ros-localhost-only",
    ]
    command.append(
        "--roughness-innovation-enable-spatial-passage-hysteresis"
        if spatial_passage
        else "--no-roughness-innovation-enable-spatial-passage-hysteresis"
    )
    if reference_root is not None:
        command.extend(
            [
                "--reference-variant-root",
                str(reference_root),
                "--allow-replay-fairness-mismatch-field",
                "roughness_innovation_profile.enable_spatial_passage_hysteresis",
            ]
        )
    if skip_existing:
        command.append("--skip-existing")
    return command


def _copy_map(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _write_side_by_side(
    destination: Path,
    bag_name: str,
    baseline_map: Path,
    post_map: Path,
) -> bool:
    if not baseline_map.is_file() or not post_map.is_file():
        return False
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-slam-2d-review")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, title, path in (
        (axes[0], "baseline SLAM", baseline_map),
        (axes[1], "post SLAM", post_map),
    ):
        axis.imshow(mpimg.imread(path))
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(bag_name)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return True


def _write_contact_sheet(comparisons: Sequence[tuple[str, Path]], output: Path) -> None:
    if not comparisons:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-slam-2d-review")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    columns = 3
    rows = math.ceil(len(comparisons) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(18, 5.2 * rows))
    axes_flat = list(getattr(axes, "flat", [axes]))
    for axis, (bag_name, image_path) in zip(axes_flat, comparisons):
        axis.imshow(mpimg.imread(image_path))
        axis.set_title(bag_name, fontsize=9)
        axis.axis("off")
    for axis in axes_flat[len(comparisons) :]:
        axis.axis("off")
    fig.suptitle("2D SLAM baseline / post map comparison")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def _rename_exchange(first: Path, second: Path) -> bool:
    """Atomically exchange two directories when Linux renameat2 is available."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 2
    if (
        renameat2(
            at_fdcwd,
            os.fsencode(first),
            at_fdcwd,
            os.fsencode(second),
            rename_exchange,
        )
        == 0
    ):
        return True
    error = ctypes.get_errno()
    if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP, errno.EXDEV}:
        return False
    raise OSError(error, os.strerror(error), f"{first} <-> {second}")


def _publish_review_directory(staged: Path, output_root: Path) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if not output_root.exists():
        os.replace(staged, output_root)
        return
    if not output_root.is_dir():
        raise RuntimeError(f"quick-review output is not a directory: {output_root}")
    if _rename_exchange(staged, output_root):
        # staged now names the previous published review.
        shutil.rmtree(staged, ignore_errors=True)
        return

    backup = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.backup-",
            dir=output_root.parent,
        )
    )
    backup.rmdir()
    os.replace(output_root, backup)
    try:
        os.replace(staged, output_root)
    except BaseException:
        os.replace(backup, output_root)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _bag_directories(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def create_map_only_review(
    *,
    baseline_root: Path,
    post_root: Path,
    comparison_root: Path,
    output_root: Path,
    expected_bags: Sequence[str] | None = None,
) -> dict[str, int]:
    baseline_bags = baseline_root / "bags"
    post_bags = post_root / "bags"
    # Kept in the API for command compatibility. Comparison images are always
    # regenerated from the two current map.png inputs.
    _ = comparison_root
    if expected_bags is None:
        names = sorted(
            _bag_directories(baseline_bags) | _bag_directories(post_bags)
        )
    else:
        names = sorted(set(expected_bags))
    if not names:
        raise RuntimeError("quick-review has no expected bag maps")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staged-",
            dir=output_root.parent,
        )
    )
    counts = {"baseline": 0, "post": 0, "comparison": 0}
    comparisons: list[tuple[str, Path]] = []
    published = False
    try:
        for directory_name in ("base_line", "post", "comparison"):
            (staged / directory_name).mkdir(parents=True)
        for name in names:
            baseline_map = baseline_bags / name / "map.png"
            post_map = post_bags / name / "map.png"
            missing = [
                str(path)
                for path in (baseline_map, post_map)
                if not path.is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"missing paired map for {name}: {', '.join(missing)}"
                )
            _copy_map(baseline_map, staged / "base_line" / f"{name}.png")
            counts["baseline"] += 1
            _copy_map(post_map, staged / "post" / f"{name}.png")
            counts["post"] += 1
            comparison_output = staged / "comparison" / f"{name}.png"
            if not _write_side_by_side(
                comparison_output,
                name,
                baseline_map,
                post_map,
            ):
                raise RuntimeError(f"failed to render paired map for {name}")
            counts["comparison"] += 1
            comparisons.append((name, comparison_output))
        expected_count = len(names)
        if any(count != expected_count for count in counts.values()):
            raise RuntimeError(
                f"incomplete quick-review render: expected={expected_count}, "
                f"counts={counts}"
            )
        contact_sheet = staged / "00_all_bags_contact_sheet.png"
        _write_contact_sheet(comparisons, contact_sheet)
        if not contact_sheet.is_file():
            raise RuntimeError("failed to render quick-review contact sheet")
        _publish_review_directory(staged, output_root)
        published = True
        return counts
    finally:
        if not published and staged.exists():
            shutil.rmtree(staged, ignore_errors=True)


def _write_commands(path: Path, commands: Iterable[Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n\n".join(shlex.join(list(command)) for command in commands) + "\n",
        encoding="utf-8",
    )


def _resolve_bag_roots(values: Sequence[str] | None) -> list[Path]:
    return [Path(value) for value in values] if values else list(DEFAULT_BAG_ROOTS)


def run_regression(args: argparse.Namespace) -> int:
    roots = _resolve_bag_roots(args.bag_root)
    records = audit_storage_integrity(
        prepare_staging(
            discover_bags(roots),
            Path(args.staging_root),
            reindex=not args.no_reindex,
        )
    )
    output_root = Path(args.output).expanduser().resolve()
    quick_root = Path(args.quick_output).expanduser().resolve()
    write_manifest(records, output_root)
    usable = [record for record in records if record.usable]
    if args.bag_name:
        requested = set(args.bag_name)
        usable = [record for record in usable if record.name in requested]
        missing = sorted(requested - {record.name for record in usable})
        if missing:
            raise RuntimeError(f"requested bags are not usable: {', '.join(missing)}")
    bag_names = [record.name for record in usable]
    if not bag_names:
        raise RuntimeError("no usable bags were found")
    replay_imu_qos_depth = (
        args.replay_imu_qos_depth
        if args.replay_imu_qos_depth > 0
        else max(record.imu_count for record in usable) + 1024
    )
    replay_imu_buffer_duration_s = (
        args.replay_imu_buffer_duration_s
        if args.replay_imu_buffer_duration_s > 0.0
        else max(record.duration_s for record in usable) + 60.0
    )
    replay_max_pending_scans = max(
        args.replay_max_pending_scans,
        max(record.scan_laserscan_count for record in usable) + 1024,
    )
    source_slam_config = Path(args.slam_config).expanduser().resolve()
    replay_slam_config = prepare_replay_slam_config(
        source_slam_config,
        output_root / "configs" / "mapper_replay.yaml",
        scan_queue_size=args.replay_scan_queue_size,
        tf_buffer_duration_s=args.replay_tf_buffer_duration_s,
        imu_buffer_duration_s=replay_imu_buffer_duration_s,
        imu_qos_depth=replay_imu_qos_depth,
        diagnostics_period_s=args.replay_diagnostics_period_s,
        pending_timeout_s=args.queue_drain_timeout_s,
        max_pending_scans=replay_max_pending_scans,
    )

    rate_suffix = _rate_label(args.rate)
    baseline_label = args.baseline_label or f"post_before_spatial_passage_{rate_suffix}"
    post_label = args.post_label or f"spatial_passage_candidate_{rate_suffix}"
    stage_root = output_root / "stage34_roughness_gate_replay"
    baseline_root = stage_root / "variant" / baseline_label
    post_root = stage_root / "variant" / post_label
    comparison_root = stage_root / "comparisons" / post_label
    common = {
        "pipeline": Path(args.pipeline).expanduser().resolve(),
        "setup_bash": Path(args.setup_bash).expanduser().resolve(),
        "staging_root": Path(args.staging_root).expanduser().resolve(),
        "output_root": output_root,
        "slam_config": replay_slam_config,
        "bag_names": bag_names,
        "rate": args.rate,
        "skip_existing": args.skip_existing,
        "ros_domain_base": args.ros_domain_base,
        "scan_qos_depth": args.scan_qos_depth,
        "post_playback_wait_s": args.post_playback_wait_s,
        "queue_drain_timeout_s": args.queue_drain_timeout_s,
        "startup_tf_drop_limit": args.startup_tf_drop_limit,
    }
    baseline_command = build_stage34_command(
        **common,
        snapshot_label=baseline_label,
        spatial_passage=False,
        reference_root=None,
    )
    post_command = build_stage34_command(
        **common,
        snapshot_label=post_label,
        spatial_passage=True,
        reference_root=baseline_root,
    )
    _write_commands(output_root / "commands.txt", (baseline_command, post_command))
    run_metadata = {
        "bag_roots": [str(path.expanduser().resolve()) for path in roots],
        "bag_count_discovered": len(records),
        "bag_count_usable": len(usable),
        "bag_names": bag_names,
        "rate": args.rate,
        "scan_qos_depth": args.scan_qos_depth,
        "post_playback_wait_s": args.post_playback_wait_s,
        "queue_drain_timeout_s": args.queue_drain_timeout_s,
        "startup_tf_drop_limit": args.startup_tf_drop_limit,
        "source_slam_config": str(source_slam_config),
        "replay_slam_config": str(replay_slam_config),
        "replay_scan_queue_size": args.replay_scan_queue_size,
        "replay_tf_buffer_duration_s": args.replay_tf_buffer_duration_s,
        "replay_imu_buffer_duration_s": replay_imu_buffer_duration_s,
        "replay_imu_qos_depth": replay_imu_qos_depth,
        "replay_diagnostics_period_s": args.replay_diagnostics_period_s,
        "replay_pending_timeout_s": args.queue_drain_timeout_s,
        "replay_max_pending_scans": replay_max_pending_scans,
        "baseline_label": baseline_label,
        "post_label": post_label,
        "quick_output": str(quick_root),
        "phase": args.phase,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        print(shlex.join(baseline_command))
        print(shlex.join(post_command))
        return 0
    if args.phase in {"all", "baseline"}:
        subprocess.run(baseline_command, check=True)
    if args.phase in {"all", "post"}:
        subprocess.run(post_command, check=True)
    if args.phase in {"all", "quick"}:
        counts = create_map_only_review(
            baseline_root=baseline_root,
            post_root=post_root,
            comparison_root=comparison_root,
            output_root=quick_root,
            expected_bags=bag_names,
        )
        print(json.dumps(counts, sort_keys=True))
    return 0


def inventory(args: argparse.Namespace) -> int:
    records = audit_storage_integrity(
        prepare_staging(
            discover_bags(_resolve_bag_roots(args.bag_root)),
            Path(args.staging_root),
            reindex=not args.no_reindex,
        )
    )
    write_manifest(records, Path(args.output).expanduser().resolve())
    print(
        json.dumps(
            {
                "total": len(records),
                "usable": sum(record.usable for record in records),
                "unusable": sum(not record.usable for record in records),
            },
            sort_keys=True,
        )
    )
    return 0


def quick_review(args: argparse.Namespace) -> int:
    expected_bags = (
        expected_bag_names_from_manifest(
            Path(args.manifest).expanduser().resolve()
        )
        if args.manifest
        else None
    )
    counts = create_map_only_review(
        baseline_root=Path(args.baseline_root).expanduser().resolve(),
        post_root=Path(args.post_root).expanduser().resolve(),
        comparison_root=Path(args.comparison_root).expanduser().resolve(),
        output_root=Path(args.output).expanduser().resolve(),
        expected_bags=expected_bags,
    )
    print(json.dumps(counts, sort_keys=True))
    return 0


def expected_bag_names_from_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"manifest records must be a list: {path}")
    return sorted(
        str(record["name"])
        for record in records
        if isinstance(record, dict)
        and record.get("status") in {"ready", "reindexed"}
        and record.get("storage_integrity_status") == "passed"
    )


def _replay_flag(result: dict[str, object], key: str) -> int:
    try:
        return int(result.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _load_json_object(path: Path) -> tuple[dict[str, object], str]:
    if not path.is_file():
        return {}, f"missing_{path.stem}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{path.stem}_read_error:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return {}, f"{path.stem}_not_object"
    return payload, ""


def _canonical_payload_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_names_for_schema(schema: int) -> tuple[str, ...] | None:
    if schema == 3:
        return REPLAY_ARTIFACT_NAMES_V3
    if schema == 4:
        return REPLAY_ARTIFACT_NAMES_V4
    if schema == 5:
        return REPLAY_ARTIFACT_NAMES_V5
    return None


def _artifact_manifest(
    bag_root: Path,
    names: Sequence[str],
) -> dict[str, dict[str, object]]:
    manifest: dict[str, dict[str, object]] = {}
    for name in names:
        path = bag_root / name
        if not path.is_file():
            continue
        manifest[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return manifest


def _required_artifacts(
    schema: int,
    result: dict[str, object],
    contract: dict[str, object],
) -> set[str]:
    names = _artifact_names_for_schema(schema)
    if names is None:
        return set()
    required = set(REPLAY_ARTIFACT_NAMES_V3)
    if schema >= 4:
        required.update(REPLAY_ARTIFACT_NAMES_V4)
    if schema < 5:
        return required

    # V5 diagnostics are conditional in the saved pipeline schema. Runtime
    # SLAM evidence is mandatory for this strict audit; inactive guard outputs
    # are required only when the corresponding node was enabled.
    required.add("runtime_parameters/slam_toolbox.yaml")
    if _replay_flag(result, "enable_laserscan_roughness_gate") == 1:
        required.update(
            {
                "guard_status_counts.csv",
                "guard_status_events.csv",
                "runtime_parameters/slam_laserscan_roughness_gate.yaml",
            }
        )
    profile = contract.get("roughness_innovation_profile", {})
    innovation_enabled = (
        isinstance(profile, dict) and profile.get("enabled") is True
    ) or result.get("runtime_slam_actual_enable_roughness_innovation_gate") is True
    if innovation_enabled:
        required.add("roughness_innovation_events.csv")
        post_confirmation_enabled = (
            isinstance(profile, dict)
            and profile.get("enable_post_confirmation") is True
        ) or result.get("runtime_slam_actual_roughness_enable_post_confirmation") is True
        if post_confirmation_enabled:
            required.add("roughness_post_confirmation_events.csv")
    return required


def _integrity_file_record(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve() if str(path) else path
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": int(exists),
        "size_bytes": resolved.stat().st_size if exists else 0,
        "sha256": _file_sha256(resolved),
    }


def _replay_integrity_manifest(
    bag_root: Path,
    result: dict[str, object],
) -> dict[str, dict[str, object]]:
    paths = {
        "replay_result": bag_root / "replay_result.json",
        "commands": bag_root / "commands.txt",
        "pose_graph_export_log": bag_root / "pose_graph_export.log",
        "launch_log": Path(str(result.get("launch_log", ""))),
        "capture_log": Path(str(result.get("capture_log", ""))),
    }
    return {name: _integrity_file_record(path) for name, path in paths.items()}


def _verify_replay_evidence(
    bag_root: Path,
    result: dict[str, object],
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "failures": [],
        "replay_provenance_status": "not_checked",
        "replay_signature_match": 0,
        "artifact_manifest_status": "not_checked",
        "replay_integrity_status": "not_checked",
        "fairness_contract_status": "not_checked",
        "fairness_contract": {},
    }
    failures = evidence["failures"]
    assert isinstance(failures, list)
    provenance, provenance_error = _load_json_object(
        bag_root / "replay_provenance.json"
    )
    if provenance_error:
        evidence["replay_provenance_status"] = provenance_error
        failures.append(provenance_error)
        return evidence

    try:
        schema = int(provenance.get("schema", 0) or 0)
    except (TypeError, ValueError):
        schema = 0
    names = _artifact_names_for_schema(schema)
    if names is None:
        evidence["replay_provenance_status"] = (
            f"unsupported_replay_provenance_schema:{schema}"
        )
        failures.append("unsupported_replay_provenance_schema")
        return evidence
    try:
        saved_schema = int(result.get("replay_provenance_schema", 0) or 0)
    except (TypeError, ValueError):
        saved_schema = 0
    if saved_schema != schema:
        evidence["replay_provenance_status"] = "replay_provenance_schema_mismatch"
        failures.append("replay_provenance_schema_mismatch")
    else:
        evidence["replay_provenance_status"] = "valid"

    signature = _canonical_payload_sha256(provenance)
    signature_match = result.get("replay_signature_sha256") == signature
    evidence["replay_signature_match"] = int(signature_match)
    if not signature_match:
        failures.append("replay_signature_mismatch")

    contract = provenance.get("fairness_contract")
    if not isinstance(contract, dict):
        evidence["fairness_contract_status"] = "missing_fairness_contract"
        failures.append("missing_fairness_contract")
        contract = {}
    elif result.get("replay_fairness_contract") != contract:
        evidence["fairness_contract_status"] = "embedded_contract_mismatch"
        failures.append("embedded_fairness_contract_mismatch")
    elif result.get("replay_fairness_contract_sha256") != _canonical_payload_sha256(
        contract
    ):
        evidence["fairness_contract_status"] = "contract_hash_mismatch"
        failures.append("fairness_contract_hash_mismatch")
    else:
        evidence["fairness_contract_status"] = "valid"
    evidence["fairness_contract"] = contract

    saved_manifest = result.get("artifact_manifest")
    current_manifest = _artifact_manifest(bag_root, names)
    required = _required_artifacts(schema, result, contract)
    missing_required = sorted(required - set(current_manifest))
    if not isinstance(saved_manifest, dict):
        evidence["artifact_manifest_status"] = "missing_artifact_manifest"
        failures.append("missing_artifact_manifest")
    elif missing_required:
        evidence["artifact_manifest_status"] = (
            "missing_required_artifacts:" + ",".join(missing_required)
        )
        failures.append("missing_required_artifacts")
    elif saved_manifest != current_manifest:
        evidence["artifact_manifest_status"] = "artifact_manifest_mismatch"
        failures.append("artifact_manifest_mismatch")
    else:
        evidence["artifact_manifest_status"] = "valid"

    if schema < 4:
        evidence["replay_integrity_status"] = "not_required_for_schema"
        return evidence
    integrity, integrity_error = _load_json_object(
        bag_root / "replay_integrity.json"
    )
    if integrity_error:
        evidence["replay_integrity_status"] = integrity_error
        failures.append(integrity_error)
    elif integrity.get("schema") != 1:
        evidence["replay_integrity_status"] = "unsupported_integrity_schema"
        failures.append("unsupported_replay_integrity_schema")
    elif integrity.get("files") != _replay_integrity_manifest(bag_root, result):
        evidence["replay_integrity_status"] = "replay_integrity_mismatch"
        failures.append("replay_integrity_mismatch")
    else:
        evidence["replay_integrity_status"] = "valid"
    return evidence


def _strict_replay_failures(
    bag_name: str,
    result: dict[str, object],
) -> list[str]:
    failures: list[str] = []
    if result.get("bag") != bag_name:
        failures.append("replay_bag_name_mismatch")
    status = str(result.get("status", ""))
    if status not in ACCEPTED_REPLAY_STATUSES:
        failures.append("status_not_accepted")

    required_one = {
        "attempted": "replay_not_attempted",
        "bag_playback_finished_observed": "playback_not_finished",
        "runtime_slam_parameter_match": "runtime_parameter_mismatch",
        "slam_input_complete": "incomplete_slam_input",
        "slam_input_log_complete": "incomplete_slam_input_log",
        "slam_recorded_scan_delivery_required": "scan_delivery_not_required",
        "slam_recorded_scan_delivery_complete": "incomplete_scan_delivery",
        "slam_recorded_scan_metadata_count_match": "scan_metadata_count_mismatch",
        "slam_recorded_scan_stamp_audit_available": "missing_scan_stamp_audit",
        "slam_recorded_scan_stamp_fingerprint_match": "scan_fingerprint_mismatch",
        "slam_recorded_scan_stamp_sequence_complete": "scan_sequence_mismatch",
        "roughness_imu_delivery_complete": "incomplete_imu_delivery",
        "roughness_imu_delivery_fingerprint_match": "imu_fingerprint_mismatch",
        "subscription_drain_complete": "incomplete_subscription_drain",
        "subscription_finalize_pending_success": "pending_finalize_failed",
        "queue_drain_enabled": "queue_drain_disabled",
        "queue_drain_complete": "incomplete_queue_drain",
        "pose_graph_captured": "pose_graph_not_captured",
        "pose_graph_coverage_complete": "incomplete_pose_graph_coverage",
    }
    for key, reason in required_one.items():
        if _replay_flag(result, key) != 1:
            failures.append(reason)

    required_zero = {
        "launch_return_code_after_stop": "launch_return_code_nonzero",
        "capture_return_code_after_stop": "capture_return_code_nonzero",
        "runtime_slam_parameter_dump_return_code": "runtime_dump_failed",
        "runtime_slam_parameter_mismatch_count": "runtime_parameter_mismatch_count",
        "slam_recorded_scan_missing_count": "recorded_scan_missing",
        "slam_recorded_scan_extra_count": "recorded_scan_extra",
        "slam_recorded_scan_unmatched_startup_stamp_count": (
            "unmatched_startup_scan"
        ),
        "slam_unexpected_scan_drop_count": "unexpected_scan_drop",
        "roughness_imu_delivery_count_delta": "imu_delivery_count_delta",
        "slam_queue_direct_latest_pending_depth": "pending_queue_not_empty",
        "slam_queue_direct_latest_scan_depth": "scan_queue_not_empty",
        "slam_queue_direct_post_playback_pending_depth": (
            "post_playback_pending_queue_not_empty"
        ),
        "slam_queue_direct_post_playback_scan_depth": (
            "post_playback_scan_queue_not_empty"
        ),
        "pose_graph_serialize_return_code": "pose_graph_serialize_failed",
        "pose_graph_serialize_service_result": "pose_graph_service_failed",
        "pose_graph_export_return_code": "pose_graph_export_failed",
    }
    for key, reason in required_zero.items():
        try:
            value = int(result.get(key, -1))
        except (TypeError, ValueError):
            value = -1
        if value != 0:
            failures.append(reason)

    expected_statuses = {
        "runtime_slam_parameter_status": {"matched"},
        "slam_recorded_scan_bag_audit_status": {"valid"},
        "slam_recorded_scan_delivery_status": {"complete"},
        "queue_drain_status": {"drained_zero_depth"},
        "queue_drain_evidence": {"direct_node_diagnostics"},
    }
    for key, expected in expected_statuses.items():
        if result.get(key) not in expected:
            failures.append(f"invalid_{key}")
    for key in ("slam_input_status", "slam_input_log_status"):
        if not str(result.get(key, "")).startswith("complete"):
            failures.append(f"invalid_{key}")

    count_pairs = (
        (
            "subscription_drain_effective_expected_scan_count",
            "subscription_drain_received_scan_count",
            "subscription_scan_count_mismatch",
        ),
        (
            "subscription_drain_expected_imu_count",
            "subscription_drain_received_imu_count",
            "subscription_imu_count_mismatch",
        ),
        (
            "roughness_imu_delivery_expected_count",
            "roughness_imu_delivery_received_count",
            "imu_delivery_count_mismatch",
        ),
    )
    for expected_key, received_key, reason in count_pairs:
        if result.get(expected_key) != result.get(received_key):
            failures.append(reason)

    stationary = status == "map_and_stationary_pose_graph"
    expected_coverage = (
        "stationary_single_node_complete" if stationary else "complete"
    )
    if result.get("pose_graph_coverage_status") != expected_coverage:
        failures.append("invalid_pose_graph_coverage_status")
    if stationary:
        if _replay_flag(result, "pose_graph_stationary_single_node") != 1:
            failures.append("stationary_pose_graph_not_confirmed")
    elif _replay_flag(result, "pose_graph_motion_gap_complete") != 1:
        failures.append("incomplete_pose_graph_motion_coverage")
    try:
        trajectory_count = int(result.get("pose_graph_trajectory_count", 0) or 0)
    except (TypeError, ValueError):
        trajectory_count = 0
    if trajectory_count < (1 if stationary else 2):
        failures.append("insufficient_pose_graph_trajectory")
    return failures


def _new_audit_row(variant: str, bag_name: str) -> dict[str, object]:
    row = {column: "" for column in REPLAY_AUDIT_COLUMNS}
    row.update({"variant": variant, "bag": bag_name, "audit_pass": 0})
    return row


def audit_replay_variant(
    variant: str,
    variant_root: Path,
    expected_bags: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    bags_root = variant_root / "bags"
    expected_set = set(expected_bags)
    actual_set = _bag_directories(bags_root)
    for bag_name in expected_bags:
        bag_root = bags_root / bag_name
        replay_result = bag_root / "replay_result.json"
        map_png = bag_root / "map.png"
        result, load_error = _load_json_object(replay_result)
        status = str(result.get("status", ""))
        evidence = _verify_replay_evidence(bag_root, result) if result else {
            "failures": [],
            "replay_provenance_status": "not_checked",
            "replay_signature_match": 0,
            "artifact_manifest_status": "not_checked",
            "replay_integrity_status": "not_checked",
            "fairness_contract_status": "not_checked",
        }
        failures = ([load_error] if load_error else [])
        if result:
            failures.extend(_strict_replay_failures(bag_name, result))
            failures.extend(str(item) for item in evidence["failures"])
        if not map_png.is_file():
            failures.append("missing_map")
        row = _new_audit_row(variant, bag_name)
        row.update(
            {
                "variant": variant,
                "bag": bag_name,
                "audit_pass": int(not failures),
                "failure_reasons": ";".join(failures),
                "status": status,
                "map_present": int(map_png.is_file()),
                "runtime_parameter_match": _replay_flag(
                    result, "runtime_slam_parameter_match"
                ),
                "slam_input_complete": _replay_flag(
                    result, "slam_input_complete"
                ),
                "scan_delivery_complete": _replay_flag(
                    result, "slam_recorded_scan_delivery_complete"
                ),
                "imu_delivery_complete": _replay_flag(
                    result, "roughness_imu_delivery_complete"
                ),
                "imu_fingerprint_match": _replay_flag(
                    result,
                    "roughness_imu_delivery_fingerprint_match",
                ),
                "queue_drain_complete": _replay_flag(
                    result, "queue_drain_complete"
                ),
                "queue_drain_status": result.get("queue_drain_status", ""),
                "queue_drain_evidence": result.get(
                    "queue_drain_evidence", ""
                ),
                "pose_graph_captured": _replay_flag(
                    result, "pose_graph_captured"
                ),
                "pose_graph_coverage_complete": _replay_flag(
                    result, "pose_graph_coverage_complete"
                ),
                "pose_graph_coverage_status": result.get(
                    "pose_graph_coverage_status", ""
                ),
                "pose_graph_trajectory_count": result.get(
                    "pose_graph_trajectory_count", ""
                ),
                "stationary_replay": _replay_flag(
                    result, "stationary_replay"
                ),
                "trajectory_map_frame_path_length_m": result.get(
                    "trajectory_map_frame_path_length_m", ""
                ),
                "trajectory_odom_frame_path_length_m": result.get(
                    "trajectory_odom_frame_path_length_m", ""
                ),
                "roughness_received_scan_count": result.get(
                    "roughness_innovation_received_scan_count", ""
                ),
                "roughness_received_imu_count": result.get(
                    "roughness_imu_delivery_received_count", ""
                ),
                "roughness_expected_imu_count": result.get(
                    "roughness_imu_delivery_expected_count", ""
                ),
                "roughness_spatial_passage_count": result.get(
                    "roughness_near_chain_spatial_passage_count", ""
                ),
                "roughness_recovery_count": result.get(
                    "roughness_near_chain_recovery_count", ""
                ),
                "subscription_drain_complete": _replay_flag(
                    result, "subscription_drain_complete"
                ),
                "subscription_finalize_pending_success": _replay_flag(
                    result, "subscription_finalize_pending_success"
                ),
                "replay_provenance_status": evidence[
                    "replay_provenance_status"
                ],
                "replay_signature_match": evidence["replay_signature_match"],
                "artifact_manifest_status": evidence[
                    "artifact_manifest_status"
                ],
                "replay_integrity_status": evidence[
                    "replay_integrity_status"
                ],
                "fairness_contract_status": evidence[
                    "fairness_contract_status"
                ],
                "fairness_mismatch_fields": "",
                "replay_result_json": str(replay_result),
                "map_png": str(map_png),
            }
        )
        rows.append(row)

    unexpected = sorted(actual_set - expected_set)
    for bag_name in unexpected:
        row = _new_audit_row(variant, bag_name)
        row.update(
            {
                "failure_reasons": "unexpected_bag_directory",
                "replay_result_json": str(
                    bags_root / bag_name / "replay_result.json"
                ),
                "map_png": str(bags_root / bag_name / "map.png"),
            }
        )
        rows.append(row)

    failures = [row for row in rows if int(row["audit_pass"]) != 1]
    summary: dict[str, object] = {
        "variant": variant,
        "variant_root": str(variant_root),
        "expected_bag_count": len(expected_bags),
        "actual_bag_count": len(actual_set),
        "empty_expected_usable_set": int(not expected_bags),
        "unexpected_bag_count": len(unexpected),
        "unexpected_bags": unexpected,
        "missing_bag_count": len(expected_set - actual_set),
        "missing_bags": sorted(expected_set - actual_set),
        "passed_bag_count": len(rows) - len(failures),
        "failed_bag_count": len(failures),
        "failed_bags": [str(row["bag"]) for row in failures],
    }
    return rows, summary


def _flatten_contract(value: object, prefix: str = "") -> dict[str, object]:
    if not isinstance(value, dict):
        return {prefix or "<root>": value}
    flattened: dict[str, object] = {}
    for key in sorted(value):
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        child = value[key]
        if isinstance(child, dict):
            flattened.update(_flatten_contract(child, child_prefix))
        else:
            flattened[child_prefix] = child
    return flattened


def _is_explicit_false(value: object) -> bool:
    if value is False:
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 0
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false"}
    return False


def _uses_direct_internal_laserscan(
    result: dict[str, object],
    provenance: dict[str, object],
    contract: dict[str, object],
) -> bool:
    contract_input = contract.get("input")
    if not isinstance(contract_input, dict):
        return False
    launch_command = provenance.get("launch_command")
    if not isinstance(launch_command, str):
        return False
    try:
        launch_tokens = set(shlex.split(launch_command))
    except ValueError:
        return False
    return (
        _is_explicit_false(result.get("enable_laserscan_roughness_gate"))
        and _is_explicit_false(result.get("start_pointcloud_projection"))
        and result.get("scan_topic") == "/scan_laserscan"
        and result.get("slam_scan_topic") == "/scan_laserscan"
        and _is_explicit_false(contract_input.get("start_pointcloud_projection"))
        and contract_input.get("scan_topic") == "/scan_laserscan"
        and "enable_laserscan_roughness_gate:=false" in launch_tokens
        and "start_pointcloud_projection:=false" in launch_tokens
        and "scan_topic:=/scan_laserscan" in launch_tokens
        and "slam_scan_topic:=/scan_laserscan" in launch_tokens
    )


def _storage_content_identity(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    identity: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, dict):
            return None
        if not all(key in entry for key in ("name", "size_bytes", "sha256")):
            return None
        identity.append(
            {
                "name": entry["name"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
        )
    return identity


def _normalize_proven_equivalent_fairness_fields(
    contracts: list[dict[str, object]],
    results: list[dict[str, object]],
    provenances: list[dict[str, object]],
) -> list[str]:
    equivalent_fields: list[str] = []

    bag_entries: list[dict[str, object]] = []
    for contract in contracts:
        bag = contract.get("bag")
        if not isinstance(bag, dict):
            bag_entries = []
            break
        bag_entries.append(bag)
    if len(bag_entries) == 2:
        fingerprints = [entry.get("storage_fingerprint") for entry in bag_entries]
        identities = [_storage_content_identity(value) for value in fingerprints]
        if (
            identities[0] is not None
            and identities[0] == identities[1]
            and fingerprints[0] != fingerprints[1]
        ):
            bag_entries[0]["storage_fingerprint"] = identities[0]
            bag_entries[1]["storage_fingerprint"] = identities[1]
            equivalent_fields.append("bag.storage_fingerprint[*].mtime_ns")

    if all(
        _uses_direct_internal_laserscan(result, provenance, contract)
        for result, provenance, contract in zip(results, provenances, contracts)
    ):
        backends = [contract.get("backend_asset_sha256") for contract in contracts]
        if all(isinstance(backend, dict) for backend in backends):
            for key in INACTIVE_LASERSCAN_GUARD_ASSET_FIELDS:
                values = [backend.get(key) for backend in backends]
                if values[0] != values[1]:
                    equivalent_fields.append(f"backend_asset_sha256.{key}")
                for backend in backends:
                    backend.pop(key, None)

            launch_commands = [
                str(provenance.get("launch_command", ""))
                for provenance in provenances
            ]
            if all("slam_from_bag_launch.py" not in command for command in launch_commands):
                values = [backend.get("slam_from_bag_launch") for backend in backends]
                if values[0] != values[1]:
                    equivalent_fields.append(
                        "backend_asset_sha256.slam_from_bag_launch"
                    )
                for backend in backends:
                    backend.pop("slam_from_bag_launch", None)

    return sorted(equivalent_fields)


def _pair_fairness_status(
    baseline_root: Path,
    post_root: Path,
    bag_name: str,
) -> tuple[str, list[str], list[str]]:
    contracts: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    provenances: list[dict[str, object]] = []
    for root in (baseline_root, post_root):
        bag_root = root / "bags" / bag_name
        result, result_error = _load_json_object(bag_root / "replay_result.json")
        provenance, provenance_error = _load_json_object(
            bag_root / "replay_provenance.json"
        )
        contract = provenance.get("fairness_contract")
        if (
            result_error
            or provenance_error
            or not isinstance(contract, dict)
            or result.get("replay_signature_sha256")
            != _canonical_payload_sha256(provenance)
            or result.get("replay_fairness_contract") != contract
            or result.get("replay_fairness_contract_sha256")
            != _canonical_payload_sha256(contract)
        ):
            return "unverifiable", [], []
        results.append(result)
        provenances.append(provenance)
        contracts.append(copy.deepcopy(contract))

    equivalent_fields = _normalize_proven_equivalent_fairness_fields(
        contracts,
        results,
        provenances,
    )

    baseline_flat = _flatten_contract(contracts[0])
    post_flat = _flatten_contract(contracts[1])
    fields = sorted(set(baseline_flat) | set(post_flat))
    mismatches = [
        field
        for field in fields
        if (
            field not in baseline_flat
            or field not in post_flat
            or _canonical_payload_sha256(baseline_flat[field])
            != _canonical_payload_sha256(post_flat[field])
        )
    ]
    if mismatches != [ALLOWED_FAIRNESS_DIFFERENCE]:
        return "unexpected_difference", mismatches, equivalent_fields
    if (
        baseline_flat.get(ALLOWED_FAIRNESS_DIFFERENCE) is not False
        or post_flat.get(ALLOWED_FAIRNESS_DIFFERENCE) is not True
    ):
        return "invalid_spatial_passage_direction", mismatches, equivalent_fields
    return "matched_intended_difference", mismatches, equivalent_fields


def _append_audit_failure(row: dict[str, object], reason: str) -> None:
    reasons = [item for item in str(row.get("failure_reasons", "")).split(";") if item]
    if reason not in reasons:
        reasons.append(reason)
    row["failure_reasons"] = ";".join(reasons)
    row["audit_pass"] = 0


def _refresh_variant_summary(
    rows: Sequence[dict[str, object]],
    summary: dict[str, object],
) -> None:
    failures = [row for row in rows if int(row["audit_pass"]) != 1]
    summary["passed_bag_count"] = len(rows) - len(failures)
    summary["failed_bag_count"] = len(failures)
    summary["failed_bags"] = [str(row["bag"]) for row in failures]


def audit_results(args: argparse.Namespace) -> int:
    expected_bags = expected_bag_names_from_manifest(
        Path(args.manifest).expanduser().resolve()
    )
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    post_root = Path(args.post_root).expanduser().resolve()
    baseline_rows, baseline_summary = audit_replay_variant(
        "baseline",
        baseline_root,
        expected_bags,
    )
    post_rows, post_summary = audit_replay_variant(
        "post",
        post_root,
        expected_bags,
    )
    baseline_by_bag = {str(row["bag"]): row for row in baseline_rows}
    post_by_bag = {str(row["bag"]): row for row in post_rows}
    fairness_failed: list[str] = []
    for bag_name in expected_bags:
        status, mismatch_fields, equivalent_fields = _pair_fairness_status(
            baseline_root,
            post_root,
            bag_name,
        )
        fields_text = ";".join(mismatch_fields)
        for row in (baseline_by_bag[bag_name], post_by_bag[bag_name]):
            row["fairness_contract_status"] = status
            row["fairness_mismatch_fields"] = fields_text
            row["fairness_equivalent_fields"] = ";".join(equivalent_fields)
            if status != "matched_intended_difference":
                _append_audit_failure(row, f"fairness_{status}")
        if status != "matched_intended_difference":
            fairness_failed.append(bag_name)
    _refresh_variant_summary(baseline_rows, baseline_summary)
    _refresh_variant_summary(post_rows, post_summary)
    rows = baseline_rows + post_rows
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "replay_integrity_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLAY_AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "expected_bag_count": len(expected_bags),
        "empty_expected_usable_set": int(not expected_bags),
        "overall_complete": int(
            bool(expected_bags)
            and int(baseline_summary["failed_bag_count"]) == 0
            and int(post_summary["failed_bag_count"]) == 0
            and not fairness_failed
        ),
        "baseline": baseline_summary,
        "post": post_summary,
        "fairness": {
            "allowed_difference": ALLOWED_FAIRNESS_DIFFERENCE,
            "expected_direction": "baseline=false,post=true",
            "passed_bag_count": len(expected_bags) - len(fairness_failed),
            "failed_bag_count": len(fairness_failed),
            "failed_bags": fairness_failed,
        },
        "audit_csv": str(csv_path),
    }
    summary_path = output / "replay_integrity_audit.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if int(summary["overall_complete"]) == 1 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--bag-root", action="append")
    inventory_parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    inventory_parser.add_argument("--output", required=True)
    inventory_parser.add_argument("--no-reindex", action="store_true")
    inventory_parser.set_defaults(func=inventory)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--bag-root", action="append")
    run_parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    run_parser.add_argument("--output", default=str(DEFAULT_OUTPUT_ROOT))
    run_parser.add_argument("--quick-output", default=str(DEFAULT_QUICK_ROOT))
    run_parser.add_argument("--pipeline", default=str(DEFAULT_PIPELINE))
    run_parser.add_argument("--setup-bash", default=str(DEFAULT_SETUP_BASH))
    run_parser.add_argument("--slam-config", default=str(DEFAULT_SLAM_CONFIG))
    run_parser.add_argument("--rate", type=float, default=5.0)
    run_parser.add_argument("--ros-domain-base", type=int, default=180)
    run_parser.add_argument("--scan-qos-depth", type=int, default=20000)
    run_parser.add_argument("--post-playback-wait-s", type=float, default=60.0)
    run_parser.add_argument("--queue-drain-timeout-s", type=float, default=600.0)
    run_parser.add_argument("--startup-tf-drop-limit", type=int, default=10)
    run_parser.add_argument("--replay-scan-queue-size", type=int, default=20000)
    run_parser.add_argument(
        "--replay-tf-buffer-duration-s", type=float, default=1800.0
    )
    run_parser.add_argument(
        "--replay-imu-buffer-duration-s",
        type=float,
        default=0.0,
        help="Replay IMU history in seconds; 0 selects longest bag plus margin.",
    )
    run_parser.add_argument(
        "--replay-imu-qos-depth",
        type=int,
        default=0,
        help="Reliable replay IMU queue depth; 0 selects largest bag count plus margin.",
    )
    run_parser.add_argument(
        "--replay-diagnostics-period-s", type=float, default=2.0
    )
    run_parser.add_argument(
        "--replay-max-pending-scans", type=int, default=20000
    )
    run_parser.add_argument("--baseline-label", default="")
    run_parser.add_argument("--post-label", default="")
    run_parser.add_argument("--bag-name", action="append")
    run_parser.add_argument(
        "--phase", choices=("all", "baseline", "post", "quick"), default="all"
    )
    run_parser.add_argument("--skip-existing", action="store_true")
    run_parser.add_argument("--no-reindex", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=run_regression)

    quick_parser = subparsers.add_parser("quick-review")
    quick_parser.add_argument("--baseline-root", required=True)
    quick_parser.add_argument("--post-root", required=True)
    quick_parser.add_argument("--comparison-root", required=True)
    quick_parser.add_argument("--manifest")
    quick_parser.add_argument("--output", default=str(DEFAULT_QUICK_ROOT))
    quick_parser.set_defaults(func=quick_review)

    audit_parser = subparsers.add_parser("audit-results")
    audit_parser.add_argument("--manifest", required=True)
    audit_parser.add_argument("--baseline-root", required=True)
    audit_parser.add_argument("--post-root", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.set_defaults(func=audit_results)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
