#!/usr/bin/env python3
"""Run resumable control/adaptive replays over the verified field manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


ADAPTIVE_PROFILE = (
    "roughness_innovation_profile.adaptive_roughness_constraint_profile"
)

ALLOWED_ADAPTIVE_DIFFERENCES = (
    "backend_asset_sha256.slam_config",
    f"{ADAPTIVE_PROFILE}.roughness_constraint_mode",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_calibrated",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_response_bad",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_response_good",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_translation_good_m",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_translation_bad_m",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_yaw_good_deg",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_yaw_bad_deg",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_xy_covariance_good_m2",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_xy_covariance_bad_m2",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_yaw_covariance_good_rad2",
    f"{ADAPTIVE_PROFILE}.roughness_near_chain_quality_yaw_covariance_bad_rad2",
    f"{ADAPTIVE_PROFILE}.roughness_adaptive_downweight_risk",
    f"{ADAPTIVE_PROFILE}.roughness_adaptive_reject_risk",
    f"{ADAPTIVE_PROFILE}.roughness_adaptive_dynamic_threshold_max",
)

ACCEPTED_STATUSES = {
    "map_and_map_trajectory",
    "map_and_stationary_pose_graph",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_replay_assets(
    args: argparse.Namespace,
    config: Path,
) -> dict[str, str]:
    workspace = args.workspace.expanduser().resolve()
    assets = {
        "lossless_adapter_sha256": file_sha256(args.pipeline),
        "runtime_slam_node_sha256": file_sha256(
            workspace
            / "install/slam_2d_core/lib/slam_2d_core/"
            "roughness_aware_slam_toolbox_node"
        ),
        "replay_pose_graph_exporter_sha256": file_sha256(
            workspace
            / "install/slam_2d_core/lib/slam_2d_core/"
            "slam_pose_graph_exporter"
        ),
        "runtime_cmake_cache_sha256": file_sha256(
            workspace / "build/slam_2d_core/CMakeCache.txt"
        ),
        "slam_config_sha256": file_sha256(config),
    }
    if any(len(value) != 64 for value in assets.values()):
        raise ValueError("current replay asset hash is incomplete")
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.home() / "gole_ws")
    parser.add_argument("--setup-bash", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--adaptive-config", type=Path, required=True)
    parser.add_argument(
        "--control-label", default="field41_intervention_off_release"
    )
    parser.add_argument(
        "--adaptive-label", default="field41_adaptive_quality_calibrated"
    )
    parser.add_argument(
        "--phase", choices=("control", "adaptive", "all"), default="all"
    )
    parser.add_argument("--bag-play-rate", type=float, default=5.0)
    parser.add_argument("--ros-domain-base", type=int, default=180)
    parser.add_argument("--ros-domain-stride", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--scan-qos-depth", type=int, default=20000)
    parser.add_argument("--queue-drain-timeout-s", type=float, default=600.0)
    parser.add_argument("--queue-drain-quiet-s", type=float, default=60.0)
    parser.add_argument("--startup-tf-drop-limit", type=int, default=10)
    parser.add_argument(
        "--bag",
        action="append",
        default=[],
        help="Run only this manifest bag; repeat to select multiple bags.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compatible-lossless-adapter-sha256",
        action="append",
        default=[],
        help=(
            "Explicitly allow a replay produced by this behavior-compatible "
            "lossless adapter hash; repeat for multiple audited versions."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def usable_bag_names(manifest: Path) -> list[str]:
    payload = json.loads(manifest.expanduser().read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    names = [
        str(row["name"])
        for row in records
        if isinstance(row, dict)
        and row.get("status") in {"ready", "reindexed"}
        and row.get("storage_integrity_status") == "passed"
    ]
    if not names:
        raise ValueError("manifest contains no usable bags")
    if len(names) != len(set(names)):
        raise ValueError("manifest contains duplicate usable bag names")
    return names


def usable_bag_durations(manifest: Path) -> dict[str, float]:
    payload = json.loads(manifest.expanduser().read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest records must be a list")
    durations: dict[str, float] = {}
    for row in records:
        if (
            not isinstance(row, dict)
            or row.get("status") not in {"ready", "reindexed"}
            or row.get("storage_integrity_status") != "passed"
        ):
            continue
        name = str(row["name"])
        try:
            duration = float(row.get("duration_s", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        durations[name] = max(duration, 0.0)
    return durations


def balanced_bag_shards(
    bag_names: list[str],
    durations: dict[str, float],
    jobs: int,
) -> list[list[str]]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if not bag_names:
        return []
    shard_count = min(jobs, len(bag_names))
    shards: list[list[str]] = [[] for _ in range(shard_count)]
    loads = [0.0] * shard_count
    manifest_order = {name: index for index, name in enumerate(bag_names)}
    ordered = sorted(
        bag_names,
        key=lambda value: (-durations.get(value, 0.0), manifest_order[value]),
    )
    for shard_index, name in enumerate(ordered[:shard_count]):
        shards[shard_index].append(name)
        loads[shard_index] += durations.get(name, 0.0)
    for name in ordered[shard_count:]:
        shard_index = min(
            range(shard_count),
            key=lambda index: (loads[index], index),
        )
        shards[shard_index].append(name)
        loads[shard_index] += durations.get(name, 0.0)
    for shard in shards:
        shard.sort(key=manifest_order.__getitem__)
    return shards


def shard_domain_bases(
    shards: list[list[str]],
    base: int,
    stride: int,
) -> list[int]:
    if base < 0 or stride < 1:
        raise ValueError("ROS domain base must be nonnegative and stride positive")
    bases = [base + index * stride for index in range(len(shards))]
    ranges: list[set[int]] = []
    for shard, shard_base in zip(shards, bases):
        domains = set(range(shard_base + 1, shard_base + len(shard) + 1))
        if domains and max(domains) > 230:
            raise ValueError("parallel replay ROS domains must remain in 0..230")
        if any(domains & prior for prior in ranges):
            raise ValueError("parallel replay ROS domain ranges overlap")
        ranges.append(domains)
    return bases


def select_bag_names(
    available: list[str],
    requested: list[str],
    limit: int,
) -> list[str]:
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    if len(requested) != len(set(requested)):
        raise ValueError("requested bag names must be unique")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("requested bags are not usable: " + ", ".join(unknown))
    selected = (
        [name for name in available if name in set(requested)]
        if requested else list(available)
    )
    return selected[:limit] if limit > 0 else selected


def validate_inputs(args: argparse.Namespace, bag_names: list[str]) -> None:
    required_files = (
        args.manifest,
        args.setup_bash,
        args.pipeline,
        args.control_config,
        args.adaptive_config,
    )
    required_directories = (
        args.staging_root,
        args.workspace,
    )
    missing = [
        str(path.expanduser().resolve())
        for path in required_files
        if not path.expanduser().is_file()
    ]
    missing.extend(
        str(path.expanduser().resolve())
        for path in required_directories
        if not path.expanduser().is_dir()
    )
    staging_root = args.staging_root.expanduser()
    missing.extend(
        str((staging_root / name).resolve())
        for name in bag_names
        if not (staging_root / name).is_dir()
    )
    if missing:
        raise FileNotFoundError(
            "missing required regression inputs: " + ", ".join(missing)
        )
    invalid_hashes = [
        value
        for value in getattr(
            args,
            "compatible_lossless_adapter_sha256",
            [],
        )
        if re.fullmatch(r"[0-9a-f]{64}", value) is None
    ]
    if invalid_hashes:
        raise ValueError(
            "invalid compatible lossless adapter SHA-256: "
            + ", ".join(invalid_hashes)
        )


def common_command(
    args: argparse.Namespace,
    bag_names: list[str],
    *,
    ros_domain_base: int | None = None,
    skip_existing: bool | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(args.pipeline.expanduser().resolve()),
        "stage34-roughness-gate-replay",
        "--bag-root",
        str(args.staging_root.expanduser().resolve()),
        "--output",
        str(args.output_root.expanduser().resolve()),
        "--workspace",
        str(args.workspace.expanduser().resolve()),
        "--setup-bash",
        str(args.setup_bash.expanduser().resolve()),
        "--stage32-label",
        "roughness_divergence_20260629",
        "--stage31-label",
        "all_compare_overlay_20260625",
        "--max-bags",
        "0",
        "--ros-domain-base",
        str(args.ros_domain_base if ros_domain_base is None else ros_domain_base),
        "--bag-play-rate",
        f"{args.bag_play_rate:g}",
        "--launch-start-delay-s",
        "15",
        "--wait-for-slam-queue-drain",
        "--queue-drain-timeout-s",
        f"{args.queue_drain_timeout_s:g}",
        "--queue-drain-quiet-s",
        f"{args.queue_drain_quiet_s:g}",
        "--slam-input-startup-drop-grace-s",
        "2",
        "--slam-input-max-startup-tf-drop-count",
        str(args.startup_tf_drop_limit),
        "--slam-input-min-recorded-scan-delivery-ratio",
        "0.999",
        "--max-recorded-scan-delivery-count-delta",
        "1",
        "--pose-graph-max-motion-gap-m",
        "2.0",
        "--no-projection-enable-gravity-alignment",
        "--projection-roughness-max-scan-imu-dt-s",
        "0.2",
        "--use-roughness-innovation-slam-toolbox",
        "--roughness-innovation-scan-qos-reliable",
        "--roughness-innovation-scan-qos-depth",
        str(args.scan_qos_depth),
        "--ros-localhost-only",
    ]
    if args.skip_existing if skip_existing is None else skip_existing:
        command.append("--skip-existing")
    command.extend(["--bag-list", *bag_names])
    return command


def phase_command(
    args: argparse.Namespace,
    bag_names: list[str],
    phase: str,
    *,
    ros_domain_base: int | None = None,
    skip_existing: bool | None = None,
) -> list[str]:
    command = common_command(
        args,
        bag_names,
        ros_domain_base=ros_domain_base,
        skip_existing=skip_existing,
    )
    bag_list_index = command.index("--bag-list")
    phase_args: list[str]
    if phase == "control":
        phase_args = [
            "--snapshot-label",
            args.control_label,
            "--slam-config",
            str(args.control_config.expanduser().resolve()),
        ]
    elif phase == "adaptive":
        reference = (
            args.output_root.expanduser().resolve()
            / "stage34_roughness_gate_replay"
            / "variant"
            / args.control_label
        )
        phase_args = [
            "--snapshot-label",
            args.adaptive_label,
            "--slam-config",
            str(args.adaptive_config.expanduser().resolve()),
            "--reference-variant-root",
            str(reference),
        ]
        for field in ALLOWED_ADAPTIVE_DIFFERENCES:
            phase_args.extend(
                ["--allow-replay-fairness-mismatch-field", field]
            )
    else:
        raise ValueError(f"unsupported phase: {phase}")
    command[bag_list_index:bag_list_index] = phase_args
    return command


def phase_commands(
    args: argparse.Namespace,
    bag_names: list[str],
    durations: dict[str, float],
    phase: str,
) -> tuple[list[list[str]], list[str] | None]:
    shards = balanced_bag_shards(bag_names, durations, args.jobs)
    bases = shard_domain_bases(
        shards,
        args.ros_domain_base,
        args.ros_domain_stride,
    )
    commands = [
        phase_command(
            args,
            shard,
            phase,
            ros_domain_base=base,
        )
        for shard, base in zip(shards, bases)
    ]
    consolidation = None
    if len(commands) > 1:
        consolidation = phase_command(
            args,
            bag_names,
            phase,
            ros_domain_base=args.ros_domain_base,
            skip_existing=True,
        )
    return commands, consolidation


def run_commands(commands: list[list[str]]) -> None:
    if len(commands) == 1:
        subprocess.run(commands[0], check=True)
        return
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = [
            executor.submit(subprocess.run, command, check=True)
            for command in commands
        ]
        for future in futures:
            future.result()


def validate_variant(
    output_root: Path,
    label: str,
    bag_names: list[str],
    expected_assets: dict[str, str] | None = None,
    expected_mode: str | None = None,
    compatible_adapter_sha256s: set[str] | None = None,
) -> dict[str, Any]:
    variant_root = (
        output_root.expanduser().resolve()
        / "stage34_roughness_gate_replay"
        / "variant"
        / label
        / "bags"
    )
    missing: list[str] = []
    invalid: dict[str, str] = {}
    observed_adapter_sha256s: Counter[str] = Counter()
    accepted_adapter_sha256s = set(compatible_adapter_sha256s or ())
    if expected_assets is not None:
        accepted_adapter_sha256s.add(
            expected_assets.get("lossless_adapter_sha256", "")
        )
    for name in bag_names:
        result_path = variant_root / name / "replay_result.json"
        if not result_path.is_file():
            missing.append(name)
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance_path = result_path.with_name("replay_provenance.json")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            provenance = {}
        fairness = provenance.get("fairness_contract", {})
        if not isinstance(fairness, dict):
            fairness = {}
        backend = fairness.get("backend_asset_sha256", {})
        if not isinstance(backend, dict):
            backend = {}
        runtime_build = fairness.get("slam_2d_runtime_build", {})
        if not isinstance(runtime_build, dict):
            runtime_build = {}
        adapter = provenance.get("lossless_stage34_adapter", {})
        if not isinstance(adapter, dict):
            adapter = {}
        actual_assets = {
            "lossless_adapter_sha256": str(adapter.get("sha256", "")),
            "runtime_slam_node_sha256": str(
                backend.get("roughness_innovation_node", "")
            ),
            "replay_pose_graph_exporter_sha256": str(
                backend.get("pose_graph_exporter", "")
            ),
            "runtime_cmake_cache_sha256": str(
                runtime_build.get("cmake_cache_sha256", "")
            ),
            "slam_config_sha256": str(backend.get("slam_config", "")),
        }
        observed_adapter_sha256s[actual_assets["lossless_adapter_sha256"]] += 1
        try:
            adaptive_parameters = json.loads(str(
                result.get(
                    "runtime_slam_adaptive_constraint_parameter_actual_json",
                    "{}",
                )
            ))
        except json.JSONDecodeError:
            adaptive_parameters = {}
        if not isinstance(adaptive_parameters, dict):
            adaptive_parameters = {}
        status = str(result.get("status", ""))
        assets_match = expected_assets is None or (
            actual_assets["lossless_adapter_sha256"]
            in accepted_adapter_sha256s
            and all(
                actual_assets.get(key) == value
                for key, value in expected_assets.items()
                if key != "lossless_adapter_sha256"
            )
        )
        complete = (
            status in ACCEPTED_STATUSES
            and int(result.get("queue_drain_complete", 0) or 0) == 1
            and int(result.get("slam_recorded_scan_delivery_complete", 0) or 0) == 1
            and int(result.get("roughness_imu_delivery_complete", 0) or 0) == 1
            and result.get("pose_graph_coverage_status")
            in {"complete", "stationary_single_node_complete"}
            and int(result.get("runtime_slam_parameter_match", 0) or 0) == 1
            and int(
                result.get(
                    "runtime_slam_adaptive_constraint_parameter_match", 0
                ) or 0
            ) == 1
            and assets_match
            and (
                expected_mode is None
                or adaptive_parameters.get("roughness_constraint_mode")
                == expected_mode
            )
        )
        if not complete:
            invalid[name] = status
    report = {
        "label": label,
        "expected": len(bag_names),
        "complete": len(bag_names) - len(missing) - len(invalid),
        "missing": missing,
        "invalid": invalid,
        "expected_assets": expected_assets or {},
        "accepted_lossless_adapter_sha256s": sorted(
            value for value in accepted_adapter_sha256s if value
        ),
        "observed_lossless_adapter_sha256_counts": dict(
            sorted(observed_adapter_sha256s.items())
        ),
        "expected_mode": expected_mode or "",
    }
    if missing or invalid:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main() -> int:
    args = parse_args()
    names = select_bag_names(
        usable_bag_names(args.manifest),
        args.bag,
        args.limit,
    )
    durations = usable_bag_durations(args.manifest)
    validate_inputs(args, names)
    phases = ("control", "adaptive") if args.phase == "all" else (args.phase,)
    phase_execution = [
        phase_commands(args, names, durations, phase) for phase in phases
    ]
    command_dir = args.output_root.expanduser().resolve() / "field_regression"
    command_dir.mkdir(parents=True, exist_ok=True)
    documented_commands: list[str] = []
    for phase, (commands, consolidation) in zip(phases, phase_execution):
        documented_commands.append(f"# {phase} shards")
        documented_commands.extend(shlex.join(command) for command in commands)
        if consolidation is not None:
            documented_commands.append(f"# {phase} combined summary pass")
            documented_commands.append(shlex.join(consolidation))
    (command_dir / "adaptive_field_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + "\n\n".join(documented_commands)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "bags": len(names),
            "jobs": min(args.jobs, len(names)),
            "phases": phases,
        },
        sort_keys=True,
    ))
    if args.dry_run:
        for commands, consolidation in phase_execution:
            for command in commands:
                print(shlex.join(command))
            if consolidation is not None:
                print(shlex.join(consolidation))
        return 0
    for phase, (commands, consolidation) in zip(phases, phase_execution):
        control_assets = current_replay_assets(args, args.control_config)
        compatible_adapters = set(
            getattr(
                args,
                "compatible_lossless_adapter_sha256",
                [],
            )
        )
        if phase == "adaptive":
            validate_variant(
                args.output_root,
                args.control_label,
                names,
                control_assets,
                "intervention_off",
                compatible_adapters,
            )
        run_commands(commands)
        if consolidation is not None:
            subprocess.run(consolidation, check=True)
        phase_config = (
            args.control_config if phase == "control" else args.adaptive_config
        )
        report = validate_variant(
            args.output_root,
            args.control_label if phase == "control" else args.adaptive_label,
            names,
            current_replay_assets(args, phase_config),
            (
                "intervention_off"
                if phase == "control"
                else "adaptive_quality_aware"
            ),
            compatible_adapters,
        )
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
