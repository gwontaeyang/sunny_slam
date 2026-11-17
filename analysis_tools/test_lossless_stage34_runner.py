import csv
import importlib.util
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name("lossless_stage34_runner.py")
SPEC = importlib.util.spec_from_file_location("lossless_stage34_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def valid_semantic_replay(**overrides):
    replay = {
        "status": "map_and_map_trajectory",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_dump_return_code": 0,
        "runtime_slam_parameter_count": 120,
        "runtime_slam_pose_graph_anchor_parameter_configured": 0,
        "runtime_guard_parameter_match": 1,
        "runtime_guard_parameter_status": "not_required",
        "runtime_guard_parameter_mismatch_count": 0,
        "slam_input_complete": 1,
        "slam_recorded_scan_delivery_required": 1,
        "slam_recorded_scan_delivery_complete": 1,
        "slam_recorded_scan_delivery_status": "complete",
        "slam_recorded_scan_delivery_count_delta": 0,
        "slam_recorded_scan_delivery_ratio": 1.0,
        "roughness_imu_delivery_complete": 1,
        "roughness_imu_delivery_fingerprint_match": 1,
        "roughness_imu_delivery_expected_count": 20,
        "roughness_imu_delivery_received_count": 20,
        "roughness_imu_delivery_expected_fingerprint_fnv1a64": (
            "0123456789abcdef"
        ),
        "roughness_imu_received_fingerprint_fnv1a64": (
            "0123456789abcdef"
        ),
        "queue_drain_enabled": 1,
        "queue_drain_complete": 1,
        "queue_drain_evidence": "direct_node_diagnostics",
        "subscription_drain_complete": 1,
        "subscription_finalize_pending_success": 1,
        "slam_queue_worker_completion_verified": 1,
        "slam_queue_direct_post_playback_worker_queued": 10,
        "slam_queue_direct_post_playback_worker_in_flight": 0,
        "slam_queue_direct_post_playback_worker_completed": 10,
        "subscription_playback_failure": "",
        "subscription_drain_effective_expected_scan_count": 10,
        "subscription_drain_received_scan_count": 10,
        "subscription_drain_expected_imu_count": 20,
        "subscription_drain_received_imu_count": 20,
        "pose_graph_captured": 1,
        "pose_graph_coverage_complete": 1,
        "enable_laserscan_roughness_gate": 0,
        "roughness_innovation_enabled": 0,
    }
    replay.update(overrides)
    return replay


def valid_semantic_context(**overrides):
    replay = valid_semantic_replay(**overrides)
    return replay, {"slam_input_complete": 1}, {
        "pose_graph_coverage_complete": 1
    }


def test_enrich_stage34_rows_with_exact_roughness_place_metrics(tmp_path):
    variant_dir = tmp_path / "variant" / "bags" / "bag"
    variant_dir.mkdir(parents=True)
    (variant_dir / "replay_result.json").write_text(
        json.dumps(
            {
                "status": "map_and_map_trajectory",
                "roughness_place_direction_hypothesis_count": 10,
                "roughness_place_direction_override_count": 1,
                "roughness_place_direction_selection_reason_counts_json": {
                    "lower_quality_risk": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "bag": "bag",
            "variant_dir": str(variant_dir),
            "variant_roughness_place_direction_override_count": "stale",
        }
    ]

    enriched = MODULE.enrich_stage34_rows_with_roughness_place_metrics(rows)

    assert enriched is rows
    assert rows[0]["variant_roughness_place_metrics_status"] == "loaded"
    assert rows[0]["variant_roughness_place_metrics_count"] == 3
    assert rows[0]["variant_roughness_place_direction_hypothesis_count"] == 10
    assert rows[0]["variant_roughness_place_direction_override_count"] == 1
    assert json.loads(
        rows[0][
            "variant_roughness_place_direction_selection_reason_counts_json"
        ]
    ) == {"lower_quality_risk": 1}
    assert "variant_status" not in rows[0]


@pytest.mark.parametrize(
    ("row", "expected_status"),
    [
        ({}, "missing_variant_dir"),
        ({"variant_dir": "missing"}, "missing_replay_result"),
    ],
)
def test_enrich_stage34_rows_reports_missing_evidence(
    tmp_path,
    monkeypatch,
    row,
    expected_status,
):
    monkeypatch.chdir(tmp_path)

    MODULE.enrich_stage34_rows_with_roughness_place_metrics([row])

    assert row["variant_roughness_place_metrics_status"] == expected_status
    assert row["variant_roughness_place_metrics_count"] == 0


def test_stage34_wrapper_enriches_without_scan_conditioning(tmp_path):
    variant_dir = tmp_path / "variant" / "bags" / "bag"
    variant_dir.mkdir(parents=True)
    (variant_dir / "replay_result.json").write_text(
        json.dumps({"roughness_place_reference_chain_event_count": 4}),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    def write_csv(path, rows):
        rows = list(rows)
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def original_stage34(args):
        csv_path = (
            Path(args.output)
            / "stage34_roughness_gate_replay"
            / "roughness_gate_replay_shadow.csv"
        )
        csv_path.parent.mkdir(parents=True)
        write_csv(csv_path, [{"bag": "bag", "variant_dir": str(variant_dir)}])

    pipeline = SimpleNamespace(
        stage30_run_one=lambda *args: {},
        stage30_wait_for_slam_queue_drain=lambda *args: {},
        stage30_queue_log_fields_for_args=lambda *args: {},
        stage30_capture_final_pose_graph=lambda *args: {},
        stage30_roughness_innovation_log_fields=lambda *args: {},
        stage30_capture_slam_parameters=lambda *args: {},
        stage30_replay_provenance_payload=lambda *args: {},
        stage30_pose_graph_coverage_fields=lambda *args: {},
        stage30_existing_replay_rerun_reason=lambda *args: "",
        stage30_bag_storage_fingerprint=lambda *args: [],
        stage34_roughness_gate_replay=original_stage34,
        stage34_comparison_status=lambda **kwargs: "valid",
        load_slam_toolbox_params=lambda path: {},
        sanitize_name=lambda value: value,
        read_csv_rows=lambda path: list(csv.DictReader(path.open())),
        write_csv=write_csv,
    )
    MODULE.install_recorded_scan_drain_guard(pipeline)

    pipeline.stage34_roughness_gate_replay(
        SimpleNamespace(
            output=str(output),
            snapshot_label="shadow",
            slam_config="",
        )
    )

    csv_path = (
        output
        / "stage34_roughness_gate_replay"
        / "roughness_gate_replay_shadow.csv"
    )
    row = next(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    assert row["variant_roughness_place_metrics_status"] == "loaded"
    assert row["variant_roughness_place_reference_chain_event_count"] == "4"
    assert "slam_owned_scan_conditioning_node_parity_relaxed" not in row


def test_pose_graph_timeout_recovers_fresh_stable_serialization(
    tmp_path,
    monkeypatch,
):
    bag_out = tmp_path / "bag"
    bag_out.mkdir()
    trajectory_csv = bag_out / "trajectory_pose_graph_final.csv"
    export_log = bag_out / "pose_graph_export.log"

    def original_capture(args, output, env):
        (output / "final_pose_graph.posegraph").write_bytes(b"posegraph")
        (output / "final_pose_graph.data").write_bytes(b"data")
        export_log.write_text("initial timeout\n", encoding="utf-8")
        return {
            "pose_graph_stem": str(output / "final_pose_graph"),
            "pose_graph_trajectory_csv": str(trajectory_csv),
            "pose_graph_export_log": str(export_log),
            "pose_graph_export_command": "offline-export",
            "pose_graph_serialize_return_code": 124,
            "pose_graph_serialize_error": "timeout",
            "pose_graph_export_return_code": -1,
            "pose_graph_export_error": "serialize_command_failed",
            "pose_graph_captured": 0,
        }

    def run(command, **kwargs):
        assert command == ["bash", "-lc", "offline-export"]
        trajectory_csv.write_text("id,x\n1,0\n2,1\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="exported\n", stderr="")

    pipeline = SimpleNamespace(
        stage30_run_one=lambda *args: {},
        stage30_wait_for_slam_queue_drain=lambda *args: {},
        stage30_queue_log_fields_for_args=lambda *args: {},
        stage30_capture_final_pose_graph=original_capture,
        stage30_roughness_innovation_log_fields=lambda *args: {},
        stage30_capture_slam_parameters=lambda *args: {},
        stage30_replay_provenance_payload=lambda *args: {},
        stage30_pose_graph_coverage_fields=lambda *args: {},
        stage30_existing_replay_rerun_reason=lambda *args: "",
        stage30_bag_storage_fingerprint=lambda *args: [],
        stage34_roughness_gate_replay=lambda *args: None,
        stage34_comparison_status=lambda **kwargs: "valid",
        read_csv_rows=lambda path: list(csv.DictReader(path.open())),
    )
    monkeypatch.setattr(MODULE, "POSE_GRAPH_SERIALIZATION_STABLE_WINDOW_S", 0.0)
    monkeypatch.setattr(MODULE, "POSE_GRAPH_SERIALIZATION_COMPLETION_GRACE_S", 0.3)
    monkeypatch.setattr(MODULE.subprocess, "run", run)

    MODULE.install_recorded_scan_drain_guard(pipeline)
    fields = pipeline.stage30_capture_final_pose_graph(
        SimpleNamespace(workspace=str(tmp_path)), bag_out, {}
    )

    assert fields["pose_graph_serialize_return_code"] == 124
    assert fields["pose_graph_serialize_timeout_recovery_attempted"] == 1
    assert fields["pose_graph_serialize_timeout_recovered"] == 1
    assert fields["pose_graph_serialize_timeout_recovery_status"] == (
        "recovered_from_fresh_stable_files"
    )
    assert fields["pose_graph_captured"] == 1
    assert fields["pose_graph_trajectory_count"] == 2
    assert "pose_graph_export_error" not in fields
    assert "running offline exporter" in export_log.read_text(encoding="utf-8")


def test_pose_graph_timeout_does_not_recover_stale_serialization(
    tmp_path,
    monkeypatch,
):
    bag_out = tmp_path / "bag"
    bag_out.mkdir()
    posegraph = bag_out / "final_pose_graph.posegraph"
    data = bag_out / "final_pose_graph.data"
    posegraph.write_bytes(b"stale")
    data.write_bytes(b"stale")
    os.utime(posegraph, (1, 1))
    os.utime(data, (1, 1))

    def original_capture(args, output, env):
        return {
            "pose_graph_stem": str(output / "final_pose_graph"),
            "pose_graph_trajectory_csv": str(
                output / "trajectory_pose_graph_final.csv"
            ),
            "pose_graph_export_log": str(output / "pose_graph_export.log"),
            "pose_graph_export_command": "must-not-run",
            "pose_graph_serialize_return_code": 124,
            "pose_graph_captured": 0,
        }

    pipeline = SimpleNamespace(
        stage30_run_one=lambda *args: {},
        stage30_wait_for_slam_queue_drain=lambda *args: {},
        stage30_queue_log_fields_for_args=lambda *args: {},
        stage30_capture_final_pose_graph=original_capture,
        stage30_roughness_innovation_log_fields=lambda *args: {},
        stage30_capture_slam_parameters=lambda *args: {},
        stage30_replay_provenance_payload=lambda *args: {},
        stage30_pose_graph_coverage_fields=lambda *args: {},
        stage30_existing_replay_rerun_reason=lambda *args: "",
        stage30_bag_storage_fingerprint=lambda *args: [],
        stage34_roughness_gate_replay=lambda *args: None,
        stage34_comparison_status=lambda **kwargs: "valid",
    )
    monkeypatch.setattr(MODULE, "POSE_GRAPH_SERIALIZATION_COMPLETION_GRACE_S", 0.0)
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("stale files must not be exported"),
    )

    MODULE.install_recorded_scan_drain_guard(pipeline)
    fields = pipeline.stage30_capture_final_pose_graph(
        SimpleNamespace(workspace=str(tmp_path)), bag_out, {}
    )

    assert fields["pose_graph_serialize_timeout_recovery_attempted"] == 1
    assert fields["pose_graph_serialize_timeout_recovered"] == 0
    assert fields["pose_graph_serialize_timeout_recovery_status"] == (
        "fresh_serialized_files_not_stable"
    )
    assert fields["pose_graph_captured"] == 0


def test_slam_2d_only_scope_accepts_direct_laserscan():
    MODULE.validate_slam_2d_only_scope(
        {
            "scan_topic": "/scan_laserscan",
            "start_pointcloud_projection": 0,
        },
        SimpleNamespace(enable_laserscan_roughness_gate=False),
    )


@pytest.mark.parametrize(
    ("row", "args", "message"),
    [
        (
            {
                "scan_topic": "/scan_laserscan",
                "start_pointcloud_projection": 0,
            },
            SimpleNamespace(enable_laserscan_roughness_gate=True),
            "external LaserScan guard",
        ),
        (
            {
                "scan_topic": "/scan_laserscan",
                "start_pointcloud_projection": 1,
            },
            SimpleNamespace(enable_laserscan_roughness_gate=False),
            "PointCloud projection",
        ),
        (
            {"scan_topic": "/scan", "start_pointcloud_projection": 0},
            SimpleNamespace(enable_laserscan_roughness_gate=False),
            "direct /scan_laserscan",
        ),
    ],
)
def test_slam_2d_only_scope_rejects_external_or_non_laserscan_paths(
    row,
    args,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        MODULE.validate_slam_2d_only_scope(row, args)


def test_saved_expected_imu_audit_requires_exact_replay_evidence(tmp_path):
    replay_path = tmp_path / "replay_result.json"
    replay_path.write_text(
        json.dumps(
            {
                "roughness_imu_delivery_expected_count": 42,
                "roughness_imu_delivery_expected_fingerprint_fnv1a64": (
                    "0123456789abcdef"
                ),
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.saved_expected_imu_audit(tmp_path) == {
        "count": 42,
        "fingerprint_fnv1a64": "0123456789abcdef",
        "source": "saved_replay",
    }

    replay_path.write_text(
        json.dumps(
            {
                "roughness_imu_delivery_expected_count": "42",
                "roughness_imu_delivery_expected_fingerprint_fnv1a64": (
                    "0123456789abcdef"
                ),
            }
        ),
        encoding="utf-8",
    )
    assert MODULE.saved_expected_imu_audit(tmp_path) is None


def test_saved_bag_fingerprint_reuses_exact_stats_and_metadata_hash(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    metadata = bag / "metadata.yaml"
    database = bag / "bag_0.db3"
    metadata.write_text("metadata", encoding="utf-8")
    database.write_bytes(b"database")
    fingerprint = []
    for path in (database, metadata):
        stat = path.stat()
        fingerprint.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": MODULE._file_sha256(path),
            }
        )
    fingerprint.sort(key=lambda item: item["name"])
    bag_out = tmp_path / "output"
    bag_out.mkdir()
    (bag_out / "replay_provenance.json").write_text(
        json.dumps(
            {
                "bag_path": str(bag.resolve()),
                "bag_storage_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.saved_bag_storage_fingerprint(
        {"path": str(bag)}, bag_out
    ) == fingerprint

    database.write_bytes(b"changed-longer")
    assert MODULE.saved_bag_storage_fingerprint(
        {"path": str(bag)}, bag_out
    ) is None


def test_saved_bag_fingerprint_rejects_metadata_content_with_same_stats(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    metadata = bag / "metadata.yaml"
    database = bag / "bag_0.db3"
    metadata.write_bytes(b"original")
    database.write_bytes(b"database")
    fingerprint = []
    for path in (database, metadata):
        stat = path.stat()
        fingerprint.append(
            {
                "name": path.name,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": MODULE._file_sha256(path),
            }
        )
    fingerprint.sort(key=lambda item: item["name"])
    bag_out = tmp_path / "output"
    bag_out.mkdir()
    (bag_out / "replay_provenance.json").write_text(
        json.dumps(
            {
                "bag_path": str(bag.resolve()),
                "bag_storage_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    stat = metadata.stat()
    metadata.write_bytes(b"modified")
    MODULE.os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert MODULE.saved_bag_storage_fingerprint(
        {"path": str(bag)}, bag_out
    ) is None


def test_latest_received_scan_count_uses_latest_summary():
    text = (
        "ROUGHNESS_INNOVATION_SUMMARY received=10 queued=2\n"
        "unrelated\n"
        "ROUGHNESS_INNOVATION_SUMMARY received=25 queued=0\n"
    )

    assert MODULE.latest_received_scan_count(text) == 25


def test_latest_received_scan_count_handles_missing_summary():
    assert MODULE.latest_received_scan_count("no diagnostics") is None


def test_worker_drain_pattern_requires_completed_in_flight_item():
    incomplete = (
        "ROUGHNESS_INNOVATION_SUMMARY received=10 queued=4 pending=0 "
        "decision_cache=0 scan_queue_depth=0 scan_worker_in_flight=1 "
        "scan_worker_completed=3 duplicate_imu=0"
    )
    complete = incomplete.replace(
        "scan_worker_in_flight=1 scan_worker_completed=3",
        "scan_worker_in_flight=0 scan_worker_completed=4",
    )

    incomplete_match = MODULE._WORKER_DRAIN_PATTERN.search(incomplete)
    complete_match = MODULE._WORKER_DRAIN_PATTERN.search(complete)

    assert incomplete_match is not None
    assert incomplete_match.groups() == ("4", "0", "1", "3")
    assert complete_match is not None
    assert complete_match.groups() == ("4", "0", "0", "4")


def test_legacy_log_normalization_hides_worker_fields_only_from_parser():
    line = (
        "ROUGHNESS_INNOVATION_SUMMARY received=10 queued=4 pending=0 "
        "decision_cache=0 scan_queue_depth=0 scan_worker_in_flight=1 "
        "scan_worker_completed=3 duplicate_imu=0"
    )

    normalized = MODULE.normalize_pose_graph_near_chain_diagnostics(line)

    assert "scan_queue_depth=0 duplicate_imu=0" in normalized
    assert "scan_worker_in_flight" not in normalized
    assert "scan_worker_completed" not in normalized
    assert MODULE._WORKER_DRAIN_PATTERN.search(line) is not None


def test_playback_finished_requires_clean_rosbag_process_marker():
    assert MODULE.playback_finished(
        "[INFO] [ros2-4]: process has finished cleanly [pid 123]"
    )
    assert not MODULE.playback_finished("Starting LaserScan-only bag playback")


def test_playback_failure_requires_rosbag_child_process():
    failed = (
        "[ERROR] [ros2-4]: process has died [pid 123, exit code 1, "
        "cmd 'ros2 bag play /tmp/bag --clock']"
    )

    assert MODULE.playback_failure(failed) == failed
    assert not MODULE.playback_failure(
        "[ERROR] [tf_relay-2]: process has died [pid 456, exit code 1]"
    )


def test_playback_failure_detects_roughness_slam_node_crash():
    failed = (
        "[ERROR] [roughness_aware_slam_toolbox_node-1]: process has died "
        "[pid 456, exit code -11, cmd '/tmp/roughness_aware_slam_toolbox_node']"
    )

    assert MODULE.playback_failure(failed) == failed


def test_monitor_playback_failure_interrupts_only_current_group(
    tmp_path, monkeypatch
):
    launch_log = tmp_path / "launch.log"
    launch_log.write_text(
        "[ERROR] [ros2-4]: process has died [pid 123, exit code 1, "
        "cmd 'ros2 bag play /tmp/bag --clock']\n",
        encoding="utf-8",
    )
    killed = []

    class FakeProcess:
        pid = 987

        @staticmethod
        def poll():
            return None

    monkeypatch.setattr(
        MODULE.os,
        "killpg",
        lambda pid, sig: killed.append((pid, sig)),
    )
    state = {}

    MODULE.monitor_playback_failure(
        launch_log,
        FakeProcess(),
        threading.Event(),
        state,
        poll_s=0.01,
    )

    assert killed == [(987, signal.SIGINT)]
    assert "ros2 bag play" in state["detail"]


def test_expected_scan_count_validates_metadata_value():
    assert MODULE.expected_scan_count({"scan_laserscan_count": "11194"}) == 11194
    assert MODULE.expected_scan_count({"scan_laserscan_count": "0"}) is None
    assert MODULE.expected_scan_count({"scan_laserscan_count": "bad"}) is None


def test_adapter_provenance_contains_self_hash_and_anchor_config(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {
            MODULE.POSE_GRAPH_ANCHOR_PARAMETER: True,
        }
    )
    args = SimpleNamespace(slam_config=str(config))
    legacy_payload = {
        "schema": 5,
        "bag_path": "/tmp/bag",
        "fairness_contract": {"roughness_innovation_profile": {}},
    }

    payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: legacy_payload,
        {},
        args,
        "launch",
        "capture",
    )

    adapter = payload["lossless_stage34_adapter"]
    assert adapter["schema"] == MODULE.ADAPTER_PROVENANCE_SCHEMA
    assert adapter["path"] == str(MODULE.ADAPTER_PATH)
    assert adapter["sha256"] == MODULE._file_sha256(MODULE.ADAPTER_PATH)
    assert adapter["pose_graph_anchor_parameter"] == {
        "name": MODULE.POSE_GRAPH_ANCHOR_PARAMETER,
        "configured": 1,
        "default": False,
        "effective": True,
        "config_path": str(config.resolve()),
        "config_status": "loaded",
        "expected": True,
    }
    assert payload["fairness_contract"]["roughness_innovation_profile"][
        MODULE.POSE_GRAPH_ANCHOR_FAIRNESS_FIELD
    ] is True
    adapter_signature = MODULE._payload_sha256(payload)
    assert MODULE._LEGACY_SIGNATURE_BY_ADAPTER_SIGNATURE[adapter_signature] == (
        MODULE._payload_sha256(legacy_payload)
    )
    compatible_legacy = MODULE._LEGACY_PAYLOAD_BY_ADAPTER_SIGNATURE[
        adapter_signature
    ]
    assert compatible_legacy["fairness_contract"][
        "roughness_innovation_profile"
    ][MODULE.POSE_GRAPH_ANCHOR_FAIRNESS_FIELD] is True
    assert "lossless_stage34_adapter" not in compatible_legacy


def test_runtime_build_provenance_records_release_binary(tmp_path):
    workspace = tmp_path / "workspace"
    executable = (
        workspace
        / "install/slam_2d_core/lib/slam_2d_core"
        / "roughness_aware_slam_toolbox_node"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"release-binary")
    cache = workspace / "build/slam_2d_core/CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_CXX_FLAGS:STRING=\n"
        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
        "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
        "CMAKE_CXX_COMPILER_VERSION:STRING=11.4.0\n",
        encoding="utf-8",
    )

    result = MODULE.runtime_build_provenance(
        SimpleNamespace(workspace=str(workspace))
    )

    assert result["status"] == "complete"
    assert result["cmake_build_type"] == "Release"
    assert result["cmake_cxx_flags_release"] == "-O3 -DNDEBUG"
    assert result["executable_sha256"] == MODULE._file_sha256(executable)
    assert result["cmake_cache_sha256"] == MODULE._file_sha256(cache)


def test_runtime_build_provenance_resolves_nested_source_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    nested_source = workspace / "src/project/package"
    nested_source.mkdir(parents=True)
    executable = (
        workspace
        / "install/slam_2d_core/lib/slam_2d_core"
        / "roughness_aware_slam_toolbox_node"
    )
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"binary")
    cache = workspace / "build/slam_2d_core/CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n", encoding="utf-8")

    result = MODULE.runtime_build_provenance(
        SimpleNamespace(workspace=str(nested_source))
    )

    assert result["status"] == "complete"
    assert result["requested_workspace"] == str(nested_source.resolve())
    assert result["workspace"] == str(workspace.resolve())


def test_adapter_fairness_includes_runtime_build_fingerprint(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    pipeline = SimpleNamespace(load_slam_toolbox_params=lambda path: {})

    payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: {
            "fairness_contract": {"roughness_innovation_profile": {}},
        },
        {},
        SimpleNamespace(slam_config=str(config), workspace=""),
        "launch",
        "capture",
    )

    expected = {"status": "workspace_not_provided"}
    assert payload["lossless_stage34_adapter"]["runtime_build"] == expected
    assert payload["fairness_contract"][
        MODULE.RUNTIME_BUILD_FAIRNESS_FIELD
    ] == expected


def test_adapter_fairness_uses_node_default_when_anchor_is_not_configured():
    pipeline = SimpleNamespace(load_slam_toolbox_params=lambda path: {})
    args = SimpleNamespace(slam_config="")

    payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: {
            "fairness_contract": {"roughness_innovation_profile": {}},
        },
        {},
        args,
        "launch",
        "capture",
    )

    adapter = payload["lossless_stage34_adapter"]
    assert adapter["pose_graph_anchor_parameter"] == {
        "name": MODULE.POSE_GRAPH_ANCHOR_PARAMETER,
        "configured": 0,
        "default": False,
        "effective": False,
        "config_path": "",
        "config_status": "not_provided",
    }
    assert payload["fairness_contract"]["roughness_innovation_profile"][
        MODULE.POSE_GRAPH_ANCHOR_FAIRNESS_FIELD
    ] is False


def test_pose_graph_log_compatibility_preserves_activity_and_new_counters(
    tmp_path,
):
    log_path = tmp_path / "launch.log"
    log_path.write_text(
        "ROUGHNESS_NEAR_CHAIN_DIAG stamp_ns=10 mapper_generation=0 "
        "state_id=1 reason=spatial_passage active=true rough_trigger=false "
        "window_complete=true rough_samples=0 score=1.0 "
        "spatial_pose_source=pose_graph_predicted "
        "pose_graph_anchor_mode=true recovery_until_stamp_ns=20\n"
        "ROUGHNESS_NEAR_CHAIN_DIAG stamp_ns=11 mapper_generation=0 "
        "state_id=2 reason=normal active=false rough_trigger=false "
        "window_complete=true rough_samples=0 score=0.2 "
        "spatial_pose_source=pose_graph_empty "
        "pose_graph_anchor_mode=true recovery_until_stamp_ns=0\n"
        "ROUGHNESS_INNOVATION_SUMMARY received=2 "
        "spatial_pose_graph_refreshes=7 "
        "spatial_pose_graph_stale_removes=1 "
        "spatial_pose_graph_invalid_removes=2 "
        "spatial_pose_graph_prediction_failures=3 "
        "spatial_pose_graph_anchor_inserts=4\n",
        encoding="utf-8",
    )
    parser_paths = []

    def legacy_parser(path, *unused):
        parser_paths.append(path)
        normalized = path.read_text(encoding="utf-8")
        assert "spatial_pose_source=" not in normalized
        assert "pose_graph_anchor_mode=" not in normalized
        return {
            "roughness_near_chain_diag_count": normalized.count(
                "ROUGHNESS_NEAR_CHAIN_DIAG"
            ),
            "roughness_near_chain_active_count": normalized.count(
                "active=true"
            ),
        }

    fields = MODULE.compatible_roughness_log_fields(
        legacy_parser,
        log_path,
    )

    assert fields["roughness_near_chain_diag_count"] == 2
    assert fields["roughness_near_chain_active_count"] == 1
    assert fields[
        "roughness_spatial_memory_pose_graph_anchor_mode_true_count"
    ] == 2
    assert json.loads(
        fields["roughness_spatial_memory_pose_source_counts_json"]
    ) == {"pose_graph_empty": 1, "pose_graph_predicted": 1}
    assert fields[
        "roughness_spatial_memory_pose_graph_refreshes_summary_last"
    ] == 7
    assert fields[
        "roughness_spatial_memory_pose_graph_prediction_failures_"
        "summary_last"
    ] == 3
    assert fields[
        "roughness_spatial_memory_pose_graph_anchor_inserts_summary_last"
    ] == 4
    assert parser_paths[0] != log_path
    assert not parser_paths[0].exists()


def test_compatible_log_fields_preserve_each_bag_diagnostic_csv(tmp_path):
    log_text = (
        "ROUGHNESS_ADAPTIVE_MEMORY stamp_ns=10 mode=intervention_off "
        "available=true strength=0.1\n"
        "ROUGHNESS_NEAR_CHAIN_CANDIDATE stamp_ns=10 candidate_index=0 "
        "backend_action=accepted outcome_parity=true\n"
    )
    log_a = tmp_path / "bag_a_launch.log"
    log_b = tmp_path / "bag_b_launch.log"
    log_a.write_text(log_text, encoding="utf-8")
    log_b.write_text(log_text.replace("stamp_ns=10", "stamp_ns=20"), encoding="utf-8")

    fields_a = MODULE.compatible_roughness_log_fields(
        lambda *unused: {},
        log_a,
    )
    fields_b = MODULE.compatible_roughness_log_fields(
        lambda *unused: {},
        log_b,
    )

    candidate_a = Path(
        fields_a["roughness_exact_near_chain_candidate_events_csv"]
    )
    candidate_b = Path(
        fields_b["roughness_exact_near_chain_candidate_events_csv"]
    )
    memory_a = Path(fields_a["roughness_adaptive_memory_events_csv"])
    memory_b = Path(fields_b["roughness_adaptive_memory_events_csv"])
    assert candidate_a.name == "bag_a_near_chain_candidate_events.csv"
    assert candidate_b.name == "bag_b_near_chain_candidate_events.csv"
    assert memory_a.name == "bag_a_adaptive_memory_events.csv"
    assert memory_b.name == "bag_b_adaptive_memory_events.csv"
    assert candidate_a != candidate_b
    assert memory_a != memory_b
    assert candidate_a.is_file() and candidate_b.is_file()
    assert memory_a.is_file() and memory_b.is_file()


def test_scan_conditioning_log_fields_capture_actions_and_summary_counters():
    text = (
        "ROUGHNESS_SCAN_CONDITION stamp_ns=100 mode=residual_filter "
        "action=residual_filtered score=3.2 rough_active=true "
        "hard_active=true imu_available=true nearest_imu_dt_s=0.01 "
        "input_valid_beams=900 output_valid_beams=880 overlap_beams=850 "
        "removed_beams=20 recovery_active=false recovery_fault=false\n"
        "ROUGHNESS_SCAN_CONDITION stamp_ns=200 mode=residual_filter "
        "action=residual_tf_failed_observe score=2.8 rough_active=true "
        "hard_active=false imu_available=false nearest_imu_dt_s=0.30 "
        "input_valid_beams=900 output_valid_beams=900 overlap_beams=0 "
        "removed_beams=0 recovery_active=true recovery_fault=true\n"
        "ROUGHNESS_INNOVATION_SUMMARY received=2 "
        "scan_conditioning_allow=2 scan_conditioning_drop=0 "
        "scan_conditioning_bridge=0 scan_conditioning_quarantine=0 "
        "scan_conditioning_stabilized=0 "
        "scan_conditioning_residual_filtered=1 "
        "scan_conditioning_fallback=1 "
        "scan_conditioning_imu_quality_reject=3 "
        "scan_conditioning_tf_failure=1\n"
    )

    fields = MODULE.scan_conditioning_log_fields(text)

    assert fields["roughness_scan_conditioning_event_count"] == 2
    assert json.loads(
        fields["roughness_scan_conditioning_action_counts_json"]
    ) == {
        "residual_filtered": 1,
        "residual_tf_failed_observe": 1,
    }
    assert fields["roughness_scan_conditioning_rough_active_count"] == 2
    assert fields["roughness_scan_conditioning_hard_active_count"] == 1
    assert fields["roughness_scan_conditioning_imu_unavailable_count"] == 1
    assert fields["roughness_scan_conditioning_recovery_fault_count"] == 1
    assert fields["roughness_scan_conditioning_removed_beams_total"] == 20
    assert fields["roughness_scan_conditioning_removed_beams_max"] == 20
    assert fields["roughness_scan_conditioning_first_stamp_ns"] == 100
    assert fields["roughness_scan_conditioning_last_stamp_ns"] == 200
    assert fields["roughness_scan_conditioning_residual_filtered_summary_last"] == 1
    assert fields["roughness_scan_conditioning_tf_failure_summary_last"] == 1


def test_roughness_place_log_fields_capture_observe_only_candidates(tmp_path):
    text = (
        "ROUGHNESS_PLACE_EVENT event_id=1 pre_stamp_ns=100 post_stamp_ns=200 "
        "pre_state_id=10 post_state_id=20 rough_observations=8 "
        "path_length_m=1.2 descriptor_bins=16 descriptor_channels=6 "
        "descriptor_valid=true reference_events=0 candidate_available=false "
        "candidate_event_id=0 candidate_direction=none candidate_score=nan "
        "candidate_shape_score=nan candidate_amplitude_score=nan "
        "candidate_length_score=nan candidate_second_score=nan "
        "candidate_margin=nan corrected_center_distance_m=1.0 "
        "corrected_post_heading_delta_deg=2.0 action=observe graph_action=none\n"
        "ROUGHNESS_PLACE_EVENT event_id=2 pre_stamp_ns=300 post_stamp_ns=400 "
        "pre_state_id=30 post_state_id=40 rough_observations=12 "
        "path_length_m=1.8 descriptor_bins=16 descriptor_channels=6 "
        "descriptor_valid=true reference_events=1 candidate_available=true "
        "candidate_event_id=1 candidate_direction=reverse candidate_score=0.82 "
        "candidate_shape_score=0.88 candidate_amplitude_score=0.80 "
        "candidate_length_score=0.78 candidate_second_score=0.57 "
        "candidate_margin=0.25 corrected_center_distance_m=0.4 "
        "corrected_post_heading_delta_deg=175.0 action=observe graph_action=none\n"
    )
    csv_path = tmp_path / "roughness_place_events.csv"

    fields = MODULE.roughness_place_log_fields(text, csv_path)

    assert fields["roughness_place_event_count"] == 2
    assert fields["roughness_place_valid_descriptor_count"] == 2
    assert fields["roughness_place_candidate_count"] == 1
    assert fields["roughness_place_graph_action_count"] == 0
    assert json.loads(fields["roughness_place_direction_counts_json"]) == {
        "none": 1,
        "reverse": 1,
    }
    assert fields["roughness_place_candidate_score_mean"] == pytest.approx(0.82)
    assert fields["roughness_place_candidate_score_max"] == pytest.approx(0.82)
    assert fields["roughness_place_candidate_margin_mean"] == pytest.approx(0.25)
    assert fields[
        "roughness_place_corrected_center_distance_mean_m"
    ] == pytest.approx(0.7)
    assert fields[
        "roughness_place_descriptor_path_length_mean_m"
    ] == pytest.approx(1.5)
    assert fields["roughness_place_events_csv"] == str(csv_path)

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["event_id"] for row in rows] == ["1", "2"]
    assert rows[1]["candidate_direction"] == "reverse"
    assert all(row["graph_action"] == "none" for row in rows)


def test_roughness_place_scan_match_log_fields_capture_fail_open_and_pair(tmp_path):
    text = (
        "ROUGHNESS_PLACE_ANCHOR_SCAN_MATCH event_id=2 candidate_event_id=1 "
        "candidate_direction=reverse anchor_role=reference_pre "
        "reference_state_id=72 query_state_id=226 state_id_separation=154 "
        "descriptor_score=0.82 descriptor_margin=0.25 available=true "
        "fail_open=false failure=none forward_available=true "
        "forward_failure=none forward_coarse_response=0.74 "
        "forward_fine_response=0.81 forward_coarse_cov_xx=0.01 "
        "forward_coarse_cov_yy=0.02 forward_coarse_cov_yawyaw=0.03 "
        "forward_fine_cov_xx=0.004 forward_fine_cov_yy=0.005 "
        "forward_fine_cov_yawyaw=0.006 "
        "forward_proposal_disagreement_translation_m=0.04 "
        "forward_proposal_disagreement_yaw_rad=0.02 reverse_available=true "
        "reverse_failure=none reverse_coarse_response=0.72 "
        "reverse_fine_response=0.79 reverse_coarse_cov_xx=0.01 "
        "reverse_coarse_cov_yy=0.02 reverse_coarse_cov_yawyaw=0.03 "
        "reverse_fine_cov_xx=0.004 reverse_fine_cov_yy=0.005 "
        "reverse_fine_cov_yawyaw=0.006 "
        "reverse_proposal_disagreement_translation_m=0.05 "
        "reverse_proposal_disagreement_yaw_rad=0.03 odom_relative_x_m=0.1 "
        "odom_relative_y_m=0.0 odom_relative_yaw_rad=3.1 "
        "matched_relative_x_m=0.11 matched_relative_y_m=0.01 "
        "matched_relative_yaw_rad=3.09 "
        "measurement_vs_odometry_translation_m=0.02 "
        "measurement_vs_odometry_yaw_rad=0.01 "
        "bidirectional_translation_error_m=0.03 "
        "bidirectional_yaw_error_rad=0.02 "
        "directional_alignment_translation_error_m=0.04 "
        "directional_alignment_yaw_error_rad=0.01 branch_alignment_x_m=1.0 "
        "branch_alignment_y_m=2.0 branch_alignment_yaw_rad=0.2 "
        "action=observe graph_action=none\n"
        "ROUGHNESS_PLACE_ANCHOR_SCAN_MATCH event_id=2 candidate_event_id=1 "
        "candidate_direction=reverse anchor_role=reference_post "
        "reference_state_id=74 query_state_id=224 state_id_separation=150 "
        "descriptor_score=0.82 descriptor_margin=0.25 available=false "
        "fail_open=true failure=forward=none_reverse=non_finite "
        "forward_available=true forward_failure=none "
        "forward_coarse_response=0.70 forward_fine_response=0.76 "
        "reverse_available=false reverse_failure=non_finite "
        "reverse_coarse_response=0 reverse_fine_response=nan "
        "action=observe graph_action=none\n"
        "ROUGHNESS_PLACE_PAIR_CONSISTENCY event_id=2 candidate_event_id=1 "
        "candidate_direction=reverse attempted=true available=false "
        "fail_open=true reference_pre_state_id=72 "
        "query_for_reference_pre_state_id=226 reference_post_state_id=74 "
        "query_for_reference_post_state_id=224 "
        "forward_alignment_translation_error_m=0.06 "
        "forward_alignment_yaw_error_rad=0.02 "
        "reverse_alignment_translation_error_m=nan "
        "reverse_alignment_yaw_error_rad=nan action=observe graph_action=none\n"
    )
    anchor_csv = tmp_path / "anchor.csv"
    pair_csv = tmp_path / "pair.csv"

    fields = MODULE.roughness_place_scan_match_log_fields(
        text,
        anchor_csv,
        pair_csv,
    )

    assert fields["roughness_place_scan_match_anchor_count"] == 2
    assert fields["roughness_place_scan_match_anchor_available_count"] == 1
    assert fields["roughness_place_scan_match_anchor_fail_open_count"] == 1
    assert fields["roughness_place_scan_match_pair_count"] == 1
    assert fields["roughness_place_scan_match_pair_available_count"] == 0
    assert fields["roughness_place_scan_match_pair_fail_open_count"] == 1
    assert fields["roughness_place_scan_match_graph_action_count"] == 0
    assert fields[
        "roughness_place_scan_match_forward_fine_response_mean"
    ] == pytest.approx(0.785)
    assert fields[
        "roughness_place_pair_forward_alignment_translation_error_m_max"
    ] == pytest.approx(0.06)
    with anchor_csv.open(encoding="utf-8", newline="") as stream:
        anchor_rows = list(csv.DictReader(stream))
    with pair_csv.open(encoding="utf-8", newline="") as stream:
        pair_rows = list(csv.DictReader(stream))
    assert [row["anchor_role"] for row in anchor_rows] == [
        "reference_pre",
        "reference_post",
    ]
    assert pair_rows[0]["fail_open"] == "true"


def test_roughness_place_reference_chain_log_fields_capture_shadow_evidence(
    tmp_path,
):
    def line(marker, values):
        return marker + " " + " ".join(f"{key}={value}" for key, value in values.items())

    context = {
        "mapper_generation": 2,
        "event_id": 4,
        "available": "true",
        "fail_open": "false",
        "failure": "none",
        "pre_anchor_state_id": 137,
        "post_anchor_state_id": 139,
        "pre_chain_count": 5,
        "pre_chain_first_state_id": 133,
        "pre_chain_last_state_id": 137,
        "pre_chain_anchor_state_gap": 2,
        "pre_chain_anchor_distance_m": 0.12,
        "pre_chain_anchor_time_gap_s": 0.08,
        "post_chain_count": 5,
        "post_chain_first_state_id": 139,
        "post_chain_last_state_id": 143,
        "post_reacquisition_reset_count": 2,
        "post_reacquisition_state_gap": 3,
        "post_reacquisition_distance_m": 0.18,
        "post_reacquisition_time_gap_s": 0.24,
        "candidate_count": 1,
        "method": "stable_side_reacquired_odom_clone_v3",
        "action": "observe",
        "graph_action": "none",
    }
    anchor_base = {
        "mapper_generation": 2,
        "event_id": 4,
        "candidate_event_id": 2,
        "candidate_rank": 1,
        "hypothesis_index": 0,
        "descriptor_direction": "reverse",
        "evaluated_direction": "forward",
        "anchor_role": "pre",
        "available": "true",
        "fail_open": "false",
        "failure": "none",
        "reference_base_scan_count": 5,
        "query_base_scan_count": 5,
        "reference_first_state_id": 133,
        "reference_last_state_id": 137,
        "query_first_state_id": 224,
        "query_last_state_id": 228,
        "reference_maximum_anchor_distance_m": 0.42,
        "query_maximum_anchor_distance_m": 0.38,
        "forward_coarse_response": 0.71,
        "forward_fine_response": 0.79,
        "reverse_coarse_response": 0.70,
        "reverse_fine_response": 0.77,
        "forward_fine_cov_xx": 0.01,
        "forward_fine_cov_yy": 0.02,
        "forward_fine_cov_yawyaw": 0.003,
        "reverse_fine_cov_xx": 0.01,
        "reverse_fine_cov_yy": 0.02,
        "reverse_fine_cov_yawyaw": 0.003,
        "bidirectional_translation_error_m": 0.03,
        "bidirectional_yaw_error_rad": 0.01,
        "method": "stable_side_reacquired_odom_clone_v3",
        "action": "observe",
        "graph_action": "none",
    }
    hypothesis_base = {
        "mapper_generation": 2,
        "event_id": 4,
        "candidate_event_id": 2,
        "candidate_rank": 1,
        "hypothesis_index": 0,
        "descriptor_direction": "reverse",
        "evaluated_direction": "forward",
        "reference_context_available": "true",
        "available": "true",
        "fail_open": "false",
        "failure": "none",
        "quality_accepted": "true",
        "quality_reason": "accepted",
        "quality_available": "true",
        "quality_risk": 0.72,
        "minimum_fine_response": 0.77,
        "forward_pair_translation_error_m": 0.04,
        "forward_pair_yaw_error_rad": 0.01,
        "reverse_pair_translation_error_m": 0.05,
        "reverse_pair_yaw_error_rad": 0.02,
        "selected": "true",
        "selection_fail_open": "false",
        "selection_reason": "lower_gate_utilization",
        "method": "stable_side_reacquired_odom_clone_v3",
        "action": "observe",
        "graph_action": "none",
    }
    second_anchor = {**anchor_base, "anchor_role": "post", "forward_fine_response": 0.81}
    second_hypothesis = {
        **hypothesis_base,
        "hypothesis_index": 1,
        "evaluated_direction": "reverse",
        "quality_risk": 0.91,
        "minimum_fine_response": 0.75,
        "selected": "false",
    }
    decision = {
        "mapper_generation": 2,
        "event_id": 4,
        "available": "true",
        "fail_open": "false",
        "quality_accepted": "true",
        "reason": "quality_accepted_correlated_duplicate",
        "candidate_event_id": 2,
        "candidate_rank": 1,
        "evaluated_direction": "forward",
        "quality_risk": 0.72,
        "minimum_fine_response": 0.77,
        "correlated_passage_duplicate": "true",
        "correlated_with_reference_event_id": 2,
        "correlated_with_query_event_id": 3,
        "reference_intervening_state_count": 0,
        "query_intervening_state_count": 1,
        "evaluation_duration_ms": 12.5,
        "method": "stable_side_reacquired_odom_clone_v3",
        "action": "observe",
        "graph_action": "none",
    }
    text = "\n".join(
        [
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT", context),
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR", anchor_base),
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_ANCHOR", second_anchor),
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS", hypothesis_base),
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_HYPOTHESIS", second_hypothesis),
            line("ROUGHNESS_PLACE_REFERENCE_CHAIN_DECISION", decision),
        ]
    )
    event_path = tmp_path / "events.csv"
    anchor_path = tmp_path / "anchors.csv"
    hypothesis_path = tmp_path / "hypotheses.csv"
    decision_path = tmp_path / "decisions.csv"

    fields = MODULE.roughness_place_reference_chain_log_fields(
        text,
        event_path,
        anchor_path,
        hypothesis_path,
        decision_path,
    )

    assert fields["roughness_place_reference_chain_event_count"] == 1
    assert fields["roughness_place_reference_chain_event_available_count"] == 1
    assert fields["roughness_place_reference_chain_anchor_count"] == 2
    assert fields["roughness_place_reference_chain_anchor_available_count"] == 2
    assert fields["roughness_place_reference_chain_hypothesis_count"] == 2
    assert fields["roughness_place_reference_chain_hypothesis_accepted_count"] == 2
    assert fields["roughness_place_reference_chain_hypothesis_selected_count"] == 1
    assert fields["roughness_place_reference_chain_decision_count"] == 1
    assert fields["roughness_place_reference_chain_decision_accepted_count"] == 1
    assert (
        fields[
            "roughness_place_reference_chain_decision_correlated_duplicate_count"
        ]
        == 1
    )
    assert fields[
        "roughness_place_reference_chain_evaluation_duration_ms_p95"
    ] == pytest.approx(12.5)
    assert fields[
        "roughness_place_reference_chain_event_pre_chain_anchor_state_gap_max"
    ] == 2
    assert fields[
        "roughness_place_reference_chain_event_post_reacquisition_reset_count_max"
    ] == 2
    assert fields[
        "roughness_place_reference_chain_event_post_reacquisition_distance_m_p95"
    ] == pytest.approx(0.18)
    assert fields[
        "roughness_place_reference_chain_event_pre_chain_anchor_time_gap_s_max"
    ] == pytest.approx(0.08)
    assert fields[
        "roughness_place_reference_chain_event_post_reacquisition_time_gap_s_max"
    ] == pytest.approx(0.24)
    assert fields["roughness_place_reference_chain_quality_risk_median"] == pytest.approx(0.815)
    assert fields[
        "roughness_place_reference_chain_anchor_forward_fine_response_max"
    ] == pytest.approx(0.81)
    assert fields["roughness_place_reference_chain_event_schema_error_count"] == 0
    assert fields["roughness_place_reference_chain_anchor_schema_error_count"] == 0
    assert fields["roughness_place_reference_chain_hypothesis_schema_error_count"] == 0
    with hypothesis_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with decision_path.open(encoding="utf-8", newline="") as stream:
        decision_rows = list(csv.DictReader(stream))
    assert [row["selected"] for row in rows] == ["true", "false"]
    assert all(
        row["method"] == "stable_side_reacquired_odom_clone_v3"
        for row in rows
    )
    assert decision_rows[0]["correlated_passage_duplicate"] == "true"

    malformed = {**context, "pre_chain_count": "not_an_integer"}
    malformed_fields = MODULE.roughness_place_reference_chain_log_fields(
        line("ROUGHNESS_PLACE_REFERENCE_CHAIN_EVENT", malformed),
        tmp_path / "malformed_events.csv",
        tmp_path / "malformed_anchors.csv",
        tmp_path / "malformed_hypotheses.csv",
    )
    assert malformed_fields[
        "roughness_place_reference_chain_event_schema_error_count"
    ] == 1


def test_roughness_place_constraint_log_fields_link_decision_and_soft_edge(tmp_path):
    decision = (
        "ROUGHNESS_PLACE_CANDIDATE_DECISION stamp_ns=400 mapper_generation=2 "
        "event_id=4 candidate_event_id=2 candidate_rank=1 event_separation_s=45 "
        "minimum_state_id_separation=87 descriptor_score=0.88 "
        "descriptor_margin=0.04 quality_accepted=true fail_open=false "
        "reason=accepted selected=true selected_anchor_role=pre "
        "selected_minimum_fine_response=0.44 "
        "maximum_pair_alignment_translation_error_m=0.33 "
        "maximum_pair_alignment_yaw_error_rad=0.018 "
        "reference_pre_state_id=137 reference_pre_unique_id=137 "
        "query_pre_state_id=224 query_pre_unique_id=224 "
        "reference_post_state_id=139 reference_post_unique_id=139 "
        "query_post_state_id=226 query_post_unique_id=226 "
        "action=soft graph_action=none experimental_field=kept\n"
    )
    constraint = (
        "ROUGHNESS_PLACE_CONSTRAINT stamp_ns=400 mapper_generation=2 event_id=4 "
        "candidate_event_id=2 candidate_rank=1 mode=soft quality_accepted=true "
        "fail_open=false reason=soft_edge_inserted selected_anchor_role=pre "
        "source_state_id=137 source_unique_id=137 target_state_id=224 "
        "target_unique_id=224 edge_type=roughness_place_soft edge_index=400 "
        "edge_count_before=400 edge_count_after=401 measurement_x_m=0.1 "
        "measurement_y_m=0.02 measurement_yaw_rad=3.1 covariance_xx=0.12 "
        "covariance_yy=0.12 covariance_yawyaw=0.003 "
        "final_covariance_xx=0.12 final_covariance_xy=0 "
        "final_covariance_yy=0.12 final_covariance_yawyaw=0.003 "
        "pre_residual_x_m=5 pre_residual_y_m=2 pre_residual_translation_m=5.38 "
        "pre_residual_yaw_rad=1.8 pre_covariance_weighted_residual_sq=120 "
        "post_residual_x_m=0.02 post_residual_y_m=0.01 "
        "post_residual_translation_m=0.022 post_residual_yaw_rad=0.01 "
        "post_covariance_weighted_residual_sq=0.05 injected=true duplicate=false "
        "correlated_passage_duplicate=false "
        "correlated_with_reference_event_id=0 "
        "correlated_with_query_event_id=0 "
        "reference_intervening_state_count=-1 "
        "query_intervening_state_count=-1 "
        "action=soft graph_action=soft_edge_inserted\n"
    )
    constraint_edge = (
        "ROUGHNESS_PLACE_CONSTRAINT_EDGE stamp_ns=400 mapper_generation=2 "
        "event_id=4 candidate_event_id=2 candidate_rank=1 mode=soft "
        "anchor_strategy=validated_pair group_size=2 edge_group_index=0 "
        "anchor_role=pre source_state_id=137 source_unique_id=137 "
        "target_state_id=224 target_unique_id=224 edge_index=400 "
        "measurement_x_m=0.1 measurement_y_m=0.02 measurement_yaw_rad=3.1 "
        "covariance_xx=0.01 covariance_xy=0 covariance_xyaw=0 "
        "covariance_yy=0.01 covariance_yyaw=0 covariance_yawyaw=0.0002 "
        "final_covariance_xx=0.01 final_covariance_xy=0 "
        "final_covariance_xyaw=0 final_covariance_yy=0.01 "
        "final_covariance_yyaw=0 final_covariance_yawyaw=0.0002 "
        "pre_residual_x_m=5 pre_residual_y_m=2 "
        "pre_residual_translation_m=5.38 pre_residual_yaw_rad=1.8 "
        "pre_covariance_weighted_residual_sq=120 post_residual_x_m=0.02 "
        "post_residual_y_m=0.01 post_residual_translation_m=0.022 "
        "post_residual_yaw_rad=0.01 post_covariance_weighted_residual_sq=0.05 "
        "injected=true group_partial=false reason=validated_pair_inserted "
        "action=soft graph_action=validated_pair_inserted\n"
    )
    decision_path = tmp_path / "decisions.csv"
    constraint_path = tmp_path / "constraints.csv"
    constraint_edge_path = tmp_path / "constraint_edges.csv"

    fields = MODULE.roughness_place_constraint_log_fields(
        decision + constraint + constraint_edge,
        decision_path,
        constraint_path,
        constraint_edge_path,
    )

    assert fields["roughness_place_candidate_decision_count"] == 1
    assert fields["roughness_place_candidate_accepted_count"] == 1
    assert fields["roughness_place_candidate_selected_count"] == 1
    assert fields["roughness_place_constraint_count"] == 1
    assert fields["roughness_place_constraint_edge_count"] == 1
    assert fields["roughness_place_constraint_edge_injected_count"] == 1
    assert fields["roughness_place_constraint_injected_count"] == 1
    assert fields["roughness_place_constraint_correlated_duplicate_count"] == 0
    assert fields["roughness_place_decision_constraint_unmatched_count"] == 0
    assert fields["roughness_place_decision_duplicate_identity_count"] == 0
    assert fields["roughness_place_decision_schema_error_count"] == 0
    assert fields["roughness_place_constraint_schema_error_count"] == 0
    assert fields["roughness_place_constraint_edge_schema_error_count"] == 0
    assert fields[
        "roughness_place_constraint_post_residual_translation_m_median"
    ] == pytest.approx(0.022)
    with decision_path.open(encoding="utf-8", newline="") as stream:
        decision_rows = list(csv.DictReader(stream))
    assert decision_rows[0]["experimental_field"] == "kept"
    with constraint_edge_path.open(encoding="utf-8", newline="") as stream:
        edge_rows = list(csv.DictReader(stream))
    assert edge_rows[0]["anchor_strategy"] == "validated_pair"


def test_roughness_place_constraint_log_fields_empty_and_malformed(tmp_path):
    decision_path = tmp_path / "empty_decisions.csv"
    constraint_path = tmp_path / "empty_constraints.csv"
    fields = MODULE.roughness_place_constraint_log_fields(
        "",
        decision_path,
        constraint_path,
    )

    assert fields["roughness_place_candidate_decision_count"] == 0
    assert fields["roughness_place_constraint_count"] == 0
    assert decision_path.read_text(encoding="utf-8").startswith("stamp_ns,")
    assert constraint_path.read_text(encoding="utf-8").startswith("stamp_ns,")

    malformed = (
        "ROUGHNESS_PLACE_CANDIDATE_DECISION stamp_ns=1 mapper_generation=0 "
        "event_id=1 candidate_event_id=2 candidate_rank=1 "
        "quality_accepted=maybe fail_open=false selected=true\n"
    )
    malformed_fields = MODULE.roughness_place_constraint_log_fields(malformed)
    assert malformed_fields["roughness_place_decision_schema_error_count"] == 1


def test_roughness_place_constraint_log_fields_capture_direction_hypotheses(
    tmp_path,
):
    text = "\n".join(
        [
            "ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS stamp_ns=400 "
            "mapper_generation=2 event_id=4 candidate_event_id=2 "
            "candidate_rank=1 hypothesis_index=0 descriptor_direction=reverse "
            "evaluated_direction=reverse preferred=true quality_accepted=true "
            "fail_open=false reason=accepted quality_available=true "
            "quality_risk=0.94 minimum_fine_response=0.40 "
            "maximum_pair_alignment_translation_error_m=0.45 "
            "maximum_pair_alignment_yaw_error_rad=0.02 selected=false "
            "direction_overridden=false selection_fail_open=false "
            "selection_reason=lower_gate_utilization action=observe "
            "graph_action=none",
            "ROUGHNESS_PLACE_DIRECTION_HYPOTHESIS stamp_ns=400 "
            "mapper_generation=2 event_id=4 candidate_event_id=2 "
            "candidate_rank=1 hypothesis_index=1 descriptor_direction=reverse "
            "evaluated_direction=forward preferred=false quality_accepted=true "
            "fail_open=false reason=accepted quality_available=true "
            "quality_risk=0.68 minimum_fine_response=0.44 "
            "maximum_pair_alignment_translation_error_m=0.12 "
            "maximum_pair_alignment_yaw_error_rad=0.01 selected=true "
            "direction_overridden=true selection_fail_open=false "
            "selection_reason=lower_gate_utilization action=observe "
            "graph_action=none",
        ]
    )
    hypotheses_path = tmp_path / "direction_hypotheses.csv"

    fields = MODULE.roughness_place_constraint_log_fields(
        text,
        direction_hypothesis_output_path=hypotheses_path,
    )

    assert fields["roughness_place_direction_hypothesis_count"] == 2
    assert fields["roughness_place_direction_hypothesis_accepted_count"] == 2
    assert fields["roughness_place_direction_selected_count"] == 1
    assert fields["roughness_place_direction_override_count"] == 1
    assert fields["roughness_place_direction_selection_fail_open_count"] == 0
    assert fields["roughness_place_direction_hypothesis_schema_error_count"] == 0
    assert fields["roughness_place_direction_quality_risk_median"] == pytest.approx(
        0.81
    )
    rows = list(csv.DictReader(hypotheses_path.open(encoding="utf-8")))
    assert [row["evaluated_direction"] for row in rows] == ["reverse", "forward"]
    assert rows[1]["direction_overridden"] == "true"


def test_adaptive_near_chain_log_fields_write_exact_candidate_csv(tmp_path):
    text = "\n".join(
        [
            "ROUGHNESS_ADAPTIVE_MEMORY stamp_ns=10 mode=adaptive_memory_observe "
            "available=true strength=0.8 event_intensity=1.0 "
            "effective_duration_s=3.5 effective_distance_m=0.55 "
            "outcome_parity=true",
            "ROUGHNESS_NEAR_CHAIN_CANDIDATE stamp_ns=10 candidate_index=0 "
            "chain_size_accepted=true normalized_response=0.12 "
            "translation_disagreement_m=0.08 yaw_disagreement_rad=0.04 "
            "xy_covariance_max_eigenvalue_m2=0.02 quality_risk=0.7 "
            "combined_risk=0.56 shadow_action=downweighted fail_open=false "
            "backend_action=accepted edge_inserted=true downweighted=false "
            "scan_match_covariance_weighted_disagreement_sq=2.0 "
            "post_covariance_weighted_residual_sq=1.0 outcome_parity=true",
        ]
    )
    candidate_path = tmp_path / "candidates.csv"
    memory_path = tmp_path / "memory.csv"
    fields = MODULE.adaptive_near_chain_log_fields(
        text,
        candidate_path,
        memory_path,
    )
    assert fields["roughness_exact_near_chain_candidate_count"] == 1
    assert fields["roughness_exact_near_chain_edge_inserted_count"] == 1
    assert fields["roughness_adaptive_memory_strength_mean"] == pytest.approx(
        0.8
    )
    assert fields["roughness_adaptive_normalized_response_p95"] == pytest.approx(
        0.12
    )
    assert candidate_path.is_file()
    assert memory_path.is_file()
    rows = list(csv.DictReader(candidate_path.open(encoding="utf-8")))
    assert rows[0]["backend_action"] == "accepted"


def test_adaptive_near_chain_log_fields_write_schema_for_zero_events(tmp_path):
    candidate_path = tmp_path / "empty_candidates.csv"
    memory_path = tmp_path / "empty_memory.csv"

    fields = MODULE.adaptive_near_chain_log_fields(
        "",
        candidate_path,
        memory_path,
    )

    assert fields["roughness_exact_near_chain_candidate_count"] == 0
    assert fields["roughness_adaptive_memory_event_count"] == 0
    with candidate_path.open(encoding="utf-8", newline="") as stream:
        assert set(csv.DictReader(stream).fieldnames or ()) >= {
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
    with memory_path.open(encoding="utf-8", newline="") as stream:
        assert set(csv.DictReader(stream).fieldnames or ()) >= {
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


def test_adapter_provenance_records_effective_scan_conditioning_config(
    tmp_path,
):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    configured = {
        "enable_roughness_scan_conditioning": True,
        "roughness_scan_conditioning_mode": "quarantine",
        "roughness_scan_conditioning_quarantine_recovery_scan_count": 7,
    }
    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: configured,
    )

    payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: {
            "fairness_contract": {"roughness_innovation_profile": {}},
        },
        {},
        SimpleNamespace(slam_config=str(config)),
        "launch",
        "capture",
    )

    recorded = payload["lossless_stage34_adapter"][
        "scan_conditioning_parameters"
    ]
    assert recorded["configured_count"] == 3
    assert recorded["expected"] == configured
    assert recorded["effective"][
        "roughness_scan_conditioning_residual_direction"
    ] == "closer"
    fairness = payload["fairness_contract"]["roughness_innovation_profile"][
        MODULE.SCAN_CONDITIONING_FAIRNESS_FIELD
    ]
    assert fairness["enable_roughness_scan_conditioning"] is True
    assert fairness["roughness_scan_conditioning_mode"] == "quarantine"
    assert fairness[
        "roughness_scan_conditioning_quarantine_recovery_scan_count"
    ] == 7


def test_adapter_provenance_records_adaptive_constraint_profile(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    configured = {
        "roughness_constraint_mode": "adaptive_memory_observe",
        "roughness_enable_near_chain_candidate_observer": True,
        "roughness_adaptive_memory_max_duration_s": 3.5,
        "roughness_adaptive_memory_max_distance_m": 0.55,
        "unrelated_parameter": 123,
    }
    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: configured,
    )

    payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: {
            "fairness_contract": {"roughness_innovation_profile": {}},
        },
        {},
        SimpleNamespace(slam_config=str(config)),
        "launch",
        "capture",
    )

    expected = dict(configured)
    expected.pop("unrelated_parameter")
    recorded = payload["lossless_stage34_adapter"][
        "adaptive_constraint_parameters"
    ]
    assert recorded["configured_count"] == len(expected)
    assert recorded["expected"] == expected
    fairness = payload["fairness_contract"]["roughness_innovation_profile"][
        MODULE.ADAPTIVE_CONSTRAINT_FAIRNESS_FIELD
    ]
    assert fairness == expected


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ({}, False),
        (
            {
                "enable_roughness_scan_conditioning": False,
                "roughness_scan_conditioning_mode": "quarantine",
            },
            False,
        ),
        (
            {
                "enable_roughness_scan_conditioning": True,
                "roughness_scan_conditioning_mode": "observe",
            },
            False,
        ),
        (
            {
                "enable_roughness_scan_conditioning": True,
                "roughness_scan_conditioning_mode": "quarantine",
            },
            True,
        ),
        (
            {
                "enable_roughness_scan_conditioning": True,
                "roughness_scan_conditioning_mode": "residual_filter",
                "roughness_scan_conditioning_residual_fallback_mode": (
                    "observe"
                ),
            },
            False,
        ),
        (
            {
                "enable_roughness_scan_conditioning": True,
                "roughness_scan_conditioning_mode": "residual_filter",
                "roughness_scan_conditioning_residual_fallback_mode": "drop",
            },
            True,
        ),
    ],
)
def test_scan_conditioning_node_parity_relaxes_only_for_drop_capable_modes(
    profile,
    expected,
):
    assert MODULE.scan_conditioning_can_change_pose_graph_node_count(
        profile
    ) is expected


def test_intentional_conditioning_keeps_retention_floor_when_parity_relaxes():
    rows = [
        {
            "bag": "field_bag",
            "pose_graph_node_parity_required": 1,
            "pose_graph_node_parity_complete": 0,
            "variant_pose_graph_node_retention_ratio": 0.93,
        }
    ]

    result = MODULE.relax_intentional_conditioning_node_parity_rows(rows)

    assert result[0]["pose_graph_node_parity_required"] == 0
    assert result[0]["pose_graph_node_parity_complete"] == 1
    assert result[0][
        "slam_owned_scan_conditioning_node_parity_relaxed"
    ] == 1
    assert result[0][
        "slam_owned_scan_conditioning_node_retention_floor"
    ] == 0.85
    assert result[0]["variant_pose_graph_node_retention_ratio"] == 0.93


def test_runtime_capture_validates_pose_graph_anchor_from_full_dump(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    dump.write_text("full parameter dump\n", encoding="utf-8")
    parameter_name = MODULE.POSE_GRAPH_ANCHOR_PARAMETER

    def parameter_mismatches(expected, actual):
        return {
            name: {"expected": value, "actual": actual.get(name, "<missing>")}
            for name, value in expected.items()
            if actual.get(name, "<missing>") != value
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {parameter_name: True},
        stage30_extract_node_parameters=lambda text, node: {
            parameter_name: True,
            "resolution": 0.02,
        },
        stage30_parameter_mismatches=parameter_mismatches,
    )
    original_fields = {
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_count": 2,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_mismatch_fields": "",
        "runtime_slam_parameter_mismatch_details_json": "{}",
        "runtime_slam_parameter_dump": str(dump),
        "runtime_slam_parameter_dump_return_code": 0,
    }

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        lambda *unused: dict(original_fields),
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(),
    )

    assert fields["runtime_slam_full_parameter_dump_captured"] == 1
    assert fields["runtime_slam_full_parameter_dump_count"] == 2
    assert fields["runtime_slam_full_parameter_dump_sha256"] == (
        MODULE._file_sha256(dump)
    )
    assert fields["runtime_slam_pose_graph_anchor_parameter_configured"] == 1
    assert fields["runtime_slam_pose_graph_anchor_parameter_match"] == 1
    assert fields[f"runtime_slam_actual_{parameter_name}"] is True
    assert fields["runtime_slam_parameter_match"] == 1


def test_runtime_capture_retries_transient_parameter_dump_failure(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    parameter_name = MODULE.POSE_GRAPH_ANCHOR_PARAMETER
    calls = []

    def capture(*unused, timeout_s=15.0):
        calls.append(timeout_s)
        if len(calls) == 1:
            dump.write_text("", encoding="utf-8")
            return {
                "runtime_slam_parameter_status": "unavailable",
                "runtime_slam_parameter_match": 0,
                "runtime_slam_parameter_mismatch_count": 1,
                "runtime_slam_parameter_mismatch_fields": "<parameter_dump>",
                "runtime_slam_parameter_mismatch_details_json": "{}",
                "runtime_slam_parameter_dump": str(dump),
                "runtime_slam_parameter_dump_return_code": 124,
            }
        dump.write_text("full parameter dump\n", encoding="utf-8")
        return {
            "runtime_slam_parameter_status": "matched",
            "runtime_slam_parameter_match": 1,
            "runtime_slam_parameter_count": 1,
            "runtime_slam_parameter_mismatch_count": 0,
            "runtime_slam_parameter_mismatch_fields": "",
            "runtime_slam_parameter_mismatch_details_json": "{}",
            "runtime_slam_parameter_dump": str(dump),
            "runtime_slam_parameter_dump_return_code": 0,
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {parameter_name: False},
        stage30_extract_node_parameters=lambda text, node: {
            parameter_name: False,
        } if text else {},
        stage30_parameter_mismatches=lambda expected, actual: {
            name: {
                "expected": value,
                "actual": actual.get(name, "<missing>"),
            }
            for name, value in expected.items()
            if actual.get(name, "<missing>") != value
        },
    )

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        capture,
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(poll=lambda: None),
    )

    assert calls == [15.0, MODULE.SLAM_PARAMETER_CAPTURE_RETRY_TIMEOUT_S]
    assert fields["runtime_slam_parameter_capture_attempt_count"] == 2
    assert fields["runtime_slam_parameter_capture_retry_count"] == 1
    assert fields["runtime_slam_parameter_capture_retry_status"] == "succeeded"
    assert fields["runtime_slam_parameter_capture_initial_return_code"] == 124
    assert fields["runtime_slam_parameter_dump_return_code"] == 0
    assert fields["runtime_slam_parameter_match"] == 1


def test_runtime_capture_does_not_retry_after_launch_exit(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    calls = []

    def capture(*unused, timeout_s=15.0):
        calls.append(timeout_s)
        dump.write_text("", encoding="utf-8")
        return {
            "runtime_slam_parameter_status": "launch_exited",
            "runtime_slam_parameter_match": 0,
            "runtime_slam_parameter_mismatch_count": 1,
            "runtime_slam_parameter_mismatch_fields": "<parameter_dump>",
            "runtime_slam_parameter_mismatch_details_json": "{}",
            "runtime_slam_parameter_dump": str(dump),
            "runtime_slam_parameter_dump_return_code": 124,
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {},
        stage30_extract_node_parameters=lambda text, node: {},
        stage30_parameter_mismatches=lambda expected, actual: {},
    )

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        capture,
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(poll=lambda: 1),
    )

    assert calls == [15.0]
    assert fields["runtime_slam_parameter_capture_attempt_count"] == 1
    assert fields["runtime_slam_parameter_capture_retry_count"] == 0
    assert fields["runtime_slam_parameter_capture_retry_status"] == "not_needed"
    assert fields["runtime_slam_parameter_match"] == 0


def test_runtime_capture_keeps_persistent_retry_failure_fail_closed(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    calls = []

    def capture(*unused, timeout_s=15.0):
        calls.append(timeout_s)
        dump.write_text("", encoding="utf-8")
        return {
            "runtime_slam_parameter_status": "unavailable",
            "runtime_slam_parameter_match": 0,
            "runtime_slam_parameter_mismatch_count": 1,
            "runtime_slam_parameter_mismatch_fields": "<parameter_dump>",
            "runtime_slam_parameter_mismatch_details_json": "{}",
            "runtime_slam_parameter_dump": str(dump),
            "runtime_slam_parameter_dump_return_code": 124,
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {},
        stage30_extract_node_parameters=lambda text, node: {},
        stage30_parameter_mismatches=lambda expected, actual: {},
    )

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        capture,
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(poll=lambda: None),
    )

    assert calls == [15.0, MODULE.SLAM_PARAMETER_CAPTURE_RETRY_TIMEOUT_S]
    assert fields["runtime_slam_parameter_capture_attempt_count"] == 2
    assert fields["runtime_slam_parameter_capture_retry_status"] == "failed"
    assert fields["runtime_slam_parameter_dump_return_code"] == 124
    assert fields["runtime_slam_parameter_match"] == 0


def test_runtime_capture_promotes_anchor_mismatch_to_overall_mismatch(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    dump.write_text("full parameter dump\n", encoding="utf-8")
    parameter_name = MODULE.POSE_GRAPH_ANCHOR_PARAMETER
    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: {parameter_name: True},
        stage30_extract_node_parameters=lambda text, node: {
            parameter_name: False,
        },
        stage30_parameter_mismatches=lambda expected, actual: {
            parameter_name: {"expected": True, "actual": False}
        },
    )
    original_fields = {
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_count": 1,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_mismatch_fields": "",
        "runtime_slam_parameter_mismatch_details_json": "{}",
        "runtime_slam_parameter_dump": str(dump),
        "runtime_slam_parameter_dump_return_code": 0,
    }

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        lambda *unused: dict(original_fields),
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(),
    )

    assert fields["runtime_slam_pose_graph_anchor_parameter_match"] == 0
    assert fields["runtime_slam_parameter_match"] == 0
    assert fields["runtime_slam_parameter_status"] == "mismatched"
    assert json.loads(fields["runtime_slam_parameter_mismatch_details_json"])[
        parameter_name
    ] == {"expected": True, "actual": False}


def test_runtime_capture_validates_all_configured_scan_conditioning_params(
    tmp_path,
):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    dump.write_text("full parameter dump\n", encoding="utf-8")
    expected = {
        "enable_roughness_scan_conditioning": True,
        "roughness_scan_conditioning_mode": "residual_filter",
        "roughness_scan_conditioning_residual_max_removed_ratio": 0.03,
    }
    actual = dict(expected)

    def parameter_mismatches(expected_values, actual_values):
        return {
            name: {
                "expected": value,
                "actual": actual_values.get(name, "<missing>"),
            }
            for name, value in expected_values.items()
            if actual_values.get(name, "<missing>") != value
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: expected,
        stage30_extract_node_parameters=lambda text, node: actual,
        stage30_parameter_mismatches=parameter_mismatches,
    )
    original_fields = {
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_count": 3,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_mismatch_fields": "",
        "runtime_slam_parameter_mismatch_details_json": "{}",
        "runtime_slam_parameter_dump": str(dump),
        "runtime_slam_parameter_dump_return_code": 0,
    }

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        lambda *unused: dict(original_fields),
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(),
    )

    assert fields[
        "runtime_slam_scan_conditioning_parameter_configured_count"
    ] == 3
    assert fields["runtime_slam_scan_conditioning_parameter_match"] == 1
    assert fields[
        "runtime_slam_scan_conditioning_parameter_status"
    ] == "matched"
    assert json.loads(
        fields["runtime_slam_scan_conditioning_parameter_actual_json"]
    ) == actual
    assert fields["runtime_slam_parameter_match"] == 1


def test_runtime_capture_promotes_scan_conditioning_mismatch(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    dump.write_text("full parameter dump\n", encoding="utf-8")
    name = "roughness_scan_conditioning_mode"
    expected = {name: "quarantine"}
    actual = {name: "observe"}
    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: expected,
        stage30_extract_node_parameters=lambda text, node: actual,
        stage30_parameter_mismatches=lambda expected_values, actual_values: {
            key: {
                "expected": value,
                "actual": actual_values.get(key, "<missing>"),
            }
            for key, value in expected_values.items()
            if actual_values.get(key, "<missing>") != value
        },
    )
    original_fields = {
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_count": 1,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_mismatch_fields": "",
        "runtime_slam_parameter_mismatch_details_json": "{}",
        "runtime_slam_parameter_dump": str(dump),
        "runtime_slam_parameter_dump_return_code": 0,
    }

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        lambda *unused: dict(original_fields),
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(),
    )

    assert fields["runtime_slam_scan_conditioning_parameter_match"] == 0
    assert fields["runtime_slam_parameter_match"] == 0
    details = json.loads(
        fields["runtime_slam_parameter_mismatch_details_json"]
    )
    assert details[name] == {"expected": "quarantine", "actual": "observe"}


def test_runtime_capture_promotes_adaptive_mode_mismatch(tmp_path):
    config = tmp_path / "mapper.yaml"
    config.write_text("placeholder\n", encoding="utf-8")
    dump = tmp_path / "slam_toolbox.yaml"
    dump.write_text("full parameter dump\n", encoding="utf-8")
    name = "roughness_constraint_mode"
    expected = {name: "adaptive_memory_observe"}
    actual = {name: "intervention_off"}

    def parameter_mismatches(expected_values, actual_values):
        return {
            key: {
                "expected": value,
                "actual": actual_values.get(key, "<missing>"),
            }
            for key, value in expected_values.items()
            if actual_values.get(key, "<missing>") != value
        }

    pipeline = SimpleNamespace(
        load_slam_toolbox_params=lambda path: expected,
        stage30_extract_node_parameters=lambda text, node: actual,
        stage30_parameter_mismatches=parameter_mismatches,
    )
    original_fields = {
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_count": 1,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_mismatch_fields": "",
        "runtime_slam_parameter_mismatch_details_json": "{}",
        "runtime_slam_parameter_dump": str(dump),
        "runtime_slam_parameter_dump_return_code": 0,
    }

    fields = MODULE.augment_slam_parameter_capture(
        pipeline,
        lambda *unused: dict(original_fields),
        SimpleNamespace(slam_config=str(config)),
        tmp_path,
        {},
        SimpleNamespace(),
    )

    assert fields["runtime_slam_adaptive_constraint_parameter_match"] == 0
    assert fields["runtime_slam_parameter_match"] == 0
    assert fields["runtime_slam_parameter_status"] == "mismatched"
    details = json.loads(
        fields["runtime_slam_parameter_mismatch_details_json"]
    )
    assert details[name] == {
        "expected": "adaptive_memory_observe",
        "actual": "intervention_off",
    }


def test_latest_received_imu_audit_uses_latest_summary():
    text = (
        "ROUGHNESS_INNOVATION_SUMMARY received=1 "
        "received_imu=10 received_imu_first_stamp_ns=100 "
        "received_imu_last_stamp_ns=200 received_imu_max_positive_gap_ns=20 "
        "received_imu_stamp_duplicate_count=1 "
        "received_imu_stamp_out_of_order_count=0 "
        "received_imu_stamp_fingerprint_fnv1a64=0123456789abcdef\n"
        "ROUGHNESS_INNOVATION_SUMMARY received=2 "
        "received_imu=20 received_imu_first_stamp_ns=100 "
        "received_imu_last_stamp_ns=300 received_imu_max_positive_gap_ns=30 "
        "received_imu_stamp_duplicate_count=2 "
        "received_imu_stamp_out_of_order_count=0 "
        "received_imu_stamp_fingerprint_fnv1a64=fedcba9876543210\n"
    )

    assert MODULE.latest_received_imu_audit(text) == {
        "count": 20,
        "first_stamp_ns": 100,
        "last_stamp_ns": 300,
        "max_positive_gap_ns": 30,
        "duplicate_count": 2,
        "out_of_order_count": 0,
        "fingerprint_fnv1a64": "fedcba9876543210",
    }


def test_rounded_stamp_exclusion_resolves_millisecond_log_stamp():
    recorded = [1_000_400_000, 1_040_000_000, 1_080_000_000]

    excluded, unmatched = MODULE.rounded_stamp_exclusion_indices(
        recorded, [1_000_000_000]
    )

    assert excluded == {0}
    assert unmatched == 0


def test_rounded_stamp_exclusion_rejects_distant_stamp():
    excluded, unmatched = MODULE.rounded_stamp_exclusion_indices(
        [1_005_000_000], [1_000_000_000]
    )

    assert excluded == set()
    assert unmatched == 1


def test_stationary_single_node_pose_graph_is_complete(tmp_path):
    (tmp_path / "map.png").write_bytes(b"map")
    pipeline = SimpleNamespace(
        stage31_load_samples=lambda path: [(10.0, 1.0, 2.0, 0.0)]
    )

    fields = MODULE.stationary_aware_pose_graph_coverage(
        pipeline,
        lambda bag_out, args: {
            "pose_graph_coverage_status": "unavailable",
            "pose_graph_coverage_complete": 0,
            "pose_graph_coverage_expected_motion_count": 1,
        },
        tmp_path,
        SimpleNamespace(),
    )

    assert fields["pose_graph_coverage_complete"] == 1
    assert fields["pose_graph_coverage_status"] == (
        "stationary_single_node_complete"
    )
    assert fields["pose_graph_stationary_single_node"] == 1


def test_single_node_does_not_cover_expected_motion(tmp_path):
    (tmp_path / "map.png").write_bytes(b"map")
    pipeline = SimpleNamespace(
        stage31_load_samples=lambda path: [(10.0, 1.0, 2.0, 0.0)]
    )

    fields = MODULE.stationary_aware_pose_graph_coverage(
        pipeline,
        lambda bag_out, args: {
            "pose_graph_coverage_status": "unavailable",
            "pose_graph_coverage_complete": 0,
            "pose_graph_coverage_expected_motion_count": 2,
        },
        tmp_path,
        SimpleNamespace(),
    )

    assert fields["pose_graph_coverage_complete"] == 0
    assert fields["pose_graph_coverage_status"] == "unavailable"


def test_existing_replay_allows_missing_inactive_v5_artifacts(tmp_path):
    v4_names = ("summary.json", "map.png", "commands.txt")
    runtime_dump = "runtime_parameters/slam_toolbox.yaml"
    v5_names = v4_names + (
        "guard_status_counts.csv",
        "guard_status_events.csv",
        "runtime_parameters/slam_laserscan_roughness_gate.yaml",
        runtime_dump,
        "roughness_innovation_events.csv",
        "roughness_post_confirmation_events.csv",
    )
    original_resolver = lambda schema: v5_names if schema >= 5 else v4_names
    pipeline = SimpleNamespace(
        STAGE30_ARTIFACT_NAMES_V4=v4_names,
        stage30_artifact_names_for_schema=original_resolver,
    )
    observed_names = []
    replay, log_fields, coverage = valid_semantic_context()
    replay.update(
        {
            "replay_provenance_schema": 5,
            "replay_signature_sha256": "signature",
            "artifact_manifest": {
                "summary.json": {},
                "map.png": {},
                "commands.txt": {},
                runtime_dump: {},
            },
        }
    )

    def original_reason(*args):
        observed_names.extend(pipeline.stage30_artifact_names_for_schema(5))
        return ""

    reason = MODULE.compatible_existing_replay_rerun_reason(
        pipeline,
        original_reason,
        tmp_path,
        {},
        replay,
        log_fields,
        coverage,
        "signature",
        True,
    )

    assert reason == ""
    assert observed_names == [
        "summary.json",
        "map.png",
        "commands.txt",
        runtime_dump,
    ]
    assert pipeline.stage30_artifact_names_for_schema is original_resolver


def test_existing_replay_requires_v5_artifacts_for_active_features(tmp_path):
    v4_names = ("summary.json", "map.png", "commands.txt")
    runtime_dump = "runtime_parameters/slam_toolbox.yaml"
    v5_names = v4_names + (
        "guard_status_counts.csv",
        "guard_status_events.csv",
        "runtime_parameters/slam_laserscan_roughness_gate.yaml",
        runtime_dump,
        "roughness_innovation_events.csv",
        "roughness_post_confirmation_events.csv",
    )
    original_resolver = lambda schema: v5_names if schema >= 5 else v4_names
    pipeline = SimpleNamespace(
        STAGE30_ARTIFACT_NAMES_V4=v4_names,
        stage30_artifact_names_for_schema=original_resolver,
    )
    replay, log_fields, coverage = valid_semantic_context(
        enable_laserscan_roughness_gate=1,
        roughness_innovation_enabled=1,
        roughness_innovation_enable_post_confirmation=1,
        runtime_guard_parameter_status="matched",
    )
    replay.update(
        {
            "replay_provenance_schema": 5,
            "replay_signature_sha256": "signature",
            "artifact_manifest": {
                name: {}
                for name in v4_names + (runtime_dump,)
            },
        }
    )
    calls = []

    def original_reason(*args):
        calls.append(tuple(pipeline.stage30_artifact_names_for_schema(5)))
        return "existing_replay_artifacts_missing"

    reason = MODULE.compatible_existing_replay_rerun_reason(
        pipeline,
        original_reason,
        tmp_path,
        {},
        replay,
        log_fields,
        coverage,
        "signature",
        True,
    )

    assert reason == "existing_replay_artifacts_missing"
    assert calls == [v5_names]


def test_existing_replay_does_not_relax_missing_core_artifacts(tmp_path):
    v4_names = ("summary.json", "map.png", "commands.txt")
    v5_names = v4_names + ("runtime_parameters/slam_toolbox.yaml",)
    original_resolver = lambda schema: v5_names if schema >= 5 else v4_names
    pipeline = SimpleNamespace(
        STAGE30_ARTIFACT_NAMES_V4=v4_names,
        stage30_artifact_names_for_schema=original_resolver,
    )
    observed_names = []

    def original_reason(*args):
        observed_names.extend(pipeline.stage30_artifact_names_for_schema(5))
        return "existing_replay_artifacts_missing"

    reason = MODULE.compatible_existing_replay_rerun_reason(
        pipeline,
        original_reason,
        tmp_path,
        {},
        {
            "replay_provenance_schema": 5,
            "artifact_manifest": {
                "summary.json": {},
                "map.png": {},
            },
        },
        {},
        {},
        "signature",
        True,
    )

    assert reason == "existing_replay_artifacts_missing"
    assert observed_names == list(v5_names)
    assert pipeline.stage30_artifact_names_for_schema is original_resolver


@pytest.mark.parametrize(
    ("overrides", "log_fields", "coverage", "expected_reason"),
    [
        (
            {"status": "map_only"},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_status_not_accepted",
        ),
        (
            {"runtime_slam_parameter_match": 2},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_runtime_parameters_not_exact",
        ),
        (
            {"slam_recorded_scan_delivery_count_delta": 1},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_scan_delivery_not_exact",
        ),
        (
            {"roughness_imu_delivery_fingerprint_match": 0},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_imu_delivery_not_exact",
        ),
        (
            {"subscription_finalize_pending_success": 0},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_queue_or_subscription_not_exact",
        ),
        (
            {"pose_graph_captured": 0},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 1},
            "existing_replay_pose_graph_not_exact",
        ),
        (
            {},
            {"slam_input_complete": 1},
            {"pose_graph_coverage_complete": 0},
            "existing_replay_pose_graph_not_exact",
        ),
    ],
)
def test_existing_replay_semantics_require_exact_success(
    overrides,
    log_fields,
    coverage,
    expected_reason,
):
    replay = valid_semantic_replay(**overrides)

    assert MODULE.existing_replay_semantic_rerun_reason(
        replay,
        log_fields,
        coverage,
    ) == expected_reason


def test_adapter_signature_accepts_exact_matching_legacy_signature_after_command_rewrite(
    tmp_path,
):
    v4_names = ("summary.json", "map.png", "commands.txt")
    runtime_dump = "runtime_parameters/slam_toolbox.yaml"
    v5_names = v4_names + (runtime_dump,)
    original_resolver = lambda schema: v5_names if schema >= 5 else v4_names
    pipeline = SimpleNamespace(
        STAGE30_ARTIFACT_NAMES_V4=v4_names,
        stage30_artifact_names_for_schema=original_resolver,
        load_slam_toolbox_params=lambda path: {},
    )
    legacy_payload = {
        "schema": 5,
        "bag_path": "/tmp/bag",
        "fairness_contract": {"profile": "same"},
    }
    enhanced_payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: legacy_payload,
        {},
        SimpleNamespace(slam_config=""),
        "launch",
        "capture",
    )
    legacy_signature = MODULE._payload_sha256(legacy_payload)
    enhanced_signature = MODULE._payload_sha256(enhanced_payload)
    fairness_signature = MODULE._payload_sha256(
        legacy_payload["fairness_contract"]
    )
    enhanced_fairness_signature = MODULE._payload_sha256(
        enhanced_payload["fairness_contract"]
    )
    replay, log_fields, coverage = valid_semantic_context()
    legacy_commands = (
        "launch=unchanged\n"
        f"replay_signature_sha256={legacy_signature}\n"
        f"replay_fairness_contract_sha256={fairness_signature}\n"
    )
    replay.update(
        {
            "replay_provenance_schema": 5,
            "replay_signature_sha256": legacy_signature,
            "replay_fairness_contract_sha256": fairness_signature,
            "artifact_manifest": {
                "summary.json": {"size_bytes": 1, "sha256": "summary"},
                "map.png": {"size_bytes": 1, "sha256": "map"},
                "commands.txt": {
                    "size_bytes": len(legacy_commands.encode()),
                    "sha256": MODULE.hashlib.sha256(
                        legacy_commands.encode()
                    ).hexdigest(),
                },
                runtime_dump: {"size_bytes": 1, "sha256": "runtime"},
            },
            "launch_log": str(tmp_path / "launch.log"),
            "capture_log": str(tmp_path / "capture.log"),
        }
    )
    current_commands = (
        "launch=unchanged\n"
        f"replay_signature_sha256={enhanced_signature}\n"
        "replay_fairness_contract_sha256="
        f"{enhanced_fairness_signature}\n"
    )
    (tmp_path / "commands.txt").write_text(current_commands, encoding="utf-8")
    current_manifest = {
        **replay["artifact_manifest"],
        "commands.txt": {
            "size_bytes": len(current_commands.encode()),
            "sha256": MODULE.hashlib.sha256(
                current_commands.encode()
            ).hexdigest(),
        },
    }
    saved_integrity = {
        "replay_result": {
            "path": str(tmp_path / "replay_result.json"),
            "exists": 1,
            "size_bytes": 10,
            "sha256": "replay",
        },
        "commands": {
            "path": str(tmp_path / "commands.txt"),
            "exists": 1,
            **replay["artifact_manifest"]["commands.txt"],
        },
        "pose_graph_export_log": {
            "path": str(tmp_path / "pose_graph_export.log"),
            "exists": 1,
            "size_bytes": 10,
            "sha256": "pose",
        },
        "launch_log": {
            "path": str(tmp_path / "launch.log"),
            "exists": 1,
            "size_bytes": 10,
            "sha256": "launch",
        },
        "capture_log": {
            "path": str(tmp_path / "capture.log"),
            "exists": 1,
            "size_bytes": 10,
            "sha256": "capture",
        },
    }
    (tmp_path / "replay_integrity.json").write_text(
        json.dumps({"schema": 1, "files": saved_integrity}),
        encoding="utf-8",
    )
    current_integrity = {
        **saved_integrity,
        "commands": {
            **saved_integrity["commands"],
            "size_bytes": len(current_commands.encode()),
            "sha256": MODULE.hashlib.sha256(
                current_commands.encode()
            ).hexdigest(),
        },
    }
    pipeline.stage30_artifact_manifest = (
        lambda bag_out, schema: current_manifest
    )
    pipeline.stage30_replay_integrity_manifest = (
        lambda bag_out, replay_result: current_integrity
    )
    observed_signatures = []

    def original_reason(*args):
        observed_signatures.append(args[5])
        return "" if args[5] == replay["replay_signature_sha256"] else "bad"

    reason = MODULE.compatible_existing_replay_rerun_reason(
        pipeline,
        original_reason,
        tmp_path,
        {
            "map_count": 1,
            "trajectory_map_frame_count": 2,
            "trajectory_odom_frame_count": 2,
        },
        replay,
        log_fields,
        coverage,
        enhanced_signature,
        True,
    )

    assert reason == ""
    assert observed_signatures == []
    assert (tmp_path / "commands.txt").read_text(encoding="utf-8") == (
        legacy_commands
    )


def test_adapter_signature_rejects_non_signature_command_change(tmp_path):
    v4_names = ("summary.json", "map.png", "commands.txt")
    runtime_dump = "runtime_parameters/slam_toolbox.yaml"
    pipeline = SimpleNamespace(
        STAGE30_ARTIFACT_NAMES_V4=v4_names,
        stage30_artifact_names_for_schema=lambda schema: v4_names
        + (runtime_dump,),
        load_slam_toolbox_params=lambda path: {},
    )
    legacy_payload = {
        "schema": 5,
        "bag_path": "/tmp/bag",
        "fairness_contract": {"profile": "same"},
    }
    enhanced_payload = MODULE.augment_replay_provenance_payload(
        pipeline,
        lambda *unused: legacy_payload,
        {},
        SimpleNamespace(slam_config=""),
        "launch",
        "capture",
    )
    legacy_signature = MODULE._payload_sha256(legacy_payload)
    enhanced_signature = MODULE._payload_sha256(enhanced_payload)
    fairness_signature = MODULE._payload_sha256(
        legacy_payload["fairness_contract"]
    )
    legacy_commands = (
        "launch=original\n"
        f"replay_signature_sha256={legacy_signature}\n"
        f"replay_fairness_contract_sha256={fairness_signature}\n"
    )
    current_commands = (
        "launch=changed\n"
        f"replay_signature_sha256={enhanced_signature}\n"
        f"replay_fairness_contract_sha256={fairness_signature}\n"
    )
    (tmp_path / "commands.txt").write_text(current_commands, encoding="utf-8")
    replay, log_fields, coverage = valid_semantic_context()
    replay.update(
        {
            "replay_provenance_schema": 5,
            "replay_signature_sha256": legacy_signature,
            "replay_fairness_contract_sha256": fairness_signature,
            "artifact_manifest": {
                "summary.json": {"size_bytes": 1, "sha256": "summary"},
                "map.png": {"size_bytes": 1, "sha256": "map"},
                "commands.txt": {
                    "size_bytes": len(legacy_commands.encode()),
                    "sha256": MODULE.hashlib.sha256(
                        legacy_commands.encode()
                    ).hexdigest(),
                },
                runtime_dump: {"size_bytes": 1, "sha256": "runtime"},
            },
        }
    )
    pipeline.stage30_artifact_manifest = lambda bag_out, schema: {
        **replay["artifact_manifest"],
        "commands.txt": {
            "size_bytes": len(current_commands.encode()),
            "sha256": MODULE.hashlib.sha256(
                current_commands.encode()
            ).hexdigest(),
        },
    }
    pipeline.stage30_replay_integrity_manifest = lambda *unused: {}

    reason = MODULE.compatible_existing_replay_rerun_reason(
        pipeline,
        lambda *unused: "original_reason_called",
        tmp_path,
        {
            "map_count": 1,
            "trajectory_map_frame_count": 2,
            "trajectory_odom_frame_count": 2,
        },
        replay,
        log_fields,
        coverage,
        enhanced_signature,
        True,
    )

    assert reason == "existing_replay_legacy_commands_mismatch"


def test_legacy_command_rewrite_restores_replay_and_fairness_hashes(tmp_path):
    saved_signature = "a" * 64
    adapter_signature = "b" * 64
    saved_fairness = "c" * 64
    current_fairness = "d" * 64
    saved_commands = (
        "launch=unchanged\n"
        f"replay_signature_sha256={saved_signature}\n"
        f"replay_fairness_contract_sha256={saved_fairness}\n"
    ).encode()
    current_commands = (
        "launch=unchanged\n"
        f"replay_signature_sha256={adapter_signature}\n"
        f"replay_fairness_contract_sha256={current_fairness}\n"
    ).encode()
    commands_path = tmp_path / "commands.txt"
    commands_path.write_bytes(current_commands)
    saved_entry = {
        "size_bytes": len(saved_commands),
        "sha256": MODULE.hashlib.sha256(saved_commands).hexdigest(),
    }
    current_entry = {
        "size_bytes": len(current_commands),
        "sha256": MODULE.hashlib.sha256(current_commands).hexdigest(),
    }
    saved_integrity = {
        name: {
            "path": str(tmp_path / name),
            "exists": 1,
            "size_bytes": 10,
            "sha256": name,
        }
        for name in (
            "replay_result",
            "commands",
            "pose_graph_export_log",
            "launch_log",
            "capture_log",
        )
    }
    saved_integrity["commands"].update(saved_entry)
    current_integrity = json.loads(json.dumps(saved_integrity))
    current_integrity["commands"].update(current_entry)
    (tmp_path / "replay_integrity.json").write_text(
        json.dumps({"schema": 1, "files": saved_integrity}),
        encoding="utf-8",
    )
    pipeline = SimpleNamespace(
        stage30_artifact_manifest=lambda bag_out, schema: {
            "commands.txt": current_entry,
        },
        stage30_replay_integrity_manifest=lambda bag_out, replay: (
            current_integrity
        ),
    )

    reason = MODULE.legacy_command_rewrite_integrity_rerun_reason(
        pipeline,
        tmp_path,
        {
            "replay_provenance_schema": 5,
            "artifact_manifest": {"commands.txt": saved_entry},
        },
        adapter_signature,
        saved_signature,
        current_fairness,
        saved_fairness,
    )

    assert reason == ""
    assert commands_path.read_bytes() == saved_commands


def test_command_rewrite_restores_saved_ros_domain_only(tmp_path):
    signature = "a" * 64
    fairness = "b" * 64
    saved_commands = (
        "ROS_DOMAIN_ID=105\n"
        "launch=unchanged\n"
        f"replay_signature_sha256={signature}\n"
        f"replay_fairness_contract_sha256={fairness}\n"
    ).encode()
    current_commands = saved_commands.replace(
        b"ROS_DOMAIN_ID=105",
        b"ROS_DOMAIN_ID=41",
    )
    commands_path = tmp_path / "commands.txt"
    commands_path.write_bytes(current_commands)
    saved_entry = {
        "size_bytes": len(saved_commands),
        "sha256": MODULE.hashlib.sha256(saved_commands).hexdigest(),
    }
    current_entry = {
        "size_bytes": len(current_commands),
        "sha256": MODULE.hashlib.sha256(current_commands).hexdigest(),
    }
    saved_integrity = {
        name: {
            "path": str(tmp_path / name),
            "exists": 1,
            "size_bytes": 10,
            "sha256": name,
        }
        for name in (
            "replay_result",
            "commands",
            "pose_graph_export_log",
            "launch_log",
            "capture_log",
        )
    }
    saved_integrity["commands"].update(saved_entry)
    current_integrity = json.loads(json.dumps(saved_integrity))
    current_integrity["commands"].update(current_entry)
    (tmp_path / "replay_integrity.json").write_text(
        json.dumps({"schema": 1, "files": saved_integrity}),
        encoding="utf-8",
    )
    pipeline = SimpleNamespace(
        stage30_artifact_manifest=lambda bag_out, schema: {
            "commands.txt": current_entry,
        },
        stage30_replay_integrity_manifest=lambda bag_out, replay: (
            current_integrity
        ),
    )

    reason = MODULE.legacy_command_rewrite_integrity_rerun_reason(
        pipeline,
        tmp_path,
        {
            "replay_provenance_schema": 5,
            "replay_signature_sha256": signature,
            "replay_fairness_contract_sha256": fairness,
            "ros_domain_id": 105,
            "artifact_manifest": {"commands.txt": saved_entry},
        },
        signature,
        signature,
        fairness,
        fairness,
    )

    assert reason == ""
    assert commands_path.read_bytes() == saved_commands


def test_rejected_reuse_restores_commands_for_the_actual_replay(tmp_path):
    commands_path = tmp_path / "commands.txt"
    current_commands = b"ROS_DOMAIN_ID=46\nlaunch=current\n"
    saved_commands = b"ROS_DOMAIN_ID=106\nlaunch=legacy\n"
    commands_path.write_bytes(saved_commands)

    MODULE.restore_current_commands_after_reuse_rejection(
        commands_path,
        current_commands,
    )

    assert commands_path.read_bytes() == current_commands


def test_rejected_reuse_without_current_commands_does_not_create_file(tmp_path):
    commands_path = tmp_path / "commands.txt"

    MODULE.restore_current_commands_after_reuse_rejection(
        commands_path,
        None,
    )

    assert not commands_path.exists()


def test_legacy_payload_allows_only_inactive_guard_and_unused_launch_sources(
    tmp_path,
):
    standard_launch = (
        "/home/user/gole_ws/src/gole_autonomy/gole_slam/"
        "slam_2d_bringup/launch/slam_from_bag_launch.py"
    )
    guard_source = (
        "/home/user/gole_ws/src/gole_autonomy/gole_localization/"
        "amcl_guard/src/amcl_guard.cpp"
    )
    current = {
        "schema": 5,
        "launch_command": (
            "ros2 launch /tmp/frozen_stage34.launch.py "
            "enable_laserscan_roughness_gate:=false"
        ),
        "source_sha256": {
            standard_launch: "current-unused-launch",
            guard_source: "current-inactive-guard",
            "/tmp/roughness.cpp": "same-core",
        },
        "fairness_contract": {
            "laserscan_guard_profile": {"enabled": False},
            "backend_asset_sha256": {
                "slam_launch_file": "same-frozen-launch",
                "slam_from_bag_launch": "current-unused-launch",
                "amcl_guard_node": "current-inactive-node",
                "amcl_guard_source": "current-inactive-guard",
                "roughness_innovation_frontend": "same-core",
            },
        },
    }
    saved = json.loads(json.dumps(current))
    saved["source_sha256"][standard_launch] = "saved-unused-launch"
    saved["source_sha256"][guard_source] = "saved-inactive-guard"
    backend = saved["fairness_contract"]["backend_asset_sha256"]
    backend["slam_from_bag_launch"] = "saved-unused-launch"
    backend["amcl_guard_node"] = "saved-inactive-node"
    backend["amcl_guard_source"] = "saved-inactive-guard"
    (tmp_path / "replay_provenance.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )
    saved_signature = MODULE._payload_sha256(saved)

    assert MODULE.compatible_saved_legacy_signature(
        tmp_path,
        {
            "replay_signature_sha256": saved_signature,
            "enable_laserscan_roughness_gate": 0,
        },
        current,
    ) == (saved_signature, "")


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("active_guard", "existing_replay_legacy_provenance_mismatch"),
        ("roughness_core", "existing_replay_legacy_provenance_mismatch"),
        ("actual_launch", "existing_replay_legacy_launch_asset_mismatch"),
        ("saved_hash", "existing_replay_saved_provenance_signature_mismatch"),
    ],
)
def test_legacy_payload_rejects_operational_provenance_changes(
    tmp_path,
    mutation,
    expected_reason,
):
    standard_launch = (
        "/home/user/gole_ws/src/gole_autonomy/gole_slam/"
        "slam_2d_bringup/launch/slam_from_bag_launch.py"
    )
    guard_source = (
        "/home/user/gole_ws/src/gole_autonomy/gole_localization/"
        "amcl_guard/src/amcl_guard.cpp"
    )
    current = {
        "schema": 5,
        "launch_command": (
            "ros2 launch /tmp/frozen_stage34.launch.py "
            "enable_laserscan_roughness_gate:=false"
        ),
        "source_sha256": {
            standard_launch: "current-unused-launch",
            guard_source: "current-guard",
            "/tmp/roughness.cpp": "current-core",
        },
        "fairness_contract": {
            "laserscan_guard_profile": {"enabled": False},
            "backend_asset_sha256": {
                "slam_launch_file": "current-frozen-launch",
                "slam_from_bag_launch": "current-unused-launch",
                "amcl_guard_source": "current-guard",
                "roughness_innovation_frontend": "current-core",
            },
        },
    }
    saved = json.loads(json.dumps(current))
    saved["source_sha256"][standard_launch] = "saved-unused-launch"
    saved["fairness_contract"]["backend_asset_sha256"][
        "slam_from_bag_launch"
    ] = "saved-unused-launch"
    if mutation == "active_guard":
        saved["fairness_contract"]["laserscan_guard_profile"]["enabled"] = True
        saved["source_sha256"][guard_source] = "saved-guard"
    elif mutation == "roughness_core":
        saved["source_sha256"]["/tmp/roughness.cpp"] = "saved-core"
    elif mutation == "actual_launch":
        saved["fairness_contract"]["backend_asset_sha256"][
            "slam_launch_file"
        ] = "saved-frozen-launch"
    (tmp_path / "replay_provenance.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )
    saved_signature = MODULE._payload_sha256(saved)
    saved_guard_enabled = 0
    if mutation == "saved_hash":
        saved_signature = "0" * 64
    elif mutation == "active_guard":
        saved_guard_enabled = 1

    assert MODULE.compatible_saved_legacy_signature(
        tmp_path,
        {
            "replay_signature_sha256": saved_signature,
            "enable_laserscan_roughness_gate": saved_guard_enabled,
        },
        current,
    ) == ("", expected_reason)
