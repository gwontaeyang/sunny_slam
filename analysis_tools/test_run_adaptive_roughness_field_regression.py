import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("run_adaptive_roughness_field_regression.py")
SPEC = importlib.util.spec_from_file_location(
    "run_adaptive_roughness_field_regression",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def arguments(tmp_path):
    return argparse.Namespace(
        manifest=tmp_path / "manifest.json",
        pipeline=tmp_path / "pipeline.py",
        staging_root=tmp_path / "bags",
        output_root=tmp_path / "output",
        workspace=tmp_path / "workspace",
        setup_bash=tmp_path / "setup.bash",
        control_config=tmp_path / "control.yaml",
        adaptive_config=tmp_path / "adaptive.yaml",
        control_label="control",
        adaptive_label="adaptive",
        bag_play_rate=5.0,
        ros_domain_base=180,
        ros_domain_stride=32,
        jobs=1,
        scan_qos_depth=20000,
        queue_drain_timeout_s=600.0,
        queue_drain_quiet_s=60.0,
        startup_tf_drop_limit=10,
        skip_existing=True,
    )


def test_usable_bag_names_filters_integrity_and_status(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "name": "A",
                        "status": "ready",
                        "storage_integrity_status": "passed",
                    },
                    {
                        "name": "B",
                        "status": "reindexed",
                        "storage_integrity_status": "passed",
                    },
                    {
                        "name": "bad",
                        "status": "storage_corrupt",
                        "storage_integrity_status": "failed",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.usable_bag_names(manifest) == ["A", "B"]


def test_select_bag_names_preserves_manifest_order():
    selected = MODULE.select_bag_names(
        ["A", "B", "C"],
        ["C", "A"],
        0,
    )

    assert selected == ["A", "C"]


def test_select_bag_names_rejects_unknown_and_duplicate_requests():
    with pytest.raises(ValueError, match="not usable"):
        MODULE.select_bag_names(["A"], ["missing"], 0)
    with pytest.raises(ValueError, match="must be unique"):
        MODULE.select_bag_names(["A"], ["A", "A"], 0)


def test_select_bag_names_applies_limit_after_explicit_filter():
    assert MODULE.select_bag_names(
        ["A", "B", "C"],
        ["B", "C"],
        1,
    ) == ["B"]


def test_balanced_bag_shards_are_deterministic_and_duration_balanced():
    names = ["A", "B", "C", "D", "E"]
    durations = {"A": 100.0, "B": 90.0, "C": 20.0, "D": 10.0, "E": 5.0}

    shards = MODULE.balanced_bag_shards(names, durations, 2)

    assert shards == [["A", "D", "E"], ["B", "C"]]
    assert sorted(name for shard in shards for name in shard) == sorted(names)


def test_balanced_bag_shards_never_emit_empty_zero_duration_shards():
    names = ["A", "B", "C"]

    shards = MODULE.balanced_bag_shards(names, {}, 3)

    assert shards == [["A"], ["B"], ["C"]]


def test_shard_domain_bases_are_disjoint_and_bounded():
    shards = [["A", "B"], ["C", "D", "E"]]

    assert MODULE.shard_domain_bases(shards, 40, 16) == [40, 56]
    with pytest.raises(ValueError, match="overlap"):
        MODULE.shard_domain_bases(shards, 40, 1)
    with pytest.raises(ValueError, match="0..230"):
        MODULE.shard_domain_bases(shards, 229, 16)


def test_phase_commands_assign_shards_and_add_combined_summary(tmp_path):
    args = arguments(tmp_path)
    args.jobs = 2
    args.ros_domain_base = 40
    args.ros_domain_stride = 16

    commands, consolidation = MODULE.phase_commands(
        args,
        ["A", "B", "C"],
        {"A": 30.0, "B": 20.0, "C": 10.0},
        "control",
    )

    assert len(commands) == 2
    assert [
        command[command.index("--ros-domain-base") + 1]
        for command in commands
    ] == ["40", "56"]
    assert consolidation is not None
    assert "--skip-existing" in consolidation
    assert consolidation[-4:] == ["--bag-list", "A", "B", "C"]


def test_adaptive_command_declares_only_expected_experiment_differences(tmp_path):
    command = MODULE.phase_command(arguments(tmp_path), ["A", "B"], "adaptive")

    assert command[command.index("--workspace") + 1] == str(
        (tmp_path / "workspace").resolve()
    )
    assert command[command.index("--snapshot-label") + 1] == "adaptive"
    assert command[command.index("--reference-variant-root") + 1].endswith(
        "/variant/control"
    )
    assert command.count("--allow-replay-fairness-mismatch-field") == len(
        MODULE.ALLOWED_ADAPTIVE_DIFFERENCES
    )
    assert command[-3:] == ["--bag-list", "A", "B"]


def test_validate_inputs_rejects_missing_setup_before_replay(tmp_path):
    args = arguments(tmp_path)
    for path in (
        args.manifest,
        args.pipeline,
        args.control_config,
        args.adaptive_config,
    ):
        path.write_text("", encoding="utf-8")
    args.staging_root.mkdir()
    (args.staging_root / "A").mkdir()
    args.workspace.mkdir()

    with pytest.raises(FileNotFoundError, match="setup.bash"):
        MODULE.validate_inputs(args, ["A"])


def test_validate_inputs_accepts_complete_replay_inputs(tmp_path):
    args = arguments(tmp_path)
    for path in (
        args.manifest,
        args.setup_bash,
        args.pipeline,
        args.control_config,
        args.adaptive_config,
    ):
        path.write_text("", encoding="utf-8")
    args.staging_root.mkdir()
    (args.staging_root / "A").mkdir()
    args.workspace.mkdir()

    MODULE.validate_inputs(args, ["A"])


def test_validate_variant_requires_complete_sensor_and_graph_evidence(tmp_path):
    output = tmp_path / "output"
    result_path = (
        output
        / "stage34_roughness_gate_replay"
        / "variant"
        / "control"
        / "bags"
        / "A"
        / "replay_result.json"
    )
    result_path.parent.mkdir(parents=True)
    payload = {
        "status": "map_and_map_trajectory",
        "queue_drain_complete": 1,
        "slam_recorded_scan_delivery_complete": 1,
        "roughness_imu_delivery_complete": 1,
        "pose_graph_coverage_status": "complete",
        "runtime_slam_parameter_match": 1,
        "runtime_slam_adaptive_constraint_parameter_match": 1,
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    assert MODULE.validate_variant(output, "control", ["A"])["complete"] == 1

    payload["queue_drain_complete"] = 0
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError):
        MODULE.validate_variant(output, "control", ["A"])


def test_validate_variant_requires_exact_current_provenance(tmp_path):
    output = tmp_path / "output"
    bag_dir = (
        output
        / "stage34_roughness_gate_replay"
        / "variant"
        / "control"
        / "bags"
        / "A"
    )
    bag_dir.mkdir(parents=True)
    expected = {
        "lossless_adapter_sha256": "a" * 64,
        "runtime_slam_node_sha256": "b" * 64,
        "replay_pose_graph_exporter_sha256": "c" * 64,
        "runtime_cmake_cache_sha256": "d" * 64,
        "slam_config_sha256": "e" * 64,
    }
    (bag_dir / "replay_result.json").write_text(
        json.dumps(
            {
                "status": "map_and_map_trajectory",
                "queue_drain_complete": 1,
                "slam_recorded_scan_delivery_complete": 1,
                "roughness_imu_delivery_complete": 1,
                "pose_graph_coverage_status": "complete",
                "runtime_slam_parameter_match": 1,
                "runtime_slam_adaptive_constraint_parameter_match": 1,
                "runtime_slam_adaptive_constraint_parameter_actual_json": (
                    '{"roughness_constraint_mode":"intervention_off"}'
                ),
            }
        ),
        encoding="utf-8",
    )
    provenance = {
        "lossless_stage34_adapter": {"sha256": expected["lossless_adapter_sha256"]},
        "fairness_contract": {
            "backend_asset_sha256": {
                "roughness_innovation_node": expected["runtime_slam_node_sha256"],
                "pose_graph_exporter": expected[
                    "replay_pose_graph_exporter_sha256"
                ],
                "slam_config": expected["slam_config_sha256"],
            },
            "slam_2d_runtime_build": {
                "cmake_cache_sha256": expected["runtime_cmake_cache_sha256"]
            },
        },
    }
    (bag_dir / "replay_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )

    assert MODULE.validate_variant(
        output, "control", ["A"], expected, "intervention_off"
    )["complete"] == 1

    expected["lossless_adapter_sha256"] = "9" * 64
    with pytest.raises(RuntimeError):
        MODULE.validate_variant(
            output, "control", ["A"], expected, "intervention_off"
        )

    report = MODULE.validate_variant(
        output,
        "control",
        ["A"],
        expected,
        "intervention_off",
        {"a" * 64},
    )
    assert report["complete"] == 1
    assert report["observed_lossless_adapter_sha256_counts"] == {
        "a" * 64: 1
    }

    expected["runtime_slam_node_sha256"] = "8" * 64
    with pytest.raises(RuntimeError):
        MODULE.validate_variant(
            output,
            "control",
            ["A"],
            expected,
            "intervention_off",
            {"a" * 64},
        )
