import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_adaptive_replay_provenance.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_adaptive_replay_provenance", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_replay(
    root,
    label,
    bag,
    domain,
    fingerprint="abc",
    nodes=10,
    status="map_and_map_trajectory",
    coverage="complete",
):
    bag_dir = (
        root
        / "stage34_roughness_gate_replay"
        / "variant"
        / label
        / "bags"
        / bag
    )
    bag_dir.mkdir(parents=True)
    commands = bag_dir / "commands.txt"
    commands.write_text(f"ROS_DOMAIN_ID={domain}\nlaunch=current\n", encoding="utf-8")
    provenance = {
        "lossless_stage34_adapter": {"sha256": "a" * 64},
        "fairness_contract": {
            "backend_asset_sha256": {
                "roughness_innovation_node": "b" * 64,
                "pose_graph_exporter": "c" * 64,
                "slam_config": "d" * 64,
            },
            "slam_2d_runtime_build": {"cmake_cache_sha256": "e" * 64},
        },
    }
    provenance_path = bag_dir / "replay_provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    result = {
        "status": status,
        "ros_domain_id": domain,
        "queue_drain_complete": 1,
        "slam_recorded_scan_delivery_complete": 1,
        "roughness_imu_delivery_complete": 1,
        "pose_graph_coverage_status": coverage,
        "runtime_slam_parameter_match": 1,
        "runtime_slam_adaptive_constraint_parameter_match": 1,
        "runtime_slam_adaptive_constraint_parameter_actual_json": json.dumps(
            {"roughness_constraint_mode": "intervention_off"}
        ),
        "slam_recorded_scan_effective_expected_count": 100,
        "slam_recorded_scan_received_count": 100,
        "slam_recorded_scan_expected_fingerprint_fnv1a64": fingerprint,
        "slam_recorded_scan_received_fingerprint_fnv1a64": fingerprint,
        "slam_recorded_scan_stamp_sequence_complete": 1,
        "pose_graph_node_count": nodes,
        "artifact_manifest": {
            "commands.txt": {
                "size_bytes": commands.stat().st_size,
                "sha256": sha256(commands),
            },
            "replay_provenance.json": {
                "size_bytes": provenance_path.stat().st_size,
                "sha256": sha256(provenance_path),
            },
        },
    }
    result_path = bag_dir / "replay_result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_audit_replay_verifies_domain_manifest_and_completion(tmp_path):
    path = make_replay(tmp_path, "control", "bag", 42)

    row = MODULE.audit_replay(path)

    assert row["replay_valid"] == 1
    assert row["domain_match"] == 1
    assert row["artifact_manifest_valid"] == 1
    assert row["mode"] == "intervention_off"


def test_audit_replay_accepts_stationary_pose_graph(tmp_path):
    path = make_replay(
        tmp_path,
        "control",
        "stationary",
        42,
        nodes=1,
        status="map_and_stationary_pose_graph",
        coverage="stationary_single_node_complete",
    )

    assert MODULE.audit_replay(path)["replay_valid"] == 1


def test_audit_replay_detects_stale_commands_and_artifact(tmp_path):
    path = make_replay(tmp_path, "control", "bag", 42)
    (path.parent / "commands.txt").write_text(
        "ROS_DOMAIN_ID=41\nlaunch=stale\n", encoding="utf-8"
    )

    row = MODULE.audit_replay(path)

    assert row["replay_valid"] == 0
    assert row["domain_match"] == 0
    assert row["artifact_manifest_valid"] == 0
    assert "commands.txt:size" in row["artifact_manifest_errors"]


def test_pair_rows_detects_scan_fingerprint_and_count_mismatch(tmp_path):
    control = MODULE.audit_replay(
        make_replay(tmp_path, "control", "bag", 42, "aaa", 10)
    )
    adaptive = MODULE.audit_replay(
        make_replay(tmp_path, "adaptive", "bag", 43, "bbb", 11)
    )

    pair = MODULE.pair_rows(
        [control, adaptive], [("control", "adaptive")]
    )[0]

    assert pair["scan_input_parity"] == 0
    assert pair["control_pose_graph_node_count"] == 10
    assert pair["adaptive_pose_graph_node_count"] == 11


def test_pair_rows_accepts_identical_scan_evidence(tmp_path):
    control = MODULE.audit_replay(
        make_replay(tmp_path, "control", "bag", 42, "same", 10)
    )
    adaptive = MODULE.audit_replay(
        make_replay(tmp_path, "adaptive", "bag", 43, "same", 10)
    )

    pair = MODULE.pair_rows(
        [control, adaptive], [("control", "adaptive")]
    )[0]

    assert pair["scan_input_parity"] == 1
