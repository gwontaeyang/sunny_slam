import base64
import importlib.util
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name("slam_2d_field_regression.py")
SPEC = importlib.util.spec_from_file_location("slam_2d_field_regression", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_bag(root: Path, name: str, *, metadata: bool, size: int) -> Path:
    bag = root / name
    bag.mkdir(parents=True)
    (bag / f"{name}_0.db3").write_bytes(b"x" * size)
    if metadata:
        (bag / "metadata.yaml").write_text(
            "rosbag2_bagfile_information:\n"
            "  duration: {nanoseconds: 1000000000}\n"
            "  topics_with_message_count:\n"
            "    - topic_metadata: {name: /scan_laserscan}\n"
            "      message_count: 10\n"
            "    - topic_metadata: {name: /imu/data}\n"
            "      message_count: 20\n",
            encoding="utf-8",
        )
    return bag


def test_discover_bags_classifies_ready_reindex_and_empty(tmp_path):
    root = tmp_path / "01_input"
    root.mkdir()
    make_bag(root, "ty_bag_ready", metadata=True, size=8)
    make_bag(root, "ty_bag_reindex", metadata=False, size=8)
    make_bag(root, "ty_bag_empty", metadata=False, size=0)

    records = {record.name: record for record in MODULE.discover_bags([root])}

    assert records["ty_bag_ready"].status == "ready"
    assert records["ty_bag_reindex"].status == "needs_reindex"
    assert records["ty_bag_empty"].status == "empty_storage"


def test_discover_bags_rejects_duplicate_names(tmp_path):
    first = tmp_path / "01_input"
    second = tmp_path / "03_input"
    first.mkdir()
    second.mkdir()
    make_bag(first, "ty_bag_same", metadata=True, size=8)
    make_bag(second, "ty_bag_same", metadata=True, size=8)

    with pytest.raises(RuntimeError, match="duplicate bag name"):
        MODULE.discover_bags([first, second])


def test_stage34_candidate_changes_only_spatial_passage_flag(tmp_path):
    common = dict(
        pipeline=tmp_path / "pipeline.py",
        setup_bash=tmp_path / "setup.bash",
        staging_root=tmp_path / "bags",
        output_root=tmp_path / "out",
        slam_config=tmp_path / "mapper.yaml",
        bag_names=["ty_bag_one"],
        snapshot_label="candidate",
        rate=5.0,
        skip_existing=True,
        ros_domain_base=180,
        scan_qos_depth=20000,
        post_playback_wait_s=60.0,
    )
    baseline = MODULE.build_stage34_command(
        **common, spatial_passage=False, reference_root=None
    )
    candidate = MODULE.build_stage34_command(
        **common,
        spatial_passage=True,
        reference_root=tmp_path / "reference",
    )

    assert "--no-roughness-innovation-enable-spatial-passage-hysteresis" in baseline
    assert "--roughness-innovation-enable-spatial-passage-hysteresis" in candidate
    assert "roughness_innovation_profile.enable_spatial_passage_hysteresis" in candidate
    assert "20000" in candidate
    startup_limit_index = candidate.index(
        "--slam-input-max-startup-tf-drop-count"
    )
    assert candidate[startup_limit_index + 1] == "10"


def test_prepare_replay_config_does_not_modify_source(tmp_path):
    source = tmp_path / "mapper.yaml"
    original = (
        "slam_toolbox:\n"
        "  ros__parameters:\n"
        "    scan_queue_size: 100\n"
        "    tf_buffer_duration: 40.0\n"
    )
    source.write_text(original, encoding="utf-8")
    destination = tmp_path / "derived/mapper_replay.yaml"

    MODULE.prepare_replay_slam_config(
        source,
        destination,
        scan_queue_size=20000,
        tf_buffer_duration_s=1800.0,
    )

    assert source.read_text(encoding="utf-8") == original
    payload = MODULE.yaml.safe_load(destination.read_text(encoding="utf-8"))
    parameters = payload["slam_toolbox"]["ros__parameters"]
    assert parameters["scan_queue_size"] == 20000
    assert parameters["tf_buffer_duration"] == 1800.0
    assert parameters["roughness_imu_buffer_duration_s"] == 1800.0
    assert parameters["roughness_imu_qos_reliable"] is True
    assert parameters["roughness_imu_qos_depth"] == 200000
    assert parameters["roughness_diagnostics_period_s"] == 2.0
    assert parameters["roughness_pending_timeout_s"] == 600.0
    assert parameters["roughness_max_pending_scans"] == 20000


def test_storage_integrity_audit_excludes_malformed_sqlite(tmp_path):
    ready = MODULE.BagRecord(
        name="ty_bag_ready",
        source_root="01_input",
        source_path=str(tmp_path / "source"),
        metadata_present=True,
        storage_files=1,
        storage_bytes=4096,
        status="ready",
        staged_path=str(tmp_path / "ready"),
    )
    corrupt = MODULE.BagRecord(
        name="ty_bag_corrupt",
        source_root="01_input",
        source_path=str(tmp_path / "source"),
        metadata_present=True,
        storage_files=1,
        storage_bytes=4096,
        status="ready",
        staged_path=str(tmp_path / "corrupt"),
    )
    for record in (ready, corrupt):
        Path(record.staged_path).mkdir()
    database = Path(ready.staged_path) / "ready.db3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE messages(id INTEGER PRIMARY KEY, data BLOB)")
    connection.commit()
    connection.close()
    (Path(corrupt.staged_path) / "corrupt.db3").write_bytes(b"not sqlite")

    MODULE.audit_storage_integrity([ready, corrupt])

    assert ready.usable
    assert ready.storage_integrity_status == "passed"
    assert not corrupt.usable
    assert corrupt.status == "storage_corrupt"
    assert corrupt.storage_integrity_status == "failed"


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_map_only_review_renders_current_maps_and_publishes_complete_set(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    comparisons = tmp_path / "comparisons"
    for root in (baseline, post):
        bag = root / "bags" / "ty_bag_one"
        bag.mkdir(parents=True)
        (bag / "map.png").write_bytes(PNG)
    comparison_bag = comparisons / "ty_bag_one"
    comparison_bag.mkdir(parents=True)
    stale_comparison = comparison_bag / "map_comparison_side_by_side.png"
    stale_comparison.write_bytes(b"stale comparison must not be copied")

    output = tmp_path / "00_latest_map_review"
    stale = output / "base_line/ty_bag_stale.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(PNG)
    counts = MODULE.create_map_only_review(
        baseline_root=baseline,
        post_root=post,
        comparison_root=comparisons,
        output_root=output,
    )

    assert counts == {"baseline": 1, "post": 1, "comparison": 1}
    assert (output / "base_line/ty_bag_one.png").is_file()
    assert (output / "post/ty_bag_one.png").is_file()
    assert (output / "comparison/ty_bag_one.png").is_file()
    assert (
        output / "comparison/ty_bag_one.png"
    ).read_bytes() != stale_comparison.read_bytes()
    assert (output / "00_all_bags_contact_sheet.png").is_file()
    assert not stale.exists()
    assert not list(output.rglob("*.csv"))
    assert not list(output.rglob("*.json"))


def test_map_only_review_preserves_published_review_when_pair_is_missing(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    baseline_bag = baseline / "bags" / "ty_bag_one"
    baseline_bag.mkdir(parents=True)
    (baseline_bag / "map.png").write_bytes(PNG)
    (post / "bags").mkdir(parents=True)
    output = tmp_path / "00_latest_map_review"
    output.mkdir()
    marker = output / "previous-review.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing paired map"):
        MODULE.create_map_only_review(
            baseline_root=baseline,
            post_root=post,
            comparison_root=tmp_path / "unused-comparisons",
            output_root=output,
            expected_bags=["ty_bag_one"],
        )

    assert marker.read_text(encoding="utf-8") == "keep me"
    assert list(output.iterdir()) == [marker]
    assert not list(tmp_path.glob(".00_latest_map_review.staged-*"))


def test_atomic_review_publish_rolls_back_when_final_rename_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "review"
    staged = tmp_path / ".review.staged"
    output.mkdir()
    staged.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_replace = MODULE.os.replace

    monkeypatch.setattr(MODULE, "_rename_exchange", lambda *_: False)

    def fail_staged_publish(source, destination):
        if Path(source) == staged and Path(destination) == output:
            raise OSError("injected publish failure")
        return original_replace(source, destination)

    monkeypatch.setattr(MODULE.os, "replace", fail_staged_publish)

    with pytest.raises(OSError, match="injected publish failure"):
        MODULE._publish_review_directory(staged, output)

    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "new.txt").exists()
    assert not list(tmp_path.glob(".review.backup-*"))


def valid_replay_result(*, stationary: bool = False) -> dict:
    result = {
        "status": (
            "map_and_stationary_pose_graph"
            if stationary
            else "map_and_map_trajectory"
        ),
        "attempted": 1,
        "bag_playback_finished_observed": 1,
        "launch_return_code_after_stop": 0,
        "capture_return_code_after_stop": 0,
        "runtime_slam_parameter_dump_return_code": 0,
        "runtime_slam_parameter_match": 1,
        "runtime_slam_parameter_mismatch_count": 0,
        "runtime_slam_parameter_status": "matched",
        "runtime_slam_actual_enable_roughness_innovation_gate": True,
        "runtime_slam_actual_roughness_enable_post_confirmation": True,
        "enable_laserscan_roughness_gate": 0,
        "slam_input_complete": 1,
        "slam_input_log_complete": 1,
        "slam_input_status": "complete",
        "slam_input_log_status": "complete",
        "slam_recorded_scan_delivery_required": 1,
        "slam_recorded_scan_delivery_complete": 1,
        "slam_recorded_scan_metadata_count_match": 1,
        "slam_recorded_scan_stamp_audit_available": 1,
        "slam_recorded_scan_stamp_fingerprint_match": 1,
        "slam_recorded_scan_stamp_sequence_complete": 1,
        "slam_recorded_scan_bag_audit_status": "valid",
        "slam_recorded_scan_delivery_status": "complete",
        "slam_recorded_scan_missing_count": 0,
        "slam_recorded_scan_extra_count": 0,
        "slam_recorded_scan_unmatched_startup_stamp_count": 0,
        "slam_unexpected_scan_drop_count": 0,
        "roughness_imu_delivery_complete": 1,
        "roughness_imu_delivery_fingerprint_match": 1,
        "roughness_imu_delivery_count_delta": 0,
        "roughness_imu_delivery_expected_count": 20,
        "roughness_imu_delivery_received_count": 20,
        "subscription_drain_complete": 1,
        "subscription_finalize_pending_success": 1,
        "subscription_drain_effective_expected_scan_count": 10,
        "subscription_drain_received_scan_count": 10,
        "subscription_drain_expected_imu_count": 20,
        "subscription_drain_received_imu_count": 20,
        "queue_drain_enabled": 1,
        "queue_drain_complete": 1,
        "queue_drain_status": "drained_zero_depth",
        "queue_drain_evidence": "direct_node_diagnostics",
        "slam_queue_direct_latest_pending_depth": 0,
        "slam_queue_direct_latest_scan_depth": 0,
        "slam_queue_direct_post_playback_pending_depth": 0,
        "slam_queue_direct_post_playback_scan_depth": 0,
        "pose_graph_captured": 1,
        "pose_graph_coverage_complete": 1,
        "pose_graph_coverage_status": (
            "stationary_single_node_complete" if stationary else "complete"
        ),
        "pose_graph_trajectory_count": 1 if stationary else 10,
        "pose_graph_serialize_return_code": 0,
        "pose_graph_serialize_service_result": 0,
        "pose_graph_export_return_code": 0,
        "stationary_replay": int(stationary),
    }
    if stationary:
        result["pose_graph_stationary_single_node"] = 1
    else:
        result["pose_graph_motion_gap_complete"] = 1
    return result


def write_replay_evidence(
    variant: Path,
    bag_name: str,
    *,
    spatial_passage: bool,
    stationary: bool = False,
    contract_extra: dict | None = None,
) -> Path:
    bag = variant / "bags" / bag_name
    bag.mkdir(parents=True)
    for name in MODULE.REPLAY_ARTIFACT_NAMES_V4:
        if name == "replay_provenance.json":
            continue
        artifact = bag / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(PNG if name.endswith(".png") else b"evidence\n")
    for name in (
        "runtime_parameters/slam_toolbox.yaml",
        "roughness_innovation_events.csv",
        "roughness_post_confirmation_events.csv",
    ):
        artifact = bag / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence\n", encoding="utf-8")

    contract = {
        "bag": {"name": bag_name},
        "playback": {"bag_play_rate": 5.0},
        "roughness_innovation_profile": {
            "enabled": True,
            "enable_post_confirmation": True,
            "enable_spatial_passage_hysteresis": spatial_passage,
        },
    }
    if contract_extra:
        contract.update(contract_extra)
    provenance = {
        "schema": 5,
        "bag_path": f"/staged/{bag_name}",
        "fairness_contract": contract,
    }
    (bag / "replay_provenance.json").write_text(
        MODULE.json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    logs = variant / "logs"
    logs.mkdir(exist_ok=True)
    launch_log = logs / f"{bag_name}_launch.log"
    capture_log = logs / f"{bag_name}_capture.log"
    launch_log.write_text("launch complete\n", encoding="utf-8")
    capture_log.write_text("capture complete\n", encoding="utf-8")
    result = valid_replay_result(stationary=stationary)
    result.update(
        {
            "bag": bag_name,
            "launch_log": str(launch_log.resolve()),
            "capture_log": str(capture_log.resolve()),
            "replay_provenance_schema": 5,
            "replay_signature_sha256": MODULE._canonical_payload_sha256(
                provenance
            ),
            "replay_fairness_contract": contract,
            "replay_fairness_contract_sha256": (
                MODULE._canonical_payload_sha256(contract)
            ),
            "artifact_manifest": MODULE._artifact_manifest(
                bag, MODULE.REPLAY_ARTIFACT_NAMES_V5
            ),
        }
    )
    (bag / "replay_result.json").write_text(
        MODULE.json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bag / "replay_integrity.json").write_text(
        MODULE.json.dumps(
            {
                "schema": 1,
                "files": MODULE._replay_integrity_manifest(bag, result),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return bag


def rewrite_result_and_integrity(bag: Path, **changes) -> None:
    result = MODULE.json.loads(
        (bag / "replay_result.json").read_text(encoding="utf-8")
    )
    result.update(changes)
    (bag / "replay_result.json").write_text(
        MODULE.json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (bag / "replay_integrity.json").write_text(
        MODULE.json.dumps(
            {
                "schema": 1,
                "files": MODULE._replay_integrity_manifest(bag, result),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def rewrite_pair_fairness_evidence(
    bag: Path,
    *,
    contract: dict,
    launch_command: str,
    **result_changes,
) -> None:
    provenance = MODULE.json.loads(
        (bag / "replay_provenance.json").read_text(encoding="utf-8")
    )
    provenance["fairness_contract"] = contract
    provenance["launch_command"] = launch_command
    (bag / "replay_provenance.json").write_text(
        MODULE.json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    result = MODULE.json.loads(
        (bag / "replay_result.json").read_text(encoding="utf-8")
    )
    result.update(result_changes)
    result["replay_signature_sha256"] = MODULE._canonical_payload_sha256(
        provenance
    )
    result["replay_fairness_contract"] = contract
    result["replay_fairness_contract_sha256"] = MODULE._canonical_payload_sha256(
        contract
    )
    (bag / "replay_result.json").write_text(
        MODULE.json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_input_manifest(path: Path, bag_names: list[str]) -> Path:
    path.write_text(
        MODULE.json.dumps(
            {
                "records": [
                    {
                        "name": name,
                        "status": "ready",
                        "storage_integrity_status": "passed",
                    }
                    for name in bag_names
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def audit_args(
    tmp_path: Path,
    manifest: Path,
    baseline: Path,
    post: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=str(manifest),
        baseline_root=str(baseline),
        post_root=str(post),
        output=str(tmp_path / "audit"),
    )


def test_replay_audit_accepts_dynamic_and_stationary_results(tmp_path):
    variant = tmp_path / "variant"
    for bag_name, stationary in (("ty_bag_move", False), ("ty_bag_stop", True)):
        write_replay_evidence(
            variant,
            bag_name,
            spatial_passage=False,
            stationary=stationary,
        )

    rows, summary = MODULE.audit_replay_variant(
        "baseline",
        variant,
        ["ty_bag_move", "ty_bag_stop"],
    )

    assert all(row["audit_pass"] == 1 for row in rows)
    assert summary["passed_bag_count"] == 2
    assert summary["failed_bag_count"] == 0


def test_replay_audit_reports_missing_result_and_map(tmp_path):
    rows, summary = MODULE.audit_replay_variant(
        "post",
        tmp_path / "variant",
        ["ty_bag_missing"],
    )

    assert rows[0]["audit_pass"] == 0
    assert "missing_replay_result" in rows[0]["failure_reasons"]
    assert "missing_map" in rows[0]["failure_reasons"]
    assert summary["failed_bags"] == ["ty_bag_missing"]


def test_replay_audit_detects_unexpected_bag_directory(tmp_path):
    variant = tmp_path / "variant"
    write_replay_evidence(
        variant, "ty_bag_expected", spatial_passage=False
    )
    (variant / "bags" / "ty_bag_extra").mkdir()

    rows, summary = MODULE.audit_replay_variant(
        "baseline", variant, ["ty_bag_expected"]
    )

    extra = next(row for row in rows if row["bag"] == "ty_bag_extra")
    assert extra["failure_reasons"] == "unexpected_bag_directory"
    assert summary["unexpected_bags"] == ["ty_bag_extra"]
    assert summary["failed_bag_count"] == 1


@pytest.mark.parametrize(
    ("target", "expected_reason"),
    (
        ("provenance", "replay_signature_mismatch"),
        ("artifact", "artifact_manifest_mismatch"),
        ("integrity", "replay_integrity_mismatch"),
    ),
)
def test_replay_audit_detects_saved_evidence_tampering(
    tmp_path, target, expected_reason
):
    variant = tmp_path / "variant"
    bag = write_replay_evidence(
        variant, "ty_bag_one", spatial_passage=False
    )
    if target == "provenance":
        provenance = MODULE.json.loads(
            (bag / "replay_provenance.json").read_text(encoding="utf-8")
        )
        provenance["bag_path"] = "/tampered"
        (bag / "replay_provenance.json").write_text(
            MODULE.json.dumps(provenance), encoding="utf-8"
        )
    elif target == "artifact":
        (bag / "map.pgm").write_bytes(b"tampered map")
    else:
        result = MODULE.json.loads(
            (bag / "replay_result.json").read_text(encoding="utf-8")
        )
        Path(result["capture_log"]).write_text("tampered log", encoding="utf-8")

    rows, _ = MODULE.audit_replay_variant(
        "baseline", variant, ["ty_bag_one"]
    )

    assert rows[0]["audit_pass"] == 0
    assert expected_reason in rows[0]["failure_reasons"]


def test_replay_audit_checks_strict_subscription_finalize_field(tmp_path):
    variant = tmp_path / "variant"
    bag = write_replay_evidence(
        variant, "ty_bag_one", spatial_passage=False
    )
    rewrite_result_and_integrity(
        bag, subscription_finalize_pending_success=0
    )

    rows, _ = MODULE.audit_replay_variant(
        "baseline", variant, ["ty_bag_one"]
    )

    assert "pending_finalize_failed" in rows[0]["failure_reasons"]


def test_audit_results_accepts_only_intended_spatial_passage_difference(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    write_replay_evidence(
        baseline, "ty_bag_one", spatial_passage=False
    )
    write_replay_evidence(post, "ty_bag_one", spatial_passage=True)
    manifest = write_input_manifest(tmp_path / "manifest.json", ["ty_bag_one"])
    args = audit_args(tmp_path, manifest, baseline, post)

    assert MODULE.audit_results(args) == 0
    summary = MODULE.json.loads(
        (Path(args.output) / "replay_integrity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["overall_complete"] == 1
    assert summary["fairness"]["failed_bag_count"] == 0


def _direct_scan_contract(bag_name: str, spatial_passage: bool) -> dict:
    return {
        "bag": {
            "name": bag_name,
            "storage_fingerprint": [
                {
                    "name": "metadata.yaml",
                    "size_bytes": 10,
                    "sha256": "a" * 64,
                    "mtime_ns": 1,
                },
                {
                    "name": f"{bag_name}_0.db3",
                    "size_bytes": 100,
                    "sha256": "b" * 64,
                    "mtime_ns": 1,
                },
            ],
        },
        "input": {
            "scan_topic": "/scan_laserscan",
            "start_pointcloud_projection": 0,
        },
        "backend_asset_sha256": {
            "amcl_guard_header": "old-header",
            "amcl_guard_node": "old-node",
            "amcl_guard_source": "old-source",
            "imu_quality_policy_source": "old-policy",
            "laser_scan_stabilizer_source": "old-stabilizer",
            "slam_from_bag_launch": "old-unused-launch",
            "slam_launch_file": "active-launch",
            "slam_toolbox_node": "active-node",
        },
        "roughness_innovation_profile": {
            "enable_spatial_passage_hysteresis": spatial_passage,
        },
    }


def test_pair_fairness_accepts_only_proven_inactive_asset_and_mtime_changes(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    bag_name = "ty_bag_one"
    baseline_bag = write_replay_evidence(
        baseline, bag_name, spatial_passage=False
    )
    post_bag = write_replay_evidence(post, bag_name, spatial_passage=True)
    baseline_contract = _direct_scan_contract(bag_name, False)
    post_contract = _direct_scan_contract(bag_name, True)
    post_contract["bag"]["storage_fingerprint"][0]["mtime_ns"] = 2
    for key in MODULE.INACTIVE_LASERSCAN_GUARD_ASSET_FIELDS:
        post_contract["backend_asset_sha256"][key] = f"new-{key}"
    post_contract["backend_asset_sha256"]["slam_from_bag_launch"] = (
        "new-unused-launch"
    )
    launch_command = (
        "ros2 launch slam_artifact_replay.launch.py "
        "enable_laserscan_roughness_gate:=false "
        "start_pointcloud_projection:=false "
        "scan_topic:=/scan_laserscan slam_scan_topic:=/scan_laserscan"
    )
    direct_result = {
        "enable_laserscan_roughness_gate": 0,
        "start_pointcloud_projection": 0,
        "scan_topic": "/scan_laserscan",
        "slam_scan_topic": "/scan_laserscan",
    }
    rewrite_pair_fairness_evidence(
        baseline_bag,
        contract=baseline_contract,
        launch_command=launch_command,
        **direct_result,
    )
    rewrite_pair_fairness_evidence(
        post_bag,
        contract=post_contract,
        launch_command=launch_command,
        **direct_result,
    )

    status, mismatches, equivalent = MODULE._pair_fairness_status(
        baseline, post, bag_name
    )

    assert status == "matched_intended_difference"
    assert mismatches == [MODULE.ALLOWED_FAIRNESS_DIFFERENCE]
    assert "bag.storage_fingerprint[*].mtime_ns" in equivalent
    assert "backend_asset_sha256.amcl_guard_node" in equivalent
    assert "backend_asset_sha256.slam_from_bag_launch" in equivalent


def test_pair_fairness_rejects_external_asset_change_without_disabled_guard_proof(
    tmp_path,
):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    bag_name = "ty_bag_one"
    baseline_bag = write_replay_evidence(
        baseline, bag_name, spatial_passage=False
    )
    post_bag = write_replay_evidence(post, bag_name, spatial_passage=True)
    baseline_contract = _direct_scan_contract(bag_name, False)
    post_contract = _direct_scan_contract(bag_name, True)
    post_contract["backend_asset_sha256"]["amcl_guard_node"] = "changed-node"
    launch_command = (
        "ros2 launch slam_artifact_replay.launch.py "
        "enable_laserscan_roughness_gate:=true "
        "start_pointcloud_projection:=false "
        "scan_topic:=/scan_laserscan slam_scan_topic:=/scan_laserscan"
    )
    enabled_result = {
        "enable_laserscan_roughness_gate": 1,
        "start_pointcloud_projection": 0,
        "scan_topic": "/scan_laserscan",
        "slam_scan_topic": "/scan_laserscan",
    }
    rewrite_pair_fairness_evidence(
        baseline_bag,
        contract=baseline_contract,
        launch_command=launch_command,
        **enabled_result,
    )
    rewrite_pair_fairness_evidence(
        post_bag,
        contract=post_contract,
        launch_command=launch_command,
        **enabled_result,
    )

    status, mismatches, equivalent = MODULE._pair_fairness_status(
        baseline, post, bag_name
    )

    assert status == "unexpected_difference"
    assert "backend_asset_sha256.amcl_guard_node" in mismatches
    assert "backend_asset_sha256.amcl_guard_node" not in equivalent


def test_pair_fairness_rejects_storage_content_change(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    bag_name = "ty_bag_one"
    baseline_bag = write_replay_evidence(
        baseline, bag_name, spatial_passage=False
    )
    post_bag = write_replay_evidence(post, bag_name, spatial_passage=True)
    baseline_contract = _direct_scan_contract(bag_name, False)
    post_contract = _direct_scan_contract(bag_name, True)
    post_contract["bag"]["storage_fingerprint"][1]["sha256"] = "c" * 64
    launch_command = (
        "ros2 launch slam_artifact_replay.launch.py "
        "enable_laserscan_roughness_gate:=false "
        "start_pointcloud_projection:=false "
        "scan_topic:=/scan_laserscan slam_scan_topic:=/scan_laserscan"
    )
    direct_result = {
        "enable_laserscan_roughness_gate": 0,
        "start_pointcloud_projection": 0,
        "scan_topic": "/scan_laserscan",
        "slam_scan_topic": "/scan_laserscan",
    }
    rewrite_pair_fairness_evidence(
        baseline_bag,
        contract=baseline_contract,
        launch_command=launch_command,
        **direct_result,
    )
    rewrite_pair_fairness_evidence(
        post_bag,
        contract=post_contract,
        launch_command=launch_command,
        **direct_result,
    )

    status, mismatches, equivalent = MODULE._pair_fairness_status(
        baseline, post, bag_name
    )

    assert status == "unexpected_difference"
    assert "bag.storage_fingerprint" in mismatches
    assert "bag.storage_fingerprint[*].mtime_ns" not in equivalent


def test_audit_results_rejects_unexpected_fairness_difference(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    write_replay_evidence(
        baseline, "ty_bag_one", spatial_passage=False
    )
    write_replay_evidence(
        post,
        "ty_bag_one",
        spatial_passage=True,
        contract_extra={"unexpected_setting": 1},
    )
    manifest = write_input_manifest(tmp_path / "manifest.json", ["ty_bag_one"])
    args = audit_args(tmp_path, manifest, baseline, post)

    assert MODULE.audit_results(args) == 1
    rows = list(
        MODULE.csv.DictReader(
            (Path(args.output) / "replay_integrity_audit.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert all("fairness_unexpected_difference" in row["failure_reasons"] for row in rows)


def test_audit_results_rejects_empty_expected_usable_set(tmp_path):
    baseline = tmp_path / "baseline"
    post = tmp_path / "post"
    (baseline / "bags").mkdir(parents=True)
    (post / "bags").mkdir(parents=True)
    manifest = write_input_manifest(tmp_path / "manifest.json", [])
    args = audit_args(tmp_path, manifest, baseline, post)

    assert MODULE.audit_results(args) == 1
    summary = MODULE.json.loads(
        (Path(args.output) / "replay_integrity_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["empty_expected_usable_set"] == 1
    assert summary["overall_complete"] == 0
