import base64
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("aggregate_slam_2d_regression.py")
SPEC = importlib.util.spec_from_file_location("aggregate_slam_2d_regression", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def valid_replay(**overrides):
    payload = {
        "status": "map_and_map_trajectory",
        "runtime_slam_parameter_match": 1,
        "slam_input_complete": 1,
        "slam_recorded_scan_delivery_complete": 1,
        "roughness_imu_delivery_complete": 1,
        "roughness_imu_delivery_fingerprint_match": 1,
        "queue_drain_complete": 1,
        "pose_graph_captured": 1,
        "pose_graph_coverage_complete": 1,
        "stationary_replay": 0,
        "pose_graph_trajectory_count": 25,
        "map_width": 100,
        "map_height": 40,
        "map_resolution_m": 0.05,
        "map_known_ratio": 0.2,
        "map_occupied_ratio": 0.03,
        "roughness_near_chain_spatial_passage_count": 4,
        "roughness_near_chain_recovery_count": 3,
    }
    payload.update(overrides)
    return payload


def write_replay(
    root: Path,
    bag: str,
    *,
    payload=None,
    result: bool = True,
    map_present: bool = True,
) -> Path:
    bag_dir = root / "bags" / bag
    bag_dir.mkdir(parents=True)
    if result:
        (bag_dir / "replay_result.json").write_text(
            json.dumps(payload or valid_replay()), encoding="utf-8"
        )
    if map_present:
        (bag_dir / "map.png").write_bytes(PNG)
    return bag_dir


def write_pairs(comparison_root: Path, bag: str, rows, *, fieldnames=None) -> Path:
    bag_dir = comparison_root / bag
    bag_dir.mkdir(parents=True, exist_ok=True)
    path = bag_dir / MODULE.DEFAULT_REVISIT_CSV
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames
            or (
                "baseline_abs_drift_m",
                "post_abs_drift_m",
                "abs_drift_improvement_m",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return bag_dir


def write_manifest(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "name",
                "source_root",
                "status",
                "storage_integrity_status",
                "duration_s",
                "scan_laserscan_count",
                "imu_count",
                "wheel_odom_count",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def collect(tmp_path: Path, **kwargs):
    return MODULE.collect_regression(
        baseline_root=tmp_path / "baseline",
        post_root=tmp_path / "post",
        comparison_root=tmp_path / "comparisons",
        **kwargs,
    )


def test_complete_bag_aggregates_proxy_without_accuracy_claim(tmp_path):
    bag = "ty_bag_complete"
    write_replay(tmp_path / "baseline", bag)
    write_replay(
        tmp_path / "post",
        bag,
        payload=valid_replay(
            map_width=80,
            map_height=50,
            roughness_near_chain_spatial_passage_count=9,
        ),
    )
    comparison = write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 1.0,
                "post_abs_drift_m": 0.5,
                "abs_drift_improvement_m": 0.5,
            },
            {
                "baseline_abs_drift_m": 0.4,
                "post_abs_drift_m": 0.6,
                "abs_drift_improvement_m": -0.2,
            },
        ],
    )
    (comparison / "map_comparison_side_by_side.png").write_bytes(PNG)

    aggregation = collect(tmp_path)
    row = aggregation["bags"][0]

    assert row["comparison_state"] == "complete_with_return_proxy"
    assert row["proxy_available"] == 1
    assert row["return_pair_count"] == 2
    assert row["baseline_return_abs_drift_mean_m"] == pytest.approx(0.7)
    assert row["post_return_abs_drift_mean_m"] == pytest.approx(0.55)
    assert row["return_abs_drift_improvement_mean_m"] == pytest.approx(0.15)
    assert row["improved_pair_count"] == 1
    assert row["regressed_pair_count"] == 1
    assert row["bag_outcome_by_mean_proxy"] == "improved"
    assert row["map_area_delta_m2"] == pytest.approx(0.0)
    assert row["comparison_map_artifact_trusted"] == 1
    contract = aggregation["summary"]["metric_contract"]
    assert contract["ground_truth_available"] is False
    assert contract["absolute_accuracy_claim_supported"] is False
    assert contract["trajectory_rendering_policy"] == "points_only_no_connected_segments"
    assert contract["trajectory_plots_generated"] is False


@pytest.mark.parametrize("value", [1, 1.0, "1", "1.0", "1e0"])
def test_replay_success_flags_accept_only_numeric_one_representations(tmp_path, value):
    bag = "ty_bag_exact_one"
    write_replay(
        tmp_path / "baseline",
        bag,
        payload=valid_replay(runtime_slam_parameter_match=value),
    )

    replay = MODULE.inspect_replay(tmp_path / "baseline", bag)

    assert replay["replay_state"] == "complete"


@pytest.mark.parametrize(
    "value",
    [
        False,
        True,
        0,
        -1,
        2,
        0.5,
        "0",
        "-1",
        "2",
        "true",
        "yes",
        "passed",
        "complete",
    ],
)
def test_replay_success_flags_reject_non_one_truthy_values(tmp_path, value):
    bag = "ty_bag_not_one"
    write_replay(
        tmp_path / "baseline",
        bag,
        payload=valid_replay(runtime_slam_parameter_match=value),
    )

    replay = MODULE.inspect_replay(tmp_path / "baseline", bag)

    assert replay["replay_state"] == "failed_integrity"
    assert "runtime_slam_parameter_match_not_exactly_one" in replay["failure_reasons"]


def test_manifest_keeps_unusable_and_incomplete_bags_explicit(tmp_path):
    manifest = tmp_path / "input_bag_manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "name": "ty_bag_interrupted",
                "source_root": "01_input",
                "status": "ready",
                "storage_integrity_status": "passed",
                "duration_s": 10,
                "scan_laserscan_count": 100,
                "imu_count": 800,
                "wheel_odom_count": 200,
            },
            {
                "name": "ty_bag_empty",
                "source_root": "03_input",
                "status": "empty_storage",
                "storage_integrity_status": "not_checked",
                "duration_s": "nan",
                "scan_laserscan_count": 0,
                "imu_count": 0,
                "wheel_odom_count": 0,
            },
        ],
    )
    write_replay(tmp_path / "baseline", "ty_bag_interrupted")
    interrupted = write_replay(
        tmp_path / "post", "ty_bag_interrupted", result=False, map_present=False
    )
    (interrupted / "commands.txt").write_text("partial run", encoding="utf-8")
    comparison = write_pairs(
        tmp_path / "comparisons",
        "ty_bag_interrupted",
        [
            {
                "baseline_abs_drift_m": 1.0,
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": 0.9,
            }
        ],
    )
    (comparison / "map_comparison_side_by_side.png").write_bytes(PNG)

    aggregation = collect(tmp_path, input_manifest=manifest)
    rows = {row["bag"]: row for row in aggregation["bags"]}

    interrupted_row = rows["ty_bag_interrupted"]
    assert interrupted_row["comparison_state"] == "post_incomplete"
    assert interrupted_row["post_replay_state"] == "missing_replay_result"
    assert interrupted_row["proxy_available"] == 0
    assert interrupted_row["return_pair_count"] == 0
    assert (
        interrupted_row["comparison_csv_error"]
        == "not_evaluated_incomplete_replay"
    )
    assert interrupted_row["comparison_map_artifact_present"] == 1
    assert interrupted_row["comparison_map_artifact_trusted"] == 0
    assert "ty_bag_interrupted" in aggregation["summary"]["incomplete_usable_bags"]

    empty_row = rows["ty_bag_empty"]
    assert empty_row["comparison_state"] == "input_unusable"
    assert empty_row["input_usable"] == 0
    assert empty_row["input_duration_s"] is None
    assert aggregation["summary"]["counts"]["all_bag_count"] == 2


def test_authoritative_manifest_reports_but_excludes_unexpected_artifacts(tmp_path):
    manifest = tmp_path / "input_bag_manifest.csv"
    write_manifest(
        manifest,
        [
            {
                "name": "ty_bag_expected",
                "source_root": "01_input",
                "status": "ready",
                "storage_integrity_status": "passed",
                "duration_s": 10,
                "scan_laserscan_count": 100,
                "imu_count": 800,
                "wheel_odom_count": 200,
            }
        ],
    )
    for bag, baseline_drift, post_drift in (
        ("ty_bag_expected", 0.4, 0.2),
        ("ty_bag_unexpected", 10.0, 0.0),
    ):
        write_replay(tmp_path / "baseline", bag)
        write_replay(tmp_path / "post", bag)
        write_pairs(
            tmp_path / "comparisons",
            bag,
            [
                {
                    "baseline_abs_drift_m": baseline_drift,
                    "post_abs_drift_m": post_drift,
                    "abs_drift_improvement_m": baseline_drift - post_drift,
                }
            ],
        )

    aggregation = collect(tmp_path, input_manifest=manifest)
    rows = {row["bag"]: row for row in aggregation["bags"]}
    unexpected = rows["ty_bag_unexpected"]

    assert unexpected["comparison_state"] == "unexpected_artifact"
    assert unexpected["manifest_member"] == 0
    assert unexpected["unexpected_artifact"] == 1
    assert unexpected["proxy_available"] == 0
    assert unexpected["comparison_csv_trust_status"] == "not_evaluated"
    assert set(unexpected["unexpected_artifact_locations"].split(",")) == {
        "baseline",
        "post",
        "comparison",
    }
    counts = aggregation["summary"]["counts"]
    assert counts["all_bag_count"] == 1
    assert counts["reported_row_count"] == 2
    assert counts["unexpected_artifact_bag_count"] == 1
    assert counts["return_proxy_bag_count"] == 1
    assert counts["pooled_return_pair_count"] == 1
    assert aggregation["summary"]["unexpected_artifacts"] == [
        {
            "bag": "ty_bag_unexpected",
            "locations": ["baseline", "post", "comparison"],
        }
    ]
    macro = aggregation["summary"]["aggregate_return_consistency_proxy"][
        "macro_equal_bag_weight"
    ]
    assert macro["bag_count"] == 1
    assert macro["mean_of_bag_return_abs_drift_improvement_mean_m"] == pytest.approx(
        0.2
    )


def test_proxy_loader_skips_invalid_rows_and_recomputes_improvement(tmp_path):
    bag = "ty_bag_malformed_pairs"
    write_replay(tmp_path / "baseline", bag)
    write_replay(tmp_path / "post", bag)
    write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 0.5,
                "post_abs_drift_m": 0.2,
                "abs_drift_improvement_m": 99.0,
            },
            {
                "baseline_abs_drift_m": "nan",
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": 0.0,
            },
            {
                "baseline_abs_drift_m": -1.0,
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": -1.1,
            },
        ],
    )

    row = collect(tmp_path)["bags"][0]

    assert row["return_pair_count"] == 1
    assert row["return_pair_skipped_row_count"] == 2
    assert row["provided_improvement_mismatch_count"] == 1
    assert row["return_abs_drift_improvement_mean_m"] == pytest.approx(0.3)


def test_stale_comparison_csv_and_map_are_exposed_and_excluded(tmp_path):
    bag = "ty_bag_stale_comparison"
    baseline_dir = write_replay(tmp_path / "baseline", bag)
    post_dir = write_replay(tmp_path / "post", bag)
    (baseline_dir / "trajectory_pose_graph_final.csv").write_text(
        "stamp_s,x_m,y_m\n0,0,0\n", encoding="utf-8"
    )
    (post_dir / "trajectory_pose_graph_final.csv").write_text(
        "stamp_s,x_m,y_m\n0,0,0\n", encoding="utf-8"
    )
    comparison = write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 1.0,
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": 0.9,
            }
        ],
    )
    map_comparison = comparison / "map_comparison_side_by_side.png"
    map_comparison.write_bytes(PNG)
    comparison_mtime_ns = max(
        (comparison / MODULE.DEFAULT_REVISIT_CSV).stat().st_mtime_ns,
        map_comparison.stat().st_mtime_ns,
    )
    newer_mtime_ns = comparison_mtime_ns + 10_000_000_000
    for source in (
        post_dir / "replay_result.json",
        post_dir / "map.png",
        post_dir / "trajectory_pose_graph_final.csv",
    ):
        os.utime(source, ns=(newer_mtime_ns, newer_mtime_ns))

    aggregation = collect(tmp_path)
    row = aggregation["bags"][0]

    assert row["comparison_state"] == "complete_replays_untrusted_comparison"
    assert row["comparison_csv_present"] == 1
    assert row["comparison_csv_trusted"] == 0
    assert row["comparison_csv_trust_status"] == "stale"
    assert "post/bags/ty_bag_stale_comparison/replay_result.json" in row[
        "comparison_csv_trust_reasons"
    ]
    assert row["comparison_map_artifact_present"] == 1
    assert row["comparison_map_artifact_trusted"] == 0
    assert row["comparison_map_artifact_trust_status"] == "stale"
    assert row["proxy_available"] == 0
    assert aggregation["summary"]["counts"]["pooled_return_pair_count"] == 0
    assert aggregation["summary"]["untrusted_comparison_csv_bags"][0][
        "status"
    ] == "stale"
    assert aggregation["summary"]["untrusted_comparison_map_bags"][0][
        "status"
    ] == "stale"
    MODULE.write_outputs(
        aggregation, tmp_path / "stale_report", create_plots=False
    )
    report = (tmp_path / "stale_report/aggregate_report.md").read_text(
        encoding="utf-8"
    )
    assert "Untrusted CSV `ty_bag_stale_comparison`: status=stale" in report
    assert "Untrusted map `ty_bag_stale_comparison`: status=stale" in report


def test_pair_identity_duplicates_and_invalid_rows_invalidate_comparison(tmp_path):
    bag = "ty_bag_duplicate_pairs"
    write_replay(tmp_path / "baseline", bag)
    write_replay(tmp_path / "post", bag)
    identity_fields = (
        "outbound_stamp_s",
        "return_stamp_s",
        "candidate_index",
        "baseline_abs_drift_m",
        "post_abs_drift_m",
        "abs_drift_improvement_m",
    )
    write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "outbound_stamp_s": 1.0,
                "return_stamp_s": 5.0,
                "candidate_index": 7,
                "baseline_abs_drift_m": 0.5,
                "post_abs_drift_m": 0.2,
                "abs_drift_improvement_m": 0.3,
            },
            {
                "outbound_stamp_s": "1.0",
                "return_stamp_s": "5.0",
                "candidate_index": "7.0",
                "baseline_abs_drift_m": 0.6,
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": 0.5,
            },
            {
                "outbound_stamp_s": "",
                "return_stamp_s": 6.0,
                "candidate_index": 8,
                "baseline_abs_drift_m": 0.6,
                "post_abs_drift_m": 0.1,
                "abs_drift_improvement_m": 0.5,
            },
        ],
        fieldnames=identity_fields,
    )

    row = collect(tmp_path)["bags"][0]

    assert row["pair_identity_columns"] == (
        "outbound_stamp_s,return_stamp_s,candidate_index"
    )
    assert row["pair_identity_validation_status"] == "invalid"
    assert row["pair_identity_duplicate_row_count"] == 1
    assert row["pair_identity_invalid_row_count"] == 1
    assert row["comparison_csv_trusted"] == 0
    assert row["comparison_csv_trust_status"] == "invalid_pair_identity"
    assert "duplicate_pair_identity:1" in row["comparison_csv_error"]
    assert row["proxy_available"] == 0


def test_complete_replays_without_pairs_are_not_reported_as_comparable(tmp_path):
    bag = "ty_bag_stationary"
    stationary = valid_replay(
        status="map_and_stationary_pose_graph",
        stationary_replay=1,
        pose_graph_trajectory_count=1,
    )
    write_replay(tmp_path / "baseline", bag, payload=stationary)
    write_replay(tmp_path / "post", bag, payload=stationary)
    write_pairs(tmp_path / "comparisons", bag, [])

    aggregation = collect(tmp_path)
    row = aggregation["bags"][0]

    assert row["replay_pair_complete"] == 1
    assert row["comparison_state"] == "complete_replays_no_return_proxy"
    assert row["proxy_available"] == 0
    assert row["comparison_csv_error"] == "no_valid_proxy_rows"
    assert aggregation["summary"]["complete_replays_without_return_proxy"] == [bag]


def test_legacy_blank_pair_csv_is_accepted_only_for_stationary_replays(tmp_path):
    bag = "ty_bag_stationary_blank"
    stationary = valid_replay(
        status="map_and_stationary_pose_graph",
        stationary_replay=1,
        pose_graph_trajectory_count=1,
    )
    write_replay(tmp_path / "baseline", bag, payload=stationary)
    write_replay(tmp_path / "post", bag, payload=stationary)
    comparison_dir = write_pairs(tmp_path / "comparisons", bag, [])
    (comparison_dir / MODULE.DEFAULT_REVISIT_CSV).write_text(
        "\r\n", encoding="utf-8"
    )

    row = collect(tmp_path)["bags"][0]

    assert row["comparison_state"] == "complete_replays_no_return_proxy"
    assert row["comparison_csv_trusted"] == 1
    assert row["comparison_csv_error"] == "no_valid_proxy_rows"


def test_legacy_blank_pair_csv_is_rejected_for_dynamic_replays(tmp_path):
    bag = "ty_bag_dynamic_blank"
    write_replay(tmp_path / "baseline", bag)
    write_replay(tmp_path / "post", bag)
    comparison_dir = write_pairs(tmp_path / "comparisons", bag, [])
    (comparison_dir / MODULE.DEFAULT_REVISIT_CSV).write_text(
        "\r\n", encoding="utf-8"
    )

    row = collect(tmp_path)["bags"][0]

    assert row["comparison_state"] == "complete_replays_untrusted_comparison"
    assert row["comparison_csv_trusted"] == 0
    assert row["comparison_csv_error"] == "empty_csv_without_header"


def test_failed_replay_integrity_blocks_stale_proxy(tmp_path):
    bag = "ty_bag_bad_delivery"
    write_replay(tmp_path / "baseline", bag)
    write_replay(
        tmp_path / "post",
        bag,
        payload=valid_replay(roughness_imu_delivery_fingerprint_match=0),
    )
    write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 1.0,
                "post_abs_drift_m": 0.0,
                "abs_drift_improvement_m": 1.0,
            }
        ],
    )

    row = collect(tmp_path)["bags"][0]

    assert row["comparison_state"] == "post_incomplete"
    assert row["post_replay_state"] == "failed_integrity"
    assert "roughness_imu_delivery_fingerprint_match_not_exactly_one" in row[
        "post_failure_reasons"
    ]
    assert row["proxy_available"] == 0


def test_writer_creates_strict_csv_json_and_paper_review_plots(tmp_path):
    bag = "ty_bag_plot"
    write_replay(tmp_path / "baseline", bag)
    write_replay(tmp_path / "post", bag)
    comparison = write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 0.4,
                "post_abs_drift_m": 0.2,
                "abs_drift_improvement_m": 0.2,
            }
        ],
    )
    (comparison / "map_comparison_side_by_side.png").write_bytes(PNG)
    aggregation = collect(tmp_path)

    outputs = MODULE.write_outputs(
        aggregation,
        tmp_path / "paper_review",
        create_plots=True,
        contact_sheet_page_size=1,
    )

    names = {path.name for path in outputs}
    assert "all_bags_summary.csv" in names
    assert "all_bags_summary.json" in names
    assert "tie_band_sweep.csv" in names
    assert "aggregate_report.md" in names
    assert "01_return_consistency_proxy_by_bag.png" in names
    assert "02_return_consistency_proxy_distribution.png" in names
    assert "03_replay_comparison_coverage.png" in names
    assert "04_map_comparison_contact_sheet_01.png" in names
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
    json_text = (tmp_path / "paper_review/all_bags_summary.json").read_text(
        encoding="utf-8"
    )
    assert "NaN" not in json_text
    payload = json.loads(json_text)
    assert payload["metric_contract"]["absolute_accuracy_claim_supported"] is False
    assert payload["tie_band_sweep"]
    tie_rows = list(
        csv.DictReader(
            (tmp_path / "paper_review/tie_band_sweep.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(tie_rows) == len(payload["tie_band_sweep"])
    assert all(row["absolute_accuracy_claim_supported"] == "False" for row in tie_rows)
    report = (tmp_path / "paper_review/aggregate_report.md").read_text(
        encoding="utf-8"
    )
    assert "not ground-truth" in report
    assert "points_only_no_connected_segments" in report


def test_configurable_tie_band_sweep_preserves_raw_proxy_metrics(tmp_path):
    bag = "ty_bag_tie_sweep"
    write_replay(tmp_path / "baseline", bag)
    write_replay(tmp_path / "post", bag)
    write_pairs(
        tmp_path / "comparisons",
        bag,
        [
            {
                "baseline_abs_drift_m": 0.50,
                "post_abs_drift_m": 0.48,
                "abs_drift_improvement_m": 0.02,
            },
            {
                "baseline_abs_drift_m": 0.50,
                "post_abs_drift_m": 0.53,
                "abs_drift_improvement_m": -0.03,
            },
        ],
    )

    aggregation = collect(
        tmp_path,
        tie_epsilon_m=0.0,
        tie_bands_m=[0.01, 0.025, 0.05],
    )
    row = aggregation["bags"][0]
    sweep = {
        entry["tie_band_m"]: entry for entry in aggregation["summary"]["tie_band_sweep"]
    }

    assert row["return_abs_drift_improvement_mean_m"] == pytest.approx(-0.005)
    assert sweep[0.01]["improved_pair_count"] == 1
    assert sweep[0.01]["regressed_pair_count"] == 1
    assert sweep[0.025]["improved_pair_count"] == 0
    assert sweep[0.025]["tied_pair_count"] == 1
    assert sweep[0.025]["regressed_pair_count"] == 1
    assert sweep[0.05]["tied_pair_count"] == 2
    assert all(
        entry["absolute_accuracy_claim_supported"] is False
        for entry in sweep.values()
    )


def test_percentile_rejects_invalid_quantile():
    assert MODULE.percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    assert MODULE.percentile([], 0.5) is None
    with pytest.raises(ValueError):
        MODULE.percentile([1.0], 1.1)
