import csv
import importlib.util
import math
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_pose_graph_rotation_constraints.py")
SPEC = importlib.util.spec_from_file_location(
    "audit_pose_graph_rotation_constraints", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def node(state, x, yaw):
    return {
        "stamp_s": str(state),
        "x_m": str(x),
        "y_m": "0",
        "yaw_rad": str(yaw),
        "source": "pose_graph_final",
        "state_id": str(state),
        "unique_id": str(state),
        "odom_x_m": str(x),
        "odom_y_m": "0",
        "odom_yaw_rad": "0",
    }


def test_normalized_angle_wraps_at_pi():
    assert math.isclose(MODULE.normalized_angle(3.0 * math.pi), math.pi)
    assert math.isclose(MODULE.normalized_angle(-3.0 * math.pi), -math.pi)


def test_percentile_interpolates_and_ignores_nan():
    assert MODULE.percentile([0.0, 10.0, math.nan], 0.5) == 5.0
    assert math.isnan(MODULE.percentile([], 0.95))


def test_edge_classification_is_evidence_limited():
    near_pairs = {(2, 8)}
    adjacent = {"source_state_id": "2", "target_state_id": "3"}
    near = {"source_state_id": "8", "target_state_id": "2"}
    unknown = {"source_state_id": "2", "target_state_id": "9"}
    assert MODULE.classify_edge(adjacent, near_pairs) == "adjacent"
    assert MODULE.classify_edge(near, near_pairs) == "near_chain"
    assert MODULE.classify_edge(unknown, near_pairs) == "unobserved_nonadjacent"


def test_analyze_dataset_detects_rotated_near_chain_edge(tmp_path):
    bag_dir = tmp_path / "variant" / "bags" / "bag_a"
    nodes = [node(0, 0.0, 0.0), node(1, 0.1, 0.0), node(2, 0.0, math.pi / 2)]
    write_csv(bag_dir / "trajectory_pose_graph_final.csv", nodes)
    edges = [
        {
            "source_state_id": "0",
            "target_state_id": "1",
            "relative_yaw_rad": "0",
            "covariance_yawyaw": "0.1",
            "final_residual_translation_m": "0",
            "final_residual_yaw_rad": "0",
        },
        {
            "source_state_id": "0",
            "target_state_id": "2",
            "relative_yaw_rad": str(math.pi / 2),
            "covariance_yawyaw": "0.1",
            "final_residual_translation_m": "0",
            "final_residual_yaw_rad": "0",
        },
    ]
    write_csv(bag_dir / "pose_graph_edges_final.csv", edges)
    candidates = [
        {
            "source_state_id": "0",
            "target_state_id": "2",
            "edge_present": "true",
            "prelinked": "false",
            "edge_inserted": "true",
            "translation_disagreement_m": "1.0",
            "yaw_disagreement_rad": str(math.pi / 4),
            "relative_yaw_rad": str(math.pi / 2),
            "normalized_response": "0.2",
            "chain_size": "8",
            "stamp_ns": "2",
            "xy_covariance_max_eigenvalue_m2": "0.1",
            "covariance_yawyaw": "0.01",
            "scan_match_covariance_weighted_disagreement_sq": "100",
            "memory_strength": "0",
            "backend_action": "pose_fused_edge_inserted",
        }
    ]
    candidate_path = tmp_path / "candidate.csv"
    write_csv(candidate_path, candidates)

    metrics, edge_rows, candidate_rows = MODULE.analyze_dataset(
        bag_dir,
        candidate_path,
        revisit_radius_m=0.5,
        minimum_state_separation=2,
        minimum_separation_fraction=0.0,
    )

    assert metrics.near_chain_edge_count == 1
    assert metrics.nonadjacent_yaw_disagreement_gt_80_deg == 1
    assert metrics.candidate_large_prediction_disagreement_count == 1
    assert edge_rows[1]["edge_class"] == "near_chain"
    assert math.isclose(candidate_rows[0]["odom_yaw_disagreement_deg"], 90.0)
