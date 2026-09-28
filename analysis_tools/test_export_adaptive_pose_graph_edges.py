import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("export_adaptive_pose_graph_edges.py")
SPEC = importlib.util.spec_from_file_location(
    "export_adaptive_pose_graph_edges", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


FIELDS = (
    "stamp_s",
    "x_m",
    "y_m",
    "yaw_rad",
    "source",
    "state_id",
    "unique_id",
    "odom_x_m",
    "odom_y_m",
    "odom_yaw_rad",
)


def write_trajectory(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def trajectory(unique_id="10", x="1.0"):
    return {
        "stamp_s": "2.0",
        "x_m": x,
        "y_m": "3.0",
        "yaw_rad": "0.2",
        "source": "pose_graph_final",
        "state_id": "4",
        "unique_id": unique_id,
        "odom_x_m": "0.9",
        "odom_y_m": "2.9",
        "odom_yaw_rad": "0.1",
    }


def test_equivalent_trajectory_accepts_numeric_format_difference(tmp_path):
    existing = tmp_path / "existing.csv"
    exported = tmp_path / "exported.csv"
    write_trajectory(existing, [trajectory(x="1")])
    write_trajectory(exported, [trajectory(x="1.0000000000000")])

    assert MODULE.trajectories_equivalent(existing, exported) == (True, 1)


def test_equivalent_trajectory_rejects_pose_or_id_change(tmp_path):
    existing = tmp_path / "existing.csv"
    exported = tmp_path / "exported.csv"
    write_trajectory(existing, [trajectory()])
    write_trajectory(exported, [trajectory(unique_id="11")])
    assert MODULE.trajectories_equivalent(existing, exported) == (False, 1)

    write_trajectory(exported, [trajectory(x="1.01")])
    assert MODULE.trajectories_equivalent(existing, exported) == (False, 1)


def test_selected_replays_filters_labels(tmp_path):
    for label in ("control", "adaptive"):
        path = (
            tmp_path
            / "stage34_roughness_gate_replay"
            / "variant"
            / label
            / "bags"
            / "bag_a"
            / "replay_result.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")

    selected = MODULE.selected_replays(tmp_path, ["control"])
    assert len(selected) == 1
    assert selected[0].parents[2].name == "control"


def test_export_one_writes_schema2_sidecar_and_rechecks_trajectory(tmp_path):
    bag_dir = tmp_path / "variant" / "control" / "bags" / "A"
    bag_dir.mkdir(parents=True)
    replay = bag_dir / "replay_result.json"
    replay.write_text("{}", encoding="utf-8")
    (bag_dir / "final_pose_graph.posegraph").write_bytes(b"posegraph")
    existing = bag_dir / "trajectory_pose_graph_final.csv"
    write_trajectory(existing, [trajectory()])

    edge_fields = sorted(MODULE.REQUIRED_EDGE_FIELDS)
    exporter = tmp_path / "fake_exporter.py"
    exporter.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib,sys\n"
        f"pathlib.Path(sys.argv[2]).write_text({(','.join(FIELDS) + chr(10) + ','.join(trajectory()[name] for name in FIELDS) + chr(10))!r})\n"
        f"pathlib.Path(sys.argv[3]).write_text({(','.join(edge_fields) + chr(10))!r})\n",
        encoding="utf-8",
    )
    exporter.chmod(0o755)
    artifact_root = tmp_path / "artifacts"

    first = MODULE.export_one(replay, exporter, artifact_root, False, 5.0)
    assert first["status"] == "exported"
    sidecar = json.loads(
        (bag_dir / "pose_graph_edges_final.provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["schema_version"] == 2
    assert sidecar["trajectory_sha256"] == MODULE.file_sha256(existing)

    changed = trajectory(x="9.0")
    write_trajectory(existing, [changed])
    second = MODULE.export_one(replay, exporter, artifact_root, False, 5.0)
    assert second["status"] == "trajectory_mismatch"
