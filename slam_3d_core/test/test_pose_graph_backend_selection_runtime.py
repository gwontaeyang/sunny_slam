import os
import re
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
import yaml


SOLVERS = ("legacy_relaxation", "ceres_se3", "g2o_se3", "gtsam_se3")
SELECTION = re.compile(
    r"Pose graph solver selected: ([a-z0-9_]+); compiled solvers: ([a-z0-9_,]+)"
)


def _probe_startup(solver, **overrides):
    token = uuid.uuid4().hex[:8]
    prefix = f"/pose_graph_selection_test/run_{token}"
    params = {
        "pose_graph_solver": solver,
        "pose_graph_correction_enabled": False,
        "pose_graph_planar_correction_enabled": False,
        "use_secondary_cloud": False,
        "use_tertiary_cloud": False,
        "use_wheel_prediction": False,
        "use_tf_transforms": False,
        "publish_tf": False,
        "deskew_enabled": False,
        "primary_cloud_topic": f"{prefix}/cloud",
        "wheel_odom_topic": f"{prefix}/wheel",
        "imu_topic": f"{prefix}/imu",
        "imu_watermark_topic": f"{prefix}/imu_watermark",
        **overrides,
    }
    command = [
        "ros2", "run", "slam_3d_core", "scan_to_map_icp_node", "--ros-args",
        "-r", f"__node:=pose_graph_selection_{token}",
        "-r", f"__ns:={prefix}",
    ]
    for name, value in params.items():
        encoded = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(["-p", f"{name}:={encoded}"])

    with tempfile.TemporaryDirectory(prefix="pose_graph_selection_") as temp_dir:
        output_path = Path(temp_dir) / "node.log"
        environment = os.environ.copy()
        environment["ROS_LOCALHOST_ONLY"] = "1"
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        with output_path.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, env=environment,
            )
            try:
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    log = output_path.read_text(encoding="utf-8")
                    returncode = process.poll()
                    if returncode is not None:
                        return returncode, output_path.read_text(encoding="utf-8")
                    if "scan_to_map_icp_node ready:" in log:
                        return None, log
                    time.sleep(0.05)
                pytest.fail(f"node startup timed out for {solver}:\n{log}")
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5.0)


@pytest.fixture(scope="module")
def compiled_solvers():
    returncode, log = _probe_startup("legacy_relaxation")
    assert returncode is None, log
    match = SELECTION.search(log)
    assert match is not None, log
    assert match.group(1) == "legacy_relaxation"
    available = match.group(2).split(",")
    assert len(available) == len(set(available)), log
    assert {"legacy_relaxation", "ceres_se3"} <= set(available), log
    assert set(available) <= set(SOLVERS), log
    return set(available)


@pytest.mark.parametrize("solver", SOLVERS)
def test_selected_backend_starts_or_fails_explicitly_if_not_compiled(solver, compiled_solvers):
    returncode, log = _probe_startup(solver)
    if solver in compiled_solvers:
        assert returncode is None, log
        match = SELECTION.search(log)
        assert match is not None, log
        assert match.group(1) == solver, log
        assert set(match.group(2).split(",")) == compiled_solvers, log
    else:
        assert returncode not in (None, 0), log
        assert f"pose_graph_solver='{solver}' is not compiled into this build" in log
        assert "compiled solvers:" in log
        assert "No fallback is performed." in log
        assert "scan_to_map_icp_node ready:" not in log


@pytest.mark.parametrize("solver", ["ceres", "GTSAM", "isam2", "unknown_solver"])
def test_invalid_backend_name_is_rejected(solver):
    returncode, log = _probe_startup(solver)
    assert returncode not in (None, 0), log
    assert f"Invalid pose_graph_solver='{solver}'" in log
    assert all(name in log for name in SOLVERS), log
    assert "scan_to_map_icp_node ready:" not in log


@pytest.mark.parametrize("solver", SOLVERS[1:])
def test_cpu_minimal_overlay_starts_with_each_compiled_backend(solver, compiled_solvers):
    if solver not in compiled_solvers:
        pytest.skip(f"{solver} is not compiled; covered by unavailable selection test")
    path = Path(__file__).resolve().parents[1] / "config/indoor_cpu_minimal_overrides.yaml"
    overrides = yaml.safe_load(path.read_text())["scan_to_map_icp_node"]["ros__parameters"]
    returncode, log = _probe_startup(solver, **overrides)
    assert returncode is None, log
    match = SELECTION.search(log)
    assert match is not None and match.group(1) == solver, log


@pytest.mark.parametrize("solver", SOLVERS[1:])
def test_nonlinear_backend_rejects_planar_correction(solver, compiled_solvers):
    if solver not in compiled_solvers:
        pytest.skip(f"{solver} is not compiled; covered by unavailable selection test")
    returncode, log = _probe_startup(solver, pose_graph_planar_correction_enabled=True)
    assert returncode not in (None, 0), log
    assert f"pose_graph_solver={solver} requires full 3D pose graph correction" in log
    assert "scan_to_map_icp_node ready:" not in log


@pytest.mark.parametrize("solver", SOLVERS)
def test_full_information_requires_a_compiled_nonlinear_backend(solver, compiled_solvers):
    if solver not in compiled_solvers:
        pytest.skip(f"{solver} is not compiled; covered by unavailable selection test")
    returncode, log = _probe_startup(
        solver, pose_graph_information_mode="registration_observability"
    )
    if solver == "legacy_relaxation":
        assert returncode not in (None, 0), log
        assert "registration_observability requires" in log
        assert "scan_to_map_icp_node ready:" not in log
    else:
        assert returncode is None, log
        match = SELECTION.search(log)
        assert match is not None, log
        assert match.group(1) == solver, log
