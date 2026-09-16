"""Read the installed node's parameters, including C++ defaults."""

from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

import yaml


def effective_parameters(overrides):
    """Return isolated copies of the cached runtime parameter values."""
    return deepcopy(_probe(json.dumps(overrides, sort_keys=True)))


@lru_cache(maxsize=64)
def _probe(encoded):
    params = json.loads(encoded)
    name = "parameter_contract_" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory(prefix=name) as directory:
        root = Path(directory)
        config = root / "params.yaml"
        config.write_text(yaml.safe_dump({name: {"ros__parameters": params}}))
        environment = os.environ.copy()
        environment["ROS_LOCALHOST_ONLY"] = "1"
        environment["ROS_LOG_DIR"] = str(root / "ros_logs")
        command = [
            "ros2", "run", "slam_3d_core", "scan_to_map_icp_node",
            "--ros-args",
            "-r", f"__node:={name}", "--params-file", str(config),
        ]
        # Prevent a metadata-only probe from consuming another test's sensors.
        topics = {
            value for key, value in params.items()
            if key.endswith("_topic") and isinstance(value, str) and value
        }
        topics.update({
            "/tf", "/tf_static", "/imu/data", "/motor_driver_node/odom",
            "/slam_3d/input/hesai_center/points_timed",
            "/slam_3d/input/hesai_front/points_timed",
            "/slam_3d/input/hesai_rear/points_timed",
        })
        remaps = {f"/{name}/topic_{index}": topic
                  for index, topic in enumerate(sorted(topics))}
        for isolated, topic in remaps.items():
            command.extend(["-r", f"{topic}:={isolated}"])
        log_path = root / "node.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                env=environment, start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 12.0
                while True:
                    output = log_path.read_text()
                    if "scan_to_map_icp_node ready:" in output:
                        break
                    if (process.poll() is not None or
                            time.monotonic() > deadline):
                        raise RuntimeError(
                            f"Parameter probe failed:\n{output}")
                    time.sleep(0.05)
                result = subprocess.run(
                    ["ros2", "param", "dump", "--no-daemon", f"/{name}"],
                    env=environment, capture_output=True, text=True,
                    timeout=15,
                    check=True,
                )
                document = yaml.safe_load(result.stdout)
                effective = document[f"/{name}"]["ros__parameters"]
                # ROS creates QoS parameter keys from the remapped topic names.
                if "qos_overrides" in effective:
                    effective["qos_overrides"] = {
                        remaps.get(topic, topic): value
                        for topic, value in effective["qos_overrides"].items()
                    }
                return effective
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
