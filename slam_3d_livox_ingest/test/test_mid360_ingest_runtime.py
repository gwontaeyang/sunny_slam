import os
import signal
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from livox_ros_driver2.msg import CustomMsg, CustomPoint
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2


def _time_msg(nanoseconds):
    from builtin_interfaces.msg import Time

    return Time(sec=nanoseconds // 1_000_000_000, nanosec=nanoseconds % 1_000_000_000)


def _point(offset_time, x, y, z, reflectivity, tag, line):
    point = CustomPoint()
    point.offset_time = offset_time
    point.x = x
    point.y = y
    point.z = z
    point.reflectivity = reflectivity
    point.tag = tag
    point.line = line
    return point


def _message(timebase, lidar_id, points, declared_count=None):
    message = CustomMsg()
    message.header.stamp = _time_msg(timebase)
    message.header.frame_id = "driver_frame"
    message.timebase = timebase
    message.point_num = len(points) if declared_count is None else declared_count
    message.lidar_id = lidar_id
    message.rsvd = [0, 0, 0]
    message.points = points
    return message


def _status_values(message):
    return {item.key: item.value for item in message.status[0].values}


def _field_map(cloud):
    return {field.name: (field.offset, field.datatype) for field in cloud.fields}


def test_dual_custom_messages_are_validated_and_preserved():
    token = uuid.uuid4().hex[:8]
    prefix = f"/mid360_ingest_test/run_{token}"
    topics = {
        "front_input": f"{prefix}/front/custom",
        "rear_input": f"{prefix}/rear/custom",
        "front_output": f"{prefix}/front/timed",
        "rear_output": f"{prefix}/rear/timed",
        "health": f"{prefix}/health",
    }
    command = [
        "ros2",
        "run",
        "slam_3d_livox_ingest",
        "mid360_ingest_node",
        "--ros-args",
        "-r",
        f"__node:=mid360_ingest_runtime_{token}",
        "-p",
        f"front_input_topic:={topics['front_input']}",
        "-p",
        f"rear_input_topic:={topics['rear_input']}",
        "-p",
        f"front_output_topic:={topics['front_output']}",
        "-p",
        f"rear_output_topic:={topics['rear_output']}",
        "-p",
        f"health_topic:={topics['health']}",
        "-p",
        "front_frame_id:=mid360_front_link",
        "-p",
        "rear_frame_id:=mid360_rear_link",
        "-p",
        "front_clock_domain:=ptp_grandmaster",
        "-p",
        "rear_clock_domain:=ptp_grandmaster",
        "-p",
        "front_time_offset_ns:=2000000",
        "-p",
        "rear_time_offset_ns:=-1000000",
    ]

    client_log_directory = tempfile.TemporaryDirectory(
        prefix="mid360_ingest_runtime_client_"
    )
    os.environ["ROS_LOG_DIR"] = client_log_directory.name
    rclpy.init()
    test_node = Node(f"mid360_ingest_runtime_client_{token}")
    sensor_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    front_publisher = test_node.create_publisher(
        CustomMsg, topics["front_input"], sensor_qos
    )
    rear_publisher = test_node.create_publisher(
        CustomMsg, topics["rear_input"], sensor_qos
    )
    front_clouds = []
    rear_clouds = []
    health_messages = []
    test_node.create_subscription(
        PointCloud2, topics["front_output"], front_clouds.append, sensor_qos
    )
    test_node.create_subscription(
        PointCloud2, topics["rear_output"], rear_clouds.append, sensor_qos
    )
    test_node.create_subscription(
        DiagnosticArray, topics["health"], health_messages.append, 20
    )

    with tempfile.TemporaryDirectory(prefix="mid360_ingest_runtime_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid,
            env=environment,
        )

        def wait_until(predicate, timeout_s=10.0):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output = process.stdout.read() if process.stdout else ""
                    raise AssertionError(
                        f"mid360_ingest_node exited with {process.returncode}:\n{output}"
                    )
                rclpy.spin_once(test_node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError("timed out waiting for ROS runtime condition")

        try:
            wait_until(
                lambda: front_publisher.get_subscription_count() == 1
                and rear_publisher.get_subscription_count() == 1
            )

            front_points = [
                _point(0, 1.0, 2.0, 3.0, 42, 7, 1),
                _point(50_000_000, 4.0, 5.0, 6.0, 84, 8, 2),
            ]
            rear_points = [
                _point(0, -1.0, -2.0, 1.0, 21, 3, 0),
                _point(40_000_000, -4.0, -5.0, 2.0, 63, 4, 3),
            ]
            front_publisher.publish(_message(20_000_000_000, 11, front_points))
            rear_publisher.publish(_message(20_001_000_000, 22, rear_points))
            wait_until(
                lambda: len(front_clouds) == 1
                and len(rear_clouds) == 1
                and len(health_messages) >= 2
            )

            front_cloud = front_clouds[-1]
            rear_cloud = rear_clouds[-1]
            assert front_cloud.header.stamp == _time_msg(20_002_000_000)
            assert rear_cloud.header.stamp == _time_msg(20_000_000_000)
            assert front_cloud.header.frame_id == "mid360_front_link"
            assert rear_cloud.header.frame_id == "mid360_rear_link"
            assert front_cloud.width == 2
            assert front_cloud.point_step == 24
            assert _field_map(front_cloud) == {
                "x": (0, 7),
                "y": (4, 7),
                "z": (8, 7),
                "intensity": (12, 7),
                "offset_time": (16, 6),
                "tag": (20, 2),
                "line": (21, 2),
                "lidar_id": (22, 2),
            }
            first = front_cloud.data[: front_cloud.point_step]
            assert struct.unpack_from("<ffffI3B", first, 0) == (
                1.0,
                2.0,
                3.0,
                42.0,
                0,
                7,
                1,
                11,
            )
            front_status = next(
                message
                for message in reversed(health_messages)
                if message.status and message.status[0].name.endswith("/front")
            )
            rear_status = next(
                message
                for message in reversed(health_messages)
                if message.status and message.status[0].name.endswith("/rear")
            )
            assert _status_values(front_status)["time_offset_ns"] == "2000000"
            assert (
                _status_values(front_status)["corrected_timebase_ns"]
                == "20002000000"
            )
            assert (
                _status_values(front_status)["clock_domain"] == "ptp_grandmaster"
            )
            assert _status_values(rear_status)["time_offset_ns"] == "-1000000"
            assert (
                _status_values(rear_status)["corrected_timebase_ns"]
                == "20000000000"
            )

            prior_front_count = len(front_clouds)
            front_publisher.publish(
                _message(
                    20_100_000_000,
                    11,
                    front_points,
                    declared_count=len(front_points) + 1,
                )
            )
            wait_until(
                lambda: any(
                    message.status
                    and message.status[0].name.endswith("/front")
                    and _status_values(message).get("contract_status")
                    == "point_count_mismatch"
                    for message in health_messages
                )
            )
            assert len(front_clouds) == prior_front_count
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5.0)
            test_node.destroy_node()
            rclpy.shutdown()
            client_log_directory.cleanup()
