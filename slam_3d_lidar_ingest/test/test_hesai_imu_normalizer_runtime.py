import math
import os
import signal
import subprocess
import tempfile
import time
import uuid

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu


def _imu(frame_id, stamp_s):
    message = Imu()
    message.header.frame_id = frame_id
    message.header.stamp.sec = int(stamp_s)
    message.header.stamp.nanosec = int(
        round((stamp_s - int(stamp_s)) * 1.0e9)
    )
    message.angular_velocity.x = 180.0
    message.angular_velocity.y = -90.0
    message.angular_velocity.z = 45.0
    message.linear_acceleration.x = 1.0
    message.linear_acceleration.y = 0.0
    message.linear_acceleration.z = -1.0
    message.angular_velocity_covariance[0] = 4.0
    message.linear_acceleration_covariance[0] = 9.0
    return message


def _status_values(message):
    return {item.key: item.value for item in message.status[0].values}


def test_hesai_imu_clock_units_and_rejections():
    token = uuid.uuid4().hex[:8]
    prefix = f"/hesai_imu_test/run_{token}"
    input_topic = f"{prefix}/raw"
    output_topic = f"{prefix}/normalized"
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_lidar_ingest",
        "hesai_imu_normalizer_node",
        "--ros-args",
        "-r",
        f"__node:=hesai_imu_runtime_{token}",
        "-p",
        f"input_topic:={input_topic}",
        "-p",
        f"output_topic:={output_topic}",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        "expected_input_frame_id:=hesai_front",
        "-p",
        "output_frame_id:=hesai_front",
        "-p",
        "clock_policy:=calibrated_sensor_time",
        "-p",
        "time_scale:=1.5",
        "-p",
        "anchor_input_header_ns:=10000000000",
        "-p",
        "anchor_output_header_ns:=20000000000",
        "-p",
        "calibration_valid_min_input_header_ns:=9000000000",
        "-p",
        "calibration_valid_max_input_header_ns:=11000000000",
        "-p",
        f"angular_velocity_scale:={math.pi / 180.0}",
        "-p",
        "linear_acceleration_scale:=9.80665",
        "-p",
        "max_output_stamp_step_s:=0.5",
        "-p",
        "diagnostic_period_s:=0.0",
    ]

    client_log_directory = tempfile.TemporaryDirectory(
        prefix="hesai_imu_runtime_client_"
    )
    os.environ["ROS_LOG_DIR"] = client_log_directory.name
    rclpy.init()
    test_node = Node(f"hesai_imu_runtime_client_{token}")
    qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
    publisher = test_node.create_publisher(Imu, input_topic, qos)
    outputs = []
    health_messages = []
    subscriptions = [
        test_node.create_subscription(Imu, output_topic, outputs.append, qos),
        test_node.create_subscription(
            DiagnosticArray,
            health_topic,
            health_messages.append,
            20,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="hesai_imu_runtime_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = temp_dir
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
                        f"IMU normalizer exited with {process.returncode}:\n"
                        f"{output}"
                    )
                rclpy.spin_once(test_node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError("timed out waiting for IMU runtime condition")

        try:
            wait_until(
                lambda: publisher.get_subscription_count() == 1
                and test_node.count_publishers(output_topic) == 1
            )
            publisher.publish(_imu("hesai_front", 10.0))
            wait_until(lambda: len(outputs) == 1)
            output = outputs[-1]
            output_stamp_ns = (
                output.header.stamp.sec * 1_000_000_000
                + output.header.stamp.nanosec
            )
            assert output_stamp_ns == 20_000_000_000
            assert math.isclose(output.angular_velocity.x, math.pi)
            assert math.isclose(output.angular_velocity.y, -math.pi / 2.0)
            assert math.isclose(output.linear_acceleration.z, -9.80665)
            assert math.isclose(
                output.angular_velocity_covariance[0],
                4.0 * (math.pi / 180.0) ** 2,
            )

            publisher.publish(_imu("hesai_front", 10.1))
            wait_until(lambda: len(outputs) == 2)
            second_stamp_ns = (
                outputs[-1].header.stamp.sec * 1_000_000_000
                + outputs[-1].header.stamp.nanosec
            )
            assert second_stamp_ns == 20_150_000_000

            publisher.publish(_imu("wrong_frame", 10.2))
            wait_until(
                lambda: any(
                    message.status
                    and message.status[0].message == "unexpected_frame_id"
                    for message in health_messages
                )
            )
            assert len(outputs) == 2

            publisher.publish(_imu("hesai_front", 12.0))
            wait_until(
                lambda: any(
                    message.status
                    and message.status[0].message
                    == "clock_calibration_out_of_range"
                    and _status_values(message).get("accepted") == "false"
                    for message in health_messages
                )
            )
            assert len(outputs) == 2
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                try:
                    process.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5.0)
            test_node.destroy_node()
            rclpy.shutdown()
            client_log_directory.cleanup()
            del subscriptions
