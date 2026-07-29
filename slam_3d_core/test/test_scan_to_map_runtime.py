import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Int64


def _time_msg(seconds: float):
    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1.0e9))
    from builtin_interfaces.msg import Time

    return Time(sec=whole, nanosec=nanoseconds)


def _status_values(message: DiagnosticArray):
    if not message.status:
        return {}
    return {item.key: item.value for item in message.status[0].values}


def test_sparse_scan_reports_completion_and_large_imu_rewind_is_dropped():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    imu_topic = f"{prefix}/imu"
    health_topic = f"{prefix}/health"
    watermark_topic = f"{prefix}/imu_watermark"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_scan_points:=100",
        "-p",
        f"imu_topic:={imu_topic}",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        f"imu_watermark_topic:={watermark_topic}",
        "-p",
        "gravity_roll_pitch_constraint_enabled:=true",
        "-p",
        "gravity_accel_max_age_s:=0.1",
        "-p",
        "imu_timestamp_reset_threshold_s:=0.5",
        "-p",
        f"odom_topic:={prefix}/odom",
        "-p",
        f"path_topic:={prefix}/path",
        "-p",
        f"local_map_topic:={prefix}/local_map",
        "-p",
        f"global_map_topic:={prefix}/global_map",
    ]

    rclpy.init()
    test_node = Node(f"scan_to_map_runtime_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    imu_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    watermark_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, cloud_qos)
    imu_publisher = test_node.create_publisher(Imu, imu_topic, imu_qos)
    health_messages = []
    watermark_messages = []
    test_node.create_subscription(
        DiagnosticArray,
        health_topic,
        health_messages.append,
        20,
    )
    test_node.create_subscription(
        Int64,
        watermark_topic,
        watermark_messages.append,
        watermark_qos,
    )

    with tempfile.TemporaryDirectory(prefix="slam_3d_core_runtime_") as temp_dir:
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
                        f"scan_to_map_icp_node exited with {process.returncode}:\n{output}"
                    )
                rclpy.spin_once(test_node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError("timed out waiting for ROS runtime condition")

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and test_node.count_publishers(health_topic) == 1
                and test_node.count_publishers(watermark_topic) == 1
            )

            first_imu = Imu()
            first_imu.header.stamp = _time_msg(10.0)
            first_imu.linear_acceleration.z = 9.81
            imu_publisher.publish(first_imu)
            wait_until(
                lambda: watermark_messages and watermark_messages[-1].data == 10_000_000_000
            )

            rewind_imu = Imu()
            rewind_imu.header.stamp = _time_msg(9.0)
            rewind_imu.linear_acceleration.z = 9.81
            imu_publisher.publish(rewind_imu)
            rewind_drain_deadline = time.monotonic() + 0.25
            while time.monotonic() < rewind_drain_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.02)

            header = Header()
            header.stamp = _time_msg(11.0)
            header.frame_id = "base_link"
            sparse_cloud = point_cloud2.create_cloud_xyz32(
                header,
                [(5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (5.0, -1.0, 0.0)],
            )
            cloud_publisher.publish(sparse_cloud)

            def matching_health():
                return [
                    message
                    for message in health_messages
                    if message.header.stamp.sec == 11
                    and message.header.stamp.nanosec == 0
                ]

            wait_until(lambda: bool(matching_health()))
            values = _status_values(matching_health()[-1])

            assert values["rejection_reason"] == "scan_too_sparse_before_downsample"
            assert values["gravity_constraint_status"] == "stale_at_cloud_stamp"
            assert values["gravity_accel_age_s"] == "-1"
            assert values["latest_imu_stamp_ns"] == "10000000000"
            assert values["imu_out_of_order_dropped_count"] == "1"
            assert values["imu_rewind_dropped_count"] == "1"
            assert values["imu_timestamp_reset_count"] == "1"
            assert (
                values["imu_timestamp_reset_policy"]
                == "detect_and_drop_without_partial_state_reset"
            )
            assert process.poll() is None
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
