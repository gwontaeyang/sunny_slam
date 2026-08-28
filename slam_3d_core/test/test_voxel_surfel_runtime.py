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
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _time_msg(seconds):
    from builtin_interfaces.msg import Time

    whole_seconds = int(seconds)
    return Time(
        sec=whole_seconds,
        nanosec=int(round((seconds - whole_seconds) * 1.0e9)),
    )


def _three_plane_cloud(stamp_s, translation_x):
    header = Header()
    header.stamp = _time_msg(stamp_s)
    header.frame_id = "base_link"
    points = []
    horizontal = [index * 0.2 - 3.0 for index in range(31)]
    vertical = [index * 0.2 for index in range(16)]
    for x in horizontal:
        for y in horizontal:
            points.append((x - translation_x, y, 0.0))
    for y in horizontal:
        for z in vertical:
            points.append((3.0 - translation_x, y, z))
    for x in horizontal:
        for z in vertical:
            points.append((x - translation_x, -3.0, z))
    return point_cloud2.create_cloud_xyz32(header, points)


def _status_values(message):
    return {item.key: item.value for item in message.status[0].values}


def test_voxel_surfel_backend_is_selected_and_reports_observability():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_surfel_runtime/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=voxel_surfel_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_tertiary_cloud:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_range_m:=0.0",
        "-p",
        "min_scan_points:=100",
        "-p",
        "scan_voxel_leaf_m:=0.2",
        "-p",
        "map_voxel_leaf_m:=0.2",
        "-p",
        "registration_method:=voxel_surfel",
        "-p",
        "surfel_voxel_size_m:=0.8",
        "-p",
        "surfel_min_points_per_voxel:=4",
        "-p",
        "surfel_max_planarity_ratio:=0.3",
        "-p",
        "surfel_max_plane_distance_m:=0.4",
        "-p",
        "surfel_max_center_distance_m:=1.4",
        "-p",
        "surfel_min_correspondences:=100",
        "-p",
        "surfel_min_correspondence_ratio:=0.05",
        "-p",
        "surfel_max_iterations:=15",
        "-p",
        "surfel_max_quality_rmse_m:=0.3",
        "-p",
        "surfel_max_quality_p95_m:=0.6",
        "-p",
        "robust_registration_enabled:=true",
        "-p",
        "robust_registration_quality_max_correspondence_m:=0.45",
        "-p",
        "min_registration_inlier_ratio:=0.05",
        "-p",
        "max_registration_inlier_rmse_m:=1.0",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        f"imu_topic:={prefix}/imu",
        "-p",
        f"imu_watermark_topic:={prefix}/imu_watermark",
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
    test_node = Node(f"voxel_surfel_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, qos)
    health_messages = []
    health_subscription = test_node.create_subscription(
        DiagnosticArray,
        health_topic,
        health_messages.append,
        20,
    )

    with tempfile.TemporaryDirectory(prefix="voxel_surfel_runtime_") as temp_dir:
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

        def health_at(stamp_s):
            stamp = _time_msg(stamp_s)
            return [
                message
                for message in health_messages
                if message.header.stamp.sec == stamp.sec
                and message.header.stamp.nanosec == stamp.nanosec
            ]

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and test_node.count_publishers(health_topic) == 1
            )

            cloud_publisher.publish(_three_plane_cloud(30.0, 0.0))
            wait_until(lambda: bool(health_at(30.0)))
            assert _status_values(health_at(30.0)[-1])["rejection_reason"] == "initialized"

            cloud_publisher.publish(_three_plane_cloud(30.2, 0.2))
            wait_until(lambda: bool(health_at(30.2)))
            values = _status_values(health_at(30.2)[-1])
            assert values["registration_method"] == "voxel_surfel"
            assert values["surfel_reason"] != "not_selected"
            assert int(values["surfel_correspondences"]) >= 100
            assert int(values["surfel_observable_rank"]) >= 3
            assert float(values["surfel_total_time_ms"]) >= 0.0
            assert float(values["registration_runtime_ms"]) > 0.0
            assert abs(
                float(values["registration_quality_max_correspondence_m"])
                - 0.45
            ) < 1.0e-6
            assert values["accepted"] == "true"
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5.0)
            test_node.destroy_subscription(health_subscription)
            test_node.destroy_node()
            rclpy.shutdown()
