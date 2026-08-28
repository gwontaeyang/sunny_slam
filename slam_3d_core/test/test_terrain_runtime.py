import math
import os
import signal
import struct
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _time_msg(seconds: float):
    whole = int(seconds)
    return Time(sec=whole, nanosec=int(round((seconds - whole) * 1.0e9)))


def _floor_cloud(stamp_s: float):
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    points = []
    for azimuth_deg in range(-50, 51, 10):
        azimuth = math.radians(azimuth_deg)
        for radial_index in range(2, 13):
            radius = radial_index * 0.25
            for lateral_jitter in (-0.006, 0.0, 0.006):
                points.append(
                    (
                        radius * math.cos(azimuth) - lateral_jitter * math.sin(azimuth),
                        radius * math.sin(azimuth) + lateral_jitter * math.cos(azimuth),
                        0.0,
                    )
                )
    return point_cloud2.create_cloud(
        Header(stamp=_time_msg(stamp_s), frame_id="base_link"),
        fields,
        points,
    )


def _status_values(message: DiagnosticArray):
    return {item.key: item.value for item in message.status[0].values}


def test_gravity_aligned_terrain_is_transient_for_late_joiner():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_terrain_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    imu_topic = f"{prefix}/imu"
    terrain_topic = f"{prefix}/terrain"
    terrain_colored_topic = f"{prefix}/terrain_colored"
    terrain_support_topic = f"{prefix}/terrain_support"
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_terrain_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_tertiary_cloud:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "use_constant_velocity_prediction:=false",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_range_m:=0.0",
        "-p",
        "min_scan_points:=20",
        "-p",
        "scan_voxel_leaf_m:=0.02",
        "-p",
        "map_voxel_leaf_m:=0.02",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "deskew_imu_wait_timeout_s:=0.5",
        "-p",
        f"imu_topic:={imu_topic}",
        "-p",
        "gravity_accel_low_pass_alpha:=1.0",
        "-p",
        "gravity_accel_time_constant_s:=0.0",
        "-p",
        "gravity_accel_max_age_s:=0.10",
        "-p",
        "terrain_evidence_enabled:=true",
        "-p",
        "terrain_require_gravity_alignment:=true",
        "-p",
        "terrain_azimuth_bins:=72",
        "-p",
        "terrain_radial_bin_size_m:=0.25",
        "-p",
        "terrain_min_range_m:=0.25",
        "-p",
        "terrain_max_range_m:=4.0",
        "-p",
        "terrain_seed_max_range_m:=1.5",
        "-p",
        "terrain_classification_voxel_size_m:=0.01",
        "-p",
        "terrain_min_points_per_bin:=2",
        "-p",
        "terrain_max_adjacent_gap_m:=0.30",
        "-p",
        "terrain_min_confirmations:=1",
        "-p",
        "terrain_confirmation_min_viewpoint_distance_m:=0.0",
        "-p",
        "terrain_min_view_label_confidence:=0.67",
        "-p",
        "terrain_min_label_confidence:=0.67",
        "-p",
        "map_publish_period_s:=3.0",
        "-p",
        f"terrain_evidence_topic:={terrain_topic}",
        "-p",
        f"terrain_evidence_colored_topic:={terrain_colored_topic}",
        "-p",
        f"terrain_support_evidence_topic:={terrain_support_topic}",
        "-p",
        f"health_topic:={health_topic}",
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
    node = Node(f"scan_to_map_terrain_runtime_client_{token}")
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
    terrain_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    cloud_publisher = node.create_publisher(PointCloud2, cloud_topic, cloud_qos)
    imu_publisher = node.create_publisher(Imu, imu_topic, imu_qos)
    health_messages = []
    node.create_subscription(DiagnosticArray, health_topic, health_messages.append, 10)
    initial_terrain_messages = []
    initial_subscription = node.create_subscription(
        PointCloud2,
        terrain_topic,
        initial_terrain_messages.append,
        terrain_qos,
    )
    initial_colored_messages = []
    colored_subscription = node.create_subscription(
        PointCloud2,
        terrain_colored_topic,
        initial_colored_messages.append,
        terrain_qos,
    )
    initial_support_messages = []
    support_subscription = node.create_subscription(
        PointCloud2,
        terrain_support_topic,
        initial_support_messages.append,
        terrain_qos,
    )

    with tempfile.TemporaryDirectory(prefix="slam_3d_terrain_runtime_") as temp_dir:
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
                rclpy.spin_once(node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError("timed out waiting for ROS runtime condition")

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and node.count_publishers(terrain_topic) == 1
                and node.count_publishers(terrain_colored_topic) == 1
                and node.count_publishers(terrain_support_topic) == 1
            )

            cloud_publisher.publish(_floor_cloud(10.01))

            no_dispatch_deadline = time.monotonic() + 0.10
            while time.monotonic() < no_dispatch_deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
            assert not health_messages

            for stamp_s in (10.0, 10.02):
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "base_link"
                imu.linear_acceleration.z = 9.81
                imu_publisher.publish(imu)

            wait_until(lambda: bool(health_messages))
            values = _status_values(health_messages[-1])
            assert values["deskew_imu_watermark_waited"] == "true"
            assert int(values["deskew_required_imu_stamp_ns"]) == 10_010_000_000
            assert values["terrain_gravity_status"] == "timestamped_gravity_aligned"
            assert int(values["terrain_last_keyframe_input_points"]) > 0
            assert int(values["terrain_last_keyframe_retained_points"]) > 0
            wait_until(
                lambda: any(message.width > 0 for message in initial_terrain_messages),
                timeout_s=6.0,
            )
            initial_message = next(
                message for message in reversed(initial_terrain_messages) if message.width > 0
            )
            assert initial_message.header.frame_id == "map"
            assert initial_message.height == 1
            wait_until(
                lambda: any(message.width > 0 for message in initial_colored_messages),
                timeout_s=1.0,
            )
            colored_message = next(
                message
                for message in reversed(initial_colored_messages)
                if message.width > 0
            )
            assert colored_message.header.frame_id == "map"
            assert colored_message.width == initial_message.width
            rgb_field = next(
                field for field in colored_message.fields if field.name == "rgb"
            )
            packed_rgb = struct.unpack_from(
                "<I", colored_message.data, rgb_field.offset
            )[0]
            assert (
                (packed_rgb >> 16) & 0xFF,
                (packed_rgb >> 8) & 0xFF,
                packed_rgb & 0xFF,
            ) == (105, 190, 105)

            wait_until(
                lambda: any(message.width > 0 for message in initial_support_messages),
                timeout_s=1.0,
            )
            support_message = next(
                message
                for message in reversed(initial_support_messages)
                if message.width > 0
            )
            assert support_message.header.frame_id == "map"
            assert support_message.width == initial_message.width
            fields = {field.name: field for field in support_message.fields}
            assert set(fields) == {
                "x",
                "y",
                "z",
                "rgb",
                "support_probability",
                "geometric_traversability_score",
                "label_confidence",
                "roughness_score",
                "label",
                "state",
                "connectivity_rejected",
                "confirmations",
                "observations",
            }
            assert struct.unpack_from(
                "<f", support_message.data, fields["support_probability"].offset
            )[0] == 1.0
            assert struct.unpack_from(
                "<B", support_message.data, fields["state"].offset
            )[0] == 2
            support_rgb = struct.unpack_from(
                "<I", support_message.data, fields["rgb"].offset
            )[0]
            assert (
                (support_rgb >> 16) & 0xFF,
                (support_rgb >> 8) & 0xFF,
                support_rgb & 0xFF,
            ) == (40, 170, 70)
            assert 0.0 < struct.unpack_from(
                "<f",
                support_message.data,
                fields["geometric_traversability_score"].offset,
            )[0] <= 1.0

            node.destroy_subscription(initial_subscription)
            late_messages = []
            late_started = time.monotonic()
            late_subscription = node.create_subscription(
                PointCloud2,
                terrain_topic,
                late_messages.append,
                terrain_qos,
            )
            wait_until(lambda: bool(late_messages), timeout_s=1.0)
            assert time.monotonic() - late_started < 1.0
            assert late_messages[-1].width == initial_message.width
            assert late_messages[-1].header.frame_id == "map"
            node.destroy_subscription(late_subscription)
            node.destroy_subscription(colored_subscription)
            node.destroy_subscription(support_subscription)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
            node.destroy_node()
            rclpy.shutdown()
