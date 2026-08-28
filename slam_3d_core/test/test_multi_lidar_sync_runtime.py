import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Int64


def _time_msg(seconds):
    from builtin_interfaces.msg import Time

    whole_seconds = int(seconds)
    return Time(
        sec=whole_seconds,
        nanosec=int(round((seconds - whole_seconds) * 1.0e9)),
    )


def _cloud(stamp_s, frame_id, y_offset, point_count=40):
    header = Header()
    header.stamp = _time_msg(stamp_s)
    header.frame_id = frame_id
    points = [
        (
            4.0 + 0.05 * (index % 10),
            y_offset + 0.1 * (index // 10),
            0.05 * (index % 4),
        )
        for index in range(point_count)
    ]
    return point_cloud2.create_cloud_xyz32(header, points)


def _horizontal_plane_cloud(stamp_s, frame_id, z_m, point_count):
    header = Header(stamp=_time_msg(stamp_s), frame_id=frame_id)
    points = [
        (
            3.0 + 0.05 * (index % 20),
            -1.0 + 0.05 * (index // 20),
            z_m,
        )
        for index in range(point_count)
    ]
    return point_cloud2.create_cloud_xyz32(header, points)


def _timed_cloud(stamp_s, frame_id, y_offset, duration_s, point_count=40):
    header = Header(stamp=_time_msg(stamp_s), frame_id=frame_id)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="offset_time",
            offset=12,
            datatype=PointField.UINT32,
            count=1,
        ),
    ]
    points = []
    for index in range(point_count):
        ratio = index / max(1, point_count - 1)
        points.append(
            (
                4.0 + 0.05 * (index % 10),
                y_offset + 0.1 * (index // 10),
                0.05 * (index % 4),
                int(round(ratio * duration_s * 1.0e9)),
            )
        )
    return point_cloud2.create_cloud(header, fields, points)


def _odom(stamp_s, x):
    message = Odometry()
    message.header.stamp = _time_msg(stamp_s)
    message.header.frame_id = "odom"
    message.child_frame_id = "base_link"
    message.pose.pose.position.x = x
    message.pose.pose.orientation.w = 1.0
    return message


def _imu(stamp_s, angular_velocity_z):
    message = Imu()
    message.header.stamp = _time_msg(stamp_s)
    message.header.frame_id = "imu_link"
    message.angular_velocity.z = angular_velocity_z
    message.orientation_covariance[0] = -1.0
    return message


def _status_values(message):
    return {item.key: item.value for item in message.status[0].values}


def test_three_clouds_are_consumed_once_and_timeout_degrades_to_primary():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_multi_lidar_sync_test/run_{token}"
    topics = {
        "primary": f"{prefix}/center",
        "secondary": f"{prefix}/front",
        "tertiary": f"{prefix}/rear",
        "wheel": f"{prefix}/wheel",
        "health": f"{prefix}/health",
    }
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=multi_lidar_sync_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={topics['primary']}",
        "-p",
        f"secondary_cloud_topic:={topics['secondary']}",
        "-p",
        f"tertiary_cloud_topic:={topics['tertiary']}",
        "-p",
        "use_secondary_cloud:=true",
        "-p",
        "use_tertiary_cloud:=true",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "cloud_sync_queue_size:=5",
        "-p",
        "max_auxiliary_stamp_delta_s:=0.01",
        "-p",
        "cloud_sync_wait_timeout_s:=0.30",
        "-p",
        "cloud_sync_reorder_window_s:=0.01",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "terrain_evidence_enabled:=true",
        "-p",
        "terrain_require_gravity_alignment:=false",
        "-p",
        "terrain_min_confirmations:=1",
        "-p",
        "terrain_traversal_confirmation_enabled:=true",
        "-p",
        f"terrain_evidence_topic:={prefix}/terrain",
        "-p",
        f"terrain_preview_evidence_topic:={prefix}/terrain_preview",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "inter_lidar_motion_compensation_enabled:=true",
        "-p",
        "inter_lidar_motion_compensation_source:=wheel_odom",
        "-p",
        "inter_lidar_motion_compensation_max_dt_s:=0.02",
        "-p",
        "inter_lidar_motion_failure_policy:=drop_auxiliary",
        "-p",
        "wheel_odom_max_time_delta_s:=0.2",
        "-p",
        "wheel_odom_max_interpolation_gap_s:=0.5",
        "-p",
        f"wheel_odom_topic:={topics['wheel']}",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_scan_points:=20",
        "-p",
        f"health_topic:={topics['health']}",
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
    test_node = Node(f"multi_lidar_sync_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    publishers = {
        name: test_node.create_publisher(PointCloud2, topics[name], qos)
        for name in ("primary", "secondary", "tertiary")
    }
    wheel_publisher = test_node.create_publisher(Odometry, topics["wheel"], qos)
    health_messages = []
    health_subscription = test_node.create_subscription(
        DiagnosticArray,
        topics["health"],
        health_messages.append,
        20,
    )

    with tempfile.TemporaryDirectory(prefix="multi_lidar_sync_runtime_") as temp_dir:
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
                lambda: all(
                    publisher.get_subscription_count() == 1
                    for publisher in publishers.values()
                )
                and wheel_publisher.get_subscription_count() == 1
                and test_node.count_publishers(topics["health"]) == 1
            )

            wheel_publisher.publish(_odom(19.9, 0.0))
            wheel_publisher.publish(_odom(20.1, 1.0))
            wrong_frame_odom = _odom(20.2, 2.0)
            wrong_frame_odom.child_frame_id = "wrong_base"
            wheel_publisher.publish(wrong_frame_odom)
            invalid_pose_odom = _odom(20.2, 2.0)
            invalid_pose_odom.pose.pose.orientation.w = 0.0
            wheel_publisher.publish(invalid_pose_odom)
            wheel_publisher.publish(_odom(18.0, -1.0))
            time.sleep(0.05)
            publishers["primary"].publish(_cloud(20.0, "center", 0.0))
            wait_deadline = time.monotonic() + 0.03
            while time.monotonic() < wait_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.01)
            assert not health_at(20.0)

            publishers["secondary"].publish(_cloud(20.002, "front", 1.0))
            publishers["tertiary"].publish(_cloud(19.999, "rear", -1.0))
            wait_until(lambda: bool(health_at(20.0)))
            synchronized_values = _status_values(health_at(20.0)[-1])
            assert synchronized_values["input_clouds_expected"] == "3"
            assert synchronized_values["input_clouds_merged"] == "3"
            assert synchronized_values["cloud_sync_resolution"] == "all_inputs_matched"
            assert synchronized_values["scan_points_before_downsample"] == "120"
            assert synchronized_values["secondary_cloud_consumed_count"] == "1"
            assert synchronized_values["tertiary_cloud_consumed_count"] == "1"
            assert synchronized_values["wheel_odom_frame_rejected_count"] == "1"
            assert (
                synchronized_values["wheel_odom_invalid_pose_rejected_count"]
                == "1"
            )
            assert (
                synchronized_values[
                    "wheel_odom_timestamp_rewind_rejected_count"
                ]
                == "1"
            )
            assert synchronized_values["inter_lidar_motion_status"] == "applied"
            assert synchronized_values["inter_lidar_motion_attempted_count"] == "2"
            assert synchronized_values["inter_lidar_motion_applied_count"] == "2"
            assert 0.009 < float(
                synchronized_values["inter_lidar_motion_max_translation_m"]
            ) < 0.011
            assert float(
                synchronized_values["max_input_cloud_stamp_delta_s"]
            ) <= 0.002001

            publishers["primary"].publish(_cloud(21.0, "center", 0.0))
            publishers["secondary"].publish(_cloud(21.002, "front", 1.0))
            publishers["tertiary"].publish(_cloud(20.999, "rear", -1.0))
            wait_until(lambda: bool(health_at(21.0)))
            unbracketed_values = _status_values(health_at(21.0)[-1])
            assert unbracketed_values["input_clouds_expected"] == "3"
            assert unbracketed_values["input_clouds_merged"] == "1"
            assert unbracketed_values["scan_points_before_downsample"] == "40"
            assert (
                unbracketed_values["inter_lidar_motion_status"]
                == "wheel_odom_unbracketed_or_unavailable"
            )
            assert unbracketed_values["inter_lidar_motion_attempted_count"] == "2"
            assert unbracketed_values["inter_lidar_motion_applied_count"] == "0"
            assert unbracketed_values["inter_lidar_motion_dropped_count"] == "2"
            assert unbracketed_values["inter_lidar_wheel_wait_timed_out"] == "true"

            wheel_publisher.publish(_odom(21.9, 10.0))
            wheel_publisher.publish(_odom(22.1, 11.0))
            time.sleep(0.05)
            publishers["primary"].publish(_cloud(22.0, "center", 0.0))
            for name, y_offset in (("secondary", 1.0), ("tertiary", -1.0)):
                publishers[name].publish(_cloud(22.04, name, y_offset))
                publishers[name].publish(_cloud(22.03, name, y_offset))
            wait_until(lambda: bool(health_at(22.0)))
            watermark_values = _status_values(health_at(22.0)[-1])
            assert watermark_values["input_clouds_expected"] == "3"
            assert watermark_values["input_clouds_merged"] == "1"
            assert watermark_values["cloud_sync_resolution"] == "timestamp_watermark"
            assert watermark_values["cloud_sync_wait_timed_out"] == "false"
            assert watermark_values["secondary_cloud_missing_count"] == "1"
            assert watermark_values["tertiary_cloud_missing_count"] == "1"
            assert watermark_values["secondary_cloud_out_of_order_count"] == "1"
            assert watermark_values["tertiary_cloud_out_of_order_count"] == "1"

            publishers["primary"].publish(_cloud(22.03, "center", 0.0))
            wait_until(lambda: bool(health_at(22.03)))
            reused_values = _status_values(health_at(22.03)[-1])
            assert reused_values["input_clouds_merged"] == "3"
            assert reused_values["secondary_cloud_consumed_count"] == "3"
            assert reused_values["tertiary_cloud_consumed_count"] == "3"
            assert reused_values["secondary_cloud_queued"] == "1"
            assert reused_values["tertiary_cloud_queued"] == "1"

            wheel_publisher.publish(_odom(22.9, 15.0))
            wheel_publisher.publish(_odom(23.2, 16.0))
            publishers["primary"].publish(_cloud(23.0, "center", 0.0))
            publishers["primary"].publish(_cloud(23.1, "center", 0.0))
            late_callback_deadline = time.monotonic() + 0.10
            while time.monotonic() < late_callback_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.01)
            assert not health_at(23.0)

            publishers["secondary"].publish(_cloud(23.0, "front", 1.0))
            publishers["tertiary"].publish(_cloud(23.0, "rear", -1.0))
            wait_until(lambda: bool(health_at(23.0)))
            late_values = _status_values(health_at(23.0)[-1])
            assert late_values["input_clouds_merged"] == "3"
            assert late_values["cloud_sync_resolution"] == "all_inputs_matched"
            assert late_values["cloud_sync_wait_timed_out"] == "false"
            assert late_values["secondary_cloud_missing_count"] == "1"
            assert late_values["tertiary_cloud_missing_count"] == "1"
            assert late_values["secondary_cloud_consumed_count"] == "4"
            assert late_values["tertiary_cloud_consumed_count"] == "4"

            wait_until(lambda: bool(health_at(23.1)))
            timeout_values = _status_values(health_at(23.1)[-1])
            assert timeout_values["input_clouds_merged"] == "1"
            assert timeout_values["cloud_sync_resolution"] == "wall_timeout"
            assert timeout_values["cloud_sync_wait_timed_out"] == "true"
            assert timeout_values["secondary_cloud_missing_count"] == "2"
            assert timeout_values["tertiary_cloud_missing_count"] == "2"

            # A stale primary can enter an underfull forward queue and survive
            # until the next callback causes overflow. Classify the eventual
            # victim by its dispatch watermark, not by callback identity.
            for burst_index in range(4):
                publishers["primary"].publish(
                    _cloud(24.0 + 0.1 * burst_index, "center", 0.0)
                )
            publishers["primary"].publish(_cloud(23.05, "center", 0.0))
            publishers["primary"].publish(_cloud(24.4, "center", 0.0))
            for stamp_s, x in (
                (23.9, 20.0),
                (24.1, 21.0),
                (24.3, 22.0),
                (24.5, 23.0),
            ):
                wheel_publisher.publish(_odom(stamp_s, x))
            for burst_index in range(5):
                stamp_s = 24.0 + 0.1 * burst_index
                publishers["secondary"].publish(
                    _cloud(stamp_s, "front", 1.0)
                )
                publishers["tertiary"].publish(
                    _cloud(stamp_s, "rear", -1.0)
                )
            wait_until(lambda: bool(health_at(24.4)))
            stale_overflow_values = _status_values(health_at(24.0)[-1])
            assert int(
                stale_overflow_values["primary_sync_queue_overflow_count"]
            ) >= 1
            assert stale_overflow_values["primary_cloud_gap_before"] == "false"

            # A not-yet-dispatched cloud can also evict itself when it arrives
            # late to a full queue. Unlike the stale case above, that is a real
            # forward-timeline omission and must mark the first survivor.
            for burst_index in range(5):
                publishers["primary"].publish(
                    _cloud(24.6 + 0.1 * burst_index, "center", 0.0)
                )
            publishers["primary"].publish(_cloud(24.5, "center", 0.0))
            for stamp_s, x in (
                (24.7, 24.0),
                (24.9, 25.0),
                (25.1, 26.0),
            ):
                wheel_publisher.publish(_odom(stamp_s, x))
            for burst_index in range(5):
                stamp_s = 24.6 + 0.1 * burst_index
                publishers["secondary"].publish(
                    _cloud(stamp_s, "front", 1.0)
                )
                publishers["tertiary"].publish(
                    _cloud(stamp_s, "rear", -1.0)
                )
            wait_until(lambda: bool(health_at(25.0)))
            forward_self_eviction = _status_values(health_at(24.6)[-1])
            assert forward_self_eviction["primary_cloud_gap_before"] == "true"
            forward_end_values = _status_values(health_at(25.0)[-1])
            breaks_before_next_overflow = int(
                forward_end_values["terrain_preview_continuity_breaks"]
            )
            if breaks_before_next_overflow == 0:
                assert forward_end_values[
                    "terrain_preview_continuity_break_pending"
                ] == "true"

            # Keep the oldest primary waiting for auxiliaries while a forward
            # burst exceeds the queue. The first surviving cloud must carry the
            # omitted-scan break into the terrain traversal timeline.
            for burst_index in range(7):
                publishers["primary"].publish(
                    _cloud(26.0 + 0.1 * burst_index, "center", 0.0)
                )
            wheel_publisher.publish(_odom(26.1, 27.0))
            wheel_publisher.publish(_odom(26.3, 28.0))
            publishers["secondary"].publish(_cloud(26.2, "front", 1.0))
            publishers["tertiary"].publish(_cloud(26.2, "rear", -1.0))
            wait_until(lambda: bool(health_at(26.2)))
            overflow_values = _status_values(health_at(26.2)[-1])
            assert int(overflow_values["primary_sync_queue_overflow_count"]) >= 2
            assert overflow_values["primary_cloud_gap_before"] == "true"
            if overflow_values["accepted"] == "true":
                assert int(
                    overflow_values["terrain_preview_continuity_breaks"]
                ) == (breaks_before_next_overflow + 1)
                assert overflow_values[
                    "terrain_preview_continuity_break_pending"
                ] == "false"
            else:
                assert int(
                    overflow_values["terrain_preview_continuity_breaks"]
                ) == breaks_before_next_overflow
                assert overflow_values[
                    "terrain_preview_continuity_break_pending"
                ] == "true"
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


def test_deterministic_sync_waits_for_nearest_auxiliary_candidate():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_nearest_sync_test/run_{token}"
    topics = {
        "primary": f"{prefix}/primary",
        "secondary": f"{prefix}/secondary",
        "health": f"{prefix}/health",
    }
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=nearest_sync_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={topics['primary']}",
        "-p",
        f"secondary_cloud_topic:={topics['secondary']}",
        "-p",
        "use_secondary_cloud:=true",
        "-p",
        "use_tertiary_cloud:=false",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "cloud_sync_queue_size:=10",
        "-p",
        "max_auxiliary_stamp_delta_s:=0.05",
        "-p",
        "cloud_sync_wait_for_nearest_candidate:=true",
        "-p",
        "cloud_sync_allow_wall_timeout_fallback:=false",
        "-p",
        "cloud_sync_wait_timeout_s:=0.10",
        "-p",
        "cloud_sync_reorder_window_s:=0.01",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "terrain_evidence_enabled:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "inter_lidar_motion_compensation_enabled:=false",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_scan_points:=20",
        "-p",
        f"health_topic:={topics['health']}",
        "-p",
        f"imu_topic:={prefix}/imu",
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
    test_node = Node(f"nearest_sync_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    primary_publisher = test_node.create_publisher(
        PointCloud2, topics["primary"], qos
    )
    secondary_publisher = test_node.create_publisher(
        PointCloud2, topics["secondary"], qos
    )
    health_messages = []
    health_subscription = test_node.create_subscription(
        DiagnosticArray,
        topics["health"],
        health_messages.append,
        20,
    )

    with tempfile.TemporaryDirectory(prefix="nearest_sync_runtime_") as temp_dir:
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
                lambda: primary_publisher.get_subscription_count() == 1
                and secondary_publisher.get_subscription_count() == 1
                and test_node.count_publishers(topics["health"]) == 1
            )

            secondary_publisher.publish(_cloud(59.98, "secondary", 2.0))
            time.sleep(0.05)
            primary_publisher.publish(_cloud(60.0, "primary", 0.0))
            wait_deadline = time.monotonic() + 0.20
            while time.monotonic() < wait_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.01)
            assert not health_at(60.0)

            secondary_publisher.publish(_cloud(60.01, "secondary", 1.0))
            secondary_publisher.publish(_cloud(60.07, "secondary", 3.0))
            wait_until(lambda: bool(health_at(60.0)))
            values = _status_values(health_at(60.0)[-1])

            assert values["input_clouds_merged"] == "2"
            assert values["cloud_sync_resolution"] == "all_inputs_matched"
            assert values["cloud_sync_wait_timed_out"] == "false"
            assert values["cloud_sync_wait_for_nearest_candidate"] == "enabled"
            assert values["cloud_sync_allow_wall_timeout_fallback"] == "disabled"
            assert 0.0099 <= float(values["max_input_cloud_stamp_delta_s"]) <= 0.0101
            assert values["secondary_cloud_stale_drop_count"] == "1"
            assert values["secondary_cloud_consumed_count"] == "1"
            assert values["secondary_cloud_queued"] == "1"
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


def test_imu_rotation_compensates_auxiliary_without_wheel_subscription():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_multi_lidar_imu_test/run_{token}"
    topics = {
        "primary": f"{prefix}/primary",
        "secondary": f"{prefix}/secondary",
        "imu": f"{prefix}/imu",
        "wheel": f"{prefix}/unused_wheel",
        "health": f"{prefix}/health",
        "watermark": f"{prefix}/imu_watermark",
    }
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=multi_lidar_imu_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={topics['primary']}",
        "-p",
        f"secondary_cloud_topic:={topics['secondary']}",
        "-p",
        "use_secondary_cloud:=true",
        "-p",
        "use_tertiary_cloud:=false",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "max_auxiliary_stamp_delta_s:=0.03",
        "-p",
        "cloud_sync_wait_timeout_s:=0.30",
        "-p",
        "cloud_sync_reorder_window_s:=0.03",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=true",
        "-p",
        "deskew_wait_for_imu_watermark:=false",
        "-p",
        "reject_cloud_on_deskew_failure:=true",
        "-p",
        "deskew_point_time_field:=offset_time",
        "-p",
        "deskew_point_time_scale:=1.0e-9",
        "-p",
        "deskew_max_scan_duration_s:=0.2",
        "-p",
        "deskew_max_imu_gap_s:=0.02",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "inter_lidar_motion_compensation_enabled:=true",
        "-p",
        "inter_lidar_motion_compensation_source:=imu_rotation",
        "-p",
        "inter_lidar_motion_compensation_max_dt_s:=0.03",
        "-p",
        "inter_lidar_imu_rotation_scale:=1.0",
        "-p",
        "inter_lidar_gap_recovery_hypothesis_enabled:=true",
        "-p",
        "inter_lidar_gap_recovery_hypothesis_min_gap_s:=0.5",
        "-p",
        "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio:=0.02",
        "-p",
        "inter_lidar_motion_failure_policy:=drop_auxiliary",
        "-p",
        f"wheel_odom_topic:={topics['wheel']}",
        "-p",
        f"imu_topic:={topics['imu']}",
        "-p",
        "imu_subscription_reliable:=true",
        "-p",
        f"imu_watermark_topic:={topics['watermark']}",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_scan_points:=20",
        "-p",
        "scan_voxel_leaf_m:=0.01",
        "-p",
        "map_voxel_leaf_m:=0.01",
        "-p",
        f"health_topic:={topics['health']}",
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
    test_node = Node(f"multi_lidar_imu_runtime_client_{token}")
    reliable_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    primary_publisher = test_node.create_publisher(
        PointCloud2, topics["primary"], reliable_qos
    )
    secondary_publisher = test_node.create_publisher(
        PointCloud2, topics["secondary"], reliable_qos
    )
    imu_publisher = test_node.create_publisher(Imu, topics["imu"], reliable_qos)
    health_messages = []
    watermark_messages = []
    health_subscription = test_node.create_subscription(
        DiagnosticArray,
        topics["health"],
        health_messages.append,
        20,
    )
    watermark_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    watermark_subscription = test_node.create_subscription(
        Int64,
        topics["watermark"],
        watermark_messages.append,
        watermark_qos,
    )

    with tempfile.TemporaryDirectory(prefix="multi_lidar_imu_runtime_") as temp_dir:
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
                lambda: primary_publisher.get_subscription_count() == 1
                and secondary_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and test_node.count_publishers(topics["health"]) == 1
                and test_node.count_publishers(topics["watermark"]) == 1
            )
            assert test_node.count_subscribers(topics["wheel"]) == 0

            for sample_index in range(18):
                stamp_s = 39.98 + sample_index * 0.01
                imu_publisher.publish(_imu(stamp_s, 1.0))
            wait_until(
                lambda: watermark_messages
                and watermark_messages[-1].data >= 40_150_000_000
            )

            primary_publisher.publish(
                _timed_cloud(40.02, "primary", 0.0, duration_s=0.10)
            )
            secondary_publisher.publish(
                _timed_cloud(40.0, "secondary", 1.0, duration_s=0.14)
            )
            wait_until(lambda: bool(health_at(40.02)))
            values = _status_values(health_at(40.02)[-1])

            assert values["input_clouds_expected"] == "2"
            assert values["input_clouds_merged"] == "2"
            assert values["deskew_status"] == "applied"
            assert values["deskew_imu_watermark_waited"] == "false"
            assert int(values["deskew_required_imu_stamp_ns"]) == 40_140_000_000
            assert int(values["imu_snapshot_end_stamp_ns"]) == 40_140_000_000
            assert values["inter_lidar_motion_compensation_source"] == "imu_rotation"
            assert float(values["inter_lidar_imu_rotation_scale"]) == 1.0
            assert values["inter_lidar_motion_status"] == "applied"
            assert values["inter_lidar_motion_attempted_count"] == "1"
            assert values["inter_lidar_motion_applied_count"] == "1"
            assert values["inter_lidar_motion_dropped_count"] == "0"
            assert float(values["inter_lidar_motion_max_translation_m"]) == 0.0
            assert 1.0 < float(values["inter_lidar_motion_max_rotation_deg"]) < 1.3

            for sample_index in range(18):
                stamp_s = 40.98 + sample_index * 0.01
                imu_publisher.publish(_imu(stamp_s, 1.0))
            wait_until(
                lambda: watermark_messages
                and watermark_messages[-1].data >= 41_150_000_000
            )

            primary_publisher.publish(
                _timed_cloud(41.02, "primary", 0.0, duration_s=0.10)
            )
            secondary_publisher.publish(
                _timed_cloud(41.0, "secondary", 1.0, duration_s=0.14)
            )
            wait_until(lambda: bool(health_at(41.02)))
            gap_values = _status_values(health_at(41.02)[-1])

            assert gap_values[
                "inter_lidar_gap_recovery_hypothesis_attempted"
            ] == "true"
            assert gap_values[
                "inter_lidar_gap_recovery_selected_input"
            ] in ("compensated", "uncompensated")
            assert float(
                gap_values["inter_lidar_gap_recovery_runtime_ms"]
            ) >= 0.0
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5.0)
            test_node.destroy_subscription(health_subscription)
            test_node.destroy_subscription(watermark_subscription)
            test_node.destroy_node()
            rclpy.shutdown()


def test_registration_only_cloud_is_used_by_frontend_but_excluded_from_map():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_registration_only_test/run_{token}"
    topics = {
        "primary": f"{prefix}/obstacles",
        "ground": f"{prefix}/ground",
        "health": f"{prefix}/health",
        "preprocessed": f"{prefix}/registration_scan",
        "global_map": f"{prefix}/global_map",
    }
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=registration_only_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={topics['primary']}",
        "-p",
        f"secondary_cloud_topic:={topics['ground']}",
        "-p",
        "use_secondary_cloud:=true",
        "-p",
        "use_tertiary_cloud:=false",
        "-p",
        "secondary_cloud_registration_only:=true",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "max_auxiliary_stamp_delta_s:=0.01",
        "-p",
        "cloud_sync_wait_timeout_s:=0.30",
        "-p",
        "cloud_sync_reorder_window_s:=0.01",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "terrain_evidence_enabled:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "inter_lidar_motion_compensation_enabled:=false",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_scan_points:=20",
        "-p",
        "scan_voxel_leaf_m:=0.01",
        "-p",
        "map_voxel_leaf_m:=0.01",
        "-p",
        "registration_only_cloud_voxel_leaf_m:=0.01",
        "-p",
        "registration_only_cloud_max_point_ratio:=0.25",
        "-p",
        "registration_only_cloud_max_points:=30",
        "-p",
        "publish_preprocessed_scan:=true",
        "-p",
        f"preprocessed_scan_topic:={topics['preprocessed']}",
        "-p",
        "map_publish_period_s:=0.10",
        "-p",
        f"health_topic:={topics['health']}",
        "-p",
        f"global_map_topic:={topics['global_map']}",
        "-p",
        f"imu_topic:={prefix}/imu",
        "-p",
        f"imu_watermark_topic:={prefix}/imu_watermark",
        "-p",
        f"wheel_odom_topic:={prefix}/wheel",
        "-p",
        f"odom_topic:={prefix}/odom",
        "-p",
        f"path_topic:={prefix}/path",
        "-p",
        f"local_map_topic:={prefix}/local_map",
    ]

    rclpy.init()
    test_node = Node(f"registration_only_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    primary_publisher = test_node.create_publisher(
        PointCloud2, topics["primary"], qos
    )
    ground_publisher = test_node.create_publisher(
        PointCloud2, topics["ground"], qos
    )
    health_messages = []
    registration_scans = []
    global_maps = []
    health_subscription = test_node.create_subscription(
        DiagnosticArray, topics["health"], health_messages.append, 20
    )
    registration_subscription = test_node.create_subscription(
        PointCloud2, topics["preprocessed"], registration_scans.append, qos
    )
    global_map_subscription = test_node.create_subscription(
        PointCloud2, topics["global_map"], global_maps.append, qos
    )

    with tempfile.TemporaryDirectory(prefix="registration_only_runtime_") as temp_dir:
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
                lambda: primary_publisher.get_subscription_count() == 1
                and ground_publisher.get_subscription_count() == 1
                and test_node.count_publishers(topics["health"]) == 1
                and test_node.count_publishers(topics["preprocessed"]) == 1
                and test_node.count_publishers(topics["global_map"]) == 1
            )

            ground_publisher.publish(
                _horizontal_plane_cloud(50.0, "base_link", 0.0, 240)
            )
            primary_publisher.publish(
                _horizontal_plane_cloud(50.0, "base_link", 1.5, 120)
            )
            wait_until(
                lambda: health_messages and registration_scans and global_maps
            )

            values = _status_values(health_messages[-1])
            assert values["input_clouds_expected"] == "2"
            assert values["input_clouds_merged"] == "2"
            assert values["secondary_cloud_role"] == "registration_only"
            assert values["registration_only_clouds_merged"] == "1"
            assert values["registration_only_points_before_balance"] == "240"
            assert values["registration_only_points_after_balance"] == "30"
            assert values["mapping_scan_points_before_downsample"] == "120"
            assert values["mapping_scan_points_after_downsample"] == "120"
            assert values["scan_points_before_downsample"] == "150"
            assert values["scan_points_after_downsample"] == "150"

            registration_z = [
                float(point[2])
                for point in point_cloud2.read_points(
                    registration_scans[-1], field_names=("x", "y", "z")
                )
            ]
            map_z = [
                float(point[2])
                for point in point_cloud2.read_points(
                    global_maps[-1], field_names=("x", "y", "z")
                )
            ]
            assert len(registration_z) == 150
            assert sum(abs(z) < 1.0e-5 for z in registration_z) == 30
            assert len(map_z) == 120
            assert all(abs(z - 1.5) < 1.0e-5 for z in map_z)

            # A support-only frame must not advance tracking or create map
            # content when the mapping/obstacle stream has no usable points.
            ground_publisher.publish(
                _horizontal_plane_cloud(51.0, "base_link", 0.0, 240)
            )
            primary_publisher.publish(
                _horizontal_plane_cloud(51.0, "base_link", 1.5, 0)
            )
            wait_until(lambda: bool(health_at(51.0)))
            support_only_values = _status_values(health_at(51.0)[-1])
            assert support_only_values[
                "registration_only_points_before_balance"
            ] == "240"
            assert support_only_values[
                "registration_only_points_after_balance"
            ] == "0"
            assert (
                support_only_values["mapping_scan_points_before_downsample"]
                == "0"
            )
            assert support_only_values["scan_points_before_downsample"] == "0"
            assert support_only_values["accepted"] == "false"
            assert support_only_values[
                "rejection_reason"
            ] == "scan_too_sparse_before_downsample"
        finally:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5.0)
            test_node.destroy_subscription(health_subscription)
            test_node.destroy_subscription(registration_subscription)
            test_node.destroy_subscription(global_map_subscription)
            test_node.destroy_node()
            rclpy.shutdown()
