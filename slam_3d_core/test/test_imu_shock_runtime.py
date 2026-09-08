import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


def _time_msg(seconds: float):
    whole = int(seconds)
    return Time(sec=whole, nanosec=int(round((seconds - whole) * 1.0e9)))


def _status_values(message: DiagnosticArray):
    return {item.key: item.value for item in message.status[0].values}


def _timed_cloud(stamp_s: float, final_offset_ns=None, base_offset_ns=0):
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
    header = Header(stamp=_time_msg(stamp_s), frame_id="base_link")
    points = []
    for point_index in range(240):
        ratio = point_index / 239.0
        offset_ns = base_offset_ns + int(round(ratio * 120_000_000))
        if final_offset_ns is not None and point_index == 239:
            offset_ns = final_offset_ns
        points.append(
            (
                4.0 + 2.0 * ratio,
                -2.0 + 4.0 * ratio,
                0.2 * (point_index % 9),
                offset_ns,
            )
        )
    return point_cloud2.create_cloud(header, fields, points)


def test_offset_time_shock_quarantines_initialization_and_keyframe_mutation():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_shock_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    imu_topic = f"{prefix}/imu"
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_shock_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "use_constant_velocity_prediction:=true",
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
        "scan_voxel_leaf_m:=0.001",
        "-p",
        "map_voxel_leaf_m:=0.001",
        "-p",
        "keyframe_translation_m:=0.0",
        "-p",
        "keyframe_rotation_deg:=0.0",
        "-p",
        "deskew_enabled:=true",
        "-p",
        "deskew_wait_for_imu_watermark:=true",
        "-p",
        "deskew_imu_wait_timeout_s:=0.5",
        "-p",
        "reject_cloud_on_deskew_failure:=true",
        "-p",
        "deskew_point_time_field:=offset_time",
        "-p",
        "deskew_max_scan_duration_s:=0.2",
        "-p",
        "deskew_max_imu_gap_s:=0.03",
        "-p",
        "imu_roughness_detection_enabled:=true",
        "-p",
        "imu_shock_policy_enabled:=true",
        "-p",
        "imu_roughness_window_s:=0.15",
        "-p",
        "imu_roughness_angular_velocity_threshold_radps:=2.0",
        "-p",
        "imu_roughness_angular_accel_threshold_radps2:=1000.0",
        "-p",
        "imu_roughness_accel_norm_delta_threshold_mps2:=1.0",
        "-p",
        "imu_roughness_linear_jerk_threshold_mps3:=100.0",
        "-p",
        "imu_shock_hold_duration_s:=0.2",
        "-p",
        "imu_shock_recovery_stable_scans:=2",
        "-p",
        "quarantine_keyframes_on_shock:=true",
        "-p",
        "quarantine_pose_on_shock:=false",
        "-p",
        "record_tracking_scan_during_shock:=true",
        "-p",
        "place_recognition_enabled:=true",
        "-p",
        "place_recognition_min_keyframes:=1",
        "-p",
        "terrain_evidence_enabled:=true",
        "-p",
        "terrain_require_gravity_alignment:=false",
        "-p",
        "terrain_min_confirmations:=1",
        "-p",
        "terrain_traversal_confirmation_enabled:=true",
        "-p",
        "terrain_traversal_max_sample_gap_s:=1.0",
        "-p",
        f"terrain_preview_evidence_topic:={prefix}/terrain_preview",
        "-p",
        f"imu_topic:={imu_topic}",
        "-p",
        "imu_subscription_reliable:=true",
        "-p",
        f"health_topic:={health_topic}",
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
        "-p",
        f"terrain_evidence_topic:={prefix}/terrain",
    ]

    rclpy.init()
    test_node = Node(f"scan_to_map_shock_runtime_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=4,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    imu_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=200,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, cloud_qos)
    imu_publisher = test_node.create_publisher(Imu, imu_topic, imu_qos)
    health_messages = []
    odom_messages = []
    path_messages = []
    test_node.create_subscription(
        DiagnosticArray,
        health_topic,
        health_messages.append,
        50,
    )
    test_node.create_subscription(
        Odometry,
        f"{prefix}/odom",
        odom_messages.append,
        50,
    )
    test_node.create_subscription(
        NavPath,
        f"{prefix}/path",
        path_messages.append,
        50,
    )

    with tempfile.TemporaryDirectory(prefix="slam_3d_core_shock_runtime_") as temp_dir:
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

        def publish_imu_window(
            start_s,
            pulse_s=None,
            include_linear_pulse=True,
            linear_pulse_x=5.0,
        ):
            for sample_index in range(16):
                stamp_s = start_s - 0.01 + sample_index * 0.01
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "base_link"
                imu.linear_acceleration.z = 9.81
                if pulse_s is not None and abs(stamp_s - pulse_s) < 1.0e-6:
                    imu.angular_velocity.z = 2.5
                    if include_linear_pulse:
                        imu.linear_acceleration.x = linear_pulse_x
                imu_publisher.publish(imu)

        def publish_incomplete_shock_window(start_s):
            for sample_index in range(3):
                stamp_s = start_s - 0.01 + sample_index * 0.01
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "base_link"
                imu.linear_acceleration.z = 9.81
                if sample_index == 2:
                    imu.angular_velocity.z = 2.5
                    imu.linear_acceleration.x = 5.0
                imu_publisher.publish(imu)

        def health_at(stamp_s):
            expected = _time_msg(stamp_s)
            return [
                message
                for message in health_messages
                if message.header.stamp.sec == expected.sec
                and message.header.stamp.nanosec == expected.nanosec
            ]

        def odom_at(stamp_s):
            expected = _time_msg(stamp_s)
            return [
                message
                for message in odom_messages
                if message.header.stamp.sec == expected.sec
                and message.header.stamp.nanosec == expected.nanosec
            ]

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and test_node.count_publishers(health_topic) == 1
            )

            # Point times are valid, but IMU coverage stops at 10.01 while the
            # cloud ends at 10.12. The pulse must survive the deskew rejection.
            publish_incomplete_shock_window(10.0)
            cloud_publisher.publish(_timed_cloud(10.0))
            wait_until(lambda: bool(health_at(10.0)))
            first = _status_values(health_at(10.0)[-1])
            assert first["deskew_status"] == "insufficient_imu_coverage"
            assert first["imu_roughness_event"] == "true"
            assert first["imu_shock_phase"] == "shock"
            assert first["imu_shock_active"] == "true"
            assert first["shock_quarantined"] == "false"
            assert first["keyframe_added"] == "false"
            assert first["keyframes"] == "0"
            assert first["rejection_reason"] == "deskew_insufficient_imu_coverage"
            assert first["constant_velocity_rebase_pending"] == "true"
            assert first["terrain_preview_pending_shock"] == "false"
            assert first["terrain_preview_unlocalized_shock_events"] == "1"
            assert first["terrain_preview_traversal_samples"] == "0"

            publish_imu_window(10.5)
            cloud_publisher.publish(_timed_cloud(10.5))
            wait_until(lambda: bool(health_at(10.5)))
            recovering = _status_values(health_at(10.5)[-1])
            assert recovering["imu_shock_phase"] == "recovering"
            assert recovering["keyframes"] == "0"
            assert recovering["terrain_preview_pending_shock"] == "false"
            assert recovering["terrain_preview_traversal_samples"] == "0"

            publish_imu_window(10.8)
            cloud_publisher.publish(_timed_cloud(10.8))
            wait_until(lambda: bool(health_at(10.8)))
            initialized = _status_values(health_at(10.8)[-1])
            assert initialized["imu_shock_phase"] == "stable"
            assert initialized["accepted"] == "true"
            assert initialized["keyframe_added"] == "true"
            assert initialized["keyframes"] == "1"
            assert initialized["terrain_preview_pending_shock"] == "false"
            assert initialized["terrain_preview_traversal_samples"] == "1"
            wait_until(lambda: bool(path_messages))
            stable_path_size = len(path_messages[-1].poses)
            assert stable_path_size == 1

            publish_imu_window(11.1, pulse_s=11.18)
            cloud_publisher.publish(_timed_cloud(11.1))
            wait_until(lambda: bool(health_at(11.1)))
            quarantined = _status_values(health_at(11.1)[-1])
            assert quarantined["accepted"] == "true"
            assert quarantined["imu_shock_phase"] == "shock"
            assert quarantined["shock_quarantine_reason"] == "imu_shock"
            assert quarantined["keyframe_quarantined"] == "true"
            assert quarantined["keyframe_added"] == "false"
            assert quarantined["keyframes"] == "1"
            assert quarantined["record_tracking_scan_during_shock"] == "enabled"
            assert quarantined["tracking_scan_updated_during_shock"] == "true"
            assert quarantined["terrain_preview_pending_shock"] == "true"
            assert quarantined["terrain_preview_pending_shock_stamp_ns"] == (
                "11180000000"
            )
            assert int(quarantined["imu_snapshot_end_stamp_ns"]) <= int(
                quarantined["deskew_required_imu_stamp_ns"]
            )
            assert int(quarantined["imu_snapshot_sample_count"]) > 0
            assert quarantined[
                "terrain_preview_continuity_break_pending"
            ] == "true"
            assert quarantined["terrain_preview_traversal_samples"] == "1"
            assert quarantined["place_recognition_attempted"] == "false"
            assert quarantined["place_recognition_skipped"] == "true"
            assert quarantined["place_recognition_skip_reason"] == (
                "imu_shock_persistent_state_quarantine"
            )
            assert quarantined[
                "constant_velocity_prediction_suppressed_by_shock"
            ] == "true"
            assert health_at(11.1)[-1].status[0].level == DiagnosticStatus.WARN
            wait_until(lambda: bool(odom_at(11.1)))
            assert odom_at(11.1)[-1].pose.covariance[0] == 100.0
            assert odom_at(11.1)[-1].pose.covariance[35] == 100.0
            wait_until(lambda: len(path_messages) >= 2)
            assert len(path_messages[-1].poses) == stable_path_size
            assert quarantined["omit_shock_poses_from_path"] == "enabled"
            assert quarantined["path_pose_appended"] == "false"

            publish_imu_window(11.5)
            cloud_publisher.publish(_timed_cloud(11.5))
            wait_until(lambda: bool(health_at(11.5)))
            recovery = _status_values(health_at(11.5)[-1])
            assert recovery["imu_shock_phase"] == "recovering"
            assert recovery["imu_shock_observation"] == "stable"
            assert recovery["constant_velocity_rebase_pending"] == "true"
            assert recovery["terrain_preview_pending_shock"] == "true"
            assert recovery["terrain_preview_traversal_samples"] == "1"
            wait_until(lambda: len(path_messages) >= 3)
            assert len(path_messages[-1].poses) == stable_path_size

            publish_imu_window(11.8)
            cloud_publisher.publish(_timed_cloud(11.8))
            wait_until(lambda: bool(health_at(11.8)))
            rebased = _status_values(health_at(11.8)[-1])
            assert rebased["imu_shock_phase"] == "stable"
            assert rebased["imu_shock_just_recovered"] == "true"
            assert rebased[
                "constant_velocity_prediction_suppressed_by_shock"
            ] == "true"
            assert rebased["constant_velocity_rebase_pending"] == "false"
            assert rebased["keyframe_added"] == "true"
            assert rebased["terrain_preview_pending_shock"] == "false"
            assert rebased["terrain_preview_continuity_break_pending"] == "false"
            assert rebased["terrain_preview_continuity_breaks"] == "0"
            assert rebased["terrain_preview_traversal_samples"] == "3"
            assert rebased["terrain_preview_unlocalized_shock_events"] == "1"
            wait_until(lambda: len(path_messages[-1].poses) == stable_path_size + 1)
            assert rebased["path_pose_appended"] == "true"

            publish_imu_window(12.1)
            cloud_publisher.publish(_timed_cloud(12.1))
            wait_until(lambda: bool(health_at(12.1)))
            post_recovery = _status_values(health_at(12.1)[-1])
            assert post_recovery["imu_shock_phase"] == "stable"
            assert post_recovery["imu_shock_just_recovered"] == "false"
            assert post_recovery[
                "constant_velocity_prediction_suppressed_by_shock"
            ] == "false"

            # A rapid yaw-only input may conservatively quarantine the common
            # frontend, but it is not a spatial terrain impact without a
            # translational acceleration/jerk response.
            publish_imu_window(12.4, pulse_s=12.48, include_linear_pulse=False)
            cloud_publisher.publish(_timed_cloud(12.4))
            wait_until(lambda: bool(health_at(12.4)))
            yaw_only = _status_values(health_at(12.4)[-1])
            assert yaw_only["imu_roughness_event"] == "true"
            assert yaw_only["terrain_preview_pending_shock"] == "false"
            assert yaw_only["terrain_preview_continuity_break_pending"] == "true"
            assert yaw_only["terrain_preview_unlocalized_shock_events"] == "1"

            publish_imu_window(12.8)
            cloud_publisher.publish(_timed_cloud(12.8))
            wait_until(lambda: bool(health_at(12.8)))
            publish_imu_window(13.1)
            cloud_publisher.publish(_timed_cloud(13.1))
            wait_until(lambda: bool(health_at(13.1)))
            assert _status_values(health_at(13.1)[-1])["imu_shock_phase"] == (
                "stable"
            )
            assert _status_values(health_at(13.1)[-1])[
                "terrain_preview_continuity_breaks"
            ] == "1"

            # The next impact has a valid pre-shock sample, but recovery is
            # outside the 1 s causal segment gate. It must be discarded rather
            # than attached to the recovered endpoint.
            publish_imu_window(13.4, pulse_s=13.48)
            cloud_publisher.publish(_timed_cloud(13.4))
            wait_until(lambda: bool(health_at(13.4)))
            delayed_shock = _status_values(health_at(13.4)[-1])
            assert delayed_shock["terrain_preview_pending_shock"] == "true"

            publish_imu_window(13.8)
            cloud_publisher.publish(_timed_cloud(13.8))
            wait_until(lambda: bool(health_at(13.8)))
            publish_imu_window(14.6)
            cloud_publisher.publish(_timed_cloud(14.6))
            wait_until(lambda: bool(health_at(14.6)))
            delayed_recovery = _status_values(health_at(14.6)[-1])
            assert delayed_recovery["imu_shock_phase"] == "stable"
            assert delayed_recovery["terrain_preview_pending_shock"] == "false"
            assert delayed_recovery["terrain_preview_continuity_breaks"] == "2"
            assert delayed_recovery["terrain_preview_unlocalized_shock_events"] == (
                "2"
            )
            assert delayed_recovery["terrain_preview_shock_passages"] == "0"

            # A stronger second pulse in one translational episode must move
            # both score and timestamp. A separately rejected scan then
            # invalidates the causal bridge instead of joining pre/post poses.
            publish_imu_window(14.7, pulse_s=14.78, linear_pulse_x=3.0)
            cloud_publisher.publish(_timed_cloud(14.7))
            wait_until(lambda: bool(health_at(14.7)))
            first_pulse = _status_values(health_at(14.7)[-1])
            assert first_pulse["terrain_preview_pending_shock"] == "true"
            assert first_pulse["terrain_preview_pending_shock_stamp_ns"] == (
                "14780000000"
            )
            first_pulse_score = float(
                first_pulse["terrain_preview_pending_shock_score"]
            )

            publish_imu_window(14.86, pulse_s=14.94, linear_pulse_x=8.0)
            cloud_publisher.publish(_timed_cloud(14.86))
            wait_until(lambda: bool(health_at(14.86)))
            stronger_pulse = _status_values(health_at(14.86)[-1])
            assert stronger_pulse["terrain_preview_pending_shock_stamp_ns"] == (
                "14940000000"
            )
            assert (
                float(stronger_pulse["terrain_preview_pending_shock_score"])
                > first_pulse_score
            )

            cloud_publisher.publish(_timed_cloud(15.02))
            wait_until(lambda: bool(health_at(15.02)))
            rejected_bridge = _status_values(health_at(15.02)[-1])
            assert rejected_bridge["accepted"] == "false"
            assert rejected_bridge[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "true"

            publish_imu_window(15.2)
            cloud_publisher.publish(_timed_cloud(15.2))
            wait_until(lambda: bool(health_at(15.2)))
            publish_imu_window(15.4)
            cloud_publisher.publish(_timed_cloud(15.4))
            wait_until(lambda: bool(health_at(15.4)))
            invalidated_recovery = _status_values(health_at(15.4)[-1])
            assert invalidated_recovery["imu_shock_phase"] == "stable"
            assert invalidated_recovery["terrain_preview_pending_shock"] == "false"
            assert invalidated_recovery[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "false"
            assert invalidated_recovery["terrain_preview_continuity_breaks"] == "3"
            assert invalidated_recovery[
                "terrain_preview_unlocalized_shock_events"
            ] == "3"
            assert invalidated_recovery["terrain_preview_shock_passages"] == "0"

            # A rejected scan leaves an explicit break. A later pulse before a
            # new stable sample cannot reopen a bridge to the old stable pose.
            cloud_publisher.publish(_timed_cloud(15.6))
            wait_until(lambda: bool(health_at(15.6)))
            rejected_before_pulse = _status_values(health_at(15.6)[-1])
            assert rejected_before_pulse["accepted"] == "false"
            assert rejected_before_pulse[
                "terrain_preview_continuity_break_pending"
            ] == "true"

            publish_imu_window(15.8, pulse_s=15.88)
            cloud_publisher.publish(_timed_cloud(15.8))
            wait_until(lambda: bool(health_at(15.8)))
            pulse_after_reject = _status_values(health_at(15.8)[-1])
            assert pulse_after_reject["terrain_preview_pending_shock"] == "false"
            assert pulse_after_reject[
                "terrain_preview_continuity_break_pending"
            ] == "true"
            assert pulse_after_reject[
                "terrain_preview_unlocalized_shock_events"
            ] == "4"

            publish_imu_window(16.1)
            cloud_publisher.publish(_timed_cloud(16.1))
            wait_until(lambda: bool(health_at(16.1)))
            publish_imu_window(16.4)
            cloud_publisher.publish(_timed_cloud(16.4))
            wait_until(lambda: bool(health_at(16.4)))
            recovered_after_reject = _status_values(health_at(16.4)[-1])
            assert recovered_after_reject["imu_shock_phase"] == "stable"
            assert recovered_after_reject["terrain_preview_continuity_breaks"] == "4"
            assert recovered_after_reject["terrain_preview_shock_passages"] == "0"

            # A pending pulse invalidated by a rejected scan must remain
            # invalid even if a quiet interval resets the episode latch and a
            # second pulse arrives before stable recovery.
            publish_imu_window(16.7, pulse_s=16.78)
            cloud_publisher.publish(_timed_cloud(16.7))
            wait_until(lambda: bool(health_at(16.7)))
            assert _status_values(health_at(16.7)[-1])[
                "terrain_preview_pending_shock"
            ] == "true"

            cloud_publisher.publish(_timed_cloud(17.0))
            wait_until(lambda: bool(health_at(17.0)))
            invalidating_reject = _status_values(health_at(17.0)[-1])
            assert invalidating_reject["accepted"] == "false"
            assert invalidating_reject[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "true"

            publish_imu_window(17.2, pulse_s=17.28, linear_pulse_x=8.0)
            cloud_publisher.publish(_timed_cloud(17.2))
            wait_until(lambda: bool(health_at(17.2)))
            second_pulse_after_reject = _status_values(health_at(17.2)[-1])
            assert second_pulse_after_reject["terrain_preview_pending_shock"] == "true"
            assert second_pulse_after_reject[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "true"

            publish_imu_window(17.5)
            cloud_publisher.publish(_timed_cloud(17.5))
            wait_until(lambda: bool(health_at(17.5)))
            publish_imu_window(17.8)
            cloud_publisher.publish(_timed_cloud(17.8))
            wait_until(lambda: bool(health_at(17.8)))
            recovered_after_invalidated_second_pulse = _status_values(
                health_at(17.8)[-1]
            )
            assert recovered_after_invalidated_second_pulse[
                "terrain_preview_pending_shock"
            ] == "false"
            assert recovered_after_invalidated_second_pulse[
                "terrain_preview_continuity_breaks"
            ] == "5"
            assert recovered_after_invalidated_second_pulse[
                "terrain_preview_unlocalized_shock_events"
            ] == "5"
            assert recovered_after_invalidated_second_pulse[
                "terrain_preview_shock_passages"
            ] == "0"

            # A late rejected diagnostic cannot break the current, newer
            # traversal timeline.
            cloud_publisher.publish(_timed_cloud(17.05))
            wait_until(lambda: bool(health_at(17.05)))
            late_rejected = _status_values(health_at(17.05)[-1])
            assert late_rejected["accepted"] == "false"
            assert late_rejected[
                "terrain_preview_continuity_break_pending"
            ] == "false"
            assert late_rejected["terrain_preview_continuity_breaks"] == "5"

            # A late cloud between the stable pre-impact sample and the pending
            # pulse is diagnostic-only. It must not invalidate that newer
            # causal bridge.
            publish_imu_window(18.1)
            cloud_publisher.publish(_timed_cloud(18.1))
            wait_until(lambda: bool(health_at(18.1)))

            publish_imu_window(18.4, pulse_s=18.48)
            cloud_publisher.publish(_timed_cloud(18.4))
            wait_until(lambda: bool(health_at(18.4)))
            pending_before_late = _status_values(health_at(18.4)[-1])
            assert pending_before_late["terrain_preview_pending_shock"] == "true"
            assert pending_before_late[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "false"

            cloud_publisher.publish(_timed_cloud(18.2))
            wait_until(lambda: bool(health_at(18.2)))
            late_during_pending = _status_values(health_at(18.2)[-1])
            assert late_during_pending["rejection_reason"] == (
                "primary_cloud_timestamp_not_increasing"
            )
            assert late_during_pending["terrain_preview_pending_shock"] == "true"
            assert late_during_pending[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "false"

            publish_imu_window(18.8)
            cloud_publisher.publish(_timed_cloud(18.8))
            wait_until(lambda: bool(health_at(18.8)))
            publish_imu_window(19.1)
            cloud_publisher.publish(_timed_cloud(19.1))
            wait_until(lambda: bool(health_at(19.1)))
            recovered_after_late = _status_values(health_at(19.1)[-1])
            assert recovered_after_late["imu_shock_phase"] == "stable"
            assert recovered_after_late["terrain_preview_pending_shock"] == "false"
            assert recovered_after_late[
                "terrain_preview_pending_shock_bridge_invalidated"
            ] == "false"
            assert recovered_after_late[
                "terrain_preview_unlocalized_shock_events"
            ] == "5"

            # One corrupt point-time must still reject deskew, but it cannot
            # move the causal IMU/shock watermark seconds into the future.
            publish_imu_window(19.4, pulse_s=19.48)
            cloud_publisher.publish(_timed_cloud(19.4))
            wait_until(lambda: bool(health_at(19.4)))
            assert _status_values(health_at(19.4)[-1])["imu_shock_phase"] == (
                "shock"
            )

            publish_imu_window(19.55)
            cloud_publisher.publish(
                _timed_cloud(19.55, final_offset_ns=2_000_000_000)
            )
            wait_until(lambda: bool(health_at(19.55)))
            corrupt_point_time = _status_values(health_at(19.55)[-1])
            assert corrupt_point_time["deskew_status"] == (
                "scan_duration_out_of_range"
            )
            assert corrupt_point_time["rejection_reason"] == (
                "deskew_scan_duration_out_of_range"
            )
            assert int(corrupt_point_time["deskew_required_imu_stamp_ns"]) == (
                19_550_000_000
            )
            assert corrupt_point_time["imu_roughness_reason"] != (
                "nonadvancing_scan_time_bounds"
            )
            assert corrupt_point_time["imu_shock_phase"] == "shock"

            publish_imu_window(19.8)
            cloud_publisher.publish(_timed_cloud(19.8))
            wait_until(lambda: bool(health_at(19.8)))
            recovering_after_corrupt_point_time = _status_values(
                health_at(19.8)[-1]
            )
            assert recovering_after_corrupt_point_time[
                "imu_roughness_reason"
            ] != "nonadvancing_scan_time_bounds"
            assert recovering_after_corrupt_point_time["imu_shock_phase"] == (
                "recovering"
            )

            publish_imu_window(20.1)
            cloud_publisher.publish(_timed_cloud(20.1))
            wait_until(lambda: bool(health_at(20.1)))
            recovered_after_corrupt_point_time = _status_values(
                health_at(20.1)[-1]
            )
            assert recovered_after_corrupt_point_time["imu_shock_phase"] == (
                "stable"
            )

            # A coherent field shifted by one full duration is subtler than a
            # single outlier: its own span is valid, but its end is too far
            # from the header and must not extend the causal IMU horizon.
            publish_imu_window(20.4)
            cloud_publisher.publish(
                _timed_cloud(20.4, base_offset_ns=200_000_000)
            )
            wait_until(lambda: bool(health_at(20.4)))
            shifted_point_times = _status_values(health_at(20.4)[-1])
            assert int(shifted_point_times["deskew_required_imu_stamp_ns"]) == (
                20_400_000_000
            )
            assert int(shifted_point_times["imu_snapshot_end_stamp_ns"]) <= (
                20_400_000_000
            )
            assert shifted_point_times["imu_roughness_reason"] != (
                "nonadvancing_scan_time_bounds"
            )

            publish_imu_window(20.7)
            cloud_publisher.publish(_timed_cloud(20.7))
            wait_until(lambda: bool(health_at(20.7)))
            assert _status_values(health_at(20.7)[-1])[
                "imu_roughness_reason"
            ] != "nonadvancing_scan_time_bounds"
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
