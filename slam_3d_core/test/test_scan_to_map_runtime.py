import math
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


def _time_msg(seconds: float):
    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1.0e9))
    from builtin_interfaces.msg import Time

    return Time(sec=whole, nanosec=nanoseconds)


def _status_values(message: DiagnosticArray):
    if not message.status:
        return {}
    return {item.key: item.value for item in message.status[0].values}


def test_online_temporal_cleanup_is_rejected_with_migration_guide():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_legacy_temporal_{token}",
        "-p",
        "temporal_static_map_enabled:=true",
    ]
    with tempfile.TemporaryDirectory(prefix="slam_3d_core_legacy_temporal_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "temporal_static_map_enabled" in output
    assert "slam_3d_map_cleanup" in output


def test_nonpositive_max_scan_duration_is_rejected():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_duration_{token}",
        "-p",
        "deskew_max_scan_duration_s:=0.0",
    ]
    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_invalid_duration_"
    ) as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "deskew_max_scan_duration_s must be finite and greater than zero" in (
        result.stdout + result.stderr
    )


def test_negative_initialization_warmup_is_rejected():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_warmup_{token}",
        "-p",
        "initialization_warmup_duration_s:=-0.1",
    ]
    with tempfile.TemporaryDirectory(prefix="slam_3d_core_invalid_warmup_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "initialization_warmup_duration_s must be finite and non-negative" in (
        result.stdout + result.stderr
    )


def test_large_pose_jump_recovery_requires_independent_geometry_validation():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_large_loop_{token}",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_enabled:=true",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_max_translation_m:=10.0",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg:=20.0",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance:=0.1",
    ]
    with tempfile.TemporaryDirectory(prefix="slam_3d_core_invalid_large_loop_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "large pose-jump recovery requires mapping-cloud refinement" in (
        result.stdout + result.stderr
    )


def test_distributed_loop_seed_requires_robust_high_confidence_path():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_loop_seed_{token}",
        "-p",
        "pose_graph_distributed_loop_seed_enabled:=true",
    ]
    with tempfile.TemporaryDirectory(prefix="slam_3d_core_invalid_loop_seed_") as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "distributed loop seeding requires large pose-jump recovery" in (
        result.stdout + result.stderr
    )


def test_verified_coarse_loop_requires_large_pose_jump_recovery():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_verified_coarse_{token}",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled:=true",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_"
        "verified_coarse_min_inlier_ratio:=0.9",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_"
        "verified_coarse_max_inlier_rmse_m:=0.2",
        "-p",
        "place_recognition_mapping_refinement_large_pose_jump_"
        "verified_coarse_min_target_coverage_ratio:=0.7",
    ]
    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_invalid_verified_coarse_"
    ) as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "verified coarse mapping-cloud acceptance requires" in (
        result.stdout + result.stderr
    )


def test_required_pose_graph_mapping_verification_requires_refinement():
    token = uuid.uuid4().hex[:8]
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_invalid_required_mapping_{token}",
        "-p",
        "place_recognition_mapping_refinement_required_for_pose_graph:=true",
    ]
    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_invalid_required_mapping_"
    ) as temp_dir:
        environment = os.environ.copy()
        environment["ROS_LOG_DIR"] = str(Path(temp_dir) / "ros_log")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            env=environment,
            check=False,
        )

    assert result.returncode == 1
    assert "required mapping-cloud pose-graph verification needs" in (
        result.stdout + result.stderr
    )


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
        "gravity_accel_scale:=-1.0",
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
            first_imu.linear_acceleration.z = -9.81
            imu_publisher.publish(first_imu)
            wait_until(
                lambda: watermark_messages and watermark_messages[-1].data == 10_000_000_000
            )

            rewind_imu = Imu()
            rewind_imu.header.stamp = _time_msg(9.0)
            rewind_imu.linear_acceleration.z = -9.81
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
            assert values["gravity_accel_scale"] == "-1"
            assert values["ground_vehicle_motion_constraint"] == "disabled"
            assert values["ground_motion_constraint_evaluated"] == "false"
            assert values["ground_motion_constraint_status"] == "not_evaluated"
            assert math.isclose(
                float(values["latest_gravity_roll_deg"]),
                0.0,
                abs_tol=1.0e-6,
            )
            assert math.isclose(
                float(values["latest_gravity_pitch_deg"]),
                0.0,
                abs_tol=1.0e-6,
            )
            assert values["latest_imu_stamp_ns"] == "10000000000"
            assert values["imu_out_of_order_dropped_count"] == "1"
            assert values["imu_rewind_dropped_count"] == "1"
            assert values["imu_timestamp_reset_count"] == "1"
            assert values["pose_graph_solver"] == "legacy_relaxation"
            assert values["pose_graph_information_mode"] == "configured_diagonal"
            assert values["registration_information_valid"] == "false"
            assert values["registration_information_status"] == "disabled"
            assert int(values["registration_information_observable_rank"]) == 0
            assert float(values["registration_information_evaluation_runtime_ms"]) == 0.0
            assert values["pose_graph_solver_status"] == "not_run"
            assert values["pose_graph_convergence_reason"] == "not_run"
            assert int(values["pose_graph_solver_stages"]) == 0
            assert int(values["pose_graph_solver_successful_steps"]) == 0
            assert int(values["pose_graph_solver_unsuccessful_steps"]) == 0
            assert int(values["pose_graph_dcs_downweighted_loop_edges"]) == 0
            assert float(values["pose_graph_dcs_min_scale"]) == 1.0
            assert float(values["pose_graph_new_loop_dcs_scale"]) == 1.0
            assert float(values["pose_graph_solver_first_stage_initial_cost"]) == 0.0
            assert float(values["pose_graph_solver_last_stage_final_cost"]) == 0.0
            assert float(values["pose_graph_solver_final_gradient_max_norm"]) == 0.0
            assert float(values["pose_graph_optimization_runtime_ms"]) == 0.0
            assert float(values["pose_graph_iteration_runtime_ms"]) == 0.0
            assert float(values["pose_graph_mean_iteration_runtime_ms"]) == 0.0
            assert int(values["pose_graph_registration_information_edges"]) == 0
            assert int(values["pose_graph_configured_diagonal_information_edges"]) == 0
            assert values["pose_graph_new_loop_information_status"] == "not_evaluated"
            assert (
                values["imu_timestamp_reset_policy"]
                == "detect_and_drop_without_partial_state_reset"
            )

            # Even a sparse rejected cloud advances the primary timeline. A
            # denser cloud arriving behind it must not reach IMU or SLAM state.
            late_header = Header()
            late_header.stamp = _time_msg(10.5)
            late_header.frame_id = "base_link"
            late_dense_cloud = point_cloud2.create_cloud_xyz32(
                late_header,
                [
                    (
                        4.0 + 0.02 * (index % 20),
                        -1.0 + 0.03 * (index // 20),
                        0.02 * (index % 7),
                    )
                    for index in range(160)
                ],
            )
            cloud_publisher.publish(late_dense_cloud)

            def late_health():
                return [
                    message
                    for message in health_messages
                    if message.header.stamp.sec == 10
                    and message.header.stamp.nanosec == 500_000_000
                ]

            wait_until(lambda: bool(late_health()))
            late_values = _status_values(late_health()[-1])
            assert late_values["accepted"] == "false"
            assert (
                late_values["rejection_reason"]
                == "primary_cloud_timestamp_not_increasing"
            )
            assert late_values["imu_shock_time_reset"] == "false"
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


def test_offset_time_cloud_waits_for_imu_watermark_before_deskew():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_deskew_test/run_{token}"
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
        f"__node:=scan_to_map_deskew_runtime_{token}",
        "-p",
        "temporal_static_map_enabled:=false",
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
        "min_range_m:=0.0",
        "-p",
        "min_scan_points:=20",
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
        "imu_require_expected_frame:=true",
        "-p",
        "imu_expected_frame:=expected_imu",
        "-p",
        "scan_voxel_leaf_m:=0.001",
        "-p",
        "map_voxel_leaf_m:=0.001",
        "-p",
        "map_publish_period_s:=0.05",
        "-p",
        f"imu_topic:={imu_topic}",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        f"imu_watermark_topic:={watermark_topic}",
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
    test_node = Node(f"scan_to_map_deskew_runtime_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    imu_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
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
    odom_messages = []
    global_map_messages = []
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
    test_node.create_subscription(
        Odometry,
        f"{prefix}/odom",
        odom_messages.append,
        20,
    )
    test_node.create_subscription(
        PointCloud2,
        f"{prefix}/global_map",
        global_map_messages.append,
        cloud_qos,
    )

    with tempfile.TemporaryDirectory(prefix="slam_3d_core_deskew_runtime_") as temp_dir:
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

            wrong_frame_imu = Imu()
            wrong_frame_imu.header.stamp = _time_msg(9.98)
            wrong_frame_imu.header.frame_id = "wrong_imu"
            wrong_frame_imu.angular_velocity.z = 100.0
            wrong_frame_imu.linear_acceleration.z = 9.81
            imu_publisher.publish(wrong_frame_imu)

            nonfinite_imu = Imu()
            nonfinite_imu.header.stamp = _time_msg(9.985)
            nonfinite_imu.header.frame_id = "expected_imu"
            nonfinite_imu.angular_velocity.x = math.nan
            nonfinite_imu.linear_acceleration.z = 9.81
            imu_publisher.publish(nonfinite_imu)
            invalid_drain_deadline = time.monotonic() + 0.2
            while time.monotonic() < invalid_drain_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.02)
            assert not watermark_messages

            for sample_index in range(7):
                stamp_s = 9.99 + sample_index * 0.01
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "expected_imu"
                imu.angular_velocity.z = 1.0
                imu.linear_acceleration.z = 9.81
                imu_publisher.publish(imu)

            wait_until(
                lambda: watermark_messages
                and watermark_messages[-1].data >= 10_050_000_000
            )

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
            header = Header()
            header.stamp = _time_msg(10.0)
            header.frame_id = "base_link"
            points = []
            for point_index in range(200):
                ratio = point_index / 199.0
                points.append(
                    (
                        5.0 + 0.5 * ratio,
                        -2.0 + 4.0 * ratio,
                        0.25 * (point_index % 7),
                        int(round(ratio * 120_000_000)),
                    )
                )
            timed_cloud = point_cloud2.create_cloud(header, fields, points)
            cloud_publisher.publish(timed_cloud)

            def matching_health():
                return [
                    message
                    for message in health_messages
                    if message.header.stamp.sec == 10
                    and message.header.stamp.nanosec == 0
                ]

            no_dispatch_deadline = time.monotonic() + 0.12
            while time.monotonic() < no_dispatch_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.02)
            assert not matching_health()

            for sample_index in range(7, 15):
                stamp_s = 9.99 + sample_index * 0.01
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "expected_imu"
                imu.angular_velocity.z = 1.0
                imu.linear_acceleration.z = 9.81
                imu_publisher.publish(imu)

            wait_until(
                lambda: watermark_messages
                and watermark_messages[-1].data >= 10_120_000_000
            )
            wait_until(lambda: bool(matching_health()))
            values = _status_values(matching_health()[-1])
            assert values["deskew_status"] == "applied"
            assert values["deskew_applied"] == "true"
            assert values["deskew_point_time_field"] == "offset_time"
            assert values["deskew_points_with_time"] == "200"
            assert values["deskew_point_count"] == "200"
            assert float(values["deskew_time_ratio"]) == 1.0
            assert abs(float(values["deskew_scan_duration_s"]) - 0.12) < 1.0e-6
            assert values["deskew_wait_for_imu_watermark"] == "enabled"
            assert values["deskew_imu_watermark_waited"] == "true"
            assert values["deskew_imu_wait_timed_out"] == "false"
            assert values["deskew_required_imu_stamp_ns"] == "10120000000"
            assert int(values["deskew_dispatch_imu_stamp_ns"]) >= 10_120_000_000
            assert values["deskew_imu_watermark_waited_count"] == "1"
            assert values["deskew_imu_wait_timeout_count"] == "0"
            assert values["imu_require_expected_frame"] == "enabled"
            assert values["imu_expected_frame"] == "expected_imu"
            assert values["imu_frame_rejected_count"] == "1"
            assert values["imu_nonfinite_rejected_count"] == "1"

            wait_until(lambda: bool(global_map_messages))
            corrected_points = [
                (float(point[0]), float(point[1]), float(point[2]))
                for point in point_cloud2.read_points(
                    global_map_messages[-1],
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ]
            # Core publishes one-pass geometry; static confirmation belongs offline.
            assert values["keyframes"] == "1"
            assert corrected_points

            expected_positive_yaw = (
                5.5 * math.cos(0.12) - 2.0 * math.sin(0.12),
                5.5 * math.sin(0.12) + 2.0 * math.cos(0.12),
                0.75,
            )
            expected_negative_yaw = (
                5.5 * math.cos(0.12) + 2.0 * math.sin(0.12),
                -5.5 * math.sin(0.12) + 2.0 * math.cos(0.12),
                0.75,
            )

            def nearest_distance(target):
                return min(
                    math.sqrt(
                        (point[0] - target[0]) ** 2
                        + (point[1] - target[1]) ** 2
                        + (point[2] - target[2]) ** 2
                    )
                    for point in corrected_points
                )

            positive_distance = nearest_distance(expected_positive_yaw)
            negative_distance = nearest_distance(expected_negative_yaw)
            assert positive_distance < 0.02
            assert positive_distance < negative_distance * 0.1

            odom_count_after_valid_scan = len(odom_messages)
            map_point_count_after_valid_scan = len(corrected_points)
            second_header = Header()
            second_header.stamp = _time_msg(10.3)
            second_header.frame_id = "base_link"
            cloud_publisher.publish(
                point_cloud2.create_cloud(second_header, fields, points)
            )

            def rejected_health():
                return [
                    message
                    for message in health_messages
                    if message.header.stamp.sec == 10
                    and message.header.stamp.nanosec == 300_000_000
                ]

            wait_until(lambda: bool(rejected_health()))
            rejected_values = _status_values(rejected_health()[-1])
            assert rejected_values["deskew_status"] == "insufficient_imu_coverage"
            assert (
                rejected_values["rejection_reason"]
                == "deskew_insufficient_imu_coverage"
            )
            assert rejected_values["deskew_imu_wait_timed_out"] == "true"
            assert rejected_values["accepted"] == "false"

            quarantine_drain_deadline = time.monotonic() + 0.2
            while time.monotonic() < quarantine_drain_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.02)
            assert len(odom_messages) == odom_count_after_valid_scan
            latest_map_points = list(
                point_cloud2.read_points(
                    global_map_messages[-1],
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            )
            assert len(latest_map_points) == map_point_count_after_valid_scan

            # Epoch-scale doubles cannot represent every nanosecond. A scan
            # exactly at the configured duration may decode a few ULPs long,
            # which must not become a false out-of-range rejection.
            epoch_s = 1_700_000_000.0
            for sample_index in range(24):
                stamp_s = epoch_s - 0.01 + sample_index * 0.01
                imu = Imu()
                imu.header.stamp = _time_msg(stamp_s)
                imu.header.frame_id = "expected_imu"
                imu.angular_velocity.z = 1.0
                imu.linear_acceleration.z = 9.81
                imu_publisher.publish(imu)
            epoch_imu_end = _time_msg(epoch_s + 0.22)
            epoch_imu_end_ns = (
                epoch_imu_end.sec * 1_000_000_000 + epoch_imu_end.nanosec
            )
            wait_until(
                lambda: watermark_messages
                and watermark_messages[-1].data >= epoch_imu_end_ns
            )

            epoch_header = Header(
                stamp=_time_msg(epoch_s),
                frame_id="base_link",
            )
            epoch_points = []
            for point_index in range(200):
                ratio = point_index / 199.0
                epoch_points.append(
                    (
                        5.0 + 0.5 * ratio,
                        -2.0 + 4.0 * ratio,
                        0.25 * (point_index % 7),
                        int(round(ratio * 200_000_000)),
                    )
                )
            cloud_publisher.publish(
                point_cloud2.create_cloud(epoch_header, fields, epoch_points)
            )

            def epoch_health():
                return [
                    message
                    for message in health_messages
                    if message.header.stamp.sec == int(epoch_s)
                    and message.header.stamp.nanosec == 0
                ]

            wait_until(lambda: bool(epoch_health()))
            epoch_values = _status_values(epoch_health()[-1])
            assert epoch_values["deskew_status"] == "applied"
            assert abs(
                float(epoch_values["deskew_scan_duration_s"]) - 0.2
            ) < 1.0e-5
            expected_epoch_end_ns = int(epoch_s) * 1_000_000_000 + 200_000_000
            assert abs(
                int(epoch_values["deskew_required_imu_stamp_ns"])
                - expected_epoch_end_ns
            ) <= 5_000
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


def test_wheel_prediction_interpolates_at_cloud_stamp_not_latest_callback():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_wheel_time_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    wheel_topic = f"{prefix}/wheel_odom"
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_wheel_time_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_wheel_prediction:=true",
        "-p",
        "use_constant_velocity_prediction:=false",
        "-p",
        f"wheel_odom_topic:={wheel_topic}",
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
        "min_scan_points:=20",
        "-p",
        "scan_voxel_leaf_m:=0.01",
        "-p",
        "map_voxel_leaf_m:=0.01",
        "-p",
        "max_icp_iterations:=2",
        "-p",
        "wheel_odom_max_time_delta_s:=0.1",
        "-p",
        "wheel_odom_max_interpolation_gap_s:=0.2",
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
    test_node = Node(f"scan_to_map_wheel_time_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, qos)
    wheel_publisher = test_node.create_publisher(Odometry, wheel_topic, qos)
    health_messages = []
    test_node.create_subscription(
        DiagnosticArray,
        health_topic,
        health_messages.append,
        20,
    )

    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_wheel_time_runtime_"
    ) as temp_dir:
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

        def publish_wheel(stamp_s, x_m):
            message = Odometry()
            message.header.stamp = _time_msg(stamp_s)
            message.header.frame_id = "odom"
            message.child_frame_id = "base_link"
            message.pose.pose.position.x = x_m
            message.pose.pose.orientation.w = 1.0
            wheel_publisher.publish(message)

        def cloud(stamp_s):
            header = Header()
            header.stamp = _time_msg(stamp_s)
            header.frame_id = "base_link"
            points = [
                (
                    4.0 + 0.05 * (index % 20),
                    -1.0 + 0.05 * ((index // 20) % 10),
                    0.05 * (index % 7),
                )
                for index in range(200)
            ]
            return point_cloud2.create_cloud_xyz32(header, points)

        def health_at(stamp_s):
            target = _time_msg(stamp_s)
            return [
                message
                for message in health_messages
                if message.header.stamp.sec == target.sec
                and message.header.stamp.nanosec == target.nanosec
            ]

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and wheel_publisher.get_subscription_count() == 1
                and test_node.count_publishers(health_topic) == 1
            )

            publish_wheel(10.0, 0.0)
            publish_wheel(10.1, 1.0)
            publish_wheel(12.0, 100.0)
            wheel_drain_deadline = time.monotonic() + 0.2
            while time.monotonic() < wheel_drain_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.02)

            cloud_publisher.publish(cloud(10.0))
            wait_until(lambda: bool(health_at(10.0)))
            cloud_publisher.publish(cloud(10.05))
            wait_until(lambda: bool(health_at(10.05)))

            values = _status_values(health_at(10.05)[-1])
            assert values["latest_wheel_stamp_ns"] == "12000000000"
            assert values["wheel_prediction_status"] == "interpolated"
            assert values["wheel_prediction_query_stamp_ns"] == "10050000000"
            assert values["wheel_prediction_lower_stamp_ns"] == "10000000000"
            assert values["wheel_prediction_upper_stamp_ns"] == "10100000000"
            assert abs(
                float(values["wheel_prediction_max_bracket_delta_s"]) - 0.05
            ) < 1.0e-9
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


def test_imu_orientation_delta_is_used_as_rotation_prediction():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_imu_orientation_test/run_{token}"
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
        f"__node:=scan_to_map_imu_orientation_runtime_{token}",
        "-p",
        f"primary_cloud_topic:={cloud_topic}",
        "-p",
        "use_secondary_cloud:=false",
        "-p",
        "use_wheel_prediction:=false",
        "-p",
        "use_constant_velocity_prediction:=false",
        "-p",
        "use_tf_transforms:=false",
        "-p",
        "deskew_enabled:=false",
        "-p",
        "deskew_wait_for_imu_watermark:=true",
        "-p",
        "deskew_imu_wait_timeout_s:=1.0",
        "-p",
        "cloud_subscription_reliable:=true",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_range_m:=0.0",
        "-p",
        "min_scan_points:=20",
        "-p",
        "scan_voxel_leaf_m:=0.01",
        "-p",
        "map_voxel_leaf_m:=0.01",
        "-p",
        "max_icp_iterations:=10",
        "-p",
        "imu_rotation_prediction_enabled:=true",
        "-p",
        "imu_rotation_prediction_source:=orientation",
        "-p",
        "imu_rotation_prediction_max_dt_s:=0.5",
        "-p",
        "imu_orientation_prediction_max_time_delta_s:=0.03",
        "-p",
        "imu_roughness_detection_enabled:=true",
        "-p",
        "imu_roughness_window_s:=0.30",
        "-p",
        "imu_roughness_angular_velocity_threshold_radps:=100.0",
        "-p",
        "imu_roughness_angular_accel_threshold_radps2:=1000.0",
        "-p",
        "imu_roughness_accel_norm_delta_threshold_mps2:=1.0",
        "-p",
        "imu_roughness_linear_jerk_threshold_mps3:=100.0",
        "-p",
        f"imu_topic:={imu_topic}",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        f"imu_watermark_topic:={watermark_topic}",
        "-p",
        f"wheel_odom_topic:={prefix}/wheel_odom",
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
    test_node = Node(f"scan_to_map_imu_orientation_runtime_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=4,
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

    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_imu_orientation_runtime_"
    ) as temp_dir:
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

        def publish_imu(stamp_s, yaw_rad, linear_accel_x=0.0):
            message = Imu()
            message.header.stamp = _time_msg(stamp_s)
            message.orientation.z = math.sin(0.5 * yaw_rad)
            message.orientation.w = math.cos(0.5 * yaw_rad)
            message.orientation_covariance[0] = 0.01
            message.linear_acceleration.x = linear_accel_x
            message.linear_acceleration.z = 9.81
            imu_publisher.publish(message)

        def cloud(stamp_s, duration_s=None):
            header = Header()
            header.stamp = _time_msg(stamp_s)
            header.frame_id = "base_link"
            points = [
                (
                    4.0 + 0.04 * (index % 25),
                    -1.0 + 0.05 * ((index // 25) % 10),
                    0.03 * ((index * 7) % 13),
                )
                for index in range(250)
            ]
            if duration_s is not None:
                fields = [
                    PointField(
                        name="x", offset=0, datatype=PointField.FLOAT32, count=1
                    ),
                    PointField(
                        name="y", offset=4, datatype=PointField.FLOAT32, count=1
                    ),
                    PointField(
                        name="z", offset=8, datatype=PointField.FLOAT32, count=1
                    ),
                    PointField(
                        name="offset_time",
                        offset=12,
                        datatype=PointField.UINT32,
                        count=1,
                    ),
                ]
                timed_points = [
                    (*point, int(round(index / 249.0 * duration_s * 1.0e9)))
                    for index, point in enumerate(points)
                ]
                return point_cloud2.create_cloud(header, fields, timed_points)
            return point_cloud2.create_cloud_xyz32(header, points)

        def health_at(stamp_s):
            target = _time_msg(stamp_s)
            return [
                message
                for message in health_messages
                if message.header.stamp.sec == target.sec
                and message.header.stamp.nanosec == target.nanosec
            ]

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and imu_publisher.get_subscription_count() == 1
                and test_node.count_publishers(health_topic) == 1
                and test_node.count_publishers(watermark_topic) == 1
            )

            imu_samples = (
                (9.99, 0.0, 9_990_000_000),
                (10.01, 0.0, 10_010_000_000),
                (10.09, 0.1, 10_090_000_000),
                (10.11, 0.1, 10_110_000_000),
            )
            for stamp_s, yaw_rad, expected_watermark_ns in imu_samples:
                publish_imu(stamp_s, yaw_rad)
                wait_until(
                    lambda expected=expected_watermark_ns: watermark_messages
                    and watermark_messages[-1].data >= expected
                )

            cloud_publisher.publish(cloud(10.0))
            wait_until(lambda: bool(health_at(10.0)))
            cloud_publisher.publish(cloud(10.1))
            wait_until(lambda: bool(health_at(10.1)))

            values = _status_values(health_at(10.1)[-1])
            assert values["imu_rotation_prediction"] == "enabled"
            assert values["imu_rotation_prediction_source"] == "orientation"
            assert values["imu_rotation_prediction_used"] == "true"
            assert values["imu_rotation_prediction_status"] == "orientation_applied"
            assert (
                abs(
                    float(
                        values[
                            "imu_orientation_prediction_max_time_delta_s"
                        ]
                    )
                    - 0.03
                )
                < 1.0e-9
            )

            # Even with deskew disabled, a cloud-bound IMU prediction must
            # wait for its causal IMU prefix. Publishing the cloud first must
            # produce the same snapshot horizon as publishing IMU first.
            cloud_publisher.publish(cloud(10.2))
            no_dispatch_deadline = time.monotonic() + 0.10
            while time.monotonic() < no_dispatch_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.01)
            assert not health_at(10.2)

            publish_imu(10.19, 0.2)
            publish_imu(10.21, 0.2)
            wait_until(lambda: bool(health_at(10.2)))
            cloud_first_values = _status_values(health_at(10.2)[-1])
            assert cloud_first_values["deskew_status"] == "disabled"
            assert cloud_first_values["deskew_imu_watermark_waited"] == "true"
            assert int(cloud_first_values["deskew_required_imu_stamp_ns"]) == (
                10_200_000_000
            )
            assert int(cloud_first_values["imu_snapshot_end_stamp_ns"]) <= (
                10_200_000_000
            )
            assert int(cloud_first_values["imu_snapshot_sample_count"]) > 0

            # A deskew-disabled timed cloud still owns the IMU interval through
            # its final point. An impact after the header must be detected now,
            # not deferred until a hypothetical next cloud.
            cloud_publisher.publish(cloud(10.4, duration_s=0.1))
            no_timed_dispatch_deadline = time.monotonic() + 0.10
            while time.monotonic() < no_timed_dispatch_deadline:
                rclpy.spin_once(test_node, timeout_sec=0.01)
            assert not health_at(10.4)

            for stamp_s, accel_x in (
                (10.39, 0.0),
                (10.42, 0.0),
                (10.46, 5.0),
                (10.50, 0.0),
            ):
                publish_imu(stamp_s, 0.2, accel_x)
            wait_until(lambda: bool(health_at(10.4)))
            timed_values = _status_values(health_at(10.4)[-1])
            assert timed_values["deskew_status"] == "disabled"
            assert timed_values["deskew_imu_watermark_waited"] == "true"
            assert int(timed_values["deskew_required_imu_stamp_ns"]) == (
                10_500_000_000
            )
            assert int(timed_values["imu_snapshot_end_stamp_ns"]) <= (
                10_500_000_000
            )
            assert timed_values["imu_roughness_event"] == "true"

            # Overlapping clouds may have increasing headers but decreasing
            # point-time ends. They must not rewind the shock-policy clock.
            cloud_publisher.publish(cloud(10.45, duration_s=0.01))
            wait_until(lambda: bool(health_at(10.45)))
            overlap_values = _status_values(health_at(10.45)[-1])
            assert overlap_values["imu_roughness_reason"] == (
                "nonadvancing_scan_time_bounds"
            )
            assert overlap_values["imu_shock_time_reset"] == "false"
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
