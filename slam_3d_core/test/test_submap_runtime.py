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
from visualization_msgs.msg import MarkerArray


def _time_msg(seconds: float):
    from builtin_interfaces.msg import Time

    whole = int(seconds)
    return Time(sec=whole, nanosec=int(round((seconds - whole) * 1.0e9)))


def _status_values(message: DiagnosticArray):
    return {item.key: item.value for item in message.status[0].values}


def _structured_cloud(stamp_s: float):
    header = Header(stamp=_time_msg(stamp_s), frame_id="base_link")
    points = []
    for x_index in range(20):
        for y_index in range(10):
            x = 2.0 + 0.15 * x_index
            y = -1.0 + 0.2 * y_index
            points.append((x, y, 0.05 * ((x_index + y_index) % 5)))
            points.append((x, -1.5, 0.1 * y_index))
    return point_cloud2.create_cloud_xyz32(header, points)


def test_completed_overlapping_submap_is_reported_after_second_keyframe():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_submap_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    health_topic = f"{prefix}/health"
    marker_topic = f"{prefix}/pose_graph_markers"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_submap_runtime_{token}",
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
        "deskew_enabled:=false",
        "-p",
        "crop_box_enabled:=false",
        "-p",
        "min_range_m:=0.0",
        "-p",
        "min_scan_points:=100",
        "-p",
        "scan_voxel_leaf_m:=0.01",
        "-p",
        "map_voxel_leaf_m:=0.02",
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
        "surfel_min_correspondences:=40",
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
        "fitness_score_threshold:=0.5",
        "-p",
        "min_registration_inlier_ratio:=0.05",
        "-p",
        "max_registration_inlier_rmse_m:=1.0",
        "-p",
        "mapping_spatial_mode:=se3",
        "-p",
        "keyframe_translation_m:=0.0",
        "-p",
        "keyframe_rotation_deg:=180.0",
        "-p",
        "keyframe_rotation_3d_deg:=180.0",
        "-p",
        "submap_enabled:=true",
        "-p",
        "submap_max_keyframes:=2",
        "-p",
        "submap_max_travel_m:=0.0",
        "-p",
        "submap_overlap_keyframes:=1",
        "-p",
        "max_local_submaps:=2",
        "-p",
        "place_recognition_enabled:=true",
        "-p",
        "place_recognition_use_submaps:=true",
        "-p",
        "place_recognition_periodic_enabled:=false",
        "-p",
        "place_recognition_mapping_cloud_refinement_enabled:=true",
        "-p",
        "place_recognition_mapping_refinement_min_observable_rank:=1",
        "-p",
        "place_recognition_mapping_refinement_max_hessian_condition:=100000000.0",
        "-p",
        "place_recognition_mapping_refinement_min_target_keyframe_observations:=2",
        "-p",
        "place_recognition_mapping_refinement_target_confirmation_voxel_m:=0.1",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        f"pose_graph_marker_topic:={marker_topic}",
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
    test_node = Node(f"scan_to_map_submap_runtime_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=4,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, cloud_qos)
    health_messages = []
    marker_messages = []
    test_node.create_subscription(DiagnosticArray, health_topic, health_messages.append, 20)
    marker_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    test_node.create_subscription(
        MarkerArray,
        marker_topic,
        marker_messages.append,
        marker_qos,
    )

    with tempfile.TemporaryDirectory(prefix="slam_3d_submap_runtime_") as temp_dir:
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
                and test_node.count_publishers(health_topic) == 1
            )
            cloud_publisher.publish(_structured_cloud(1.0))
            wait_until(lambda: any(message.header.stamp.sec == 1 for message in health_messages))
            first = next(
                message for message in reversed(health_messages) if message.header.stamp.sec == 1
            )
            first_values = _status_values(first)
            assert first_values["mapping_spatial_mode"] == "se3"
            assert first_values["submap_enabled"] == "true"
            assert first_values["keyframes"] == "1"
            assert first_values["submaps"] == "1"
            assert first_values["completed_submaps"] == "0"
            assert first_values["active_submap_keyframes"] == "1"

            cloud_publisher.publish(_structured_cloud(2.0))
            wait_until(lambda: any(message.header.stamp.sec == 2 for message in health_messages))
            second = next(
                message for message in reversed(health_messages) if message.header.stamp.sec == 2
            )
            second_values = _status_values(second)
            assert second_values["accepted"] == "true"
            assert second_values["keyframe_added"] == "true"
            assert second_values["keyframes"] == "2"
            assert second_values["submaps"] == "2"
            assert second_values["completed_submaps"] == "1"
            assert second_values["frozen_completed_submaps"] == "1"
            frozen_input_points = int(
                second_values["frozen_submap_mapping_input_points"]
            )
            frozen_confirmed_points = int(
                second_values["frozen_submap_mapping_confirmed_points"]
            )
            assert 0 < frozen_confirmed_points <= frozen_input_points
            assert second_values["active_submap_keyframes"] == "1"
            assert int(second_values["active_submap_points"]) >= 100
            wait_until(
                lambda: any(
                    marker.ns == "submap_anchors"
                    for message in marker_messages
                    for marker in message.markers
                )
            )
            submap_marker = next(
                marker
                for marker in marker_messages[-1].markers
                if marker.ns == "submap_anchors"
            )
            assert len(submap_marker.points) == 2
            assert len(submap_marker.colors) == 2
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
