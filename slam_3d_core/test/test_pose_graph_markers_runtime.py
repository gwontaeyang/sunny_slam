import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


def _time_msg(seconds: float):
    from builtin_interfaces.msg import Time

    whole = int(seconds)
    nanoseconds = int(round((seconds - whole) * 1.0e9))
    return Time(sec=whole, nanosec=nanoseconds)


def test_pose_graph_markers_publish_nodes_edges_and_latest_orientation():
    token = uuid.uuid4().hex[:8]
    prefix = f"/slam_3d_core_pose_graph_marker_test/run_{token}"
    cloud_topic = f"{prefix}/cloud"
    marker_topic = f"{prefix}/pose_graph_markers"
    command = [
        "ros2",
        "run",
        "slam_3d_core",
        "scan_to_map_icp_node",
        "--ros-args",
        "-r",
        f"__node:=scan_to_map_pose_graph_marker_runtime_{token}",
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
        "max_icp_iterations:=10",
        "-p",
        "keyframe_translation_m:=-1.0",
        "-p",
        f"pose_graph_marker_topic:={marker_topic}",
        "-p",
        f"imu_topic:={prefix}/imu",
        "-p",
        f"imu_watermark_topic:={prefix}/imu_watermark",
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
        "-p",
        f"health_topic:={prefix}/health",
    ]

    rclpy.init()
    test_node = Node(f"scan_to_map_pose_graph_marker_client_{token}")
    cloud_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=4,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    marker_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    cloud_publisher = test_node.create_publisher(PointCloud2, cloud_topic, cloud_qos)
    marker_messages = []
    test_node.create_subscription(
        MarkerArray,
        marker_topic,
        marker_messages.append,
        marker_qos,
    )

    with tempfile.TemporaryDirectory(
        prefix="slam_3d_core_pose_graph_marker_runtime_"
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

        def wait_until(predicate, timeout_s=12.0):
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

        def cloud(stamp_s):
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
            return point_cloud2.create_cloud_xyz32(header, points)

        try:
            wait_until(
                lambda: cloud_publisher.get_subscription_count() == 1
                and test_node.count_publishers(marker_topic) == 1
            )
            cloud_publisher.publish(cloud(40.0))
            wait_until(
                lambda: marker_messages
                and len(marker_messages[-1].markers[0].points) == 1
            )
            cloud_publisher.publish(cloud(40.1))
            wait_until(
                lambda: marker_messages
                and len(marker_messages[-1].markers[0].points) == 2
            )

            markers = {marker.ns: marker for marker in marker_messages[-1].markers}
            assert set(markers) == {
                "pose_graph_nodes",
                "pose_graph_odometry_edges",
                "pose_graph_loop_edges",
                "pose_graph_latest_pose",
            }

            nodes = markers["pose_graph_nodes"]
            odometry_edges = markers["pose_graph_odometry_edges"]
            loop_edges = markers["pose_graph_loop_edges"]
            latest_pose = markers["pose_graph_latest_pose"]

            for marker in markers.values():
                assert marker.header.frame_id == "map"
                assert marker.header.stamp == _time_msg(40.1)
                assert marker.action == Marker.ADD

            assert nodes.type == Marker.SPHERE_LIST
            assert len(nodes.points) == 2
            assert len(nodes.colors) == 2
            assert nodes.colors[0] != nodes.colors[1]

            assert odometry_edges.type == Marker.LINE_LIST
            assert len(odometry_edges.points) == 2
            assert loop_edges.type == Marker.LINE_LIST
            assert len(loop_edges.points) == 0
            assert loop_edges.scale.x > odometry_edges.scale.x
            assert loop_edges.color != odometry_edges.color

            assert latest_pose.type == Marker.ARROW
            assert latest_pose.scale.x > latest_pose.scale.y
            assert latest_pose.pose.orientation.w != 0.0
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
