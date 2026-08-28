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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


def _time_msg(seconds):
    from builtin_interfaces.msg import Time

    whole_seconds = int(seconds)
    return Time(
        sec=whole_seconds,
        nanosec=int(round((seconds - whole_seconds) * 1.0e9)),
    )


def _raw_hesai_cloud(frame_id, start_s, max_ring, point_count=8):
    cloud = PointCloud2()
    cloud.header.stamp = _time_msg(start_s)
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = point_count
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name="intensity",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(name="ring", offset=16, datatype=PointField.UINT16, count=1),
        PointField(
            name="timestamp",
            offset=18,
            datatype=PointField.FLOAT64,
            count=1,
        ),
    ]
    cloud.is_bigendian = False
    cloud.point_step = 26
    cloud.row_step = cloud.point_step * cloud.width
    cloud.data = bytearray(cloud.row_step)
    cloud.is_dense = True
    for index in range(point_count):
        ratio = index / (point_count - 1)
        ring = max_ring if index == point_count - 1 else index
        struct.pack_into(
            "<ffffHd",
            cloud.data,
            index * cloud.point_step,
            5.0 + ratio,
            -1.0 + 2.0 * ratio,
            0.1 * (index % 3),
            20.0 + index,
            ring,
            start_s + 0.1 * ratio,
        )
    return cloud


def _field_map(cloud):
    return {
        field.name: (field.offset, field.datatype, field.count)
        for field in cloud.fields
    }


def _status_values(message):
    return {item.key: item.value for item in message.status[0].values}


def test_three_hesai_streams_are_normalized_and_model_checked():
    token = uuid.uuid4().hex[:8]
    prefix = f"/hesai_ingest_test/run_{token}"
    sensor_names = ("front", "rear", "center")
    input_topics = {name: f"{prefix}/{name}/raw" for name in sensor_names}
    output_topics = {name: f"{prefix}/{name}/timed" for name in sensor_names}
    health_topic = f"{prefix}/health"
    command = [
        "ros2",
        "run",
        "slam_3d_lidar_ingest",
        "hesai_pointcloud_ingest_node",
        "--ros-args",
        "-r",
        f"__node:=hesai_ingest_runtime_{token}",
        "-p",
        f"front.input_topic:={input_topics['front']}",
        "-p",
        f"rear.input_topic:={input_topics['rear']}",
        "-p",
        f"center.input_topic:={input_topics['center']}",
        "-p",
        f"front.output_topic:={output_topics['front']}",
        "-p",
        f"rear.output_topic:={output_topics['rear']}",
        "-p",
        f"center.output_topic:={output_topics['center']}",
        "-p",
        f"health_topic:={health_topic}",
        "-p",
        "input_reliable:=true",
        "-p",
        "output_reliable:=true",
        "-p",
        "reject_clock_misalignment:=false",
        "-p",
        "front.expected_input_frame_id:=raw_front",
        "-p",
        "front.min_points:=1",
        "-p",
        "front.clock_policy:=calibrated_sensor_time",
        "-p",
        "front.time_scale:=1.5",
        "-p",
        "front.anchor_input_header_ns:=100000000000",
        "-p",
        "front.anchor_output_start_ns:=200000000000",
        "-p",
        "front.calibration_valid_min_input_header_ns:=99000000000",
        "-p",
        "front.calibration_valid_max_input_header_ns:=101000000000",
        "-p",
        "front.max_output_stamp_step_s:=0.5",
        "-p",
        "front.calibration_fit_rmse_ms:=1.0",
        "-p",
        "front.calibration_fit_max_abs_ms:=2.0",
        "-p",
        "front.calibration_id:=runtime_test_fit",
        "-p",
        "rear.expected_input_frame_id:=raw_rear",
        "-p",
        "rear.min_points:=1",
        "-p",
        "center.expected_input_frame_id:=raw_center",
        "-p",
        "center.min_points:=1",
    ]

    client_log_directory = tempfile.TemporaryDirectory(
        prefix="hesai_ingest_runtime_client_"
    )
    os.environ["ROS_LOG_DIR"] = client_log_directory.name
    rclpy.init()
    test_node = Node(f"hesai_ingest_runtime_client_{token}")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    publishers = {
        name: test_node.create_publisher(PointCloud2, input_topics[name], qos)
        for name in sensor_names
    }
    outputs = {name: [] for name in sensor_names}
    subscriptions = [
        test_node.create_subscription(
            PointCloud2,
            output_topics[name],
            outputs[name].append,
            qos,
        )
        for name in sensor_names
    ]
    health_messages = []
    subscriptions.append(
        test_node.create_subscription(
            DiagnosticArray,
            health_topic,
            health_messages.append,
            20,
        )
    )

    with tempfile.TemporaryDirectory(prefix="hesai_ingest_runtime_") as temp_dir:
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
                        f"hesai ingest exited with {process.returncode}:\n{output}"
                    )
                rclpy.spin_once(test_node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError("timed out waiting for ROS runtime condition")

        try:
            wait_until(
                lambda: all(
                    publisher.get_subscription_count() == 1
                    for publisher in publishers.values()
                )
                and test_node.count_publishers(health_topic) == 1
            )

            publishers["front"].publish(
                _raw_hesai_cloud("raw_front", 100.0, 127)
            )
            publishers["rear"].publish(
                _raw_hesai_cloud("raw_rear", 100.002, 127)
            )
            publishers["center"].publish(
                _raw_hesai_cloud("raw_center", 100.001, 31)
            )
            wait_until(lambda: all(outputs[name] for name in sensor_names))

            expected_fields = {
                "x": (0, PointField.FLOAT32, 1),
                "y": (4, PointField.FLOAT32, 1),
                "z": (8, PointField.FLOAT32, 1),
                "intensity": (12, PointField.FLOAT32, 1),
                "offset_time": (16, PointField.UINT32, 1),
                "ring": (20, PointField.UINT16, 1),
                "sensor_id": (22, PointField.UINT8, 1),
            }
            expected_frames = {
                "front": "raw_front",
                "rear": "raw_rear",
                "center": "raw_center",
            }
            expected_ids = {"front": 0, "rear": 1, "center": 2}
            for name in sensor_names:
                cloud = outputs[name][-1]
                assert cloud.point_step == 24
                assert cloud.width == 8
                assert _field_map(cloud) == expected_fields
                assert cloud.header.frame_id == expected_frames[name]
                first = struct.unpack_from("<ffffIHBx", cloud.data, 0)
                last = struct.unpack_from(
                    "<ffffIHBx",
                    cloud.data,
                    (cloud.width - 1) * cloud.point_step,
                )
                assert first[4] == 0
                expected_last_offset_ns = (
                    150_000_000 if name == "front" else 100_000_000
                )
                assert abs(last[4] - expected_last_offset_ns) <= 100
                assert last[5] == (31 if name == "center" else 127)
                assert first[6] == expected_ids[name]

            first_front_stamp = outputs["front"][-1].header.stamp
            first_front_stamp_ns = (
                first_front_stamp.sec * 1_000_000_000
                + first_front_stamp.nanosec
            )
            assert first_front_stamp_ns == 200_000_000_000
            time.sleep(0.15)
            publishers["front"].publish(
                _raw_hesai_cloud("raw_front", 100.2, 127)
            )
            wait_until(lambda: len(outputs["front"]) >= 2)
            second_front_stamp = outputs["front"][-1].header.stamp
            second_front_stamp_ns = (
                second_front_stamp.sec * 1_000_000_000
                + second_front_stamp.nanosec
            )
            assert second_front_stamp_ns - first_front_stamp_ns == 300_000_000

            def front_calibration_diagnostic_seen():
                return any(
                    message.status
                    and message.status[0].name.endswith("/front")
                    and _status_values(message).get("clock_policy")
                    == "calibrated_sensor_time"
                    and _status_values(message).get("time_anchor_initialized")
                    == "true"
                    and _status_values(message).get("time_anchor_source")
                    == "configured"
                    and _status_values(message).get("calibration_id")
                    == "runtime_test_fit"
                    and float(
                        _status_values(message).get(
                            "anchor_receive_residual_ms", "-1"
                        )
                    )
                    != -1.0
                    for message in health_messages
                )

            wait_until(front_calibration_diagnostic_seen)

            prior_front_count = len(outputs["front"])
            publishers["front"].publish(
                _raw_hesai_cloud("raw_front", 101.2, 127)
            )

            def front_calibration_range_rejection_seen():
                return any(
                    message.status
                    and message.status[0].name.endswith("/front")
                    and message.status[0].message
                    == "clock_calibration_out_of_range"
                    for message in health_messages
                )

            wait_until(front_calibration_range_rejection_seen)
            assert len(outputs["front"]) == prior_front_count

            publishers["front"].publish(
                _raw_hesai_cloud("raw_front", 100.9, 127)
            )

            def front_output_jump_rejection_seen():
                return any(
                    message.status
                    and message.status[0].name.endswith("/front")
                    and message.status[0].message == "output_timestamp_jump"
                    for message in health_messages
                )

            wait_until(front_output_jump_rejection_seen)
            assert len(outputs["front"]) == prior_front_count

            prior_center_count = len(outputs["center"])
            publishers["center"].publish(
                _raw_hesai_cloud("raw_center", 100.2, 32)
            )

            def center_ring_rejection_seen():
                return any(
                    message.status
                    and message.status[0].name.endswith("/center")
                    and message.status[0].message == "ring_out_of_range"
                    and _status_values(message).get("configured_max_ring") == "31"
                    for message in health_messages
                )

            wait_until(center_ring_rejection_seen)
            assert len(outputs["center"]) == prior_center_count
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
