import os
import signal
import subprocess
import time
import uuid

import numpy as np
import pytest
import rclpy
import yaml
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger


@pytest.mark.parametrize('terrain_enabled', [False, True])
def test_export_retains_mapping_views_before_classification(tmp_path, terrain_enabled):
    token = uuid.uuid4().hex[:8]
    name = f'archive_test_{token}'
    prefix = f'/{name}'
    output = tmp_path / 'archive'
    parameters = {
        'primary_cloud_topic': f'{prefix}/points',
        'secondary_cloud_topic': f'{prefix}/ground',
        'use_secondary_cloud': True,
        'secondary_cloud_registration_only': True,
        'use_tertiary_cloud': False,
        'cloud_subscription_reliable': True,
        'use_tf_transforms': False,
        'deskew_enabled': False,
        'use_wheel_prediction': False,
        'inter_lidar_motion_compensation_enabled': False,
        'terrain_evidence_enabled': terrain_enabled,
        'crop_box_enabled': False,
        'min_scan_points': 20,
        'map_voxel_leaf_m': 0.01,
        'scan_voxel_leaf_m': 0.01,
        'keyframe_archive_voxel_leaf_m': 0.01,
        'keyframe_archive_directory': str(output),
        'health_topic': f'{prefix}/health',
    }
    command = ['ros2', 'run', 'slam_3d_core', 'scan_to_map_icp_node',
               '--ros-args', '-r', f'__node:={name}']
    for key, value in parameters.items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        command.extend(['-p', f'{key}:={value}'])
    environment = os.environ.copy()
    environment['ROS_LOG_DIR'] = str(tmp_path / 'ros_log')
    rclpy.init()
    node = Node(f'archive_client_{token}')
    publisher = node.create_publisher(PointCloud2, f'{prefix}/points', 10)
    ground = node.create_publisher(PointCloud2, f'{prefix}/ground', 10)
    statuses = []
    subscription = node.create_subscription(
        DiagnosticArray, f'{prefix}/health', statuses.append, 10)
    client = node.create_client(Trigger, f'{prefix}/export_keyframes')
    with (tmp_path / 'node.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   env=environment, start_new_session=True)

        def wait_for(predicate):
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                assert process.poll() is None, (tmp_path / 'node.log').read_text()
                rclpy.spin_once(node, timeout_sec=0.05)
                if predicate():
                    return
            raise AssertionError('timed out waiting for archive runtime condition')

        try:
            wait_for(lambda: publisher.get_subscription_count() == 1
                     and ground.get_subscription_count() == 1
                     and client.service_is_ready())
            empty = client.call_async(Trigger.Request())
            wait_for(empty.done)
            assert not empty.result().success
            assert not output.exists()
            header = Header(frame_id='base_link')
            header.stamp.sec = 50
            points = [(3.0 + 0.1 * (i % 20), -1.0 + 0.1 * (i // 20), 1.5)
                      for i in range(120)]
            ground.publish(point_cloud2.create_cloud_xyz32(
                header, [(x, y, 0.0) for x, y, _ in points]))
            publisher.publish(point_cloud2.create_cloud_xyz32(header, points))
            wait_for(lambda: bool(statuses))
            exported = client.call_async(Trigger.Request())
            wait_for(exported.done)
            assert exported.result().success, exported.result().message
            manifest = yaml.safe_load((output / 'manifest.yaml').read_text())
            assert manifest['complete_session'] is True
            assert manifest['base_frame'] == 'base_link'
            assert len(manifest['keyframes']) == 1
            frame = manifest['keyframes'][0]
            assert frame['stamp_ns'] == 50_000_000_000
            assert frame['id'] == 0
            assert len(frame['scans']) == 1  # registration-only ground is excluded.
            assert frame['scans'][0]['origin_base'] == [0.0, 0.0, 0.0]
            payload = (output / frame['scans'][0]['pcd']).read_bytes()
            xyz = np.frombuffer(payload.split(b'DATA binary\n', 1)[1],
                                dtype='<f4').reshape(-1, 3)
            assert len(xyz) > 20
            assert np.allclose(xyz[:, 2], 1.5)
            values = {item.key: item.value for item in statuses[-1].status[0].values}
            assert values['map_cleanup'] == 'external_postprocess'
            assert values['global_map_semantics'] == 'unfiltered_mapping_cloud'
            repeated = client.call_async(Trigger.Request())
            wait_for(repeated.done)
            assert not repeated.result().success
            assert 'exists' in repeated.result().message
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5.0)
            node.destroy_subscription(subscription)
            node.destroy_node()
            rclpy.shutdown()
