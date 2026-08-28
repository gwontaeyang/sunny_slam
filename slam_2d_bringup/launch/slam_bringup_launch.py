#!/usr/bin/env python3

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, LogInfo,
                            OpaqueFunction, RegisterEventHandler)
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def parse_launch_bool(value, name):
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    raise RuntimeError(f'{name} must be a boolean value')


def mapper_parameter_overrides(context):
    overrides = {
        'use_lifecycle_manager': parse_launch_bool(
            LaunchConfiguration('use_lifecycle_manager').perform(context),
            'use_lifecycle_manager'),
    }
    optional_values = (
        ('use_sim_time', parse_launch_bool),
        ('throttle_scans', int),
        ('resolution', float),
    )
    for name, parser in optional_values:
        raw_value = LaunchConfiguration(name).perform(context).strip()
        if not raw_value:
            continue
        try:
            value = parser(raw_value, name) if parser is parse_launch_bool \
                else parser(raw_value)
        except ValueError as exc:
            raise RuntimeError(f'{name} has an invalid value: {raw_value}') \
                from exc
        if name == 'throttle_scans' and value <= 0:
            raise RuntimeError('throttle_scans must be positive')
        if name == 'resolution' and (
                not math.isfinite(value) or value <= 0.0):
            raise RuntimeError('resolution must be finite and positive')
        overrides[name] = value
    return overrides


def launch_slam_toolbox(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    autostart = parse_launch_bool(
        LaunchConfiguration('autostart').perform(context),
        'autostart')
    overrides = mapper_parameter_overrides(context)
    use_roughness_innovation = parse_launch_bool(
        LaunchConfiguration(
            'use_roughness_innovation_slam_toolbox').perform(context),
        'use_roughness_innovation_slam_toolbox')
    roughness_enable_near_chain_gate = parse_launch_bool(
        LaunchConfiguration(
            'roughness_innovation_enable_near_chain_gate').perform(context),
        'roughness_innovation_enable_near_chain_gate')
    roughness_enable_spatial_memory_gate = parse_launch_bool(
        LaunchConfiguration(
            'roughness_innovation_enable_spatial_memory_gate').perform(context),
        'roughness_innovation_enable_spatial_memory_gate')
    roughness_spatial_memory_use_pose_graph_anchors = parse_launch_bool(
        LaunchConfiguration(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors',
            default='false'
        ).perform(context),
        'roughness_innovation_spatial_memory_use_pose_graph_anchors')
    roughness_enable_spatial_passage_hysteresis = parse_launch_bool(
        LaunchConfiguration(
            'roughness_innovation_enable_spatial_passage_hysteresis'
        ).perform(context),
        'roughness_innovation_enable_spatial_passage_hysteresis')
    if (
            roughness_enable_near_chain_gate
            or roughness_enable_spatial_memory_gate
            or roughness_spatial_memory_use_pose_graph_anchors
            or roughness_enable_spatial_passage_hysteresis
            ) and not use_roughness_innovation:
        raise RuntimeError(
            'roughness innovation near-chain, spatial memory, and passage '
            'features require use_roughness_innovation_slam_toolbox')

    if use_roughness_innovation:
        if not autostart:
            raise RuntimeError(
                'autostart must be true for the roughness innovation '
                'slam_toolbox frontend')
        if overrides['use_lifecycle_manager']:
            raise RuntimeError(
                'the roughness innovation slam_toolbox frontend '
                'self-configures and cannot use an external lifecycle manager')
        roughness_params_file = LaunchConfiguration(
            'roughness_innovation_params_file').perform(context)
        roughness_imu_topic = LaunchConfiguration(
            'roughness_innovation_imu_topic').perform(context)
        roughness_scan_qos_reliable = parse_launch_bool(
            LaunchConfiguration(
                'roughness_innovation_scan_qos_reliable',
                default='false').perform(context),
            'roughness_innovation_scan_qos_reliable')
        roughness_scan_qos_depth_raw = LaunchConfiguration(
            'roughness_innovation_scan_qos_depth',
            default='5').perform(context)
        try:
            roughness_scan_qos_depth = int(roughness_scan_qos_depth_raw)
        except ValueError as exc:
            raise RuntimeError(
                'roughness_innovation_scan_qos_depth has an invalid value: '
                f'{roughness_scan_qos_depth_raw}') from exc
        if roughness_scan_qos_depth <= 0:
            raise RuntimeError(
                'roughness_innovation_scan_qos_depth must be positive')
        cooldown_raw = LaunchConfiguration(
            'roughness_innovation_near_chain_loop_closing_cooldown_s'
        ).perform(context)
        try:
            roughness_near_chain_loop_closing_cooldown_s = float(cooldown_raw)
        except ValueError as exc:
            raise RuntimeError(
                'roughness_innovation_near_chain_loop_closing_cooldown_s '
                f'has an invalid value: {cooldown_raw}') from exc
        if (
                not math.isfinite(
                    roughness_near_chain_loop_closing_cooldown_s)
                or roughness_near_chain_loop_closing_cooldown_s < 0.0):
            raise RuntimeError(
                'roughness_innovation_near_chain_loop_closing_cooldown_s '
                'must be finite and non-negative')
        if (
                roughness_enable_spatial_memory_gate
                and not roughness_enable_near_chain_gate):
            raise RuntimeError(
                'roughness_innovation_enable_spatial_memory_gate requires '
                'roughness_innovation_enable_near_chain_gate')
        if (
                roughness_spatial_memory_use_pose_graph_anchors
                and not roughness_enable_spatial_memory_gate):
            raise RuntimeError(
                'roughness_innovation_spatial_memory_use_pose_graph_anchors '
                'requires roughness_innovation_enable_spatial_memory_gate')
        if (
                roughness_enable_spatial_passage_hysteresis
                and (
                    not roughness_enable_near_chain_gate
                    or not roughness_enable_spatial_memory_gate)):
            raise RuntimeError(
                'roughness_innovation_enable_spatial_passage_hysteresis '
                'requires roughness_innovation_enable_near_chain_gate and '
                'roughness_innovation_enable_spatial_memory_gate')
        spatial_passage_radius_raw = LaunchConfiguration(
            'roughness_innovation_spatial_passage_radius_m'
        ).perform(context)
        try:
            roughness_spatial_passage_radius_m = float(
                spatial_passage_radius_raw)
        except ValueError as exc:
            raise RuntimeError(
                'roughness_innovation_spatial_passage_radius_m '
                f'has an invalid value: {spatial_passage_radius_raw}') from exc
        if (
                not math.isfinite(roughness_spatial_passage_radius_m)
                or roughness_spatial_passage_radius_m < 0.0):
            raise RuntimeError(
                'roughness_innovation_spatial_passage_radius_m '
                'must be finite and non-negative')
        spatial_passage_max_age_raw = LaunchConfiguration(
            'roughness_innovation_spatial_passage_max_age_s'
        ).perform(context)
        try:
            roughness_spatial_passage_max_age_s = float(
                spatial_passage_max_age_raw)
        except ValueError as exc:
            raise RuntimeError(
                'roughness_innovation_spatial_passage_max_age_s '
                f'has an invalid value: {spatial_passage_max_age_raw}') from exc
        if (
                not math.isfinite(roughness_spatial_passage_max_age_s)
                or roughness_spatial_passage_max_age_s < 0.0):
            raise RuntimeError(
                'roughness_innovation_spatial_passage_max_age_s '
                'must be finite and non-negative')
        custom_overrides = {
            **overrides,
            'enable_roughness_innovation_gate': True,
            'roughness_imu_topic': roughness_imu_topic,
            'roughness_scan_qos_reliable': roughness_scan_qos_reliable,
            'roughness_scan_qos_depth': roughness_scan_qos_depth,
            'roughness_near_chain_loop_closing_cooldown_s':
                roughness_near_chain_loop_closing_cooldown_s,
            'roughness_enable_near_chain_gate':
                roughness_enable_near_chain_gate,
            'roughness_enable_spatial_memory_gate':
                roughness_enable_spatial_memory_gate,
            'roughness_spatial_memory_use_pose_graph_anchors':
                roughness_spatial_memory_use_pose_graph_anchors,
            'roughness_enable_spatial_passage_hysteresis':
                roughness_enable_spatial_passage_hysteresis,
            'roughness_spatial_passage_radius_m':
                roughness_spatial_passage_radius_m,
            'roughness_spatial_passage_max_age_s':
                roughness_spatial_passage_max_age_s,
        }
        return [
            Node(
                parameters=[
                    params_file,
                    roughness_params_file,
                    custom_overrides,
                ],
                package='slam_2d_core',
                executable='roughness_aware_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                namespace='')
        ]

    slam_toolbox_node = LifecycleNode(
        parameters=[params_file, overrides],
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='')
    actions = [slam_toolbox_node]
    if not autostart or overrides['use_lifecycle_manager']:
        return actions

    actions.append(
        EmitEvent(
            event=ChangeState(
                lifecycle_node_matcher=matches_action(slam_toolbox_node),
                transition_id=Transition.TRANSITION_CONFIGURE)))
    actions.append(
        RegisterEventHandler(
            OnStateTransition(
                target_lifecycle_node=slam_toolbox_node,
                start_state='configuring',
                goal_state='inactive',
                entities=[
                    LogInfo(
                        msg='[LifecycleLaunch] Slamtoolbox node is activating.'),
                    EmitEvent(
                        event=ChangeState(
                            lifecycle_node_matcher=matches_action(
                                slam_toolbox_node),
                            transition_id=Transition.TRANSITION_ACTIVATE)),
                ])))
    return actions


def generate_launch_description():
    # Get package directories
    slam_2d_bringup_dir = get_package_share_directory('slam_2d_bringup')
    config_file = os.path.join(
        slam_2d_bringup_dir, 'config', 'mapper_params.yaml')
    roughness_innovation_config_file = os.path.join(
        slam_2d_bringup_dir,
        'config',
        'slam_roughness_innovation_params.yaml')

    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true',
        description='Automatically startup the slamtoolbox. '
                    'Ignored when use_lifecycle_manager is true.')

    declare_use_lifecycle_manager = DeclareLaunchArgument(
        'use_lifecycle_manager', default_value='false',
        description='Enable bond connection during node activation')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='',
        description=(
            'Optional ROS simulation clock override. Empty preserves the '
            'params_file value.'))

    declare_throttle_scans = DeclareLaunchArgument(
        'throttle_scans', default_value='',
        description=(
            'Optional scan throttle override. Empty preserves the params_file '
            'value.'))

    declare_resolution = DeclareLaunchArgument(
        'resolution', default_value='',
        description=(
            'Optional map resolution override. Empty preserves the '
            'params_file value.'))

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=config_file,
        description='Path to slam_toolbox params yaml file')
    declare_use_roughness_innovation_slam_toolbox = DeclareLaunchArgument(
        'use_roughness_innovation_slam_toolbox',
        default_value='false',
        description='Use the IMU roughness-aware slam_toolbox frontend')
    declare_roughness_innovation_params_file = DeclareLaunchArgument(
        'roughness_innovation_params_file',
        default_value=roughness_innovation_config_file,
        description='Path to roughness innovation frontend params yaml')
    declare_roughness_innovation_imu_topic = DeclareLaunchArgument(
        'roughness_innovation_imu_topic',
        default_value='/imu/data',
        description='IMU topic used to classify roughness')
    declare_roughness_innovation_scan_qos_reliable = DeclareLaunchArgument(
        'roughness_innovation_scan_qos_reliable',
        default_value='false',
        description=(
            'Request reliable LaserScan QoS only when the publisher offers it'))
    declare_roughness_innovation_scan_qos_depth = DeclareLaunchArgument(
        'roughness_innovation_scan_qos_depth',
        default_value='5',
        description='LaserScan subscription history depth')
    declare_roughness_innovation_near_chain_loop_closing_cooldown_s = (
        DeclareLaunchArgument(
            'roughness_innovation_near_chain_loop_closing_cooldown_s',
            default_value='0.0',
            description=(
                'Keep loop closing suppressed after the near-chain threshold '
                'gate releases'))
    )
    declare_roughness_innovation_enable_near_chain_gate = (
        DeclareLaunchArgument(
            'roughness_innovation_enable_near_chain_gate',
            default_value='false',
            description='Apply the roughness-aware near-chain gate')
    )
    declare_roughness_innovation_enable_spatial_passage_hysteresis = (
        DeclareLaunchArgument(
            'roughness_innovation_enable_spatial_passage_hysteresis',
            default_value='false',
            description=(
                'Extend near-chain recovery while traversing a recent '
                'remembered rough zone'))
    )
    declare_roughness_innovation_enable_spatial_memory_gate = (
        DeclareLaunchArgument(
            'roughness_innovation_enable_spatial_memory_gate',
            default_value='false',
            description=(
                'Remember odometry locations of rough scans for spatial '
                'near-chain protection'))
    )
    declare_roughness_innovation_spatial_memory_use_pose_graph_anchors = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors',
            default_value='false',
            description=(
                'Anchor rough zones to corrected SLAM pose-graph scans so '
                'loop-closure corrections also move remembered zones'))
    )
    declare_roughness_innovation_spatial_passage_radius_m = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_passage_radius_m',
            default_value='0.55',
            description=(
                'Maximum odometry distance for spatial passage hysteresis'))
    )
    declare_roughness_innovation_spatial_passage_max_age_s = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_passage_max_age_s',
            default_value='3.5',
            description=(
                'Maximum age of a rough zone used for passage hysteresis'))
    )

    ld = LaunchDescription()

    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_lifecycle_manager)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_throttle_scans)
    ld.add_action(declare_resolution)
    ld.add_action(declare_params_file)
    ld.add_action(declare_use_roughness_innovation_slam_toolbox)
    ld.add_action(declare_roughness_innovation_params_file)
    ld.add_action(declare_roughness_innovation_imu_topic)
    ld.add_action(declare_roughness_innovation_scan_qos_reliable)
    ld.add_action(declare_roughness_innovation_scan_qos_depth)
    ld.add_action(
        declare_roughness_innovation_near_chain_loop_closing_cooldown_s)
    ld.add_action(declare_roughness_innovation_enable_near_chain_gate)
    ld.add_action(declare_roughness_innovation_enable_spatial_memory_gate)
    ld.add_action(
        declare_roughness_innovation_spatial_memory_use_pose_graph_anchors)
    ld.add_action(
        declare_roughness_innovation_enable_spatial_passage_hysteresis)
    ld.add_action(declare_roughness_innovation_spatial_passage_radius_m)
    ld.add_action(declare_roughness_innovation_spatial_passage_max_age_s)
    ld.add_action(OpaqueFunction(function=launch_slam_toolbox))

    return ld
