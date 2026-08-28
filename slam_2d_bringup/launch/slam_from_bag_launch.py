#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.actions import EmitEvent
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import AndSubstitution, IfElseSubstitution, LaunchConfiguration
from launch.substitutions import NotSubstitution, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from rclpy.expand_topic_name import expand_topic_name


SLAM_BAG_PLAY_TOPICS_WITH_CLOUD = [
    '/scan',
    '/scan_laserscan',
    '/tf',
    '/tf_static',
    '/odom',
    '/motor_driver_node/odom',
    '/imu/data',
    '/imu/data_raw',
    '/front/sick_scansegment_xd/imu',
    '/rear/sick_scansegment_xd/imu',
]
SLAM_BAG_PLAY_TOPICS_LASERSCAN_ONLY = [
    '/scan_laserscan',
    '/tf',
    '/tf_static',
    '/odom',
    '/motor_driver_node/odom',
    '/imu/data',
    '/imu/data_raw',
    '/front/sick_scansegment_xd/imu',
    '/rear/sick_scansegment_xd/imu',
]


def slam_toolbox_scan_topic_substitution(
        scan_topic,
        slam_scan_topic):
    is_configured = PythonExpression(
        ["'", slam_scan_topic, "' != ''"])
    return IfElseSubstitution(
        is_configured,
        if_value=slam_scan_topic,
        else_value=scan_topic,
    )


def validate_custom_slam_toolbox_selection(context):
    anchor_enabled = LaunchConfiguration(
        'use_roughness_anchor_slam_toolbox').perform(context)
    innovation_enabled = LaunchConfiguration(
        'use_roughness_innovation_slam_toolbox').perform(context)
    truthy = {'1', 'true', 'yes', 'on'}
    if (
            anchor_enabled.strip().lower() in truthy
            and innovation_enabled.strip().lower() in truthy):
        raise RuntimeError(
            'use_roughness_anchor_slam_toolbox and '
            'use_roughness_innovation_slam_toolbox are mutually exclusive')
    return []


def validate_roughness_innovation_playback_topics(context):
    truthy = {'1', 'true', 'yes', 'on'}
    enabled = LaunchConfiguration(
        'use_roughness_innovation_slam_toolbox').perform(context)
    restricted = LaunchConfiguration('play_slam_topics_only').perform(context)
    if (
            enabled.strip().lower() not in truthy
            or restricted.strip().lower() not in truthy):
        return []
    imu_topic = expand_topic_name(
        LaunchConfiguration('roughness_innovation_imu_topic').perform(context),
        'slam_toolbox',
        '/',
    )
    with_cloud = LaunchConfiguration(
        'start_pointcloud_projection').perform(context)
    allowed_topics = (
        SLAM_BAG_PLAY_TOPICS_WITH_CLOUD
        if with_cloud.strip().lower() in truthy
        else SLAM_BAG_PLAY_TOPICS_LASERSCAN_ONLY
    )
    if imu_topic not in allowed_topics:
        raise RuntimeError(
            f"roughness_innovation_imu_topic '{imu_topic}' is excluded by "
            "play_slam_topics_only; add it to the launch topic contract or "
            "set play_slam_topics_only:=false"
        )
    return []


def validate_roughness_innovation_dependencies(context):
    truthy = {'1', 'true', 'yes', 'on'}
    custom_enabled = LaunchConfiguration(
        'use_roughness_innovation_slam_toolbox'
    ).perform(context).strip().lower() in truthy
    near_chain_enabled = LaunchConfiguration(
        'roughness_innovation_enable_near_chain_gate'
    ).perform(context).strip().lower() in truthy
    spatial_memory_enabled = LaunchConfiguration(
        'roughness_innovation_enable_spatial_memory_gate'
    ).perform(context).strip().lower() in truthy
    pose_graph_anchors_raw = LaunchConfiguration(
        'roughness_innovation_spatial_memory_use_pose_graph_anchors',
        default='false'
    ).perform(context).strip().lower()
    if pose_graph_anchors_raw not in truthy | {'0', 'false', 'no', 'off'}:
        raise RuntimeError(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors '
            'must be a boolean')
    pose_graph_anchors_enabled = pose_graph_anchors_raw in truthy
    spatial_passage_enabled = LaunchConfiguration(
        'roughness_innovation_enable_spatial_passage_hysteresis'
    ).perform(context).strip().lower() in truthy
    if (
            near_chain_enabled
            or spatial_memory_enabled
            or pose_graph_anchors_enabled
            or spatial_passage_enabled
            ) and not custom_enabled:
        raise RuntimeError(
            'roughness innovation near-chain, spatial memory, and passage '
            'features require use_roughness_innovation_slam_toolbox')
    if spatial_memory_enabled and not near_chain_enabled:
        raise RuntimeError(
            'roughness_innovation_enable_spatial_memory_gate requires '
            'roughness_innovation_enable_near_chain_gate')
    if pose_graph_anchors_enabled and not spatial_memory_enabled:
        raise RuntimeError(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors '
            'requires roughness_innovation_enable_spatial_memory_gate')
    if spatial_passage_enabled and (
            not near_chain_enabled or not spatial_memory_enabled):
        raise RuntimeError(
            'roughness_innovation_enable_spatial_passage_hysteresis requires '
            'roughness_innovation_enable_near_chain_gate and '
            'roughness_innovation_enable_spatial_memory_gate')
    return []


def generate_launch_description():
    slam_2d_bringup_dir = get_package_share_directory('slam_2d_bringup')

    bag_file = LaunchConfiguration('bag_file')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    start_slam = LaunchConfiguration('start_slam', default='true')
    scan_topic = LaunchConfiguration('scan_topic', default='/scan_laserscan')
    slam_scan_topic = LaunchConfiguration(
        'slam_scan_topic',
        default='')
    cloud_topic = LaunchConfiguration('cloud_topic', default='/scan')
    start_pointcloud_projection = LaunchConfiguration(
        'start_pointcloud_projection',
        default='false')
    slam_toolbox_scan_topic = slam_toolbox_scan_topic_substitution(
        scan_topic,
        slam_scan_topic,
    )
    projection_min_z = LaunchConfiguration(
        'projection_min_z',
        default='-0.25')
    projection_max_z = LaunchConfiguration(
        'projection_max_z',
        default='0.25')
    projection_min_publish_period_s = LaunchConfiguration(
        'projection_min_publish_period_s',
        default='0.2')
    projection_initial_skip_count = LaunchConfiguration(
        'projection_initial_skip_count',
        default='5')
    projection_drop_nonmonotonic_stamps = LaunchConfiguration(
        'projection_drop_nonmonotonic_stamps',
        default='true')
    projection_max_stamp_gap_s = LaunchConfiguration(
        'projection_max_stamp_gap_s',
        default='2.0')
    projection_post_gap_skip_count = LaunchConfiguration(
        'projection_post_gap_skip_count',
        default='0')
    projection_post_gap_skip_duration_s = LaunchConfiguration(
        'projection_post_gap_skip_duration_s',
        default='4.0')
    projection_gate_by_tf_stamp = LaunchConfiguration(
        'projection_gate_by_tf_stamp',
        default='true')
    projection_tf_gate_child_frame = LaunchConfiguration(
        'projection_tf_gate_child_frame',
        default='base_link')
    projection_max_scan_tf_lag_s = LaunchConfiguration(
        'projection_max_scan_tf_lag_s',
        default='0.2')
    projection_enable_scan_quality_gate = LaunchConfiguration(
        'projection_enable_scan_quality_gate',
        default='false')
    projection_scan_quality_min_finite_point_ratio = LaunchConfiguration(
        'projection_scan_quality_min_finite_point_ratio',
        default='0.50')
    projection_scan_quality_min_height_kept_ratio = LaunchConfiguration(
        'projection_scan_quality_min_height_kept_ratio',
        default='0.05')
    projection_scan_quality_min_range_kept_ratio = LaunchConfiguration(
        'projection_scan_quality_min_range_kept_ratio',
        default='0.05')
    projection_scan_quality_min_valid_bin_ratio = LaunchConfiguration(
        'projection_scan_quality_min_valid_bin_ratio',
        default='0.03')
    projection_scan_quality_min_valid_bins = LaunchConfiguration(
        'projection_scan_quality_min_valid_bins',
        default='20')
    projection_scan_quality_max_empty_gap_ratio = LaunchConfiguration(
        'projection_scan_quality_max_empty_gap_ratio',
        default='0.95')
    projection_scan_quality_score_gate = LaunchConfiguration(
        'projection_scan_quality_score_gate',
        default='1.0')
    projection_diagnostics_period_s = LaunchConfiguration(
        'projection_diagnostics_period_s',
        default='1.0')
    use_static_map_odom = LaunchConfiguration('use_static_map_odom', default='false')
    bag_play_rate = LaunchConfiguration('bag_play_rate', default='1.0')
    bag_start_offset_s = LaunchConfiguration('bag_start_offset_s', default='0.0')
    play_slam_topics_only = LaunchConfiguration('play_slam_topics_only', default='true')
    use_roughness_anchor_slam_toolbox = LaunchConfiguration(
        'use_roughness_anchor_slam_toolbox',
        default='false')
    use_roughness_innovation_slam_toolbox = LaunchConfiguration(
        'use_roughness_innovation_slam_toolbox',
        default='false')
    roughness_innovation_mode = LaunchConfiguration(
        'roughness_innovation_mode',
        default='observe')
    roughness_innovation_imu_topic = LaunchConfiguration(
        'roughness_innovation_imu_topic',
        default='/imu/data')
    roughness_innovation_scan_qos_reliable = LaunchConfiguration(
        'roughness_innovation_scan_qos_reliable',
        default='false')
    roughness_innovation_scan_qos_depth = LaunchConfiguration(
        'roughness_innovation_scan_qos_depth',
        default='5')
    roughness_innovation_window_before_s = LaunchConfiguration(
        'roughness_innovation_window_before_s',
        default='0.10')
    roughness_innovation_window_after_s = LaunchConfiguration(
        'roughness_innovation_window_after_s',
        default='0.10')
    roughness_innovation_score_gate = LaunchConfiguration(
        'roughness_innovation_score_gate',
        default='2.75')
    roughness_innovation_min_rough_samples = LaunchConfiguration(
        'roughness_innovation_min_rough_samples',
        default='3')
    roughness_innovation_min_rough_duration_s = LaunchConfiguration(
        'roughness_innovation_min_rough_duration_s',
        default='0.02')
    roughness_innovation_translation_gate_m = LaunchConfiguration(
        'roughness_innovation_translation_gate_m',
        default='0.04')
    roughness_innovation_yaw_gate_deg = LaunchConfiguration(
        'roughness_innovation_yaw_gate_deg',
        default='3.0')
    roughness_innovation_enable_near_chain_gate = LaunchConfiguration(
        'roughness_innovation_enable_near_chain_gate',
        default='false')
    roughness_innovation_near_chain_recovery_duration_s = LaunchConfiguration(
        'roughness_innovation_near_chain_recovery_duration_s',
        default='2.0')
    roughness_innovation_near_chain_loop_closing_cooldown_s = (
        LaunchConfiguration(
            'roughness_innovation_near_chain_loop_closing_cooldown_s',
            default='0.0')
    )
    roughness_innovation_near_chain_min_rough_samples = LaunchConfiguration(
        'roughness_innovation_near_chain_min_rough_samples',
        default='1')
    roughness_innovation_near_chain_gated_min_response = LaunchConfiguration(
        'roughness_innovation_near_chain_gated_min_response',
        default='1.1')
    roughness_innovation_disable_loop_closing_during_gate = LaunchConfiguration(
        'roughness_innovation_disable_loop_closing_during_gate',
        default='false')
    roughness_innovation_enable_spatial_memory_gate = LaunchConfiguration(
        'roughness_innovation_enable_spatial_memory_gate',
        default='false')
    roughness_innovation_spatial_memory_use_pose_graph_anchors = (
        LaunchConfiguration(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors',
            default='false')
    )
    roughness_innovation_spatial_memory_radius_m = LaunchConfiguration(
        'roughness_innovation_spatial_memory_radius_m',
        default='0.6')
    roughness_innovation_spatial_memory_min_revisit_separation_s = (
        LaunchConfiguration(
            'roughness_innovation_spatial_memory_min_revisit_separation_s',
            default='20.0')
    )
    roughness_innovation_spatial_memory_min_heading_delta_deg = (
        LaunchConfiguration(
            'roughness_innovation_spatial_memory_min_heading_delta_deg',
            default='126.0')
    )
    roughness_innovation_spatial_memory_merge_radius_m = LaunchConfiguration(
        'roughness_innovation_spatial_memory_merge_radius_m',
        default='0.10')
    roughness_innovation_spatial_memory_max_zones = LaunchConfiguration(
        'roughness_innovation_spatial_memory_max_zones',
        default='200')
    roughness_innovation_enable_spatial_passage_hysteresis = (
        LaunchConfiguration(
            'roughness_innovation_enable_spatial_passage_hysteresis',
            default='false')
    )
    roughness_innovation_spatial_passage_radius_m = LaunchConfiguration(
        'roughness_innovation_spatial_passage_radius_m',
        default='0.55')
    roughness_innovation_spatial_passage_max_age_s = LaunchConfiguration(
        'roughness_innovation_spatial_passage_max_age_s',
        default='3.5')
    roughness_innovation_enable_post_confirmation = LaunchConfiguration(
        'roughness_innovation_enable_post_confirmation',
        default='true')
    roughness_innovation_post_confirmation_max_age_s = LaunchConfiguration(
        'roughness_innovation_post_confirmation_max_age_s',
        default='2.0')
    roughness_innovation_post_confirmation_min_base_scans = LaunchConfiguration(
        'roughness_innovation_post_confirmation_min_base_scans',
        default='8')
    roughness_innovation_post_confirmation_max_base_scans = LaunchConfiguration(
        'roughness_innovation_post_confirmation_max_base_scans',
        default='10')
    roughness_innovation_post_confirmation_min_response = LaunchConfiguration(
        'roughness_innovation_post_confirmation_min_response',
        default='0.30')
    roughness_innovation_post_confirmation_min_response_gain = (
        LaunchConfiguration(
            'roughness_innovation_post_confirmation_min_response_gain',
            default='0.05')
    )
    roughness_innovation_post_confirmation_min_covariance_improvement_ratio = (
        LaunchConfiguration(
            'roughness_innovation_post_confirmation_min_covariance_improvement_ratio',
            default='1.5')
    )
    roughness_innovation_post_confirmation_translation_disagreement_gate_m = (
        LaunchConfiguration(
            'roughness_innovation_post_confirmation_translation_disagreement_gate_m',
            default='0.03')
    )
    roughness_innovation_post_confirmation_yaw_disagreement_gate_deg = (
        LaunchConfiguration(
            'roughness_innovation_post_confirmation_yaw_disagreement_gate_deg',
            default='1.0')
    )
    roughness_innovation_experimental_reject_scan_stamps_ns = LaunchConfiguration(
        'roughness_innovation_experimental_reject_scan_stamps_ns',
        default='')
    roughness_anchor_constraints_csv = LaunchConfiguration(
        'roughness_anchor_constraints_csv',
        default='')
    roughness_anchor_event_a = LaunchConfiguration('roughness_anchor_event_a', default='-1')
    roughness_anchor_event_b = LaunchConfiguration('roughness_anchor_event_b', default='-1')
    roughness_anchor_max_stamp_dt_s = LaunchConfiguration(
        'roughness_anchor_max_stamp_dt_s',
        default='8.0')
    roughness_anchor_min_scan_iou = LaunchConfiguration(
        'roughness_anchor_min_scan_iou',
        default='0.15')
    roughness_anchor_max_scan_nn_p50_m = LaunchConfiguration(
        'roughness_anchor_max_scan_nn_p50_m',
        default='0.10')
    roughness_anchor_max_abs_scan_yaw_deg = LaunchConfiguration(
        'roughness_anchor_max_abs_scan_yaw_deg',
        default='5.0')
    roughness_anchor_max_input_odom_gap_m = LaunchConfiguration(
        'roughness_anchor_max_input_odom_gap_m',
        default='0.10')
    roughness_anchor_min_pair_residual_m = LaunchConfiguration(
        'roughness_anchor_min_pair_residual_m',
        default='0.05')
    roughness_anchor_max_graph_odom_gap_m = LaunchConfiguration(
        'roughness_anchor_max_graph_odom_gap_m',
        default='0.05')
    roughness_anchor_max_constraint_pairs = LaunchConfiguration(
        'roughness_anchor_max_constraint_pairs',
        default='8')
    roughness_anchor_require_complete_pair = LaunchConfiguration(
        'roughness_anchor_require_complete_pair',
        default='true')
    roughness_anchor_translation_sigma_m = LaunchConfiguration(
        'roughness_anchor_translation_sigma_m',
        default='0.10')
    roughness_anchor_yaw_sigma_rad = LaunchConfiguration(
        'roughness_anchor_yaw_sigma_rad',
        default='1.0')

    declare_bag_file = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to the bag file to play (leave empty to skip bag playback)'
    )
    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(
            slam_2d_bringup_dir,
            'config',
            'mapper_params.yaml',
        ),
        description='Path to the slam_toolbox params YAML file'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from bag file'
    )

    declare_start_slam = DeclareLaunchArgument(
        'start_slam',
        default_value='true',
        description='Start SLAM node'
    )

    declare_scan_topic = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan_laserscan',
        description='LaserScan topic for slam_toolbox'
    )

    declare_slam_scan_topic = DeclareLaunchArgument(
        'slam_scan_topic',
        default_value='',
        description=(
            'Optional slam_toolbox LaserScan override. When empty, use '
            'scan_topic.')
    )

    declare_cloud_topic = DeclareLaunchArgument(
        'cloud_topic',
        default_value='/scan',
        description='PointCloud2 topic to project into LaserScan during bag replay'
    )

    declare_start_pointcloud_projection = DeclareLaunchArgument(
        'start_pointcloud_projection',
        default_value='false',
        description='Start PointCloud2 to LaserScan projection for bags without LaserScan'
    )

    declare_projection_min_z = DeclareLaunchArgument(
        'projection_min_z',
        default_value='-0.25',
        description='Minimum z height retained by projected LaserScan height filtering'
    )

    declare_projection_max_z = DeclareLaunchArgument(
        'projection_max_z',
        default_value='0.25',
        description='Maximum z height retained by projected LaserScan height filtering'
    )

    declare_projection_min_publish_period_s = DeclareLaunchArgument(
        'projection_min_publish_period_s',
        default_value='0.2',
        description=(
            'Minimum ROS-stamp period between projected LaserScan messages '
            '(0.2 = 5 Hz, matching the default slam_toolbox minimum_time_interval)'
        )
    )

    declare_projection_initial_skip_count = DeclareLaunchArgument(
        'projection_initial_skip_count',
        default_value='5',
        description=(
            'Drop this many initial PointCloud2 inputs before publishing '
            'projected LaserScan'
        )
    )

    declare_projection_drop_nonmonotonic_stamps = DeclareLaunchArgument(
        'projection_drop_nonmonotonic_stamps',
        default_value='true',
        description='Drop projected replay clouds whose header stamp moves backward'
    )

    declare_projection_max_stamp_gap_s = DeclareLaunchArgument(
        'projection_max_stamp_gap_s',
        default_value='2.0',
        description='Treat larger PointCloud2 header stamp gaps as replay discontinuities'
    )

    declare_projection_post_gap_skip_count = DeclareLaunchArgument(
        'projection_post_gap_skip_count',
        default_value='0',
        description='Drop this many PointCloud2 inputs after a replay timestamp gap'
    )

    declare_projection_post_gap_skip_duration_s = DeclareLaunchArgument(
        'projection_post_gap_skip_duration_s',
        default_value='4.0',
        description=(
            'Drop projected replay clouds for this many stamp seconds after '
            'a timestamp gap'
        )
    )

    declare_projection_gate_by_tf_stamp = DeclareLaunchArgument(
        'projection_gate_by_tf_stamp',
        default_value='true',
        description='Drop projected replay clouds newer than the latest dynamic base TF'
    )

    declare_projection_tf_gate_child_frame = DeclareLaunchArgument(
        'projection_tf_gate_child_frame',
        default_value='base_link',
        description='Dynamic TF child frame used to gate projected replay clouds'
    )

    declare_projection_max_scan_tf_lag_s = DeclareLaunchArgument(
        'projection_max_scan_tf_lag_s',
        default_value='0.2',
        description='Allowed projected scan lead over the latest dynamic base TF stamp'
    )

    declare_projection_enable_scan_quality_gate = DeclareLaunchArgument(
        'projection_enable_scan_quality_gate',
        default_value='false',
        description='Suppress projected LaserScan updates when projected scan quality is too poor'
    )

    declare_projection_scan_quality_min_finite_point_ratio = DeclareLaunchArgument(
        'projection_scan_quality_min_finite_point_ratio',
        default_value='0.50',
        description='Minimum finite/raw PointCloud2 point ratio before scan-quality suppression'
    )

    declare_projection_scan_quality_min_height_kept_ratio = DeclareLaunchArgument(
        'projection_scan_quality_min_height_kept_ratio',
        default_value='0.05',
        description='Minimum height-filtered/finite point ratio before scan-quality suppression'
    )

    declare_projection_scan_quality_min_range_kept_ratio = DeclareLaunchArgument(
        'projection_scan_quality_min_range_kept_ratio',
        default_value='0.05',
        description=(
            'Minimum range-filtered/height-filtered point ratio before scan-quality suppression'
        )
    )

    declare_projection_scan_quality_min_valid_bin_ratio = DeclareLaunchArgument(
        'projection_scan_quality_min_valid_bin_ratio',
        default_value='0.03',
        description='Minimum valid projected LaserScan bin ratio before scan-quality suppression'
    )

    declare_projection_scan_quality_min_valid_bins = DeclareLaunchArgument(
        'projection_scan_quality_min_valid_bins',
        default_value='20',
        description='Minimum valid projected LaserScan bin count before scan-quality suppression'
    )

    declare_projection_scan_quality_max_empty_gap_ratio = DeclareLaunchArgument(
        'projection_scan_quality_max_empty_gap_ratio',
        default_value='0.95',
        description='Maximum contiguous empty-bin ratio allowed before scan-quality suppression'
    )

    declare_projection_scan_quality_score_gate = DeclareLaunchArgument(
        'projection_scan_quality_score_gate',
        default_value='1.0',
        description='Bad-scan score threshold for projected scan-quality suppression'
    )

    declare_projection_diagnostics_period_s = DeclareLaunchArgument(
        'projection_diagnostics_period_s',
        default_value='1.0',
        description='Period for projected scan diagnostic counter publication; 0 disables it'
    )

    declare_use_static_map_odom = DeclareLaunchArgument(
        'use_static_map_odom',
        default_value='false',
        description=(
            'Use static transform publisher for map->odom '
            '(workaround if slam_toolbox does not publish)'
        )
    )

    declare_bag_play_rate = DeclareLaunchArgument(
        'bag_play_rate',
        default_value='1.0',
        description=(
            'Bag file playback rate '
            '(1.0 = real-time, 0.5 = half speed, 2.0 = double speed)'
        )
    )

    declare_bag_start_offset_s = DeclareLaunchArgument(
        'bag_start_offset_s',
        default_value='0.0',
        description='Skip this many seconds from the start of the bag before playback'
    )

    declare_play_slam_topics_only = DeclareLaunchArgument(
        'play_slam_topics_only',
        default_value='true',
        description='Only replay SLAM-relevant topics from the bag'
    )

    declare_use_roughness_anchor_slam_toolbox = DeclareLaunchArgument(
        'use_roughness_anchor_slam_toolbox',
        default_value='false',
        description=(
            'Use experimental slam_toolbox node that treats roughness events as '
            'candidate graph anchors; keep false unless validated on the target bags'
        )
    )

    declare_use_roughness_innovation_slam_toolbox = DeclareLaunchArgument(
        'use_roughness_innovation_slam_toolbox',
        default_value='false',
        description=(
            'Use the experimental IMU-window plus scan-match innovation SLAM '
            'frontend; disabled until bag validation is complete'
        )
    )
    declare_roughness_innovation_mode = DeclareLaunchArgument(
        'roughness_innovation_mode',
        default_value='observe',
        description='Roughness innovation policy; online operation is observe-only'
    )
    declare_roughness_innovation_imu_topic = DeclareLaunchArgument(
        'roughness_innovation_imu_topic',
        default_value='/imu/data',
        description='IMU topic used by the roughness innovation frontend'
    )
    declare_roughness_innovation_scan_qos_reliable = DeclareLaunchArgument(
        'roughness_innovation_scan_qos_reliable',
        default_value='false',
        description=(
            'Request reliable LaserScan QoS only when the publisher offers it')
    )
    declare_roughness_innovation_scan_qos_depth = DeclareLaunchArgument(
        'roughness_innovation_scan_qos_depth',
        default_value='5',
        description='LaserScan subscription history depth'
    )
    declare_roughness_innovation_window_before_s = DeclareLaunchArgument(
        'roughness_innovation_window_before_s',
        default_value='0.10',
        description='IMU roughness window before each accepted LaserScan'
    )
    declare_roughness_innovation_window_after_s = DeclareLaunchArgument(
        'roughness_innovation_window_after_s',
        default_value='0.10',
        description='IMU roughness lookahead after each accepted LaserScan'
    )
    declare_roughness_innovation_score_gate = DeclareLaunchArgument(
        'roughness_innovation_score_gate',
        default_value='2.75',
        description='Minimum normalized IMU roughness score'
    )
    declare_roughness_innovation_min_rough_samples = DeclareLaunchArgument(
        'roughness_innovation_min_rough_samples',
        default_value='3',
        description='Minimum consecutive above-gate IMU samples'
    )
    declare_roughness_innovation_min_rough_duration_s = DeclareLaunchArgument(
        'roughness_innovation_min_rough_duration_s',
        default_value='0.02',
        description='Minimum consecutive above-gate IMU duration'
    )
    declare_roughness_innovation_translation_gate_m = DeclareLaunchArgument(
        'roughness_innovation_translation_gate_m',
        default_value='0.04',
        description='Minimum scan-match translation innovation in metres'
    )
    declare_roughness_innovation_yaw_gate_deg = DeclareLaunchArgument(
        'roughness_innovation_yaw_gate_deg',
        default_value='3.0',
        description='Minimum absolute scan-match yaw innovation in degrees'
    )
    declare_roughness_innovation_enable_near_chain_gate = DeclareLaunchArgument(
        'roughness_innovation_enable_near_chain_gate',
        default_value='false',
        description=(
            'Raise only the Karto near-chain acceptance threshold during an '
            'IMU roughness event and its recovery interval'
        )
    )
    declare_roughness_innovation_near_chain_recovery_duration_s = (
        DeclareLaunchArgument(
            'roughness_innovation_near_chain_recovery_duration_s',
            default_value='2.0',
            description='Scan-timestamp recovery hold after a roughness event'
        )
    )
    declare_roughness_innovation_near_chain_loop_closing_cooldown_s = (
        DeclareLaunchArgument(
            'roughness_innovation_near_chain_loop_closing_cooldown_s',
            default_value='0.0',
            description=(
                'Keep loop closing suppressed after the near-chain threshold '
                'gate releases'
            )
        )
    )
    declare_roughness_innovation_near_chain_min_rough_samples = (
        DeclareLaunchArgument(
            'roughness_innovation_near_chain_min_rough_samples',
            default_value='1',
            description='Minimum above-gate IMU samples for near-chain gating'
        )
    )
    declare_roughness_innovation_near_chain_gated_min_response = (
        DeclareLaunchArgument(
            'roughness_innovation_near_chain_gated_min_response',
            default_value='1.1',
            description=(
                'Temporary Karto LinkMatchMinimumResponseFine while the '
                'near-chain gate is active'
            )
        )
    )
    declare_roughness_innovation_disable_loop_closing_during_gate = (
        DeclareLaunchArgument(
            'roughness_innovation_disable_loop_closing_during_gate',
            default_value='false',
            description=(
                'Temporarily disable global Karto loop closure while the '
                'roughness near-chain gate is active'
            )
        )
    )
    declare_roughness_innovation_enable_spatial_memory_gate = (
        DeclareLaunchArgument(
            'roughness_innovation_enable_spatial_memory_gate',
            default_value='false',
            description=(
                'Gate scan matching when odometry revisits a remembered '
                'IMU-rough location'
            )
        )
    )
    declare_roughness_innovation_spatial_memory_use_pose_graph_anchors = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_use_pose_graph_anchors',
            default_value='false',
            description=(
                'Anchor rough zones to corrected SLAM pose-graph scans so '
                'loop-closure corrections also move remembered zones'
            )
        )
    )
    declare_roughness_innovation_spatial_memory_radius_m = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_radius_m',
            default_value='0.6',
            description='Maximum odometry distance to a remembered rough zone'
        )
    )
    declare_roughness_innovation_spatial_memory_min_revisit_separation_s = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_min_revisit_separation_s',
            default_value='20.0',
            description='Minimum elapsed scan time before a rough-zone revisit'
        )
    )
    declare_roughness_innovation_spatial_memory_min_heading_delta_deg = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_min_heading_delta_deg',
            default_value='126.0',
            description='Minimum heading change for a rough-zone revisit'
        )
    )
    declare_roughness_innovation_spatial_memory_merge_radius_m = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_merge_radius_m',
            default_value='0.10',
            description='Same-pass rough-zone sample merge radius'
        )
    )
    declare_roughness_innovation_spatial_memory_max_zones = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_memory_max_zones',
            default_value='200',
            description='Maximum session-local rough-zone samples'
        )
    )
    declare_roughness_innovation_enable_spatial_passage_hysteresis = (
        DeclareLaunchArgument(
            'roughness_innovation_enable_spatial_passage_hysteresis',
            default_value='false',
            description=(
                'Extend near-chain recovery while traversing a recent '
                'remembered rough zone'
            )
        )
    )
    declare_roughness_innovation_spatial_passage_radius_m = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_passage_radius_m',
            default_value='0.55',
            description=(
                'Maximum odometry distance for spatial passage hysteresis'
            )
        )
    )
    declare_roughness_innovation_spatial_passage_max_age_s = (
        DeclareLaunchArgument(
            'roughness_innovation_spatial_passage_max_age_s',
            default_value='3.5',
            description=(
                'Maximum age of a rough zone used for passage hysteresis'
            )
        )
    )
    declare_roughness_innovation_enable_post_confirmation = DeclareLaunchArgument(
        'roughness_innovation_enable_post_confirmation',
        default_value='true',
        description=(
            'Observe the first stable scan after a rough innovation against a '
            'frozen pre-event scan buffer'
        )
    )
    declare_roughness_innovation_post_confirmation_max_age_s = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_max_age_s',
            default_value='2.0',
            description='Maximum rough-to-stable confirmation interval'
        )
    )
    declare_roughness_innovation_post_confirmation_min_base_scans = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_min_base_scans',
            default_value='8',
            description='Minimum frozen pre-event scans required'
        )
    )
    declare_roughness_innovation_post_confirmation_max_base_scans = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_max_base_scans',
            default_value='10',
            description='Maximum frozen pre-event scans retained'
        )
    )
    declare_roughness_innovation_post_confirmation_min_response = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_min_response',
            default_value='0.30',
            description='Minimum post-event frozen-submap match response'
        )
    )
    declare_roughness_innovation_post_confirmation_min_response_gain = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_min_response_gain',
            default_value='0.05',
            description='Minimum post-event response improvement'
        )
    )
    declare_roughness_innovation_post_confirmation_min_covariance_improvement_ratio = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_min_covariance_improvement_ratio',
            default_value='1.5',
            description='Minimum rough-to-post covariance improvement ratio'
        )
    )
    declare_roughness_innovation_post_confirmation_translation_disagreement_gate_m = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_translation_disagreement_gate_m',
            default_value='0.03',
            description='Minimum map-to-odom translation disagreement'
        )
    )
    declare_roughness_innovation_post_confirmation_yaw_disagreement_gate_deg = (
        DeclareLaunchArgument(
            'roughness_innovation_post_confirmation_yaw_disagreement_gate_deg',
            default_value='1.0',
            description='Minimum map-to-odom yaw disagreement in degrees'
        )
    )
    declare_roughness_innovation_experimental_reject_scan_stamps_ns = (
        DeclareLaunchArgument(
            'roughness_innovation_experimental_reject_scan_stamps_ns',
            default_value='',
            description=(
                'Comma-separated exact LaserScan stamps for bounded bag-only '
                'leave-one-out experiments; never set during robot operation'
            )
        )
    )

    declare_roughness_anchor_constraints_csv = DeclareLaunchArgument(
        'roughness_anchor_constraints_csv',
        default_value='',
        description=(
            'CSV generated by roughness_anchor_scan_check.py; candidates still '
            'require scan and graph-odom gates before edge insertion'
        )
    )

    declare_roughness_anchor_event_a = DeclareLaunchArgument(
        'roughness_anchor_event_a',
        default_value='-1',
        description='Optional event_a filter for roughness-anchor constraints'
    )

    declare_roughness_anchor_event_b = DeclareLaunchArgument(
        'roughness_anchor_event_b',
        default_value='-1',
        description='Optional event_b filter for roughness-anchor constraints'
    )

    declare_roughness_anchor_max_stamp_dt_s = DeclareLaunchArgument(
        'roughness_anchor_max_stamp_dt_s',
        default_value='8.0',
        description='Maximum anchor timestamp to graph-node timestamp difference'
    )

    declare_roughness_anchor_min_scan_iou = DeclareLaunchArgument(
        'roughness_anchor_min_scan_iou',
        default_value='0.15',
        description='Minimum scan IoU required for a roughness-anchor graph constraint'
    )

    declare_roughness_anchor_max_scan_nn_p50_m = DeclareLaunchArgument(
        'roughness_anchor_max_scan_nn_p50_m',
        default_value='0.10',
        description='Maximum odom-frame scan nearest-neighbor p50 for anchor acceptance'
    )

    declare_roughness_anchor_max_abs_scan_yaw_deg = DeclareLaunchArgument(
        'roughness_anchor_max_abs_scan_yaw_deg',
        default_value='5.0',
        description='Maximum scan yaw correction magnitude for anchor acceptance'
    )

    declare_roughness_anchor_max_input_odom_gap_m = DeclareLaunchArgument(
        'roughness_anchor_max_input_odom_gap_m',
        default_value='0.10',
        description='Maximum input odom XY gap for a stable roughness anchor'
    )

    declare_roughness_anchor_min_pair_residual_m = DeclareLaunchArgument(
        'roughness_anchor_min_pair_residual_m',
        default_value='0.05',
        description='Minimum pair-level baseline SLAM-minus-odom residual worth constraining'
    )

    declare_roughness_anchor_max_graph_odom_gap_m = DeclareLaunchArgument(
        'roughness_anchor_max_graph_odom_gap_m',
        default_value='0.05',
        description=(
            'Maximum graph-node odom XY gap allowed before injecting an anchor edge; '
            'strict by default to avoid false roughness anchors'
        )
    )

    declare_roughness_anchor_max_constraint_pairs = DeclareLaunchArgument(
        'roughness_anchor_max_constraint_pairs',
        default_value='8',
        description='Maximum number of confirmed roughness event pairs to inject'
    )

    declare_roughness_anchor_require_complete_pair = DeclareLaunchArgument(
        'roughness_anchor_require_complete_pair',
        default_value='true',
        description='Require both stable pre/post anchor roles to pass every gate'
    )

    declare_roughness_anchor_translation_sigma_m = DeclareLaunchArgument(
        'roughness_anchor_translation_sigma_m',
        default_value='0.10',
        description='Translation sigma used for injected roughness-anchor graph edges'
    )

    declare_roughness_anchor_yaw_sigma_rad = DeclareLaunchArgument(
        'roughness_anchor_yaw_sigma_rad',
        default_value='1.0',
        description='Yaw sigma used for injected roughness-anchor graph edges'
    )

    standard_slam_condition = AndSubstitution(
        start_slam,
        AndSubstitution(
            NotSubstitution(use_roughness_anchor_slam_toolbox),
            NotSubstitution(use_roughness_innovation_slam_toolbox),
        ),
    )
    roughness_anchor_condition = AndSubstitution(
        start_slam,
        AndSubstitution(
            use_roughness_anchor_slam_toolbox,
            NotSubstitution(use_roughness_innovation_slam_toolbox),
        ),
    )
    roughness_innovation_condition = AndSubstitution(
        start_slam,
        AndSubstitution(
            use_roughness_innovation_slam_toolbox,
            NotSubstitution(use_roughness_anchor_slam_toolbox),
        ),
    )

    slam_toolbox_node = LifecycleNode(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': slam_toolbox_scan_topic,
                'log_level': 'debug'
            }
        ],
        condition=IfCondition(standard_slam_condition)
    )

    roughness_anchor_slam_toolbox_node = Node(
        package='slam_2d_core',
        executable='roughness_anchor_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': slam_toolbox_scan_topic,
                'log_level': 'debug',
                'enable_roughness_anchor_constraints': True,
                'roughness_anchor_constraints_csv': roughness_anchor_constraints_csv,
                'roughness_anchor_event_a': roughness_anchor_event_a,
                'roughness_anchor_event_b': roughness_anchor_event_b,
                'roughness_anchor_max_stamp_dt_s': roughness_anchor_max_stamp_dt_s,
                'roughness_anchor_min_scan_iou': roughness_anchor_min_scan_iou,
                'roughness_anchor_max_scan_nn_p50_m': roughness_anchor_max_scan_nn_p50_m,
                'roughness_anchor_max_abs_scan_yaw_deg': roughness_anchor_max_abs_scan_yaw_deg,
                'roughness_anchor_max_input_odom_gap_m': roughness_anchor_max_input_odom_gap_m,
                'roughness_anchor_min_pair_residual_m': roughness_anchor_min_pair_residual_m,
                'roughness_anchor_max_graph_odom_gap_m': roughness_anchor_max_graph_odom_gap_m,
                'roughness_anchor_max_constraint_pairs': roughness_anchor_max_constraint_pairs,
                'roughness_anchor_require_complete_pair': roughness_anchor_require_complete_pair,
                'roughness_anchor_translation_sigma_m': roughness_anchor_translation_sigma_m,
                'roughness_anchor_yaw_sigma_rad': roughness_anchor_yaw_sigma_rad,
            }
        ],
        condition=IfCondition(roughness_anchor_condition)
    )

    roughness_innovation_slam_toolbox_node = Node(
        package='slam_2d_core',
        executable='roughness_aware_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            params_file,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': slam_toolbox_scan_topic,
                'log_level': 'debug',
                'enable_roughness_innovation_gate': True,
                'roughness_innovation_mode': roughness_innovation_mode,
                'roughness_imu_topic': roughness_innovation_imu_topic,
                'roughness_scan_qos_reliable':
                    roughness_innovation_scan_qos_reliable,
                'roughness_scan_qos_depth':
                    roughness_innovation_scan_qos_depth,
                'roughness_window_before_s':
                    roughness_innovation_window_before_s,
                'roughness_window_after_s':
                    roughness_innovation_window_after_s,
                'roughness_score_gate': roughness_innovation_score_gate,
                'roughness_min_rough_samples':
                    roughness_innovation_min_rough_samples,
                'roughness_min_rough_duration_s':
                    roughness_innovation_min_rough_duration_s,
                'roughness_translation_innovation_gate_m':
                    roughness_innovation_translation_gate_m,
                'roughness_yaw_innovation_gate_deg':
                    roughness_innovation_yaw_gate_deg,
                'roughness_enable_near_chain_gate':
                    roughness_innovation_enable_near_chain_gate,
                'roughness_near_chain_recovery_duration_s':
                    roughness_innovation_near_chain_recovery_duration_s,
                'roughness_near_chain_loop_closing_cooldown_s':
                    roughness_innovation_near_chain_loop_closing_cooldown_s,
                'roughness_near_chain_min_rough_samples':
                    roughness_innovation_near_chain_min_rough_samples,
                'roughness_near_chain_gated_min_response':
                    roughness_innovation_near_chain_gated_min_response,
                'roughness_disable_loop_closing_during_near_chain_gate':
                    roughness_innovation_disable_loop_closing_during_gate,
                'roughness_enable_spatial_memory_gate':
                    roughness_innovation_enable_spatial_memory_gate,
                'roughness_spatial_memory_use_pose_graph_anchors':
                    roughness_innovation_spatial_memory_use_pose_graph_anchors,
                'roughness_spatial_memory_radius_m':
                    roughness_innovation_spatial_memory_radius_m,
                'roughness_spatial_memory_min_revisit_separation_s':
                    roughness_innovation_spatial_memory_min_revisit_separation_s,
                'roughness_spatial_memory_min_heading_delta_deg':
                    roughness_innovation_spatial_memory_min_heading_delta_deg,
                'roughness_spatial_memory_merge_radius_m':
                    roughness_innovation_spatial_memory_merge_radius_m,
                'roughness_spatial_memory_max_zones':
                    roughness_innovation_spatial_memory_max_zones,
                'roughness_enable_spatial_passage_hysteresis':
                    roughness_innovation_enable_spatial_passage_hysteresis,
                'roughness_spatial_passage_radius_m':
                    roughness_innovation_spatial_passage_radius_m,
                'roughness_spatial_passage_max_age_s':
                    roughness_innovation_spatial_passage_max_age_s,
                'roughness_enable_post_confirmation':
                    roughness_innovation_enable_post_confirmation,
                'roughness_post_confirmation_max_age_s':
                    roughness_innovation_post_confirmation_max_age_s,
                'roughness_post_confirmation_min_base_scans':
                    roughness_innovation_post_confirmation_min_base_scans,
                'roughness_post_confirmation_max_base_scans':
                    roughness_innovation_post_confirmation_max_base_scans,
                'roughness_post_confirmation_min_response':
                    roughness_innovation_post_confirmation_min_response,
                'roughness_post_confirmation_min_response_gain':
                    roughness_innovation_post_confirmation_min_response_gain,
                'roughness_post_confirmation_min_covariance_improvement_ratio':
                    roughness_innovation_post_confirmation_min_covariance_improvement_ratio,
                'roughness_post_confirmation_translation_disagreement_gate_m':
                    roughness_innovation_post_confirmation_translation_disagreement_gate_m,
                'roughness_post_confirmation_yaw_disagreement_gate_deg':
                    roughness_innovation_post_confirmation_yaw_disagreement_gate_deg,
                'roughness_experimental_reject_scan_stamps_ns':
                    roughness_innovation_experimental_reject_scan_stamps_ns,
            }
        ],
        condition=IfCondition(roughness_innovation_condition)
    )

    configure_event = TimerAction(
        period=1.0,
        actions=[
            LogInfo(msg="[LifecycleLaunch] Configuring slam_toolbox."),
            EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_toolbox_node),
                    transition_id=Transition.TRANSITION_CONFIGURE
                )
            )
        ],
        condition=IfCondition(standard_slam_condition)
    )

    activate_event = TimerAction(
        period=4.0,
        actions=[
            LogInfo(msg="[LifecycleLaunch] Activating slam_toolbox."),
            EmitEvent(
                event=ChangeState(
                    lifecycle_node_matcher=matches_action(slam_toolbox_node),
                    transition_id=Transition.TRANSITION_ACTIVATE
                )
            )
        ],
        condition=IfCondition(standard_slam_condition)
    )

    bag_play_with_cloud_delayed = TimerAction(
        period=7.0,
        actions=[
            LogInfo(msg="[LifecycleLaunch] Starting LaserScan+PointCloud bag playback."),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', bag_file,
                    '--clock', '--rate', bag_play_rate,
                    '--start-offset', bag_start_offset_s,
                    '--disable-keyboard-controls',
                    '--topics',
                    *SLAM_BAG_PLAY_TOPICS_WITH_CLOUD,
                ],
                output='screen'
            )
        ],
        condition=IfCondition(AndSubstitution(
            AndSubstitution(
                AndSubstitution(
                    start_slam,
                    PythonExpression(['"', bag_file, '" != ""'])
                ),
                play_slam_topics_only
            ),
            start_pointcloud_projection
        ))
    )

    bag_play_laserscan_only_delayed = TimerAction(
        period=7.0,
        actions=[
            LogInfo(msg="[LifecycleLaunch] Starting LaserScan-only bag playback."),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', bag_file,
                    '--clock', '--rate', bag_play_rate,
                    '--start-offset', bag_start_offset_s,
                    '--disable-keyboard-controls',
                    '--topics',
                    *SLAM_BAG_PLAY_TOPICS_LASERSCAN_ONLY,
                ],
                output='screen'
            )
        ],
        condition=IfCondition(AndSubstitution(
            AndSubstitution(
                AndSubstitution(
                    start_slam,
                    PythonExpression(['"', bag_file, '" != ""'])
                ),
                play_slam_topics_only
            ),
            NotSubstitution(start_pointcloud_projection)
        ))
    )

    bag_play_all_topics_delayed = TimerAction(
        period=7.0,
        actions=[
            LogInfo(msg="[LifecycleLaunch] Starting full bag playback."),
            ExecuteProcess(
                cmd=[
                    'ros2', 'bag', 'play', bag_file,
                    '--clock', '--rate', bag_play_rate,
                    '--start-offset', bag_start_offset_s,
                    '--disable-keyboard-controls',
                ],
                output='screen'
            )
        ],
        condition=IfCondition(AndSubstitution(
            AndSubstitution(
                start_slam,
                PythonExpression(['"', bag_file, '" != ""'])
            ),
            NotSubstitution(play_slam_topics_only)
        ))
    )

    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_file,
            '--clock', '--rate', bag_play_rate,
            '--start-offset', bag_start_offset_s,
            '--disable-keyboard-controls',
        ],
        output='screen',
        condition=IfCondition(AndSubstitution(
            PythonExpression(['"', bag_file, '" != ""']),
            NotSubstitution(start_slam)
        ))
    )

    static_map_odom_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_odom_tf',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
        condition=IfCondition(use_static_map_odom)
    )

    pointcloud_projection_node = Node(
        package='lidar_preprocessing',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan_node',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'input_topic': cloud_topic,
            'output_topic': scan_topic,
            'output_frame': 'base_link',
            'required_input_frame': 'base_link',
            'use_height_filter': True,
            'min_z': projection_min_z,
            'max_z': projection_max_z,
            'min_publish_period_s': projection_min_publish_period_s,
            'initial_skip_count': projection_initial_skip_count,
            'drop_nonmonotonic_stamps': projection_drop_nonmonotonic_stamps,
            'max_stamp_gap_s': projection_max_stamp_gap_s,
            'post_gap_skip_count': projection_post_gap_skip_count,
            'post_gap_skip_duration_s': projection_post_gap_skip_duration_s,
            'gate_by_tf_stamp': projection_gate_by_tf_stamp,
            'tf_topic': '/tf',
            'tf_gate_child_frame': projection_tf_gate_child_frame,
            'tf_gate_child_frames': ['base_link'],
            'max_scan_tf_lag_s': projection_max_scan_tf_lag_s,
            'enable_scan_quality_gate': projection_enable_scan_quality_gate,
            'scan_quality_min_finite_point_ratio':
                projection_scan_quality_min_finite_point_ratio,
            'scan_quality_min_height_kept_ratio':
                projection_scan_quality_min_height_kept_ratio,
            'scan_quality_min_range_kept_ratio':
                projection_scan_quality_min_range_kept_ratio,
            'scan_quality_min_valid_bin_ratio':
                projection_scan_quality_min_valid_bin_ratio,
            'scan_quality_min_valid_bins': projection_scan_quality_min_valid_bins,
            'scan_quality_max_empty_gap_ratio':
                projection_scan_quality_max_empty_gap_ratio,
            'scan_quality_score_gate': projection_scan_quality_score_gate,
            'diagnostics_period_s': projection_diagnostics_period_s,
        }],
        condition=IfCondition(start_pointcloud_projection),
    )

    return LaunchDescription([
        declare_bag_file,
        declare_params_file,
        declare_use_sim_time,
        declare_start_slam,
        declare_scan_topic,
        declare_slam_scan_topic,
        declare_cloud_topic,
        declare_start_pointcloud_projection,
        declare_projection_min_z,
        declare_projection_max_z,
        declare_projection_min_publish_period_s,
        declare_projection_initial_skip_count,
        declare_projection_drop_nonmonotonic_stamps,
        declare_projection_max_stamp_gap_s,
        declare_projection_post_gap_skip_count,
        declare_projection_post_gap_skip_duration_s,
        declare_projection_gate_by_tf_stamp,
        declare_projection_tf_gate_child_frame,
        declare_projection_max_scan_tf_lag_s,
        declare_projection_enable_scan_quality_gate,
        declare_projection_scan_quality_min_finite_point_ratio,
        declare_projection_scan_quality_min_height_kept_ratio,
        declare_projection_scan_quality_min_range_kept_ratio,
        declare_projection_scan_quality_min_valid_bin_ratio,
        declare_projection_scan_quality_min_valid_bins,
        declare_projection_scan_quality_max_empty_gap_ratio,
        declare_projection_scan_quality_score_gate,
        declare_projection_diagnostics_period_s,
        declare_use_static_map_odom,
        declare_bag_play_rate,
        declare_bag_start_offset_s,
        declare_play_slam_topics_only,
        declare_use_roughness_anchor_slam_toolbox,
        declare_use_roughness_innovation_slam_toolbox,
        declare_roughness_innovation_mode,
        declare_roughness_innovation_imu_topic,
        declare_roughness_innovation_scan_qos_reliable,
        declare_roughness_innovation_scan_qos_depth,
        declare_roughness_innovation_window_before_s,
        declare_roughness_innovation_window_after_s,
        declare_roughness_innovation_score_gate,
        declare_roughness_innovation_min_rough_samples,
        declare_roughness_innovation_min_rough_duration_s,
        declare_roughness_innovation_translation_gate_m,
        declare_roughness_innovation_yaw_gate_deg,
        declare_roughness_innovation_enable_near_chain_gate,
        declare_roughness_innovation_near_chain_recovery_duration_s,
        declare_roughness_innovation_near_chain_loop_closing_cooldown_s,
        declare_roughness_innovation_near_chain_min_rough_samples,
        declare_roughness_innovation_near_chain_gated_min_response,
        declare_roughness_innovation_disable_loop_closing_during_gate,
        declare_roughness_innovation_enable_spatial_memory_gate,
        declare_roughness_innovation_spatial_memory_use_pose_graph_anchors,
        declare_roughness_innovation_spatial_memory_radius_m,
        declare_roughness_innovation_spatial_memory_min_revisit_separation_s,
        declare_roughness_innovation_spatial_memory_min_heading_delta_deg,
        declare_roughness_innovation_spatial_memory_merge_radius_m,
        declare_roughness_innovation_spatial_memory_max_zones,
        declare_roughness_innovation_enable_spatial_passage_hysteresis,
        declare_roughness_innovation_spatial_passage_radius_m,
        declare_roughness_innovation_spatial_passage_max_age_s,
        declare_roughness_innovation_enable_post_confirmation,
        declare_roughness_innovation_post_confirmation_max_age_s,
        declare_roughness_innovation_post_confirmation_min_base_scans,
        declare_roughness_innovation_post_confirmation_max_base_scans,
        declare_roughness_innovation_post_confirmation_min_response,
        declare_roughness_innovation_post_confirmation_min_response_gain,
        declare_roughness_innovation_post_confirmation_min_covariance_improvement_ratio,
        declare_roughness_innovation_post_confirmation_translation_disagreement_gate_m,
        declare_roughness_innovation_post_confirmation_yaw_disagreement_gate_deg,
        declare_roughness_innovation_experimental_reject_scan_stamps_ns,
        declare_roughness_anchor_constraints_csv,
        declare_roughness_anchor_event_a,
        declare_roughness_anchor_event_b,
        declare_roughness_anchor_max_stamp_dt_s,
        declare_roughness_anchor_min_scan_iou,
        declare_roughness_anchor_max_scan_nn_p50_m,
        declare_roughness_anchor_max_abs_scan_yaw_deg,
        declare_roughness_anchor_max_input_odom_gap_m,
        declare_roughness_anchor_min_pair_residual_m,
        declare_roughness_anchor_max_graph_odom_gap_m,
        declare_roughness_anchor_max_constraint_pairs,
        declare_roughness_anchor_require_complete_pair,
        declare_roughness_anchor_translation_sigma_m,
        declare_roughness_anchor_yaw_sigma_rad,
        OpaqueFunction(function=validate_custom_slam_toolbox_selection),
        OpaqueFunction(function=validate_roughness_innovation_playback_topics),
        OpaqueFunction(function=validate_roughness_innovation_dependencies),
        slam_toolbox_node,
        roughness_anchor_slam_toolbox_node,
        roughness_innovation_slam_toolbox_node,
        pointcloud_projection_node,
        static_map_odom_tf,
        configure_event,
        activate_event,
        bag_play_with_cloud_delayed,
        bag_play_laserscan_only_delayed,
        bag_play_all_topics_delayed,
        bag_play,
        LogInfo(msg=['Bag file will be played after SLAM node is active: ', bag_file]),
        LogInfo(msg=['SLAM scan topic: ', scan_topic]),
        LogInfo(msg=['SLAM toolbox input scan topic: ', slam_toolbox_scan_topic]),
        LogInfo(msg=['PointCloud2 projection enabled: ', start_pointcloud_projection]),
        LogInfo(msg=['SLAM-only bag playback: ', play_slam_topics_only]),
        LogInfo(msg='SLAM mapping started. Use map_saver to save the map when done.'),
        LogInfo(msg=['Static map->odom TF: ', use_static_map_odom]),
    ])
