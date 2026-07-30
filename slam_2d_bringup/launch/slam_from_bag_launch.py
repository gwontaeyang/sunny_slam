#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.actions import EmitEvent
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import AndSubstitution, LaunchConfiguration
from launch.substitutions import NotSubstitution, PythonExpression
from launch_ros.actions import LifecycleNode, Node
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


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


def generate_launch_description():
    slam_2d_bringup_dir = get_package_share_directory('slam_2d_bringup')

    bag_file = LaunchConfiguration('bag_file')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    start_slam = LaunchConfiguration('start_slam', default='true')
    scan_topic = LaunchConfiguration('scan_topic', default='/scan_laserscan')
    slam_scan_topic = LaunchConfiguration('slam_scan_topic', default='/scan_laserscan')
    cloud_topic = LaunchConfiguration('cloud_topic', default='/scan')
    start_pointcloud_projection = LaunchConfiguration(
        'start_pointcloud_projection',
        default='false')
    enable_laserscan_roughness_gate = LaunchConfiguration(
        'enable_laserscan_roughness_gate',
        default='false')
    laserscan_roughness_gate_mode = LaunchConfiguration(
        'laserscan_roughness_gate_mode',
        default='drop')
    laserscan_roughness_use_nearest_imu_sample = LaunchConfiguration(
        'laserscan_roughness_use_nearest_imu_sample',
        default='true')
    laserscan_roughness_imu_buffer_duration_s = LaunchConfiguration(
        'laserscan_roughness_imu_buffer_duration_s',
        default='1.0')
    laserscan_roughness_stabilize_max_source_age_s = LaunchConfiguration(
        'laserscan_roughness_stabilize_max_source_age_s',
        default='0.35')
    laserscan_roughness_stabilize_max_event_duration_s = LaunchConfiguration(
        'laserscan_roughness_stabilize_max_event_duration_s',
        default='0.5')
    laserscan_roughness_stabilize_min_publish_period_s = LaunchConfiguration(
        'laserscan_roughness_stabilize_min_publish_period_s',
        default='0.1')
    laserscan_roughness_stabilize_max_translation_m = LaunchConfiguration(
        'laserscan_roughness_stabilize_max_translation_m',
        default='0.2')
    laserscan_roughness_stabilize_max_yaw_rad = LaunchConfiguration(
        'laserscan_roughness_stabilize_max_yaw_rad',
        default='0.15')
    laserscan_roughness_stabilize_max_roll_pitch_rad = LaunchConfiguration(
        'laserscan_roughness_stabilize_max_roll_pitch_rad',
        default='0.05')
    laserscan_roughness_stabilize_min_cached_valid_ratio = LaunchConfiguration(
        'laserscan_roughness_stabilize_min_cached_valid_ratio',
        default='0.2')
    laserscan_roughness_stabilize_min_output_valid_ratio = LaunchConfiguration(
        'laserscan_roughness_stabilize_min_output_valid_ratio',
        default='0.15')
    laserscan_roughness_stabilize_min_output_valid_beams = LaunchConfiguration(
        'laserscan_roughness_stabilize_min_output_valid_beams',
        default='20')
    laserscan_roughness_stabilize_fallback_mode = LaunchConfiguration(
        'laserscan_roughness_stabilize_fallback_mode',
        default='drop')
    laserscan_roughness_stabilize_require_return_revisit = LaunchConfiguration(
        'laserscan_roughness_stabilize_require_return_revisit',
        default='false')
    laserscan_roughness_adaptive_use_travel_heading = LaunchConfiguration(
        'laserscan_roughness_adaptive_use_travel_heading',
        default='false')
    laserscan_roughness_adaptive_min_travel_distance_m = LaunchConfiguration(
        'laserscan_roughness_adaptive_min_travel_distance_m',
        default='0.25')
    laserscan_roughness_adaptive_travel_heading_window_s = LaunchConfiguration(
        'laserscan_roughness_adaptive_travel_heading_window_s',
        default='4.0')
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
    projection_enable_gravity_alignment = LaunchConfiguration(
        'projection_enable_gravity_alignment',
        default='false')
    projection_gravity_max_scan_imu_dt_s = LaunchConfiguration(
        'projection_gravity_max_scan_imu_dt_s',
        default='0.05')
    projection_gravity_roll_scale = LaunchConfiguration(
        'projection_gravity_roll_scale',
        default='1.0')
    projection_gravity_pitch_scale = LaunchConfiguration(
        'projection_gravity_pitch_scale',
        default='1.0')
    projection_enable_imu_roughness_gate = LaunchConfiguration(
        'projection_enable_imu_roughness_gate',
        default='true')
    projection_imu_topic = LaunchConfiguration(
        'projection_imu_topic',
        default='/imu/data')
    projection_roughness_tilt_threshold_deg = LaunchConfiguration(
        'projection_roughness_tilt_threshold_deg',
        default='2.0')
    projection_roughness_gyro_xy_threshold_rad_s = LaunchConfiguration(
        'projection_roughness_gyro_xy_threshold_rad_s',
        default='0.05')
    projection_roughness_accel_norm_deviation_threshold_m_s2 = LaunchConfiguration(
        'projection_roughness_accel_norm_deviation_threshold_m_s2',
        default='0.35')
    projection_roughness_tilt_rate_threshold_deg_s = LaunchConfiguration(
        'projection_roughness_tilt_rate_threshold_deg_s',
        default='5.0')
    projection_roughness_accel_norm_rate_threshold_m_s3 = LaunchConfiguration(
        'projection_roughness_accel_norm_rate_threshold_m_s3',
        default='80.0')
    projection_roughness_max_scan_imu_dt_s = LaunchConfiguration(
        'projection_roughness_max_scan_imu_dt_s',
        default='0.2')
    projection_roughness_hold_duration_s = LaunchConfiguration(
        'projection_roughness_hold_duration_s',
        default='0.35')
    projection_roughness_hard_hold_duration_s = LaunchConfiguration(
        'projection_roughness_hard_hold_duration_s',
        default='0.0')
    projection_roughness_score_gate = LaunchConfiguration(
        'projection_roughness_score_gate',
        default='1.0')
    projection_roughness_score_hard_gate = LaunchConfiguration(
        'projection_roughness_score_hard_gate',
        default='2.0')
    projection_roughness_gate_mode = LaunchConfiguration(
        'projection_roughness_gate_mode',
        default='drop')
    projection_roughness_throttle_min_publish_period_s = LaunchConfiguration(
        'projection_roughness_throttle_min_publish_period_s',
        default='0.5')
    projection_roughness_throttle_drop_hard = LaunchConfiguration(
        'projection_roughness_throttle_drop_hard',
        default='true')
    projection_roughness_max_continuous_drop_duration_s = LaunchConfiguration(
        'projection_roughness_max_continuous_drop_duration_s',
        default='0.0')
    projection_roughness_adaptive_require_return_revisit = LaunchConfiguration(
        'projection_roughness_adaptive_require_return_revisit',
        default='true')
    projection_roughness_adaptive_revisit_radius_m = LaunchConfiguration(
        'projection_roughness_adaptive_revisit_radius_m',
        default='0.8')
    projection_roughness_adaptive_min_separation_s = LaunchConfiguration(
        'projection_roughness_adaptive_min_separation_s',
        default='20.0')
    projection_roughness_adaptive_min_heading_delta_rad = LaunchConfiguration(
        'projection_roughness_adaptive_min_heading_delta_rad',
        default='2.2')
    projection_roughness_adaptive_history_duration_s = LaunchConfiguration(
        'projection_roughness_adaptive_history_duration_s',
        default='900.0')
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
        default_value='/scan_laserscan',
        description='LaserScan topic consumed directly by slam_toolbox'
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

    declare_enable_laserscan_roughness_gate = DeclareLaunchArgument(
        'enable_laserscan_roughness_gate',
        default_value='false',
        description='Filter the LaserScan stream for SLAM using the IMU roughness gate'
    )

    declare_laserscan_roughness_gate_mode = DeclareLaunchArgument(
        'laserscan_roughness_gate_mode',
        default_value='drop',
        description='LaserScan roughness policy, independent of point-cloud projection'
    )
    declare_laserscan_roughness_use_nearest_imu_sample = DeclareLaunchArgument(
        'laserscan_roughness_use_nearest_imu_sample',
        default_value='true',
        description='Use the nearest buffered IMU sample when gating LaserScan input'
    )

    declare_laserscan_roughness_imu_buffer_duration_s = DeclareLaunchArgument(
        'laserscan_roughness_imu_buffer_duration_s',
        default_value='1.0',
        description='IMU history duration for LaserScan roughness gating'
    )
    declare_laserscan_roughness_stabilize_max_source_age_s = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_max_source_age_s',
        default_value='0.35')
    declare_laserscan_roughness_stabilize_max_event_duration_s = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_max_event_duration_s',
        default_value='0.5')
    declare_laserscan_roughness_stabilize_min_publish_period_s = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_min_publish_period_s',
        default_value='0.1')
    declare_laserscan_roughness_stabilize_max_translation_m = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_max_translation_m',
        default_value='0.2')
    declare_laserscan_roughness_stabilize_max_yaw_rad = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_max_yaw_rad',
        default_value='0.15')
    declare_laserscan_roughness_stabilize_max_roll_pitch_rad = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_max_roll_pitch_rad',
        default_value='0.05')
    declare_laserscan_roughness_stabilize_min_cached_valid_ratio = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_min_cached_valid_ratio',
        default_value='0.2')
    declare_laserscan_roughness_stabilize_min_output_valid_ratio = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_min_output_valid_ratio',
        default_value='0.15')
    declare_laserscan_roughness_stabilize_min_output_valid_beams = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_min_output_valid_beams',
        default_value='20')
    declare_laserscan_roughness_stabilize_fallback_mode = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_fallback_mode',
        default_value='drop')
    declare_laserscan_roughness_stabilize_require_return_revisit = DeclareLaunchArgument(
        'laserscan_roughness_stabilize_require_return_revisit',
        default_value='false')
    declare_laserscan_roughness_adaptive_use_travel_heading = DeclareLaunchArgument(
        'laserscan_roughness_adaptive_use_travel_heading',
        default_value='false')
    declare_laserscan_roughness_adaptive_min_travel_distance_m = DeclareLaunchArgument(
        'laserscan_roughness_adaptive_min_travel_distance_m',
        default_value='0.25')
    declare_laserscan_roughness_adaptive_travel_heading_window_s = DeclareLaunchArgument(
        'laserscan_roughness_adaptive_travel_heading_window_s',
        default_value='4.0')

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

    declare_projection_enable_gravity_alignment = DeclareLaunchArgument(
        'projection_enable_gravity_alignment',
        default_value='false',
        description='Apply IMU roll/pitch compensation before projected LaserScan height filtering'
    )

    declare_projection_gravity_max_scan_imu_dt_s = DeclareLaunchArgument(
        'projection_gravity_max_scan_imu_dt_s',
        default_value='0.05',
        description='Maximum scan-to-IMU timestamp difference for projection gravity alignment'
    )

    declare_projection_gravity_roll_scale = DeclareLaunchArgument(
        'projection_gravity_roll_scale',
        default_value='1.0',
        description='Scale applied to IMU roll before projection gravity alignment'
    )

    declare_projection_gravity_pitch_scale = DeclareLaunchArgument(
        'projection_gravity_pitch_scale',
        default_value='1.0',
        description='Scale applied to IMU pitch before projection gravity alignment'
    )

    declare_projection_enable_imu_roughness_gate = DeclareLaunchArgument(
        'projection_enable_imu_roughness_gate',
        default_value='true',
        description='Suppress projected LaserScan updates during IMU-detected non-planar roughness'
    )

    declare_projection_imu_topic = DeclareLaunchArgument(
        'projection_imu_topic',
        default_value='/imu/data',
        description='IMU topic used by projected scan roughness gate'
    )

    declare_projection_roughness_tilt_threshold_deg = DeclareLaunchArgument(
        'projection_roughness_tilt_threshold_deg',
        default_value='2.0',
        description='Roll/pitch tilt threshold for projected scan roughness gate'
    )

    declare_projection_roughness_gyro_xy_threshold_rad_s = DeclareLaunchArgument(
        'projection_roughness_gyro_xy_threshold_rad_s',
        default_value='0.05',
        description='Roll/pitch angular velocity threshold for projected scan roughness gate'
    )

    declare_projection_roughness_accel_norm_deviation_threshold_m_s2 = DeclareLaunchArgument(
        'projection_roughness_accel_norm_deviation_threshold_m_s2',
        default_value='0.35',
        description='Acceleration norm deviation threshold for projected scan roughness gate'
    )

    declare_projection_roughness_tilt_rate_threshold_deg_s = DeclareLaunchArgument(
        'projection_roughness_tilt_rate_threshold_deg_s',
        default_value='5.0',
        description='Roll/pitch tilt-rate threshold for projected scan roughness gate'
    )

    declare_projection_roughness_accel_norm_rate_threshold_m_s3 = DeclareLaunchArgument(
        'projection_roughness_accel_norm_rate_threshold_m_s3',
        default_value='80.0',
        description='Acceleration norm rate threshold for projected scan roughness gate'
    )

    declare_projection_roughness_max_scan_imu_dt_s = DeclareLaunchArgument(
        'projection_roughness_max_scan_imu_dt_s',
        default_value='0.2',
        description='Maximum scan-to-IMU timestamp difference for roughness gate'
    )

    declare_projection_roughness_hold_duration_s = DeclareLaunchArgument(
        'projection_roughness_hold_duration_s',
        default_value='0.35',
        description='Hold projected scan suppression after roughness clears'
    )

    declare_projection_roughness_hard_hold_duration_s = DeclareLaunchArgument(
        'projection_roughness_hard_hold_duration_s',
        default_value='0.0',
        description='Longer projected scan suppression hold after hard roughness; 0 disables it'
    )

    declare_projection_roughness_score_gate = DeclareLaunchArgument(
        'projection_roughness_score_gate',
        default_value='1.0',
        description='Roughness score threshold for projected scan suppression'
    )

    declare_projection_roughness_score_hard_gate = DeclareLaunchArgument(
        'projection_roughness_score_hard_gate',
        default_value='2.0',
        description='Roughness score threshold logged as hard projected scan suppression'
    )

    declare_projection_roughness_gate_mode = DeclareLaunchArgument(
        'projection_roughness_gate_mode',
        default_value='drop',
        description='IMU roughness gate mode: drop, throttle, observe, or adaptive'
    )

    declare_projection_roughness_throttle_min_publish_period_s = DeclareLaunchArgument(
        'projection_roughness_throttle_min_publish_period_s',
        default_value='0.5',
        description='Minimum period for allowing scans during roughness when gate mode is throttle'
    )

    declare_projection_roughness_throttle_drop_hard = DeclareLaunchArgument(
        'projection_roughness_throttle_drop_hard',
        default_value='true',
        description='Keep dropping hard roughness scans when roughness gate mode is throttle'
    )

    declare_projection_roughness_max_continuous_drop_duration_s = DeclareLaunchArgument(
        'projection_roughness_max_continuous_drop_duration_s',
        default_value='0.0',
        description=(
            'Maximum continuous roughness scan suppression before allowing one bridge scan; '
            'zero preserves unbounded hard-scan suppression'
        )
    )

    declare_projection_roughness_adaptive_require_return_revisit = DeclareLaunchArgument(
        'projection_roughness_adaptive_require_return_revisit',
        default_value='true',
        description=(
            'Only apply adaptive roughness throttling near opposite-direction odom revisits'
        )
    )

    declare_projection_roughness_adaptive_revisit_radius_m = DeclareLaunchArgument(
        'projection_roughness_adaptive_revisit_radius_m',
        default_value='0.8',
        description='Odom XY radius for adaptive roughness return-revisit detection'
    )

    declare_projection_roughness_adaptive_min_separation_s = DeclareLaunchArgument(
        'projection_roughness_adaptive_min_separation_s',
        default_value='20.0',
        description='Minimum time separation for adaptive roughness return-revisit detection'
    )

    declare_projection_roughness_adaptive_min_heading_delta_rad = DeclareLaunchArgument(
        'projection_roughness_adaptive_min_heading_delta_rad',
        default_value='2.2',
        description='Minimum heading delta for adaptive roughness return-revisit detection'
    )

    declare_projection_roughness_adaptive_history_duration_s = DeclareLaunchArgument(
        'projection_roughness_adaptive_history_duration_s',
        default_value='900.0',
        description='Odom history duration for adaptive roughness return-revisit detection'
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

    slam_config = os.path.join(
        slam_2d_bringup_dir,
        'config',
        'roughness_anchor_mapper_params.yaml',
    )

    slam_toolbox_node = LifecycleNode(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            slam_config,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': slam_scan_topic,
                'log_level': 'debug'
            }
        ],
        condition=IfCondition(
            AndSubstitution(start_slam, NotSubstitution(use_roughness_anchor_slam_toolbox))
        )
    )

    roughness_anchor_slam_toolbox_node = Node(
        package='slam_2d_core',
        executable='roughness_anchor_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        namespace='',
        parameters=[
            slam_config,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': slam_scan_topic,
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
        condition=IfCondition(AndSubstitution(start_slam, use_roughness_anchor_slam_toolbox))
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
        condition=IfCondition(
            AndSubstitution(start_slam, NotSubstitution(use_roughness_anchor_slam_toolbox))
        )
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
        condition=IfCondition(
            AndSubstitution(start_slam, NotSubstitution(use_roughness_anchor_slam_toolbox))
        )
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
            'enable_gravity_alignment': projection_enable_gravity_alignment,
            'gravity_max_scan_imu_dt_s': projection_gravity_max_scan_imu_dt_s,
            'gravity_roll_scale': projection_gravity_roll_scale,
            'gravity_pitch_scale': projection_gravity_pitch_scale,
            'enable_imu_roughness_gate': projection_enable_imu_roughness_gate,
            'imu_topic': projection_imu_topic,
            'roughness_tilt_threshold_deg': projection_roughness_tilt_threshold_deg,
            'roughness_gyro_xy_threshold_rad_s': projection_roughness_gyro_xy_threshold_rad_s,
            'roughness_accel_norm_deviation_threshold_m_s2':
                projection_roughness_accel_norm_deviation_threshold_m_s2,
            'roughness_tilt_rate_threshold_deg_s':
                projection_roughness_tilt_rate_threshold_deg_s,
            'roughness_accel_norm_rate_threshold_m_s3':
                projection_roughness_accel_norm_rate_threshold_m_s3,
            'roughness_max_scan_imu_dt_s': projection_roughness_max_scan_imu_dt_s,
            'roughness_hold_duration_s': projection_roughness_hold_duration_s,
            'roughness_hard_hold_duration_s': projection_roughness_hard_hold_duration_s,
            'roughness_score_gate': projection_roughness_score_gate,
            'roughness_score_hard_gate': projection_roughness_score_hard_gate,
            'roughness_gate_mode': projection_roughness_gate_mode,
            'roughness_throttle_min_publish_period_s':
                projection_roughness_throttle_min_publish_period_s,
            'roughness_throttle_drop_hard': projection_roughness_throttle_drop_hard,
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

    laserscan_roughness_gate_node = Node(
        package='amcl_guard',
        executable='amcl_guard_node',
        name='slam_laserscan_roughness_gate',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enabled': True,
            'input_scan_topic': scan_topic,
            'output_scan_topic': slam_scan_topic,
            'imu_topic': projection_imu_topic,
            'require_good_pose': False,
            'min_valid_beam_ratio': 0.0,
            'max_near_beam_ratio': 1.0,
            'max_hold_duration': 3600.0,
            'drop_scans_after_timeout': True,
            'publish_status_topics': True,
            'enable_imu_roughness_gate': True,
            'tilt_threshold_deg': projection_roughness_tilt_threshold_deg,
            'gyro_xy_threshold_rad_s': projection_roughness_gyro_xy_threshold_rad_s,
            'accel_norm_deviation_threshold_m_s2':
                projection_roughness_accel_norm_deviation_threshold_m_s2,
            'tilt_rate_threshold_deg_s': projection_roughness_tilt_rate_threshold_deg_s,
            'accel_norm_rate_threshold_m_s3':
                projection_roughness_accel_norm_rate_threshold_m_s3,
            'max_scan_imu_dt_s': projection_roughness_max_scan_imu_dt_s,
            'imu_roughness_hold_duration_s': projection_roughness_hold_duration_s,
            'imu_roughness_hard_hold_duration_s':
                projection_roughness_hard_hold_duration_s,
            'roughness_score_gate': projection_roughness_score_gate,
            'roughness_score_hard_gate': projection_roughness_score_hard_gate,
            'roughness_gate_mode': laserscan_roughness_gate_mode,
            'roughness_throttle_min_publish_period_s':
                projection_roughness_throttle_min_publish_period_s,
            'roughness_throttle_drop_hard': projection_roughness_throttle_drop_hard,
            'roughness_max_continuous_drop_duration_s':
                projection_roughness_max_continuous_drop_duration_s,
            'roughness_stabilize_max_source_age_s':
                laserscan_roughness_stabilize_max_source_age_s,
            'roughness_stabilize_max_event_duration_s':
                laserscan_roughness_stabilize_max_event_duration_s,
            'roughness_stabilize_min_publish_period_s':
                laserscan_roughness_stabilize_min_publish_period_s,
            'roughness_stabilize_max_translation_m':
                laserscan_roughness_stabilize_max_translation_m,
            'roughness_stabilize_max_yaw_rad':
                laserscan_roughness_stabilize_max_yaw_rad,
            'roughness_stabilize_max_roll_pitch_rad':
                laserscan_roughness_stabilize_max_roll_pitch_rad,
            'roughness_stabilize_min_cached_valid_ratio':
                laserscan_roughness_stabilize_min_cached_valid_ratio,
            'roughness_stabilize_min_output_valid_ratio':
                laserscan_roughness_stabilize_min_output_valid_ratio,
            'roughness_stabilize_min_output_valid_beams':
                laserscan_roughness_stabilize_min_output_valid_beams,
            'roughness_stabilize_fallback_mode':
                laserscan_roughness_stabilize_fallback_mode,
            'roughness_stabilize_require_return_revisit':
                laserscan_roughness_stabilize_require_return_revisit,
            'roughness_adaptive_require_return_revisit':
                projection_roughness_adaptive_require_return_revisit,
            'roughness_adaptive_revisit_radius_m':
                projection_roughness_adaptive_revisit_radius_m,
            'roughness_adaptive_min_separation_s':
                projection_roughness_adaptive_min_separation_s,
            'roughness_adaptive_min_heading_delta_rad':
                projection_roughness_adaptive_min_heading_delta_rad,
            'roughness_adaptive_history_duration_s':
                projection_roughness_adaptive_history_duration_s,
            'roughness_adaptive_use_travel_heading':
                laserscan_roughness_adaptive_use_travel_heading,
            'roughness_adaptive_min_travel_distance_m':
                laserscan_roughness_adaptive_min_travel_distance_m,
            'roughness_adaptive_travel_heading_window_s':
                laserscan_roughness_adaptive_travel_heading_window_s,
            'use_nearest_imu_sample': laserscan_roughness_use_nearest_imu_sample,
            'imu_buffer_duration_s': laserscan_roughness_imu_buffer_duration_s,
        }],
        condition=IfCondition(enable_laserscan_roughness_gate),
    )

    return LaunchDescription([
        declare_bag_file,
        declare_use_sim_time,
        declare_start_slam,
        declare_scan_topic,
        declare_slam_scan_topic,
        declare_cloud_topic,
        declare_start_pointcloud_projection,
        declare_enable_laserscan_roughness_gate,
        declare_laserscan_roughness_gate_mode,
        declare_laserscan_roughness_use_nearest_imu_sample,
        declare_laserscan_roughness_imu_buffer_duration_s,
        declare_laserscan_roughness_stabilize_max_source_age_s,
        declare_laserscan_roughness_stabilize_max_event_duration_s,
        declare_laserscan_roughness_stabilize_min_publish_period_s,
        declare_laserscan_roughness_stabilize_max_translation_m,
        declare_laserscan_roughness_stabilize_max_yaw_rad,
        declare_laserscan_roughness_stabilize_max_roll_pitch_rad,
        declare_laserscan_roughness_stabilize_min_cached_valid_ratio,
        declare_laserscan_roughness_stabilize_min_output_valid_ratio,
        declare_laserscan_roughness_stabilize_min_output_valid_beams,
        declare_laserscan_roughness_stabilize_fallback_mode,
        declare_laserscan_roughness_stabilize_require_return_revisit,
        declare_laserscan_roughness_adaptive_use_travel_heading,
        declare_laserscan_roughness_adaptive_min_travel_distance_m,
        declare_laserscan_roughness_adaptive_travel_heading_window_s,
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
        declare_projection_enable_gravity_alignment,
        declare_projection_gravity_max_scan_imu_dt_s,
        declare_projection_gravity_roll_scale,
        declare_projection_gravity_pitch_scale,
        declare_projection_enable_imu_roughness_gate,
        declare_projection_imu_topic,
        declare_projection_roughness_tilt_threshold_deg,
        declare_projection_roughness_gyro_xy_threshold_rad_s,
        declare_projection_roughness_accel_norm_deviation_threshold_m_s2,
        declare_projection_roughness_tilt_rate_threshold_deg_s,
        declare_projection_roughness_accel_norm_rate_threshold_m_s3,
        declare_projection_roughness_max_scan_imu_dt_s,
        declare_projection_roughness_hold_duration_s,
        declare_projection_roughness_hard_hold_duration_s,
        declare_projection_roughness_score_gate,
        declare_projection_roughness_score_hard_gate,
        declare_projection_roughness_gate_mode,
        declare_projection_roughness_throttle_min_publish_period_s,
        declare_projection_roughness_throttle_drop_hard,
        declare_projection_roughness_max_continuous_drop_duration_s,
        declare_projection_roughness_adaptive_require_return_revisit,
        declare_projection_roughness_adaptive_revisit_radius_m,
        declare_projection_roughness_adaptive_min_separation_s,
        declare_projection_roughness_adaptive_min_heading_delta_rad,
        declare_projection_roughness_adaptive_history_duration_s,
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
        slam_toolbox_node,
        roughness_anchor_slam_toolbox_node,
        pointcloud_projection_node,
        laserscan_roughness_gate_node,
        static_map_odom_tf,
        configure_event,
        activate_event,
        bag_play_with_cloud_delayed,
        bag_play_laserscan_only_delayed,
        bag_play_all_topics_delayed,
        bag_play,
        LogInfo(msg=['Bag file will be played after SLAM node is active: ', bag_file]),
        LogInfo(msg=['SLAM scan topic: ', scan_topic]),
        LogInfo(msg=['SLAM toolbox input scan topic: ', slam_scan_topic]),
        LogInfo(msg=['PointCloud2 projection enabled: ', start_pointcloud_projection]),
        LogInfo(msg=['LaserScan roughness gate enabled: ', enable_laserscan_roughness_gate]),
        LogInfo(
            msg=[
                'LaserScan roughness nearest IMU: ',
                laserscan_roughness_use_nearest_imu_sample
            ]
        ),
        LogInfo(msg=['SLAM-only bag playback: ', play_slam_topics_only]),
        LogInfo(msg='SLAM mapping started. Use map_saver to save the map when done.'),
        LogInfo(msg=['Static map->odom TF: ', use_static_map_odom]),
    ])
