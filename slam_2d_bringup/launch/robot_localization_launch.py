#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    slam_2d_bringup_dir = get_package_share_directory('slam_2d_bringup')

    # Path to the config file
    config_file = os.path.join(
        slam_2d_bringup_dir, 'config', 'ekf_localization_params.yaml')

    # Launch argument for config file override
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        default_value=config_file,
        description='Path to the EKF localization config file'
    )

    # Launch argument for node name
    declare_node_name = DeclareLaunchArgument(
        'node_name',
        default_value='ekf_filter_node',
        description='Name of the EKF filter node'
    )

    # Launch argument for output
    declare_output = DeclareLaunchArgument(
        'output',
        default_value='screen',
        description='Output destination (screen or log)'
    )

    # EKF Localization Node
    ekf_filter_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name=LaunchConfiguration('node_name'),
        output=LaunchConfiguration('output'),
        parameters=[
            LaunchConfiguration('config_file'),
        ],
        remappings=[
            ('odom0', '/motor_driver_node/odom'),
            ('imu0', '/imu/data'),
        ]
    )

    base_footprint_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_footprint_to_base_link_tf',
        arguments=[
            '--x', '0',
            '--y', '0',
            '--z', '0',
            '--roll', '0',
            '--pitch', '0',
            '--yaw', '0',
            '--frame-id', 'base_footprint',
            '--child-frame-id', 'base_link',
        ],
        output='screen',
    )

    return LaunchDescription([
        declare_config_file,
        declare_node_name,
        declare_output,
        base_footprint_tf,
        ekf_filter_node,
    ])
