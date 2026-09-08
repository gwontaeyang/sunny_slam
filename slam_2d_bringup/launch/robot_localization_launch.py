#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # Launch argument for config file override
    declare_config_file = DeclareLaunchArgument(
        'config_file',
        description='Required path to an explicitly selected EKF localization config file'
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

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the ROS simulation clock'
    )

    # EKF Localization Node
    ekf_filter_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name=LaunchConfiguration('node_name'),
        output=LaunchConfiguration('output'),
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'),
                    value_type=bool,
                ),
            },
        ],
        remappings=[
            ('odom0', '/motor_driver_node/odom'),
            ('imu0', '/imu/data'),
        ]
    )

    return LaunchDescription([
        declare_config_file,
        declare_node_name,
        declare_output,
        declare_use_sim_time,
        ekf_filter_node,
    ])
