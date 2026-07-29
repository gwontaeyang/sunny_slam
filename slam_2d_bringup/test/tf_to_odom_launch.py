#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch arguments
    parent_frame_arg = DeclareLaunchArgument(
        'parent_frame',
        default_value='odom',
        description='Parent frame for TF lookup (default: odom)'
    )

    child_frame_arg = DeclareLaunchArgument(
        'child_frame',
        default_value='base_link',
        description='Child frame for TF lookup (default: base_link)'
    )

    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic_name',
        default_value='odom_wheel',
        description='Odometry topic name to publish (default: odom)'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='50.0',
        description='Publish rate in Hz (default: 50.0)'
    )

    # TF to Odometry node
    tf_to_odom_node = Node(
        package='slam_2d_core',
        executable='tf_to_odom_node',
        name='tf_to_odom_node',
        output='screen',
        parameters=[{
            'parent_frame': LaunchConfiguration('parent_frame'),
            'child_frame': LaunchConfiguration('child_frame'),
            'odom_topic_name': LaunchConfiguration('odom_topic_name'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )

    return LaunchDescription([
        parent_frame_arg,
        child_frame_arg,
        odom_topic_arg,
        publish_rate_arg,
        tf_to_odom_node,
    ])
