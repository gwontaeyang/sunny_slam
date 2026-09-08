#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("slam_3d_lidar_ingest")
    default_params = os.path.join(
        package_share,
        "config",
        "hesai_jt128_xt32_ingest_params.yaml",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation time",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params,
                description="Two JT128 plus one XT32 ingest parameter file",
            ),
            Node(
                package="slam_3d_lidar_ingest",
                executable="hesai_pointcloud_ingest_node",
                name="hesai_pointcloud_ingest",
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
