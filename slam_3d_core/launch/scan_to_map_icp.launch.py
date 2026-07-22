#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_dir = get_package_share_directory("slam_3d_core")
    default_params = os.path.join(pkg_dir, "config", "scan_to_map_icp_params.yaml")

    declarations = [
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Parameter file for the scan-to-map ICP/GICP learning node",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time",
        ),
        DeclareLaunchArgument(
            "primary_cloud_topic",
            default_value="/slam_3d/input/mid360_front/points_debug",
            description="Primary Mid-360 PointCloud2 debug stream",
        ),
        DeclareLaunchArgument(
            "secondary_cloud_topic",
            default_value="/slam_3d/input/mid360_rear/points_debug",
            description="Secondary Mid-360 PointCloud2 debug stream",
        ),
        DeclareLaunchArgument(
            "use_secondary_cloud",
            default_value="true",
            description="Fuse the latest secondary cloud when timestamp skew is acceptable",
        ),
        DeclareLaunchArgument(
            "wheel_odom_topic",
            default_value="/motor_driver_node/odom",
            description="Existing GOLE wheel odometry topic used as a motion prior",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/imu/data",
            description="Existing external IMU topic; currently recorded for diagnostics",
        ),
        DeclareLaunchArgument(
            "registration_method",
            default_value="gicp",
            description="Registration method: gicp or icp",
        ),
        DeclareLaunchArgument(
            "map_frame",
            default_value="map",
            description="Map frame for prototype odometry and map clouds",
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Robot base frame",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description="Publish map -> base_link TF. Keep false with existing localization/AMCL.",
        ),
        DeclareLaunchArgument(
            "output",
            default_value="screen",
            description="Node output mode",
        ),
    ]

    node = Node(
        package="slam_3d_core",
        executable="scan_to_map_icp_node",
        name="scan_to_map_icp_node",
        output=LaunchConfiguration("output"),
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"),
                    value_type=bool,
                ),
                "primary_cloud_topic": LaunchConfiguration("primary_cloud_topic"),
                "secondary_cloud_topic": LaunchConfiguration("secondary_cloud_topic"),
                "use_secondary_cloud": ParameterValue(
                    LaunchConfiguration("use_secondary_cloud"),
                    value_type=bool,
                ),
                "wheel_odom_topic": LaunchConfiguration("wheel_odom_topic"),
                "imu_topic": LaunchConfiguration("imu_topic"),
                "registration_method": LaunchConfiguration("registration_method"),
                "map_frame": LaunchConfiguration("map_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
                "publish_tf": ParameterValue(
                    LaunchConfiguration("publish_tf"),
                    value_type=bool,
                ),
            },
        ],
    )

    return LaunchDescription(declarations + [node])
