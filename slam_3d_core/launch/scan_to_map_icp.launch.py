#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _parameter_overrides(context):
    overrides = {}

    for name in (
        "primary_cloud_topic",
        "secondary_cloud_topic",
        "tertiary_cloud_topic",
        "wheel_odom_topic",
        "imu_topic",
        "inter_lidar_motion_compensation_source",
        "map_frame",
        "odom_frame",
        "base_frame",
        "pose_graph_marker_topic",
    ):
        if LaunchConfiguration(name).perform(context).strip():
            overrides[name] = LaunchConfiguration(name)

    for name, value_type in (
        ("use_sim_time", bool),
        ("use_secondary_cloud", bool),
        ("use_tertiary_cloud", bool),
        ("cloud_subscription_reliable", bool),
        ("cloud_subscription_depth", int),
        ("use_wheel_prediction", bool),
        ("publish_tf", bool),
    ):
        if LaunchConfiguration(name).perform(context).strip():
            overrides[name] = ParameterValue(
                LaunchConfiguration(name),
                value_type=value_type,
            )

    if "odom_frame" in overrides:
        overrides["wheel_odom_expected_frame"] = LaunchConfiguration(
            "odom_frame"
        )
    if "base_frame" in overrides:
        overrides["wheel_odom_expected_child_frame"] = LaunchConfiguration(
            "base_frame"
        )
    registration_method = LaunchConfiguration("registration_method").perform(
        context
    ).strip()
    if registration_method:
        overrides["registration_method"] = registration_method
    return overrides


def _launch_node(context):
    return [
        Node(
            package="slam_3d_core",
            executable="scan_to_map_icp_node",
            name="scan_to_map_icp_node",
            output=LaunchConfiguration("output"),
            parameters=[
                LaunchConfiguration("params_file"),
                _parameter_overrides(context),
            ],
        )
    ]


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
            default_value="",
            description="Optional override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "pose_graph_marker_topic",
            default_value="",
            description="Optional override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "primary_cloud_topic",
            default_value="",
            description="Optional primary cloud override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "secondary_cloud_topic",
            default_value="",
            description="Optional secondary cloud override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "tertiary_cloud_topic",
            default_value="",
            description="Optional tertiary cloud override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "use_secondary_cloud",
            default_value="",
            description="Optional front JT128 toggle; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "use_tertiary_cloud",
            default_value="",
            description="Optional rear JT128 toggle; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "cloud_subscription_reliable",
            default_value="",
            description="Optional cloud QoS override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "cloud_subscription_depth",
            default_value="",
            description="Optional cloud history override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "wheel_odom_topic",
            default_value="",
            description="Optional wheel odometry override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "use_wheel_prediction",
            default_value="",
            description="Optional wheel prediction toggle; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "inter_lidar_motion_compensation_source",
            default_value="",
            description=(
                "Optional inter-LiDAR motion source override; empty preserves "
                "params_file"
            ),
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="",
            description="Optional IMU override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "registration_method",
            default_value="",
            description=(
                "Optional registration override: gicp, icp, or voxel_surfel. "
                "Empty preserves params_file."
            ),
        ),
        DeclareLaunchArgument(
            "map_frame",
            default_value="",
            description="Optional map frame override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "odom_frame",
            default_value="",
            description="Optional odometry frame override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="",
            description="Optional robot base frame override; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="",
            description="Optional TF toggle; empty preserves params_file",
        ),
        DeclareLaunchArgument(
            "output",
            default_value="screen",
            description="Node output mode",
        ),
    ]

    return LaunchDescription(declarations + [OpaqueFunction(function=_launch_node)])
