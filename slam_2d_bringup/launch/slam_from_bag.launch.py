#!/usr/bin/env python3
"""Replay LaserScan, IMU and odometry into the project-owned 2D SLAM node."""

import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def value(context, name, default=""):
    return LaunchConfiguration(name, default=default).perform(context).strip()


def boolean(context, name, default="false"):
    raw = value(context, name, default).lower()
    if raw not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
        raise RuntimeError(f"{name} must be a boolean")
    return raw in {"true", "1", "yes", "on"}


def validate_replay(context):
    start_slam = boolean(context, "start_slam")
    static_tf = boolean(context, "use_static_map_odom")
    if start_slam and static_tf:
        raise RuntimeError("SLAM and a static map->odom publisher cannot run together")
    if boolean(context, "start_pointcloud_projection"):
        raise RuntimeError("Replay expects LaserScan input; run pointcloud preprocessing separately")
    bag = value(context, "bag_file")
    if bag and not Path(bag).expanduser().exists():
        raise RuntimeError(f"Bag path does not exist: {bag}")
    if bag and not boolean(context, "use_sim_time"):
        raise RuntimeError("Bag replay requires use_sim_time:=true")
    for name, allow_zero in (("bag_play_rate", False), ("bag_start_offset_s", True),
                             ("slam_ready_timeout_s", False)):
        try:
            number = float(value(context, name))
        except ValueError as exc:
            raise RuntimeError(f"{name} must be numeric") from exc
        if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
            raise RuntimeError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    actions = []
    if value(context, "slam_scan_topic"):
        actions.append(SetLaunchConfiguration("scan_topic", value(context, "slam_scan_topic")))
    if static_tf:
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="static_map_odom_tf", output="screen",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--frame-id", "map", "--child-frame-id", "odom"],
        ))
    return actions


def playback_actions(context):
    bag = value(context, "bag_file")
    if not bag:
        return []
    scan_topic = value(context, "scan_topic", "/scan_laserscan")
    imu_topic = value(context, "roughness_innovation_imu_topic")
    if not imu_topic:
        params = value(context, "roughness_innovation_params_file")
        profile = yaml.safe_load(Path(params).expanduser().read_text()) if params else {}
        imu_topic = (profile or {}).get("slam_toolbox", {}).get("ros__parameters", {}).get(
            "roughness_imu_topic", "/imu/data")
    if boolean(context, "start_slam"):
        share = Path(get_package_share_directory("slam_2d_bringup"))
        command = ["python3", str(share / "scripts/play_after_slam_ready.py"),
                   "--bag", str(Path(bag).expanduser()),
                   "--rate", value(context, "bag_play_rate"),
                   "--start-offset", value(context, "bag_start_offset_s"),
                   "--scan-topic", scan_topic,
                   "--timeout", value(context, "slam_ready_timeout_s")]
    else:
        command = ["ros2", "bag", "play", str(Path(bag).expanduser()), "--clock",
                   "--rate", value(context, "bag_play_rate"),
                   "--start-offset", value(context, "bag_start_offset_s"),
                   "--disable-keyboard-controls"]
    if boolean(context, "play_slam_topics_only"):
        topics = [scan_topic, str(imu_topic), "/tf", "/tf_static", "/odom",
                  "/motor_driver_node/odom", "/imu/data_raw"]
        command += ["--topics", *dict.fromkeys(topics)]
    return [ExecuteProcess(cmd=command, output="screen")]


def generate_launch_description():
    share = Path(get_package_share_directory("slam_2d_bringup"))
    return LaunchDescription([
        DeclareLaunchArgument("bag_file", default_value="",
                              description="Bag path; empty starts SLAM without playback."),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("start_slam", default_value="true"),
        DeclareLaunchArgument("scan_topic", default_value="/scan_laserscan"),
        DeclareLaunchArgument("slam_scan_topic", default_value="",
                              description="Legacy scan-topic override; empty uses scan_topic."),
        DeclareLaunchArgument("bag_play_rate", default_value="1.0"),
        DeclareLaunchArgument("bag_start_offset_s", default_value="0.0"),
        DeclareLaunchArgument("slam_ready_timeout_s", default_value="30.0"),
        DeclareLaunchArgument("play_slam_topics_only", default_value="true"),
        DeclareLaunchArgument("use_static_map_odom", default_value="false"),
        DeclareLaunchArgument("start_pointcloud_projection", default_value="false",
                              description="Legacy option; preprocessing must be launched separately."),
        OpaqueFunction(function=validate_replay),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / "launch/slam.launch.py")),
            condition=IfCondition(LaunchConfiguration("start_slam")),
        ),
        OpaqueFunction(function=playback_actions),
    ])
