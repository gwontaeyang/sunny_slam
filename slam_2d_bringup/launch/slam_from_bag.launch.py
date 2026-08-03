#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    NotSubstitution,
    PythonExpression,
)
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    slam_2d_bringup_dir = get_package_share_directory("slam_2d_bringup")
    default_params_file = os.path.join(
        slam_2d_bringup_dir,
        "config",
        "mapper_params.yaml",
    )

    bag_file = LaunchConfiguration("bag_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_slam = LaunchConfiguration("start_slam")
    scan_topic = LaunchConfiguration("scan_topic")
    base_frame = LaunchConfiguration("base_frame")
    use_static_map_odom = LaunchConfiguration("use_static_map_odom")
    bag_play_rate = LaunchConfiguration("bag_play_rate")
    params_file = LaunchConfiguration("params_file")

    declare_bag_file = DeclareLaunchArgument(
        "bag_file",
        default_value="",
        description="Path to the bag file to play. Leave empty to skip playback.",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock from bag playback.",
    )
    declare_start_slam = DeclareLaunchArgument(
        "start_slam",
        default_value="true",
        description="Start slam_toolbox before playing the bag.",
    )
    declare_scan_topic = DeclareLaunchArgument(
        "scan_topic",
        default_value="/scan_laserscan",
        description="LaserScan topic used by slam_toolbox.",
    )
    declare_base_frame = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="Robot base frame used by slam_toolbox.",
    )
    declare_use_static_map_odom = DeclareLaunchArgument(
        "use_static_map_odom",
        default_value="false",
        description="Publish a static map->odom transform for offline debugging.",
    )
    declare_bag_play_rate = DeclareLaunchArgument(
        "bag_play_rate",
        default_value="1.0",
        description="Bag playback rate.",
    )
    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to slam_toolbox params YAML file.",
    )

    slam_toolbox_node = LifecycleNode(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        namespace="",
        parameters=[
            params_file,
            {
                "use_sim_time": use_sim_time,
                "scan_topic": scan_topic,
                "base_frame": base_frame,
            },
        ],
        condition=IfCondition(start_slam),
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(start_slam),
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[slam_2d_bringup] Activating slam_toolbox."),
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(slam_toolbox_node),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    )
                ),
            ],
        ),
        condition=IfCondition(start_slam),
    )

    bag_play_after_slam = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=slam_toolbox_node,
            start_state="activating",
            goal_state="active",
            entities=[
                LogInfo(
                    msg="[slam_2d_bringup] SLAM is active. Starting bag playback."),
                TimerAction(
                    period=3.0,
                    actions=[
                        ExecuteProcess(
                            cmd=[
                                "ros2",
                                "bag",
                                "play",
                                bag_file,
                                "--clock",
                                "--rate",
                                bag_play_rate,
                            ],
                            output="screen",
                        )
                    ],
                ),
            ],
        ),
        condition=IfCondition(
            AndSubstitution(
                start_slam,
                PythonExpression(['"', bag_file, '" != ""']),
            )
        ),
    )

    bag_play_only = ExecuteProcess(
        cmd=["ros2", "bag", "play", bag_file, "--clock", "--rate", bag_play_rate],
        output="screen",
        condition=IfCondition(
            AndSubstitution(
                PythonExpression(['"', bag_file, '" != ""']),
                NotSubstitution(start_slam),
            )
        ),
    )

    static_map_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_map_odom_tf",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        output="screen",
        condition=IfCondition(use_static_map_odom),
    )

    return LaunchDescription(
        [
            declare_bag_file,
            declare_use_sim_time,
            declare_start_slam,
            declare_scan_topic,
            declare_base_frame,
            declare_use_static_map_odom,
            declare_bag_play_rate,
            declare_params_file,
            slam_toolbox_node,
            static_map_odom_tf,
            configure_event,
            activate_event,
            bag_play_after_slam,
            bag_play_only,
            LogInfo(msg=["SLAM scan topic: ", scan_topic]),
        ]
    )
