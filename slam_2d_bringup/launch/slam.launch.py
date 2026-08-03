#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import AndSubstitution, LaunchConfiguration, NotSubstitution
from launch_ros.actions import LifecycleNode
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

    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    use_lifecycle_manager = LaunchConfiguration("use_lifecycle_manager")
    params_file = LaunchConfiguration("params_file")
    scan_topic = LaunchConfiguration("scan_topic")
    base_frame = LaunchConfiguration("base_frame")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock when true.",
    )
    declare_type = DeclareLaunchArgument(
        "type",
        default_value="apart",
        description="Autonomy system type. Reserved for shared bringup compatibility.",
    )
    declare_robot_model = DeclareLaunchArgument(
        "robot_model",
        default_value="rover",
        description="Robot model name. Reserved for shared bringup compatibility.",
    )
    declare_mode = DeclareLaunchArgument(
        "mode",
        default_value="SLAM",
        description="Autonomy mode. Reserved for shared bringup compatibility.",
    )
    declare_autostart = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically configure and activate slam_toolbox.",
    )
    declare_use_lifecycle_manager = DeclareLaunchArgument(
        "use_lifecycle_manager",
        default_value="false",
        description="Leave lifecycle transitions to an external manager when true.",
    )
    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=default_params_file,
        description="Path to slam_toolbox params YAML file.",
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
                "use_lifecycle_manager": use_lifecycle_manager,
            },
        ],
    )

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))),
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
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager))),
    )

    return LaunchDescription(
        [
            declare_use_sim_time,
            declare_robot_model,
            declare_type,
            declare_mode,
            declare_autostart,
            declare_use_lifecycle_manager,
            declare_params_file,
            declare_scan_topic,
            declare_base_frame,
            slam_toolbox_node,
            configure_event,
            activate_event,
        ]
    )
