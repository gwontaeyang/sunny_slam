#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import (
    AndSubstitution,
    LaunchConfiguration,
    NotSubstitution,
)
from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition


def validate_roughness_innovation_frontend(context):
    truthy = {"1", "true", "yes", "on"}
    custom_enabled = LaunchConfiguration(
        "use_roughness_innovation_slam_toolbox"
    ).perform(context).strip().lower() in truthy
    near_chain_enabled = LaunchConfiguration(
        "roughness_innovation_enable_near_chain_gate",
        default="false",
    ).perform(context).strip().lower() in truthy
    spatial_memory_enabled = LaunchConfiguration(
        "roughness_innovation_enable_spatial_memory_gate",
        default="false",
    ).perform(context).strip().lower() in truthy
    pose_graph_anchors_raw = LaunchConfiguration(
        "roughness_innovation_spatial_memory_use_pose_graph_anchors",
        default="false",
    ).perform(context).strip().lower()
    if pose_graph_anchors_raw not in truthy | {"0", "false", "no", "off"}:
        raise RuntimeError(
            "roughness_innovation_spatial_memory_use_pose_graph_anchors "
            "must be a boolean"
        )
    pose_graph_anchors_enabled = pose_graph_anchors_raw in truthy
    spatial_passage_enabled = LaunchConfiguration(
        "roughness_innovation_enable_spatial_passage_hysteresis",
        default="false",
    ).perform(context).strip().lower() in truthy
    if (
        near_chain_enabled
        or spatial_memory_enabled
        or pose_graph_anchors_enabled
        or spatial_passage_enabled
    ) and not custom_enabled:
        raise RuntimeError(
            "roughness innovation near-chain, spatial memory, and passage "
            "features require use_roughness_innovation_slam_toolbox"
        )
    if not custom_enabled:
        return []

    lifecycle_manager_enabled = LaunchConfiguration(
        "use_lifecycle_manager"
    ).perform(context).strip().lower() in truthy
    if lifecycle_manager_enabled:
        raise RuntimeError(
            "the roughness innovation slam_toolbox frontend self-configures "
            "and cannot use an external lifecycle manager"
        )

    autostart_enabled = LaunchConfiguration(
        "autostart"
    ).perform(context).strip().lower() in truthy
    if not autostart_enabled:
        raise RuntimeError(
            "autostart must be true for the roughness innovation "
            "slam_toolbox frontend"
        )
    if spatial_memory_enabled and not near_chain_enabled:
        raise RuntimeError(
            "roughness_innovation_enable_spatial_memory_gate requires "
            "roughness_innovation_enable_near_chain_gate"
        )
    if pose_graph_anchors_enabled and not spatial_memory_enabled:
        raise RuntimeError(
            "roughness_innovation_spatial_memory_use_pose_graph_anchors "
            "requires roughness_innovation_enable_spatial_memory_gate"
        )
    if spatial_passage_enabled and (
        not near_chain_enabled or not spatial_memory_enabled
    ):
        raise RuntimeError(
            "roughness_innovation_enable_spatial_passage_hysteresis requires "
            "roughness_innovation_enable_near_chain_gate and "
            "roughness_innovation_enable_spatial_memory_gate"
        )
    return []


def generate_launch_description():
    slam_2d_bringup_dir = get_package_share_directory("slam_2d_bringup")
    default_params_file = os.path.join(
        slam_2d_bringup_dir,
        "config",
        "mapper_params.yaml",
    )
    default_roughness_innovation_params_file = os.path.join(
        slam_2d_bringup_dir,
        "config",
        "slam_roughness_innovation_params.yaml",
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    use_lifecycle_manager = LaunchConfiguration("use_lifecycle_manager")
    params_file = LaunchConfiguration("params_file")
    scan_topic = LaunchConfiguration("scan_topic")
    base_frame = LaunchConfiguration("base_frame")
    use_roughness_innovation_slam_toolbox = LaunchConfiguration(
        "use_roughness_innovation_slam_toolbox"
    )
    roughness_innovation_params_file = LaunchConfiguration(
        "roughness_innovation_params_file"
    )
    roughness_innovation_imu_topic = LaunchConfiguration(
        "roughness_innovation_imu_topic"
    )
    roughness_innovation_scan_qos_reliable = LaunchConfiguration(
        "roughness_innovation_scan_qos_reliable"
    )
    roughness_innovation_scan_qos_depth = LaunchConfiguration(
        "roughness_innovation_scan_qos_depth"
    )
    roughness_innovation_near_chain_loop_closing_cooldown_s = (
        LaunchConfiguration(
            "roughness_innovation_near_chain_loop_closing_cooldown_s"
        )
    )
    roughness_innovation_enable_near_chain_gate = LaunchConfiguration(
        "roughness_innovation_enable_near_chain_gate"
    )
    roughness_innovation_enable_spatial_memory_gate = LaunchConfiguration(
        "roughness_innovation_enable_spatial_memory_gate"
    )
    roughness_innovation_spatial_memory_use_pose_graph_anchors = (
        LaunchConfiguration(
            "roughness_innovation_spatial_memory_use_pose_graph_anchors"
        )
    )
    roughness_innovation_enable_spatial_passage_hysteresis = (
        LaunchConfiguration(
            "roughness_innovation_enable_spatial_passage_hysteresis"
        )
    )
    roughness_innovation_spatial_passage_radius_m = LaunchConfiguration(
        "roughness_innovation_spatial_passage_radius_m"
    )
    roughness_innovation_spatial_passage_max_age_s = LaunchConfiguration(
        "roughness_innovation_spatial_passage_max_age_s"
    )
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
    declare_use_roughness_innovation_slam_toolbox = DeclareLaunchArgument(
        "use_roughness_innovation_slam_toolbox",
        default_value="true",
        description=(
            "Use the custom 2D SLAM frontend (default). Set false for stock Toolbox."
        ),
    )
    declare_roughness_innovation_params_file = DeclareLaunchArgument(
        "roughness_innovation_params_file",
        default_value=default_roughness_innovation_params_file,
        description="Parameters for the roughness innovation frontend.",
    )
    declare_roughness_innovation_imu_topic = DeclareLaunchArgument(
        "roughness_innovation_imu_topic",
        default_value="/imu/data",
        description=(
            "IMU topic used only to classify roughness in the custom "
            "slam_toolbox frontend."
        ),
    )
    declare_roughness_innovation_scan_qos_reliable = DeclareLaunchArgument(
        "roughness_innovation_scan_qos_reliable",
        default_value="false",
        description=(
            "Request reliable LaserScan QoS only when the publisher offers it."
        ),
    )
    declare_roughness_innovation_scan_qos_depth = DeclareLaunchArgument(
        "roughness_innovation_scan_qos_depth",
        default_value="5",
        description="LaserScan subscription history depth.",
    )
    declare_roughness_innovation_near_chain_loop_closing_cooldown_s = (
        DeclareLaunchArgument(
            "roughness_innovation_near_chain_loop_closing_cooldown_s",
            default_value="0.0",
            description=(
                "Keep loop closing suppressed after the near-chain threshold "
                "gate releases."
            ),
        )
    )
    declare_roughness_innovation_enable_near_chain_gate = (
        DeclareLaunchArgument(
            "roughness_innovation_enable_near_chain_gate",
            default_value="false",
            description=(
                "Apply the roughness-aware near-chain scan-matching gate."
            ),
        )
    )
    declare_roughness_innovation_enable_spatial_memory_gate = (
        DeclareLaunchArgument(
            "roughness_innovation_enable_spatial_memory_gate",
            default_value="false",
            description=(
                "Remember odometry locations of rough scans for spatial "
                "near-chain protection."
            ),
        )
    )
    declare_roughness_innovation_spatial_memory_use_pose_graph_anchors = (
        DeclareLaunchArgument(
            "roughness_innovation_spatial_memory_use_pose_graph_anchors",
            default_value="false",
            description=(
                "Anchor rough zones to corrected SLAM pose-graph scans so "
                "loop-closure corrections also move the remembered zones."
            ),
        )
    )
    declare_roughness_innovation_enable_spatial_passage_hysteresis = (
        DeclareLaunchArgument(
            "roughness_innovation_enable_spatial_passage_hysteresis",
            default_value="false",
            description=(
                "Extend near-chain recovery while traversing a recent "
                "remembered rough zone."
            ),
        )
    )
    declare_roughness_innovation_spatial_passage_radius_m = (
        DeclareLaunchArgument(
            "roughness_innovation_spatial_passage_radius_m",
            default_value="0.55",
            description=(
                "Maximum odometry distance for spatial passage hysteresis."
            ),
        )
    )
    declare_roughness_innovation_spatial_passage_max_age_s = (
        DeclareLaunchArgument(
            "roughness_innovation_spatial_passage_max_age_s",
            default_value="3.5",
            description=(
                "Maximum age of a rough zone used for passage hysteresis."
            ),
        )
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
        condition=IfCondition(
            NotSubstitution(use_roughness_innovation_slam_toolbox)
        ),
    )

    roughness_innovation_slam_toolbox_node = Node(
        package="slam_2d_core",
        executable="roughness_aware_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        namespace="",
        parameters=[
            params_file,
            roughness_innovation_params_file,
            {
                "use_sim_time": use_sim_time,
                "scan_topic": scan_topic,
                "base_frame": base_frame,
                "use_lifecycle_manager": False,
                "enable_roughness_innovation_gate": True,
                "roughness_imu_topic": roughness_innovation_imu_topic,
                "roughness_scan_qos_reliable":
                    roughness_innovation_scan_qos_reliable,
                "roughness_scan_qos_depth":
                    roughness_innovation_scan_qos_depth,
                "roughness_near_chain_loop_closing_cooldown_s":
                    roughness_innovation_near_chain_loop_closing_cooldown_s,
                "roughness_enable_near_chain_gate":
                    roughness_innovation_enable_near_chain_gate,
                "roughness_enable_spatial_memory_gate":
                    roughness_innovation_enable_spatial_memory_gate,
                "roughness_spatial_memory_use_pose_graph_anchors":
                    roughness_innovation_spatial_memory_use_pose_graph_anchors,
                "roughness_enable_spatial_passage_hysteresis":
                    roughness_innovation_enable_spatial_passage_hysteresis,
                "roughness_spatial_passage_radius_m":
                    roughness_innovation_spatial_passage_radius_m,
                "roughness_spatial_passage_max_age_s":
                    roughness_innovation_spatial_passage_max_age_s,
            },
        ],
        condition=IfCondition(use_roughness_innovation_slam_toolbox),
    )

    standard_autostart_condition = AndSubstitution(
        autostart,
        AndSubstitution(
            NotSubstitution(use_lifecycle_manager),
            NotSubstitution(use_roughness_innovation_slam_toolbox),
        ),
    )
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        ),
        condition=IfCondition(standard_autostart_condition),
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
        condition=IfCondition(standard_autostart_condition),
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
            declare_use_roughness_innovation_slam_toolbox,
            declare_roughness_innovation_params_file,
            declare_roughness_innovation_imu_topic,
            declare_roughness_innovation_scan_qos_reliable,
            declare_roughness_innovation_scan_qos_depth,
            declare_roughness_innovation_near_chain_loop_closing_cooldown_s,
            declare_roughness_innovation_enable_near_chain_gate,
            declare_roughness_innovation_enable_spatial_memory_gate,
            declare_roughness_innovation_spatial_memory_use_pose_graph_anchors,
            declare_roughness_innovation_enable_spatial_passage_hysteresis,
            declare_roughness_innovation_spatial_passage_radius_m,
            declare_roughness_innovation_spatial_passage_max_age_s,
            OpaqueFunction(function=validate_roughness_innovation_frontend),
            slam_toolbox_node,
            roughness_innovation_slam_toolbox_node,
            configure_event,
            activate_event,
        ]
    )
