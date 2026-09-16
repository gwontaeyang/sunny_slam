#!/usr/bin/env python3
"""Run the project-owned 2D SLAM node with existing sensor and odometry TF."""

import math
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# Empty optional arguments preserve the selected YAML profile.
PARAMETER_ARGUMENTS = {
    "throttle_scans": ("throttle_scans", int),
    "resolution": ("resolution", float),
    "roughness_innovation_mode": ("roughness_innovation_mode", str),
    "roughness_innovation_imu_topic": ("roughness_imu_topic", str),
    "roughness_innovation_translation_gate_m": ("roughness_translation_innovation_gate_m", float),
    "roughness_innovation_yaw_gate_deg": ("roughness_yaw_innovation_gate_deg", float),
    "roughness_innovation_disable_loop_closing_during_gate": (
        "roughness_disable_loop_closing_during_near_chain_gate", bool),
}
for _suffix, _type in (
    ("scan_qos_reliable", bool), ("scan_qos_depth", int),
    ("window_before_s", float), ("window_after_s", float),
    ("score_gate", float), ("min_rough_samples", int),
    ("min_rough_duration_s", float), ("enable_near_chain_gate", bool),
    ("near_chain_recovery_duration_s", float),
    ("near_chain_loop_closing_cooldown_s", float),
    ("near_chain_min_rough_samples", int), ("near_chain_gated_min_response", float),
    ("enable_spatial_memory_gate", bool), ("spatial_memory_use_pose_graph_anchors", bool),
    ("spatial_memory_radius_m", float), ("spatial_memory_min_revisit_separation_s", float),
    ("spatial_memory_min_heading_delta_deg", float), ("spatial_memory_merge_radius_m", float),
    ("spatial_memory_max_zones", int), ("enable_spatial_passage_hysteresis", bool),
    ("spatial_passage_radius_m", float), ("spatial_passage_max_age_s", float),
    ("enable_post_confirmation", bool), ("post_confirmation_max_age_s", float),
    ("post_confirmation_min_base_scans", int), ("post_confirmation_max_base_scans", int),
    ("post_confirmation_min_response", float), ("post_confirmation_min_response_gain", float),
    ("post_confirmation_min_covariance_improvement_ratio", float),
    ("post_confirmation_translation_disagreement_gate_m", float),
    ("post_confirmation_yaw_disagreement_gate_deg", float),
    ("experimental_reject_scan_stamps_ns", str),
):
    PARAMETER_ARGUMENTS["roughness_innovation_" + _suffix] = ("roughness_" + _suffix, _type)


def parse_bool(value, name):
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


def launch_slam(context):
    def value(name):
        return LaunchConfiguration(name).perform(context).strip()

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

    overrides = {
        "use_sim_time": parse_bool(value("use_sim_time"), "use_sim_time"),
        "use_lifecycle_manager": False,
        "scan_topic": value("scan_topic"),
        "base_frame": value("base_frame"),
    }
    for argument, (parameter, kind) in PARAMETER_ARGUMENTS.items():
        raw = value(argument)
        if not raw:
            continue
        try:
            parsed = parse_bool(raw, argument) if kind is bool else kind(raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid {argument}: {raw}") from exc
        if kind is float and not math.isfinite(parsed):
            raise RuntimeError(f"{argument} must be finite")
        if argument in {"throttle_scans", "resolution", "roughness_innovation_scan_qos_depth"}:
            if parsed <= 0:
                raise RuntimeError(f"{argument} must be positive")
        overrides[parameter] = parsed
    effective.update(overrides)
    for dependent, required in (
        ("roughness_enable_spatial_memory_gate", "roughness_enable_near_chain_gate"),
        ("roughness_spatial_memory_use_pose_graph_anchors", "roughness_enable_spatial_memory_gate"),
        ("roughness_enable_spatial_passage_hysteresis", "roughness_enable_spatial_memory_gate"),
    ):
        if effective.get(dependent, False) and not effective.get(required, False):
            raise RuntimeError(f"{dependent} requires {required}")

    return [Node(
        package="slam_2d_core",
        executable="roughness_aware_slam_toolbox_node",
        name="slam_toolbox",
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
    )]

    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=matches_action(slam_toolbox_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
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
