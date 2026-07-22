#!/usr/bin/env python3

import os

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _bool_arg(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ("1", "true", "yes", "on")


def _resolve_launch_file(package_share, launch_file):
    if os.path.isabs(launch_file):
        return launch_file if os.path.exists(launch_file) else None

    candidates = [
        os.path.join(package_share, "launch", launch_file),
        os.path.join(package_share, "launch_ROS2", launch_file),
        os.path.join(package_share, launch_file),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _include_livox_driver(context):
    if not _bool_arg(context, "run_livox_driver"):
        return []

    package_name = LaunchConfiguration("livox_driver_package").perform(context)
    launch_file = LaunchConfiguration("livox_driver_launch_file").perform(context)

    try:
        package_share = get_package_share_directory(package_name)
    except PackageNotFoundError:
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] Livox driver package '{package_name}' was not found. "
                    "Use this launch as the GOLE SLAM architecture contract until sensor bringup "
                    "is installed and configured."
                )
            )
        ]

    launch_path = _resolve_launch_file(package_share, launch_file)
    if launch_path is None:
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] Livox launch file '{launch_file}' was not found in "
                    f"package '{package_name}'."
                )
            )
        ]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_path),
            launch_arguments={"use_sim_time": LaunchConfiguration("use_sim_time")}.items(),
        )
    ]


def _include_learning_slam_core(context):
    if not _bool_arg(context, "run_learning_slam_core"):
        return []

    try:
        package_share = get_package_share_directory("slam_3d_core")
    except PackageNotFoundError:
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] slam_3d_core was not found. Build "
                    "the GOLE learning SLAM core package before enabling "
                    "run_learning_slam_core."
                )
            )
        ]

    launch_path = os.path.join(package_share, "launch", "scan_to_map_icp.launch.py")
    if not os.path.exists(launch_path):
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] slam_3d_core launch file was not found: "
                    f"{launch_path}"
                )
            )
        ]

    params_file = LaunchConfiguration("learning_slam_params_file").perform(context)
    if not params_file:
        params_file = os.path.join(package_share, "config", "scan_to_map_icp_params.yaml")

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_path),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "params_file": params_file,
                "primary_cloud_topic": LaunchConfiguration(
                    "learning_slam_primary_cloud_topic"
                ),
                "secondary_cloud_topic": LaunchConfiguration(
                    "learning_slam_secondary_cloud_topic"
                ),
                "use_secondary_cloud": LaunchConfiguration(
                    "learning_slam_use_secondary_cloud"
                ),
                "registration_method": LaunchConfiguration(
                    "learning_slam_registration_method"
                ),
                "publish_tf": LaunchConfiguration("learning_slam_publish_tf"),
            }.items(),
        )
    ]


def _log_pipeline_contract(context):
    mode = LaunchConfiguration("pipeline_mode").perform(context)
    config_file = LaunchConfiguration("pipeline_config_file").perform(context)
    return [
        LogInfo(
            msg=(
                "[slam_3d_bringup] GOLE large-scale outdoor 3D SLAM pipeline mode: "
                f"{mode}. This launch owns sensor-driver/TF/EKF/RViz scaffolding "
                "and can optionally start the GOLE learning scan-to-map core; "
                "the production LIO frontend, submap manager, pose graph backend, "
                f"loop closure, and map store are specified in {config_file}."
            )
        )
    ]


def _static_tf_node(name, child_prefix):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="screen",
        arguments=[
            "--x",
            LaunchConfiguration(f"{child_prefix}_x"),
            "--y",
            LaunchConfiguration(f"{child_prefix}_y"),
            "--z",
            LaunchConfiguration(f"{child_prefix}_z"),
            "--roll",
            LaunchConfiguration(f"{child_prefix}_roll"),
            "--pitch",
            LaunchConfiguration(f"{child_prefix}_pitch"),
            "--yaw",
            LaunchConfiguration(f"{child_prefix}_yaw"),
            "--frame-id",
            LaunchConfiguration("base_frame"),
            "--child-frame-id",
            LaunchConfiguration(f"{child_prefix}_frame"),
        ],
        condition=IfCondition(LaunchConfiguration("run_static_tf")),
    )


def generate_launch_description():
    pkg_dir = get_package_share_directory("slam_3d_bringup")
    default_ekf_params = os.path.join(
        pkg_dir,
        "config",
        "dual_mid360_ekf_3d_params.yaml",
    )
    default_pipeline_config = os.path.join(
        pkg_dir,
        "config",
        "large_scale_outdoor_pipeline.yaml",
    )
    default_rviz_config = os.path.join(pkg_dir, "rviz", "dual_mid360_3d_slam.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")

    declarations = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation time",
        ),
        DeclareLaunchArgument(
            "pipeline_mode",
            default_value="architecture_only",
            description=(
                "Development mode label: architecture_only, sensor_recording, "
                "frontend_dev, mapping_dev, or localization_dev"
            ),
        ),
        DeclareLaunchArgument(
            "pipeline_config_file",
            default_value=default_pipeline_config,
            description="GOLE large-scale outdoor 3D SLAM architecture/config contract",
        ),
        DeclareLaunchArgument(
            "run_livox_driver",
            default_value="false",
            description="Include Livox ROS Driver 2 after two-Mid-360 device config is ready",
        ),
        DeclareLaunchArgument(
            "livox_driver_package",
            default_value="livox_ros_driver2",
            description="Package name for Livox ROS Driver 2",
        ),
        DeclareLaunchArgument(
            "livox_driver_launch_file",
            default_value="msg_MID360_launch.py",
            description="Livox ROS 2 launch file. launch/ and launch_ROS2/ are both searched.",
        ),
        DeclareLaunchArgument(
            "run_static_tf",
            default_value="false",
            description="Publish placeholder base_link -> Mid-360 static transforms",
        ),
        DeclareLaunchArgument(
            "run_wheel_imu_ekf",
            default_value="false",
            description=(
                "Run robot_localization EKF for wheel odom + external IMU as a reference "
                "stream. It must not own TF during 3D SLAM mapping."
            ),
        ),
        DeclareLaunchArgument(
            "ekf_params_file",
            default_value=default_ekf_params,
            description="3D robot_localization EKF parameter file",
        ),
        DeclareLaunchArgument(
            "run_learning_slam_core",
            default_value="false",
            description="Run the GOLE-owned learning scan-to-map ICP/GICP SLAM core",
        ),
        DeclareLaunchArgument(
            "learning_slam_params_file",
            default_value="",
            description=(
                "Parameter file for slam_3d_core. Empty uses the package default "
                "scan_to_map_icp_params.yaml."
            ),
        ),
        DeclareLaunchArgument(
            "learning_slam_primary_cloud_topic",
            default_value="/slam_3d/input/mid360_front/points_debug",
            description="Primary PointCloud2 stream for the learning SLAM core",
        ),
        DeclareLaunchArgument(
            "learning_slam_secondary_cloud_topic",
            default_value="/slam_3d/input/mid360_rear/points_debug",
            description="Secondary PointCloud2 stream for the learning SLAM core",
        ),
        DeclareLaunchArgument(
            "learning_slam_use_secondary_cloud",
            default_value="true",
            description="Fuse the secondary cloud in the learning SLAM core",
        ),
        DeclareLaunchArgument(
            "learning_slam_registration_method",
            default_value="gicp",
            description="Learning core registration method: gicp or icp",
        ),
        DeclareLaunchArgument(
            "learning_slam_publish_tf",
            default_value="false",
            description=(
                "Allow slam_3d_core to publish TF. Keep false unless existing "
                "localization/AMCL TF owners are disabled."
            ),
        ),
        DeclareLaunchArgument(
            "run_rviz",
            default_value="false",
            description="Start RViz with the 3D SLAM development display",
        ),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=default_rviz_config,
            description="RViz config for 3D SLAM development",
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Robot base frame",
        ),
        DeclareLaunchArgument(
            "front_frame",
            default_value="mid360_front_link",
            description="Front Mid-360 body frame after measured extrinsic calibration",
        ),
        DeclareLaunchArgument(
            "front_x",
            default_value="0.45",
            description="Front Mid-360 x offset from base_link [m]",
        ),
        DeclareLaunchArgument(
            "front_y",
            default_value="0.0",
            description="Front Mid-360 y offset [m]",
        ),
        DeclareLaunchArgument(
            "front_z",
            default_value="0.52",
            description="Front Mid-360 z offset [m]",
        ),
        DeclareLaunchArgument(
            "front_roll",
            default_value="0.0",
            description="Front Mid-360 roll [rad]",
        ),
        DeclareLaunchArgument(
            "front_pitch",
            default_value="0.0",
            description="Front Mid-360 pitch [rad]",
        ),
        DeclareLaunchArgument(
            "front_yaw",
            default_value="0.0",
            description="Front Mid-360 yaw [rad]",
        ),
        DeclareLaunchArgument(
            "rear_frame",
            default_value="mid360_rear_link",
            description="Rear Mid-360 body frame after measured extrinsic calibration",
        ),
        DeclareLaunchArgument(
            "rear_x",
            default_value="-0.45",
            description="Rear Mid-360 x offset [m]",
        ),
        DeclareLaunchArgument(
            "rear_y",
            default_value="0.0",
            description="Rear Mid-360 y offset [m]",
        ),
        DeclareLaunchArgument(
            "rear_z",
            default_value="0.52",
            description="Rear Mid-360 z offset [m]",
        ),
        DeclareLaunchArgument(
            "rear_roll",
            default_value="0.0",
            description="Rear Mid-360 roll [rad]",
        ),
        DeclareLaunchArgument(
            "rear_pitch",
            default_value="0.0",
            description="Rear Mid-360 pitch [rad]",
        ),
        DeclareLaunchArgument(
            "rear_yaw",
            default_value="3.141592653589793",
            description="Rear Mid-360 yaw [rad]",
        ),
    ]

    wheel_imu_ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="slam_3d_wheel_imu_ekf_node",
        output="screen",
        parameters=[
            LaunchConfiguration("ekf_params_file"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("odometry/filtered", "/slam_3d/reference/wheel_imu_odom"),
            ("accel/filtered", "/slam_3d/reference/wheel_imu_accel"),
        ],
        condition=IfCondition(LaunchConfiguration("run_wheel_imu_ekf")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="slam_3d_rviz",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config_file")],
        condition=IfCondition(LaunchConfiguration("run_rviz")),
    )

    return LaunchDescription(
        declarations
        + [
            OpaqueFunction(function=_log_pipeline_contract),
            _static_tf_node("base_to_mid360_front_tf", "front"),
            _static_tf_node("base_to_mid360_rear_tf", "rear"),
            wheel_imu_ekf,
            OpaqueFunction(function=_include_livox_driver),
            OpaqueFunction(function=_include_learning_slam_core),
            rviz,
        ]
    )
