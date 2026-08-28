#!/usr/bin/env python3

import os

from ament_index_python.packages import PackageNotFoundError
from ament_index_python.packages import get_package_prefix
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo
from launch.actions import OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _bool_arg(context, name):
    return LaunchConfiguration(name).perform(context).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _is_indoor_profile(context):
    profile = context.launch_configurations.get(
        "learning_slam_sensor_profile", "custom"
    ).strip()
    return profile.startswith("indoor_")


def _resolve_pipeline_config(context):
    configured = LaunchConfiguration("pipeline_config_file").perform(context).strip()
    if configured and configured != "auto":
        return configured

    package_share = get_package_share_directory("slam_3d_bringup")
    config_name = (
        "indoor_pipeline.yaml"
        if _is_indoor_profile(context)
        else "large_scale_outdoor_pipeline.yaml"
    )
    return os.path.join(package_share, "config", config_name)


def _include_hesai_driver(context):
    if not _bool_arg(context, "run_hesai_driver"):
        return []

    config_file = LaunchConfiguration("hesai_driver_config_file").perform(context)
    if not config_file or not os.path.isfile(config_file):
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] run_hesai_driver requires an existing "
                    "hesai_driver_config_file. Hardware receive was not started."
                )
            )
        ]

    package_name = LaunchConfiguration("hesai_driver_package").perform(context)
    try:
        package_prefix = get_package_prefix(package_name)
    except PackageNotFoundError:
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] Hesai driver package '{package_name}' "
                    "was not found."
                )
            )
        ]

    executable_name = "hesai_ros_driver_node"
    executable = os.path.join(
        package_prefix,
        "lib",
        package_name,
        executable_name,
    )
    if not os.path.isfile(executable):
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] '{executable_name}' was not found in "
                    f"package '{package_name}'."
                )
            )
        ]

    return [
        Node(
            package=package_name,
            executable=executable_name,
            name="hesai_ros_driver_node",
            output="screen",
            parameters=[{"config_path": config_file}],
        )
    ]


def _include_lidar_ingest(context):
    if not _bool_arg(context, "run_lidar_ingest"):
        return []

    package_name = "slam_3d_lidar_ingest"
    try:
        package_prefix = get_package_prefix(package_name)
        package_share = get_package_share_directory(package_name)
    except PackageNotFoundError:
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] slam_3d_lidar_ingest was not found. "
                    "Build the ingest package before enabling run_lidar_ingest."
                )
            )
        ]

    executable_name = "hesai_pointcloud_ingest_node"
    executable = os.path.join(
        package_prefix,
        "lib",
        package_name,
        executable_name,
    )
    if not os.path.isfile(executable):
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] '{executable_name}' is unavailable in "
                    "slam_3d_lidar_ingest."
                )
            )
        ]

    params_file = LaunchConfiguration("lidar_ingest_params_file").perform(context)
    if not params_file:
        params_file = os.path.join(
            package_share,
            "config",
            "hesai_jt128_xt32_ingest_params.yaml",
        )

    return [
        Node(
            package=package_name,
            executable=executable_name,
            name="hesai_pointcloud_ingest",
            output="screen",
            parameters=[
                params_file,
                {
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                    "front.input_topic": LaunchConfiguration(
                        "hesai_front_raw_topic"
                    ),
                    "rear.input_topic": LaunchConfiguration(
                        "hesai_rear_raw_topic"
                    ),
                    "center.input_topic": LaunchConfiguration(
                        "hesai_center_raw_topic"
                    ),
                    "front.output_topic": LaunchConfiguration(
                        "hesai_front_timed_topic"
                    ),
                    "rear.output_topic": LaunchConfiguration(
                        "hesai_rear_timed_topic"
                    ),
                    "center.output_topic": LaunchConfiguration(
                        "hesai_center_timed_topic"
                    ),
                    "front.expected_input_frame_id": LaunchConfiguration(
                        "front_frame"
                    ),
                    "rear.expected_input_frame_id": LaunchConfiguration(
                        "rear_frame"
                    ),
                    "center.expected_input_frame_id": LaunchConfiguration(
                        "center_frame"
                    ),
                    "health_topic": LaunchConfiguration(
                        "lidar_ingest_health_topic"
                    ),
                },
            ],
        )
    ]


def _include_lidar_imu_normalizer(context):
    if not _bool_arg(context, "run_lidar_imu_normalizer"):
        return []

    package_name = "slam_3d_lidar_ingest"
    try:
        package_prefix = get_package_prefix(package_name)
        package_share = get_package_share_directory(package_name)
    except PackageNotFoundError:
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] slam_3d_lidar_ingest was not found. "
                    "Build it before enabling run_lidar_imu_normalizer."
                )
            )
        ]

    executable_name = "hesai_imu_normalizer_node"
    executable = os.path.join(
        package_prefix,
        "lib",
        package_name,
        executable_name,
    )
    if not os.path.isfile(executable):
        return [
            LogInfo(
                msg=(
                    f"[slam_3d_bringup] '{executable_name}' is unavailable in "
                    "slam_3d_lidar_ingest."
                )
            )
        ]

    params_file = LaunchConfiguration(
        "lidar_imu_normalizer_params_file"
    ).perform(context)
    if not params_file:
        params_file = os.path.join(
            package_share,
            "config",
            "hesai_lidar_imu_normalizer_params.yaml",
        )

    return [
        Node(
            package=package_name,
            executable=executable_name,
            name="hesai_imu_normalizer",
            output="screen",
            parameters=[
                params_file,
                {
                    "use_sim_time": ParameterValue(
                        LaunchConfiguration("use_sim_time"),
                        value_type=bool,
                    ),
                    "input_topic": LaunchConfiguration(
                        "hesai_lidar_imu_raw_topic"
                    ),
                    "output_topic": LaunchConfiguration(
                        "hesai_lidar_imu_timed_topic"
                    ),
                    "expected_input_frame_id": LaunchConfiguration(
                        "hesai_lidar_imu_frame"
                    ),
                    "output_frame_id": LaunchConfiguration(
                        "hesai_lidar_imu_frame"
                    ),
                    "health_topic": LaunchConfiguration(
                        "lidar_imu_health_topic"
                    ),
                },
            ],
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
                    "[slam_3d_bringup] slam_3d_core was not found. Build it "
                    "before enabling run_learning_slam_core."
                )
            )
        ]

    launch_path = os.path.join(
        package_share,
        "launch",
        "scan_to_map_icp.launch.py",
    )
    if not os.path.isfile(launch_path):
        return [
            LogInfo(
                msg=f"[slam_3d_bringup] slam_3d_core launch is missing: {launch_path}"
            )
        ]

    profile = context.launch_configurations.get(
        "learning_slam_sensor_profile", "custom"
    ).strip()
    supported_profiles = {
        "custom",
        "indoor_xt32_single",
        "indoor_dual_jt128",
    }
    if profile not in supported_profiles:
        supported = ", ".join(sorted(supported_profiles))
        raise RuntimeError(
            f"Unsupported learning_slam_sensor_profile '{profile}'. "
            f"Expected one of: {supported}"
        )

    params_file = LaunchConfiguration("learning_slam_params_file").perform(context)
    if not params_file and profile == "indoor_xt32_single":
        params_file = os.path.join(
            package_share,
            "config",
            "indoor_xt32_single_params.yaml",
        )
    elif not params_file and profile == "indoor_dual_jt128":
        params_file = os.path.join(
            package_share,
            "config",
            "indoor_dual_jt128_params.yaml",
        )
    elif not params_file:
        params_file = os.path.join(
            package_share,
            "config",
            "scan_to_map_icp_params.yaml",
        )

    primary_cloud_topic = LaunchConfiguration("hesai_center_timed_topic")
    secondary_cloud_topic = LaunchConfiguration("hesai_front_timed_topic")
    tertiary_cloud_topic = LaunchConfiguration("hesai_rear_timed_topic")
    use_secondary_cloud = LaunchConfiguration(
        "learning_slam_use_secondary_cloud"
    )
    use_tertiary_cloud = LaunchConfiguration("learning_slam_use_tertiary_cloud")
    imu_topic = LaunchConfiguration("imu_topic")
    use_wheel_prediction = TextSubstitution(text="")
    inter_lidar_motion_source = TextSubstitution(text="")

    if profile == "indoor_xt32_single":
        use_secondary_cloud = TextSubstitution(text="false")
        use_tertiary_cloud = TextSubstitution(text="false")
        imu_topic = TextSubstitution(text="/imu/data")
        use_wheel_prediction = TextSubstitution(text="false")
        inter_lidar_motion_source = TextSubstitution(text="imu_rotation")
    elif profile == "indoor_dual_jt128":
        primary_cloud_topic = LaunchConfiguration("hesai_front_timed_topic")
        secondary_cloud_topic = LaunchConfiguration("hesai_rear_timed_topic")
        tertiary_cloud_topic = LaunchConfiguration("hesai_center_timed_topic")
        use_secondary_cloud = TextSubstitution(text="true")
        use_tertiary_cloud = TextSubstitution(text="false")
        imu_topic = TextSubstitution(text="/imu/data")
        use_wheel_prediction = TextSubstitution(text="false")
        inter_lidar_motion_source = TextSubstitution(text="imu_rotation")

    wheel_odom_topic = LaunchConfiguration("wheel_odom_topic")
    if profile.startswith("indoor_"):
        wheel_odom_topic = TextSubstitution(text="/unused/wheel_odom_disabled")

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(launch_path),
            launch_arguments={
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "params_file": params_file,
                "primary_cloud_topic": primary_cloud_topic,
                "secondary_cloud_topic": secondary_cloud_topic,
                "tertiary_cloud_topic": tertiary_cloud_topic,
                "use_secondary_cloud": use_secondary_cloud,
                "use_tertiary_cloud": use_tertiary_cloud,
                "registration_method": LaunchConfiguration(
                    "learning_slam_registration_method"
                ),
                "cloud_subscription_reliable": LaunchConfiguration(
                    "learning_slam_cloud_qos_reliable"
                ),
                "cloud_subscription_depth": LaunchConfiguration(
                    "learning_slam_cloud_qos_depth"
                ),
                "wheel_odom_topic": wheel_odom_topic,
                "use_wheel_prediction": use_wheel_prediction,
                "inter_lidar_motion_compensation_source": inter_lidar_motion_source,
                "imu_topic": imu_topic,
                "map_frame": LaunchConfiguration("map_frame"),
                "odom_frame": LaunchConfiguration("odom_frame"),
                "base_frame": LaunchConfiguration("base_frame"),
                "publish_tf": LaunchConfiguration("learning_slam_publish_tf"),
            }.items(),
        )
    ]


def _static_tf_node(name, prefix):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=name,
        output="screen",
        arguments=[
            "--x",
            LaunchConfiguration(f"{prefix}_x"),
            "--y",
            LaunchConfiguration(f"{prefix}_y"),
            "--z",
            LaunchConfiguration(f"{prefix}_z"),
            "--qx",
            LaunchConfiguration(f"{prefix}_qx"),
            "--qy",
            LaunchConfiguration(f"{prefix}_qy"),
            "--qz",
            LaunchConfiguration(f"{prefix}_qz"),
            "--qw",
            LaunchConfiguration(f"{prefix}_qw"),
            "--frame-id",
            LaunchConfiguration("base_frame"),
            "--child-frame-id",
            LaunchConfiguration(f"{prefix}_frame"),
        ],
        condition=IfCondition(LaunchConfiguration("run_static_tf")),
    )


def _wheel_imu_ekf_parameter_overrides():
    return {
        "use_sim_time": ParameterValue(
            LaunchConfiguration("use_sim_time"),
            value_type=bool,
        ),
        "odom0": LaunchConfiguration("wheel_odom_topic"),
        "imu0": LaunchConfiguration("imu_topic"),
        "map_frame": LaunchConfiguration("map_frame"),
        "odom_frame": LaunchConfiguration("odom_frame"),
        "base_link_frame": LaunchConfiguration("base_frame"),
        "world_frame": LaunchConfiguration("odom_frame"),
    }


def _include_wheel_imu_ekf(context):
    if not _bool_arg(context, "run_wheel_imu_ekf"):
        return []
    if _is_indoor_profile(context):
        return [
            LogInfo(
                msg=(
                    "[slam_3d_bringup] run_wheel_imu_ekf was ignored because "
                    "indoor sensor profiles enforce the no-wheel estimator contract."
                )
            )
        ]

    return [
        Node(
            package="robot_localization",
            executable="ekf_node",
            name="slam_3d_wheel_imu_ekf_node",
            output="screen",
            parameters=[
                LaunchConfiguration("ekf_params_file"),
                _wheel_imu_ekf_parameter_overrides(),
            ],
            remappings=[
                ("odometry/filtered", "/slam_3d/reference/wheel_imu_odom"),
                ("accel/filtered", "/slam_3d/reference/wheel_imu_accel"),
            ],
        )
    ]


def _log_pipeline_contract(context):
    mode = LaunchConfiguration("pipeline_mode").perform(context)
    config_file = _resolve_pipeline_config(context)
    wheel_contract = (
        "wheel input disabled by the indoor profile"
        if _is_indoor_profile(context)
        else "wheel input controlled by the selected core parameters"
    )
    return [
        LogInfo(
            msg=(
                "[slam_3d_bringup] Hesai JT128/JT128/XT32 pipeline mode: "
                f"{mode}. Contract: {config_file}. Driver, cloud ingest, IMU "
                f"normalizer, core, EKF, static TF, and RViz remain "
                f"independently gated; {wheel_contract}."
            )
        )
    ]


def _declare_transform(prefix, frame, translation, quaternion):
    x, y, z = translation
    qx, qy, qz, qw = quaternion
    return [
        DeclareLaunchArgument(
            f"{prefix}_frame",
            default_value=frame,
            description=f"{prefix} Hesai frame",
        ),
        DeclareLaunchArgument(f"{prefix}_x", default_value=str(x)),
        DeclareLaunchArgument(f"{prefix}_y", default_value=str(y)),
        DeclareLaunchArgument(f"{prefix}_z", default_value=str(z)),
        DeclareLaunchArgument(f"{prefix}_qx", default_value=str(qx)),
        DeclareLaunchArgument(f"{prefix}_qy", default_value=str(qy)),
        DeclareLaunchArgument(f"{prefix}_qz", default_value=str(qz)),
        DeclareLaunchArgument(f"{prefix}_qw", default_value=str(qw)),
    ]


def generate_launch_description():
    package_share = get_package_share_directory("slam_3d_bringup")
    default_ekf_params = os.path.join(
        package_share,
        "config",
        "wheel_imu_ekf_3d_params.yaml",
    )
    default_rviz_config = os.path.join(
        package_share,
        "rviz",
        "hesai_jt128_xt32_3d_slam.rviz",
    )

    declarations = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument(
            "pipeline_mode",
            default_value="architecture_only",
            description=(
                "architecture_only, bag_replay, frontend_dev, mapping_dev, "
                "or localization_dev"
            ),
        ),
        DeclareLaunchArgument(
            "pipeline_config_file",
            default_value="auto",
            description=(
                "Pipeline contract to report. 'auto' selects indoor_pipeline.yaml "
                "for indoor sensor profiles and the outdoor contract otherwise."
            ),
        ),
        DeclareLaunchArgument("run_hesai_driver", default_value="false"),
        DeclareLaunchArgument(
            "hesai_driver_package",
            default_value="hesai_ros_driver",
        ),
        DeclareLaunchArgument(
            "hesai_driver_config_file",
            default_value="",
            description=(
                "Explicit three-device Hesai driver YAML. Required when "
                "run_hesai_driver is true."
            ),
        ),
        DeclareLaunchArgument("run_lidar_ingest", default_value="false"),
        DeclareLaunchArgument("lidar_ingest_params_file", default_value=""),
        DeclareLaunchArgument(
            "run_lidar_imu_normalizer",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "lidar_imu_normalizer_params_file",
            default_value="",
        ),
        DeclareLaunchArgument(
            "hesai_lidar_imu_raw_topic",
            default_value="/hesai/front/lidar_imu",
        ),
        DeclareLaunchArgument(
            "hesai_lidar_imu_timed_topic",
            default_value="/slam_3d/input/hesai_front/imu",
        ),
        DeclareLaunchArgument(
            "hesai_lidar_imu_frame",
            default_value="hesai_front",
        ),
        DeclareLaunchArgument(
            "lidar_imu_health_topic",
            default_value="/slam_3d/ingest/imu_health",
        ),
        DeclareLaunchArgument(
            "hesai_front_raw_topic",
            default_value="/hesai/front/lidar_points",
        ),
        DeclareLaunchArgument(
            "hesai_rear_raw_topic",
            default_value="/hesai/rear/lidar_points",
        ),
        DeclareLaunchArgument(
            "hesai_center_raw_topic",
            default_value="/hesai/center/lidar_points",
        ),
        DeclareLaunchArgument(
            "hesai_front_timed_topic",
            default_value="/slam_3d/input/hesai_front/points_timed",
        ),
        DeclareLaunchArgument(
            "hesai_rear_timed_topic",
            default_value="/slam_3d/input/hesai_rear/points_timed",
        ),
        DeclareLaunchArgument(
            "hesai_center_timed_topic",
            default_value="/slam_3d/input/hesai_center/points_timed",
        ),
        DeclareLaunchArgument(
            "lidar_ingest_health_topic",
            default_value="/slam_3d/ingest/health",
        ),
        DeclareLaunchArgument("run_learning_slam_core", default_value="false"),
        DeclareLaunchArgument(
            "learning_slam_sensor_profile",
            default_value="custom",
            description=(
                "custom, indoor_xt32_single, or indoor_dual_jt128. Indoor "
                "profiles select the matching core YAML, cloud topology, and "
                "/imu/data without wheel prediction."
            ),
        ),
        DeclareLaunchArgument("learning_slam_params_file", default_value=""),
        DeclareLaunchArgument(
            "learning_slam_use_secondary_cloud",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "learning_slam_use_tertiary_cloud",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "learning_slam_registration_method",
            default_value="",
            description="Optional override; empty preserves the selected params file",
        ),
        DeclareLaunchArgument(
            "learning_slam_cloud_qos_reliable",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "learning_slam_cloud_qos_depth",
            default_value="5",
        ),
        DeclareLaunchArgument(
            "learning_slam_publish_tf",
            default_value="false",
        ),
        DeclareLaunchArgument("run_wheel_imu_ekf", default_value="false"),
        DeclareLaunchArgument(
            "wheel_odom_topic",
            default_value="/motor_driver_node/odom",
        ),
        DeclareLaunchArgument(
            "imu_topic",
            default_value="/slam_3d/input/hesai_front/imu",
            description=(
                "Normalized front JT128 built-in IMU used by deskew and the "
                "optional reference EKF"
            ),
        ),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("odom_frame", default_value="odom"),
        DeclareLaunchArgument(
            "ekf_params_file",
            default_value=default_ekf_params,
        ),
        DeclareLaunchArgument("run_static_tf", default_value="false"),
        DeclareLaunchArgument("base_frame", default_value="base_link"),
        DeclareLaunchArgument("run_rviz", default_value="false"),
        DeclareLaunchArgument(
            "rviz_config_file",
            default_value=default_rviz_config,
        ),
    ]

    # Captured from 02_samsung_geoje /tf_static. Calibration status is unverified.
    declarations += _declare_transform(
        "front",
        "hesai_front",
        (0.55357, 0.36857, 1.2991),
        (0.9255365, -0.3786584, 0.0000034, 0.00000139),
    )
    declarations += _declare_transform(
        "rear",
        "hesai_rear",
        (-0.55357, -0.36857, 1.2991),
        (0.3866995, 0.9222058, 0.00000142, -0.00000339),
    )
    declarations += _declare_transform(
        "center",
        "hesai_center",
        (0.36, 0.0, 1.5496),
        (0.0, 0.0, 0.707108, 0.707105),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="slam_3d_rviz",
        output="screen",
        arguments=[
            "-d",
            LaunchConfiguration("rviz_config_file"),
            "-f",
            LaunchConfiguration("map_frame"),
        ],
        remappings=[
            (
                "/slam_3d/input/hesai_front/points_timed",
                LaunchConfiguration("hesai_front_timed_topic"),
            ),
            (
                "/slam_3d/input/hesai_rear/points_timed",
                LaunchConfiguration("hesai_rear_timed_topic"),
            ),
            (
                "/slam_3d/input/hesai_center/points_timed",
                LaunchConfiguration("hesai_center_timed_topic"),
            ),
        ],
        condition=IfCondition(LaunchConfiguration("run_rviz")),
    )

    return LaunchDescription(
        declarations
        + [
            OpaqueFunction(function=_log_pipeline_contract),
            _static_tf_node("base_to_hesai_front_tf", "front"),
            _static_tf_node("base_to_hesai_rear_tf", "rear"),
            _static_tf_node("base_to_hesai_center_tf", "center"),
            OpaqueFunction(function=_include_wheel_imu_ekf),
            OpaqueFunction(function=_include_hesai_driver),
            OpaqueFunction(function=_include_lidar_ingest),
            OpaqueFunction(function=_include_lidar_imu_normalizer),
            OpaqueFunction(function=_include_learning_slam_core),
            rviz,
        ]
    )
