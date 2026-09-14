#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import GroupAction
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression


def create_package_launch(package_name, launch_file, launch_args=None, condition=None):
    kwargs = {
        "launch_description_source": PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(package_name), "launch", launch_file)
        ),
        "launch_arguments": (launch_args or {}).items(),
    }
    if condition is not None:
        kwargs["condition"] = condition
    return GroupAction(actions=[IncludeLaunchDescription(**kwargs)], scoped=True)


def nd_condition():
    robot_model = LaunchConfiguration("robot_model")
    return IfCondition(PythonExpression(["'", robot_model, "'.lower() == 'nd'"]))


def non_nd_slam_type_condition(expected_type):
    slam_type = LaunchConfiguration("slam_type")
    robot_model = LaunchConfiguration("robot_model")
    return IfCondition(
        PythonExpression(
            [
                "'",
                slam_type,
                "'.lower() == '",
                expected_type,
                "' and '",
                robot_model,
                "'.lower() != 'nd'",
            ]
        )
    )


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    robot_model = LaunchConfiguration("robot_model")
    system_type = LaunchConfiguration("type")
    mode = LaunchConfiguration("mode")

    common_args = {
        "use_sim_time": use_sim_time,
        "robot_model": robot_model,
        "type": system_type,
        "mode": mode,
    }

    declarations = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Use simulation clock when true.",
        ),
        DeclareLaunchArgument(
            "robot_model",
            default_value="rover",
            description="Robot model name.",
        ),
        DeclareLaunchArgument(
            "type",
            default_value="apart",
            description="Autonomy system type.",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="SLAM",
            description="Autonomy mode.",
        ),
        DeclareLaunchArgument(
            "slam_type",
            default_value="2d",
            description="SLAM stack to launch for non-ND robots: 2d or 3d.",
        ),
        DeclareLaunchArgument(
            "liosam_params_file",
            default_value=os.path.join(
                get_package_share_directory("lio_sam"),
                "config",
                "params.yaml",
            ),
            description="LIO-SAM parameter file used by the ND SLAM path.",
        ),
    ]

    return LaunchDescription(
        declarations
        + [
            create_package_launch(
                "slam_2d_bringup",
                "slam.launch.py",
                common_args,
                condition=non_nd_slam_type_condition("2d"),
            ),
            create_package_launch(
                "slam_3d_bringup",
                "dual_mid360_3d_slam.launch.py",
                common_args,
                condition=non_nd_slam_type_condition("3d"),
            ),
            create_package_launch(
                "lio_sam",
                "run.launch.py",
                {
                    "params_file": LaunchConfiguration("liosam_params_file"),
                    "use_sim_time": use_sim_time,
                    "use_rviz": "false",
                    "publish_static_map_to_odom": "false",
                },
                condition=nd_condition(),
            ),
        ]
    )
