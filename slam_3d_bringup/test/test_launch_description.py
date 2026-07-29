import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription


def _load_package_launch_module(package_name, launch_file, module_name):
    package_share = Path(get_package_share_directory(package_name))
    launch_path = package_share / "launch" / launch_file
    spec = importlib.util.spec_from_file_location(
        module_name,
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_launch_module():
    return _load_package_launch_module(
        "slam_3d_bringup",
        "dual_mid360_3d_slam.launch.py",
        "dual_mid360_3d_slam_launch",
    )


def _default_value(action):
    context = LaunchContext()
    return "".join(substitution.perform(context) for substitution in action.default_value)


def test_launch_description_defaults_are_safe_for_scaffold():
    module = _load_launch_module()

    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)

    declarations = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert _default_value(declarations["pipeline_mode"]) == "architecture_only"
    assert _default_value(declarations["run_livox_driver"]) == "false"
    assert _default_value(declarations["run_static_tf"]) == "false"
    assert _default_value(declarations["run_wheel_imu_ekf"]) == "false"
    assert _default_value(declarations["run_learning_slam_core"]) == "false"
    assert _default_value(declarations["learning_slam_cloud_qos_reliable"]) == "false"
    assert _default_value(declarations["learning_slam_cloud_qos_depth"]) == "5"
    assert _default_value(declarations["learning_slam_publish_tf"]) == "false"
    assert _default_value(declarations["run_rviz"]) == "false"
    assert "large_scale_outdoor_pipeline.yaml" in _default_value(
        declarations["pipeline_config_file"]
    )
    assert "slam_backend" not in declarations
    assert "fast_lio_package" not in declarations
    assert "glim_package" not in declarations
    assert "kiss_icp_package" not in declarations


def test_learning_core_cloud_qos_arguments_are_forwarded():
    module = _load_launch_module()

    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_learning_slam_core": "true",
            "learning_slam_params_file": "",
            "learning_slam_cloud_qos_reliable": "true",
            "learning_slam_cloud_qos_depth": "17",
        }
    )

    actions = module._include_learning_slam_core(context)

    assert len(actions) == 1
    assert isinstance(actions[0], IncludeLaunchDescription)
    forwarded_arguments = dict(actions[0].launch_arguments)
    assert (
        forwarded_arguments["cloud_subscription_reliable"].perform(context)
        == context.launch_configurations["learning_slam_cloud_qos_reliable"]
    )
    assert (
        forwarded_arguments["cloud_subscription_depth"].perform(context)
        == context.launch_configurations["learning_slam_cloud_qos_depth"]
    )

    core_module = _load_package_launch_module(
        "slam_3d_core",
        "scan_to_map_icp.launch.py",
        "scan_to_map_icp_launch",
    )
    core_declarations = {
        entity.name
        for entity in core_module.generate_launch_description().entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(forwarded_arguments).issubset(core_declarations)
