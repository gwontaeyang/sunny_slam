import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument


def _load_launch_module():
    package_share = Path(get_package_share_directory("slam_3d_bringup"))
    launch_path = package_share / "launch" / "dual_mid360_3d_slam.launch.py"
    spec = importlib.util.spec_from_file_location(
        "dual_mid360_3d_slam_launch",
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert _default_value(declarations["learning_slam_publish_tf"]) == "false"
    assert _default_value(declarations["run_rviz"]) == "false"
    assert "large_scale_outdoor_pipeline.yaml" in _default_value(
        declarations["pipeline_config_file"]
    )
    assert "slam_backend" not in declarations
    assert "fast_lio_package" not in declarations
    assert "glim_package" not in declarations
    assert "kiss_icp_package" not in declarations
