"""Launch contracts for the project SLAM executable; no ROS processes are started."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


ROOT = Path(__file__).resolve().parents[1]


def load_module(relative, monkeypatch):
    monkeypatch.setattr("ament_index_python.packages.get_package_share_directory", lambda _: str(ROOT))
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_"), ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "get_package_share_directory"):
        monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(ROOT))
    return module


def declared_context(module, overrides=None):
    context = LaunchContext()
    context.launch_configurations.update(overrides or {})
    description = module.generate_launch_description()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)
    return context, description


def effective_parameters(node, context):
    # launch_ros does not expose the unevaluated parameter list publicly.
    result = {}
    for item in evaluate_parameters(context, node._Node__parameters):
        if isinstance(item, Path):
            result.update(yaml.safe_load(item.read_text())["slam_toolbox"]["ros__parameters"])
        else:
            result.update(item)
    return result


def test_default_launch_runs_project_node_and_preserves_yaml(monkeypatch):
    module = load_module("launch/slam.launch.py", monkeypatch)
    context, _ = declared_context(module)
    nodes = module.launch_slam(context)
    assert len(nodes) == 1 and isinstance(nodes[0], Node)
    node = nodes[0]
    assert node.node_package == "slam_2d_core"
    assert node.node_executable == "roughness_aware_slam_toolbox_node"
    params = effective_parameters(node, context)
    assert params["scan_topic"] == "/scan_laserscan"
    assert params["base_frame"] == "base_link"
    assert params["use_sim_time"] is False
    assert params["roughness_imu_topic"] == "/imu/data"
    profile = yaml.safe_load((ROOT / "config/slam_roughness_innovation_params.yaml").read_text())
    assert params["roughness_enable_near_chain_gate"] == profile["slam_toolbox"]["ros__parameters"][
        "roughness_enable_near_chain_gate"]


def test_legacy_parameter_overrides_are_typed(monkeypatch):
    module = load_module("launch/slam.launch.py", monkeypatch)
    context, _ = declared_context(module, {
        "scan_topic": "/robot/scan", "base_frame": "base_footprint", "use_sim_time": "true",
        "roughness_innovation_imu_topic": "/robot/imu",
        "roughness_innovation_scan_qos_depth": "17",
        "roughness_innovation_scan_qos_reliable": "true",
        "roughness_innovation_translation_gate_m": "0.08",
        "resolution": "0.05", "throttle_scans": "2",
    })
    params = effective_parameters(module.launch_slam(context)[0], context)
    assert params["scan_topic"] == "/robot/scan"
    assert params["base_frame"] == "base_footprint"
    assert params["use_sim_time"] is True
    assert params["roughness_imu_topic"] == "/robot/imu"
    assert params["roughness_scan_qos_depth"] == 17
    assert params["roughness_scan_qos_reliable"] is True
    assert params["roughness_translation_innovation_gate_m"] == pytest.approx(0.08)
    assert params["resolution"] == pytest.approx(0.05)
    assert params["throttle_scans"] == 2


@pytest.mark.parametrize("overrides, message", [
    ({"use_roughness_innovation_slam_toolbox": "false"}, "must be true"),
    ({"use_roughness_anchor_slam_toolbox": "true"}, "anchor"),
    ({"autostart": "false"}, "self-configures"),
    ({"use_lifecycle_manager": "true"}, "lifecycle"),
    ({"roughness_innovation_scan_qos_depth": "0"}, "positive"),
    ({"resolution": "nan"}, "finite"),
    ({"roughness_innovation_scan_qos_reliable": "maybe"}, "boolean"),
    ({"roughness_innovation_enable_near_chain_gate": "false",
      "roughness_innovation_enable_spatial_memory_gate": "true"}, "requires"),
    ({"params_file": "/nonexistent/slam_parameters.yaml"}, "does not exist"),
])
def test_invalid_configuration_fails_before_process_creation(monkeypatch, overrides, message):
    module = load_module("launch/slam.launch.py", monkeypatch)
    context, _ = declared_context(module, overrides)
    with pytest.raises(RuntimeError, match=message):
        module.launch_slam(context)


@pytest.mark.parametrize("filename,target", [
    ("slam_bringup_launch.py", "slam.launch.py"),
    ("slam_from_bag_launch.py", "slam_from_bag.launch.py"),
])
def test_alias_delegates_to_package_launch(monkeypatch, filename, target):
    module = load_module("launch/" + filename, monkeypatch)
    description = module.generate_launch_description()
    assert len(description.entities) == 1
    include = description.entities[0]
    assert isinstance(include, IncludeLaunchDescription)
    include.launch_description_source.get_launch_description(LaunchContext())
    assert include.launch_description_source.location == str(
        ROOT / "launch" / target)


def test_bag_defaults_to_sim_time_without_playback(monkeypatch):
    module = load_module("launch/slam_from_bag.launch.py", monkeypatch)
    context, description = declared_context(module)
    assert context.launch_configurations["use_sim_time"] == "true"
    assert module.validate_replay(context) == []
    assert module.playback_actions(context) == []
    includes = [item for item in description.entities if isinstance(item, IncludeLaunchDescription)]
    assert len(includes) == 1
    includes[0].launch_description_source.get_launch_description(context)
    assert includes[0].launch_description_source.location == str(
        ROOT / "launch/slam.launch.py")


@pytest.mark.parametrize("overrides, message", [
    ({"use_static_map_odom": "true"}, "cannot run together"),
    ({"bag_play_rate": "0"}, "positive"),
    ({"bag_start_offset_s": "-1"}, "non-negative"),
    ({"start_pointcloud_projection": "true"}, "preprocessing"),
])
def test_bag_rejects_invalid_ownership_and_playback(monkeypatch, overrides, message):
    module = load_module("launch/slam_from_bag.launch.py", monkeypatch)
    context, _ = declared_context(module, overrides)
    with pytest.raises(RuntimeError, match=message):
        module.validate_replay(context)


def test_bag_player_waits_for_slam_and_filters_selected_topics(monkeypatch, tmp_path):
    module = load_module("launch/slam_from_bag.launch.py", monkeypatch)
    bag = tmp_path / "input_bag"
    bag.mkdir()
    context, _ = declared_context(module, {
        "bag_file": str(bag), "scan_topic": "/robot/scan",
        "roughness_innovation_imu_topic": "/robot/imu",
        "bag_play_rate": "0.5", "bag_start_offset_s": "2.0",
    })
    module.validate_replay(context)
    actions = module.playback_actions(context)
    assert len(actions) == 1 and isinstance(actions[0], ExecuteProcess)
    command = [perform_substitutions(context, arg) for arg in actions[0].cmd]
    assert command[:2] == ["python3", str(ROOT / "scripts/play_after_slam_ready.py")]
    assert command[command.index("--rate") + 1] == "0.5"
    assert command[command.index("--start-offset") + 1] == "2.0"
    assert "/robot/scan" in command and "/robot/imu" in command
    assert "/cmd_vel" not in command


def test_ekf_requires_explicit_config(monkeypatch):
    module = load_module("launch/robot_localization_launch.py", monkeypatch)
    arguments = [item for item in module.generate_launch_description().entities
                 if isinstance(item, DeclareLaunchArgument)]
    config = next(item for item in arguments if item.name == "config_file")
    assert config.default_value is None


def readiness_fixture(monkeypatch, ready):
    module = load_module("scripts/play_after_slam_ready.py", monkeypatch)
    future = Mock()
    future.done.return_value = ready
    future.result.return_value = SimpleNamespace(values=[])
    client = Mock()
    client.service_is_ready.return_value = True
    client.call_async.return_value = future
    node = Mock()
    node.create_client.return_value = client
    node.get_subscriptions_info_by_topic.return_value = [
        SimpleNamespace(node_name="slam_toolbox", node_namespace="/",
                        topic_type="sensor_msgs/msg/LaserScan")]
    monkeypatch.setattr(module.rclpy, "create_node", lambda _: node)
    monkeypatch.setattr(module.rclpy, "ok", lambda: True)
    monkeypatch.setattr(module.rclpy, "spin_once", lambda *args, **kwargs: None)
    times = iter(index * 0.1 for index in range(1000))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times))
    return module, node


def test_readiness_requires_executor_response_and_scan_subscription(monkeypatch):
    module, node = readiness_fixture(monkeypatch, True)
    module.wait_for_slam("/robot/scan", 1.0)
    node.get_subscriptions_info_by_topic.assert_called_with("/robot/scan")
    node.destroy_node.assert_called_once()


def test_readiness_timeout_is_bounded_and_releases_node(monkeypatch):
    module, node = readiness_fixture(monkeypatch, False)
    with pytest.raises(RuntimeError, match="within 1s"):
        module.wait_for_slam("/robot/scan", 1.0)
    node.destroy_node.assert_called_once()
