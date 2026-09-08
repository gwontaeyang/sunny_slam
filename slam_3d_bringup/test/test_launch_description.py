import importlib.util
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import LogInfo


def _load_package_launch_module(package_name, launch_file, module_name):
    package_share = Path(get_package_share_directory(package_name))
    launch_path = package_share / "launch" / launch_file
    spec = importlib.util.spec_from_file_location(module_name, launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_launch_module():
    return _load_package_launch_module(
        "slam_3d_bringup",
        "hesai_jt128_xt32_3d_slam.launch.py",
        "hesai_jt128_xt32_3d_slam_launch",
    )


def _default_value(action):
    context = LaunchContext()
    return "".join(
        substitution.perform(context) for substitution in action.default_value
    )


def test_launch_description_defaults_do_not_start_hardware_or_tf():
    module = _load_launch_module()
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    declarations = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert _default_value(declarations["pipeline_mode"]) == "architecture_only"
    assert _default_value(declarations["pipeline_config_file"]) == "auto"
    assert _default_value(declarations["run_hesai_driver"]) == "false"
    assert _default_value(declarations["run_lidar_ingest"]) == "false"
    assert _default_value(declarations["run_lidar_imu_normalizer"]) == "false"
    assert _default_value(declarations["run_static_tf"]) == "false"
    assert _default_value(declarations["run_wheel_imu_ekf"]) == "false"
    assert _default_value(declarations["run_learning_slam_core"]) == "false"
    assert _default_value(declarations["learning_slam_sensor_profile"]) == "custom"
    assert _default_value(declarations["run_rviz"]) == "false"
    assert _default_value(declarations["hesai_driver_config_file"]) == ""
    assert _default_value(declarations["learning_slam_cloud_qos_reliable"]) == "true"
    assert _default_value(declarations["learning_slam_publish_tf"]) == "false"
    assert _default_value(declarations["learning_slam_registration_method"]) == ""
    assert _default_value(declarations["hesai_center_timed_topic"]).endswith(
        "/hesai_center/points_timed"
    )
    assert _default_value(declarations["hesai_front_timed_topic"]).endswith(
        "/hesai_front/points_timed"
    )
    assert _default_value(declarations["hesai_rear_timed_topic"]).endswith(
        "/hesai_rear/points_timed"
    )
    assert _default_value(declarations["front_frame"]) == "hesai_front"
    assert _default_value(declarations["rear_frame"]) == "hesai_rear"
    assert _default_value(declarations["center_frame"]) == "hesai_center"
    assert (
        _default_value(declarations["hesai_lidar_imu_raw_topic"])
        == "/hesai/front/lidar_imu"
    )
    assert _default_value(
        declarations["hesai_lidar_imu_timed_topic"]
    ).endswith("/hesai_front/imu")
    assert _default_value(declarations["wheel_odom_topic"]) == "/motor_driver_node/odom"
    assert (
        _default_value(declarations["imu_topic"])
        == "/slam_3d/input/hesai_front/imu"
    )
    assert _default_value(declarations["map_frame"]) == "map"
    assert _default_value(declarations["odom_frame"]) == "odom"
    assert _default_value(declarations["base_frame"]) == "base_link"


def test_lidar_ingest_disabled_path_needs_no_installed_adapter():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["run_lidar_ingest"] = "false"

    assert module._include_lidar_ingest(context) == []


def test_lidar_ingest_missing_executable_logs_without_launching(monkeypatch):
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["run_lidar_ingest"] = "true"
    monkeypatch.setattr(
        module,
        "get_package_prefix",
        lambda package_name: "/tmp/missing_lidar_ingest",
    )
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda package_name: "/tmp/missing_lidar_ingest/share",
    )

    actions = module._include_lidar_ingest(context)

    assert len(actions) == 1
    assert isinstance(actions[0], LogInfo)
    message = "".join(
        substitution.perform(context) for substitution in actions[0].msg
    )
    assert "hesai_pointcloud_ingest_node" in message


def test_lidar_ingest_forwards_topics_and_expected_frames(monkeypatch):
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_lidar_ingest": "true",
            "lidar_ingest_params_file": "",
            "use_sim_time": "true",
            "hesai_front_raw_topic": "/test/front/raw",
            "hesai_rear_raw_topic": "/test/rear/raw",
            "hesai_center_raw_topic": "/test/center/raw",
            "hesai_front_timed_topic": "/test/front/timed",
            "hesai_rear_timed_topic": "/test/rear/timed",
            "hesai_center_timed_topic": "/test/center/timed",
            "lidar_ingest_health_topic": "/test/ingest/health",
            "front_frame": "test_front",
            "rear_frame": "test_rear",
            "center_frame": "test_center",
        }
    )
    monkeypatch.setattr(module, "get_package_prefix", lambda _: "/tmp/installed")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _: "/tmp/installed/share",
    )
    monkeypatch.setattr(module.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)

    actions = module._include_lidar_ingest(context)

    assert len(actions) == 1
    overrides = actions[0]["parameters"][1]
    expected = {
        "front.input_topic": "/test/front/raw",
        "rear.input_topic": "/test/rear/raw",
        "center.input_topic": "/test/center/raw",
        "front.output_topic": "/test/front/timed",
        "rear.output_topic": "/test/rear/timed",
        "center.output_topic": "/test/center/timed",
        "front.expected_input_frame_id": "test_front",
        "rear.expected_input_frame_id": "test_rear",
        "center.expected_input_frame_id": "test_center",
        "health_topic": "/test/ingest/health",
    }
    for key, value in expected.items():
        assert overrides[key].perform(context) == value


def test_lidar_imu_normalizer_forwards_clocked_topics(monkeypatch):
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_lidar_imu_normalizer": "true",
            "lidar_imu_normalizer_params_file": "",
            "use_sim_time": "true",
            "hesai_lidar_imu_raw_topic": "/test/lidar_imu/raw",
            "hesai_lidar_imu_timed_topic": "/test/lidar_imu/timed",
            "hesai_lidar_imu_frame": "test_lidar_imu",
            "lidar_imu_health_topic": "/test/lidar_imu/health",
        }
    )
    monkeypatch.setattr(module, "get_package_prefix", lambda _: "/tmp/installed")
    monkeypatch.setattr(
        module,
        "get_package_share_directory",
        lambda _: "/tmp/installed/share",
    )
    monkeypatch.setattr(module.os.path, "isfile", lambda _: True)
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)

    actions = module._include_lidar_imu_normalizer(context)

    assert len(actions) == 1
    assert actions[0]["executable"] == "hesai_imu_normalizer_node"
    overrides = actions[0]["parameters"][1]
    expected = {
        "input_topic": "/test/lidar_imu/raw",
        "output_topic": "/test/lidar_imu/timed",
        "expected_input_frame_id": "test_lidar_imu",
        "output_frame_id": "test_lidar_imu",
        "health_topic": "/test/lidar_imu/health",
    }
    for key, value in expected.items():
        assert overrides[key].perform(context) == value


def test_hesai_driver_requires_explicit_existing_config():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations["run_hesai_driver"] = "true"
    context.launch_configurations["hesai_driver_config_file"] = ""

    actions = module._include_hesai_driver(context)

    assert len(actions) == 1
    assert isinstance(actions[0], LogInfo)
    message = "".join(
        substitution.perform(context) for substitution in actions[0].msg
    )
    assert "requires an existing" in message


def test_learning_core_forwards_all_three_clouds_and_qos():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_learning_slam_core": "true",
            "learning_slam_params_file": "",
            "hesai_center_timed_topic": "/test/center",
            "hesai_front_timed_topic": "/test/front",
            "hesai_rear_timed_topic": "/test/rear",
            "learning_slam_use_secondary_cloud": "true",
            "learning_slam_use_tertiary_cloud": "true",
            "learning_slam_registration_method": "gicp",
            "learning_slam_cloud_qos_reliable": "true",
            "learning_slam_cloud_qos_depth": "17",
            "learning_slam_publish_tf": "false",
            "wheel_odom_topic": "/test/wheel",
            "imu_topic": "/test/imu",
            "map_frame": "test_map",
            "odom_frame": "test_odom",
            "base_frame": "test_base",
            "use_sim_time": "false",
        }
    )

    actions = module._include_learning_slam_core(context)

    assert len(actions) == 1
    assert isinstance(actions[0], IncludeLaunchDescription)
    forwarded = dict(actions[0].launch_arguments)
    assert forwarded["primary_cloud_topic"].perform(context) == "/test/center"
    assert forwarded["secondary_cloud_topic"].perform(context) == "/test/front"
    assert forwarded["tertiary_cloud_topic"].perform(context) == "/test/rear"
    assert forwarded["use_tertiary_cloud"].perform(context) == "true"
    assert forwarded["cloud_subscription_reliable"].perform(context) == "true"
    assert forwarded["cloud_subscription_depth"].perform(context) == "17"
    assert forwarded["wheel_odom_topic"].perform(context) == "/test/wheel"
    assert forwarded["imu_topic"].perform(context) == "/test/imu"
    assert forwarded["map_frame"].perform(context) == "test_map"
    assert forwarded["odom_frame"].perform(context) == "test_odom"
    assert forwarded["base_frame"].perform(context) == "test_base"

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
    assert set(forwarded).issubset(core_declarations)


def test_learning_core_indoor_profiles_select_cloud_topology_and_main_imu():
    module = _load_launch_module()
    common = {
        "run_learning_slam_core": "true",
        "learning_slam_params_file": "",
        "hesai_center_timed_topic": "/test/center",
        "hesai_front_timed_topic": "/test/front",
        "hesai_rear_timed_topic": "/test/rear",
        "learning_slam_use_secondary_cloud": "true",
        "learning_slam_use_tertiary_cloud": "true",
        "learning_slam_registration_method": "",
        "learning_slam_cloud_qos_reliable": "true",
        "learning_slam_cloud_qos_depth": "10",
        "learning_slam_publish_tf": "false",
        "wheel_odom_topic": "/test/wheel",
        "imu_topic": "/test/legacy_imu",
        "map_frame": "map",
        "odom_frame": "odom",
        "base_frame": "base_link",
        "use_sim_time": "false",
    }

    expected_profiles = {
        "indoor_xt32_single": {
            "params": "indoor_xt32_single_params.yaml",
            "primary": "/test/center",
            "secondary": "/test/front",
            "tertiary": "/test/rear",
            "use_secondary": "false",
            "use_tertiary": "false",
        },
        "indoor_dual_jt128": {
            "params": "indoor_dual_jt128_params.yaml",
            "primary": "/test/front",
            "secondary": "/test/rear",
            "tertiary": "/test/center",
            "use_secondary": "true",
            "use_tertiary": "false",
        },
    }

    for profile, expected in expected_profiles.items():
        context = LaunchContext()
        context.launch_configurations.update(common)
        context.launch_configurations["learning_slam_sensor_profile"] = profile

        actions = module._include_learning_slam_core(context)

        assert len(actions) == 1
        forwarded = dict(actions[0].launch_arguments)
        assert forwarded["params_file"].endswith(expected["params"])
        assert forwarded["primary_cloud_topic"].perform(context) == expected["primary"]
        assert (
            forwarded["secondary_cloud_topic"].perform(context)
            == expected["secondary"]
        )
        assert (
            forwarded["tertiary_cloud_topic"].perform(context)
            == expected["tertiary"]
        )
        assert (
            forwarded["use_secondary_cloud"].perform(context)
            == expected["use_secondary"]
        )
        assert (
            forwarded["use_tertiary_cloud"].perform(context)
            == expected["use_tertiary"]
        )
        assert forwarded["imu_topic"].perform(context) == "/imu/data"
        assert forwarded["use_wheel_prediction"].perform(context) == "false"
        assert (
            forwarded["inter_lidar_motion_compensation_source"].perform(context)
            == "imu_rotation"
        )
        assert (
            forwarded["wheel_odom_topic"].perform(context)
            == "/unused/wheel_odom_disabled"
        )


def test_pipeline_contract_auto_selects_indoor_without_changing_custom_default():
    module = _load_launch_module()

    indoor_context = LaunchContext()
    indoor_context.launch_configurations.update(
        {
            "pipeline_config_file": "auto",
            "learning_slam_sensor_profile": "indoor_xt32_single",
        }
    )
    assert module._resolve_pipeline_config(indoor_context).endswith(
        "/config/indoor_pipeline.yaml"
    )

    custom_context = LaunchContext()
    custom_context.launch_configurations.update(
        {
            "pipeline_config_file": "auto",
            "learning_slam_sensor_profile": "custom",
        }
    )
    assert module._resolve_pipeline_config(custom_context).endswith(
        "/config/large_scale_outdoor_pipeline.yaml"
    )

    explicit_context = LaunchContext()
    explicit_context.launch_configurations.update(
        {
            "pipeline_config_file": "/tmp/custom_pipeline.yaml",
            "learning_slam_sensor_profile": "indoor_xt32_single",
        }
    )
    assert (
        module._resolve_pipeline_config(explicit_context)
        == "/tmp/custom_pipeline.yaml"
    )


def test_wheel_imu_ekf_forwards_topics_and_frames():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_sim_time": "true",
            "wheel_odom_topic": "/test/wheel",
            "imu_topic": "/test/imu",
            "map_frame": "test_map",
            "odom_frame": "test_odom",
            "base_frame": "test_base",
        }
    )

    overrides = module._wheel_imu_ekf_parameter_overrides()

    assert overrides["odom0"].perform(context) == "/test/wheel"
    assert overrides["imu0"].perform(context) == "/test/imu"
    assert overrides["map_frame"].perform(context) == "test_map"
    assert overrides["odom_frame"].perform(context) == "test_odom"
    assert overrides["base_link_frame"].perform(context) == "test_base"
    assert overrides["world_frame"].perform(context) == "test_odom"


def test_indoor_profile_rejects_optional_wheel_imu_ekf():
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_wheel_imu_ekf": "true",
            "learning_slam_sensor_profile": "indoor_xt32_single",
        }
    )

    actions = module._include_wheel_imu_ekf(context)

    assert len(actions) == 1
    assert isinstance(actions[0], LogInfo)
    message = "".join(
        substitution.perform(context) for substitution in actions[0].msg
    )
    assert "no-wheel estimator contract" in message


def test_custom_profile_keeps_optional_wheel_imu_ekf(monkeypatch):
    module = _load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "run_wheel_imu_ekf": "true",
            "learning_slam_sensor_profile": "custom",
        }
    )
    monkeypatch.setattr(module, "Node", lambda **kwargs: kwargs)

    actions = module._include_wheel_imu_ekf(context)

    assert len(actions) == 1
    assert actions[0]["package"] == "robot_localization"
    assert actions[0]["executable"] == "ekf_node"
