import importlib.util
from pathlib import Path

import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch_ros.utilities import evaluate_parameters


def load_launch_module():
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    launch_path = package_share / "launch" / "slam.launch.py"
    spec = importlib.util.spec_from_file_location(
        "slam_2d_bringup_slam_launch",
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bag_launch_module():
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    launch_path = package_share / "launch" / "slam_from_bag_launch.py"
    spec = importlib.util.spec_from_file_location(
        "slam_2d_bringup_slam_from_bag_launch",
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bringup_launch_module():
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    launch_path = package_share / "launch" / "slam_bringup_launch.py"
    spec = importlib.util.spec_from_file_location(
        "slam_2d_bringup_slam_bringup_launch",
        launch_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_value(action):
    context = LaunchContext()
    return "".join(
        substitution.perform(context) for substitution in action.default_value
    )


def context_with_declared_defaults(launch_description, overrides=None):
    context = LaunchContext()
    for entity in launch_description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            context.launch_configurations[entity.name] = default_value(entity)
    context.launch_configurations.update(overrides or {})
    return context


def node_parameters(launch_description, context, node_name):
    node = next(
        entity for entity in launch_description.entities
        if isinstance(entity, Node)
        and getattr(entity, "_Node__node_name", None) == node_name
    )
    evaluated = evaluate_parameters(
        context,
        getattr(node, "_Node__parameters"),
    )
    assert len(evaluated) == 1
    return evaluated[0]


def test_roughness_innovation_is_wired_but_disabled_by_default():
    module = load_launch_module()
    launch_description = module.generate_launch_description()

    assert isinstance(launch_description, LaunchDescription)
    declarations = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    nodes = [
        entity for entity in launch_description.entities
        if isinstance(entity, Node)
    ]

    assert default_value(declarations["scan_topic"]) == "/scan_laserscan"
    assert (
        default_value(
            declarations["use_roughness_innovation_slam_toolbox"]
        )
        == "false"
    )
    assert (
        default_value(
            declarations["roughness_innovation_scan_qos_reliable"]
        )
        == "false"
    )
    assert (
        default_value(declarations["roughness_innovation_scan_qos_depth"])
        == "5"
    )
    assert (
        Path(
            default_value(
                declarations["roughness_innovation_params_file"]
            )
        ).name
        == "slam_roughness_innovation_params.yaml"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_near_chain_loop_closing_cooldown_s"
            ]
        )
        == "0.0"
    )
    assert (
        default_value(
            declarations["roughness_innovation_enable_near_chain_gate"]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_enable_spatial_memory_gate"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_spatial_memory_use_pose_graph_anchors"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_enable_spatial_passage_hysteresis"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations["roughness_innovation_spatial_passage_radius_m"]
        )
        == "0.55"
    )
    assert (
        default_value(
            declarations["roughness_innovation_spatial_passage_max_age_s"]
        )
        == "3.5"
    )
    assert len(nodes) == 2
    assert {
        getattr(node, "_Node__node_executable", None) for node in nodes
    } == {
        "sync_slam_toolbox_node",
        "roughness_aware_slam_toolbox_node",
    }


@pytest.mark.parametrize(
    ("autostart", "use_lifecycle_manager", "message"),
    [
        ("false", "false", "autostart must be true"),
        ("true", "true", "cannot use an external lifecycle manager"),
    ],
)
def test_live_roughness_innovation_frontend_rejects_lifecycle_mismatch(
        autostart, use_lifecycle_manager, message):
    module = load_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "use_lifecycle_manager": use_lifecycle_manager,
            "autostart": autostart,
        }
    )

    with pytest.raises(RuntimeError, match=message):
        module.validate_roughness_innovation_frontend(context)


def test_live_roughness_innovation_parameters_are_layered_on_mapper_config():
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(launch_description)
    node = next(
        entity for entity in launch_description.entities
        if isinstance(entity, Node)
        and getattr(entity, "_Node__node_executable", None)
        == "roughness_aware_slam_toolbox_node"
    )
    evaluated = evaluate_parameters(
        context,
        getattr(node, "_Node__parameters"),
    )

    assert len(evaluated) == 3
    assert evaluated[2]["enable_roughness_innovation_gate"] is True
    assert evaluated[2]["roughness_imu_topic"] == "/imu/data"
    assert evaluated[2]["scan_topic"] == "/scan_laserscan"
    assert evaluated[2]["use_lifecycle_manager"] is False
    assert evaluated[2]["roughness_enable_near_chain_gate"] is False
    assert evaluated[2]["roughness_enable_spatial_memory_gate"] is False
    assert (
        evaluated[2]["roughness_spatial_memory_use_pose_graph_anchors"]
        is False
    )
    assert (
        evaluated[2]["roughness_enable_spatial_passage_hysteresis"]
        is False
    )
    assert evaluated[2]["roughness_spatial_passage_radius_m"] == 0.55
    assert evaluated[2]["roughness_spatial_passage_max_age_s"] == 3.5


def test_live_spatial_passage_requires_spatial_memory():
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(
        launch_description,
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_enable_near_chain_gate": "true",
            "roughness_innovation_enable_spatial_memory_gate": "false",
            "roughness_innovation_enable_spatial_passage_hysteresis": "true",
        },
    )

    with pytest.raises(RuntimeError, match="requires"):
        module.validate_roughness_innovation_frontend(context)


def test_live_spatial_memory_requires_near_chain_gate():
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(
        launch_description,
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_enable_near_chain_gate": "false",
            "roughness_innovation_enable_spatial_memory_gate": "true",
        },
    )

    with pytest.raises(RuntimeError, match="near_chain"):
        module.validate_roughness_innovation_frontend(context)


@pytest.mark.parametrize("value", ["true", "maybe"])
def test_live_pose_graph_anchor_validation(value):
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(
        launch_description,
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_enable_near_chain_gate": "true",
            "roughness_innovation_enable_spatial_memory_gate": "false",
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                value,
        },
    )

    expected = "requires" if value == "true" else "must be a boolean"
    with pytest.raises(RuntimeError, match=expected):
        module.validate_roughness_innovation_frontend(context)


def test_live_roughness_features_require_custom_frontend():
    module = load_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(
        launch_description,
        {
            "use_roughness_innovation_slam_toolbox": "false",
            "roughness_innovation_enable_near_chain_gate": "true",
        },
    )

    with pytest.raises(RuntimeError, match="require"):
        module.validate_roughness_innovation_frontend(context)


def test_bag_roughness_innovation_frontend_is_opt_in_and_observe_by_default():
    module = load_bag_launch_module()
    launch_description = module.generate_launch_description()
    declarations = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    executables = {
        getattr(entity, "_Node__node_executable", None)
        for entity in launch_description.entities
        if isinstance(entity, Node)
    }

    assert (
        default_value(
            declarations["use_roughness_innovation_slam_toolbox"]
        )
        == "false"
    )
    assert default_value(declarations["roughness_innovation_mode"]) == "observe"
    assert (
        default_value(
            declarations["roughness_innovation_scan_qos_reliable"]
        )
        == "false"
    )
    assert (
        default_value(declarations["roughness_innovation_scan_qos_depth"])
        == "5"
    )
    assert default_value(declarations["roughness_innovation_min_rough_samples"]) == "3"
    assert (
        default_value(
            declarations["roughness_innovation_enable_near_chain_gate"]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_near_chain_recovery_duration_s"
            ]
        )
        == "2.0"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_near_chain_loop_closing_cooldown_s"
            ]
        )
        == "0.0"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_near_chain_min_rough_samples"
            ]
        )
        == "1"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_near_chain_gated_min_response"
            ]
        )
        == "1.1"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_disable_loop_closing_during_gate"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_enable_spatial_memory_gate"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_spatial_memory_use_pose_graph_anchors"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations["roughness_innovation_spatial_memory_radius_m"]
        )
        == "0.6"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_spatial_memory_min_revisit_separation_s"
            ]
        )
        == "20.0"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_enable_spatial_passage_hysteresis"
            ]
        )
        == "false"
    )
    assert (
        default_value(
            declarations["roughness_innovation_spatial_passage_radius_m"]
        )
        == "0.55"
    )
    assert (
        default_value(
            declarations["roughness_innovation_spatial_passage_max_age_s"]
        )
        == "3.5"
    )
    assert (
        default_value(
            declarations["roughness_innovation_min_rough_duration_s"]
        )
        == "0.02"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_experimental_reject_scan_stamps_ns"
            ]
        )
        == ""
    )
    assert (
        default_value(
            declarations["roughness_innovation_enable_post_confirmation"]
        )
        == "true"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_post_confirmation_min_base_scans"
            ]
        )
        == "8"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_post_confirmation_max_base_scans"
            ]
        )
        == "10"
    )
    assert (
        default_value(
            declarations[
                "roughness_innovation_post_confirmation_max_age_s"
            ]
        )
        == "2.0"
    )
    assert "roughness_aware_slam_toolbox_node" in executables


def test_bag_post_confirmation_parameters_are_wired_to_the_custom_node():
    module = load_bag_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(launch_description)
    node = next(
        entity for entity in launch_description.entities
        if isinstance(entity, Node)
        and getattr(entity, "_Node__node_executable", None)
        == "roughness_aware_slam_toolbox_node"
    )
    evaluated = evaluate_parameters(
        context,
        getattr(node, "_Node__parameters"),
    )
    assert len(evaluated) == 2
    params = evaluated[1]

    assert params["roughness_enable_post_confirmation"] is True
    assert params["roughness_post_confirmation_max_age_s"] == 2.0
    assert params["roughness_post_confirmation_min_base_scans"] == 8
    assert params["roughness_post_confirmation_max_base_scans"] == 10
    assert params["roughness_post_confirmation_min_response"] == 0.30
    assert params["roughness_post_confirmation_min_response_gain"] == 0.05
    assert params["roughness_enable_near_chain_gate"] is False
    assert params["roughness_near_chain_recovery_duration_s"] == 2.0
    assert params["roughness_near_chain_loop_closing_cooldown_s"] == 0.0
    assert params["roughness_near_chain_min_rough_samples"] == 1
    assert params["roughness_near_chain_gated_min_response"] == 1.1
    assert (
        params["roughness_disable_loop_closing_during_near_chain_gate"]
        is False
    )
    assert params["roughness_enable_spatial_memory_gate"] is False
    assert (
        params["roughness_spatial_memory_use_pose_graph_anchors"] is False
    )
    assert params["roughness_spatial_memory_radius_m"] == 0.6
    assert (
        params["roughness_spatial_memory_min_revisit_separation_s"]
        == 20.0
    )
    assert (
        params["roughness_spatial_memory_min_heading_delta_deg"]
        == 126.0
    )
    assert params["roughness_spatial_memory_merge_radius_m"] == 0.10
    assert params["roughness_spatial_memory_max_zones"] == 200
    assert params["roughness_enable_spatial_passage_hysteresis"] is False
    assert params["roughness_spatial_passage_radius_m"] == 0.55
    assert params["roughness_spatial_passage_max_age_s"] == 3.5
    assert (
        params[
            "roughness_post_confirmation_min_covariance_improvement_ratio"
        ]
        == 1.5
    )


def test_bag_replay_defaults_to_the_standard_mapper_configuration():
    module = load_bag_launch_module()
    launch_description = module.generate_launch_description()
    declarations = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert Path(default_value(declarations["params_file"])).name == "mapper_params.yaml"


def test_bag_custom_slam_frontends_are_mutually_exclusive():
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_anchor_slam_toolbox": "true",
            "use_roughness_innovation_slam_toolbox": "true",
        }
    )

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        module.validate_custom_slam_toolbox_selection(context)


@pytest.mark.parametrize(
    (
        "custom",
        "near_chain",
        "spatial_memory",
        "pose_graph_anchors",
        "spatial_passage",
    ),
    [
        ("false", "true", "false", "false", "false"),
        ("true", "false", "true", "false", "false"),
        ("true", "true", "false", "true", "false"),
        ("true", "true", "false", "false", "true"),
    ],
)
def test_bag_roughness_feature_dependencies_are_enforced(
        custom, near_chain, spatial_memory, pose_graph_anchors,
        spatial_passage):
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": custom,
            "roughness_innovation_enable_near_chain_gate": near_chain,
            "roughness_innovation_enable_spatial_memory_gate": spatial_memory,
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                pose_graph_anchors,
            "roughness_innovation_enable_spatial_passage_hysteresis":
                spatial_passage,
        }
    )

    with pytest.raises(RuntimeError, match="require"):
        module.validate_roughness_innovation_dependencies(context)


def test_bag_complete_roughness_feature_chain_is_accepted():
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_enable_near_chain_gate": "true",
            "roughness_innovation_enable_spatial_memory_gate": "true",
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                "true",
            "roughness_innovation_enable_spatial_passage_hysteresis": "true",
        }
    )

    assert module.validate_roughness_innovation_dependencies(context) == []


def test_bag_pose_graph_anchor_rejects_invalid_boolean():
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_enable_near_chain_gate": "true",
            "roughness_innovation_enable_spatial_memory_gate": "true",
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                "maybe",
            "roughness_innovation_enable_spatial_passage_hysteresis":
                "false",
        }
    )

    with pytest.raises(RuntimeError, match="must be a boolean"):
        module.validate_roughness_innovation_dependencies(context)


def test_bag_restricted_playback_rejects_an_unlisted_innovation_imu_topic():
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "play_slam_topics_only": "true",
            "roughness_innovation_imu_topic": "/custom/imu",
            "start_pointcloud_projection": "false",
        }
    )

    with pytest.raises(RuntimeError, match="excluded by play_slam_topics_only"):
        module.validate_roughness_innovation_playback_topics(context)


def test_bag_full_playback_accepts_a_custom_innovation_imu_topic():
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "play_slam_topics_only": "false",
            "roughness_innovation_imu_topic": "/custom/imu",
            "start_pointcloud_projection": "false",
        }
    )

    assert module.validate_roughness_innovation_playback_topics(context) == []


@pytest.mark.parametrize(
    ("scan_topic", "slam_scan_topic", "expected"),
    [
        ("/raw_scan", "", "/raw_scan"),
        ("/raw_scan", "/pre_filtered_scan", "/pre_filtered_scan"),
    ],
)
def test_bag_slam_scan_topic_selection_preserves_explicit_override(
        scan_topic, slam_scan_topic, expected):
    module = load_bag_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "scan_topic": scan_topic,
            "slam_scan_topic": slam_scan_topic,
        }
    )
    selected = module.slam_toolbox_scan_topic_substitution(
        module.LaunchConfiguration("scan_topic"),
        module.LaunchConfiguration("slam_scan_topic"),
    )

    assert selected.perform(context) == expected


def test_slam_bringup_preserves_custom_yaml_values_without_overrides():
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_lifecycle_manager": "false",
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
        }
    )

    assert module.mapper_parameter_overrides(context) == {
        "use_lifecycle_manager": False,
    }


def test_slam_bringup_applies_only_explicit_typed_overrides():
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_lifecycle_manager": "true",
            "use_sim_time": "true",
            "throttle_scans": "3",
            "resolution": "0.03",
        }
    )

    assert module.mapper_parameter_overrides(context) == {
        "use_lifecycle_manager": True,
        "use_sim_time": True,
        "throttle_scans": 3,
        "resolution": 0.03,
    }


def test_slam_bringup_custom_frontend_is_opt_in_and_uses_profile():
    module = load_bringup_launch_module()
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "params_file": str(
                package_share / "config" / "mapper_params.yaml"
            ),
            "autostart": "true",
            "use_lifecycle_manager": "false",
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_params_file": str(
                package_share
                / "config"
                / "slam_roughness_innovation_params.yaml"
            ),
            "roughness_innovation_imu_topic": "/imu/data",
            "roughness_innovation_scan_qos_reliable": "true",
            "roughness_innovation_scan_qos_depth": "2000",
            "roughness_innovation_near_chain_loop_closing_cooldown_s": "1.1",
            "roughness_innovation_enable_near_chain_gate": "true",
            "roughness_innovation_enable_spatial_memory_gate": "true",
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                "true",
            "roughness_innovation_enable_spatial_passage_hysteresis": "true",
            "roughness_innovation_spatial_passage_radius_m": "0.65",
            "roughness_innovation_spatial_passage_max_age_s": "4.25",
        }
    )

    actions = module.launch_slam_toolbox(context)

    assert len(actions) == 1
    assert getattr(actions[0], "_Node__node_executable", None) == (
        "roughness_aware_slam_toolbox_node"
    )
    evaluated = evaluate_parameters(
        context,
        getattr(actions[0], "_Node__parameters"),
    )
    assert len(evaluated) == 3
    assert evaluated[2]["enable_roughness_innovation_gate"] is True
    assert evaluated[2]["roughness_imu_topic"] == "/imu/data"
    assert evaluated[2]["roughness_scan_qos_reliable"] is True
    assert evaluated[2]["roughness_scan_qos_depth"] == 2000
    assert (
        evaluated[2]["roughness_near_chain_loop_closing_cooldown_s"]
        == 1.1
    )
    assert evaluated[2]["roughness_enable_near_chain_gate"] is True
    assert (
        evaluated[2]["roughness_enable_spatial_memory_gate"]
        is True
    )
    assert (
        evaluated[2]["roughness_spatial_memory_use_pose_graph_anchors"]
        is True
    )
    assert (
        evaluated[2]["roughness_enable_spatial_passage_hysteresis"]
        is True
    )
    assert evaluated[2]["roughness_spatial_passage_radius_m"] == 0.65
    assert evaluated[2]["roughness_spatial_passage_max_age_s"] == 4.25


@pytest.mark.parametrize(
    ("autostart", "use_lifecycle_manager"),
    [
        ("false", "false"),
        ("true", "true"),
    ],
)
def test_slam_bringup_custom_frontend_rejects_lifecycle_mismatch(
        autostart, use_lifecycle_manager):
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "params_file": "/tmp/mapper.yaml",
            "autostart": autostart,
            "use_lifecycle_manager": use_lifecycle_manager,
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_params_file": "/tmp/roughness.yaml",
            "roughness_innovation_imu_topic": "/imu/data",
            "roughness_innovation_near_chain_loop_closing_cooldown_s": "0.0",
            "roughness_innovation_enable_near_chain_gate": "false",
            "roughness_innovation_enable_spatial_memory_gate": "false",
            "roughness_innovation_enable_spatial_passage_hysteresis": "false",
        }
    )

    with pytest.raises(RuntimeError):
        module.launch_slam_toolbox(context)


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "invalid"])
def test_slam_bringup_rejects_invalid_loop_closing_cooldown(value):
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "params_file": "/tmp/mapper.yaml",
            "autostart": "true",
            "use_lifecycle_manager": "false",
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_params_file": "/tmp/roughness.yaml",
            "roughness_innovation_imu_topic": "/imu/data",
            "roughness_innovation_near_chain_loop_closing_cooldown_s": value,
            "roughness_innovation_enable_near_chain_gate": "false",
            "roughness_innovation_enable_spatial_memory_gate": "false",
            "roughness_innovation_enable_spatial_passage_hysteresis": "false",
        }
    )

    with pytest.raises(RuntimeError, match="cooldown"):
        module.launch_slam_toolbox(context)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("roughness_innovation_enable_spatial_passage_hysteresis", "maybe"),
        ("roughness_innovation_spatial_passage_radius_m", "-0.1"),
        ("roughness_innovation_spatial_passage_radius_m", "nan"),
        ("roughness_innovation_spatial_passage_radius_m", "inf"),
        ("roughness_innovation_spatial_passage_radius_m", "invalid"),
        ("roughness_innovation_spatial_passage_max_age_s", "-0.1"),
        ("roughness_innovation_spatial_passage_max_age_s", "nan"),
        ("roughness_innovation_spatial_passage_max_age_s", "inf"),
        ("roughness_innovation_spatial_passage_max_age_s", "invalid"),
    ],
)
def test_slam_bringup_rejects_invalid_spatial_passage_overrides(name, value):
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "params_file": "/tmp/mapper.yaml",
            "autostart": "true",
            "use_lifecycle_manager": "false",
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_params_file": "/tmp/roughness.yaml",
            "roughness_innovation_imu_topic": "/imu/data",
            "roughness_innovation_near_chain_loop_closing_cooldown_s": "0.0",
            "roughness_innovation_enable_near_chain_gate": "false",
            "roughness_innovation_enable_spatial_memory_gate": "false",
            "roughness_innovation_enable_spatial_passage_hysteresis": "false",
            "roughness_innovation_spatial_passage_radius_m": "0.55",
            "roughness_innovation_spatial_passage_max_age_s": "3.5",
            name: value,
        }
    )

    with pytest.raises(RuntimeError, match="spatial_passage"):
        module.launch_slam_toolbox(context)


def test_slam_bringup_rejects_invalid_pose_graph_anchor_boolean():
    module = load_bringup_launch_module()
    launch_description = module.generate_launch_description()
    context = context_with_declared_defaults(
        launch_description,
        {
            "use_roughness_innovation_slam_toolbox": "true",
            "roughness_innovation_spatial_memory_use_pose_graph_anchors":
                "maybe",
        },
    )

    with pytest.raises(RuntimeError, match="pose_graph_anchors"):
        module.launch_slam_toolbox(context)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("use_sim_time", "maybe"),
        ("throttle_scans", "0"),
        ("resolution", "-0.01"),
        ("resolution", "nan"),
        ("resolution", "inf"),
    ],
)
def test_slam_bringup_rejects_invalid_explicit_overrides(name, value):
    module = load_bringup_launch_module()
    context = LaunchContext()
    context.launch_configurations.update(
        {
            "use_lifecycle_manager": "false",
            "use_sim_time": "",
            "throttle_scans": "",
            "resolution": "",
            name: value,
        }
    )

    with pytest.raises(RuntimeError):
        module.mapper_parameter_overrides(context)


def test_roughness_innovation_profile_observes_without_graph_intervention():
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    config_path = (
        package_share
        / "config"
        / "slam_roughness_innovation_params.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["slam_toolbox"]["ros__parameters"]

    assert params["enable_roughness_innovation_gate"] is True
    assert params["roughness_innovation_mode"] == "observe"
    assert params["roughness_constraint_mode"] == "adaptive_memory_observe"
    assert params["roughness_enable_near_chain_gate"] is True
    # The 1.10 value remains available only when legacy_fixed_memory is
    # selected explicitly; adaptive_memory_observe never applies it.
    assert params["roughness_near_chain_gated_min_response"] > 1.0
    assert (
        params[
            "roughness_disable_loop_closing_during_near_chain_gate"
        ]
        is False
    )
    assert params["roughness_enable_spatial_memory_gate"] is False
    assert params["roughness_spatial_memory_use_pose_graph_anchors"] is False
    assert params["roughness_near_chain_loop_closing_cooldown_s"] == 0.0
    assert params["roughness_spatial_memory_radius_m"] == 0.6
    assert params["roughness_enable_spatial_passage_hysteresis"] is False
    assert params["roughness_spatial_passage_radius_m"] == 0.55
    assert params["roughness_spatial_passage_max_age_s"] == 3.5
    assert (
        params["roughness_spatial_memory_min_revisit_separation_s"]
        == 20.0
    )
    assert (
        params["roughness_spatial_memory_min_heading_delta_deg"]
        == 126.0
    )
    assert params["roughness_enable_place_recognition_observer"] is True
    assert params["roughness_enable_place_scan_match_observer"] is True
    assert params["roughness_place_evaluate_both_directions"] is False
    assert params["roughness_place_enable_reference_chain_observer"] is False
    assert params["roughness_place_reference_chain_scan_count"] == 5
    assert params["roughness_place_reference_chain_max_distance_m"] == 1.0
    assert params["roughness_place_reference_chain_max_state_gap"] == 5
    assert params["roughness_place_reference_chain_max_time_gap_s"] == 5.0
    assert params["roughness_place_candidate_top_k"] == 5
    assert params["roughness_place_min_revisit_separation_s"] == 20.0
    assert params["roughness_place_min_state_separation"] == 20
    assert params["roughness_place_constraint_mode"] == "observe"
    assert params["roughness_place_constraint_anchor_strategy"] == "best_single"
    assert params["roughness_place_constraint_group_covariance_scale"] == 1.0
    assert params["roughness_experimental_reject_scan_stamps_ns"] == ""


def test_mapper_uses_the_slam_toolbox_scan_queue_parameter():
    package_share = Path(get_package_share_directory("slam_2d_bringup"))
    config_path = package_share / "config" / "mapper_params.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["slam_toolbox"]["ros__parameters"]

    assert params["scan_queue_size"] >= 10
    assert "message_filter_queue_size" not in params
