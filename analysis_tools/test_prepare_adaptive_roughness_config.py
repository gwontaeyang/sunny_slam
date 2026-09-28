import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("prepare_adaptive_roughness_config.py")
SPEC = importlib.util.spec_from_file_location(
    "prepare_adaptive_roughness_config",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def base_config():
    return {
        "slam_toolbox": {
            "ros__parameters": {
                "scan_topic": "/scan_laserscan",
                "link_match_minimum_response_fine": 0.08,
                "custom_existing_value": 7,
            }
        }
    }


@pytest.mark.parametrize("mode", MODULE.MODES)
def test_each_mode_is_explicit_and_preserves_base(mode):
    result = MODULE.adaptive_config(base_config(), mode)
    parameters = result["slam_toolbox"]["ros__parameters"]
    assert parameters["roughness_constraint_mode"] == mode
    assert parameters["scan_topic"] == "/scan_laserscan"
    assert parameters["custom_existing_value"] == 7
    assert parameters["roughness_enable_near_chain_gate"] == (
        mode == "legacy_fixed_memory"
    )
    assert parameters["roughness_place_constraint_mode"] == "observe"
    assert parameters["roughness_place_constraint_anchor_strategy"] == "best_single"
    assert parameters["roughness_place_evaluate_both_directions"] is False
    assert parameters["roughness_place_enable_reference_chain_observer"] is False
    assert parameters["roughness_place_reference_chain_scan_count"] == 5
    assert parameters["roughness_place_reference_chain_max_distance_m"] == 1.0
    assert parameters["roughness_place_reference_chain_max_state_gap"] == 5
    assert parameters["roughness_place_reference_chain_max_time_gap_s"] == 5.0
    assert parameters["roughness_place_constraint_group_covariance_scale"] == 1.0


def test_adaptive_modes_disable_fixed_passage_memory():
    base = base_config()
    parameters = base["slam_toolbox"]["ros__parameters"]
    parameters["roughness_enable_spatial_memory_gate"] = True
    parameters["roughness_enable_spatial_passage_hysteresis"] = True
    result = MODULE.adaptive_config(base, "adaptive_memory_observe")
    configured = result["slam_toolbox"]["ros__parameters"]
    assert configured["roughness_enable_spatial_memory_gate"] is False
    assert configured["roughness_enable_spatial_passage_hysteresis"] is False


def test_quality_calibration_requires_explicit_opt_in():
    default = MODULE.adaptive_config(base_config(), "adaptive_quality_aware")
    calibrated = MODULE.adaptive_config(
        base_config(),
        "adaptive_quality_aware",
        quality_calibrated=True,
    )
    assert default["slam_toolbox"]["ros__parameters"][
        "roughness_near_chain_quality_calibrated"
    ] is False
    assert calibrated["slam_toolbox"]["ros__parameters"][
        "roughness_near_chain_quality_calibrated"
    ] is True


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        MODULE.adaptive_config(base_config(), "unknown")


def test_calibration_overrides_are_applied_and_enable_quality_model(tmp_path):
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(
        "parameters:\n"
        "  roughness_near_chain_quality_calibrated: true\n"
        "  roughness_near_chain_quality_response_bad: 0.21\n"
        "  roughness_near_chain_quality_response_good: 0.43\n",
        encoding="utf-8",
    )

    result = MODULE.adaptive_config(
        base_config(),
        "adaptive_quality_aware",
        parameter_overrides=MODULE.load_parameter_overrides(calibration),
    )
    parameters = result["slam_toolbox"]["ros__parameters"]

    assert parameters["roughness_near_chain_quality_calibrated"] is True
    assert parameters["roughness_near_chain_quality_response_bad"] == 0.21
    assert parameters["roughness_near_chain_quality_response_good"] == 0.43


def test_place_constraint_soft_mode_requires_explicit_override():
    result = MODULE.adaptive_config(
        base_config(),
        "adaptive_quality_aware",
        parameter_overrides={"roughness_place_constraint_mode": "soft"},
    )
    assert result["slam_toolbox"]["ros__parameters"][
        "roughness_place_constraint_mode"
    ] == "soft"


def test_calibration_rejects_unknown_parameter():
    with pytest.raises(ValueError, match="unsupported parameters"):
        MODULE.adaptive_config(
            base_config(),
            "adaptive_quality_aware",
            parameter_overrides={"roughness_unknown": 1.0},
        )
