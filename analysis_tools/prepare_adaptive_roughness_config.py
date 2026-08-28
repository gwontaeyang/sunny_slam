#!/usr/bin/env python3
"""Create an isolated SLAM replay config for adaptive roughness experiments."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


MODES = (
    "intervention_off",
    "legacy_fixed_memory",
    "adaptive_memory_observe",
    "adaptive_memory_only",
    "adaptive_quality_aware",
)

ADAPTIVE_DEFAULTS: dict[str, Any] = {
    "roughness_enable_near_chain_candidate_observer": True,
    "roughness_enable_loop_closure_candidate_observer": True,
    "roughness_enable_place_recognition_observer": True,
    "roughness_enable_place_scan_match_observer": True,
    "roughness_place_evaluate_both_directions": False,
    "roughness_place_enable_reference_chain_observer": False,
    "roughness_place_reference_chain_scan_count": 5,
    "roughness_place_reference_chain_max_distance_m": 1.0,
    "roughness_place_reference_chain_max_state_gap": 5,
    "roughness_place_reference_chain_max_time_gap_s": 5.0,
    "roughness_place_descriptor_bins": 16,
    "roughness_place_max_observations_per_event": 512,
    "roughness_place_max_events": 200,
    "roughness_place_candidate_top_k": 5,
    "roughness_place_min_revisit_separation_s": 20.0,
    "roughness_place_min_state_separation": 20,
    "roughness_place_constraint_mode": "observe",
    "roughness_place_constraint_anchor_strategy": "best_single",
    "roughness_place_min_fine_response": 0.30,
    "roughness_place_max_proposal_disagreement_translation_m": 0.30,
    "roughness_place_max_proposal_disagreement_yaw_deg": 5.0,
    "roughness_place_max_fine_xy_covariance_m2": 0.04,
    "roughness_place_max_fine_yaw_covariance_deg2": 25.0,
    "roughness_place_max_bidirectional_translation_error_m": 0.10,
    "roughness_place_max_bidirectional_yaw_error_deg": 2.0,
    "roughness_place_max_pair_alignment_translation_error_m": 0.50,
    "roughness_place_max_pair_alignment_yaw_error_deg": 3.0,
    "roughness_place_constraint_pair_covariance_scale": 1.0,
    "roughness_place_constraint_group_covariance_scale": 1.0,
    "roughness_place_constraint_xy_variance_floor_m2": 0.01,
    "roughness_place_constraint_yaw_variance_floor_deg2": 1.0,
    "roughness_near_chain_observer_min_parity_candidates": 20,
    "roughness_adaptive_memory_trigger_score": -1.0,
    "roughness_adaptive_memory_saturation_score": 5.5,
    "roughness_adaptive_memory_recovery_score": 1.0,
    "roughness_adaptive_memory_min_duration_s": 0.5,
    "roughness_adaptive_memory_max_duration_s": 3.5,
    "roughness_adaptive_memory_min_distance_m": 0.10,
    "roughness_adaptive_memory_max_distance_m": 0.55,
    "roughness_adaptive_memory_recovery_time_constant_s": 0.5,
    "roughness_near_chain_quality_calibrated": False,
    "roughness_near_chain_quality_response_bad": 0.08,
    "roughness_near_chain_quality_response_good": 0.30,
    "roughness_near_chain_quality_translation_good_m": 0.02,
    "roughness_near_chain_quality_translation_bad_m": 0.10,
    "roughness_near_chain_quality_yaw_good_deg": 1.0,
    "roughness_near_chain_quality_yaw_bad_deg": 5.0,
    "roughness_near_chain_quality_xy_covariance_good_m2": 0.0025,
    "roughness_near_chain_quality_xy_covariance_bad_m2": 0.04,
    "roughness_near_chain_quality_yaw_covariance_good_rad2": 0.0003046174,
    "roughness_near_chain_quality_yaw_covariance_bad_rad2": 0.0304617420,
    "roughness_adaptive_downweight_risk": 0.35,
    "roughness_adaptive_reject_risk": 0.80,
    "roughness_adaptive_minimum_information_scale": 0.10,
    "roughness_adaptive_dynamic_threshold_max": 1.0,
    "roughness_enable_rotation_consistency_guard": True,
    "roughness_rotation_history_size": 64,
    "roughness_rotation_minimum_history_batches": 8,
    "roughness_rotation_outlier_sigma": 6.0,
    "roughness_rotation_extreme_yaw_sigma": 12.0,
    "roughness_rotation_yaw_scale_floor_deg": 0.4,
    "roughness_rotation_translation_scale_floor_m": 0.03,
    "roughness_rotation_weighted_disagreement_scale_floor": 0.25,
    "roughness_rotation_chain_overlap_trigger": 0.75,
    "roughness_rotation_maximum_repeat_state_gap": 2,
    "roughness_rotation_stable_recovery_batches": 3,
}


def adaptive_config(
    base: dict[str, Any],
    mode: str,
    *,
    quality_calibrated: bool = False,
    parameter_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode: {mode}")
    result = copy.deepcopy(base)
    node = result.get("slam_toolbox")
    if not isinstance(node, dict):
        raise ValueError("base config is missing slam_toolbox mapping")
    parameters = node.get("ros__parameters")
    if not isinstance(parameters, dict):
        raise ValueError("base config is missing slam_toolbox.ros__parameters")
    parameters.update(ADAPTIVE_DEFAULTS)
    overrides = parameter_overrides or {}
    unknown = sorted(set(overrides) - set(ADAPTIVE_DEFAULTS))
    if unknown:
        raise ValueError(
            "calibration contains unsupported parameters: " + ", ".join(unknown)
        )
    parameters.update(overrides)
    parameters["roughness_constraint_mode"] = mode
    parameters["roughness_near_chain_quality_calibrated"] = (
        quality_calibrated
        or bool(overrides.get("roughness_near_chain_quality_calibrated", False))
    )
    parameters["roughness_enable_near_chain_gate"] = (
        mode == "legacy_fixed_memory"
    )
    if mode != "legacy_fixed_memory":
        parameters["roughness_enable_spatial_memory_gate"] = False
        parameters["roughness_spatial_memory_use_pose_graph_anchors"] = False
        parameters["roughness_enable_spatial_passage_hysteresis"] = False
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--quality-calibrated",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "YAML/JSON mapping of derived adaptive parameters, optionally "
            "nested below a 'parameters' key. Unknown keys fail closed."
        ),
    )
    return parser.parse_args()


def load_parameter_overrides(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    value = yaml.safe_load(path.expanduser().read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("calibration must contain a YAML/JSON mapping")
    if "parameters" in value:
        value = value["parameters"]
    if not isinstance(value, dict):
        raise ValueError("calibration parameters must be a mapping")
    return value


def main() -> int:
    args = parse_args()
    base_path = args.base.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(base, dict):
        raise ValueError("base config must contain a YAML mapping")
    result = adaptive_config(
        base,
        args.mode,
        quality_calibrated=args.quality_calibrated,
        parameter_overrides=load_parameter_overrides(args.calibration),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
