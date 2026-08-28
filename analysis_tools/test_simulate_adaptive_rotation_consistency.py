import math

import pytest

from simulate_adaptive_rotation_consistency import (
    PolicyConfig,
    dynamic_threshold,
    robust_limit,
    simulate_rows,
)


def row(
    target: int,
    yaw_deg: float,
    translation_m: float,
    *,
    response: float = 0.3,
    memory: float = 0.0,
    chain_start: int = 10,
):
    return {
        "stamp_ns": str(target * 1_000_000),
        "candidate_index": "0",
        "source_state_id": str(chain_start + 5),
        "target_state_id": str(target),
        "chain_size_accepted": "true",
        "chain_state_ids": ",".join(
            str(value) for value in range(chain_start, chain_start + 20)
        ),
        "normalized_response": str(response),
        "yaw_disagreement_rad": str(math.radians(yaw_deg)),
        "translation_disagreement_m": str(translation_m),
        "scan_match_covariance_weighted_disagreement_sq": str(
            max(yaw_deg * yaw_deg + translation_m * translation_m, 0.0)
        ),
        "memory_strength": str(memory),
    }


def stable_history(count: int = 8):
    return [row(index, 0.2, 0.02) for index in range(1, count + 1)]


def test_robust_limit_uses_mad_floor():
    result = robust_limit([0.2] * 8, 8, 6.0, 0.4, 12.0)
    assert result.available
    assert result.scale == 0.4
    assert result.limit == pytest.approx(2.6)
    assert result.extreme_limit == pytest.approx(5.0)


def test_roughness_prior_rejects_first_extreme_candidate_after_warmup():
    rows = stable_history() + [row(9, 16.0, 0.7, memory=0.8)]
    decisions = simulate_rows(rows)
    last = decisions[-1]
    assert last["candidate_risky"] == 1
    assert last["memory_trigger"] == 1
    assert last["applied_reject"] == 1


def test_repeated_overlapping_chain_triggers_without_imu():
    rows = stable_history()
    rows.extend(
        [
            row(9, 16.0, 0.7, memory=0.0),
            row(10, 15.0, 0.65, memory=0.0, chain_start=11),
        ]
    )
    decisions = simulate_rows(rows)
    assert decisions[-2]["applied_reject"] == 0
    assert decisions[-1]["repeated_trigger"] == 1
    assert decisions[-1]["applied_reject"] == 1


def test_missing_quality_fails_open():
    rows = stable_history() + [row(9, 16.0, 0.7, memory=1.0)]
    rows[-1]["scan_match_covariance_weighted_disagreement_sq"] = "nan"
    assert simulate_rows(rows)[-1]["applied_reject"] == 0


def test_overlapping_response_order_fails_open():
    config = PolicyConfig()
    candidates = [
        {"normalized_response": "0.4"},
        {"normalized_response": "0.3"},
    ]
    threshold, fail_open = dynamic_threshold(
        candidates, [True, False], config
    )
    assert math.isnan(threshold)
    assert fail_open


def test_simulation_is_deterministic():
    rows = stable_history() + [row(9, 16.0, 0.7, memory=0.8)]
    assert simulate_rows(rows) == simulate_rows(rows)
