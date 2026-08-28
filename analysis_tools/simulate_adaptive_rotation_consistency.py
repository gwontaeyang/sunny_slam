#!/usr/bin/env python3
"""Shadow-simulate an adaptive near-chain rotation consistency guard.

The input is the exact Karto ``LinkNearChains`` candidate trace.  This tool
does not claim that a candidate is a ground-truth outlier.  It asks a narrower
question: can a rolling robust model identify repeated, mutually correlated
pose corrections before they are fused into every consecutive graph node?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class PolicyConfig:
    history_size: int = 64
    minimum_history_batches: int = 8
    outlier_sigma: float = 6.0
    extreme_yaw_sigma: float = 12.0
    yaw_scale_floor_deg: float = 0.4
    translation_scale_floor_m: float = 0.03
    consistency_scale_floor: float = 0.25
    chain_overlap_trigger: float = 0.75
    maximum_repeat_state_gap: int = 2
    stable_recovery_batches: int = 3
    response_epsilon: float = 2.0e-6
    maximum_dynamic_threshold: float = 1.0


@dataclass(frozen=True)
class RobustLimit:
    available: bool = False
    median: float = math.nan
    scale: float = math.nan
    limit: float = math.nan
    extreme_limit: float = math.nan


@dataclass
class PolicyState:
    yaw_history_deg: deque[float]
    translation_history_m: deque[float]
    consistency_history: deque[float]
    stable_batch_count: int = 0
    latch_active: bool = False
    previous_outlier_target: int = -1
    previous_outlier_chains: tuple[frozenset[int], ...] = ()

    @classmethod
    def create(cls, config: PolicyConfig) -> "PolicyState":
        return cls(
            yaw_history_deg=deque(maxlen=config.history_size),
            translation_history_m=deque(maxlen=config.history_size),
            consistency_history=deque(maxlen=config.history_size),
        )


def finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def int_value(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def chain_ids(value: Any) -> frozenset[int]:
    text = str(value or "").strip()
    if not text or text == "none":
        return frozenset()
    result: set[int] = set()
    for token in text.split(","):
        try:
            result.add(int(token))
        except ValueError:
            return frozenset()
    return frozenset(result)


def robust_limit(
    values: Sequence[float],
    minimum_samples: int,
    sigma_multiplier: float,
    scale_floor: float,
    extreme_sigma_multiplier: float | None = None,
) -> RobustLimit:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < minimum_samples:
        return RobustLimit()
    center = statistics.median(finite)
    mad = statistics.median(abs(value - center) for value in finite)
    scale = max(MAD_TO_SIGMA * mad, scale_floor)
    extreme_sigma = (
        sigma_multiplier
        if extreme_sigma_multiplier is None
        else extreme_sigma_multiplier
    )
    return RobustLimit(
        available=True,
        median=center,
        scale=scale,
        limit=center + sigma_multiplier * scale,
        extreme_limit=center + extreme_sigma * scale,
    )


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def maximum_prior_overlap(
    chain: frozenset[int],
    prior: Iterable[frozenset[int]],
) -> float:
    return max((jaccard(chain, item) for item in prior), default=0.0)


def dynamic_threshold(
    rows: Sequence[dict[str, Any]],
    reject_flags: Sequence[bool],
    config: PolicyConfig,
) -> tuple[float, bool]:
    responses = [finite_float(row.get("normalized_response")) for row in rows]
    if len(responses) != len(reject_flags) or any(
        not math.isfinite(value) for value in responses
    ):
        return math.nan, True
    rejected = [
        value for value, reject in zip(responses, reject_flags) if reject
    ]
    if not rejected:
        return math.nan, False
    preserved = [
        value for value, reject in zip(responses, reject_flags) if not reject
    ]
    highest_rejected = max(rejected)
    if preserved:
        lowest_preserved = min(preserved)
        if highest_rejected + 2.0 * config.response_epsilon >= lowest_preserved:
            return math.nan, True
        requested = 0.5 * (highest_rejected + lowest_preserved)
    else:
        requested = highest_rejected + 2.0 * config.response_epsilon
    if requested > config.maximum_dynamic_threshold:
        return math.nan, True
    return requested, False


def group_candidates(
    rows: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int_value(row.get("target_state_id")),
            int_value(row.get("stamp_ns"), 0),
            int_value(row.get("candidate_index"), 0),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    key: tuple[int, int] | None = None
    for row in ordered:
        current = (
            int_value(row.get("target_state_id")),
            int_value(row.get("stamp_ns"), 0),
        )
        if current != key:
            groups.append([])
            key = current
        groups[-1].append(row)
    return groups


def candidate_features(row: dict[str, Any]) -> tuple[float, float, float]:
    yaw_rad = finite_float(row.get("yaw_disagreement_rad"))
    yaw_deg = abs(math.degrees(yaw_rad)) if math.isfinite(yaw_rad) else math.nan
    translation = finite_float(row.get("translation_disagreement_m"))
    weighted = finite_float(
        row.get("scan_match_covariance_weighted_disagreement_sq")
    )
    consistency = math.log1p(weighted) if weighted >= 0.0 else math.nan
    return yaw_deg, translation, consistency


def simulate_rows(
    rows: Sequence[dict[str, Any]],
    config: PolicyConfig = PolicyConfig(),
) -> list[dict[str, Any]]:
    state = PolicyState.create(config)
    decisions: list[dict[str, Any]] = []
    for batch in group_candidates(rows):
        eligible = [row for row in batch if is_true(row.get("chain_size_accepted"))]
        if not eligible:
            continue
        target = int_value(eligible[0].get("target_state_id"))
        yaw_limit = robust_limit(
            list(state.yaw_history_deg),
            config.minimum_history_batches,
            config.outlier_sigma,
            config.yaw_scale_floor_deg,
            config.extreme_yaw_sigma,
        )
        translation_limit = robust_limit(
            list(state.translation_history_m),
            config.minimum_history_batches,
            config.outlier_sigma,
            config.translation_scale_floor_m,
        )
        consistency_limit = robust_limit(
            list(state.consistency_history),
            config.minimum_history_batches,
            config.outlier_sigma,
            config.consistency_scale_floor,
        )
        model_available = (
            yaw_limit.available
            and translation_limit.available
            and consistency_limit.available
        )

        features = [candidate_features(row) for row in eligible]
        risky: list[bool] = []
        overlaps: list[float] = []
        for row, (yaw_deg, translation, consistency) in zip(eligible, features):
            finite = all(
                math.isfinite(value)
                for value in (yaw_deg, translation, consistency)
            )
            yaw_outlier = finite and yaw_deg > yaw_limit.limit
            translation_outlier = finite and translation > translation_limit.limit
            consistency_outlier = finite and consistency > consistency_limit.limit
            yaw_extreme = finite and yaw_deg > yaw_limit.extreme_limit
            risky.append(
                model_available
                and (yaw_extreme or (yaw_outlier and translation_outlier))
                and consistency_outlier
            )
            overlaps.append(
                maximum_prior_overlap(
                    chain_ids(row.get("chain_state_ids")),
                    state.previous_outlier_chains,
                )
            )

        memory_strength = max(
            (finite_float(row.get("memory_strength")) for row in eligible),
            default=math.nan,
        )
        memory_trigger = math.isfinite(memory_strength) and memory_strength > 0.0
        repeated_trigger = (
            state.previous_outlier_target >= 0
            and 0 < target - state.previous_outlier_target
            <= config.maximum_repeat_state_gap
            and any(
                risk and overlap >= config.chain_overlap_trigger
                for risk, overlap in zip(risky, overlaps)
            )
        )
        trigger = any(risky) and (
            state.latch_active or memory_trigger or repeated_trigger
        )
        proposed_rejects = [risk and trigger for risk in risky]
        requested_threshold, fail_open = dynamic_threshold(
            eligible, proposed_rejects, config
        )
        applied_rejects = (
            [False] * len(eligible) if fail_open else proposed_rejects
        )

        if any(risky):
            state.previous_outlier_target = target
            state.previous_outlier_chains = tuple(
                chain_ids(row.get("chain_state_ids"))
                for row, risk in zip(eligible, risky)
                if risk
            )
            state.stable_batch_count = 0
            if trigger and not fail_open:
                state.latch_active = True
        else:
            state.previous_outlier_target = -1
            state.previous_outlier_chains = ()
            if state.latch_active:
                state.stable_batch_count += 1
                if state.stable_batch_count >= config.stable_recovery_batches:
                    state.latch_active = False
                    state.stable_batch_count = 0

        for row, values, risk, overlap, proposed, applied in zip(
            eligible,
            features,
            risky,
            overlaps,
            proposed_rejects,
            applied_rejects,
        ):
            yaw_deg, translation, consistency = values
            decisions.append(
                {
                    "stamp_ns": int_value(row.get("stamp_ns"), 0),
                    "source_state_id": int_value(row.get("source_state_id")),
                    "target_state_id": target,
                    "normalized_response": finite_float(
                        row.get("normalized_response")
                    ),
                    "yaw_disagreement_deg": yaw_deg,
                    "translation_disagreement_m": translation,
                    "log1p_covariance_weighted_disagreement": consistency,
                    "memory_strength": memory_strength,
                    "history_count": len(state.yaw_history_deg),
                    "yaw_limit_deg": yaw_limit.limit,
                    "yaw_extreme_limit_deg": yaw_limit.extreme_limit,
                    "translation_limit_m": translation_limit.limit,
                    "consistency_limit": consistency_limit.limit,
                    "chain_overlap": overlap,
                    "model_available": int(model_available),
                    "candidate_risky": int(risk),
                    "memory_trigger": int(memory_trigger),
                    "repeated_trigger": int(repeated_trigger),
                    "latch_active": int(state.latch_active),
                    "proposed_reject": int(proposed),
                    "fail_open": int(fail_open),
                    "applied_reject": int(applied),
                    "requested_threshold": requested_threshold,
                }
            )

        stable_features = [
            values for values, risk in zip(features, risky) if not risk
        ]
        if stable_features:
            state.yaw_history_deg.append(
                statistics.median(value[0] for value in stable_features)
            )
            state.translation_history_m.append(
                statistics.median(value[1] for value in stable_features)
            )
            state.consistency_history.append(
                statistics.median(value[2] for value in stable_features)
            )
    return decisions


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*_near_chain_candidate_events.csv"))


def bag_name(path: Path) -> str:
    suffix = "_near_chain_candidate_events.csv"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-bag", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PolicyConfig()
    all_decisions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for path in candidate_files(args.candidate_root.expanduser().resolve()):
        name = bag_name(path)
        decisions = simulate_rows(read_rows(path), config)
        for row in decisions:
            row["bag"] = name
        all_decisions.extend(decisions)
        summaries.append(
            {
                "bag": name,
                "eligible_candidate_count": len(decisions),
                "model_available_count": sum(
                    row["model_available"] for row in decisions
                ),
                "risky_candidate_count": sum(
                    row["candidate_risky"] for row in decisions
                ),
                "proposed_reject_count": sum(
                    row["proposed_reject"] for row in decisions
                ),
                "applied_reject_count": sum(
                    row["applied_reject"] for row in decisions
                ),
                "fail_open_count": sum(row["fail_open"] for row in decisions),
                "first_applied_reject_target": next(
                    (
                        row["target_state_id"]
                        for row in decisions
                        if row["applied_reject"]
                    ),
                    -1,
                ),
            }
        )
    output = args.output_dir.expanduser().resolve()
    write_rows(output / "csv" / "candidate_decisions.csv", all_decisions)
    write_rows(output / "csv" / "dataset_summary.csv", summaries)
    (output / "report").mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "config": asdict(config),
        "dataset_count": len(summaries),
        "candidate_count": len(all_decisions),
        "risky_candidate_count": sum(
            row["candidate_risky"] for row in all_decisions
        ),
        "applied_reject_count": sum(
            row["applied_reject"] for row in all_decisions
        ),
        "fail_open_count": sum(row["fail_open"] for row in all_decisions),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Adaptive rotation-consistency shadow simulation",
        "",
        "This is a candidate-policy simulation, not a replay accuracy result.",
        "It uses only evidence available before Karto fuses each optional near-chain candidate.",
        "",
        f"- Datasets: {payload['dataset_count']}",
        f"- Eligible candidates: {payload['candidate_count']}",
        f"- Robust-risk candidates: {payload['risky_candidate_count']}",
        f"- Dynamically separable proposed rejects: {payload['applied_reject_count']}",
        f"- Response-order fail-open candidates: {payload['fail_open_count']}",
        "",
        "| bag | eligible | risky | proposed | separable reject | fail-open | first target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            f"| {row['bag']} | {row['eligible_candidate_count']} | "
            f"{row['risky_candidate_count']} | {row['proposed_reject_count']} | "
            f"{row['applied_reject_count']} | {row['fail_open_count']} | "
            f"{row['first_applied_reject_target']} |"
        )
    (output / "report" / "rotation_consistency_shadow.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
