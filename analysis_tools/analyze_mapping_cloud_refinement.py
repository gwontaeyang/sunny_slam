#!/usr/bin/env python3
"""Summarize completed-submap mapping-cloud loop refinements."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


PREFIX = "place_recognition_mapping_refinement_"


def _value(record, name, default=""):
    return record.get("values", {}).get(name, default)


def _bool(record, name):
    return str(_value(record, name, "false")).lower() == "true"


def _float(record, name, default=math.nan):
    try:
        return float(_value(record, name, default))
    except (TypeError, ValueError):
        return default


def _int(record, name, default=-1):
    value = _float(record, name, float(default))
    return int(value) if math.isfinite(value) else default


def load_events(path):
    records = []
    first_stamp_ns = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("name") != "scan_to_map_icp":
                continue
            stamp_ns = int(record["stamp_ns"])
            if first_stamp_ns is None:
                first_stamp_ns = stamp_ns
            if not _bool(record, f"{PREFIX}attempted"):
                continue
            records.append(event_from_record(record, first_stamp_ns))
    return records


def event_from_record(record, first_stamp_ns):
    stamp_ns = int(record["stamp_ns"])
    fields = {
        "stamp_ns": stamp_ns,
        "elapsed_s": (stamp_ns - first_stamp_ns) * 1.0e-9,
        "status": _value(record, f"{PREFIX}status", "missing"),
        "applied": _bool(record, f"{PREFIX}applied"),
        "place_recognition_accepted": _bool(record, "place_recognition_accepted"),
        "candidate_keyframe_index": _int(record, "place_recognition_candidate_index"),
        "candidate_submap_id": _int(record, "place_recognition_candidate_submap_id"),
        "source_points": _int(record, f"{PREFIX}source_points", 0),
        "target_points": _int(record, f"{PREFIX}target_points", 0),
        "before_inlier_ratio": _float(record, f"{PREFIX}before_inlier_ratio"),
        "after_inlier_ratio": _float(record, f"{PREFIX}after_inlier_ratio"),
        "before_inlier_rmse_m": _float(record, f"{PREFIX}before_inlier_rmse_m"),
        "after_inlier_rmse_m": _float(record, f"{PREFIX}after_inlier_rmse_m"),
        "before_target_coverage_ratio": _float(
            record, f"{PREFIX}before_target_coverage_ratio"
        ),
        "after_target_coverage_ratio": _float(
            record, f"{PREFIX}after_target_coverage_ratio"
        ),
        "before_score": _float(record, f"{PREFIX}before_score"),
        "after_score": _float(record, f"{PREFIX}after_score"),
        "primary_before_score": _float(record, f"{PREFIX}primary_before_score"),
        "primary_after_score": _float(record, f"{PREFIX}primary_after_score"),
        "correction_translation_m": _float(
            record, f"{PREFIX}correction_translation_m"
        ),
        "correction_rotation_deg": _float(
            record, f"{PREFIX}correction_rotation_deg"
        ),
        "prediction_jump_translation_m": _float(
            record, "place_recognition_prediction_jump_translation_m"
        ),
        "prediction_jump_rotation_deg": _float(
            record, "place_recognition_prediction_jump_rotation_deg"
        ),
        "map_correction_translation_m": _float(
            record, "place_recognition_map_correction_translation_m"
        ),
        "map_correction_rotation_deg": _float(
            record, "place_recognition_map_correction_rotation_deg"
        ),
        "alignment_correction_translation_m": _float(
            record, "place_recognition_alignment_correction_translation_m"
        ),
        "alignment_correction_rotation_deg": _float(
            record, "place_recognition_alignment_correction_rotation_deg"
        ),
        "mapping_refinement_alignment_translation_limit_m": _float(
            record,
            "place_recognition_mapping_refinement_keyframe_quarantine_"
            "correction_translation_m",
        ),
        "mapping_refinement_alignment_rotation_limit_deg": _float(
            record,
            "place_recognition_mapping_refinement_keyframe_quarantine_"
            "correction_rotation_deg",
        ),
        "minimum_target_coverage_ratio": _float(
            record, f"{PREFIX}min_target_coverage_ratio"
        ),
        "observable_rank": _int(record, f"{PREFIX}observable_rank", 0),
        "hessian_condition": _float(record, f"{PREFIX}hessian_condition"),
        "runtime_ms": _float(record, f"{PREFIX}runtime_ms"),
        "pose_graph_correction_applied": _bool(record, "pose_graph_correction_applied"),
        "pose_graph_new_loop_rejected": _bool(record, "pose_graph_new_loop_rejected"),
        "pose_graph_rejection_reason": _value(record, "pose_graph_rejection_reason"),
        "pose_graph_solver_status": _value(record, "pose_graph_solver_status"),
        "pose_graph_residual_before": _float(record, "pose_graph_residual_before"),
        "pose_graph_residual_after": _float(record, "pose_graph_residual_after"),
    }
    fields["score_change"] = fields["after_score"] - fields["before_score"]
    fields["score_reduction"] = fields["before_score"] - fields["after_score"]
    fields["rmse_improvement_m"] = (
        fields["before_inlier_rmse_m"] - fields["after_inlier_rmse_m"]
    )
    fields["primary_score_change"] = (
        fields["primary_after_score"] - fields["primary_before_score"]
    )
    return fields


def _finite_values(events, name, applied_only=False):
    values = [
        event[name]
        for event in events
        if (not applied_only or event["applied"])
        and isinstance(event[name], (int, float))
        and math.isfinite(event[name])
    ]
    return np.asarray(values, dtype=float)


def _distribution(values):
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarize(events):
    applied = [event for event in events if event["applied"]]
    downstream_accepted = [
        event
        for event in applied
        if event["place_recognition_accepted"]
        and event["pose_graph_correction_applied"]
    ]
    downstream_rejections = Counter(
        event["pose_graph_rejection_reason"] or "unspecified"
        for event in applied
        if not event["pose_graph_correction_applied"]
    )
    return {
        "attempted_count": len(events),
        "applied_count": len(applied),
        "applied_ratio": len(applied) / len(events) if events else 0.0,
        "status_counts": dict(sorted(Counter(event["status"] for event in events).items())),
        "pose_graph_correction_applied_count": sum(
            event["pose_graph_correction_applied"] for event in events
        ),
        "mapping_refinement_pose_graph_applied_count": len(downstream_accepted),
        "mapping_refinement_downstream_rejection_counts": dict(
            sorted(downstream_rejections.items())
        ),
        "pose_graph_new_loop_rejected_count": sum(
            event["pose_graph_new_loop_rejected"] for event in events
        ),
        "applied_score_change": _distribution(
            _finite_values(events, "score_change", applied_only=True)
        ),
        "applied_score_reduction": _distribution(
            _finite_values(events, "score_reduction", applied_only=True)
        ),
        "applied_rmse_improvement_m": _distribution(
            _finite_values(events, "rmse_improvement_m", applied_only=True)
        ),
        "applied_target_coverage_ratio": _distribution(
            _finite_values(events, "after_target_coverage_ratio", applied_only=True)
        ),
        "applied_primary_score_change": _distribution(
            _finite_values(events, "primary_score_change", applied_only=True)
        ),
        "applied_translation_correction_m": _distribution(
            _finite_values(events, "correction_translation_m", applied_only=True)
        ),
        "applied_rotation_correction_deg": _distribution(
            _finite_values(events, "correction_rotation_deg", applied_only=True)
        ),
        "applied_map_correction_translation_m": _distribution(
            _finite_values(events, "map_correction_translation_m", applied_only=True)
        ),
        "applied_alignment_correction_translation_m": _distribution(
            _finite_values(
                events, "alignment_correction_translation_m", applied_only=True
            )
        ),
        "applied_alignment_correction_rotation_deg": _distribution(
            _finite_values(events, "alignment_correction_rotation_deg", applied_only=True)
        ),
        "applied_prediction_jump_translation_m": _distribution(
            _finite_values(events, "prediction_jump_translation_m", applied_only=True)
        ),
        "attempt_runtime_ms": _distribution(_finite_values(events, "runtime_ms")),
    }


def write_csv(events, path):
    if not events:
        Path(path).write_text("stamp_ns,elapsed_s,status,applied\n", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]))
        writer.writeheader()
        writer.writerows(events)


def plot_events(events, path, dataset):
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    figure.suptitle(f"{dataset}: historical mapping-cloud loop refinement")
    if not events:
        for axis in axes.flat:
            axis.axis("off")
        axes[1, 0].text(0.5, 0.5, "No refinement attempts", ha="center", va="center")
        figure.savefig(path, dpi=180)
        plt.close(figure)
        return

    elapsed = np.asarray([event["elapsed_s"] for event in events])
    applied = np.asarray([event["applied"] for event in events])

    def series(name):
        return np.asarray([event[name] for event in events], dtype=float)

    axes[0, 0].plot(elapsed, series("before_score"), "o-", label="before")
    axes[0, 0].plot(elapsed, series("after_score"), "o-", label="after")
    axes[0, 0].set_ylabel("Obstacle quality cost (lower is better)")
    axes[0, 0].legend()

    axes[0, 1].plot(elapsed, series("before_inlier_rmse_m"), "o-", label="before")
    axes[0, 1].plot(elapsed, series("after_inlier_rmse_m"), "o-", label="after")
    axes[0, 1].set_ylabel("Obstacle inlier RMSE [m]")
    axes[0, 1].legend()
    coverage_axis = axes[0, 1].twinx()
    coverage_axis.plot(
        elapsed,
        series("after_target_coverage_ratio"),
        "s--",
        color="tab:green",
        alpha=0.65,
        label="target coverage",
    )
    coverage_axis.set_ylabel("Confirmed target coverage ratio")

    axes[1, 0].plot(
        elapsed,
        series("correction_translation_m"),
        "o-",
        label="local refinement [m]",
    )
    axes[1, 0].plot(
        elapsed,
        series("map_correction_translation_m"),
        "s-",
        label="total map correction [m]",
    )
    alignment_limit = series("mapping_refinement_alignment_translation_limit_m")
    finite_limit = np.isfinite(alignment_limit) & (alignment_limit > 0.0)
    axes[1, 0].plot(
        elapsed[finite_limit],
        alignment_limit[finite_limit],
        "k--",
        label="conditional alignment gate [m]",
    )
    axes[1, 0].set_ylabel("Translation correction [m]")
    axes[1, 0].legend()

    axes[1, 1].plot(elapsed, series("observable_rank"), "o-", color="tab:green")
    axes[1, 1].set_ylabel("Observable rank")
    condition_axis = axes[1, 1].twinx()
    condition = series("hessian_condition")
    valid_condition = np.isfinite(condition) & (condition > 0.0)
    condition_axis.plot(
        elapsed[valid_condition], condition[valid_condition], "s--", color="tab:red"
    )
    condition_axis.set_yscale("log")
    condition_axis.set_ylabel("Hessian condition (log)")

    axes[2, 0].plot(elapsed, series("primary_before_score"), "o-", label="primary before")
    axes[2, 0].plot(elapsed, series("primary_after_score"), "o-", label="primary after")
    axes[2, 0].set_ylabel("Ground+obstacle quality cost (lower is better)")
    axes[2, 0].legend()

    axes[2, 1].plot(elapsed, series("pose_graph_residual_before"), "o-", label="graph before")
    axes[2, 1].plot(elapsed, series("pose_graph_residual_after"), "o-", label="graph after")
    axes[2, 1].set_ylabel("Pose graph residual")
    axes[2, 1].legend()

    for axis in axes.flat:
        axis.set_xlabel("Bag elapsed time [s]")
        axis.grid(True, alpha=0.25)
        for x in elapsed[applied]:
            axis.axvline(x, color="tab:green", alpha=0.12)

    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("health_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(args.health_jsonl)
    summary = summarize(events)
    summary["dataset"] = args.dataset
    summary["input_health_jsonl"] = str(args.health_jsonl)

    write_csv(events, args.output_dir / "mapping_cloud_refinement_events.csv")
    (args.output_dir / "mapping_cloud_refinement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_events(
        events,
        args.output_dir / "mapping_cloud_refinement_diagnostics.png",
        args.dataset,
    )
    plot_events(
        events,
        args.output_dir / "mapping_cloud_refinement_diagnostics.svg",
        args.dataset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
