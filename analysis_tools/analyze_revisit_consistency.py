#!/usr/bin/env python3
"""Measure vertical and attitude consistency at long-separated XY revisits."""

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "none"


POSE_FIELDS = (
    "x_m",
    "y_m",
    "z_m",
    "qx",
    "qy",
    "qz",
    "qw",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
)


def load_trajectory(path):
    poses = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                pose = {name: float(row[name]) for name in POSE_FIELDS}
                pose["stamp_ns"] = int(row["stamp_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(pose[name]) for name in POSE_FIELDS):
                poses.append(pose)
    poses.sort(key=lambda pose: pose["stamp_ns"])
    if not poses:
        return poses

    poses[0]["elapsed_s"] = 0.0
    poses[0]["cumulative_distance_m"] = 0.0
    first_stamp_ns = poses[0]["stamp_ns"]
    for index in range(1, len(poses)):
        previous = poses[index - 1]
        pose = poses[index]
        pose["elapsed_s"] = (pose["stamp_ns"] - first_stamp_ns) * 1.0e-9
        step = math.hypot(pose["x_m"] - previous["x_m"], pose["y_m"] - previous["y_m"])
        pose["cumulative_distance_m"] = previous["cumulative_distance_m"] + step
    return poses


def wrapped_angle_difference_deg(lhs, rhs):
    return (lhs - rhs + 180.0) % 360.0 - 180.0


def body_up_vector(pose):
    quaternion = np.asarray(
        [pose["qx"], pose["qy"], pose["qz"], pose["qw"]], dtype=float
    )
    norm = np.linalg.norm(quaternion)
    if not math.isfinite(norm) or norm <= 1.0e-12:
        return np.asarray([math.nan, math.nan, math.nan])
    x, y, z, w = quaternion / norm
    return np.asarray(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]
    )


def body_up_difference_deg(lhs, rhs):
    dot = float(np.dot(body_up_vector(lhs), body_up_vector(rhs)))
    if not math.isfinite(dot):
        return math.nan
    return math.degrees(math.acos(np.clip(dot, -1.0, 1.0)))


def pair_metrics(poses, current_index, historical_index):
    current = poses[current_index]
    historical = poses[historical_index]
    return {
        "current_index": current_index,
        "historical_index": historical_index,
        "current_stamp_ns": current["stamp_ns"],
        "historical_stamp_ns": historical["stamp_ns"],
        "current_elapsed_s": current["elapsed_s"],
        "historical_elapsed_s": historical["elapsed_s"],
        "time_separation_s": current["elapsed_s"] - historical["elapsed_s"],
        "path_separation_m": current["cumulative_distance_m"]
        - historical["cumulative_distance_m"],
        "xy_distance_m": math.hypot(
            current["x_m"] - historical["x_m"],
            current["y_m"] - historical["y_m"],
        ),
        "z_difference_m": current["z_m"] - historical["z_m"],
        "abs_z_difference_m": abs(current["z_m"] - historical["z_m"]),
        "roll_difference_deg": wrapped_angle_difference_deg(
            current["roll_deg"], historical["roll_deg"]
        ),
        "pitch_difference_deg": wrapped_angle_difference_deg(
            current["pitch_deg"], historical["pitch_deg"]
        ),
        "body_up_difference_deg": body_up_difference_deg(current, historical),
        "yaw_difference_deg": wrapped_angle_difference_deg(
            current["yaw_deg"], historical["yaw_deg"]
        ),
    }


def nearest_revisit_pairs(
    poses,
    min_time_separation_s=300.0,
    min_path_separation_m=20.0,
):
    pairs = []
    for current_index, current in enumerate(poses):
        eligible_indices = [
            historical_index
            for historical_index in range(current_index)
            if current["elapsed_s"] - poses[historical_index]["elapsed_s"]
            >= min_time_separation_s
            and current["cumulative_distance_m"]
            - poses[historical_index]["cumulative_distance_m"]
            >= min_path_separation_m
        ]
        if not eligible_indices:
            continue
        historical_index = min(
            eligible_indices,
            key=lambda index: math.hypot(
                current["x_m"] - poses[index]["x_m"],
                current["y_m"] - poses[index]["y_m"],
            ),
        )
        pairs.append(pair_metrics(poses, current_index, historical_index))
    return pairs


def pair_manifest_from_pairs(pairs):
    return [
        {
            "current_stamp_ns": pair["current_stamp_ns"],
            "historical_stamp_ns": pair["historical_stamp_ns"],
        }
        for pair in pairs
    ]


def load_pair_manifest(path):
    pairs = []
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                current_stamp_ns = int(row["current_stamp_ns"])
                historical_stamp_ns = int(row["historical_stamp_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if current_stamp_ns > historical_stamp_ns:
                pairs.append(
                    {
                        "current_stamp_ns": current_stamp_ns,
                        "historical_stamp_ns": historical_stamp_ns,
                    }
                )
    return pairs


def nearest_pose_index(poses, stamp_ns):
    if not poses:
        return None
    stamps = [pose["stamp_ns"] for pose in poses]
    right = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if right < len(stamps):
        candidates.append(right)
    if right > 0:
        candidates.append(right - 1)
    return min(candidates, key=lambda index: abs(stamps[index] - stamp_ns))


def evaluate_pair_manifest(poses, manifest, max_timestamp_delta_s=0.2):
    evaluated = []
    max_delta_ns = max(0.0, max_timestamp_delta_s) * 1.0e9
    for requested in manifest:
        current_index = nearest_pose_index(poses, requested["current_stamp_ns"])
        historical_index = nearest_pose_index(poses, requested["historical_stamp_ns"])
        if current_index is None or historical_index is None:
            continue
        current_delta_ns = poses[current_index]["stamp_ns"] - requested["current_stamp_ns"]
        historical_delta_ns = (
            poses[historical_index]["stamp_ns"] - requested["historical_stamp_ns"]
        )
        if (
            abs(current_delta_ns) > max_delta_ns
            or abs(historical_delta_ns) > max_delta_ns
            or current_index <= historical_index
        ):
            continue
        metrics = pair_metrics(poses, current_index, historical_index)
        metrics.update(
            {
                "requested_current_stamp_ns": requested["current_stamp_ns"],
                "requested_historical_stamp_ns": requested["historical_stamp_ns"],
                "current_stamp_offset_s": current_delta_ns * 1.0e-9,
                "historical_stamp_offset_s": historical_delta_ns * 1.0e-9,
            }
        )
        evaluated.append(metrics)
    return evaluated


def distribution(values):
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def summarize(poses, pairs, xy_gates_m):
    summary = {
        "pose_count": len(poses),
        "nearest_historical_pair_count": len(pairs),
        "xy_gate_metrics": {},
    }
    if poses:
        summary.update(
            {
                "path_length_xy_m": poses[-1]["cumulative_distance_m"],
                "endpoint_displacement_xy_m": math.hypot(
                    poses[-1]["x_m"] - poses[0]["x_m"],
                    poses[-1]["y_m"] - poses[0]["y_m"],
                ),
                "endpoint_z_difference_m": poses[-1]["z_m"] - poses[0]["z_m"],
                "trajectory_z_range_m": max(pose["z_m"] for pose in poses)
                - min(pose["z_m"] for pose in poses),
            }
        )
    for gate in xy_gates_m:
        selected = [pair for pair in pairs if pair["xy_distance_m"] <= gate]
        summary["xy_gate_metrics"][f"{gate:g}"] = {
            "sample_count": len(selected),
            "abs_z_difference_m": distribution(
                pair["abs_z_difference_m"] for pair in selected
            ),
            "abs_roll_difference_deg": distribution(
                abs(pair["roll_difference_deg"]) for pair in selected
            ),
            "abs_pitch_difference_deg": distribution(
                abs(pair["pitch_difference_deg"]) for pair in selected
            ),
            "body_up_difference_deg": distribution(
                pair["body_up_difference_deg"] for pair in selected
            ),
            "abs_yaw_difference_deg": distribution(
                abs(pair["yaw_difference_deg"]) for pair in selected
            ),
        }
    return summary


def write_pairs(pairs, path):
    if not pairs:
        Path(path).write_text("current_index,historical_index\n", encoding="utf-8")
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)


def write_pair_manifest(pairs, path):
    fields = ("current_stamp_ns", "historical_stamp_ns")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)


def plot_consistency(poses, pairs, path, dataset, baseline_poses=None):
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    figure.suptitle(f"{dataset}: long-separated revisit consistency")

    def values(items, name):
        return np.asarray([item[name] for item in items], dtype=float)

    def sampled(items, maximum):
        if not items or len(items) <= maximum:
            return items
        indices = np.linspace(0, len(items) - 1, maximum, dtype=int)
        return [items[index] for index in indices]

    display_poses = sampled(poses, 500)
    display_baseline = sampled(baseline_poses, 500) if baseline_poses else None
    if display_poses:
        axes[0, 0].plot(
            values(display_poses, "x_m"),
            values(display_poses, "y_m"),
            label="candidate",
        )
        axes[0, 1].plot(
            values(display_poses, "elapsed_s"),
            values(display_poses, "z_m"),
            label="candidate",
        )
    if display_baseline:
        axes[0, 0].plot(
            values(display_baseline, "x_m"),
            values(display_baseline, "y_m"),
            alpha=0.7,
            label="baseline",
        )
        axes[0, 1].plot(
            values(display_baseline, "elapsed_s"),
            values(display_baseline, "z_m"),
            alpha=0.7,
            label="baseline",
        )
    axes[0, 0].set_xlabel("Map X [m]")
    axes[0, 0].set_ylabel("Map Y [m]")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend()
    axes[0, 1].set_xlabel("Bag elapsed time [s]")
    axes[0, 1].set_ylabel("Map Z [m]")
    axes[0, 1].legend()

    if pairs:
        display_pairs = sampled(pairs, 400)
        axes[1, 0].scatter(
            values(display_pairs, "xy_distance_m"),
            values(display_pairs, "abs_z_difference_m"),
            s=10,
            alpha=0.55,
        )
        close_pairs = [pair for pair in pairs if pair["xy_distance_m"] <= 1.0]
        axes[1, 1].hist(
            values(close_pairs, "abs_z_difference_m") if close_pairs else [],
            bins=24,
            color="tab:blue",
            alpha=0.8,
        )
    axes[1, 0].axvline(0.5, color="tab:red", linestyle="--", label="0.5 m XY gate")
    axes[1, 0].set_xlabel("Nearest historical XY distance [m]")
    axes[1, 0].set_ylabel("Absolute Z difference [m]")
    axes[1, 0].legend()
    axes[1, 1].set_xlabel("Absolute Z difference [m], XY <= 1.0 m")
    axes[1, 1].set_ylabel("Sample count")

    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--baseline-trajectory", type=Path)
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--pair-time-tolerance-s", type=float, default=0.2)
    parser.add_argument("--min-time-separation-s", type=float, default=300.0)
    parser.add_argument("--min-path-separation-m", type=float, default=20.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    poses = load_trajectory(args.trajectory_csv)
    baseline_poses = (
        load_trajectory(args.baseline_trajectory) if args.baseline_trajectory else None
    )
    if args.pair_manifest:
        pair_manifest = load_pair_manifest(args.pair_manifest)
        pair_selection_source = "provided_manifest"
    elif baseline_poses is not None:
        baseline_seed_pairs = nearest_revisit_pairs(
            baseline_poses,
            min_time_separation_s=args.min_time_separation_s,
            min_path_separation_m=args.min_path_separation_m,
        )
        pair_manifest = pair_manifest_from_pairs(baseline_seed_pairs)
        pair_selection_source = "baseline_trajectory"
    else:
        candidate_seed_pairs = nearest_revisit_pairs(
            poses,
            min_time_separation_s=args.min_time_separation_s,
            min_path_separation_m=args.min_path_separation_m,
        )
        pair_manifest = pair_manifest_from_pairs(candidate_seed_pairs)
        pair_selection_source = "candidate_trajectory"
    pairs = evaluate_pair_manifest(
        poses,
        pair_manifest,
        max_timestamp_delta_s=args.pair_time_tolerance_s,
    )
    xy_gates_m = (0.25, 0.5, 1.0, 2.0)
    summary = summarize(poses, pairs, xy_gates_m)
    summary.update(
        {
            "dataset": args.dataset,
            "input_trajectory_csv": str(args.trajectory_csv),
            "min_time_separation_s": args.min_time_separation_s,
            "min_path_separation_m": args.min_path_separation_m,
            "pair_selection_source": pair_selection_source,
            "pair_manifest_requested_count": len(pair_manifest),
            "pair_manifest_matched_count": len(pairs),
            "pair_time_tolerance_s": args.pair_time_tolerance_s,
            "note": "Gate counts are correlated trajectory samples, not independent loops.",
            "attitude_note": (
                "body_up_difference_deg is heading-invariant; raw roll/pitch/yaw "
                "differences remain diagnostic only."
            ),
        }
    )
    baseline_pairs = None
    if baseline_poses is not None:
        baseline_pairs = evaluate_pair_manifest(
            baseline_poses,
            pair_manifest,
            max_timestamp_delta_s=args.pair_time_tolerance_s,
        )
        summary["baseline"] = summarize(baseline_poses, baseline_pairs, xy_gates_m)
        summary["baseline_pair_manifest_matched_count"] = len(baseline_pairs)
        summary["baseline_input_trajectory_csv"] = str(args.baseline_trajectory)

    write_pair_manifest(pair_manifest, args.output_dir / "revisit_pair_manifest.csv")
    write_pairs(pairs, args.output_dir / "revisit_pairs.csv")
    if baseline_pairs is not None:
        write_pairs(baseline_pairs, args.output_dir / "baseline_revisit_pairs.csv")
    (args.output_dir / "revisit_consistency_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_consistency(
        poses,
        pairs,
        args.output_dir / "revisit_consistency.png",
        args.dataset,
        baseline_poses,
    )
    plot_consistency(
        poses,
        pairs,
        args.output_dir / "revisit_consistency.svg",
        args.dataset,
        baseline_poses,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
