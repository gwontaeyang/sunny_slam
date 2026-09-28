#!/usr/bin/env python3
"""Compare roughness-place pose graphs with point-only map overlays."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image


MAX_RENDERED_TRAJECTORY_POINTS = 180


@dataclass(frozen=True)
class Variant:
    label: str
    root: Path
    poses: dict[int, dict[str, float]]
    map_image: np.ndarray
    extent: tuple[float, float, float, float]


def parse_variant(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be LABEL=BAG_OUTPUT_DIR")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("variant must be LABEL=BAG_OUTPUT_DIR")
    return label, Path(raw_path).expanduser().resolve()


def parse_pair(value: str) -> tuple[int, int]:
    try:
        left, right = value.split(":", 1)
        pair = int(left), int(right)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("state pair must be SOURCE:TARGET") from exc
    if pair[0] < 0 or pair[1] < 0 or pair[0] == pair[1]:
        raise argparse.ArgumentTypeError("state IDs must be distinct and non-negative")
    return pair


def load_poses(path: Path) -> dict[int, dict[str, float]]:
    poses: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            state_id = int(row["state_id"])
            pose = {
                "stamp_s": float(row["stamp_s"]),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "yaw_rad": float(row["yaw_rad"]),
            }
            if state_id in poses:
                raise ValueError(f"duplicate state_id {state_id} in {path}")
            if not all(math.isfinite(item) for item in pose.values()):
                raise ValueError(f"non-finite state_id {state_id} in {path}")
            poses[state_id] = pose
    if not poses:
        raise ValueError(f"no poses in {path}")
    return poses


def load_map(root: Path) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with (root / "map.yaml").open(encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    image = np.asarray(Image.open(root / metadata["image"]).convert("L"))
    resolution_m = float(metadata["resolution"])
    origin = metadata["origin"]
    origin_x_m = float(origin[0])
    origin_y_m = float(origin[1])
    origin_yaw_rad = float(origin[2])
    if abs(origin_yaw_rad) > 1.0e-9:
        raise ValueError("rotated map origins are not supported")
    height, width = image.shape
    extent = (
        origin_x_m,
        origin_x_m + width * resolution_m,
        origin_y_m,
        origin_y_m + height * resolution_m,
    )
    return image, extent


def load_variant(label: str, root: Path) -> Variant:
    poses = load_poses(root / "trajectory_pose_graph_final.csv")
    image, extent = load_map(root)
    return Variant(label, root, poses, image, extent)


def pair_metrics(variant: Variant, pair: tuple[int, int]) -> dict[str, Any]:
    source_id, target_id = pair
    if source_id not in variant.poses or target_id not in variant.poses:
        raise ValueError(
            f"{variant.label} is missing state pair {source_id}:{target_id}"
        )
    source = variant.poses[source_id]
    target = variant.poses[target_id]
    translation_m = math.hypot(
        source["x_m"] - target["x_m"],
        source["y_m"] - target["y_m"],
    )
    yaw_rad = abs(math.remainder(
        source["yaw_rad"] - target["yaw_rad"], 2.0 * math.pi
    ))
    return {
        "variant": variant.label,
        "source_state_id": source_id,
        "target_state_id": target_id,
        "translation_gap_m": translation_m,
        "yaw_gap_deg": math.degrees(yaw_rad),
    }


def variant_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == label]
    translation = [float(row["translation_gap_m"]) for row in selected]
    yaw = [float(row["yaw_gap_deg"]) for row in selected]
    return {
        "variant": label,
        "pair_count": len(selected),
        "translation_gap_mean_m": sum(translation) / len(translation),
        "translation_gap_median_m": float(np.median(translation)),
        "translation_gap_max_m": max(translation),
        "yaw_gap_mean_deg": sum(yaw) / len(yaw),
        "yaw_gap_median_deg": float(np.median(yaw)),
        "yaw_gap_max_deg": max(yaw),
    }


def trajectory_render_indices(state_count: int) -> tuple[np.ndarray, int]:
    """Return an evenly strided point set that remains visually discrete."""
    if state_count <= 0:
        raise ValueError("state_count must be positive")
    stride = max(1, math.ceil(state_count / MAX_RENDERED_TRAJECTORY_POINTS))
    indices = np.arange(0, state_count, stride, dtype=int)
    if indices[-1] != state_count - 1:
        indices = np.append(indices, state_count - 1)
    return indices, stride


def draw_variant(
    axis: Any,
    variant: Variant,
    pairs: list[tuple[int, int]],
    common_bounds: tuple[float, float, float, float],
) -> None:
    axis.imshow(
        variant.map_image,
        cmap="gray",
        vmin=0,
        vmax=255,
        origin="upper",
        extent=variant.extent,
    )
    state_ids = sorted(variant.poses)
    trajectory = np.asarray([
        [variant.poses[state_id]["x_m"], variant.poses[state_id]["y_m"]]
        for state_id in state_ids
    ])
    render_indices, render_stride = trajectory_render_indices(len(trajectory))
    rendered_trajectory = trajectory[render_indices]
    axis.scatter(
        rendered_trajectory[:, 0],
        rendered_trajectory[:, 1],
        s=10,
        facecolors="none",
        edgecolors="#0072B2",
        alpha=0.9,
        linewidths=0.65,
        label=f"pose-graph states (points, 1/{render_stride})",
    )
    source_ids = sorted({pair[0] for pair in pairs})
    target_ids = sorted({pair[1] for pair in pairs})
    source = np.asarray([
        [variant.poses[state_id]["x_m"], variant.poses[state_id]["y_m"]]
        for state_id in source_ids
    ])
    target = np.asarray([
        [variant.poses[state_id]["x_m"], variant.poses[state_id]["y_m"]]
        for state_id in target_ids
    ])
    axis.scatter(
        source[:, 0], source[:, 1], s=38, marker="o", color="#009E73",
        edgecolors="white", linewidths=0.6, label="outbound anchors", zorder=4,
    )
    axis.scatter(
        target[:, 0], target[:, 1], s=42, marker="x", color="#D55E00",
        linewidths=1.4, label="return anchors", zorder=5,
    )
    axis.set_xlim(common_bounds[0], common_bounds[1])
    axis.set_ylim(common_bounds[2], common_bounds[3])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.set_title(variant.label)
    axis.grid(alpha=0.18)


def plot_comparison(
    variants: list[Variant],
    pairs: list[tuple[int, int]],
    summaries: list[dict[str, Any]],
    output: Path,
) -> None:
    x_min = min(variant.extent[0] for variant in variants)
    x_max = max(variant.extent[1] for variant in variants)
    y_min = min(variant.extent[2] for variant in variants)
    y_max = max(variant.extent[3] for variant in variants)
    common_bounds = (x_min, x_max, y_min, y_max)
    figure, axes = plt.subplots(
        1, len(variants), figsize=(6.4 * len(variants), 7.2), squeeze=False
    )
    for axis, variant, summary in zip(axes[0], variants, summaries):
        draw_variant(axis, variant, pairs, common_bounds)
        axis.text(
            0.02,
            0.02,
            "mean anchor gap\n"
            f"{summary['translation_gap_mean_m']:.3f} m / "
            f"{summary['yaw_gap_mean_deg']:.2f} deg",
            transform=axis.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "0.6"},
        )
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.suptitle(
        "Roughness-place return closure (point-only trajectories)", y=0.995
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=True,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", type=parse_variant, required=True,
        help="repeat LABEL=BAG_OUTPUT_DIR",
    )
    parser.add_argument(
        "--state-pair", action="append", type=parse_pair, required=True,
        help="repeat SOURCE:TARGET",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = [load_variant(label, path) for label, path in args.variant]
    labels = [variant.label for variant in variants]
    if len(set(labels)) != len(labels):
        raise ValueError("variant labels must be unique")
    rows = [
        pair_metrics(variant, pair)
        for variant in variants
        for pair in args.state_pair
    ]
    summaries = [variant_summary(rows, variant.label) for variant in variants]

    with (args.output_dir / "return_anchor_gaps.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "trajectory_rendering": "points_only",
        "state_pairs": [list(pair) for pair in args.state_pair],
        "variants": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    plot_comparison(
        variants,
        args.state_pair,
        summaries,
        args.output_dir / "map_pose_graph_points_comparison.png",
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
