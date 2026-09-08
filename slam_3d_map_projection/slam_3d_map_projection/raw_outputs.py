# Copyright 2026 Gole Robotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Artifacts that keep the 3D evidence map and planning grid together."""

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from .configuration import ProjectionConfig  # noqa: E402
from .outputs import write_trinary_map  # noqa: E402
from .path_planning import PathPlan, PathPlanningConfig  # noqa: E402
from .pcd_io import write_xyzrgb_pcd  # noqa: E402
from .projection import GridClass, PointClass, ProjectionResult  # noqa: E402
from .visibility import (  # noqa: E402
    EvidenceDecision,
    ObstacleCandidates,
    RawVisibilityConfig,
    VisibilityEvidence,
    VisibilityRefinement,
)


_BASE_COLORS = {
    PointClass.FLOOR: np.asarray([0, 158, 115], dtype=np.uint8),
    PointClass.BODY_OBSTACLE: np.asarray([117, 112, 179], dtype=np.uint8),
    PointClass.HIGH_OVERHANG: np.asarray([0, 114, 178], dtype=np.uint8),
    PointClass.BELOW_FLOOR: np.asarray([110, 110, 110], dtype=np.uint8),
    PointClass.UNREFERENCED: np.asarray([204, 121, 167], dtype=np.uint8),
}
_EVIDENCE_COLORS = {
    EvidenceDecision.CONFIRMED: np.asarray([213, 94, 0], dtype=np.uint8),
    EvidenceDecision.UNCERTAIN: np.asarray([230, 159, 0], dtype=np.uint8),
    EvidenceDecision.CONTRADICTED: np.asarray([0, 158, 180], dtype=np.uint8),
}


def visibility_point_colors(
    point_classes: np.ndarray,
    point_decisions: np.ndarray,
) -> np.ndarray:
    """Color 3D map points by geometry class and raw visibility decision."""
    colors = np.zeros((point_classes.shape[0], 3), dtype=np.uint8)
    for label, color in _BASE_COLORS.items():
        colors[point_classes == label] = color
    for decision, color in _EVIDENCE_COLORS.items():
        colors[point_decisions == decision] = color
    return colors


def _equal_xy_limits(axis, points: np.ndarray) -> None:
    low = np.quantile(points[:, :2], 0.005, axis=0)
    high = np.quantile(points[:, :2], 0.995, axis=0)
    center = 0.5 * (low + high)
    radius = max(0.5 * float(np.max(high - low)), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)


def create_combined_review(
    path: Path,
    dataset_label: str,
    points: np.ndarray,
    colors: np.ndarray,
    trajectory: np.ndarray,
    refinement: VisibilityRefinement,
    projection: ProjectionResult,
    plan: PathPlan,
) -> None:
    """Render 3D evidence views and the path-generation grid in one figure."""
    max_points = 60_000
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    else:
        indices = np.arange(points.shape[0])
    shown_points = points[indices]
    shown_colors = colors[indices].astype(np.float64) / 255.0
    z_low, z_high = np.quantile(shown_points[:, 2], [0.005, 0.995])
    plan_xy = plan.poses_xy_yaw[:, :2]

    figure = plt.figure(figsize=(19, 14))
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.86,
        hspace=0.24,
        wspace=0.16,
    )
    perspective = figure.add_subplot(2, 2, 1, projection="3d")
    perspective.scatter(
        shown_points[:, 0],
        shown_points[:, 1],
        shown_points[:, 2],
        c=shown_colors,
        s=0.35,
        linewidths=0,
        alpha=0.8,
    )
    perspective.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="#00A6D6",
        linewidth=1.2,
        label="optimized trajectory",
    )
    perspective.set_title("3D visibility-classified map")
    perspective.set_xlabel("map x [m]")
    perspective.set_ylabel("map y [m]")
    perspective.set_zlabel("map z [m]")
    perspective.set_zlim(z_low, z_high)
    perspective.view_init(elev=35.0, azim=-55.0)

    top = figure.add_subplot(2, 2, 2)
    top.scatter(
        shown_points[:, 0],
        shown_points[:, 1],
        c=shown_colors,
        s=0.45,
        linewidths=0,
        alpha=0.8,
    )
    top.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#00A6D6",
        linewidth=1.0,
        alpha=0.8,
    )
    top.plot(plan_xy[:, 0], plan_xy[:, 1], color="#CC0077", linewidth=2.2)
    top.scatter(
        [plan_xy[0, 0], plan_xy[-1, 0]],
        [plan_xy[0, 1], plan_xy[-1, 1]],
        c=["#009E73", "#D55E00"],
        s=45,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    top.set_title("3D map top view with generated path")
    top.set_xlabel("map x [m]")
    top.set_ylabel("map y [m]")
    top.set_aspect("equal", adjustable="box")
    top.set_xlim(projection.geometry.origin_x_m, projection.geometry.max_x_m)
    top.set_ylim(projection.geometry.origin_y_m, projection.geometry.max_y_m)
    top.grid(linewidth=0.3, alpha=0.4)

    side = figure.add_subplot(2, 2, 3)
    side.scatter(
        shown_points[:, 0],
        shown_points[:, 2],
        c=shown_colors,
        s=0.45,
        linewidths=0,
        alpha=0.8,
    )
    side.plot(
        trajectory[:, 0],
        trajectory[:, 2],
        color="#00A6D6",
        linewidth=1.0,
    )
    side.set_title("3D map front projection")
    side.set_xlabel("map x [m]")
    side.set_ylabel("map z [m]")
    side.set_ylim(z_low, z_high)
    side.grid(linewidth=0.3, alpha=0.4)

    grid = figure.add_subplot(2, 2, 4)
    display = np.zeros(refinement.occupancy.shape, dtype=np.uint8)
    display[refinement.occupancy == GridClass.UNKNOWN] = 0
    display[refinement.occupancy == GridClass.FREE] = 1
    display[refinement.occupancy == GridClass.OCCUPIED] = 2
    geometry = projection.geometry
    extent = [
        geometry.origin_x_m,
        geometry.max_x_m,
        geometry.origin_y_m,
        geometry.max_y_m,
    ]
    grid.imshow(
        display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#8A8A8A", "#FFFFFF", "#111111"]),
        vmin=0,
        vmax=2,
    )
    contradicted_rows, contradicted_columns = np.nonzero(
        refinement.contradicted_endpoint_mask
    )
    if contradicted_rows.size:
        grid.scatter(
            geometry.origin_x_m
            + (contradicted_columns + 0.5) * geometry.resolution_m,
            geometry.origin_y_m
            + (contradicted_rows + 0.5) * geometry.resolution_m,
            color="#00BFC4",
            s=2,
            linewidths=0,
            label="raw miss contradicted",
        )
    grid.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#00A6D6",
        linewidth=1.0,
        alpha=0.8,
    )
    lattice_invalid = ~plan.trajectory_lattice_valid
    if np.any(lattice_invalid):
        grid.scatter(
            trajectory[lattice_invalid, 0],
            trajectory[lattice_invalid, 1],
            color="#D55E00",
            marker="x",
            s=18,
            linewidths=0.8,
            label="recorded pose invalid in lattice",
            zorder=4,
        )
    grid.plot(plan_xy[:, 0], plan_xy[:, 1], color="#CC0077", linewidth=2.2)
    grid.scatter(
        [plan_xy[0, 0], plan_xy[-1, 0]],
        [plan_xy[0, 1], plan_xy[-1, 1]],
        c=["#009E73", "#D55E00"],
        s=45,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    grid.set_title(
        "Path grid; UNKNOWN blocked; swept/exact/lattice collisions="
        f"{plan.metrics['path_exact_swept_collision_sample_count']}/"
        f"{plan.metrics['path_exact_footprint_collision_pose_count']}/"
        f"{plan.metrics['path_collision_pose_count']}\n"
        "trajectory exact/lattice valid="
        f"{100.0 * plan.metrics['trajectory_exact_footprint_valid_fraction']:.1f}%/"
        f"{100.0 * plan.metrics['trajectory_lattice_footprint_valid_fraction']:.1f}%"
    )
    grid.set_xlabel("map x [m]")
    grid.set_ylabel("map y [m]")
    grid.set_aspect("equal", adjustable="box")
    grid.grid(linewidth=0.25, alpha=0.35)

    legend = [
        Patch(color="#009E73", label="floor"),
        Patch(color="#D55E00", label="confirmed obstacle"),
        Patch(color="#E69F00", label="uncertain obstacle"),
        Patch(color="#00BFC4", label="miss-contradicted obstacle"),
        Patch(color="#0072B2", label="high overhang"),
        Patch(color="#8A8A8A", label="unknown / below-floor"),
        Line2D([0], [0], color="#00A6D6", label="optimized trajectory"),
        Line2D([0], [0], color="#CC0077", linewidth=2, label="generated path"),
        Line2D(
            [0],
            [0],
            color="#D55E00",
            marker="x",
            linestyle="none",
            label="recorded pose invalid in lattice",
        ),
    ]
    figure.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=True,
    )
    figure.suptitle(
        f"{dataset_label}: raw-scan visibility 3D map and footprint-aware grid path\n"
        "Hit/miss evidence is height-banded; empty or occluded beams are not free",
        fontsize=16,
        y=0.985,
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_raw_visibility_outputs(
    output_dir: Path,
    map_name: str,
    dataset_label: str,
    points: np.ndarray,
    trajectory: np.ndarray,
    projection: ProjectionResult,
    candidates: ObstacleCandidates,
    evidence: VisibilityEvidence,
    refinement: VisibilityRefinement,
    plan: PathPlan,
    projection_config: ProjectionConfig,
    visibility_config: RawVisibilityConfig,
    planning_config: PathPlanningConfig,
) -> Dict[str, Path]:
    """Write a paired 3D PCD, Nav2 grid, generated path, and review evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    map_paths = write_trinary_map(
        output_dir,
        map_name,
        refinement.occupancy,
        projection.geometry,
        projection_config,
    )
    colors = visibility_point_colors(
        projection.point_classes,
        refinement.point_decisions,
    )
    pcd_path = output_dir / f"{map_name}_visibility_rgb.pcd"
    write_xyzrgb_pcd(pcd_path, points, colors)

    layers_path = output_dir / f"{map_name}_visibility_layers.npz"
    np.savez_compressed(
        layers_path,
        occupancy=refinement.occupancy,
        retained_endpoint_mask=refinement.retained_endpoint_mask,
        bridge_mask=refinement.bridge_mask,
        contradicted_endpoint_mask=refinement.contradicted_endpoint_mask,
        confirmed_endpoint_mask=refinement.confirmed_endpoint_mask,
        uncertain_endpoint_mask=refinement.uncertain_endpoint_mask,
        recovered_free_mask=refinement.recovered_free_mask,
        candidate_points_map=candidates.points_map,
        candidate_band_indices=candidates.band_indices,
        candidate_coarse_rows=candidates.coarse_rows,
        candidate_coarse_columns=candidates.coarse_columns,
        candidate_decisions=evidence.decisions,
        candidate_hit_scan_count=evidence.hit_scan_count,
        candidate_miss_scan_count=evidence.miss_scan_count,
        candidate_occluded_scan_count=evidence.occluded_scan_count,
        candidate_ambiguous_scan_count=evidence.ambiguous_scan_count,
        candidate_unobserved_scan_count=evidence.unobserved_scan_count,
        evidence_origin_xy_m=candidates.coarse_origin_xy_m,
        evidence_resolution_m=np.asarray(candidates.coarse_resolution_m),
        map_origin_xy_m=np.asarray(
            [projection.geometry.origin_x_m, projection.geometry.origin_y_m]
        ),
        map_resolution_m=np.asarray(projection.geometry.resolution_m),
        path_states=plan.states,
        path_poses_xy_yaw=plan.poses_xy_yaw,
        path_exact_valid=plan.path_exact_valid,
        path_swept_poses_xy_yaw=plan.path_swept_poses_xy_yaw,
        path_swept_exact_valid=plan.path_swept_exact_valid,
        trajectory_xyz=trajectory[:, :3],
        trajectory_yaw_rad=projection.trajectory_yaw_rad,
        trajectory_exact_footprint_valid=plan.trajectory_exact_valid,
        trajectory_lattice_footprint_valid=plan.trajectory_lattice_valid,
    )

    path_csv = output_dir / f"{map_name}_planned_path.csv"
    with path_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pose_index", "x_m", "y_m", "yaw_rad"])
        for index, pose in enumerate(plan.poses_xy_yaw):
            writer.writerow([index, *pose.tolist()])

    review_path = output_dir / f"{map_name}_3d_grid_path_review.png"
    create_combined_review(
        review_path,
        dataset_label,
        points,
        colors,
        trajectory,
        refinement,
        projection,
        plan,
    )
    metrics = {
        "dataset": dataset_label,
        "projection": projection.metrics,
        "raw_visibility": evidence.scan_metrics,
        "visibility_refinement": refinement.metrics,
        "path_planning": plan.metrics,
        "parameters": {
            "projection": projection_config.to_dict(),
            "raw_visibility": asdict(visibility_config),
            "path_planning": asdict(planning_config),
        },
        "outputs": {
            "map_image": map_paths["map_image"].name,
            "map_yaml": map_paths["map_yaml"].name,
            "visibility_pcd": pcd_path.name,
            "visibility_layers": layers_path.name,
            "planned_path": path_csv.name,
            "combined_review": review_path.name,
        },
    }
    metrics_path = output_dir / f"{map_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return {
        **map_paths,
        "visibility_pcd": pcd_path,
        "visibility_layers": layers_path,
        "planned_path": path_csv,
        "combined_review": review_path,
        "metrics": metrics_path,
    }
