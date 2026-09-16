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

"""Graphical review output for projected indoor navigation maps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from .outputs import point_class_colors  # noqa: E402
from .path_planning import PathPlan  # noqa: E402
from .projection import GridClass, ProjectionResult  # noqa: E402


def _plot_trajectory(axis, trajectory, result, **kwargs):
    """Plot only trajectory segments accepted for swept-footprint interpolation."""
    section_start = 0
    section_ends = list(np.flatnonzero(~result.trajectory_segment_valid) + 1)
    section_ends.append(trajectory.shape[0])
    for section_end in section_ends:
        section = trajectory[section_start:section_end]
        if section.shape[0] == 1:
            axis.scatter(section[:, 0], section[:, 1], s=4.0, **kwargs)
        elif section.shape[0] > 1:
            axis.plot(section[:, 0], section[:, 1], **kwargs)
        section_start = section_end


def write_review_figure(
    path: Path,
    dataset_label: str,
    points: np.ndarray,
    trajectory: np.ndarray,
    result: ProjectionResult,
) -> None:
    """Render classifications, final occupancy, terrain slope, and summary counts."""
    geometry = result.geometry
    extent = [
        geometry.origin_x_m,
        geometry.max_x_m,
        geometry.origin_y_m,
        geometry.max_y_m,
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    max_points = 120_000
    stride = max(1, int(np.ceil(points.shape[0] / max_points)))
    selected = np.arange(0, points.shape[0], stride)
    colors = point_class_colors(result.point_classes[selected]).astype(float) / 255.0
    axes[0, 0].scatter(
        points[selected, 0],
        points[selected, 1],
        c=colors,
        s=1.0,
        alpha=0.55,
        rasterized=True,
    )
    _plot_trajectory(
        axes[0, 0], trajectory, result, color="#00A6D6", linewidth=2.0
    )
    axes[0, 0].set_title("3D point classes projected to XY")

    display = np.zeros(result.occupancy.shape, dtype=np.uint8)
    display[result.occupancy == GridClass.UNKNOWN] = 0
    display[result.occupancy == GridClass.FREE] = 1
    display[result.occupancy == GridClass.OCCUPIED] = 2
    occupancy_cmap = ListedColormap(["#8A8A8A", "#FFFFFF", "#1B1B1B"])
    axes[0, 1].imshow(
        display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=occupancy_cmap,
        vmin=0,
        vmax=2,
    )
    completed_free = np.ma.masked_where(
        ~result.enclosed_unknown_filled_mask,
        result.enclosed_unknown_filled_mask,
    )
    axes[0, 1].imshow(
        completed_free,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#56B4E9"]),
        alpha=0.25,
    )
    _plot_trajectory(
        axes[0, 1], trajectory, result, color="#00A6D6", linewidth=1.5
    )
    hazard_rows, hazard_columns = np.nonzero(
        result.terrain_hazard_mask
        & ~result.terrain_hazard_cleared_by_trajectory_mask
    )
    if hazard_rows.size:
        hazard_x = (
            geometry.origin_x_m
            + (hazard_columns + 0.5) * geometry.resolution_m
        )
        hazard_y = (
            geometry.origin_y_m + (hazard_rows + 0.5) * geometry.resolution_m
        )
        axes[0, 1].scatter(
            hazard_x,
            hazard_y,
            color="#CC79A7",
            marker="s",
            s=3.0,
            alpha=0.75,
            rasterized=True,
            label="terrain hazard -> unknown",
        )
    ramp_rows, ramp_columns = np.nonzero(result.ramp_mask)
    if ramp_rows.size:
        ramp_x = geometry.origin_x_m + (ramp_columns + 0.5) * geometry.resolution_m
        ramp_y = geometry.origin_y_m + (ramp_rows + 0.5) * geometry.resolution_m
        axes[0, 1].scatter(
            ramp_x, ramp_y, color="#E69F00", s=2.0, alpha=0.7, rasterized=True
        )
    map_legend = []
    if np.any(result.enclosed_unknown_filled_mask):
        map_legend.append(Patch(color="#56B4E9", label="enclosed floor -> free"))
    if hazard_rows.size:
        map_legend.append(
            Line2D(
                [],
                [],
                color="#CC79A7",
                marker="s",
                linestyle="None",
                label="terrain hazard -> unknown",
            )
        )
    if map_legend:
        axes[0, 1].legend(handles=map_legend, loc="upper right")
    axes[0, 1].set_title("Nav2 trinary grid; ramp cells in orange")

    slope = np.ma.masked_invalid(result.floor_slope_deg)
    finite_slope = result.floor_slope_deg[np.isfinite(result.floor_slope_deg)]
    slope_vmax = (
        max(15.0, float(np.percentile(finite_slope, 95)))
        if finite_slope.size
        else 15.0
    )
    slope_image = axes[1, 0].imshow(
        slope,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=slope_vmax,
    )
    _plot_trajectory(
        axes[1, 0], trajectory, result, color="#D55E00", linewidth=1.2
    )
    axes[1, 0].set_title("Estimated local floor slope [deg]")
    figure.colorbar(slope_image, ax=axes[1, 0], label="slope [deg]")

    evidence_display = np.zeros(result.occupancy.shape, dtype=np.uint8)
    evidence_display[result.obstacle_endpoint_mask] = 1
    evidence_display[result.obstacle_bridge_mask] = 2
    evidence_display[result.obstacle_dilation_mask] = 3
    evidence_display[result.obstacle_cleared_by_trajectory_mask] = 4
    evidence_colors = ["#EEEEEE", "#D55E00", "#7B3294", "#0072B2", "#00A6D6"]
    axes[1, 1].imshow(
        evidence_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(evidence_colors),
        vmin=0,
        vmax=4,
    )
    _plot_trajectory(
        axes[1, 1], trajectory, result, color="#1B1B1B", linewidth=0.8
    )
    evidence = result.metrics["obstacle_evidence"]
    derived_percent = 100.0 * evidence["final_derived_fraction"]
    axes[1, 1].set_title(
        "Obstacle provenance; "
        f"derived final cells {derived_percent:.1f}%"
    )
    axes[1, 1].legend(
        handles=[
            Patch(color=evidence_colors[1], label="merged-PCD endpoint"),
            Patch(color=evidence_colors[2], label="closing bridge only"),
            Patch(color=evidence_colors[3], label="dilation only"),
            Patch(color=evidence_colors[4], label="cleared by driven footprint"),
        ],
        loc="upper right",
    )

    for axis in axes.flat:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
        axis.grid(linewidth=0.25, alpha=0.5)
    figure.suptitle(
        f"{dataset_label}: full 3D SLAM PCD to conservative 2D navigation grid"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def write_path_review_figure(
    path: Path,
    dataset_label: str,
    trajectory: np.ndarray,
    result: ProjectionResult,
    plan: PathPlan,
) -> None:
    """Render footprint-aware route connectivity and clearance evidence."""
    geometry = result.geometry
    extent = [
        geometry.origin_x_m,
        geometry.max_x_m,
        geometry.origin_y_m,
        geometry.max_y_m,
    ]
    display = np.zeros(result.occupancy.shape, dtype=np.uint8)
    display[result.occupancy == GridClass.FREE] = 1
    display[result.occupancy == GridClass.OCCUPIED] = 2
    occupancy_cmap = ListedColormap(["#8A8A8A", "#FFFFFF", "#1B1B1B"])

    figure, axes = plt.subplots(1, 2, figsize=(18, 9))
    figure.subplots_adjust(
        left=0.055,
        right=0.965,
        bottom=0.16,
        top=0.90,
        wspace=0.16,
    )
    axes[0].imshow(
        display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=occupancy_cmap,
        vmin=0,
        vmax=2,
    )
    ramp_overlay = np.ma.masked_where(~result.ramp_mask, result.ramp_mask)
    axes[0].imshow(
        ramp_overlay,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#E69F00"]),
        alpha=0.65,
    )
    _plot_trajectory(
        axes[0], trajectory, result, color="#00A6D6", linewidth=0.9, alpha=0.8
    )
    lattice_invalid = ~plan.trajectory_lattice_valid
    if np.any(lattice_invalid):
        axes[0].scatter(
            trajectory[lattice_invalid, 0],
            trajectory[lattice_invalid, 1],
            color="#D55E00",
            marker="x",
            s=13,
            linewidths=0.7,
            zorder=4,
        )
    path_xy = plan.poses_xy_yaw[:, :2]
    axes[0].plot(path_xy[:, 0], path_xy[:, 1], color="#CC0077", linewidth=2.0)
    axes[0].scatter(
        [path_xy[0, 0], path_xy[-1, 0]],
        [path_xy[0, 1], path_xy[-1, 1]],
        c=["#009E73", "#D55E00"],
        s=45,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    axes[0].set_title(
        "Nav2 grid and generated route; UNKNOWN is blocked\n"
        f"path={plan.metrics['path_length_m']:.1f} m, "
        f"expansions={plan.metrics['expansions']}"
    )

    blocked = result.occupancy != GridClass.FREE
    clearance_m = (
        ndimage.distance_transform_edt(~blocked) * geometry.resolution_m
    )
    free_clearance = np.ma.masked_where(blocked, clearance_m)
    finite_clearance = clearance_m[~blocked]
    clearance_vmax = (
        max(1.0, float(np.percentile(finite_clearance, 95)))
        if finite_clearance.size
        else 1.0
    )
    clearance_image = axes[1].imshow(
        free_clearance,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=clearance_vmax,
    )
    axes[1].plot(path_xy[:, 0], path_xy[:, 1], color="#CC0077", linewidth=2.0)
    axes[1].scatter(
        [path_xy[0, 0], path_xy[-1, 0]],
        [path_xy[0, 1], path_xy[-1, 1]],
        c=["#009E73", "#D55E00"],
        s=45,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )
    figure.colorbar(clearance_image, ax=axes[1], label="blocked-cell clearance [m]")
    axes[1].set_title(
        "Footprint path clearance\n"
        f"minimum={plan.metrics['path_minimum_blocked_cell_clearance_m']:.2f} m, "
        f"swept collisions={plan.metrics['path_exact_swept_collision_sample_count']}"
    )

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
        axis.grid(linewidth=0.25, alpha=0.35)
    figure.legend(
        handles=[
            Patch(color="#8A8A8A", label="unknown / blocked"),
            Patch(color="#FFFFFF", ec="#777777", label="traversable floor"),
            Patch(color="#1B1B1B", label="wall / obstacle"),
            Patch(color="#E69F00", label="traversable ramp"),
            Line2D([0], [0], color="#00A6D6", label="optimized trajectory"),
            Line2D([0], [0], color="#CC0077", linewidth=2, label="planned path"),
            Line2D(
                [0],
                [0],
                color="#D55E00",
                marker="x",
                linestyle="none",
                label="lattice-invalid recorded pose",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
    )
    figure.suptitle(f"{dataset_label}: floor/person-filtered 2D planning audit")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
