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
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from .outputs import point_class_colors  # noqa: E402
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
    if hazard_rows.size:
        axes[0, 1].legend(loc="upper right")
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
