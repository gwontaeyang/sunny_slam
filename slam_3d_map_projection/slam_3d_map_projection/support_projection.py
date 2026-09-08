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

"""Project soft 3D support evidence into a continuous, non-occupancy grid."""

from dataclasses import dataclass
from enum import IntEnum
import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from .projection import GridClass, GridGeometry  # noqa: E402


class TerrainSupportState(IntEnum):
    """Values shared with slam_3d_core::TerrainSupportState."""

    UNRESOLVED = 0
    PROBABLE_SUPPORT = 1
    CONFIRMED_SUPPORT = 2
    PROBABLE_HAZARD = 3
    CONFIRMED_HAZARD = 4


@dataclass
class SupportGridResult:
    """Continuous support layers aligned with one occupancy projection."""

    geometry: GridGeometry
    support_state: np.ndarray
    support_probability: np.ndarray
    geometric_traversability_score: np.ndarray
    traversal_cost: np.ndarray
    evidence_count: np.ndarray
    hard_hazard_mask: np.ndarray
    metrics: Dict[str, object]


_REQUIRED_FIELDS = {
    "x",
    "y",
    "z",
    "support_probability",
    "geometric_traversability_score",
    "label",
    "state",
    "confirmations",
    "observations",
}


def _validate_evidence(records: np.ndarray) -> None:
    if records.dtype.names is None:
        raise ValueError("support evidence must be a structured array")
    missing = sorted(_REQUIRED_FIELDS - set(records.dtype.names))
    if missing:
        raise ValueError(f"support evidence is missing fields: {', '.join(missing)}")
    finite_fields = (
        "x",
        "y",
        "z",
        "support_probability",
        "geometric_traversability_score",
    )
    if any(not np.isfinite(records[name]).all() for name in finite_fields):
        raise ValueError("support evidence contains non-finite values")
    probability = records["support_probability"].astype(np.float64)
    score = records["geometric_traversability_score"].astype(np.float64)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("support_probability must be in [0, 1]")
    if np.any((score < 0.0) | (score > 1.0)):
        raise ValueError("geometric_traversability_score must be in [0, 1]")
    states = records["state"].astype(np.int64)
    valid_states = np.asarray([state.value for state in TerrainSupportState])
    if not np.isin(states, valid_states).all():
        raise ValueError("support evidence contains an unknown state value")
    labels = records["label"].astype(np.int64)
    if np.any((labels < 0) | (labels > 5)):
        raise ValueError("support evidence contains an unknown terrain label")
    confirmations = records["confirmations"].astype(np.int64)
    observations = records["observations"].astype(np.int64)
    if np.any(confirmations < 0) or np.any(observations < confirmations):
        raise ValueError("support evidence observation counts are inconsistent")


def _disk_offsets(radius_m: float, resolution_m: float) -> np.ndarray:
    radius_cells = int(np.ceil(radius_m / resolution_m))
    if radius_cells <= 0:
        return np.asarray([[0, 0]], dtype=np.int64)
    offsets = []
    for row in range(-radius_cells, radius_cells + 1):
        for column in range(-radius_cells, radius_cells + 1):
            if np.hypot(row, column) * resolution_m <= radius_m + 1.0e-9:
                offsets.append((row, column))
    return np.asarray(offsets, dtype=np.int64)


def project_support_evidence(
    records: np.ndarray,
    geometry: GridGeometry,
    occupancy: np.ndarray,
    evidence_radius_m: float = 0.10,
    probable_support_min_cost: float = 0.35,
) -> SupportGridResult:
    """Rasterize support confidence without converting true unknown into free."""
    _validate_evidence(records)
    occupancy = np.asarray(occupancy)
    expected_shape = (geometry.height, geometry.width)
    if occupancy.shape != expected_shape:
        raise ValueError("occupancy and support grid geometry shapes do not match")
    if evidence_radius_m < 0.0:
        raise ValueError("evidence_radius_m cannot be negative")
    if not 0.0 <= probable_support_min_cost <= 1.0:
        raise ValueError("probable_support_min_cost must be in [0, 1]")

    shape = expected_shape
    support_state = np.full(shape, TerrainSupportState.UNRESOLVED, dtype=np.uint8)
    support_probability = np.full(shape, np.nan, dtype=np.float32)
    geometric_score = np.full(shape, np.nan, dtype=np.float32)
    traversal_cost = np.full(shape, np.nan, dtype=np.float32)
    evidence_count = np.zeros(shape, dtype=np.uint16)
    hard_hazard = occupancy == GridClass.OCCUPIED
    traversal_cost[hard_hazard] = 1.0

    xy = np.column_stack((records["x"], records["y"])).astype(np.float64)
    columns, rows = geometry.world_to_grid(xy)
    offsets = _disk_offsets(evidence_radius_m, geometry.resolution_m)
    cell_evidence: Dict[int, list[int]] = {}
    for index, (center_row, center_column) in enumerate(zip(rows, columns)):
        for row_offset, column_offset in offsets:
            row = int(center_row + row_offset)
            column = int(center_column + column_offset)
            if 0 <= row < geometry.height and 0 <= column < geometry.width:
                cell_evidence.setdefault(row * geometry.width + column, []).append(index)

    state_values = records["state"].astype(np.uint8)
    probabilities = records["support_probability"].astype(np.float64)
    scores = records["geometric_traversability_score"].astype(np.float64)
    observations = np.maximum(records["observations"].astype(np.float64), 1.0)
    for flat_index, indices in cell_evidence.items():
        row, column = divmod(flat_index, geometry.width)
        selected = np.asarray(indices, dtype=np.int64)
        states = state_values[selected]
        evidence_count[row, column] = min(len(indices), np.iinfo(np.uint16).max)

        if np.any(states == TerrainSupportState.CONFIRMED_HAZARD):
            support_state[row, column] = TerrainSupportState.CONFIRMED_HAZARD
            hard_hazard[row, column] = True
            traversal_cost[row, column] = 1.0
            continue
        if np.any(states == TerrainSupportState.PROBABLE_HAZARD):
            support_state[row, column] = TerrainSupportState.PROBABLE_HAZARD
            traversal_cost[row, column] = 1.0
            continue
        support = np.isin(
            states,
            [
                TerrainSupportState.PROBABLE_SUPPORT,
                TerrainSupportState.CONFIRMED_SUPPORT,
            ],
        )
        if np.any(states == TerrainSupportState.UNRESOLVED) or not np.any(support):
            support_state[row, column] = TerrainSupportState.UNRESOLVED
            continue

        support_indices = selected[support]
        weights = observations[support_indices]
        probability = float(np.average(probabilities[support_indices], weights=weights))
        score = float(np.average(scores[support_indices], weights=weights))
        confirmed = np.any(states[support] == TerrainSupportState.CONFIRMED_SUPPORT)
        state = (
            TerrainSupportState.CONFIRMED_SUPPORT
            if confirmed
            else TerrainSupportState.PROBABLE_SUPPORT
        )
        support_state[row, column] = state
        support_probability[row, column] = probability
        geometric_score[row, column] = score
        cost = 1.0 - score
        if state == TerrainSupportState.PROBABLE_SUPPORT:
            cost = max(cost, probable_support_min_cost)
        traversal_cost[row, column] = np.clip(cost, 0.0, 1.0)

    traversal_cost[hard_hazard] = 1.0
    support_mask = np.isin(
        support_state,
        [
            TerrainSupportState.PROBABLE_SUPPORT,
            TerrainSupportState.CONFIRMED_SUPPORT,
        ],
    )
    support_overridden_by_hard_hazard = support_mask & hard_hazard
    metrics = {
        "evidence_points": int(len(records)),
        "evidence_radius_m": float(evidence_radius_m),
        "probable_support_min_cost": float(probable_support_min_cost),
        "confirmed_support_cells": int(
            np.count_nonzero(support_state == TerrainSupportState.CONFIRMED_SUPPORT)
        ),
        "probable_support_cells": int(
            np.count_nonzero(support_state == TerrainSupportState.PROBABLE_SUPPORT)
        ),
        "unresolved_evidence_cells": int(
            np.count_nonzero(
                (support_state == TerrainSupportState.UNRESOLVED)
                & (evidence_count > 0)
            )
        ),
        "probable_hazard_cells": int(
            np.count_nonzero(support_state == TerrainSupportState.PROBABLE_HAZARD)
        ),
        "confirmed_hazard_cells": int(
            np.count_nonzero(support_state == TerrainSupportState.CONFIRMED_HAZARD)
        ),
        "hard_hazard_cells": int(np.count_nonzero(hard_hazard)),
        "support_cells_overridden_by_hard_hazard": int(
            np.count_nonzero(support_overridden_by_hard_hazard)
        ),
        "finite_cost_cells": int(np.count_nonzero(np.isfinite(traversal_cost))),
    }
    return SupportGridResult(
        geometry=geometry,
        support_state=support_state,
        support_probability=support_probability,
        geometric_traversability_score=geometric_score,
        traversal_cost=traversal_cost,
        evidence_count=evidence_count,
        hard_hazard_mask=hard_hazard,
        metrics=metrics,
    )


def write_support_grid_outputs(
    output_dir: Path,
    map_name: str,
    result: SupportGridResult,
) -> Dict[str, Path]:
    """Write lossless support arrays plus a documented 8-bit cost preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_path = output_dir / f"{map_name}_support_layers.npz"
    np.savez_compressed(
        layers_path,
        support_state=result.support_state,
        support_probability=result.support_probability,
        geometric_traversability_score=result.geometric_traversability_score,
        traversal_cost=result.traversal_cost,
        evidence_count=result.evidence_count,
        hard_hazard_mask=result.hard_hazard_mask,
        origin_xy_m=np.asarray(
            [result.geometry.origin_x_m, result.geometry.origin_y_m],
            dtype=np.float64,
        ),
        resolution_m=np.asarray(result.geometry.resolution_m, dtype=np.float64),
    )

    cost_pixels = np.full(result.traversal_cost.shape, 255, dtype=np.uint8)
    finite = np.isfinite(result.traversal_cost)
    cost_pixels[finite] = np.rint(
        np.clip(result.traversal_cost[finite], 0.0, 1.0) * 252.0
    ).astype(np.uint8)
    cost_pixels[result.hard_hazard_mask] = 254
    cost_path = output_dir / f"{map_name}_traversability_cost.pgm"
    stored = np.flipud(cost_pixels)
    with cost_path.open("wb") as stream:
        stream.write(f"P5\n{stored.shape[1]} {stored.shape[0]}\n255\n".encode("ascii"))
        stream.write(stored.tobytes())

    metrics_path = output_dir / f"{map_name}_support_metrics.json"
    metadata = dict(result.metrics)
    metadata.update(
        {
            "frame": "map",
            "origin_xy_m": [
                result.geometry.origin_x_m,
                result.geometry.origin_y_m,
            ],
            "resolution_m": result.geometry.resolution_m,
            "cost_encoding": {
                "0": "lowest finite traversal cost",
                "1..252": "scaled finite traversal cost",
                "254": "hard hazard or occupied",
                "255": "unknown/no finite cost",
            },
        }
    )
    metrics_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {
        "support_layers": layers_path,
        "traversability_cost": cost_path,
        "support_metrics": metrics_path,
    }


def write_support_review_figure(
    path: Path,
    dataset_label: str,
    result: SupportGridResult,
) -> None:
    """Render support state, vote ratio, score, and continuous traversal cost."""
    geometry = result.geometry
    extent = [
        geometry.origin_x_m,
        geometry.max_x_m,
        geometry.origin_y_m,
        geometry.max_y_m,
    ]
    figure, axes = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)

    state_colors = ["#919191", "#AFE155", "#28AA46", "#F5CD37", "#D73737"]
    state_names = [
        "unresolved",
        "probable support",
        "confirmed support",
        "probable hazard",
        "confirmed hazard",
    ]
    state_cmap = ListedColormap(state_colors)
    state_cmap.set_bad("#FFFFFF")
    state_display = np.ma.masked_where(
        result.evidence_count == 0,
        result.support_state,
    )
    axes[0, 0].imshow(
        state_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=state_cmap,
        norm=BoundaryNorm(np.arange(-0.5, 5.5, 1.0), state_cmap.N),
    )
    support_mask = np.isin(
        result.support_state,
        [
            TerrainSupportState.PROBABLE_SUPPORT,
            TerrainSupportState.CONFIRMED_SUPPORT,
        ],
    )
    overridden_support = support_mask & result.hard_hazard_mask
    override_display = np.ma.masked_where(
        ~overridden_support,
        np.ones_like(result.support_state),
    )
    axes[0, 0].imshow(
        override_display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#111111"]),
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].legend(
        handles=[
            Patch(color=color, label=name)
            for color, name in zip(state_colors, state_names)
        ]
        + [Patch(color="#111111", label="support overridden by hard hazard")],
        loc="upper right",
    )
    axes[0, 0].set_title("Terrain support state; white means no terrain evidence")

    probability = np.ma.masked_invalid(result.support_probability)
    probability_image = axes[0, 1].imshow(
        probability,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("Support vote ratio; not a calibrated probability")
    figure.colorbar(probability_image, ax=axes[0, 1], label="support vote ratio [0, 1]")

    score = np.ma.masked_invalid(result.geometric_traversability_score)
    score_image = axes[1, 0].imshow(
        score,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 0].set_title("Geometry-only traversability score; robot limits not applied")
    figure.colorbar(score_image, ax=axes[1, 0], label="geometric score [0, 1]")

    cost = np.ma.masked_invalid(result.traversal_cost)
    cost_cmap = plt.get_cmap("RdYlGn_r").copy()
    cost_cmap.set_bad("#919191")
    cost_image = axes[1, 1].imshow(
        cost,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=cost_cmap,
        vmin=0.0,
        vmax=1.0,
    )
    axes[1, 1].set_title("Continuous traversal cost; gray is unknown")
    figure.colorbar(cost_image, ax=axes[1, 1], label="traversal cost [0, 1]")

    for axis in axes.flat:
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("map x [m]")
        axis.set_ylabel("map y [m]")
        axis.grid(linewidth=0.25, alpha=0.35)
    metrics = result.metrics
    figure.suptitle(
        f"{dataset_label}: probabilistic terrain support surface | "
        f"confirmed={metrics['confirmed_support_cells']}, "
        f"probable={metrics['probable_support_cells']}, "
        f"unresolved={metrics['unresolved_evidence_cells']}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
