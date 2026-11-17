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

"""Multi-resolution final rasterization for navigation grids."""

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Dict

import numpy as np
from scipy import ndimage

from .projection import GridClass, GridGeometry


@dataclass
class RefinedNavigationGrid:
    """A fine navigation raster and provenance retained from coarse evidence."""

    geometry: GridGeometry
    occupancy: np.ndarray
    ramp_mask: np.ndarray
    terrain_hazard_mask: np.ndarray
    obstacle_endpoint_mask: np.ndarray
    obstacle_bridge_mask: np.ndarray
    enclosure_bridge_mask: np.ndarray
    inferred_enclosure_wall_mask: np.ndarray
    enclosed_unknown_filled_mask: np.ndarray
    metrics: Dict[str, object]


def _disk(radius_m: float, resolution_m: float) -> np.ndarray:
    radius_cells = int(np.ceil(radius_m / resolution_m))
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    rows, columns = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]
    radius_squared = (radius_m / resolution_m) ** 2 + 1.0e-9
    return rows * rows + columns * columns <= radius_squared


def _fine_geometry(
    source: GridGeometry,
    output_resolution_m: float,
    max_grid_cells: int,
) -> GridGeometry:
    source_geometry_values = (
        source.origin_x_m,
        source.origin_y_m,
        source.resolution_m,
    )
    if not all(math.isfinite(value) for value in source_geometry_values):
        raise ValueError("source geometry values must be finite")
    if source.resolution_m <= 0.0 or source.width <= 0 or source.height <= 0:
        raise ValueError("source geometry resolution and dimensions must be positive")
    if not math.isfinite(output_resolution_m) or output_resolution_m <= 0.0:
        raise ValueError("output_resolution_m must be finite and positive")
    if output_resolution_m > source.resolution_m:
        raise ValueError("output resolution cannot be coarser than source evidence")
    if isinstance(max_grid_cells, bool) or not isinstance(max_grid_cells, Integral):
        raise ValueError("max_grid_cells must be an integer")
    if max_grid_cells <= 0:
        raise ValueError("max_grid_cells must be positive")
    width = int(math.ceil((source.max_x_m - source.origin_x_m) / output_resolution_m))
    height = int(
        math.ceil((source.max_y_m - source.origin_y_m) / output_resolution_m)
    )
    if width * height > max_grid_cells:
        raise ValueError(
            f"refined grid has {width * height} cells, above max_grid_cells="
            f"{max_grid_cells}"
        )
    return GridGeometry(
        origin_x_m=source.origin_x_m,
        origin_y_m=source.origin_y_m,
        resolution_m=output_resolution_m,
        width=width,
        height=height,
    )


def _nearest_resample(
    source_array: np.ndarray,
    source_geometry: GridGeometry,
    target_geometry: GridGeometry,
) -> np.ndarray:
    expected_shape = (source_geometry.height, source_geometry.width)
    if source_array.shape != expected_shape:
        raise ValueError(
            f"source layer shape {source_array.shape} does not match {expected_shape}"
        )
    target_x = (
        target_geometry.origin_x_m
        + (np.arange(target_geometry.width, dtype=np.float64) + 0.5)
        * target_geometry.resolution_m
    )
    target_y = (
        target_geometry.origin_y_m
        + (np.arange(target_geometry.height, dtype=np.float64) + 0.5)
        * target_geometry.resolution_m
    )
    source_columns = np.floor(
        (target_x - source_geometry.origin_x_m) / source_geometry.resolution_m
    ).astype(np.int64)
    source_rows = np.floor(
        (target_y - source_geometry.origin_y_m) / source_geometry.resolution_m
    ).astype(np.int64)
    source_columns = np.clip(source_columns, 0, source_geometry.width - 1)
    source_rows = np.clip(source_rows, 0, source_geometry.height - 1)
    return source_array[np.ix_(source_rows, source_columns)]


def refine_navigation_grid(
    coarse_occupancy: np.ndarray,
    coarse_geometry: GridGeometry,
    planning_obstacle_points: np.ndarray,
    coarse_ramp_mask: np.ndarray,
    coarse_terrain_hazard_mask: np.ndarray,
    coarse_enclosed_unknown_filled_mask: np.ndarray,
    output_resolution_m: float = 0.02,
    obstacle_closing_radius_m: float = 0.10,
    enclosure_closing_radius_m: float = 0.16,
    fill_enclosed_unknown_as_free: bool = False,
    max_grid_cells: int = 25_000_000,
) -> RefinedNavigationGrid:
    """Retain coarse terrain semantics while rerasterizing obstacles at 2 cm."""
    coarse_occupancy = np.asarray(coarse_occupancy)
    expected_shape = (coarse_geometry.height, coarse_geometry.width)
    if coarse_occupancy.shape != expected_shape:
        raise ValueError("coarse occupancy shape does not match its geometry")
    valid_values = np.isin(
        coarse_occupancy,
        [GridClass.UNKNOWN, GridClass.FREE, GridClass.OCCUPIED],
    )
    if not np.all(valid_values):
        raise ValueError("coarse occupancy violates the trinary contract")
    obstacle_points = np.asarray(planning_obstacle_points, dtype=np.float64)
    if obstacle_points.ndim != 2 or obstacle_points.shape[1] < 2:
        raise ValueError("planning obstacle points must have at least XY columns")
    if obstacle_points.shape[0] == 0 or not np.all(np.isfinite(obstacle_points[:, :2])):
        raise ValueError("planning obstacle points must be nonempty and finite")
    if (
        not math.isfinite(obstacle_closing_radius_m)
        or obstacle_closing_radius_m < 0.0
    ):
        raise ValueError("obstacle_closing_radius_m must be finite and nonnegative")
    if not math.isfinite(enclosure_closing_radius_m):
        raise ValueError("enclosure_closing_radius_m must be finite")
    if enclosure_closing_radius_m < obstacle_closing_radius_m:
        raise ValueError(
            "enclosure_closing_radius_m cannot be smaller than the final "
            "obstacle closing radius"
        )
    if not isinstance(fill_enclosed_unknown_as_free, bool):
        raise ValueError("fill_enclosed_unknown_as_free must be a boolean")

    geometry = _fine_geometry(
        coarse_geometry,
        output_resolution_m,
        max_grid_cells,
    )
    coarse_free = _nearest_resample(
        coarse_occupancy == GridClass.FREE,
        coarse_geometry,
        geometry,
    )
    terrain_hazard = _nearest_resample(
        np.asarray(coarse_terrain_hazard_mask, dtype=bool),
        coarse_geometry,
        geometry,
    )
    ramp = _nearest_resample(
        np.asarray(coarse_ramp_mask, dtype=bool),
        coarse_geometry,
        geometry,
    )
    coarse_completed = _nearest_resample(
        np.asarray(coarse_enclosed_unknown_filled_mask, dtype=bool),
        coarse_geometry,
        geometry,
    )

    columns, rows = geometry.world_to_grid(obstacle_points[:, :2])
    valid = (
        (columns >= 0)
        & (columns < geometry.width)
        & (rows >= 0)
        & (rows < geometry.height)
    )
    endpoint = np.zeros((geometry.height, geometry.width), dtype=bool)
    endpoint[rows[valid], columns[valid]] = True
    closed = endpoint.copy()
    if obstacle_closing_radius_m > 0.0:
        closed |= ndimage.binary_closing(
            endpoint,
            structure=_disk(obstacle_closing_radius_m, output_resolution_m),
        )
    bridge = closed & ~endpoint

    enclosure_wall = closed.copy()
    if enclosure_closing_radius_m > obstacle_closing_radius_m:
        enclosure_wall |= ndimage.binary_closing(
            endpoint,
            structure=_disk(enclosure_closing_radius_m, output_resolution_m),
        )
    enclosure_bridge = enclosure_wall & ~closed

    source_free = coarse_free & ~closed
    known_floor_or_wall = source_free | enclosure_wall
    enclosed_floor_candidate = (
        ndimage.binary_fill_holes(known_floor_or_wall)
        & ~known_floor_or_wall
        & ~terrain_hazard
    )

    # A coarse completion mask marks topology-inferred floor, not direct floor
    # evidence. Remove it again when the conservative no-fill policy is active.
    free = source_free.copy()
    if not fill_enclosed_unknown_as_free:
        free &= ~coarse_completed
    newly_completed = np.zeros_like(free)
    if fill_enclosed_unknown_as_free:
        newly_completed = enclosed_floor_candidate
        free |= newly_completed
    completed = (coarse_completed | newly_completed) & free
    ramp &= free

    occupancy = np.full(free.shape, GridClass.UNKNOWN, dtype=np.int8)
    occupancy[free] = GridClass.FREE
    occupancy[closed] = GridClass.OCCUPIED
    # Infer only the short wall bridges required by the completed topology, but
    # keep that wall decision independent of whether enclosed floor is exported
    # as FREE or UNKNOWN.
    topology_free = source_free | enclosed_floor_candidate
    topology_unknown = ~(topology_free | closed)
    topology_border_seed = np.zeros_like(topology_unknown)
    topology_border_seed[0, :] = topology_unknown[0, :]
    topology_border_seed[-1, :] = topology_unknown[-1, :]
    topology_border_seed[:, 0] = topology_unknown[:, 0]
    topology_border_seed[:, -1] = topology_unknown[:, -1]
    topology_exterior_unknown = ndimage.binary_propagation(
        topology_border_seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=topology_unknown,
    )
    topology_interior_unknown = topology_unknown & ~topology_exterior_unknown
    inferred_enclosure_wall = (
        topology_interior_unknown & enclosure_bridge & ~terrain_hazard
    )
    occupancy[inferred_enclosure_wall] = GridClass.OCCUPIED

    unknown = occupancy == GridClass.UNKNOWN
    border_seed = np.zeros_like(unknown)
    border_seed[0, :] = unknown[0, :]
    border_seed[-1, :] = unknown[-1, :]
    border_seed[:, 0] = unknown[:, 0]
    border_seed[:, -1] = unknown[:, -1]
    exterior_unknown = ndimage.binary_propagation(
        border_seed,
        structure=np.ones((3, 3), dtype=bool),
        mask=unknown,
    )
    interior_unknown = unknown & ~exterior_unknown
    remaining_interior_unknown = interior_unknown & ~inferred_enclosure_wall
    cell_area = output_resolution_m * output_resolution_m
    metrics = {
        "method": "coarse_terrain_semantics_fine_obstacle_raster",
        "source_resolution_m": coarse_geometry.resolution_m,
        "output_resolution_m": output_resolution_m,
        "width": geometry.width,
        "height": geometry.height,
        "origin": [geometry.origin_x_m, geometry.origin_y_m, 0.0],
        "source_obstacle_points": int(obstacle_points.shape[0]),
        "source_obstacle_points_in_grid": int(np.count_nonzero(valid)),
        "obstacle_endpoint_cells": int(np.count_nonzero(endpoint)),
        "obstacle_bridge_cells": int(np.count_nonzero(bridge)),
        "obstacle_closing_radius_m": obstacle_closing_radius_m,
        "enclosure_closing_radius_m": enclosure_closing_radius_m,
        "enclosure_bridge_cells_before_selective_wall_inference": int(
            np.count_nonzero(enclosure_bridge)
        ),
        "inferred_enclosure_wall_cells": int(
            np.count_nonzero(inferred_enclosure_wall)
        ),
        "fill_enclosed_unknown_as_free": fill_enclosed_unknown_as_free,
        "enclosed_floor_candidate_cells": int(
            np.count_nonzero(enclosed_floor_candidate)
        ),
        "newly_completed_free_cells": int(np.count_nonzero(newly_completed)),
        "completed_free_cells_total": int(np.count_nonzero(completed)),
        "cell_counts": {
            "free": int(np.count_nonzero(occupancy == GridClass.FREE)),
            "occupied": int(np.count_nonzero(occupancy == GridClass.OCCUPIED)),
            "unknown": int(np.count_nonzero(occupancy == GridClass.UNKNOWN)),
        },
        "unknown_provenance": {
            "exterior_cells": int(np.count_nonzero(exterior_unknown)),
            "interior_cells_before_wall_inference": int(
                np.count_nonzero(interior_unknown)
            ),
            "interior_cells_after_wall_inference": int(
                np.count_nonzero(remaining_interior_unknown)
            ),
            "interior_nonhazard_cells_after_wall_inference": int(
                np.count_nonzero(remaining_interior_unknown & ~terrain_hazard)
            ),
        },
        "areas_m2": {
            "free": float(np.count_nonzero(occupancy == GridClass.FREE) * cell_area),
            "occupied": float(
                np.count_nonzero(occupancy == GridClass.OCCUPIED) * cell_area
            ),
            "newly_completed_free": float(
                np.count_nonzero(newly_completed) * cell_area
            ),
        },
        "exterior_unknown_preserved": True,
        "terrain_hazards_preserved_unknown_unless_coarse_free": True,
        "source_point_resolution_caveat": (
            "fine raster cannot recreate geometry removed by upstream voxelization"
        ),
    }
    return RefinedNavigationGrid(
        geometry=geometry,
        occupancy=occupancy,
        ramp_mask=ramp,
        terrain_hazard_mask=terrain_hazard,
        obstacle_endpoint_mask=endpoint,
        obstacle_bridge_mask=bridge,
        enclosure_bridge_mask=enclosure_bridge,
        inferred_enclosure_wall_mask=inferred_enclosure_wall,
        enclosed_unknown_filled_mask=completed,
        metrics=metrics,
    )
