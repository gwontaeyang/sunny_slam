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

"""CLI for inexpensive 2 cm rerasterization of a validated terrain grid."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from .configuration import load_projection_config  # noqa: E402
from .multiresolution import refine_navigation_grid  # noqa: E402
from .outputs import write_trinary_map  # noqa: E402
from .pcd_io import read_pcd_xyz  # noqa: E402
from .projection import GridClass, GridGeometry  # noqa: E402


def _parser():
    parser = argparse.ArgumentParser(
        description="Rerasterize validated terrain and obstacle evidence at 2 cm."
    )
    parser.add_argument("--terrain-layers", required=True, type=Path)
    parser.add_argument("--planning-obstacles-pcd", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--map-name", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--output-resolution-m", type=float, default=0.02)
    parser.add_argument("--obstacle-closing-radius-m", type=float, default=0.10)
    parser.add_argument("--enclosure-closing-radius-m", type=float, default=0.16)
    parser.add_argument("--max-grid-cells", type=int, default=25_000_000)
    parser.add_argument(
        "--fill-enclosed-unknown-as-free",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def _write_review(path, dataset_label, result):
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
    figure, axis = plt.subplots(figsize=(10, 14), constrained_layout=True)
    axis.imshow(
        display,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#8A8A8A", "#FFFFFF", "#1B1B1B"]),
        vmin=0,
        vmax=2,
    )
    has_completed_floor = np.any(result.enclosed_unknown_filled_mask)
    if has_completed_floor:
        completed = np.ma.masked_where(
            ~result.enclosed_unknown_filled_mask,
            result.enclosed_unknown_filled_mask,
        )
        axis.imshow(
            completed,
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=ListedColormap(["#56B4E9"]),
            alpha=0.28,
        )
    counts = result.metrics["cell_counts"]
    axis.set_title(
        f"{dataset_label}\n"
        f"2 cm Nav2 grid: free={counts['free']:,}, "
        f"occupied={counts['occupied']:,}, unknown={counts['unknown']:,}"
    )
    axis.set_xlabel("map x [m]")
    axis.set_ylabel("map y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(linewidth=0.25, alpha=0.35)
    legend_handles = [
        Patch(color="#FFFFFF", label="FREE"),
        Patch(color="#1B1B1B", label="OCCUPIED"),
        Patch(color="#8A8A8A", label="UNKNOWN"),
    ]
    if has_completed_floor:
        legend_handles.append(
            Patch(color="#56B4E9", label="enclosed floor inferred FREE")
        )
    axis.legend(handles=legend_handles, loc="upper right")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    args = _parser().parse_args()
    config = load_projection_config(args.config)
    with np.load(args.terrain_layers) as layers:
        occupancy = np.asarray(layers["occupancy"])
        origin = np.asarray(layers["origin_xy_m"], dtype=np.float64)
        resolution = float(layers["resolution_m"])
        ramp = np.asarray(layers["ramp_mask"], dtype=bool)
        terrain_hazard = np.asarray(layers["terrain_hazard_mask"], dtype=bool)
        coarse_completed = (
            np.asarray(layers["enclosed_unknown_filled_mask"], dtype=bool)
            if "enclosed_unknown_filled_mask" in layers
            else np.zeros(occupancy.shape, dtype=bool)
        )
    if not np.isclose(resolution, config.resolution_m):
        raise ValueError(
            "terrain layer resolution does not match the projection config"
        )
    geometry = GridGeometry(
        origin_x_m=float(origin[0]),
        origin_y_m=float(origin[1]),
        resolution_m=resolution,
        width=int(occupancy.shape[1]),
        height=int(occupancy.shape[0]),
    )
    obstacle_points = read_pcd_xyz(args.planning_obstacles_pcd)
    result = refine_navigation_grid(
        occupancy,
        geometry,
        obstacle_points,
        ramp,
        terrain_hazard,
        coarse_completed,
        output_resolution_m=args.output_resolution_m,
        obstacle_closing_radius_m=args.obstacle_closing_radius_m,
        enclosure_closing_radius_m=args.enclosure_closing_radius_m,
        fill_enclosed_unknown_as_free=args.fill_enclosed_unknown_as_free,
        max_grid_cells=args.max_grid_cells,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_outputs = write_trinary_map(
        args.output_dir,
        args.map_name,
        result.occupancy,
        result.geometry,
        config,
    )
    layers_path = args.output_dir / f"{args.map_name}_multires_layers.npz"
    np.savez_compressed(
        layers_path,
        occupancy=result.occupancy,
        ramp_mask=result.ramp_mask,
        terrain_hazard_mask=result.terrain_hazard_mask,
        obstacle_endpoint_mask=result.obstacle_endpoint_mask,
        obstacle_bridge_mask=result.obstacle_bridge_mask,
        enclosure_bridge_mask=result.enclosure_bridge_mask,
        inferred_enclosure_wall_mask=result.inferred_enclosure_wall_mask,
        enclosed_unknown_filled_mask=result.enclosed_unknown_filled_mask,
        origin_xy_m=np.asarray(
            [result.geometry.origin_x_m, result.geometry.origin_y_m],
            dtype=np.float64,
        ),
        resolution_m=np.asarray(result.geometry.resolution_m, dtype=np.float64),
    )
    review_path = args.output_dir / f"{args.map_name}_review.png"
    _write_review(review_path, args.dataset_label, result)
    metrics = dict(result.metrics)
    metrics["inputs"] = {
        "terrain_layers": str(args.terrain_layers),
        "planning_obstacles_pcd": str(args.planning_obstacles_pcd),
    }
    metrics["outputs"] = {
        **{name: path.name for name, path in map_outputs.items()},
        "multires_layers": layers_path.name,
        "review": review_path.name,
    }
    metrics_path = args.output_dir / f"{args.map_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
