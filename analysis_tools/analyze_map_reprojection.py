#!/usr/bin/env python3
"""Compare global-map PCD snapshots around a pose-graph correction."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def read_xyz_pcd(path):
    path = Path(path)
    header = {}
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header is incomplete: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                break
        payload = stream.read()

    fields = header.get("FIELDS", [])
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
    if header.get("DATA") != ["binary"]:
        raise ValueError("only binary PCD input is supported")
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD field metadata lengths differ")

    formats = []
    for size, value_type, count in zip(sizes, types, counts):
        if value_type == "F" and size == 4:
            scalar = "<f4"
        elif value_type == "F" and size == 8:
            scalar = "<f8"
        elif value_type == "U" and size in (1, 2, 4, 8):
            scalar = f"<u{size}"
        elif value_type == "I" and size in (1, 2, 4, 8):
            scalar = f"<i{size}"
        else:
            raise ValueError(f"unsupported PCD field type: {value_type}{size}")
        formats.append((scalar, count) if count > 1 else scalar)
    dtype = np.dtype({"names": fields, "formats": formats})
    expected_bytes = points * dtype.itemsize
    if len(payload) < expected_bytes:
        raise ValueError("PCD payload is shorter than POINTS metadata")
    records = np.frombuffer(payload[:expected_bytes], dtype=dtype, count=points)
    if not all(name in records.dtype.names for name in ("x", "y", "z")):
        raise ValueError("PCD input does not contain x/y/z fields")
    xyz = np.column_stack([records[name] for name in ("x", "y", "z")]).astype(float)
    return xyz[np.isfinite(xyz).all(axis=1)]


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def deterministic_sample(points, maximum):
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
    return points[indices]


def compare_maps(before, after, maximum_query_points=200_000):
    before_query = deterministic_sample(before, maximum_query_points)
    after_query = deterministic_sample(after, maximum_query_points)
    before_to_after = cKDTree(after).query(before_query, workers=1)[0]
    after_to_before = cKDTree(before).query(after_query, workers=1)[0]
    thresholds = (0.01, 0.05, 0.10, 0.25, 0.50)
    return {
        "before_points": int(len(before)),
        "after_points": int(len(after)),
        "point_count_change": int(len(after) - len(before)),
        "before_centroid_m": np.mean(before, axis=0).tolist(),
        "after_centroid_m": np.mean(after, axis=0).tolist(),
        "centroid_shift_m": float(
            np.linalg.norm(np.mean(after, axis=0) - np.mean(before, axis=0))
        ),
        "before_to_after_nearest_m": distribution(before_to_after),
        "after_to_before_nearest_m": distribution(after_to_before),
        "before_points_over_threshold_ratio": {
            f"{threshold:g}": float(np.mean(before_to_after > threshold))
            for threshold in thresholds
        },
        "after_points_over_threshold_ratio": {
            f"{threshold:g}": float(np.mean(after_to_before > threshold))
            for threshold in thresholds
        },
    }, before_query, after_query, before_to_after


def plot_comparison(before, after, distances, output_path, dataset):
    figure, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    figure.suptitle(f"{dataset}: global map before/after pose-graph correction")
    display_before = deterministic_sample(before, 35_000)
    display_after = deterministic_sample(after, 35_000)
    axes[0, 0].scatter(display_before[:, 0], display_before[:, 1], s=1, alpha=0.3)
    axes[0, 0].scatter(display_after[:, 0], display_after[:, 1], s=1, alpha=0.3)
    axes[0, 0].set_xlabel("Map X [m]")
    axes[0, 0].set_ylabel("Map Y [m]")
    axes[0, 0].set_aspect("equal", adjustable="datalim")
    axes[0, 1].scatter(display_before[:, 0], display_before[:, 2], s=1, alpha=0.3)
    axes[0, 1].scatter(display_after[:, 0], display_after[:, 2], s=1, alpha=0.3)
    axes[0, 1].set_xlabel("Map X [m]")
    axes[0, 1].set_ylabel("Map Z [m]")
    finite = distances[np.isfinite(distances)]
    axes[1, 0].hist(np.clip(finite, 0.0, 1.0), bins=60)
    axes[1, 0].set_xlabel("Before-to-after nearest distance [m], clipped at 1 m")
    axes[1, 0].set_ylabel("Point count")
    sorted_distances = np.sort(finite)
    cdf = np.arange(1, len(sorted_distances) + 1) / max(1, len(sorted_distances))
    axes[1, 1].plot(sorted_distances, cdf)
    axes[1, 1].set_xlim(0.0, min(1.0, max(0.1, float(np.percentile(finite, 99)))))
    axes[1, 1].set_xlabel("Before-to-after nearest distance [m]")
    axes[1, 1].set_ylabel("Empirical CDF")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before_pcd", type=Path)
    parser.add_argument("after_pcd", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", default="dataset")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before = read_xyz_pcd(args.before_pcd)
    after = read_xyz_pcd(args.after_pcd)
    if not len(before) or not len(after):
        raise ValueError("both map snapshots must contain finite XYZ points")
    summary, before_query, after_query, distances = compare_maps(before, after)
    summary.update(
        {
            "dataset": args.dataset,
            "before_pcd": str(args.before_pcd),
            "after_pcd": str(args.after_pcd),
            "interpretation_note": (
                "Nearest distance includes voxel re-selection and any scan added between "
                "map publications; it is evidence of map regeneration, not per-point identity."
            ),
        }
    )
    (args.output_dir / "map_reprojection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_comparison(
        before_query,
        after_query,
        distances,
        args.output_dir / "map_reprojection_overlay.png",
        args.dataset,
    )
    plot_comparison(
        before_query,
        after_query,
        distances,
        args.output_dir / "map_reprojection_overlay.svg",
        args.dataset,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
