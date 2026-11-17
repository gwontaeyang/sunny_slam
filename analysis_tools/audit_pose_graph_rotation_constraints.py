#!/usr/bin/env python3
"""Audit pose-graph rotation consistency from completed SLAM replays."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class DatasetMetrics:
    bag: str
    node_count: int
    edge_count: int
    adjacent_edge_count: int
    near_chain_edge_count: int
    unobserved_nonadjacent_edge_count: int
    nonadjacent_yaw_disagreement_gt_30_deg: int
    nonadjacent_yaw_disagreement_gt_80_deg: int
    endpoint_correction_xy_m: float
    endpoint_correction_yaw_deg: float
    maximum_correction_xy_m: float
    maximum_correction_yaw_deg: float
    revisit_pair_count: int
    revisit_graph_distance_median_m: float
    revisit_graph_distance_p95_m: float
    revisit_graph_distance_max_m: float
    revisit_yaw_error_p95_deg: float
    exact_candidate_count: int
    accepted_near_chain_candidate_count: int
    candidate_prediction_translation_p95_m: float
    candidate_prediction_yaw_p95_deg: float
    candidate_odom_yaw_disagreement_p95_deg: float
    candidate_large_prediction_disagreement_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plot-bag", action="append", default=[])
    parser.add_argument("--odom-revisit-radius-m", type=float, default=0.5)
    parser.add_argument("--minimum-state-separation", type=int, default=30)
    parser.add_argument("--minimum-separation-fraction", type=float, default=0.2)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def to_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def normalized_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def percentile(values: Iterable[float], quantile: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    if len(finite) == 1:
        return finite[0]
    index = (len(finite) - 1) * min(max(quantile, 0.0), 1.0)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    fraction = index - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def candidate_log_path(variant_root: Path, bag: str) -> Path:
    direct = variant_root / "logs" / f"{variant_root.name}_variant"
    candidates = sorted(direct.glob(f"{bag}_near_chain_candidate_events.csv"))
    if candidates:
        return candidates[-1]
    recursive = sorted(
        (variant_root / "logs").glob(
            f"**/{bag}_near_chain_candidate_events.csv"
        )
    )
    return recursive[-1] if recursive else Path()


def pair_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def accepted_near_chain_pairs(
    candidates: Sequence[dict[str, str]],
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for row in candidates:
        if not is_true(row.get("edge_present")) or is_true(row.get("prelinked")):
            continue
        source = to_int(row.get("source_state_id"))
        target = to_int(row.get("target_state_id"))
        if source >= 0 and target >= 0:
            pairs.add(pair_key(source, target))
    return pairs


def classify_edge(
    edge: dict[str, str],
    near_chain_pairs: set[tuple[int, int]],
) -> str:
    source = to_int(edge.get("source_state_id"))
    target = to_int(edge.get("target_state_id"))
    if source < 0 or target < 0:
        return "invalid"
    if abs(target - source) == 1:
        return "adjacent"
    if pair_key(source, target) in near_chain_pairs:
        return "near_chain"
    # The existing observer does not expose the running-chain endpoint. This
    # bucket may contain a running-chain edge or a global loop edge, so keep the
    # name deliberately evidence-limited.
    return "unobserved_nonadjacent"


def yaw_disagreement_to_odom(
    source: int,
    target: int,
    measured_yaw_rad: float,
    nodes_by_state: dict[int, dict[str, str]],
) -> float:
    if source not in nodes_by_state or target not in nodes_by_state:
        return math.nan
    source_yaw = to_float(nodes_by_state[source].get("odom_yaw_rad"))
    target_yaw = to_float(nodes_by_state[target].get("odom_yaw_rad"))
    if not all(math.isfinite(value) for value in (source_yaw, target_yaw, measured_yaw_rad)):
        return math.nan
    odom_relative = normalized_angle(target_yaw - source_yaw)
    return abs(math.degrees(normalized_angle(measured_yaw_rad - odom_relative)))


def correction_values(node: dict[str, str]) -> tuple[float, float]:
    x = to_float(node.get("x_m"))
    y = to_float(node.get("y_m"))
    yaw = to_float(node.get("yaw_rad"))
    odom_x = to_float(node.get("odom_x_m"))
    odom_y = to_float(node.get("odom_y_m"))
    odom_yaw = to_float(node.get("odom_yaw_rad"))
    if not all(
        math.isfinite(value)
        for value in (x, y, yaw, odom_x, odom_y, odom_yaw)
    ):
        return math.nan, math.nan
    return (
        math.hypot(x - odom_x, y - odom_y),
        abs(math.degrees(normalized_angle(yaw - odom_yaw))),
    )


def revisit_metrics(
    nodes: Sequence[dict[str, str]],
    radius_m: float,
    minimum_separation: int,
) -> tuple[int, float, float, float, float]:
    graph_distances: list[float] = []
    yaw_errors_deg: list[float] = []
    for source_index, source in enumerate(nodes):
        for target_index in range(source_index + minimum_separation, len(nodes)):
            target = nodes[target_index]
            source_odom_x = to_float(source.get("odom_x_m"))
            source_odom_y = to_float(source.get("odom_y_m"))
            target_odom_x = to_float(target.get("odom_x_m"))
            target_odom_y = to_float(target.get("odom_y_m"))
            if not all(
                math.isfinite(value)
                for value in (
                    source_odom_x,
                    source_odom_y,
                    target_odom_x,
                    target_odom_y,
                )
            ):
                continue
            if math.hypot(
                target_odom_x - source_odom_x,
                target_odom_y - source_odom_y,
            ) > radius_m:
                continue
            source_x = to_float(source.get("x_m"))
            source_y = to_float(source.get("y_m"))
            target_x = to_float(target.get("x_m"))
            target_y = to_float(target.get("y_m"))
            source_yaw = to_float(source.get("yaw_rad"))
            target_yaw = to_float(target.get("yaw_rad"))
            source_odom_yaw = to_float(source.get("odom_yaw_rad"))
            target_odom_yaw = to_float(target.get("odom_yaw_rad"))
            if not all(
                math.isfinite(value)
                for value in (
                    source_x,
                    source_y,
                    target_x,
                    target_y,
                    source_yaw,
                    target_yaw,
                    source_odom_yaw,
                    target_odom_yaw,
                )
            ):
                continue
            graph_distances.append(math.hypot(target_x - source_x, target_y - source_y))
            graph_delta_yaw = normalized_angle(target_yaw - source_yaw)
            odom_delta_yaw = normalized_angle(target_odom_yaw - source_odom_yaw)
            yaw_errors_deg.append(
                abs(math.degrees(normalized_angle(graph_delta_yaw - odom_delta_yaw)))
            )
    return (
        len(graph_distances),
        statistics.median(graph_distances) if graph_distances else math.nan,
        percentile(graph_distances, 0.95),
        max(graph_distances) if graph_distances else math.nan,
        percentile(yaw_errors_deg, 0.95),
    )


def analyze_dataset(
    bag_dir: Path,
    candidate_path: Path,
    revisit_radius_m: float,
    minimum_state_separation: int,
    minimum_separation_fraction: float,
) -> tuple[DatasetMetrics, list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = read_csv(bag_dir / "trajectory_pose_graph_final.csv")
    edges = read_csv(bag_dir / "pose_graph_edges_final.csv")
    candidates = read_csv(candidate_path) if candidate_path.is_file() else []
    nodes_by_state = {to_int(row.get("state_id")): row for row in nodes}
    near_pairs = accepted_near_chain_pairs(candidates)

    edge_rows: list[dict[str, Any]] = []
    edge_classes: list[str] = []
    nonadjacent_yaw: list[float] = []
    for edge in edges:
        edge_class = classify_edge(edge, near_pairs)
        source = to_int(edge.get("source_state_id"))
        target = to_int(edge.get("target_state_id"))
        yaw_disagreement_deg = yaw_disagreement_to_odom(
            source,
            target,
            to_float(edge.get("relative_yaw_rad")),
            nodes_by_state,
        )
        edge_classes.append(edge_class)
        if edge_class != "adjacent" and math.isfinite(yaw_disagreement_deg):
            nonadjacent_yaw.append(yaw_disagreement_deg)
        edge_rows.append(
            {
                "bag": bag_dir.name,
                "edge_class": edge_class,
                "source_state_id": source,
                "target_state_id": target,
                "state_gap": abs(target - source),
                "relative_yaw_rad": to_float(edge.get("relative_yaw_rad")),
                "odom_yaw_disagreement_deg": yaw_disagreement_deg,
                "covariance_yawyaw": to_float(edge.get("covariance_yawyaw")),
                "final_residual_translation_m": to_float(
                    edge.get("final_residual_translation_m")
                ),
                "final_residual_yaw_deg": math.degrees(
                    to_float(edge.get("final_residual_yaw_rad"))
                ),
            }
        )

    candidate_rows: list[dict[str, Any]] = []
    candidate_translation: list[float] = []
    candidate_yaw_deg: list[float] = []
    candidate_odom_yaw_deg: list[float] = []
    accepted_count = 0
    large_prediction_disagreement_count = 0
    for row in candidates:
        source = to_int(row.get("source_state_id"))
        target = to_int(row.get("target_state_id"))
        translation = to_float(row.get("translation_disagreement_m"))
        yaw_rad = to_float(row.get("yaw_disagreement_rad"))
        yaw_deg = math.degrees(yaw_rad) if math.isfinite(yaw_rad) else math.nan
        odom_yaw_deg = yaw_disagreement_to_odom(
            source,
            target,
            to_float(row.get("relative_yaw_rad")),
            nodes_by_state,
        )
        edge_inserted = is_true(row.get("edge_inserted"))
        if edge_inserted:
            accepted_count += 1
        if math.isfinite(translation):
            candidate_translation.append(translation)
        if math.isfinite(yaw_deg):
            candidate_yaw_deg.append(yaw_deg)
        if math.isfinite(odom_yaw_deg):
            candidate_odom_yaw_deg.append(odom_yaw_deg)
        # Descriptive audit flag only. It is not an online gate.
        if (
            edge_inserted
            and math.isfinite(translation)
            and math.isfinite(yaw_deg)
            and translation >= 0.75
            and yaw_deg >= 10.0
        ):
            large_prediction_disagreement_count += 1
        candidate_rows.append(
            {
                "bag": bag_dir.name,
                "stamp_ns": to_int(row.get("stamp_ns"), 0),
                "source_state_id": source,
                "target_state_id": target,
                "state_gap": abs(target - source),
                "chain_size": to_int(row.get("chain_size"), 0),
                "normalized_response": to_float(row.get("normalized_response")),
                "prediction_translation_disagreement_m": translation,
                "prediction_yaw_disagreement_deg": yaw_deg,
                "odom_yaw_disagreement_deg": odom_yaw_deg,
                "xy_covariance_max_eigenvalue_m2": to_float(
                    row.get("xy_covariance_max_eigenvalue_m2")
                ),
                "yaw_covariance_rad2": to_float(row.get("covariance_yawyaw")),
                "covariance_weighted_disagreement_sq": to_float(
                    row.get("scan_match_covariance_weighted_disagreement_sq")
                ),
                "memory_strength": to_float(row.get("memory_strength")),
                "edge_inserted": int(edge_inserted),
                "backend_action": row.get("backend_action", ""),
            }
        )

    corrections = [correction_values(node) for node in nodes]
    endpoint_xy, endpoint_yaw = corrections[-1] if corrections else (math.nan, math.nan)
    minimum_separation = max(
        minimum_state_separation,
        int(round(len(nodes) * minimum_separation_fraction)),
    )
    revisit = revisit_metrics(
        nodes,
        revisit_radius_m,
        minimum_separation,
    )
    metrics = DatasetMetrics(
        bag=bag_dir.name,
        node_count=len(nodes),
        edge_count=len(edges),
        adjacent_edge_count=edge_classes.count("adjacent"),
        near_chain_edge_count=edge_classes.count("near_chain"),
        unobserved_nonadjacent_edge_count=edge_classes.count(
            "unobserved_nonadjacent"
        ),
        nonadjacent_yaw_disagreement_gt_30_deg=sum(
            value > 30.0 for value in nonadjacent_yaw
        ),
        nonadjacent_yaw_disagreement_gt_80_deg=sum(
            value > 80.0 for value in nonadjacent_yaw
        ),
        endpoint_correction_xy_m=endpoint_xy,
        endpoint_correction_yaw_deg=endpoint_yaw,
        maximum_correction_xy_m=max(
            (value[0] for value in corrections if math.isfinite(value[0])),
            default=math.nan,
        ),
        maximum_correction_yaw_deg=max(
            (value[1] for value in corrections if math.isfinite(value[1])),
            default=math.nan,
        ),
        revisit_pair_count=revisit[0],
        revisit_graph_distance_median_m=revisit[1],
        revisit_graph_distance_p95_m=revisit[2],
        revisit_graph_distance_max_m=revisit[3],
        revisit_yaw_error_p95_deg=revisit[4],
        exact_candidate_count=len(candidates),
        accepted_near_chain_candidate_count=accepted_count,
        candidate_prediction_translation_p95_m=percentile(
            candidate_translation, 0.95
        ),
        candidate_prediction_yaw_p95_deg=percentile(candidate_yaw_deg, 0.95),
        candidate_odom_yaw_disagreement_p95_deg=percentile(
            candidate_odom_yaw_deg, 0.95
        ),
        candidate_large_prediction_disagreement_count=(
            large_prediction_disagreement_count
        ),
    )
    return metrics, edge_rows, candidate_rows


def write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_dataset(
    bag_dir: Path,
    candidate_rows: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = load_matplotlib()
    nodes = read_csv(bag_dir / "trajectory_pose_graph_final.csv")
    if not nodes:
        return
    plot_dir = output_dir / "plots" / bag_dir.name
    plot_dir.mkdir(parents=True, exist_ok=True)

    graph_x = [to_float(row.get("x_m")) for row in nodes]
    graph_y = [to_float(row.get("y_m")) for row in nodes]
    odom_x = [to_float(row.get("odom_x_m")) for row in nodes]
    odom_y = [to_float(row.get("odom_y_m")) for row in nodes]
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.scatter(odom_x, odom_y, s=8, alpha=0.65, label="wheel odom")
    axis.scatter(graph_x, graph_y, s=8, alpha=0.75, label="pose graph")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"{bag_dir.name}: pose graph vs odometry (points only)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "trajectory_pose_graph_vs_odom_points.png", dpi=180)
    plt.close(figure)

    state_ids = [to_int(row.get("state_id")) for row in nodes]
    corrections = [correction_values(row) for row in nodes]
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].scatter(state_ids, [value[0] for value in corrections], s=9)
    axes[0].set_ylabel("XY correction [m]")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(state_ids, [value[1] for value in corrections], s=9)
    axes[1].set_ylabel("absolute yaw correction [deg]")
    axes[1].set_xlabel("pose-graph state ID")
    axes[1].grid(True, alpha=0.25)
    figure.suptitle(f"{bag_dir.name}: graph correction relative to odometry")
    figure.tight_layout()
    figure.savefig(plot_dir / "graph_correction_by_state.png", dpi=180)
    plt.close(figure)

    finite_candidates = [
        row
        for row in candidate_rows
        if math.isfinite(row["prediction_yaw_disagreement_deg"])
    ]
    if finite_candidates:
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        targets = [row["target_state_id"] for row in finite_candidates]
        axes[0].scatter(
            targets,
            [row["normalized_response"] for row in finite_candidates],
            s=12,
        )
        axes[0].set_ylabel("response")
        axes[1].scatter(
            targets,
            [
                row["prediction_translation_disagreement_m"]
                for row in finite_candidates
            ],
            s=12,
        )
        axes[1].set_ylabel("prediction XY disagreement [m]")
        axes[2].scatter(
            targets,
            [row["prediction_yaw_disagreement_deg"] for row in finite_candidates],
            s=12,
        )
        axes[2].set_ylabel("prediction yaw disagreement [deg]")
        axes[2].set_xlabel("target state ID")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.suptitle(f"{bag_dir.name}: exact optional near-chain candidates")
        figure.tight_layout()
        figure.savefig(plot_dir / "near_chain_candidate_quality_by_state.png", dpi=180)
        plt.close(figure)


def plot_aggregate(
    metrics: Sequence[DatasetMetrics],
    candidates: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = load_matplotlib()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    ordered = sorted(metrics, key=lambda row: row.endpoint_correction_yaw_deg)
    figure, axis = plt.subplots(figsize=(12, max(6, 0.25 * len(ordered))))
    axis.barh(
        [row.bag for row in ordered],
        [row.endpoint_correction_yaw_deg for row in ordered],
    )
    axis.set_xlabel("absolute endpoint graph-to-odom yaw correction [deg]")
    axis.set_ylabel("dataset")
    axis.grid(True, axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "endpoint_yaw_correction_all_datasets.png", dpi=180)
    plt.close(figure)

    finite = [
        row
        for row in candidates
        if math.isfinite(row["normalized_response"])
        and math.isfinite(row["prediction_yaw_disagreement_deg"])
    ]
    if finite:
        figure, axis = plt.subplots(figsize=(10, 7))
        colors = [
            row["prediction_translation_disagreement_m"] for row in finite
        ]
        scatter = axis.scatter(
            [row["normalized_response"] for row in finite],
            [row["prediction_yaw_disagreement_deg"] for row in finite],
            c=colors,
            s=10,
            alpha=0.5,
            cmap="viridis",
        )
        axis.set_xlabel("normalized response")
        axis.set_ylabel("prediction yaw disagreement [deg]")
        axis.grid(True, alpha=0.25)
        colorbar = figure.colorbar(scatter, ax=axis)
        colorbar.set_label("prediction XY disagreement [m]")
        figure.tight_layout()
        figure.savefig(plot_dir / "candidate_response_vs_yaw_disagreement.png", dpi=180)
        plt.close(figure)


def report_text(metrics: Sequence[DatasetMetrics], variant_root: Path) -> str:
    catastrophic = [row for row in metrics if row.endpoint_correction_yaw_deg > 30.0]
    high_edges = [
        row for row in metrics if row.nonadjacent_yaw_disagreement_gt_30_deg > 0
    ]
    lines = [
        "# Pose-graph rotation constraint audit",
        "",
        f"- Variant: `{variant_root}`",
        f"- Datasets with exported graph evidence: {len(metrics)}",
        (
            "- Datasets with endpoint graph-to-odom yaw correction above 30 deg: "
            f"{len(catastrophic)}"
        ),
        (
            "- Datasets with a nonadjacent edge yaw disagreement above 30 deg "
            f"against the odometry proxy: {len(high_edges)}"
        ),
        "",
        "## Evidence limits",
        "",
        "- Wheel odometry is a comparison proxy, not ground truth.",
        "- Existing logs identify exact optional near-chain edges, but do not expose "
        "running-chain endpoints or accepted global-loop endpoint IDs.",
        "- Therefore remaining nonadjacent edges are labeled "
        "`unobserved_nonadjacent`, not `global_loop`.",
        "- The descriptive 0.75 m / 10 deg count is not an online rejection rule.",
        "",
        "## Highest endpoint yaw corrections",
        "",
        "| bag | yaw [deg] | XY [m] | near-chain | >30 deg nonadj edges |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        metrics,
        key=lambda item: item.endpoint_correction_yaw_deg,
        reverse=True,
    )[:12]:
        lines.append(
            f"| {row.bag} | {row.endpoint_correction_yaw_deg:.3f} | "
            f"{row.endpoint_correction_xy_m:.3f} | {row.near_chain_edge_count} | "
            f"{row.nonadjacent_yaw_disagreement_gt_30_deg} |"
        )
    lines.extend(
        [
            "",
            "## Main interpretation",
            "",
            "The exported graph can distinguish a catastrophic rotation event from "
            "ordinary sub-meter return mismatch. The exact candidate trace must be "
            "used before changing global loop-search angles: a bad optional near-chain "
            "factor changes the current scan pose before Karto attempts global loop "
            "closure.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.odom_revisit_radius_m <= 0.0:
        raise ValueError("--odom-revisit-radius-m must be positive")
    if args.minimum_state_separation < 1:
        raise ValueError("--minimum-state-separation must be positive")
    if not 0.0 <= args.minimum_separation_fraction <= 1.0:
        raise ValueError("--minimum-separation-fraction must be in [0, 1]")

    bag_root = args.variant_root / "bags"
    bag_dirs = sorted(path for path in bag_root.iterdir() if path.is_dir())
    metrics: list[DatasetMetrics] = []
    all_edges: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    selected_plots = set(args.plot_bag)
    for bag_dir in bag_dirs:
        if not (bag_dir / "trajectory_pose_graph_final.csv").is_file():
            continue
        if not (bag_dir / "pose_graph_edges_final.csv").is_file():
            continue
        candidate_path = candidate_log_path(args.variant_root, bag_dir.name)
        dataset, edges, candidates = analyze_dataset(
            bag_dir,
            candidate_path,
            args.odom_revisit_radius_m,
            args.minimum_state_separation,
            args.minimum_separation_fraction,
        )
        metrics.append(dataset)
        all_edges.extend(edges)
        all_candidates.extend(candidates)
        if bag_dir.name in selected_plots:
            plot_dataset(bag_dir, candidates, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = [asdict(row) for row in metrics]
    write_rows(args.output_dir / "csv" / "dataset_metrics.csv", metric_rows)
    write_rows(args.output_dir / "csv" / "edge_metrics.csv", all_edges)
    write_rows(
        args.output_dir / "csv" / "near_chain_candidate_metrics.csv",
        all_candidates,
    )
    plot_aggregate(metrics, all_candidates, args.output_dir)
    (args.output_dir / "report").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report" / "rotation_constraint_audit.md").write_text(
        report_text(metrics, args.variant_root),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "variant_root": str(args.variant_root),
        "dataset_count": len(metrics),
        "edge_count": len(all_edges),
        "candidate_count": len(all_candidates),
        "endpoint_yaw_gt_30_dataset_count": sum(
            row.endpoint_correction_yaw_deg > 30.0 for row in metrics
        ),
        "nonadjacent_yaw_gt_30_dataset_count": sum(
            row.nonadjacent_yaw_disagreement_gt_30_deg > 0 for row in metrics
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
