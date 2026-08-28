#!/usr/bin/env python3
"""Build the validated Site C Q1/Q3 LaserScan shards in one source pass."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Deque, Iterable, Sequence

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import LaserScan
from sensor_msgs_py import point_cloud2
import yaml


NS = 1_000_000_000
SOURCE_SCAN_TOPIC = "/scan"
OUTPUT_SCAN_TOPIC = "/scan_laserscan"
STATIC_TOPIC = "/tf_static"
SELECTED_NON_SCAN_TOPICS = {
    "/tf",
    STATIC_TOPIC,
    "/odom",
    "/motor_driver_node/odom",
    "/imu/data",
    "/imu/data_raw",
}

ANGLE_MIN = -math.pi
ANGLE_MAX = math.pi
ANGLE_INCREMENT = 0.001745
RANGE_MIN = 0.05
RANGE_MAX = 50.0

# Frozen from the result-blind Site C segmentation report. End indices are
# exclusive. Q2/Q4 are intentionally omitted from the requested regression.
DEFAULT_SEGMENTS = ("site_c_q1:0:8399", "site_c_q3:14133:19867")
HISTORICAL_TOPIC_COUNTS = {
    "site_c_q1": {
        "/imu/data": 67136,
        "/motor_driver_node/odom": 13431,
        "/odom": 16784,
        OUTPUT_SCAN_TOPIC: 8399,
        "/tf": 46567,
        STATIC_TOPIC: 6,
    },
    "site_c_q3": {
        "/imu/data": 46268,
        "/motor_driver_node/odom": 9253,
        "/odom": 11559,
        OUTPUT_SCAN_TOPIC: 5734,
        "/tf": 31776,
        STATIC_TOPIC: 6,
    },
}
HISTORICAL_NOMINAL_BOUNDARIES_NS = {
    "site_c_q1": (1785821483679498517, 1785821819611225179),
    "site_c_q3": (1785822048951928255, 1785822278291358295),
}


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    scan_start: int
    scan_end_exclusive: int


@dataclass
class SegmentState:
    spec: SegmentSpec
    path: Path
    writer: rosbag2_py.SequentialWriter
    counts: dict[str, int]
    active: bool = False
    nominal_start_ns: int | None = None
    nominal_end_ns_exclusive: int | None = None
    first_written_ns: int | None = None
    last_written_ns: int | None = None
    context_message_count: int = 0
    static_context_message_count: int = 0

    def write(self, topic: str, data: bytes, timestamp: int) -> None:
        self.writer.write(topic, data, timestamp)
        self.counts[topic] += 1
        if self.first_written_ns is None:
            self.first_written_ns = timestamp
        self.last_written_ns = timestamp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def storage_id(path: Path) -> str:
    metadata_path = path / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = metadata.get("rosbag2_bagfile_information", metadata)
    return str(info.get("storage_identifier", "sqlite3"))


def parse_segment(value: str) -> SegmentSpec:
    fields = value.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "segment must be NAME:SCAN_START:SCAN_END_EXCLUSIVE"
        )
    name = fields[0].strip()
    try:
        start = int(fields[1])
        end = int(fields[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("segment indices must be integers") from error
    if not name or "/" in name:
        raise argparse.ArgumentTypeError("segment name must be a plain directory name")
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError("segment must satisfy 0 <= start < end")
    return SegmentSpec(name=name, scan_start=start, scan_end_exclusive=end)


def validate_segments(segments: Sequence[SegmentSpec]) -> list[SegmentSpec]:
    if not segments:
        raise ValueError("at least one segment is required")
    ordered = sorted(segments, key=lambda item: item.scan_start)
    if len({segment.name for segment in ordered}) != len(ordered):
        raise ValueError("segment names must be unique")
    for previous, current in zip(ordered, ordered[1:]):
        if previous.scan_end_exclusive > current.scan_start:
            raise ValueError("segment scan ranges must not overlap")
    return ordered


def project_cloud(message: Any) -> LaserScan:
    """Apply the previously validated deployed azimuth-bin projection."""
    scan = LaserScan()
    scan.header = message.header
    scan.header.frame_id = "base_link"
    scan.angle_min = ANGLE_MIN
    scan.angle_max = ANGLE_MAX
    scan.angle_increment = ANGLE_INCREMENT
    scan.range_min = RANGE_MIN
    scan.range_max = RANGE_MAX

    angle_min = np.float32(ANGLE_MIN)
    angle_max = np.float32(ANGLE_MAX)
    angle_increment = np.float32(ANGLE_INCREMENT)
    range_min = np.float32(RANGE_MIN)
    range_max = np.float32(RANGE_MAX)
    bin_count = int(math.floor(float((angle_max - angle_min) / angle_increment))) + 1
    ranges = np.full(bin_count, np.inf, dtype=np.float32)

    xyz = point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"])
    finite = np.isfinite(xyz).all(axis=1)
    xy = xyz[finite, :2].astype(np.float32, copy=False)
    radius = np.hypot(xy[:, 0], xy[:, 1], dtype=np.float32)
    angle = np.arctan2(xy[:, 1], xy[:, 0], dtype=np.float32)
    keep = (
        (radius >= range_min)
        & (radius <= range_max)
        & (angle >= angle_min)
        & (angle <= angle_max)
    )
    bins = np.floor((angle[keep] - angle_min) / angle_increment).astype(np.int64)
    values = radius[keep]
    valid_bins = (bins >= 0) & (bins < bin_count)
    np.minimum.at(ranges, bins[valid_bins], values[valid_bins])
    scan.ranges = ranges.tolist()
    return scan


def open_reader(path: Path, topics: Iterable[str]) -> rosbag2_py.SequentialReader:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id=storage_id(path)),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(topics)))
    return reader


def create_states(
    output_root: Path,
    specs: Sequence[SegmentSpec],
    source_topics: dict[str, Any],
) -> list[SegmentState]:
    available_non_scan = sorted(SELECTED_NON_SCAN_TOPICS & source_topics.keys())
    output_topics = available_non_scan + [OUTPUT_SCAN_TOPIC]
    states: list[SegmentState] = []
    for spec in specs:
        path = output_root / spec.name
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        for topic in available_non_scan:
            writer.create_topic(source_topics[topic])
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=OUTPUT_SCAN_TOPIC,
                type="sensor_msgs/msg/LaserScan",
                serialization_format="cdr",
            )
        )
        states.append(
            SegmentState(
                spec=spec,
                path=path,
                writer=writer,
                counts={topic: 0 for topic in output_topics},
                active=spec.scan_start == 0,
            )
        )
    return states


def historical_count_check(name: str, counts: dict[str, int]) -> dict[str, Any]:
    expected = HISTORICAL_TOPIC_COUNTS.get(name)
    if expected is None:
        return {"status": "not_available", "expected": {}, "mismatches": {}}
    mismatches = {
        topic: {"expected": count, "actual": counts.get(topic, 0)}
        for topic, count in expected.items()
        if counts.get(topic, 0) != count
    }
    return {
        "status": "matched" if not mismatches else "mismatched",
        "expected": expected,
        "mismatches": mismatches,
    }


def historical_non_scan_timestamp_allowed(name: str, timestamp: int) -> bool:
    boundaries = HISTORICAL_NOMINAL_BOUNDARIES_NS.get(name)
    return boundaries is None or boundaries[0] <= timestamp < boundaries[1]


def historical_static_timestamp_allowed(name: str, timestamp: int) -> bool:
    boundaries = HISTORICAL_NOMINAL_BOUNDARIES_NS.get(name)
    return boundaries is None or timestamp < boundaries[1]


def build(
    source_bag: Path,
    output_root: Path,
    specs: Sequence[SegmentSpec],
    startup_context_s: float,
) -> dict[str, Any]:
    specs = validate_segments(specs)
    if startup_context_s < 0.0:
        raise ValueError("startup context must be non-negative")
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)

    probe = open_reader(source_bag, [])
    source_topics = {topic.name: topic for topic in probe.get_all_topics_and_types()}
    del probe
    if SOURCE_SCAN_TOPIC not in source_topics:
        raise RuntimeError(f"source bag has no {SOURCE_SCAN_TOPIC}")
    available_non_scan = SELECTED_NON_SCAN_TOPICS & source_topics.keys()
    states = create_states(output_root, specs, source_topics)
    state_by_start = {state.spec.scan_start: state for state in states}
    state_by_end = {state.spec.scan_end_exclusive: state for state in states}

    reader = open_reader(source_bag, available_non_scan | {SOURCE_SCAN_TOPIC})
    cloud_type = get_message(source_topics[SOURCE_SCAN_TOPIC].type)
    context_ns = int(round(startup_context_s * NS))
    context_buffer: Deque[tuple[str, bytes, int]] = deque()
    static_messages: list[bytes] = []
    scan_index = 0
    source_first_stamp_ns: int | None = None
    source_last_stamp_ns: int | None = None
    max_end = max(spec.scan_end_exclusive for spec in specs)

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if source_first_stamp_ns is None:
            source_first_stamp_ns = timestamp
        source_last_stamp_ns = timestamp

        if topic != SOURCE_SCAN_TOPIC:
            if topic == STATIC_TOPIC:
                static_messages.append(data)
                for state in states:
                    if (
                        state.active
                        and state.spec.scan_start == 0
                        and historical_static_timestamp_allowed(
                            state.spec.name, timestamp
                        )
                    ):
                        state.write(topic, data, timestamp)
                continue

            context_buffer.append((topic, data, timestamp))
            while context_buffer and timestamp - context_buffer[0][2] > context_ns:
                context_buffer.popleft()
            for state in states:
                if state.active and historical_non_scan_timestamp_allowed(
                    state.spec.name, timestamp
                ):
                    state.write(topic, data, timestamp)
            continue

        ending = state_by_end.get(scan_index)
        if ending is not None:
            ending.nominal_end_ns_exclusive = timestamp
            ending.active = False
            if scan_index >= max_end:
                break

        starting = state_by_start.get(scan_index)
        if starting is not None:
            starting.nominal_start_ns = timestamp
            if starting.spec.scan_start != 0:
                context_start_ns = timestamp - context_ns
                for static_data in static_messages:
                    starting.write(STATIC_TOPIC, static_data, context_start_ns)
                    starting.static_context_message_count += 1
                for buffered_topic, buffered_data, buffered_stamp in context_buffer:
                    if context_start_ns <= buffered_stamp < timestamp:
                        starting.write(buffered_topic, buffered_data, buffered_stamp)
                        starting.context_message_count += 1
            starting.active = True

        active_states = [state for state in states if state.active]
        if active_states:
            cloud = deserialize_message(data, cloud_type)
            serialized_scan = serialize_message(project_cloud(cloud))
            for state in active_states:
                state.write(OUTPUT_SCAN_TOPIC, serialized_scan, timestamp)
        scan_index += 1

    if scan_index < max_end:
        raise RuntimeError(
            f"source ended before scan index {max_end}: observed {scan_index} scans"
        )

    rows = []
    for state in states:
        if state.nominal_start_ns is None or state.nominal_end_ns_exclusive is None:
            raise RuntimeError(f"segment boundaries were not observed: {state.spec.name}")
        rows.append(
            {
                "name": state.spec.name,
                "path": str(state.path),
                "scan_index_start": state.spec.scan_start,
                "scan_index_end_exclusive": state.spec.scan_end_exclusive,
                "nominal_start_ns": state.nominal_start_ns,
                "nominal_end_ns_exclusive": state.nominal_end_ns_exclusive,
                "written_start_ns": state.first_written_ns,
                "written_end_ns": state.last_written_ns,
                "written_duration_s": (
                    (state.last_written_ns - state.first_written_ns) / NS
                    if state.first_written_ns is not None
                    and state.last_written_ns is not None
                    else 0.0
                ),
                "context_message_count": state.context_message_count,
                "static_context_message_count": state.static_context_message_count,
                "topic_counts": state.counts,
                "historical_topic_count_contract": historical_count_check(
                    state.spec.name, state.counts
                ),
            }
        )

    # Closing writers flushes SQLite and metadata.yaml before consumers start.
    for state in states:
        state.writer = None  # type: ignore[assignment]
    states.clear()

    metadata_path = source_bag / "metadata.yaml"
    database_files = sorted(source_bag.glob("*.db3"))
    return {
        "schema": 1,
        "source_bag": str(source_bag),
        "source_bag_modified": False,
        "source_metadata_sha256": sha256_file(metadata_path),
        "source_database_files": [
            {"path": str(path), "size_bytes": path.stat().st_size}
            for path in database_files
        ],
        "output_root": str(output_root),
        "startup_context_s": startup_context_s,
        "projection": {
            "angle_min_rad": ANGLE_MIN,
            "angle_max_rad": ANGLE_MAX,
            "angle_increment_rad": ANGLE_INCREMENT,
            "range_min_m": RANGE_MIN,
            "range_max_m": RANGE_MAX,
            "height_filter": False,
            "output_frame": "base_link",
            "binning": "nearest range per floor-indexed azimuth bin",
        },
        "source_first_read_stamp_ns": source_first_stamp_ns,
        "source_last_read_stamp_ns": source_last_stamp_ns,
        "source_scans_read_through_boundary": scan_index,
        "segments": rows,
        "all_historical_topic_count_contracts_match": all(
            row["historical_topic_count_contract"]["status"] == "matched"
            for row in rows
            if row["name"] in HISTORICAL_TOPIC_COUNTS
        ),
        "command": " ".join(sys.argv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact Site C Q1/Q3 LaserScan shards without a full derived bag."
    )
    parser.add_argument("--source-bag", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--segment",
        action="append",
        type=parse_segment,
        help="NAME:SCAN_START:SCAN_END_EXCLUSIVE; repeat for multiple shards",
    )
    parser.add_argument("--startup-context-s", type=float, default=2.0)
    args = parser.parse_args()
    segments = args.segment or [parse_segment(value) for value in DEFAULT_SEGMENTS]
    report = build(
        source_bag=args.source_bag,
        output_root=args.output_root,
        specs=segments,
        startup_context_s=args.startup_context_s,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_historical_topic_count_contracts_match"]:
        raise SystemExit("Site C shard counts differ from the frozen historical contract")


if __name__ == "__main__":
    main()
