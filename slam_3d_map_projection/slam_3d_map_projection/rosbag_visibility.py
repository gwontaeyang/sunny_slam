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

"""ROS 2 bag adapter for the pure visibility evidence core."""

from pathlib import Path
from typing import Optional

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs_py import point_cloud2

from .trajectory_io import TrajectoryData
from .visibility import (
    ObstacleCandidates,
    RawVisibilityConfig,
    VisibilityAccumulator,
    VisibilityEvidence,
)


def accumulate_bag_visibility(
    bag_path: Path,
    candidates: ObstacleCandidates,
    trajectory: TrajectoryData,
    config: RawVisibilityConfig,
    max_scans: Optional[int] = None,
    progress_interval: int = 100,
) -> VisibilityEvidence:
    """Read one PointCloud2 topic and accumulate candidate visibility states."""
    if max_scans is not None and max_scans <= 0:
        raise ValueError("max_scans must be positive when provided")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topic_types = {
        item.name: item.type for item in reader.get_all_topics_and_types()
    }
    if config.topic not in topic_types:
        raise ValueError(f"raw bag does not contain topic {config.topic}")
    message_type = get_message(topic_types[config.topic])
    accumulator = VisibilityAccumulator(candidates, trajectory, config)
    point_time_spans_s = []
    header_to_first_point_s = []

    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic != config.topic:
            continue
        if max_scans is not None and accumulator.raw_scan_count >= max_scans:
            break
        message = deserialize_message(serialized, message_type)
        records = point_cloud2.read_points(
            message,
            field_names=["x", "y", "z", "ring", "timestamp"],
            skip_nans=False,
        )
        points_sensor = np.column_stack(
            [records[name] for name in ("x", "y", "z")]
        ).astype(np.float64, copy=False)
        rings = np.asarray(records["ring"])
        timestamps_s = np.asarray(records["timestamp"], dtype=np.float64)
        header_timestamp_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        finite_time = np.isfinite(timestamps_s)
        if np.any(finite_time):
            point_time_spans_s.append(
                float(
                    np.max(timestamps_s[finite_time])
                    - np.min(timestamps_s[finite_time])
                )
            )
            header_to_first_point_s.append(
                float(header_timestamp_s - np.min(timestamps_s[finite_time]))
            )
        accumulator.observe_scan(
            points_sensor,
            rings,
            timestamps_s,
            header_timestamp_s,
            message.header.frame_id,
        )
        if (
            progress_interval > 0
            and accumulator.raw_scan_count % progress_interval == 0
        ):
            print(
                "raw visibility: "
                f"{accumulator.raw_scan_count} scans read, "
                f"{accumulator.accepted_scan_count} accepted",
                flush=True,
            )

    evidence = accumulator.finish()
    evidence.scan_metrics.update(
        {
            "bag_path": str(bag_path),
            "topic": config.topic,
            "expected_frame_id": config.expected_frame_id,
            "point_time_span_s_median": (
                float(np.median(point_time_spans_s))
                if point_time_spans_s
                else None
            ),
            "point_time_span_s_p95": (
                float(np.percentile(point_time_spans_s, 95))
                if point_time_spans_s
                else None
            ),
            "header_to_first_point_s_median": (
                float(np.median(header_to_first_point_s))
                if header_to_first_point_s
                else None
            ),
        }
    )
    return evidence
