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

"""Contracts for continuous terrain support projection."""

from pathlib import Path

import numpy as np

from slam_3d_map_projection.pcd_io import read_binary_pcd_records
from slam_3d_map_projection.projection import GridClass, GridGeometry
from slam_3d_map_projection.support_projection import (
    TerrainSupportState,
    project_support_evidence,
    write_support_grid_outputs,
    write_support_review_figure,
)


SUPPORT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("rgb", "<f4"),
        ("support_probability", "<f4"),
        ("geometric_traversability_score", "<f4"),
        ("label_confidence", "<f4"),
        ("roughness_score", "<f4"),
        ("label", "u1"),
        ("state", "u1"),
        ("connectivity_rejected", "u1"),
        ("confirmations", "<u4"),
        ("observations", "<u4"),
    ]
)


def _records() -> np.ndarray:
    records = np.zeros(6, dtype=SUPPORT_DTYPE)
    records["x"] = [0.15, 0.25, 0.35, 0.15, 0.25, 0.35]
    records["y"] = [0.15, 0.15, 0.15, 0.25, 0.25, 0.35]
    records["support_probability"] = [1.0, 1.0, 0.6, 0.0, 0.0, 1.0]
    records["geometric_traversability_score"] = [0.9, 0.5, 0.0, 0.0, 0.0, 0.8]
    records["label"] = [1, 1, 1, 3, 4, 1]
    records["state"] = [2, 1, 0, 3, 4, 2]
    records["confirmations"] = [2, 1, 1, 1, 2, 2]
    records["observations"] = [2, 1, 2, 1, 2, 2]
    return records


def _write_support_pcd(path: Path, records: np.ndarray) -> None:
    fields = list(records.dtype.names or ())
    sizes = [records.dtype[name].itemsize for name in fields]
    types = [
        "F" if records.dtype[name].kind == "f" else "U"
        for name in fields
    ]
    header = (
        "VERSION 0.7\n"
        f"FIELDS {' '.join(fields)}\n"
        f"SIZE {' '.join(str(value) for value in sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join('1' for _ in fields)}\n"
        f"WIDTH {len(records)}\n"
        "HEIGHT 1\n"
        f"POINTS {len(records)}\n"
        "DATA binary\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def test_binary_reader_preserves_support_fields(tmp_path):
    path = tmp_path / "support.pcd"
    expected = _records()
    _write_support_pcd(path, expected)

    observed = read_binary_pcd_records(
        path,
        required_fields=("support_probability", "state", "observations"),
    )

    assert observed.dtype == expected.dtype
    np.testing.assert_array_equal(observed, expected)


def test_soft_support_cost_is_separate_from_occupancy(tmp_path):
    geometry = GridGeometry(0.0, 0.0, 0.1, 5, 5)
    occupancy = np.full((5, 5), GridClass.UNKNOWN, dtype=np.int16)
    occupancy[3, 3] = GridClass.OCCUPIED

    result = project_support_evidence(
        _records(),
        geometry,
        occupancy,
        evidence_radius_m=0.0,
        probable_support_min_cost=0.35,
    )

    assert result.support_state[1, 1] == TerrainSupportState.CONFIRMED_SUPPORT
    assert np.isclose(result.traversal_cost[1, 1], 0.1)
    assert result.support_state[1, 2] == TerrainSupportState.PROBABLE_SUPPORT
    assert np.isclose(result.traversal_cost[1, 2], 0.5)
    assert occupancy[1, 2] == GridClass.UNKNOWN
    assert result.support_state[1, 3] == TerrainSupportState.UNRESOLVED
    assert np.isnan(result.traversal_cost[1, 3])
    assert result.support_state[2, 1] == TerrainSupportState.PROBABLE_HAZARD
    assert result.traversal_cost[2, 1] == 1.0
    assert not result.hard_hazard_mask[2, 1]
    assert result.support_state[2, 2] == TerrainSupportState.CONFIRMED_HAZARD
    assert result.hard_hazard_mask[2, 2]
    assert result.hard_hazard_mask[3, 3]
    assert result.traversal_cost[3, 3] == 1.0
    assert result.metrics["support_cells_overridden_by_hard_hazard"] == 1
    assert np.isnan(result.traversal_cost[4, 4])

    outputs = write_support_grid_outputs(tmp_path, "room", result)
    with np.load(outputs["support_layers"]) as layers:
        np.testing.assert_array_equal(layers["support_state"], result.support_state)
        np.testing.assert_allclose(
            layers["traversal_cost"], result.traversal_cost, equal_nan=True
        )
    with outputs["traversability_cost"].open("rb") as stream:
        assert stream.readline().strip() == b"P5"
        assert stream.readline().strip() == b"5 5"
        assert stream.readline().strip() == b"255"
        pixels = np.frombuffer(stream.read(), dtype=np.uint8).reshape(5, 5)
    pixels = np.flipud(pixels)
    assert pixels[1, 1] == round(0.1 * 252)
    assert pixels[1, 3] == 255
    assert pixels[2, 2] == 254

    review_path = tmp_path / "room_support_review.png"
    write_support_review_figure(review_path, "synthetic room", result)
    assert review_path.stat().st_size > 10_000
