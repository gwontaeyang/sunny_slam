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

"""Small, dependency-light PCD readers and writers used by the projector."""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


_NUMPY_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def _read_header(stream) -> Tuple[Dict[str, List[str]], str]:
    header: Dict[str, List[str]] = {}
    while True:
        raw_line = stream.readline()
        if not raw_line:
            raise ValueError("PCD header ended before DATA")
        try:
            line = raw_line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PCD header is not ASCII") from error
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        key = tokens[0].upper()
        header[key] = tokens[1:]
        if key == "DATA":
            if not header[key]:
                raise ValueError("PCD DATA entry is empty")
            return header, header[key][0].lower()


def _pcd_layout(header: Dict[str, List[str]]) -> Tuple[np.dtype, Sequence[str]]:
    fields = header.get("FIELDS") or header.get("FIELD")
    if not fields:
        raise ValueError("PCD header has no FIELDS entry")
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = [value.upper() for value in header.get("TYPE", [])]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT lengths differ")

    dtype_fields = []
    scalar_names = []
    for name, size, value_type, count in zip(fields, sizes, types, counts):
        numpy_type = _NUMPY_TYPES.get((value_type, size))
        if numpy_type is None:
            raise ValueError(f"unsupported PCD type {value_type}{size} for field {name}")
        if count <= 0:
            raise ValueError(f"invalid COUNT for PCD field {name}")
        if count == 1:
            dtype_fields.append((name, numpy_type))
            scalar_names.append(name)
        else:
            dtype_fields.append((name, numpy_type, (count,)))
            scalar_names.extend(f"{name}_{index}" for index in range(count))
    return np.dtype(dtype_fields), scalar_names


def read_pcd_xyz(path: Path) -> np.ndarray:
    """Read finite XYZ points from an ASCII or binary PCD file."""
    with path.open("rb") as stream:
        header, data_mode = _read_header(stream)
        dtype, scalar_names = _pcd_layout(header)
        point_count = int((header.get("POINTS") or ["0"])[0])
        if point_count <= 0:
            width = int((header.get("WIDTH") or ["0"])[0])
            height = int((header.get("HEIGHT") or ["1"])[0])
            point_count = width * height
        if point_count <= 0:
            raise ValueError("PCD point count must be positive")

        if data_mode == "binary":
            expected_bytes = point_count * dtype.itemsize
            payload = stream.read(expected_bytes)
            if len(payload) != expected_bytes:
                raise ValueError(
                    f"binary PCD payload is truncated: {len(payload)} != {expected_bytes}"
                )
            records = np.frombuffer(payload, dtype=dtype, count=point_count)
            try:
                points = np.column_stack([records[name] for name in ("x", "y", "z")])
            except ValueError as error:
                raise ValueError("PCD must contain scalar x, y, and z fields") from error
        elif data_mode == "ascii":
            values = np.fromstring(stream.read().decode("ascii"), sep=" ")
            scalar_count = len(scalar_names)
            if values.size != point_count * scalar_count:
                raise ValueError("ASCII PCD payload size does not match its header")
            values = values.reshape(point_count, scalar_count)
            try:
                xyz_indices = [scalar_names.index(name) for name in ("x", "y", "z")]
            except ValueError as error:
                raise ValueError("PCD must contain scalar x, y, and z fields") from error
            points = values[:, xyz_indices]
        elif data_mode == "binary_compressed":
            raise ValueError("binary_compressed PCD is not supported; convert it to binary first")
        else:
            raise ValueError(f"unsupported PCD DATA mode: {data_mode}")

    points = np.asarray(points, dtype=np.float64)
    return points[np.isfinite(points).all(axis=1)]


def read_binary_pcd_records(path: Path, required_fields: Sequence[str] = ()) -> np.ndarray:
    """Read a binary PCD while preserving all scalar field names and types."""
    with path.open("rb") as stream:
        header, data_mode = _read_header(stream)
        if data_mode != "binary":
            raise ValueError("structured PCD input must use DATA binary")
        dtype, _ = _pcd_layout(header)
        if any(dtype[name].shape for name in dtype.names or ()):
            raise ValueError("structured PCD input must contain only scalar fields")
        missing = sorted(set(required_fields) - set(dtype.names or ()))
        if missing:
            raise ValueError(f"PCD is missing required fields: {', '.join(missing)}")
        point_count = int((header.get("POINTS") or ["0"])[0])
        if point_count < 0:
            raise ValueError("PCD point count cannot be negative")
        expected_bytes = point_count * dtype.itemsize
        payload = stream.read(expected_bytes)
        if len(payload) != expected_bytes:
            raise ValueError(
                f"binary PCD payload is truncated: {len(payload)} != {expected_bytes}"
            )
        if stream.read(1):
            raise ValueError("binary PCD payload contains trailing bytes")
    return np.frombuffer(payload, dtype=dtype, count=point_count).copy()


def write_xyzrgb_pcd(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary XYZRGB PCD that is directly viewable in pcl_viewer."""
    points = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if rgb.shape != points.shape:
        raise ValueError("rgb must have shape (N, 3)")

    packed_rgb = (
        rgb[:, 0].astype(np.uint32) << 16
        | rgb[:, 1].astype(np.uint32) << 8
        | rgb[:, 2].astype(np.uint32)
    )
    records = np.empty(
        points.shape[0],
        dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]),
    )
    records["x"] = points[:, 0]
    records["y"] = points[:, 1]
    records["z"] = points[:, 2]
    records["rgb"] = packed_rgb.view(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(records)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(records)}\n"
        "DATA binary\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())
