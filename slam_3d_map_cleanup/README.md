# slam_3d_map_cleanup

Independent, offline temporal map cleanup. `temporal_map_cleanup` is a CLI,
not a ROS node: no subscriptions, services, TF lookup, bag replay, hardware,
or runtime dependency on `slam_3d_core`. The library and CLI implementation
are separate from the thin executable entry point.

The temporal accumulator and its original 15 regression tests were copied
from `slam_3d_core`. Only the namespace/header path and default configuration
changed. Hit/miss counting, ray traversal, prior-free evidence, promotion,
retention, and demotion numerics are unchanged. No additional filtering
heuristics or semantic model are introduced.

## Limits

A single merged PCD is insufficient: it has lost observation timestamps,
keyframe boundaries, sensor viewpoints, and free-space ray history. Use the
keyframe archive, not the core's exported raw map PCD.

This is **not complete semantic human recognition**. Unconfirmed evidence
can include people, other moving objects, and static surfaces seen too few
times. A stationary person can become confirmed; sparsely observed walls
can remain unconfirmed. Validate the cleaned map before downstream use.

## Build and Run

Dependencies: `ament_cmake`, Eigen3, PCL `common`/`io`, and yaml-cpp. Tests
also use `ament_cmake_gtest`, Python 3, and PyYAML. No ROS client library is
linked. With these dependencies already available:

```bash
cmake -S /path/to/slam_3d_map_cleanup -B /tmp/slam_3d_map_cleanup_build \
  -DCMAKE_PREFIX_PATH=/opt/ros/humble -DPython3_EXECUTABLE=/usr/bin/python3 \
  -DBUILD_TESTING=ON
cmake --build /tmp/slam_3d_map_cleanup_build -j2
ctest --test-dir /tmp/slam_3d_map_cleanup_build --output-on-failure

/tmp/slam_3d_map_cleanup_build/temporal_map_cleanup \
  --input /path/to/archive \
  --output-dir /path/to/new_cleanup_result \
  --config /path/to/slam_3d_map_cleanup/config/indoor.yaml
```

`--config` is optional. `--help` prints usage. Exit status is 0 for success,
2 for argument errors, and 1 for invalid data/configuration or IO failures.
The executable installs to `bin/temporal_map_cleanup`. This package also
supports ordinary `colcon build --packages-select slam_3d_map_cleanup`.

## Archive Contract: Schema 1

`ARCHIVE_DIR/manifest.yaml` is one YAML document:

```yaml
schema_version: 1
snapshot_scope: retained_keyframes
pruned_keyframes: 0
complete_session: true
point_frame: base_frame
point_source: deskewed_mapping_only_before_terrain_classification
map_frame: map
base_frame: base_link
map_z_filter:
  enabled: true
  min_m: -1.0
  max_m: 2.0
keyframes:
  - id: 0
    stamp_ns: 1700000000000000000
    pose: [0, 0, 0, 0, 0, 0, 1]
    scans:
      - pcd: keyframes/000000_sensor00.pcd
        origin_base: [0.2, 0.0, 0.8]
      - pcd: keyframes/000000_sensor01.pcd
        origin_base: [-0.2, 0.0, 0.8]
```

- Units are meters and nanoseconds. `pose` is
  `[tx, ty, tz, qx, qy, qz, qw]`, mapping **base -> map**. Use the final
  post-loop-optimization pose snapshot for every keyframe. A fresh run
  reprojects all observations; no accumulated pre-loop map is reused.
- Each PCD contains deskewed, mapping-only XYZ in **base_frame**, not sensor
  frame and not map frame. Do not apply sensor extrinsics again. Scalar
  FLOAT32 or FLOAT64 XYZ fields are accepted through PCL IO; other fields
  are ignored. PCD VIEWPOINT metadata does not override `origin_base` or pose.
- Both endpoints and each `origin_base` are transformed with the same pose.
  Inclusive `min_m <= world_z <= max_m` filtering applies to endpoints after
  transformation. Removed endpoints also provide no rays. Sensor origins
  are not clipped to the Z band, and retained rays use the original algorithm.
- All sensor views in a merged keyframe enter **one** observation. Duplicate
  voxel hits or ray misses across sensors cannot count as multiple observations.
- Keyframes must be nonempty and listed in strictly increasing `id` **and**
  `stamp_ns` order. IDs are unsigned 64-bit decimal integers (gaps allowed);
  stamps are signed 64-bit decimal integers. The CLI does not silently sort.
  Timestamps are converted to seconds relative to the first stamp to preserve
  small intervals at large epoch timestamps without int64 subtraction overflow.
- Frames must be nonempty. Poses, origins, XYZ, and Z bounds must be finite.
  Quaternions must have unit norm within `1e-6`; only accepted roundoff is
  normalized. Reversed Z bounds are invalid even when filtering is disabled.
  Coordinates must fit the accumulator's int64 voxel keys and FLOAT32 output.
- The five snapshot metadata fields shown above are optional for older schema-1
  archives. When present, their types/declared scope and point conventions are
  validated and preserved in both `stats.yaml` and the result manifest.
  `pruned_keyframes` is uint64; when both completeness fields are supplied,
  `complete_session` must equal `(pruned_keyframes == 0)`, matching the writer.
  Contradictions are rejected. Pruned or incomplete snapshots emit a
  CLI warning and are explicitly not full-session maps. Missing completeness
  metadata is recorded as null/unknown, never inferred to mean complete.
  The preserved `point_frame` describes archive input; `output_point_frame`
  explicitly identifies the output PCD frame, equal to `map_frame`.
- Each keyframe needs at least one scan, each input scan needs points, and
  repeated PCD paths within one keyframe are rejected. A view or entire map
  becoming empty **after** the Z filter is valid.
- PCD paths must be relative and resolve to regular files inside the archive.
  Missing files, escaping symlinks/paths, unsupported schemas, missing fields,
  duplicate/unknown YAML keys, and multiple YAML documents fail closed.
- Finish exporting the archive before cleanup and do not mutate it during a
  run. The reader is not a snapshot or a security boundary against concurrent
  filesystem mutation. An archive writer should publish its manifest last.

## Configuration

Plain YAML uses exactly the `TemporalStaticMapConfig` member names, with no
ROS parameter wrapper. Partial mappings override defaults; `{}` uses defaults.
Unknown/duplicate keys, invalid types, nonfinite numbers, invalid ranges, and
unrepresentable traversal sizes are errors. See `config/indoor.yaml` for the
complete baseline:

```yaml
enabled: true
voxel_size_m: 0.22
min_hit_observations: 3
min_observation_span_s: 1.0
min_viewpoint_distance_m: 0.6
ray_clearing_enabled: true
ray_endpoint_margin_m: 0.45
max_ray_length_m: 30.0
max_rays_per_observation: 2000
prior_free_enabled: true
min_prior_free_observations: 4
max_prior_free_observations: 8
prior_free_max_sensor_vertical_drop_m: 0.9
promotion_min_hit_ratio: 0.35
retention_min_hit_ratio: 0.3
min_miss_observations_for_demotion: 4
```

As in the original accumulator, `enabled: false` disables all accumulation:
the archive is still validated, but every output map is empty. It does not
mean raw-map passthrough. The original time/viewpoint promotion gate is an
OR gate, not an AND gate; zero values retain their original semantics.

## Outputs and Completion

`--output-dir` must be a new directory with an existing parent. Existing
directories, files, symlinks (including dangling ones), input aliases, and
input/output nesting are rejected. No input file is changed.

| File | Meaning |
| --- | --- |
| `map_static.pcd` | Confirmed temporal candidate voxel centroids, map frame |
| `map_unfiltered.pcd` | **All temporal candidate voxels**, confirmed + unconfirmed |
| `map_rejected.pcd` | Unconfirmed candidates, **not semantic person-only** |
| `map_evidence_rgb.pcd` | All candidates, green `(0,255,0)` confirmed / orange `(255,165,0)` unconfirmed |
| `stats.yaml` | Effective configuration, frames/Z filter, point counts, evidence counters, timing in ms |
| `manifest.yaml` | Final completion marker with `status: complete` and output paths |

`map_unfiltered.pcd` is explicitly **not byte-identical to the core raw voxel
map**. It uses this accumulator's voxel grid and averages per-observation
centroids with temporal observation weighting. Confirmed and unconfirmed
PCDs partition its candidate voxels. RGB evidence is always emitted for
inspection in a PCD viewer. Nonempty outputs are binary PCD; empty outputs
are valid zero-point ASCII PCDs with fields, WIDTH 0, HEIGHT 1, and POINTS 0.
No dummy geometry is added to empty evidence.

Input validation and accumulation finish before the output directory is
created. All PCD files and stats are written before the completion manifest,
which is published by same-directory rename. **Only `manifest.yaml` with
`status: complete` denotes success**; individual PCD files or stats do not.
On an output error the CLI returns nonzero and attempts to write `failure.yaml`.
A crash or unwritable disk may leave an incomplete directory without that
failure file, but without a success manifest. Such a directory is never
overwritten automatically. This is a logical completion protocol, not a
power-loss durability guarantee (`fsync` is not performed).

`processing` timing includes archive validation, PCD reads, projection, and
accumulation; `pcd_output` includes map extraction and PCD writes.
`total_before_metadata` excludes final YAML serialization and publication.
Memory holds the temporal candidate/prior-free voxel maps, archive metadata,
one merged keyframe, and output clouds, not all raw scans at once.

## Verification

The original 15 accumulator tests are preserved. Additional C++ and real
CLI subprocess tests cover transformed sensor origins and ray evidence,
post-loop poses, map-frame Z filtering, multisensor one-vote behavior,
empty confirmed/RGB output, configuration validation, invalid manifests/PCDs,
path escape, timestamp precision, and output overwrite protection. Tests
use small temporary synthetic archives; no bag replay or hardware is needed.
