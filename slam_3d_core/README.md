# slam_3d_core

## Compute-first indoor preset

Load `config/indoor_compute_budget_overrides.yaml` after the calibrated indoor
sensor profile. For example, with the current installed package:

```bash
ros2 run slam_3d_core scan_to_map_icp_node --ros-args \
  --params-file <calibrated_indoor_sensor_profile.yaml> \
  --params-file <indoor_compute_budget_overrides.yaml>
```

Pair it with the ingest package's `compute_budget_overrides.yaml` (4,000 points
per sensor). The preset limits range to 22 m, scan/map voxels to 0.40/0.35 m,
surfel iterations to 5, local keyframes to 16 and active submaps to 2. Keyframes,
place recognition, and global map publication run less
often or with smaller budgets. Scan frequency, IMU deskew, shock handling,
3D pose graph and robust loop validation remain enabled. Queue polling runs at
50 ms instead of 5 ms; this is not scan decimation (dispatch is capped at
20 scans/s). This preset explicitly selects the voxel-surfel backend; its
iteration limits do not apply to ICP/GICP.

Online terrain classification and preview/traversal evidence are disabled;
these topics no longer supply usable terrain maps in this preset. The core
map remains 3D and temporally unfiltered; cleanup and projection run offline.
Dense wall detail, weak-feature tracking, offline cleanup evidence, and loop
recall may decrease. Changing
the raster resolution downstream cannot recover discarded geometry.

Benchmark CPU seconds for ingest plus core on identical bag data. Wall time at
a fixed replay rate and parallel speedups alone do not measure reduced compute.
The 20% CPU target must be measured on the intended input and machine.

Before the offline-cleanup split, one 90 s replay of a recorded fused dual-JT128
stream measured 28.32 -> 5.24
CPU seconds for core plus ingest (18.5%), with all 211 scans delivered and
identical 197 accepted / 14 deskew-rejected scan timestamps. Both runs failed
the 95% deskew-coverage gate because the recorded IMU had missing coverage/gaps.
There were no loop events. Upstream fusion, one-time timestamp sorting, bag
playback, and plots are excluded from that CPU measurement.

This is an experimental compute limit, NOT a quality-preserving replacement:
the then-online confirmed map fell from 12,590 to 679 points, and estimated path
length changed from 8.22 to 15.87 m (no ground truth). These are historical
measurements, not validation of the archive/offline-cleanup workflow. The
compute preset's point-budget settings remain unchanged by this split.
Separate physical sensor streams each receive the point cap, so
their combined budget and end-to-end CPU must be measured again. Evidence is
under `~/analysis_outputs/slam_3d/37_compute_budget/`.

Learning-oriented GOLE 3D SLAM core code. This is direct C++ implementation,
not a wrapper around FAST-LIO, LIO-SAM, GLIM, KISS-ICP, or another open-source
runtime.

The first node is `scan_to_map_icp_node`. It is intentionally smaller than a
production LiDAR-inertial SLAM stack so the algorithm can be edited and studied
inside this repository.

## Current Algorithm

```text
Canonical center XT32 plus queued front/rear JT128 PointCloud2 streams
  -> bounded timestamp pairing with one-time auxiliary scan consumption
  -> per-sensor TF transform to base_link and packet-level gyro rotation deskew
  -> wheel-odom motion compensation from each auxiliary stamp to the XT32 stamp
  -> merge available clouds at the primary base-frame time
  -> split mapping clouds from capped registration-only support clouds
  -> range/z/self crop
  -> voxel downsample, with optional smaller adaptive voxel when a scan becomes
     too sparse after downsampling
  -> wheel odom relative-motion prediction, or bounded constant-velocity
     fallback when wheel odom is unavailable
  -> optional IMU gyro rotation prediction between scan header stamps
  -> optional scan-to-scan odometry prediction for odom-free dataset replay
  -> scan-to-local-submap-map voxel-surfel, ICP, or GICP registration
  -> optional trimmed coarse registration plus narrow-distance refinement
  -> mapping-cloud-only Scan Context-style historical scan retrieval,
     combined-cloud submap geometry validation, and transactional multi-stage
     SE(3) pose-graph optimization
  -> shock/recovery state gate for CV prediction, place recognition, and
     persistent keyframe mutation
  -> temporally unfiltered global mapping cloud and opt-in keyframe archive export
  -> optional gravity-aligned keyframe-local floor/ramp/step/observed-drop/staircase
     evidence with independent-viewpoint voting
  -> optional observer-only terrain preview memory: LiDAR geometry candidate,
     stable/shock IMU traversal association, and graph-correction reprojection
  -> /slam_3d/frontend/odom, /slam_3d/frontend/path, map clouds, diagnostics
  -> optional /slam_3d/frontend/deskewed_points parity/debug output
```

Implemented in:

- `include/slam_3d_core/scan_to_map_icp.hpp`
- `src/scan_to_map_icp.cpp`
- `src/scan_to_map_icp_node.cpp`

This is the editable baseline for learning scan registration, keyframe policy,
local map construction, and three-cloud synchronization.

## Pose Graph Backend

`pose_graph_solver=ceres_se3` selects the indoor backend in
`pose_graph_optimizer.hpp/.cpp`. The original edge-wise relaxation remains as
`legacy_relaxation` for replay comparison and planar legacy profiles.

Each keyframe state is a 3D translation plus a unit quaternion. Ceres applies
rotation updates through `EigenQuaternionParameterization`, so the quaternion
stays on the SO(3) manifold while the first keyframe is fixed to remove gauge
freedom. For an edge measurement `Z_ij`, the six residual components are the
relative translation error in frame `i` and the multiplicative quaternion
error `2 vec(q_error)`. Every edge carries a symmetric full `6x6` information
matrix in `[tx, ty, tz, rx, ry, rz]` residual order. The
`configured_diagonal` mode reproduces the original independent translation and
rotation weights and remains the indoor default after full-bag regression.
The opt-in `registration_observability` mode evaluates a robust point-to-plane
Gauss-Newton Hessian at the final accepted pose, transported from map-frame
`[rx, ry, rz, tx, ty, tz]` perturbations into the edge residual frame, and
eigen-decomposed after normalization by the configured weights. Weak geometric
directions are reduced to a bounded floor while no direction can become
stronger than the configured baseline. Current-scan loop measurements include
the latest-keyframe-to-scan lever arm. A failed quality gate, invalid Hessian,
or changed loop linearization point falls back to the configured diagonal
matrix. Ceres obtains the Jacobians with automatic differentiation and solves
the sparse normal system with Levenberg-Marquardt and sparse Cholesky.

Loop optimization is deliberately staged:

1. Descriptor, temporal exclusion, submap consistency, and point-cloud
   registration reject weak loop candidates before graph mutation.
2. A verified large loop may receive the existing path-distributed SE(3) seed
   to move the initial estimate into the nonlinear solver's convergence basin.
3. Ceres solves the entire connected graph with Cauchy scales of `4x`, `2x`,
   and `1x`. Sequential LiDAR edges remain quadratic; only loop edges are
   robustified.
4. At every stage, Dynamic Covariance Scaling (DCS) computes
   `s=min(1, 2*phi/(phi+chi2))` for every loop and scales its information by
   `s^2`. The next stage therefore sees the residual consistency discovered by
   the preceding solve.
5. A new loop whose final DCS scale is below
   `pose_graph_min_new_loop_robust_weight` is removed. The candidate solution
   must also improve the separately evaluated fixed Cauchy objective and pass
   the existing residual and ground-motion gates.
6. Only then are keyframe poses committed. Every keyframe-local cloud, submap,
   path sample, and terrain sample is reprojected. Archive export captures the
   latest corrected poses with immutable local sensor views; any failed stage
   leaves the graph and map unchanged.

The first-stage and last-stage Ceres costs are reported separately because the
robust scale and DCS weights change between stages; they are not directly
comparable. The fixed-kernel `pose_graph_robust_objective_before/after` values
remain the transactional acceptance metric. Solver stages, accepted/rejected
LM steps, final gradient, DCS downweight count, and new-loop DCS scale are also
published in `/slam_3d/health`.
The Cauchy scale and DCS `phi` operate in the information-weighted residual
space, not in meters alone. Changing edge information therefore requires a
joint replay calibration of both robust parameters.

When enabled, the registration Hessian is used only for its directional shape.
It is not reported as a statistically calibrated ICP covariance: nearest-neighbor
rematching, shared submap points, robust weighting, and correlated LiDAR
samples violate the independent fixed-correspondence assumptions needed for
that claim. The configured information remains the explicit magnitude bound.

Primary references for this implementation are the Ceres 3D pose-graph
example, Agarwal et al.'s Dynamic Covariance Scaling, Suenderhauf and Protzel's
Switchable Constraints, and Grisetti et al.'s graph-SLAM tutorial:

- https://ceres-solver.readthedocs.io/latest/nnls_tutorial.html
- https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/slam/pose_graph_3d/pose_graph_3d_error_term.h
- https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/agarwal13icraws.pdf
- https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf
- https://doi.org/10.1109/MITS.2010.939925
- https://research.google.com/pubs/archive/45466.pdf
- https://gisbi-kim.github.io/publications/gkim-2018-iros.pdf
- https://arxiv.org/abs/1909.08605
- https://censi.science/pub/research/2007-icra-icpcov.pdf
- https://arxiv.org/abs/1410.7632
- https://arxiv.org/abs/2211.16335

Switchable Constraints and Graduated Non-Convexity are alternative robust
outer-loop strategies, not extra kernels stacked into the current solve.
Applying Cauchy, DCS, switch variables, and GNC to the same loop simultaneously
would make the effective information difficult to interpret. DCS is used here
because its closed-form scale adds no latent graph state. A switch-variable or
GNC backend should be evaluated as a controlled A/B against DCS instead.

## Offline Map Cleanup Migration

`/slam_3d/map/global_cloud` is now temporally unfiltered mapping geometry. Normal
range/z/self filters, mapping voxelization, and registration-only input exclusion
still apply. It is not the confirmed-static output of `slam_3d_map_cleanup`.
Registration targets, submaps, descriptors, loop verification, and the pose graph
remain core responsibilities; offline cleanup does not feed back into them.

`temporal_static_map_enabled=true` throws `std::invalid_argument` with migration
guidance. `false` is accepted only as a legacy guard. Remove all other
`temporal_static_map_*` parameters and the former temporal-candidates topic.
Cleanup thresholds now live in
[`slam_3d_map_cleanup/config/indoor.yaml`](../slam_3d_map_cleanup/config/indoor.yaml).

Configure the archive before mapping in the selected core parameter file:

```yaml
scan_to_map_icp_node:
  ros__parameters:
    keyframe_archive_directory: /absolute/path/to/new_keyframe_archive
    keyframe_archive_voxel_leaf_m: 0.22
```

The default directory is `''`: it disables archive-specific scan retention and
export without disabling ordinary SLAM keyframes or terrain evidence. The
archive keeps mapping-only, deskewed base-frame sensor views before terrain
classification, with sensor origins, keyframe stamps, stable monotonic IDs, and
the latest corrected SE(3) poses. These views are filtered/downsampled evidence,
not original raw scans. The manifest reports `snapshot_scope: retained_keyframes`
and `pruned_keyframes`; entries removed by `max_keyframes` are not recovered.

After mapping and any graph correction, export while the node is still running:

```bash
ros2 service call /scan_to_map_icp_node/export_keyframes std_srvs/srv/Trigger '{}'
```

The service is private `~/export_keyframes`; the launch default node name is
`/scan_to_map_icp_node`. Use the renamed/namespaced node path when launch overrides
it. Trigger uses the configured archive directory, not a path in its request.
Check `success` and the response message before starting the separate CLI:

```bash
ros2 run slam_3d_map_cleanup temporal_map_cleanup \
  --input /absolute/path/to/new_keyframe_archive \
  --output-dir /absolute/path/to/new_cleaned_map \
  --config install/slam_3d_map_cleanup/share/slam_3d_map_cleanup/config/indoor.yaml
```

Use a new output directory with an existing parent, separate from the archive.
`map_static.pcd` contains confirmed geometry; `map_rejected.pcd` contains
unconfirmed evidence, not semantic person-only labels. `map_unfiltered.pcd`
contains all cleanup candidate voxel centroids, not the original core global
cloud. Each synchronized keyframe remains one observation across its sensor
views. Re-export and rerun cleanup after later graph corrections; use a trajectory
from the same corrected state for projection. Cleanup is geometric persistence
evidence, not person detection or object tracking.

An auxiliary input marked `secondary_cloud_registration_only` or
`tertiary_cloud_registration_only` is voxelized and capped relative to the
mapping-cloud point count. It remains available to frontend registration,
registration submaps, and geometric loop validation. It does not enter place
descriptors or the published/global map, and it cannot create a scan when the
mapping input is empty. This supports using a dense ground stream for ramp
observability while keeping saved obstacle PCDs ground-removed.

`place_recognition_registration_only_vertical_recovery_m` optionally lets
place recognition inspect a candidate outside the normal vertical-separation
gate, but only when both the current scan and the historical submap contain
registration-only support points. The normal gate remains authoritative for
obstacle-only scans, and descriptor, geometric-registration, consecutive-match,
and robust pose-graph checks still apply. This is intended to recover a return
through a previously observed ramp after accumulated vertical drift without
globally allowing cross-floor loop candidates.
`place_recognition_registration_only_consistency_max_translation_m` can also
provide a bounded consecutive-correction tolerance for that recovery path. It
is used only when both the current and immediately preceding matches used the
registration-only vertical recovery; normal loop candidates keep
`place_recognition_submap_consistency_max_translation_m`.
`place_recognition_registration_only_consistency_max_gap_s` bounds how long a
temporarily missing recovery candidate may preserve that state. Recovery
matches may also cross one adjacent submap, but the same transform and robust
backend checks are still required.

An independent point-to-plane registration core is also implemented in:

- `include/slam_3d_core/voxel_surfel_registration.hpp`
- `src/voxel_surfel_registration.cpp`
- `test/test_voxel_surfel_registration.cpp`

It builds planar voxel surfels, applies Huber IRLS, projects updates onto the
observable Hessian subspace, and reports rank, condition, weak directions,
residuals, and timing. Select it with `registration_method=voxel_surfel`.
Synthetic numerical tests and the bounded Geoje Hesai replays exercise this path
independently of the ICP/GICP implementations.

The indoor XT32 default is `config/indoor_xt32_single_params.yaml`: voxel-surfel
registration at the 10 Hz scan rate plus a per-scan bounded LiDAR-only
constant-velocity initial guess. The original slower GICP comparison is kept in
`config/indoor_xt32_single_gicp_quality_params.yaml`. Both consume only the
selected LiDAR stream and `/imu/data`; wheel prediction is disabled. The
indoor profiles require `cv7_link` and use its native XYZ convention with
`deskew_imu_to_base_rpy_deg=[0,0,0]`. The commissioned contract publishes an
identity `base_link->cv7_link` rotation. The older `04_gangnam` recordings
that needed a replay-only `roll=180 deg` workaround remain reproducible from
their archived experiment configs; that workaround is no longer a general
indoor default. Physical mounting and LiDAR-to-IMU clock offset still need
live calibration and this setting must not be copied to another IMU frame.
The default 80-iteration pose-graph cap was selected by a 40/80 iteration A/B
on the `04_gangnam` return loop; it is a dataset-specific convergence setting,
not a substitute for reference-trajectory accuracy validation.

The validated `04_gangnam` bags advertise RELIABLE `/imu/data`, so the indoor
profiles set `imu_subscription_reliable=true`. This prevents transport loss
from changing the deskewed point cloud under CPU load. A live deployment must
first verify that its publisher also offers RELIABLE QoS. For each cloud, the
core copies one IMU snapshot and truncates it at the cloud set's required
point-time horizon. Deskew, roughness evaluation, and optional IMU rotation
prediction all consume that same causal prefix, so callback-dependent future
samples cannot alter one scan midway through processing. When watermark waiting
is enabled, it applies to every cloud-bound IMU consumer even if deskew itself
is disabled. Point-time bounds are decoded independently of deskew, so an IMU
impact after the cloud header but before its final point remains part of that
cloud's roughness observation. A selected auxiliary LiDAR's point-time end also
extends the snapshot horizon, including when waiting is disabled and only the
already available IMU prefix can be used. Overlapping clouds whose point-time
end does not advance cannot rewind or falsely advance the shock-policy clock.
`deskew_max_scan_duration_s` must be finite and greater than zero. Decoded
bounds must fit that duration, and both endpoints must remain within one such
duration of the cloud header. An outlier or uniformly shifted point-time field
can still fail deskew diagnostics, but it cannot gain a second duration of IMU
wait or shock-policy look-ahead. These comparisons include only a
timestamp-magnitude ULP tolerance so epoch-scale ROS times at the exact duration
limit are not rejected by double rounding.

Indoor terrain evidence is intentionally separate from the map cloud. Each
accepted keyframe retains a terrain-specific 0.04 m cloud before the
coarser map voxel filter. A timestamped, low-pass IMU acceleration snapshot
levels that cloud in a yaw-neutral gravity frame; classification is skipped
when the snapshot is unavailable or stale. Labels therefore stay attached to
the original keyframe-local points and are reprojected when pose-graph poses
change. A height gate and a supported near-base height cluster prevent a
ceiling-only, high-shelf-only, or isolated low return from seeding a floor.
Voxel labels require independent viewpoints and a configurable
vote confidence, so later contradictory floor evidence can revoke an earlier
hazard label. `/slam_3d/terrain/evidence` uses `PointXYZI`, with intensity
`1=floor`, `2=ramp`, `3=positive_step`, `4=directly_observed_lower_floor`, and
`5=staircase`.
An empty transient-local cloud is published to clear stale downstream state.

The confirmed-only semantic contract remains unchanged. A separate transient-local
`/slam_3d/terrain/support_evidence` PointCloud2 retains the intermediate evidence
instead of forcing it into a floor/non-floor bit. Its states are
`0=unresolved`, `1=probable_support`, `2=confirmed_support`,
`3=probable_hazard`, and `4=confirmed_hazard`. Fields include the empirical
support-vote ratio, semantic-label confidence, observation counts, geometric
roughness, a geometry-only traversability score, and whether a probable support
voxel failed direct connectivity to confirmed support. One hazard vote vetoes
support promotion. A one-view floor/ramp voxel can become probable only when it
is close in XY and height to a directly confirmed support voxel; probable voxels
never recursively validate other probable voxels. The vote ratio is not a
calibrated posterior probability, and the traversability score does not include
robot footprint, clearance, slope capability, or controller dynamics.

The indoor profiles also enable an observer-only terrain preview memory on
`/slam_3d/terrain/preview_evidence`. Its PointXYZI intensity is
`10 * terrain_label + preview_state`, where preview states are
`1=tentative`, `2=geometry_confirmed`, `3=traversal_confirmed`, and
`4=unsafe_traversal_conflict`. A central radial-profile interpolation residual
adds small rough-floor candidates that the semantic step/ramp classifier can
miss. A candidate counts as an advance preview only when it was first observed
at least 0.75 m in front of the robot and no more than 1.00 m laterally from
the heading axis. The current footprint association is a conservative 0.75 m
XY disk, not a calibrated chassis or per-wheel polygon.

Traversal evidence is causal and separate from LiDAR geometry confidence. A
normal passage can confirm that the robot occupied a candidate. An IMU impact
is localized by timestamp interpolation between valid stable pre-impact and
stable-recovery poses; an event without bounded time and travel distance is
discarded instead of being assigned to the recovery endpoint. Angular velocity
and angular acceleration are excluded from this terrain-response score so a
yaw manoeuvre alone cannot label floor roughness. A yaw-only quarantine or
rejected registration stores a continuity break, preventing replay from
inventing a straight traversal between the stable poses on either side.
If a stronger translational pulse arrives in the same response episode, both
the retained score and its physical timestamp move to that pulse. Any rejected
or pose-quarantined scan between the endpoints invalidates the pending shock
bridge; a later recovery cannot erase that independent continuity break.
Primary-cloud queue overflow tags the first surviving cloud with the same
break, reported as `primary_cloud_gap_before`, only when a pending forward
cloud was actually omitted. A stale arrival that is itself discarded from a
full queue, or that remains in an underfull queue until a later overflow, does
not create a false break. Every primary cloud, including a
sparse or deskew-failed rejection, advances the monotonic input watermark; a
duplicate or older timestamp is rejected before deskew, IMU policy,
registration, or terrain state can be updated.
Observed-drop/staircase proximity creates an unsafe conflict instead of a safe
traversal confirmation. Geometry and traversal samples remain attached to
keyframe-local state and are interpolated/replayed when pose-graph poses change.
This observer does not modify scan registration, keyframe selection, loop
acceptance, graph edges, poses, or the map cloud. It is therefore diagnostic
terrain memory, not yet a localization constraint or a navigation safety
decision.

While the IMU shock policy is active, odometry remains published with inflated
covariance for observability, but those uncertain samples are omitted from the
published frontend path. Persistent map, keyframe, submap, and loop state are
also quarantined until consecutive fresh stable scans complete recovery.

This is geometric evidence, not a navigation safety guarantee. No return is
always unknown, never a drop. Staircase evidence requires at least three
same-direction risers with consistent riser height, tread spacing, and aggregate
slope in one gravity-aligned radial profile. This remains a geometric candidate,
not a traversability guarantee. Per-sensor ray miss/occlusion evidence,
confidence fields beyond diagnostics, and 2D projection remain deferred.

## Deliberate Limitations

- The core global map retains transient geometry. The separate offline cleanup
  is geometric persistence evidence, not a person detector: a stationary person
  can be promoted and a surface seen only once can remain tentative. Archive
  export covers only retained keyframes and cannot reconstruct pruned history.
- IMU data is buffered for an optional gyro-only rotation deskew step. This MVP
  requires a `PointCloud2` field such as `timestamp`, `time`, `t`, or
  `offset_time`; it does not estimate translation during the scan and is not
  IMU preintegration.
- `imu_subscription_reliable` defaults to `false` for publisher compatibility.
  Enable it only after `ros2 topic info -v /imu/data` confirms a RELIABLE
  publisher. `/slam_3d/health` reports the configured reliability, causal
  snapshot endpoint, and snapshot sample count.
- `slam_3d_lidar_ingest` validates two Hesai JT128 streams and one XT32 stream,
  then emits separate canonical PointCloud2 clouds with `offset_time` in
  nanoseconds relative to `header.stamp`. In `02_samsung_geoje`, timestamps are
  packet-level values repeated for 256 points, not calibrated channel firing
  times. The bounded replay profile applies a deterministic affine fit from
  each sensor clock to bag record time. This offline approximation must not be
  described as tightly synchronized LIO input.
- The center XT32 triggers registration. Front and rear JT128 scans are selected
  from timestamp-sorted bounded queues by normalized start stamp and consumed
  at most once. A per-sensor timestamp watermark plus
  `cloud_sync_reorder_window_s` decides that a match can no longer arrive.
  Primary-stream progress is deliberately not treated as proof of an
  auxiliary dropout because ROS 2 does not guarantee callback order across
  topics. If an auxiliary stream stalls, `cloud_sync_wait_timeout_s` is the
  final fallback. Size `cloud_sync_queue_size` for at least the expected
  primary rate multiplied by that timeout, plus margin.
- If that primary queue still overflows, the oldest cloud is discarded and the
  first survivor carries a persistent terrain continuity break. Health output
  exposes both `primary_sync_queue_overflow_count` and
  `primary_cloud_gap_before`.
- PointCloud subscriptions use a reentrant callback group while registration
  runs from a separate wall-timer callback. Slow registration therefore cannot
  prevent the other LiDAR queues from receiving matching scans.
- `inter_lidar_motion_compensation_enabled` transforms an already deskewed
  auxiliary cloud into the primary base-frame time before concatenation.
  `inter_lidar_motion_compensation_source=wheel_odom` preserves the original
  exact-or-bracketed wheel-pose interpolation, including translation and
  rotation. `imu_rotation` instead integrates the configured `/imu/data` gyro
  over the scan-start offset and corrects rotation only; it does not subscribe
  to wheel odometry or estimate translation. The dedicated
  `inter_lidar_imu_rotation_scale` multiplies only this inter-scan gyro delta,
  independently of per-point deskew, and is intended for calibration A/B
  checks; the nominal value is `1.0`. `inter_lidar_motion_compensation_max_dt_s`
  bounds either correction. When
  `inter_lidar_gap_recovery_hypothesis_enabled=true`, a scan arriving after
  `inter_lidar_gap_recovery_hypothesis_min_gap_s` is also registered without
  the inter-LiDAR correction. The uncompensated scan is selected only when its
  fitness/inlier quality score improves by the configured minimum ratio; only
  the selected scan enters tracking, keyframes, and submaps. This bounded
  second hypothesis avoids committing a compensation-sensitive GICP local
  minimum after an IMU/data gap while leaving normal-cadence runtime unchanged.
  With the default
  `inter_lidar_motion_failure_policy=drop_auxiliary`, a missing wheel bracket or
  invalid IMU interval drops that auxiliary cloud and raises degraded
  diagnostics instead of contaminating the merged scan.
- Map publication takes immutable local/global cloud snapshots with `try_lock`
  and performs PointCloud2 conversion and DDS publication outside the frontend
  state lock. A slow map subscriber therefore cannot occupy an executor thread
  while registration owns the state.
- Loop closure remains a learning-oriented batch backend rather than a
  production incremental smoother. Consecutive LiDAR edges and verified loop
  edges are solved as a connected sparse SE(3) graph with Ceres in indoor
  profiles. The optional registration-observability mode supplies bounded full
  `6x6` directional information, but it is not a calibrated covariance
  estimate and requires a dataset-specific replay regression before use. Graph
  solving is synchronous, and there are no IMU preintegration, bias, velocity,
  or covariance states. The legacy edge-wise relaxation remains available only
  for comparison and planar legacy profiles.
- Terrain labels are approach-direction dependent: the same ledge can appear as
  a positive step from one side and an observed drop from the other. The
  preview state must not be injected as a direct loop edge. A future
  direction-invariant ledge primitive and independently validated terrain
  likelihood may first be used as a loop conflict diagnostic or veto; adding
  another strong factor from the same LiDAR geometry would otherwise
  double-count correlated evidence.
- The internal graph is published as
  `visualization_msgs/msg/MarkerArray` on
  `/slam_3d/backend/pose_graph_markers` by default. Keyframe spheres use a
  chronological color ramp, consecutive odometry edges are thin green lines,
  accepted loop edges are thicker magenta lines, and an arrow shows the latest
  keyframe orientation. The topic is configurable with
  `pose_graph_marker_topic` and uses reliable transient-local QoS so RViz and
  recorders can capture the latest graph after joining.
- Pose-graph diagnostics report end-to-end optimization time, solver-loop time,
  mean time per iteration, iteration count, solver status, convergence reason,
  residuals before/after, and robust loop weights. These fields distinguish a
  graph correction that merely reached its iteration cap from one that met the
  step tolerance.
- The constant-velocity fallback is only an ICP initial guess. It is not a
  replacement for IMU preintegration, deskew, or a backend optimizer.
- `imu_rotation_prediction_enabled` defaults to `false`. When enabled,
  `imu_rotation_prediction_source=angular_velocity` integrates buffered gyro
  samples between consecutive processed scan stamps.
  `imu_rotation_prediction_source=orientation` instead interpolates valid IMU
  quaternions near both scan stamps and uses their relative rotation. Both modes
  preserve translation from the existing motion prior and replace only the
  predicted orientation. Orientation samples outside
  `imu_orientation_prediction_max_time_delta_s` are rejected. Neither mode
  estimates translation, velocity, bias, or gravity; they are narrow replay and
  debugging aids. The Geoje v11 experiment exercised the orientation path but
  increased max-iteration surfel exits, so its checked-in replay profile keeps
  rotation prediction disabled.
  `imu_rotation_prediction_scale` is separate from
  `deskew_angular_velocity_scale` so replay can test the inter-scan rotation
  sign without corrupting per-point deskew.
- `imu_relative_roll_pitch_constraint_enabled` defaults to `false`. When
  enabled, the first accepted cloud's time-interpolated IMU quaternion defines
  the map attitude gauge. Later candidates keep LiDAR translation and yaw but
  use the initial-to-current IMU relative roll/pitch. Samples farther than
  `imu_relative_attitude_max_time_delta_s` are ignored, and
  `imu_relative_roll_pitch_max_correction_deg` bounds each scan's roll and
  pitch correction independently (`0.0` means unbounded). Initialization waits
  for a valid orientation instead of silently choosing a different gauge. This
  guard requires a verified quaternion convention and IMU-to-base extrinsic;
  it is not IMU preintegration and does not estimate velocity, bias, yaw, or
  translation.
- IMU ingestion uses a dedicated mutually-exclusive callback group and the node
  executable runs a two-thread executor. Registration, map publication, and the
  other default callbacks remain mutually exclusive, while the IMU buffer can
  continue receiving high-rate samples during a slow ICP or pose-graph update.
  Map diagnostics record both total publication runtime and the terrain-support
  cloud portion so middleware backpressure can be separated from frontend cost.
  The primary LiDAR uses a mutually-exclusive callback group so expensive point
  time-bound extraction cannot let a later primary scan overtake an earlier
  one. Auxiliary inputs retain their reentrant group and timestamp-sorted sync
  queues so multi-LiDAR ingestion remains concurrent.
  `imu_subscription_depth` defaults to 400 instead of the depth-5 sensor QoS
  default so a one-second 400 Hz burst is not silently reduced to its last few
  samples. Access to the deskew/gravity buffer remains protected by
  `imu_mutex_`. The cloud dispatcher captures the IMU buffer and watermark in
  one snapshot, so deskew cannot observe a different buffer from the one used
  to release that cloud. Each accepted IMU callback also publishes its timestamp on the
  transient-local reliable `/slam_3d/imu_watermark_ns` topic. Offline replay
  can wait for this watermark before publishing a cloud, but it proves callback
  progress only; it cannot prove that every intermediate best-effort IMU sample
  was delivered. The reliable DDS publish occurs after releasing `imu_mutex_`
  so middleware backpressure cannot block deskew/gravity readers inside the IMU
  critical section.
- Point clouds keep ROS sensor-data `best_effort` QoS by default so the node
  remains compatible with live LiDAR drivers. Set
  `cloud_subscription_reliable:=true` only when the publisher also offers
  reliable QoS, such as a bounded offline bag replay where silently losing a
  multi-megabyte cloud would invalidate an A/B comparison. This setting applies
  to all enabled LiDAR streams. `cloud_subscription_depth` bounds DDS history,
  while `cloud_sync_queue_size` independently bounds each application queue, so
  reliable transport is not a substitute for replay backpressure.
- Registration is accepted only when convergence, bounded-distance fitness, and
  prediction-to-result correction limits pass. The correction gate also checks
  aligned-cloud inlier ratio/RMSE against the local map so a small set of close
  correspondences cannot hide weak overlap.
- `registration_gap_relaxation_enabled` keeps the normal translation-correction
  limit at ordinary scan cadence. After a gap of at least
  `registration_gap_relaxation_min_gap_s`, it may raise that limit to the lower
  of `registration_gap_relaxation_max_translation_m` and scan gap multiplied by
  `registration_gap_relaxation_max_speed_mps`. Relaxation also requires the
  configured minimum inlier ratio and maximum inlier RMSE. Diagnostics expose
  the measured gap, effective gate, and whether the wider gate was available.
  This is a bounded dropout-recovery policy, not a motion estimate or permission
  to accept low-overlap registration.
- `robust_registration_enabled` defaults to `false`. When enabled, point-to-point
  ICP discards the farthest correspondence fraction using PCL's trimmed
  correspondence rejector, performs the normal wide-distance registration, and
  then starts a second refinement from that pose with
  `robust_registration_refinement_max_correspondence_m`. GICP uses the same
  coarse-to-fine schedule but does not use PCL's ICP-only trimmed rejector.
  Acceptance fitness and overlap for ICP, GICP, and voxel-surfel are recomputed
  with the independent, narrower
  `robust_registration_quality_max_correspondence_m`, so a large convergence
  radius cannot make every point appear to be an inlier. Diagnostics retain
  coarse and refinement fitness/inlier/RMSE values and whether refinement was
  selected; a converged refinement replaces the coarse result only when its
  fitness divided by strict-distance inlier ratio improves. This is an
  inspectable robust-registration experiment, not a substitute for the planned
  tightly coupled LIO or X-ICP-style directional localizability constraints.
  HILTI 2023 site2 handheld4 smoke replays did not justify enabling it by
  default: trim 0.85 plus 0.8 m refinement increased 420-frame wall time from
  212.2 s to 514.8 s, introduced 13 processing wait timeouts, and increased the
  trajectory closure gap from 1.64 m to 6.65 m. A no-trim 1.0 m refinement also
  increased the 120-frame wall time from 84.8 s to 195.7 s with six processing
  wait timeouts. Keep the strict-distance metrics for diagnosis, but treat the
  refinement and trimming controls as rejected tuning candidates unless a new
  dataset-specific experiment proves otherwise.
- `adaptive_scan_voxel_enabled` defaults to `false`. When enabled, the scan is
  first downsampled with `scan_voxel_leaf_m`; if the resulting scan has fewer
  than `adaptive_scan_voxel_min_points`, it is re-sampled with
  `adaptive_scan_voxel_leaf_m`. This mirrors the AdaLIO observation that narrow
  or degenerate HILTI-like spaces can lose too many correspondences under a
  fixed coarse voxel.
- `quarantine_keyframes_on_shock` defaults to `false`. When enabled, accepted
  poses are still published, but scans whose scan-matching correction exceeds
  the quarantine thresholds or whose IMU window is in the explicit shock
  policy are not inserted as keyframes. This keeps sudden threshold/door-frame
  shocks from immediately contaminating the persistent map.
  The same shock frame is also kept out of the constant-velocity accepted-pose
  history, so one bad correction does not become the velocity seed for the next
  few frames. By default it is also excluded from scan-to-scan tracking.
  Shock and recovery frames also skip place recognition before its cooldown and
  candidate-consistency state can be mutated. Recovery requires stable IMU
  evidence; missing, duplicate, or non-increasing samples do not advance it.
  The first recovered accepted pose rebases the CV history instead of applying
  the pre-shock velocity delta. Published shock poses carry large covariance
  even when pose publication itself is not quarantined.
- `quarantine_pose_on_shock` also defaults to `false`. When enabled with shock
  quarantine, the node publishes the motion-prior prediction instead of the
  large ICP correction for shock frames. This reduces visible odometry shaking
  at the cost of temporarily trusting the motion prior more than scan matching.
- `record_tracking_scan_on_pose_quarantine` defaults to `false`. When enabled,
  a pose-quarantined shock scan is still excluded from keyframes/global map, but
  is kept as the next scan-to-scan tracking reference at the quarantined
  motion-prior pose. This is an experimental recovery aid for shaky construction
  segments where skipping every shock scan makes the next scan match against a
  stale tracking frame.
- `record_tracking_scan_during_shock` defaults to `false`. When enabled, every
  successfully deskewed shock/recovery scan may advance only the ephemeral
  adjacent-scan reference, including after a rejected scan-to-map result when
  the scan-to-scan prediction itself passed. It never inserts that scan into a
  keyframe, submap, global map, accepted-pose history, or pose-graph edge. Use
  this when a long rough segment would otherwise exceed
  `scan_to_scan_max_reference_age_s`; diagnostics distinguish shock updates and
  post-rejection updates.
- `pose_quarantine_tracking_reset_interval` defaults to `0` (disabled). A
  positive value re-seeds only every Nth consecutive pose-quarantined scan as
  the scan-to-scan reference. It is a bounded alternative to recording every
  shock scan: the persistent map and accepted-pose history remain untouched,
  and diagnostics report the consecutive count and each tracking reset.
- `place_recognition_enabled` defaults to `false`. It stores a lightweight
  polar scan descriptor per keyframe and tests older keyframe candidates with
  ICP when local registration is weak. `place_recognition_descriptor_bins`
  controls the yaw sectors. `place_recognition_descriptor_rings > 1` switches
  the descriptor to a Scan-Context-like ring/sector height grid while the
  default `1` keeps the older polar range descriptor for compatibility.
  `place_recognition_max_yaw_offset_deg` can reject descriptor shifts that
  imply a large heading jump before spending ICP and pose-graph work on them.
  `place_recognition_shock_relocalization_cooldown_s` can throttle repeated
  relocalization attempts while a shock correction is being quarantined; this is
  intended for construction-site impact segments where retrying old-keyframe ICP
  every scan mostly increases latency and false candidates.
  `place_recognition_periodic_enabled`
  also defaults to `false`; when enabled, accepted scans that are far enough
  from the last keyframe can periodically probe old keyframes too. This matters
  on HILTI-like construction data because scan-to-local-map can remain
  numerically accepted while the global x/y/yaw estimate slowly drifts through a
  repeated space. Candidate matches are still bounded by descriptor, inlier,
  RMSE, pose-jump, and interval gates so a descriptor match cannot snap the live
  odometry across the map in one frame.
- `pose_graph_correction_enabled` defaults to `false`. When enabled and a place
  recognition match is accepted, the node adds a loop edge from the matched
  keyframe to the latest keyframe, optimizes the stored keyframe poses with the
  consecutive odometry edges, reprojects each keyframe's base-frame cloud into
  the corrected map frame, and updates path history by interpolating per-node
  correction transforms by timestamp. The live pose is then moved by the latest
  keyframe correction applied to the current motion-prior pose, so loop closure
  does not become an unbounded single-frame snap. Diagnostics include
  `pose_graph_edges`, `pose_graph_loop_edges`, residual before/after, optimizer
  iteration count, max per-iteration update, registration-information edge
  count, minimum observable rank, Hessian condition, and minimum directional
  gain. This is still intentionally lightweight and lacks statistically
  calibrated covariance modeling, IMU preintegration, and incremental
  relinearization.
  `place_recognition_mapping_cloud_refinement_enabled` adds an optional second
  geometric check using the map-output cloud. Its target is limited to a
  completed historical submap, frozen in one anchor-keyframe frame before the
  revisit. Only voxels observed by the configured number of distinct keyframes
  are retained, and recent keyframes are excluded before candidate generation.
  Source-to-target inlier quality, reverse target coverage, six-axis
  observability, Hessian conditioning, local correction size, overlap-score
  reduction, and non-regression of the primary registration must all pass. An
  accepted source/target submap pair contributes one loop edge by default.
  `pose_graph_duplicate_submap_pair_override_enabled` can raise that limit to
  `pose_graph_duplicate_submap_pair_override_max_edges`, but only for another
  loop that independently passes the high-confidence minimum-interval override
  and selects a historical keyframe anchor not already used by that submap
  pair. This permits a bounded second indoor-return anchor while rejecting
  repeated scans against the same anchor. Both settings default to the original
  one-edge behavior. These checks prevent the active scan from reinforcing its
  own target and reject small repeated patches, but the confirmed voxels are
  still correlated LiDAR evidence rather than ground truth. With
  `pose_graph_information_mode=registration_observability`, the accepted
  fixed-pose point-to-plane Hessian supplies only the bounded directional shape
  of the loop information matrix. It deliberately does not claim an inverse
  Hessian is a calibrated registration covariance.
  `place_recognition_mapping_refinement_required_for_pose_graph` makes this
  frozen-map check a fail-closed prerequisite for every pose-graph loop. A
  candidate that passes the ordinary pose-jump gate cannot bypass a failed
  mapping-cloud check. This stricter behavior is opt-in for compatibility.
  `place_recognition_mapping_refinement_large_pose_jump_enabled` is a separate,
  fail-closed recovery path for a physically valid revisit whose accumulated
  frontend drift is already larger than the normal pose-jump gate. A candidate
  may enter the extended translation/rotation window only when its descriptor
  passes the dedicated tighter limit. It is not accepted until the frozen
  mapping-cloud refinement above passes. The normal jump gate remains unchanged
  for every other candidate. All large-jump settings default to disabled/zero.
  A coarse pose can already be stationary at the correct frozen-map alignment,
  so requiring the second optimizer to improve it by a fixed percentage can
  create a false negative. The separate
  `place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled`
  policy handles only that case: the normal refinement must have passed every
  geometry, correction, rank, and conditioning gate and stopped specifically at
  `insufficient_improvement`; both its before/after poses must then pass stricter
  absolute inlier, RMSE, reverse-coverage, and bounded score-regression gates.
  The coarse pose is retained rather than replaced by the non-improving result.
  This policy also defaults to disabled with zero thresholds.
  Once such a large-loop measurement is verified, its LiDAR SE(3) target is
  passed to the graph unchanged instead of being re-clamped against the
  still-drifted active pose. IMU/ground consistency is then enforced by the
  graph's ground-sequence and same-layer constraints.
  `pose_graph_distributed_loop_seed_enabled` can then distribute the verified
  endpoint SE(3) correction over cumulative keyframe travel before the selected
  graph solver. The historical anchor is fixed, the newest node is seeded
  exactly at the loop target, and every stored keyframe cloud is reprojected
  only if the seeded graph strictly lowers the robust objective and all normal
  post-optimization residual checks pass. This option additionally requires
  robust loop weighting and the independent descriptor, inlier, RMSE, and
  reverse-coverage gates used by the minimum-interval override. It is an
  historical loop-edge check: inlier ratio/RMSE come from the selected
  candidate/submap registration, not from a later quality refresh against the
  still-drifted active local map. The exact values used by the backend are
  published as `pose_graph_constraint_*` diagnostics. It is an
  initialization aid for the nonlinear solver, not permission to accept an
  unverified map snap.
  `pose_graph_correction_min_interval_s` is a global throttle between accepted
  graph corrections. A long interval can suppress a distinct return-to-start
  loop shortly after another loop, even though exact source/target submap pairs
  are already de-duplicated. The optional
  `pose_graph_minimum_interval_override_enabled` permits such a loop to proceed
  only when the map-output alignment was geometrically verified and the descriptor distance,
  primary-cloud inlier ratio/RMSE, and reverse target coverage all satisfy the
  dedicated override thresholds. All normal correction-size, residual, robust
  weight, and optimizer validation gates still run. The override defaults to
  `false`; health diagnostics expose its elapsed time, decision status, and
  whether it was applied.
  A deployment that is known to contain only one traversed floor can also set
  `pose_graph_minimum_interval_override_single_floor_z_enabled=true`. For an
  accepted high-confidence interval override, the backend then selects the
  nearest historical keyframe in the matched submap within the configured XY
  radius, transports its Z to the current loop measurement, and seeds that Z
  correction continuously from the historical anchor to the graph endpoint.
  This is explicitly disabled by default because the same-XY/same-Z assumption
  is invalid in multi-storey maps, bridges, and vertically overlapping paths.
  `pose_graph_minimum_interval_override_single_floor_z_weight_multiplier`
  strengthens only the constraint-frame z residual of that confirmed
  single-floor loop edge. For gravity-aligned ground-robot keyframes this is
  approximately the vertical residual. XY and rotation retain the normal loop
  weights, and registration-only vertical-recovery loops are not strengthened.
  `pose_graph_robust_loop_enabled` applies Cauchy loss to loop edges only.
  Consecutive LiDAR edges retain their quadratic information. With
  `pose_graph_solver=ceres_se3`, the scale is annealed by
  `pose_graph_ceres_robust_scale_factors`, while DCS independently updates each
  loop information matrix between solves. Raw squared residual and the fixed
  base-scale Cauchy objective are reported separately so transactional
  acceptance is not confused with changing stage costs.
  `pose_graph_min_new_loop_robust_weight` rejects a newly proposed loop before
  optimization when the Cauchy kernel already regards it as an extreme outlier.
  The same threshold rejects the edge after optimization when final DCS still
  regards the new loop as inconsistent. It only applies when robust loop
  weighting is enabled. Enabling the
  robust loop kernel also makes pose-graph validation mandatory for an accepted
  place match; if the proposed loop is rejected, the frontend restores the
  pre-place-recognition registration instead of retaining the unvalidated pose
  snap. This includes the relocalization path where local registration failed:
  a rejected backend loop restores the rejected local result and enters the
  normal fallback/rejection path. The same rollback occurs when validation is
  required but pose-graph correction itself is disabled, so a contradictory
  configuration fails closed. Every accepted place candidate first receives the
  same motion, planar, and gravity bounds as the normal commit path, even when
  backend validation is optional. Candidate-map ICP correction, bounded jump
  from the current prediction, and the equivalent left map correction are
  reported separately. Shock safety gates candidate-alignment correction and
  bounded prediction jump; the map-origin-dependent left translation is
  diagnostic only. A rejected candidate is restored before graph optimization,
  so a quarantined pose cannot mutate the persistent graph and then be discarded
  only from the frontend.
  `pose_graph_max_residual_after` and `pose_graph_max_residual_ratio` can reject
  a loop edge even when optimization technically reduces residual; this is useful
  for repeated construction-site geometry where a false loop can lower the
  scalar graph error but still leave meter-scale inconsistency.
  `pose_graph_max_loop_residual_before` is a cheaper pre-optimization gate for
  the prospective loop edge itself. Keep it at `0.0` for disabled/default runs;
  enable it in HILTI-style sweeps when rejected loop candidates are spending too
  much time in the full graph solve.
  `pose_graph_planar_correction_enabled` can be enabled for HILTI-like shaky
  handheld or mostly-ground-robot tests so loop closure corrects x/y/yaw while
  leaving z/roll/pitch to IMU/deskew and local registration. This avoids a bad
  3D loop match tilting an entire construction-site map. It therefore requires
  `pose_graph_solver=legacy_relaxation`; `ceres_se3` intentionally requires
  full 3D.
- `planar_pose_constraint_enabled` defaults to `false`. When enabled, accepted
  frontend poses keep x/y/yaw from registration but force roll/pitch to the map
  gravity frame and optionally clamp z correction around the motion prior with
  `planar_pose_max_z_correction_m`. This is useful when an offline dataset has
  no valid IMU orientation and scan matching starts fitting repeated vertical
  structures by tilting keyframes. It is a debugging constraint, not a full
  replacement for gravity-aligned IMU preintegration.
- `ground_vehicle_motion_constraint_enabled` defaults to `false`. When enabled,
  each candidate translation is expressed in the previous committed body frame
  and its body-z component is bounded before the gravity/planar constraint.
  A zero `ground_vehicle_nominal_max_body_z_step_m` is a hard nonholonomic
  constraint, not a disabled limit. Forward motion on a pitched body still
  changes map z, so ramps remain possible while weak map-z registration drift
  is removed. `ground_vehicle_roughness_max_body_z_step_m` may permit a larger
  body-z step during a recent IMU roughness event; keep both limits equal when
  field evidence does not justify that relaxation. The constraint is applied
  to normal registration, fallback, and local-map bootstrap commit paths.
- `gravity_roll_pitch_constraint_enabled` can be combined with the planar pose
  constraint. It low-pass filters IMU linear acceleration, rejects samples whose
  acceleration norm is outside `gravity_accel_min_norm_mps2` to
  `gravity_accel_max_norm_mps2`, and uses the resulting gravity roll/pitch with
  scan-matched yaw. This is more appropriate than forcing zero roll/pitch for a
  handheld or a robot crossing a bump, but it still ignores acceleration bias
  and dynamic acceleration except for the simple norm gate.
  `gravity_accel_time_constant_s > 0` changes the filter to
  `alpha = 1 - exp(-dt/tau)` using IMU timestamps, making its physical response
  independent of callback frequency. `0.0` preserves the legacy fixed
  `gravity_accel_low_pass_alpha` behavior for reproducible baseline replays.
  Each cloud captures the newest filtered gravity sample whose timestamp is not
  later than the cloud timestamp from the same dispatch-time sensor snapshot
  and reuses that immutable sample throughout frontend, keyframe terrain
  classification, and backend constraints. This prevents a concurrently
  arriving future IMU sample from changing roll/pitch halfway through one cloud.
  A gravity-only configuration always uses the same cloud-stamp watermark wait,
  independently of `deskew_wait_for_imu_watermark`, so cloud-first and
  IMU-first callback order select the same causal prefix.
  `gravity_accel_max_age_s` defaults to `0.1`; an older snapshot is reported as
  `stale_at_cloud_stamp` and is not applied.
  Out-of-order IMU samples are dropped to keep deskew and rotation integration
  ordered. `imu_reorder_tolerance_s` separates small reorder events from larger
  rewinds in diagnostics, while a backward jump of at least
  `imu_timestamp_reset_threshold_s` is counted as a reset candidate and dropped.
  The node deliberately does not reset only the IMU/gravity buffers because that
  would leave LiDAR trajectory state, cooldown stamps, keyframes, and path on a
  different timeline. Restart the complete SLAM node after a real ROS clock
  rewind.
- Filtered scans that are already below `min_scan_points` before downsampling
  publish a `scan_too_sparse_before_downsample` health record. Offline replay
  can therefore distinguish an intentional sparse-frame rejection from a hung
  callback.
- `imu_roughness_detection_enabled` defaults to `false`. When enabled, the node
  computes per-scan IMU roughness diagnostics over the latest scan interval or
  `imu_roughness_window_s`, whichever is shorter. Diagnostics include max gyro
  norm, angular acceleration, linear acceleration norm change, and linear jerk.
  Detection alone is diagnostic and can select the optional relaxed
  ground-vehicle body-z limit; it does not change the frontend state or inject
  an IMU translation estimate into the pose.
- `imu_shock_policy_enabled` defaults to `false` independently of roughness
  detection so existing outdoor profiles retain their previous behavior. When
  enabled, a roughness event enters `shock`, remains active for
  `imu_shock_hold_duration_s`, and then requires
  `imu_shock_recovery_stable_scans` consecutive stable scans before returning
  through `recovering` to `stable`. During `shock` or `recovering`, the indoor
  profiles suppress the constant-velocity prediction and quarantine
  initialization, keyframes, tracking history, submap transitions, place
  recognition, and persistent map mutation. The registered pose may still be
  published when `quarantine_pose_on_shock=false`; this avoids freezing odometry
  while preventing uncertain scans from becoming long-lived map evidence.
  Roughness is evaluated over the decoded minimum/maximum point times, not only
  the PointCloud2 header stamp, so an impact near the end of a scan affects that
  same scan. Gravity filtering also excludes detected impact samples while this
  policy is active. `/slam_3d/health` reports the phase, hold/recovery state,
  triggering metrics, constant-velocity suppression, and quarantine reason.
- `terrain_evidence_enabled` defaults to `false` and is enabled by the indoor
  profiles. It classifies gravity/map-aligned keyframe points into intensity
  labels `1=floor`, `2=ramp`, `3=positive step`, `4=observed drop`, and
  `5=staircase`, then publishes the persistent PointXYZI cloud on
  `/slam_3d/terrain/evidence`.
  Classification uses lower-height radial profiles, coherent slope runs, and
  supported height discontinuities. Indoor profiles additionally require each
  radial ramp candidate to belong to a gravity-relative robust local 3D plane
  with configured point count, length, lateral width, RMSE, and inlier-ratio
  support. This removes wall-edge and single-ray inclines that have slope but
  no traversable surface width. Missing returns alone remain `unknown` and
  never become a drop. A terrain voxel is published only after
  `terrain_min_confirmations` observations whose sensor viewpoints are separated
  by `terrain_confirmation_min_viewpoint_distance_m`. A single viewpoint casts
  no vote when its dominant label is below
  `terrain_min_view_label_confidence`; tied floor/hazard points therefore remain
  ambiguous instead of being promoted to a hazard. The floor seed height gate is
  applied to every radial-profile bin so shelves cannot join a valid floor seed
  found in another azimuth. Labels stay attached to keyframe-local points and
  are reprojected after pose-graph corrections. This
  output is 3D evidence for later traversability/projection work, not a Nav2
  occupancy or costmap product. Merged dual-LiDAR clouds currently preserve
  geometry but not per-sensor ray identity, so negative-obstacle occlusion
  reasoning must remain a future ingest/backend stage.
  The XT32 profiles use a 6 m seed window and a base-height prior with
  independently supported height modes because the
  sensor's useful downward floor returns in the indoor replay begin near 3 m;
  the old 3 m/20% seed selected wall and shelf bands around 0.7 m as floor. The
  supported mode also prevents a lower stair landing from replacing the current
  robot floor merely because more low returns are visible.
  Dual-JT128 keeps its own denser-cloud parameters rather than inheriting this
  sensor-specific tuning.
  `/slam_3d/terrain/evidence_colored` publishes the same confirmed voxels as
  PointXYZRGB for direct review in RViz or PCL: floor `(105,190,105)`, ramp
  `(45,165,255)`, positive step `(255,165,35)`, directly observed lower support
  `(235,55,55)`, and staircase `(190,90,245)`. The red points represent an
  observed lower surface, not an inferred empty void.
  `/slam_3d/terrain/support_evidence` additionally publishes every retained
  support/hazard voxel with explicit state and confidence fields. Confirmed floor
  is dark green, probable floor is light green, confirmed ramp is blue, probable
  ramp is cyan, probable hazard is yellow, confirmed hazards keep their semantic
  colors, and unresolved evidence is gray. This topic is intended for offline
  continuous-cost projection and review. It does not authorize unknown space as
  free and is not a direct Nav2 occupancy-map input.
- The indoor profiles currently leave `gravity_roll_pitch_constraint_enabled`,
  `imu_relative_roll_pitch_constraint_enabled`, and
  `ground_vehicle_motion_constraint_enabled` off. In the bag03 full A/B,
  enabling the uncalibrated accelerometer/body-z pair increased trajectory z
  range from the previous 0.359 m to 1.265 m; disabling it while retaining IMU
  deskew and shock detection produced 0.267 m. Re-enable only after physical
  IMU extrinsic, accelerometer bias, and dynamic-acceleration rejection are
  measured for the installed sensor.
- `map_z_filter_enabled` defaults to `false`. It filters only the stored
  keyframe map clouds after transforming them into the map frame. The incoming
  scan and scan-matching local map policy are otherwise unchanged. This is
  useful for PCD review runs where gravity/registration outliers create sparse
  high or low map points that dominate the viewer scale.
- `use_scan_to_scan_prediction` and `use_scan_to_scan_fallback` default to
  `false`. They are intended for offline odom-free experiments such as HILTI
  bag replay, not as the final production LIO frontend.
  The scan-to-scan IMU initial guess is integrated from the timestamp of the
  actual retained tracking scan, which can be older than the previous incoming
  cloud during pose quarantine. `scan_to_scan_max_reference_age_s > 0` rejects
  a stale tracking reference before registration; `0.0` leaves the age gate
  disabled for compatibility.
- `deskew_enabled` defaults to `false`. When enabled, the node reads per-point
  acquisition time from the incoming `PointCloud2`, checks IMU coverage across
  the scan, integrates angular velocity, and rotates each point back to a
  reference time clamped inside the observed scan time range. The
  `/slam_3d/health` diagnostics include
  `deskew_status`, `deskew_applied`, point-time coverage, and scan duration so
  bag replays can verify whether deskew really ran.
  `deskew_time_bin_s` defaults to `0.001` so the integrated rotation is cached
  at 1 ms resolution instead of recomputing an exponential map for every raw
  point.
  `deskew_point_time_offset_s` shifts decoded point timestamps into the IMU
  clock when a calibration file reports LiDAR-to-IMU time offset.
  `imu_require_expected_frame` can reject an IMU stream whose frame does not
  match `imu_expected_frame`; non-finite gyro or acceleration samples are
  always rejected before they can advance the deskew watermark.
- `add_scan_to_scan_fallback_keyframes` defaults to `false`. A fallback pose can
  preserve odometry continuity, but it is not scan-to-map verified and should
  not contaminate the map by default.
- `motion_prior_fallback_max_*` limits default to `0.0` (disabled). For
  odom-free HILTI-style replays, enable these caps to keep scan-to-scan,
  motion-prior fallback, or the first re-acquired scan-to-map pose from running
  away in z or jumping far from the last scan-to-map-verified pose during a
  construction-site shock segment.
- `publish_tf` defaults to `false` so this node does not fight the existing
  `localization` node or AMCL/Nav2 TF owners. When explicitly enabled, this
  learning frontend publishes its estimated `map -> base_link` transform
  directly; it does not yet implement a `map -> odom -> base_link` split.

The production direction remains a FAST-LIO2-style LIO frontend plus a
submap/pose-graph backend. This package is the first runnable learning layer
under that architecture.

## Run

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py
```

Launch arguments that overlap node parameters default to an empty value. An
empty argument preserves the selected `params_file`; only explicitly supplied
launch values override that profile.

Useful overrides:

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py registration_method:=icp
ros2 launch slam_3d_core scan_to_map_icp.launch.py use_secondary_cloud:=false
ros2 launch slam_3d_core scan_to_map_icp.launch.py use_tertiary_cloud:=false
ros2 launch slam_3d_core scan_to_map_icp.launch.py publish_tf:=true
```

For the Geoje XT32-only LiDAR-IMU deskew and robust pose-graph replay profile:

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py \
  params_file:=$(ros2 pkg prefix slam_3d_core)/share/slam_3d_core/config/geoje_xt32_lidar_imu_pose_graph_params.yaml
```

This profile uses the front JT128 IMU only as the XT32 deskew source. Its
`deskew_imu_to_base_*` rotation must be revalidated whenever the sensor mount,
driver frame convention, or IMU axis convention changes.

For the Geoje XT32 plus `/imu/data` profile with wheel odometry disabled:

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py \
  params_file:=$(ros2 pkg prefix slam_3d_core)/share/slam_3d_core/config/geoje_xt32_main_imu_no_wheel_pose_graph_params.yaml
```

This profile consumes only the XT32 timed cloud and `/imu/data`. It sets both
wheel-dependent features to false, which prevents the core from creating a
wheel odometry subscription. The recorded `base_link -> cv7_link` rotation is
identity; IMU gaps longer than the configured deskew bound quarantine the
affected cloud instead of being extrapolated. The recorded accelerometer has
approximately `-9.8 m/s^2` on `cv7_link` z while stationary, so
`gravity_accel_scale=-1.0` corrects the acceleration used by the roll/pitch
gravity constraint without changing the gyro used for point deskew. A bounded
body-frame z constraint removes motion normal to the vehicle plane. Because the
constraint is expressed in the previous gravity-aligned body frame, forward
motion on a ramp still produces map elevation while the weak surfel z mode
cannot accumulate. Full-SE(3) constant-velocity prediction is enabled only as
the scan-to-map initial guess; the registered candidate is still passed through
the body-frame z and gravity constraints before it becomes a committed pose.

Only use `publish_tf:=true` when no other node owns a conflicting transform to
`base_link`. The production backend should eventually publish `map -> odom`
while a state estimator owns `odom -> base_link`; that split is not implemented
by this learning frontend.

## Main Parameters

Parameters live in `config/scan_to_map_icp_params.yaml`.

- Inputs:
  - `/slam_3d/input/hesai_center/points_timed` as the XT32 trigger
  - `/slam_3d/input/hesai_front/points_timed` for the front JT128
  - `/slam_3d/input/hesai_rear/points_timed` for the rear JT128
  - explicitly remapped legacy dataset PointCloud2 topics for offline replay
  - `/motor_driver_node/odom`
  - `/imu/data`
- Outputs:
  - `/slam_3d/frontend/odom`
  - `/slam_3d/frontend/path`
  - `/slam_3d/map/active_submap`
  - `/slam_3d/map/global_cloud` as temporally unfiltered mapping geometry
  - `/slam_3d/terrain/evidence` as persistent indoor 3D terrain labels when
    `terrain_evidence_enabled=true`
  - `/slam_3d/terrain/evidence_colored` as the fixed semantic RGB view of the
    same confirmed terrain voxels
  - `/slam_3d/health`
  - `/slam_3d/imu_watermark_ns`
  - `/slam_3d/frontend/deskewed_points` only when
    `publish_preprocessed_scan=true`; this is the base-frame scan after deskew,
    range/z/self crop, and adaptive voxel filtering

## Next Code Milestones

1. Run longer Geoje and independent outdoor datasets with fixed replay windows,
   then promote parameters only when map overlays and trajectory metrics agree.
2. Add a geometry-observable test topic for inter-LiDAR translation and
   rotation direction, beyond the current correction magnitude/status tests.
3. Extend deskew from packet-level gyro-only rotation correction to corrected
   channel firing time plus full LIO propagation with
   measured IMU-to-LiDAR extrinsics and translation compensation.
4. Replace wheel odom pose-delta prediction with a covariance-weighted planar
   velocity prior and slip rejection.
5. Promote the optional ring/sector descriptor into a tested Scan Context or
   Quatro-style place recognition module, then evaluate an incremental
   GTSAM/iSAM2 smoother against the current batch Ceres backend.
6. Add point-wise deskew covariance and vibration-aware registration weights so
   an impact is handled continuously in addition to frame-level quarantine.
7. Calibrate the new directional pose-graph information against independent
   corridor, stair, and open-space datasets, then compare it with a
   rematching-aware registration covariance estimator.
8. Preserve sensor/ring identity through multi-LiDAR ingest, then add explicit
   free-space and occlusion evidence for negative-obstacle confirmation.

## References

- Hesai ROS Driver 2.0: https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0
- Hesai LiDAR SDK 2.0: https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0
- FAST-LIO / FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- GLIM: https://github.com/koide3/glim
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- Trimmed ICP: https://doi.org/10.1109/ICPR.2002.1047997
- X-ICP: https://arxiv.org/abs/2211.16335
- An Accurate Closed-form Estimate of ICP's Covariance (Censi):
  https://censi.science/pub/research/2007-icra-icpcov.pdf
- On the Covariance of ICP-based Scan-matching Techniques (Bonnabel et al.):
  https://arxiv.org/abs/1410.7632
- Vibration-Aware LiDAR-Inertial Odometry: https://arxiv.org/abs/2507.04311
- TRAVEL traversability estimation: https://arxiv.org/abs/2206.03190
- Fast Staircase Detection and Estimation: https://arxiv.org/abs/2211.00610
- G-VOM elevation mapping: https://arxiv.org/abs/2109.13176
- Wildcat continuous-time LiDAR-inertial SLAM: https://arxiv.org/abs/2205.12595
- Dynamic Covariance Scaling: https://doi.org/10.1109/ICRA.2013.6630557
- Ceres Solver 3D pose graph example:
  https://ceres-solver.readthedocs.io/latest/nnls_tutorial.html
- Switchable Constraints for Robust Pose Graph SLAM:
  https://nikosuenderhauf.github.io/assets/papers/IROS12-switchableConstraints.pdf
- PCL registration module:
  https://pointclouds.org/documentation/group__registration.html
