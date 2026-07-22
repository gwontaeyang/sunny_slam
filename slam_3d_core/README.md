# slam_3d_core

Learning-oriented GOLE 3D SLAM core code. This is direct C++ implementation,
not a wrapper around FAST-LIO, LIO-SAM, GLIM, KISS-ICP, or another open-source
runtime.

The first node is `scan_to_map_icp_node`. It is intentionally smaller than a
production LiDAR-inertial SLAM stack so the algorithm can be edited and studied
inside this repository.

## Current Algorithm

```text
PointCloud2 front/rear Mid-360 debug streams
  -> TF transform into base_link
  -> optional IMU gyro rotation deskew when PointCloud2 per-point time fields
     are preserved
  -> range/z/self crop
  -> voxel downsample, with optional smaller adaptive voxel when a scan becomes
     too sparse after downsampling
  -> wheel odom relative-motion prediction, or bounded constant-velocity
     fallback when wheel odom is unavailable
  -> optional IMU gyro rotation prediction between scan header stamps
  -> optional scan-to-scan odometry prediction for odom-free dataset replay
  -> scan-to-local-keyframe-map ICP or GICP
  -> optional trimmed coarse registration plus narrow-distance refinement
  -> optional place recognition and lightweight pose-graph relaxation
  -> keyframe cloud map
  -> /slam_3d/frontend/odom, /slam_3d/frontend/path, map clouds, diagnostics
```

Implemented in:

- `include/slam_3d_core/scan_to_map_icp.hpp`
- `src/scan_to_map_icp.cpp`
- `src/scan_to_map_icp_node.cpp`

This is the editable baseline for learning scan registration, keyframe policy,
local map construction, and dual-cloud synchronization.

## Deliberate Limitations

- IMU data is buffered for an optional gyro-only rotation deskew step. This MVP
  requires a `PointCloud2` field such as `timestamp`, `time`, `t`, or
  `offset_time`; it does not estimate translation during the scan and is not
  IMU preintegration.
- Mid-360 raw/custom fields are still not preserved by the current debug-cloud
  producers. Deskew only has enough information when the upstream replay or raw
  ingest publishes `PointCloud2` with per-point time fields intact.
- There is still no production-grade loop closure, submap persistence, or
  relocalization. The current backend is a small internal keyframe pose graph:
  odometry edges are added between consecutive keyframes, a loop edge is added
  when place recognition is accepted, and a bounded iterative relaxation moves
  keyframe poses, reprojects their stored base-frame clouds, and updates the
  published path history. It is designed to be edited and inspected before
  replacing it with GTSAM/iSAM2 or another robust factor-graph backend.
- The constant-velocity fallback is only an ICP initial guess. It is not a
  replacement for IMU preintegration, deskew, or a backend optimizer.
- `imu_rotation_prediction_enabled` defaults to `false`. When enabled, the node
  integrates buffered gyro samples between consecutive processed scan header
  stamps, preserves the translation from the existing motion prior, replaces
  the predicted orientation with the previous pose plus that gyro delta, and
  uses the same rotation as the initial guess for optional scan-to-scan
  prediction. It does not estimate translation, velocity, bias, or gravity; it
  is a narrow shock robustness aid for odom-free replay and debugging.
  `imu_rotation_prediction_scale` is separate from
  `deskew_angular_velocity_scale` so replay can test the inter-scan rotation
  sign without corrupting per-point deskew.
- IMU ingestion uses a dedicated mutually-exclusive callback group and the node
  executable runs a two-thread executor. Registration, map publication, and the
  other default callbacks remain mutually exclusive, while the IMU buffer can
  continue receiving high-rate samples during a slow ICP or pose-graph update.
  `imu_subscription_depth` defaults to 400 instead of the depth-5 sensor QoS
  default so a one-second 400 Hz burst is not silently reduced to its last few
  samples. Access to the deskew/gravity buffer remains protected by
  `imu_mutex_`.
- Registration is accepted only when convergence, bounded-distance fitness, and
  prediction-to-result correction limits pass. The correction gate also checks
  aligned-cloud inlier ratio/RMSE against the local map so a small set of close
  correspondences cannot hide weak overlap.
- `robust_registration_enabled` defaults to `false`. When enabled, point-to-point
  ICP discards the farthest correspondence fraction using PCL's trimmed
  correspondence rejector, performs the normal wide-distance registration, and
  then starts a second refinement from that pose with
  `robust_registration_refinement_max_correspondence_m`. GICP uses the same
  coarse-to-fine schedule but does not use PCL's ICP-only trimmed rejector.
  Acceptance fitness and overlap are recomputed with the independent, narrower
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
  the quarantine thresholds are not inserted as keyframes. This keeps sudden
  threshold/door-frame shocks from immediately contaminating the persistent map.
  The same shock frame is also kept out of the scan-to-scan tracking history and
  constant-velocity accepted-pose history, so one bad correction does not become
  the seed for the next few frames.
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
  iteration count, and max per-iteration update. This is still intentionally
  lightweight and lacks covariance modeling, IMU preintegration, and incremental
  relinearization.
  `pose_graph_robust_loop_enabled` optionally applies a Cauchy IRLS weight to
  loop edges only. Consecutive odometry edges retain their configured weights.
  `pose_graph_loop_cauchy_scale` controls the transition from least-squares-like
  behavior to strong downweighting. Raw squared residual and robust objective
  are reported separately, together with the newest/minimum loop weight and the
  count of downweighted loop edges. This is an inspectable approximation of the
  robust pose-graph treatment used by systems such as Wildcat, not a replacement
  for a sparse nonlinear solver with properly modeled factor covariances.
  `pose_graph_max_residual_after` and `pose_graph_max_residual_ratio` can reject
  a loop edge even when optimization technically reduces residual; this is useful
  for repeated construction-site geometry where a false loop can lower the
  scalar graph error but still leave meter-scale inconsistency.
  `pose_graph_max_loop_residual_before` is a cheaper pre-optimization gate for
  the prospective loop edge itself. Keep it at `0.0` for disabled/default runs;
  enable it in HILTI-style sweeps when rejected loop candidates are spending too
  much time in full graph relaxation.
  `pose_graph_planar_correction_enabled` can be enabled for HILTI-like shaky
  handheld or mostly-ground-robot tests so loop closure corrects x/y/yaw while
  leaving z/roll/pitch to IMU/deskew and local registration. This avoids a bad
  3D loop match tilting an entire construction-site map.
- `planar_pose_constraint_enabled` defaults to `false`. When enabled, accepted
  frontend poses keep x/y/yaw from registration but force roll/pitch to the map
  gravity frame and optionally clamp z correction around the motion prior with
  `planar_pose_max_z_correction_m`. This is useful when an offline dataset has
  no valid IMU orientation and scan matching starts fitting repeated vertical
  structures by tilting keyframes. It is a debugging constraint, not a full
  replacement for gravity-aligned IMU preintegration.
- `gravity_roll_pitch_constraint_enabled` can be combined with the planar pose
  constraint. It low-pass filters IMU linear acceleration, rejects samples whose
  acceleration norm is outside `gravity_accel_min_norm_mps2` to
  `gravity_accel_max_norm_mps2`, and uses the resulting gravity roll/pitch with
  scan-matched yaw. This is more appropriate than forcing zero roll/pitch for a
  handheld or a robot crossing a bump, but it still ignores acceleration bias
  and dynamic acceleration except for the simple norm gate.
- `imu_roughness_detection_enabled` defaults to `false`. When enabled, the node
  computes per-scan IMU roughness diagnostics over the latest scan interval or
  `imu_roughness_window_s`, whichever is shorter. Diagnostics include max gyro
  norm, angular acceleration, linear acceleration norm change, and linear jerk.
  This does not change the pose estimate yet; it is for finding construction
  shocks such as threshold crossings before using those events to gate
  keyframes, tracking history, or loop validation.
- `map_z_filter_enabled` defaults to `false`. It filters only the stored
  keyframe map clouds after transforming them into the map frame. The incoming
  scan and scan-matching local map policy are otherwise unchanged. This is
  useful for PCD review runs where gravity/registration outliers create sparse
  high or low map points that dominate the viewer scale.
- `use_scan_to_scan_prediction` and `use_scan_to_scan_fallback` default to
  `false`. They are intended for offline odom-free experiments such as HILTI
  bag replay, not as the final production LIO frontend.
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
- `add_scan_to_scan_fallback_keyframes` defaults to `false`. A fallback pose can
  preserve odometry continuity, but it is not scan-to-map verified and should
  not contaminate the map by default.
- `motion_prior_fallback_max_*` limits default to `0.0` (disabled). For
  odom-free HILTI-style replays, enable these caps to keep scan-to-scan,
  motion-prior fallback, or the first re-acquired scan-to-map pose from running
  away in z or jumping far from the last scan-to-map-verified pose during a
  construction-site shock segment.
- `publish_tf` defaults to `false` so this node does not fight the existing
  `localization` node or AMCL/Nav2 TF owners.

The production direction remains a FAST-LIO2-style LIO frontend plus a
submap/pose-graph backend. This package is the first runnable learning layer
under that architecture.

## Run

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py
```

Useful overrides:

```bash
ros2 launch slam_3d_core scan_to_map_icp.launch.py registration_method:=icp
ros2 launch slam_3d_core scan_to_map_icp.launch.py use_secondary_cloud:=false
ros2 launch slam_3d_core scan_to_map_icp.launch.py publish_tf:=true
```

Only use `publish_tf:=true` when existing `odom -> base_link` and `map -> odom`
publishers are disabled.

## Main Parameters

Parameters live in `config/scan_to_map_icp_params.yaml`.

- Inputs:
  - `/slam_3d/input/mid360_front/points_debug`
  - `/slam_3d/input/mid360_rear/points_debug`
  - `/motor_driver_node/odom`
  - `/imu/data`
- Outputs:
  - `/slam_3d/frontend/odom`
  - `/slam_3d/frontend/path`
  - `/slam_3d/map/active_submap`
  - `/slam_3d/map/global_cloud`
  - `/slam_3d/health`

## Next Code Milestones

1. Add a synthetic registration unit test that creates two translated clouds
   and verifies ICP/GICP convergence.
2. Add a Livox raw/custom ingest layer that preserves per-point time,
   reflectivity, tag, and source sensor id.
3. Extend deskew from gyro-only rotation correction to full LIO propagation with
   measured IMU-to-LiDAR extrinsics and translation compensation.
4. Replace wheel odom pose-delta prediction with a covariance-weighted planar
   velocity prior and slip rejection.
5. Promote the optional ring/sector descriptor into a tested Scan Context or
   Quatro-style place recognition module and replace the internal relaxation
   backend with a robust GTSAM/iSAM2 factor graph.
6. Add shock/dropout diagnostics from IMU and odom deltas so keyframe
   quarantine is driven by sensor residuals, not only ICP correction size.
7. Split keyframes into submaps and add a pose graph backend with loop closure.

## References

- Livox ROS Driver 2: https://github.com/Livox-SDK/livox_ros_driver2
- FAST-LIO / FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- GLIM: https://github.com/koide3/glim
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- Trimmed ICP: https://doi.org/10.1109/ICPR.2002.1047997
- X-ICP: https://arxiv.org/abs/2211.16335
- Wildcat continuous-time LiDAR-inertial SLAM: https://arxiv.org/abs/2205.12595
- Dynamic Covariance Scaling: https://doi.org/10.1109/ICRA.2013.6630557
- PCL registration module:
  https://pointclouds.org/documentation/group__registration.html
