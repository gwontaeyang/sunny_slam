# Large-Scale Outdoor 3D SLAM Architecture

## Scope

The current robot uses two Hesai JT128 sensors, one Hesai XT32, wheel odometry,
and an external IMU. The target environment is a large outdoor construction
site with intermittent shocks, wheel slip, repeated structures, and long loops.

The implementation is intentionally split into inspectable layers:

1. hardware driver;
2. input contract and time normalization;
3. multi-LiDAR queue synchronization;
4. per-sensor transform and deskew;
5. scan-to-local-map frontend;
6. submaps and place recognition;
7. robust pose graph and map persistence.

## Sensor Boundary

The official Hesai ROS 2 wrapper publishes all three sensors as PointCloud2.
The Geoje data uses the same field layout for JT128 and XT32, but their point
counts and ring ranges differ. `slam_3d_lidar_ingest` validates those
model-specific constraints and emits one canonical cloud per sensor.

It intentionally does not merge LiDARs. Merging before TF and deskew would make
one cloud contain points acquired in different frames and time intervals.

Canonical packet-time equation:

```text
t_packet = cloud.header.stamp + offset_time * 1e-9
```

This is not yet a channel firing-time equation. The observed Hesai timestamp is
repeated for 256 points. A production wrapper should publish SDK-corrected
firing time if channel-level deskew is required.

## Time Architecture

Production requires a common measured clock domain across:

- front JT128;
- rear JT128;
- center XT32;
- host ROS clock;
- external IMU.

The implementation currently checks header-to-receive delta and monotonicity,
but that is not proof of PTP lock. PTP status, offset, rate, and resets must be
monitored independently.

The Geoje bags violate the common-clock requirement. Their bounded replay
profile fits a fixed affine conversion from each sensor header clock to bag
record time minus observed scan duration. Fixed anchors remove callback-order
jitter and the fitted scale models local sensor-clock drift. Transport latency,
recorder latency, and true LiDAR-to-IMU offset remain unknown. Use it for
geometry, repeatability, and robustness experiments only.

## Multi-LiDAR Synchronization

XT32 is the trigger sensor because it provides the center 360-degree stream.
Each JT128 has a bounded queue. For every XT32 scan, the frontend:

1. inserts every stream in timestamp order and counts reordering;
2. removes JT128 messages that are irrecoverably older than the allowed window;
3. selects and consumes the nearest front and rear start stamps exactly once;
4. uses each auxiliary sensor's timestamp watermark plus a bounded reorder
   window to decide that a match can no longer arrive;
5. if an auxiliary stream stops, uses primary-sensor time progress beyond a
   dropout horizon to resolve the missing input deterministically;
6. retains wall timeout only for the final primary scan or complete stream
   stall, then records degraded input;
7. waits for a wheel timestamp watermark before requiring exact/interpolated
   wheel brackets for inter-LiDAR motion compensation.

This replaces the old latest-secondary snapshot behavior, which could reuse one
scan more than once or miss a delayed scan.

## Frontend Order

For every selected cloud:

1. look up `base_link <- sensor_frame`;
2. reject that sensor cloud if TF is unavailable;
3. parse `offset_time`;
4. transform sensor-frame points to `base_link`;
5. integrate base-frame gyro samples across the packet-time interval;
6. rotate points to that sensor scan's reference time.

After per-sensor deskew, wheel odom is interpolated at the auxiliary and XT32
start stamps. Each available auxiliary cloud is transformed from its acquisition
base pose into the XT32 base pose. Only then are clouds concatenated,
body-cropped, range-filtered, voxelized, and registered to the local map.

Current deskew is rotation-only. Full construction-site robustness requires
velocity, bias, gravity, and translation propagation in an error-state LIO.

## Registration And Shock Handling

The editable baseline supports ICP, GICP, and a direct voxel-surfel
point-to-plane scan-to-local-map implementation with:

- wheel odometry and constant-velocity initial guesses;
- optional IMU rotation prediction;
- robust trimmed refinement;
- correction magnitude and overlap checks;
- IMU roughness detection;
- pose and keyframe quarantine during shocks;
- scan-context-family place candidates;
- geometric loop verification;
- a selectable pose-graph backend: legacy edge-wise relaxation for the held
  outdoor profile, or staged sparse Ceres SE(3) least squares with Cauchy and
  DCS for profiles that have completed replay validation.

The Geoje bounded replay currently selects voxel-surfel. GICP remains the
production-profile default until a synchronized live commissioning dataset
supports a broader A/B decision.

## Production Target

The next frontend should be an iterated error-state LIO with position, velocity,
orientation, accelerometer bias, gyroscope bias, and gravity. Optional
calibration states can include per-LiDAR time offsets and extrinsics after
observability is established.

The backend should operate on persistent overlapping submaps, use verified loop
factors with robust kernels, and report solver convergence, timing, residuals,
and accepted/rejected loop evidence.

## Required Validation

1. Synthetic field, frame, ring, point-count, and queue tests.
2. Static Geoje contract replay.
3. Center/front/rear single-sensor geometry maps.
4. Three-LiDAR map with packet deskew on/off.
5. Shock-segment trajectory and map-thickness ablation.
6. Repeated-area loop closure before/after overlays.
7. LiDAR dropout, timestamp reset, and TF-missing tests.
8. Live static PTP recording.
9. Live slow loop, then rough large loop.
10. Long-duration memory and map persistence test.

## TF Ownership

The development stack publishes under `/slam_3d/...` and does not publish TF by
default. The current learning core has only an optional direct
`map -> base_link` prototype transform. A future mapping backend may own the
production `map -> odom -> base_link` split only after existing localization
and AMCL owners are disabled.
