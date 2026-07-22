# GOLE Dual Mid-360 Large-Scale Outdoor 3D SLAM Architecture

## Objective

Build a GOLE-owned 3D SLAM stack for very large outdoor environments using:

- Two Livox Mid-360 LiDARs.
- Existing wheel odometry from `/motor_driver_node/odom`.
- Existing external IMU from `/imu/data`.

This package is the contract and bringup shell for that stack. It should not be
treated as a wrapper around a third-party SLAM backend.

## Target Pipeline

```text
Mid-360 front raw  ─┐
                   ├─ sensor ingest ─ time/extrinsic validation ─┐
Mid-360 rear raw   ┘                                             │
                                                                  ├─ LIO frontend
/imu/data ───────────────────── IMU propagation / deskew ─────────┤
                                                                  │
/motor_driver_node/odom ─────── wheel velocity/slip prior ────────┘

LIO frontend ─ keyframes ─ submap manager ─ pose graph backend ─ map store
                       │                       │
                       │                       └─ loop closures / graph metrics
                       └─ local odometry / local map / degeneracy metrics

map store ─ localization mode ─ controlled bridge to Nav2 odom/TF contract
```

## Module Boundaries

### 1. Sensor Ingest

Responsibilities:

- Receive each Mid-360 stream independently.
- Preserve Livox per-point offset time and any tag/line/intensity fields.
- Publish PointCloud2 only as debug/recording output after the raw contract is
  preserved.
- Record packet counts, header stamps, hardware stamps, and drop counters.

Non-goals:

- No PointXYZ conversion before LIO.
- No naive front/rear cloud concatenation as the primary LIO input.
- No 2D LaserScan projection before mapping.

### 2. Calibration And Time Validation

Responsibilities:

- Maintain `base_link -> mid360_front_link`, `base_link -> mid360_rear_link`,
  `base_link -> imu_link`, and LiDAR-to-LiDAR extrinsics.
- Measure front/rear LiDAR-to-IMU and wheel-to-IMU timing offsets.
- Refuse high-confidence SLAM evaluation until timestamp skew is measured.

Large outdoor mapping will hide small calibration errors until loops are long.
The first milestone is therefore not mapping, but repeatable static and short
loop bags that prove timing and extrinsics.

### 3. Frontend LIO

Recommended first implementation:

- FAST-LIO2-style iterated EKF.
- Direct point-to-local-map residuals.
- IMU propagation and per-point deskew.
- Two LiDAR measurement batches update one robot state.
- Wheel odometry provides planar velocity prior and slip detection.

The frontend publishes local odometry and local map products, but it should not
claim global consistency. It must expose covariance/degeneracy so the backend
can decide how strongly to trust each keyframe edge.

### 4. Submap Manager

Responsibilities:

- Convert stable keyframes into overlapping submaps.
- Downsample submaps independently.
- Attach descriptors and bounding boxes.
- Keep the active local submap hot in memory and older submaps tile-indexed.

Large sites should not be handled as one dense PCD. Submap storage is a first
class requirement, not a later optimization.

### 5. Pose Graph Backend

Responsibilities:

- Add odometry factors from frontend keyframes/submaps.
- Add loop factors only after descriptor retrieval and geometric verification.
- Use robust kernels and switchable constraints.
- Publish optimization timing, residuals, node/edge counts, and loop stats.

Optional factors:

- Wheel planar motion factor.
- GNSS factor if available later.
- Manual anchor or survey control factor.

### 6. Loop Closure

Two-stage loop closure is required:

1. Candidate search by descriptor, such as Scan Context family descriptors.
2. Geometric verification by submap registration, such as GICP/NDT/Quatro-like
   registration, with overlap/fitness/transform gates.

Every rejected loop must be logged with the rejection reason. Silent loop
failures make large maps impossible to debug.

### 7. Map Store And Localization

Map store format should include:

- Submap point cloud file.
- Submap pose.
- Descriptor.
- Bounding box/tile index.
- Pose graph nodes/edges.
- Calibration and pipeline version metadata.

Localization mode should load only nearby submaps and publish a controlled
bridge to the existing GOLE Nav2 contract after integration review.

## Existing GOLE Integration

Existing consumers expect:

- `/motor_driver_node/odom` from `motor_driver`.
- `/imu/data` from the external IMU stack.
- `/odom` and `map -> odom -> base_link` for Nav2/localization workflows.
- `base_link` as robot base frame.

Conflict to avoid:

- `gole_localization/odom_estimation` publishes `/odom` and `odom -> base_link`.
- AMCL/localization in `gole_apart` owns `map -> odom` during 2D navigation.
- A 3D SLAM mapping run must not publish the same TF chain at the same time
  unless the existing localization/AMCL stack is intentionally disabled.

## Milestones

1. Topic and bag contract.
2. Static time/extrinsic validation.
3. Single-front Mid-360 LIO.
4. Single-rear Mid-360 LIO.
5. Dual measurement update against one state.
6. Keyframe and submap manager.
7. Pose graph with odometry edges only.
8. Loop closure candidate generation.
9. Geometric loop verification and robust graph optimization.
10. Persistent submap store.
11. Localization mode and Nav2 bridge.

## Done Criteria For First Outdoor Prototype

- Processes a 10+ minute dual-Mid-360 outdoor bag offline.
- Runs frontend faster than bag time on the development PC.
- Keeps bounded memory with submaps.
- Produces trajectory, local submaps, global map, and pose graph artifacts.
- Reports drift, z drift, roll/pitch, loop statistics, and runtime metrics.
- Clearly rejects bad loops instead of accepting visually plausible but wrong
  closures.
