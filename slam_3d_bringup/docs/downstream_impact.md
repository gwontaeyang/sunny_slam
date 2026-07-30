# Downstream Impact For 3D SLAM Integration

This document records the local code paths that will be affected when the
large-scale 3D SLAM stack starts publishing navigation-facing odometry or TF.

## Current Odometry And TF Owners

### Wheel odometry source

- Package: `gole_core/device/driver/motor_driver`
- Topic: `/motor_driver_node/odom`
- Evidence:
  - `device/driver/motor_driver/src/motor_driver.cpp` creates `~/odom`.
  - Odometry messages use `odom` and `base_link` frame ids.

This is an input to 3D SLAM. It should not become the final map-frame pose.

### Existing rover localization

- Package: `gole_localization/odom_estimation`
- Publishes: `/odom`
- Publishes TF: `odom -> base_link`
- Inputs: `/motor_driver_node/odom`, `/imu/data`
- Evidence:
  - `localization/src/localization.cpp` creates publisher `/odom`.
  - `localization/src/localization.cpp` sends `odom -> base_link` in the timer.

Impact:

- During 3D SLAM mapping, do not run this node as a TF owner.
- If it must run for diagnostics, it should be topic-only or namespaced away
  from `/odom` and `odom -> base_link`.

### Apartment navigation localization

- Package: `gole_apart/slam/localization_bringup`
- Owner: AMCL/map server stack for 2D localization.
- `gole_apart/navigation/nav_bringup` includes localization when
  `localization:=true`.
- Nav2 config consumes:
  - `global_frame: map`
  - `robot_base_frame: base_link`
  - `odom_topic: /odom` or `odom`

Impact:

- 3D SLAM must not publish `map -> odom` while AMCL owns that transform.
- Nav2 bridge from `/slam_3d/backend/odom` to `/odom` should be introduced only
  after a dedicated navigation integration review.

### MicroStrain example launch files

- Package: `gole_third_party/microstrain_inertial`
- Example launch files can publish static transforms such as `map -> base_link`
  and placeholder sensor extrinsics.

Impact:

- Do not use example IMU launch files unchanged in a 3D SLAM mapping run.
- Use production IMU driver config with explicit `base_link -> imu_link`
  extrinsic and no conflicting map/base static transform.

## Current LiDAR Products

### `lidar_merger`

- Package: `gole_core/device/lidar/lidar_merger`
- Inputs: SICK/picoScan-style PointCloud2 topics.
- Output: `/scan` PointCloud2 and `/scan_laserscan`.
- Frame: `base_link` output after TF transform.

Impact:

- This remains useful for perception and 2D localization products.
- It is not a raw Mid-360 LIO input path.

### `lidar_preprocessing`

- Package: `gole_core/device/lidar/lidar_filter`
- Converts `sensor_msgs/PointCloud2` to `pcl::PointXYZ`.

Impact:

- Do not place it before the 3D LIO frontend.
- It may be used after registration for visualization, downsampled map products,
  or consumer-specific filtering.

## Plant And Apartment Consumers

Known consumers of `/odom`, `map`, `odom`, and `base_link` include:

- `gole_apart/navigation/nav_bringup/config/navigator_params.yaml`
- `gole_apart/navigation/nav_bringup/config/controller_params.yaml`
- `gole_apart/navigation/nav_bt_navigator`
- `gole_apart/navigation/nav_bt_node`
- `gole_plant/motion_controller`
- `gole_plant/docking_module`

Rule:

- Keep all 3D SLAM outputs under `/slam_3d/...` until the bridge to existing
  navigation topics is intentionally designed and tested.

## Integration Modes

### Architecture-only mode

- Default `slam_3d_bringup` behavior.
- No SLAM TF.
- No static placeholder TF unless explicitly requested.
- Safe for source-tree and launch argument checks.

### Sensor recording mode

- Run Livox driver and IMU/wheel recording.
- No 3D SLAM TF.
- Record raw Livox custom topics, validated `points_timed`, ingest diagnostics,
  IMU, wheel odom, `/tf`, and `/tf_static`. Debug PointCloud2 is optional.

### Mapping mode

- Disable existing rover localization TF owner.
- Disable AMCL/localization stack.
- 3D SLAM backend owns `map -> odom -> base_link`.
- Publish navigation bridge only after review.

### Localization mode

- Load submap store and graph metadata.
- Publish `map -> odom -> base_link` only as the selected localization owner.
- Keep AMCL disabled unless explicitly comparing localization stacks offline.

## Review Checklist Before Publishing `/odom`

- Exactly one TF owner for `map -> odom -> base_link`.
- `/slam_3d/backend/odom` covariance is populated and bounded.
- Nav2 local/global costmaps are tested against the 3D SLAM pose.
- `odom` continuity is verified across loop closures.
- map-frame jumps are represented in `map -> odom`, not in `odom -> base_link`.
- Emergency stop and manual control workflows are unaffected.
