# slam_3d_bringup

GOLE-owned 3D SLAM architecture and bringup contract for very large outdoor
mapping with two Livox Mid-360 LiDARs, the existing wheel odometry topic, and
the existing external IMU topic.

This package is not an open-source SLAM wrapper. Public SLAM projects are used
only as references for algorithm and system design. The first editable GOLE
learning implementation lives in `slam_3d_core` as a direct C++ scan-to-map
ICP/GICP frontend.

## Architecture Decision

Use a split large-scale mapping pipeline:

```text
Dual Mid-360 raw data + IMU + wheel odom
  -> time/extrinsic validation
  -> per-point deskew
  -> FAST-LIO2-style internal LIO frontend
  -> keyframes
  -> submap manager
  -> pose graph backend
  -> loop closure
  -> persistent submap map store
  -> optional localization/Nav2 bridge
```

The first production-oriented direction is:

- **Frontend**: tightly coupled LiDAR-inertial odometry with an iterated EKF,
  direct local-map registration, per-point deskew, and wheel odometry as a
  planar velocity/slip prior.
- **Dual LiDAR fusion**: two Mid-360 streams are two measurement sets against
  one robot state. Do not run two independent LIO odometries and average them
  for production.
- **Backend**: keyframe/submap pose graph with robust loop closure, submap
  storage, runtime metrics, and bounded memory.
- **Map format**: submaps plus descriptors and graph metadata, not one global
  dense PCD.
- **Downstream integration**: 3D SLAM owns `map -> odom -> base_link` only in
  explicit mapping/localization modes where the existing 2D localization/AMCL
  owners are disabled.

Detailed design:

- `docs/large_scale_outdoor_architecture.md`
- `docs/downstream_impact.md`
- `config/large_scale_outdoor_pipeline.yaml`
- `config/dual_mid360_topic_contract.yaml`
- `docs/reference_matrix.md`

## Existing GOLE Inputs

| Input | Topic | Current owner | Use in 3D SLAM |
| --- | --- | --- | --- |
| Wheel odometry | `/motor_driver_node/odom` | `gole_core/device/driver/motor_driver` | Planar velocity prior, slip detection, sanity reference |
| IMU | `/imu/data` | external/MicroStrain IMU stack | Propagation, gravity, deskew |
| Front Mid-360 | `/slam_3d/input/mid360_front/livox` | Livox driver config | Raw CustomMsg stream |
| Rear Mid-360 | `/slam_3d/input/mid360_rear/livox` | Livox driver config | Raw CustomMsg stream |

`slam_3d_livox_ingest` now validates both raw streams and publishes separate
`points_timed` clouds. These retain Livox `offset_time`, source identity,
reflectivity, tag, and line. They are the input to the current learning core.
Lossy `points_debug` streams remain visualization and legacy-dataset paths.
The future tightly coupled frontend should consume the same validated timing
contract without concatenating the sensors before deskew.

## Why Not Simple PointCloud Merge

The existing `lidar_merger` and `lidar_preprocessing` packages are useful after
registration, for perception, or for 2D localization products. They are not
safe before the 3D LIO frontend because `lidar_preprocessing` converts clouds
to `pcl::PointXYZ`, and LIO needs per-point time/source/intensity metadata for
deskew and proper measurement weighting.

For the dual Mid-360 frontend, preserve:

- source sensor id,
- per-point timestamp or time offset,
- intensity/reflectivity,
- tag/line fields when available,
- calibrated extrinsics,
- source-specific covariance/health state.

## Launch Usage

The launch file is intentionally conservative. By default it starts no SLAM
backend, no placeholder TF, no EKF, and no RViz:

```bash
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py
```

Useful development checks:

```bash
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py --show-args
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py run_static_tf:=true
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py run_wheel_imu_ekf:=true
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py run_mid360_ingest:=true
ros2 launch slam_3d_bringup dual_mid360_3d_slam.launch.py run_learning_slam_core:=true
```

Core arguments:

- `pipeline_mode`: label for the current development phase. Default
  `architecture_only`.
- `pipeline_config_file`: path to `large_scale_outdoor_pipeline.yaml`.
- `run_livox_driver`: include Livox ROS Driver 2 after the two-device config is
  ready. Default `false`.
- `run_mid360_ingest`: validate raw `CustomMsg` timing and emit canonical
  front/rear `points_timed` clouds. Default `false`; the executable is available
  only when `livox_ros_driver2` was present while building
  `slam_3d_livox_ingest`.
- `run_wheel_imu_ekf`: run a reference wheel/IMU EKF. Default `false`.
- `run_learning_slam_core`: run `slam_3d_core/scan_to_map_icp_node`.
  Default `false`.
- `learning_slam_publish_tf`: allow the learning core to publish TF. Default
  `false`; keep it off unless existing localization/AMCL TF owners are
  disabled.
- `run_static_tf`: publish placeholder base-to-LiDAR TF. Default `false`.
- `run_rviz`: open the development RViz view. Default `false`.

Placeholder extrinsics:

- Front: `front_x:=0.45 front_y:=0.0 front_z:=0.52 front_yaw:=0.0`
- Rear: `rear_x:=-0.45 rear_y:=0.0 rear_z:=0.52 rear_yaw:=3.141592653589793`

Replace these with measured CAD/calibration values before any bag validation.
The stock Livox `msg_MID360_launch.py` uses custom messages but one shared
LiDAR topic. A dual unit must use `multi_topic=1` and map each device topic to
the explicit front/rear raw topics.

## TF Ownership

Existing stacks already use the standard navigation chain:

```text
map -> odom -> base_link
```

Important owners:

- `gole_localization/odom_estimation` publishes `/odom` and `odom -> base_link`.
- `gole_apart` AMCL/localization owns `map -> odom` during 2D navigation.
- Nav2 components consume `map`, `odom`, `base_link`, and `/odom`.

Therefore 3D SLAM mapping must run in a mode where existing localization/AMCL
TF publishers are disabled, or it must publish topic-only outputs without TF.

## Validation Ladder

1. Static dual-Mid-360 + IMU bag for timestamp and extrinsic checks.
2. Single-front Mid-360 LIO bag.
3. Single-rear Mid-360 LIO bag.
4. Short dual-Mid-360 loop bag.
5. Long outdoor loop with repeated structures.
6. Open-field or low-feature route for degeneracy.
7. Rough ground/slope route for roll, pitch, and z drift.
8. One-LiDAR dropout route.
9. Wheel-slip route.
10. 30-60 minute runtime/memory route.

Required plots and metrics:

- XY and 3D trajectory.
- Top/down and side map projections.
- z drift, roll/pitch stability, and closure error.
- Frontend residuals and degeneracy score.
- Loop candidate, accepted/rejected, inlier ratio, fitness, and transform gate.
- Backend optimization timing and residual distributions.
- Submap count, memory, map publish cost, and real-time factor.

## References

- Livox Mid-360 specs: https://www.livoxtech.com/mid-360/specs
- Livox ROS Driver 2: https://github.com/Livox-SDK/livox_ros_driver2
- FAST-LIO / FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- GLIM: https://github.com/koide3/glim
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- hdl_graph_slam: https://github.com/koide3/hdl_graph_slam
- Cartographer algorithm walkthrough:
  https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html
