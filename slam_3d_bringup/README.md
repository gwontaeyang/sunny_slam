# slam_3d_bringup

Connects Hesai ingest and the 3D SLAM core for indoor XT32 or dual-JT128 profiles
and a generic three-LiDAR profile. All `run_*` gates and `learning_slam_publish_tf`
default to false: no hardware, ingest, IMU normalizer, core, EKF, static TF or RViz starts.

## Build and Inspect

With ROS 2 Humble and package dependencies installed:
```bash
cd ~/gole_ws
source /opt/ros/humble/setup.bash
colcon build --packages-up-to slam_3d_lidar_ingest slam_3d_core slam_3d_bringup
source install/setup.bash
ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py --show-args
```
Ceres is required; g2o/GTSAM are optional. See [core build/solver options](../slam_3d_core/README.md#pose-graph-backend).

## Indoor Replay

For a bag already containing canonical timed clouds, `/imu/data` and calibrated sensor TF:
Clouds must be RELIABLE; for BEST_EFFORT use `learning_slam_cloud_qos_reliable:=false`.
```bash
ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py \
  use_sim_time:=true pipeline_mode:=bag_replay \
  run_learning_slam_core:=true learning_slam_sensor_profile:=indoor_xt32_single
```
In another sourced terminal, run `ros2 bag play <bag_directory> --clock`; stop both with Ctrl-C.
Driver and static TF stay off. Do not publish duplicate recorded TF or normalized cloud streams.
For raw-only bags, also enable `run_lidar_ingest:=true` and provide a calibrated
`lidar_ingest_params_file:=/absolute/path/to/ingest.yaml`; see [ingest](../slam_3d_lidar_ingest/README.md).

| `learning_slam_sensor_profile` | LiDAR input | IMU / wheel |
|---|---|---|
| `indoor_xt32_single` | center XT32 | `/imu/data` / disabled |
| `indoor_dual_jt128` | front + rear JT128 | `/imu/data` / disabled |
| `custom` (default) | selected core YAML and launch overrides | profile-dependent |

Indoor profiles force wheel prediction off, select IMU rotation for inter-LiDAR compensation,
and ignore `run_wheel_imu_ekf:=true`. XT32-only has inter-LiDAR compensation disabled.
`learning_slam_params_file` overrides the selected YAML, not the indoor launch's no-wheel contract.
`pipeline_mode` is descriptive, not a safety switch; `run_*` gates control startup.
`pipeline_config_file=auto` reports `indoor_pipeline.yaml` for indoor profiles, otherwise `large_scale_outdoor_pipeline.yaml`.

## Parameters and CPU

Core YAML selects startup `parameter_profile` defaults; explicit ROS overrides take precedence.
Keep calibration in YAML; shared tuning is in [parameter_profile.cpp](../slam_3d_core/src/parameter_profile.cpp).
Set `learning_slam_registration_method:=icp`, `gicp`, `fast_gicp`, `ndt`, or `voxel_surfel`;
empty preserves the selected YAML. FastGICP is optional; unavailable/unknown methods fail startup.
See [core parameters](../slam_3d_core/README.md#parameters-and-cpu) for presets and dependency options.
For solver selection or multiple CPU overlays, use the [core run command](../slam_3d_core/README.md#build-and-run)
with `-p pose_graph_solver:=ceres_se3`, `g2o_se3`, or `gtsam_se3`; bringup has no solver/overlay-list argument.
Pair core/ingest CPU overlays after calibrated profiles. They reduce map/tracking quality;
retain original LiDAR/IMU bags because sparse maps/archives cannot recover discarded points.

## Deployment Contract

- Validate correction/firetime files, device IPs and distinct UDP ports; the driver template is not commissioned.
- Calibrate `base_link -> hesai_*` TF and IMU axes/units/time offset in one measured clock domain.
  Indoor profiles require `cv7_link`, identity IMU-to-base rotation and a RELIABLE `/imu/data` publisher.
- Legacy Geoje clock-fit profiles are bounded replay approximations, not synchronized LIO or live calibration.
- Keep exactly one map/base TF owner. The core's opt-in TF is direct `map -> base_link`, not an odom split.
  Restart the core after a clock rewind; missing TF/IMU coverage is not repaired by a parameter preset.
- `/slam_3d/map/global_cloud` is temporally unfiltered. Enable keyframe archival before mapping,
  export after graph correction while the core runs, then use [offline cleanup](../slam_3d_core/README.md#offline-map-cleanup-migration).
  Cleanup is not launched here; project `map_static.pcd` with a trajectory from the same corrected state.
