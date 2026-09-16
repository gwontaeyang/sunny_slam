# slam_3d_lidar_ingest

Validates and normalizes separate Hesai JT128/JT128/XT32 PointCloud2 streams.
No merging, extrinsic transform or SLAM; source `header.frame_id` is preserved.

## Build and Run

With ROS 2 Humble and dependencies installed; no hardware driver is started:
```bash
cd ~/gole_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select slam_3d_lidar_ingest
source install/setup.bash
INGEST=$(ros2 pkg prefix slam_3d_lidar_ingest)/share/slam_3d_lidar_ingest/config
ros2 run slam_3d_lidar_ingest hesai_pointcloud_ingest_node --ros-args \
  --params-file "$INGEST/hesai_jt128_xt32_ingest_params.yaml" -p use_sim_time:=true
```
Replay a clock-compatible raw bag with `--clock`; stop with Ctrl-C.
For legacy Geoje bags only, select `geoje_legacy_bag_ingest_params.yaml`: its bounded affine
clock fit is an approximation, not verified LiDAR/IMU synchronization or live calibration.

## Sensor Contract

| Sensor / frame | Raw topic | Canonical topic |
|---|---|---|
| JT128 / `hesai_front` | `/hesai/front/lidar_points` | `/slam_3d/input/hesai_front/points_timed` |
| JT128 / `hesai_rear` | `/hesai/rear/lidar_points` | `/slam_3d/input/hesai_rear/points_timed` |
| XT32 / `hesai_center` | `/hesai/center/lidar_points` | `/slam_3d/input/hesai_center/points_timed` |

Canonical fields: `x/y/z/intensity FLOAT32`, `offset_time UINT32`, `ring UINT16`, `sensor_id UINT8`.
Point time is `header.stamp + offset_time * 1e-9` seconds; packet timestamps are not channel firing times.
Malformed fields, frame/ring/count mismatches and invalid timing are rejected; inspect `/slam_3d/ingest/health`.
Production requires a measured common LiDAR/host/IMU clock; the 0.25 s receive-time gate does not prove PTP lock.
Keep sensor topics, frames, units, clock policy/offsets and calibration validity bounds explicit;
omitted generic parameters use node defaults. Do not remove calibration values just because they equal defaults.
Optional `hesai_imu_normalizer_node` maps the IMU clock and scales units to rad/s and m/s^2; it does not rotate axes.

## CPU Overlays

After the calibrated ingest profile, add `--params-file "$INGEST/compute_budget_overrides.yaml"`
(4,000 points per sensor), paired with core's `indoor_compute_budget_overrides.yaml`.
Alternatively pair `cpu_minimal_overrides.yaml` (2,000 per sensor) with `indoor_cpu_minimal_overrides.yaml`.
Default `max_output_points=0` is uncapped. Validation still checks raw points; selected points retain timing/rings.
Caps are per sensor, not per merged scan. These overlays trade map/tracking quality for CPU.
Keep the original LiDAR/IMU bag: reduced clouds, maps and archives cannot restore discarded points.
