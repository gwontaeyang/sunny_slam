# slam_3d_core

Editable C++ 3D SLAM: LiDAR/IMU deskew, selectable point-cloud registration,
keyframes, submaps, loop closure, and SE(3) pose graphs. This is not a full LIO estimator.

## Build and Run

With ROS 2 Humble and dependencies installed; this starts only the core, not hardware:
```bash
cd ~/gole_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select slam_3d_core
source install/setup.bash
CORE=$(ros2 pkg prefix slam_3d_core)/share/slam_3d_core/config
ros2 run slam_3d_core scan_to_map_icp_node --ros-args \
  --params-file "$CORE/indoor_xt32_single_params.yaml" \
  -p use_sim_time:=true -p publish_tf:=false -p pose_graph_solver:=ceres_se3
```
Supply canonical clouds, IMU and TF via bag `--clock`; stop with Ctrl-C. See [bringup](../slam_3d_bringup/README.md).

## Sensor Contract

- `indoor_xt32_single_params.yaml`: center XT32; `indoor_dual_jt128_params.yaml`: front/rear JT128.
  Both use `/imu/data` with no wheel input. The generic base profile is not a no-wheel indoor profile.
- Canonical points use `header.stamp + offset_time * 1e-9`; see [ingest](../slam_3d_lidar_ingest/README.md).
  Calibrate LiDAR/IMU clocks and `base_link -> hesai_*` TF; missing TF/deskew coverage rejects clouds.
- Indoor IMU frame is `cv7_link`, native XYZ with identity IMU-to-base rotation; verify mounting,
  units, clock offset and RELIABLE publisher QoS. Do not copy these values to another IMU.
- `publish_tf=false` avoids TF ownership conflicts. When enabled, only direct `map -> base_link`
  is provided, not `map -> odom -> base_link`. Restart the core after a ROS clock rewind.
- Outputs: `/slam_3d/frontend/odom`, `/slam_3d/frontend/path`, `/slam_3d/map/global_cloud`, `/slam_3d/health`.

## Parameters and CPU

`parameter_profile` selects startup-only, read-only defaults; omitted selector = `default` (unchanged bare-node behavior).
Explicit ROS YAML/CLI/`NodeOptions` overrides take precedence over presets, then [API defaults](src/scan_to_map_icp.cpp).
Edit shared tuning in [src/parameter_profile.cpp](src/parameter_profile.cpp); keep sensor calibration explicit in YAML.
`config/scan_to_map_icp_params.yaml` selects `hesai_triple`; the three indoor YAML files select their matching indoor presets.
Select at startup: `-p registration_method:=icp`, `gicp`, `fast_gicp`, `ndt`, or `voxel_surfel`.
The matcher applies to tracking and loop geometry; `pose_graph_solver` is independent.
FastGICP needs the separately installed [CPU library](https://github.com/koide3/fast_gicp).
CMake `SLAM_3D_WITH_FAST_GICP=AUTO` detects it; `ON` requires it; `OFF` excludes it.
Unknown/unavailable methods fail startup without fallback. `/slam_3d/health` lists compiled methods.
Optional overrides: `fast_gicp_num_threads=1`, `ndt_resolution_m=1.0`, `ndt_step_size=0.1`,
`ndt_transformation_epsilon=0.01`. Existing `max_icp_iterations` caps all PCL matchers.
NDT needs populated voxel distributions; ICP trimming/distance-refinement do not apply to NDT.
Coarse CPU overlays may need a larger `ndt_resolution_m`; validate on the intended bag.

| `parameter_profile` | `registration_method` | `pose_graph_solver` | `mapping_spatial_mode` |
|---|---|---|---|
| `default`, `hesai_triple` | `gicp` | `legacy_relaxation` | `planar` |
| `indoor_xt32` | `voxel_surfel` | `ceres_se3` | `se3` |
| `indoor_xt32_gicp`, `indoor_dual_jt128` | `gicp` | `ceres_se3` | `se3` |

Load the calibrated profile first, then an optional CPU overlay with another `--params-file`:

| Core overlay (`$CORE/`) | Paired ingest overlay (`config/`) | Points per sensor |
|---|---|---:|
| `indoor_compute_budget_overrides.yaml` | `compute_budget_overrides.yaml` | 4,000 |
| `indoor_cpu_minimal_overrides.yaml` | `cpu_minimal_overrides.yaml` | 2,000 |

Presets reduce geometry/tracking quality, disable online terrain evidence/preview, and leave solver/calibration unchanged.
Retain original LiDAR/IMU bags: sparse maps/archives cannot recover discarded points. Cleanup remains offline.

## Pose Graph Backend

Select at startup with `-p pose_graph_solver:=ceres_se3`, `g2o_se3`, or `gtsam_se3`.
All three use their library's native batch Levenberg-Marquardt (LM), not an incremental solver:
- `ceres_se3`: Ceres Solver; required dependency and indoor default.
- `g2o_se3`: optional g2o, `OptimizationAlgorithmLevenberg`.
- `gtsam_se3`: optional GTSAM, `LevenbergMarquardtOptimizer`, **not iSAM2**.

The generic node/base default is `legacy_relaxation`. Unknown/unavailable solvers fail startup,
even with graph correction disabled; no fallback. Nonlinear solvers require `pose_graph_planar_correction_enabled=false`.
CMake `SLAM_3D_WITH_G2O` / `SLAM_3D_WITH_GTSAM`: `AUTO` detects dependencies, `ON` requires them, `OFF` excludes them.
Append `--cmake-args -DSLAM_3D_WITH_G2O=ON -DSLAM_3D_WITH_GTSAM=ON` to require both optional adapters.
Install optional dependencies separately; baseline rosdep does not require them.
`config/pose_graph_*_solver_overrides.yaml` remain compatibility alternatives to `-p`.
Historical `pose_graph_ceres_*` schedule keys are shared; threads/nonmonotonic steps are Ceres-only.
Stopping criteria/statistics differ by backend; inspect compiled solvers and results in `/slam_3d/health`.

## Offline Map Cleanup Migration

`global_cloud` is temporally unfiltered geometry, not a confirmed-static map.
At node startup, set `keyframe_archive_directory` via YAML or `-p` to a new absolute path.
Default `''` disables export; changing this parameter after startup does not enable archival.
After final graph correction, export while the core is running; the destination must not exist:
```bash
ros2 service call /scan_to_map_icp_node/export_keyframes std_srvs/srv/Trigger '{}'
```
Adjust the service path for a renamed node. Check `success`, then run the separately distributed
`slam_3d_map_cleanup` package; it is not included in this SLAM branch.
Archives contain retained, filtered keyframes only; pruning loses earlier evidence. Use `map_static.pcd`
and a matching corrected trajectory for projection. `temporal_static_map_enabled=true` is rejected.
