# slam_3d_bringup

This package wires the current 3D SLAM development stack for two Hesai JT128
sensors and one Hesai XT32. The default launch is inert: it does not start the
hardware driver, publish static TF, run SLAM, or take ownership of navigation
TF until each corresponding argument is enabled.

## Verified Geoje Input

`02_samsung_geoje` contains:

| Sensor | Raw topic | Frame | Points | Rings |
|---|---|---|---:|---:|
| front JT128 | `/hesai/front/lidar_points` | `hesai_front` | 230,400 | 128 |
| rear JT128 | `/hesai/rear/lidar_points` | `hesai_rear` | 230,400 | 128 |
| center XT32 | `/hesai/center/lidar_points` | `hesai_center` | 128,000 | 32 |

All three use `sensor_msgs/msg/PointCloud2` with a 26-byte input layout:
`x/y/z/intensity FLOAT32`, `ring UINT16`, and `timestamp FLOAT64` at byte 18.
One timestamp is repeated for 256 consecutive points. It is packet time, not
calibrated channel firing time.

The bags do not have one verified LiDAR/IMU clock domain. JT128, XT32, and IMU
timestamps use incompatible epochs or rates depending on the recording.
Constant offsets cannot repair this. Existing bag replay is therefore an
approximate geometry experiment, not evidence of tightly synchronized LIO.

## Package Roles

`slam_3d_lidar_ingest`:

- validates frame, fields, model point count, ring range, timestamps, and scan
  duration;
- converts each sensor separately to
  `header.stamp + offset_time:uint32 nanoseconds`;
- publishes packet-time quality diagnostics;
- does not transform, merge, register, or integrate IMU.

`slam_3d_core`:

- supports an XT32-only indoor profile and a dual-JT128 indoor profile in
  addition to the legacy three-LiDAR development topology;
- uses XT32 as the primary trigger in the legacy three-LiDAR topology and
  matches front/rear JT128 scans from bounded queues, consuming each once;
- transforms each sensor to `base_link`, packet-level deskews it, and uses
  timestamp-interpolated wheel odom to move auxiliary clouds to the XT32 time;
- rejects a sensor cloud when TF is missing in the production profile;
- runs editable ICP, GICP, or voxel-surfel scan-to-map registration plus
  keyframes, shock quarantine, place recognition, and either the legacy
  edge-wise pose graph or the indoor staged Ceres SE(3) backend;
- uses one cloud-bounded causal IMU snapshot for deskew, roughness, and optional
  gyro prediction. Watermark waiting covers all enabled IMU consumers and the
  latest selected auxiliary point-time end. The validated indoor bag profiles
  request RELIABLE IMU QoS.

## Main Files

- `launch/hesai_jt128_xt32_3d_slam.launch.py`
- `config/hesai_jt128_xt32_topic_contract.yaml`
- `config/indoor_pipeline.yaml`
- `config/large_scale_outdoor_pipeline.yaml`
- `config/hesai_jt128_xt32_driver.template.yaml`
- `config/wheel_imu_ekf_3d_params.yaml`
- `docs/indoor_outdoor_profile_contract.md`
- `rviz/hesai_jt128_xt32_3d_slam.rviz`

The driver template is intentionally not launchable as-is. Device IPs, host
interface, correction files, firetime files, and PTP behavior must be filled
from the commissioned installation. The devices must also be configured to
send to distinct host UDP ports; the template assigns front/rear/center to
2368/2369/2370 because the Hesai SDK creates a receive socket per LiDAR.

## Inspect Launch

```bash
ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py --show-args
```

Safe defaults:

- `run_hesai_driver:=false`
- `run_lidar_ingest:=false`
- `run_static_tf:=false`
- `run_wheel_imu_ekf:=false`
- `run_learning_slam_core:=false`
- `learning_slam_publish_tf:=false`
- `run_rviz:=false`

The Hesai bringup's `imu_topic` defaults to the normalized front JT128 stream,
`/slam_3d/input/hesai_front/imu`. The core profile should also enable
`imu_require_expected_frame` for the commissioned IMU frame before deskew is
used live.

For indoor development, select one of the explicit sensor profiles. Both use
`/imu/data`, disable wheel prediction, enable SE(3) keyframes and overlapping
submaps, and publish temporally unfiltered geometry on `/slam_3d/map/global_cloud`.
Temporal-static cleanup is a separate offline `slam_3d_map_cleanup` step using
[`config/indoor.yaml`](../slam_3d_map_cleanup/config/indoor.yaml), not a core or
bringup process. `temporal_static_map_enabled=true` is rejected with migration
guidance; `false` is a legacy guard only. Remove the other old temporal parameters.

To retain cleanup inputs, set `keyframe_archive_directory` to a new absolute
archive path in the selected core parameter file before mapping. Its default
`''` disables archive scan retention/export; `keyframe_archive_voxel_leaf_m`
defaults to `0.22`. After graph correction, call the private Trigger service
while the core is still running:

```bash
ros2 service call /scan_to_map_icp_node/export_keyframes std_srvs/srv/Trigger '{}'
```

The launch default node is `/scan_to_map_icp_node`; adjust the service path when
renaming or namespacing it. Check the response before running `temporal_map_cleanup`
offline. The archive contains immutable mapping-only sensor views and one snapshot
of corrected poses. Its manifest records `snapshot_scope: retained_keyframes`,
`pruned_keyframes`, and stable monotonic keyframe IDs; it is not a full-session
archive after pruning. See the [core migration workflow](../slam_3d_core/README.md#offline-map-cleanup-migration).
Feed the resulting `map_static.pcd`, not a core `global_cloud` capture, into the
cleaned-map projection workflow with a trajectory from the same corrected state.

Semantic person detection and object tracking remain outside the pipeline. The current
profiles require `cv7_link` and use its native XYZ convention with identity
IMU-to-base rotation. The older `04_gangnam` roll-180 workaround remains in
archived replay configs only. Physical mounting, time offset, Xsens, and other
IMU frames require separate calibration and profiles:

```bash
# One XT32
ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py \
  run_learning_slam_core:=true \
  learning_slam_sensor_profile:=indoor_xt32_single

# Two JT128 sensors
ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py \
  run_learning_slam_core:=true \
  learning_slam_sensor_profile:=indoor_dual_jt128
```

These commands still do not start the Hesai hardware driver or publish static
TF unless those launch arguments are enabled explicitly.

The launch argument `pipeline_config_file` defaults to `auto`. It reports
`indoor_pipeline.yaml` for either indoor sensor profile and keeps
`large_scale_outdoor_pipeline.yaml` for `custom`. The indoor profiles also
replace the forwarded wheel topic with `/unused/wheel_odom_disabled`, force
wheel prediction off, and force inter-LiDAR compensation to use IMU rotation.
They also ignore `run_wheel_imu_ekf:=true`, so an indoor profile cannot create
a wheel subscription through either estimator path. Indoor and outdoor tuning
remains in separate profile files while registration, keyframes, submaps,
place recognition, and pose-graph code stay in the shared `slam_3d_core`
package.
The XT32 indoor default uses voxel-surfel registration with a bounded
LiDAR-only constant-velocity initial guess. For a slower quality-reference A/B
run, override `learning_slam_params_file` with
`slam_3d_core/config/indoor_xt32_single_gicp_quality_params.yaml`; neither file
subscribes to wheel odometry.

Both indoor profiles run the Ceres SE(3) graph with the full-bag validated
`pose_graph_information_mode=configured_diagonal` default. The opt-in
`registration_observability` mode lets accepted keyframe and loop registrations
contribute a bounded full `6x6` information shape from the fixed-pose
point-to-plane Hessian. The configured diagonal weights remain its upper bound
and fallback for invalid geometry or a changed loop linearization point. This
mode is an observability heuristic, not a calibrated ICP covariance estimate,
and must pass a representative replay regression before profile activation.

Both indoor profiles enable an IMU shock state machine and persistent 3D
terrain evidence. A detected impact suppresses the constant-velocity seed and
prevents initialization, keyframe, submap, loop, and map mutation until the
configured stable-scan recovery count is reached; registration output remains
available as high-covariance odometry for diagnostics, while active-shock poses
are omitted from `/slam_3d/frontend/path`. `/slam_3d/terrain/evidence` uses PointXYZI intensity
labels `1=floor`, `2=ramp`, `3=positive step`, `4=observed drop`, and
`5=staircase`. Staircase evidence requires at least three height/depth-consistent
risers in one gravity-aligned radial profile and the same independent-viewpoint
confirmation as every persistent label. A missing return is not a drop, and
this topic is intentionally not projected to a 2D navigation grid in the
current indoor SLAM stage.

Ramp candidates also pass a robust gravity-relative local-plane gate for
surface length, lateral width, residual RMSE, and inlier ratio. The matching
PointXYZRGB topic `/slam_3d/terrain/evidence_colored` uses green for floor,
blue for ramp, orange for positive step, red for directly observed lower
support, and purple for staircase. The supplied RViz profile enables this
semantic cloud by default.

The RViz profile also enables `/slam_3d/terrain/support_evidence`, which preserves
gray-zone evidence as a five-state support surface: unresolved, probable support,
confirmed support, probable hazard, and confirmed hazard. Dark/light green and
blue/cyan distinguish confirmed/probable floor and ramp; yellow marks probable
hazard and gray remains unresolved. This does not weaken the existing confirmed
semantic topic. Probable support requires direct proximity to confirmed support,
cannot propagate through other probable points, and is exported as a nonzero
continuous traversal cost rather than binary free space.

The same profiles publish observer-only preview memory on
`/slam_3d/terrain/preview_evidence`. Intensity encodes
`10 * terrain_label + state`, with state `1=tentative`,
`2=geometry_confirmed`, `3=traversal_confirmed`, and
`4=unsafe_traversal_conflict`. LiDAR can create the candidate before the robot
reaches it; a preview lead-time diagnostic requires that its first observation
was at least 0.75 m ahead and within 1.00 m laterally of the heading axis.
`/imu/data` then records stable response at the accepted pose or localizes a
translational shock by timestamp interpolation between bounded stable poses.
Yaw-only motion cannot create a terrain shock, and an event without valid
stable bounds is discarded. Quarantine and rejected registrations add a
persistent traversal continuity break so replay cannot connect their stable
endpoints with a fabricated straight passage. This topic is rebuilt after
pose-graph correction, but it cannot alter registration, keyframes, loops,
poses, or map output in the current stage.
A stronger pulse updates both the response score and physical event time. Any
separately rejected or pose-quarantined scan invalidates that pending bridge.
Primary queue overflow marks the first surviving cloud with a persistent break,
but only when a pending forward cloud was omitted; a stale arrival that removes
itself does not fabricate a gap. Sparse and deskew-failed clouds still advance
the primary timestamp watermark, and a late timestamp is rejected before any
IMU or terrain state update. Point-time bounds remain active with deskew off,
so post-header IMU impacts are attributed to the cloud that actually observed
them. Gravity and roughness use the same dispatch-time sensor snapshot.
Point-time bounds outside the positive `deskew_max_scan_duration_s`, or with
either endpoint farther than that duration from the cloud header, are excluded
from IMU/shock watermark advancement. Gravity-only operation waits to the cloud
stamp independently of the deskew wait switch. Stale queue victims are
identified by the last dispatched primary stamp even when they entered before
the queue became full.
`imu_subscription_reliable=true` is valid only when the live publisher offers
RELIABLE QoS; verify the deployed `/imu/data` endpoint before bringup.

## Geoje Bag Replay

Use the legacy ingest and core profiles only for the existing recordings:

```bash
source /opt/ros/humble/setup.bash
source ~/gole_ws/install/setup.bash

ros2 launch slam_3d_bringup hesai_jt128_xt32_3d_slam.launch.py \
  use_sim_time:=true \
  pipeline_mode:=bag_replay \
  run_lidar_ingest:=true \
  lidar_ingest_params_file:=$(ros2 pkg prefix slam_3d_lidar_ingest)/share/slam_3d_lidar_ingest/config/geoje_legacy_bag_ingest_params.yaml \
  run_learning_slam_core:=true \
  learning_slam_params_file:=$(ros2 pkg prefix slam_3d_core)/share/slam_3d_core/config/geoje_legacy_scan_to_map_params.yaml
```

In a separate shell, play a bag directory with `/clock`:

```bash
ros2 bag play <geoje_bag_directory> --clock
```

Keep `run_hesai_driver:=false` and `run_static_tf:=false` for bag replay. The
bag already supplies raw clouds and recorded `/tf_static`.

The bounded derived Geoje bag under `~/analysis_outputs/` omits `/tf_static`.
Its reproducible replay script therefore sets `run_static_tf:=true` and uses the
recorded transforms copied into launch arguments. Those transforms are replay
evidence, not a verified live calibration.

## Current Geoje Evidence

The current 24 s candidate is `v24`, using fixed affine legacy replay time,
per-auxiliary timestamp watermarks, packet-level rotation deskew,
timestamp-interpolated wheel prediction, and inter-LiDAR wheel-time motion
compensation. It accepted all 72 XT32 scans with no ingest rejection or
primary queue overflow and produced 114,322 map points. A repeat run matched
all 72 poses to `7.028e-10 m` translation RMSE. The no-inter-LiDAR-motion
`v14` result remains the control because wheel odometry is not ground truth.

Results and PCD-only bag directories:

```text
~/analysis_outputs/slam_3d/03_hesai_geoje_ingest/comparisons/
~/analysis_outputs/slam_3d/03_hesai_geoje_ingest/pcd_full_maps/
```

## Live Production Gate

Before enabling the Hesai driver and production ingest:

1. Verify correction and firetime files for each serial number.
2. Verify the three LiDAR topics and frame IDs independently.
3. Verify PTP lock, offset, drift, and reset diagnostics for all LiDARs and the
   host.
4. Verify IMU timestamps are in the same measured clock domain.
5. Calibrate `base_link -> hesai_*` and `base_link -> IMU` extrinsics.
6. Record a static bag and confirm zero-motion map thickness and timestamp
   residuals.
7. Record a low-speed loop before rough construction-site motion.

The current production clock check only compares header and ROS receive time
against a threshold. It does not prove PTP lock. The official Hesai wrapper
must be extended or its PTP topics monitored before claiming synchronized LIO.

## TF Ownership

All outputs remain under `/slam_3d/...`. `publish_tf` is false by default.
The current learning core can publish only a direct `map -> base_link`
prototype transform; it does not yet provide the production
`map -> odom -> base_link` split. During mapping, exactly one stack may own the
map/base TF path. Existing localization and AMCL TF owners must be disabled
before either prototype or future backend TF publication is enabled.

## References

- Hesai ROS Driver 2.0: https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0
- Hesai LiDAR SDK 2.0: https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0
- FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- HDIF: https://arxiv.org/abs/2209.10788
- TerraPN: https://arxiv.org/abs/2202.12873
- Self-Supervised Traversability Prediction by Learning to Reconstruct Safe Terrain: https://arxiv.org/abs/2208.01329
