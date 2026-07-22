# Large-Scale Outdoor 3D SLAM Reference Matrix

This is a design reference, not an import plan. The GOLE stack should own the
pipeline boundaries, topics, map format, health metrics, and downstream TF
contract even when individual algorithms are prototyped from public work.

## Method Comparison

| Reference | Useful idea | Why it is not enough by itself | GOLE design takeaway |
| --- | --- | --- | --- |
| FAST-LIO2 / FAST-LIO family | Real-time tightly-coupled LiDAR-inertial odometry with direct point-to-map updates and Livox-friendly assumptions. | It is primarily an odometry/frontend architecture. Large outdoor operation still needs map lifecycle, loop closure, persistence, health metrics, and navigation integration. | Use an iterated-EKF/direct local-map frontend pattern for the first internal LIO implementation. |
| LIO-SAM | Factor graph separation between IMU preintegration, LiDAR odometry, GPS, and loop factors. | Its original assumptions around point cloud organization and sensor model are not a direct Mid-360 raw/custom-message fit. | Use its factor separation and keyframe graph layout as backend design reference. |
| FAST-LIO-SAM variants | Practical combination of FAST-LIO odometry and pose graph optimization. | Public variants differ in quality and are usually glue code around specific forks. | Copy the architectural split, not the dependency graph. |
| GLIM | Modern range-inertial mapping structure with frontend/backend separation, submaps, and global registration concepts. | It is a large stack with substantial dependencies; using it directly would hide GOLE-specific contracts. | Use it as a reference for submap/global registration and fixed-lag/global optimization boundaries. |
| KISS-ICP / KISS-SLAM | Simple LiDAR-only odometry and map-closure baselines with ROS 2 usability. | No IMU/wheel coupling, so it is not the final answer for Mid-360 outdoor mapping. | Keep it as an offline geometry smoke-test baseline for bags. |
| hdl_graph_slam | Classic 3D graph SLAM constraints, including IMU/GPS/floor-style priors. | ROS 1 and spinning-LiDAR era assumptions; not a Livox LIO frontend. | Good source for backend factor types and diagnostic visualizations. |
| Cartographer | Submap lifecycle, scan-to-submap matching, background constraint building, and map serialization concepts. | It is not a good direct Mid-360 LIO frontend for this project. | Use its submap and background optimization ideas for map storage and loop work scheduling. |

## Architecture Decision

For very large outdoor areas, build a split system:

1. **Sensor ingest and calibration layer**
   - Preserve Livox CustomMsg point timing and source information.
   - Track front/rear LiDAR, LiDAR-IMU, and base extrinsics separately.
   - Measure time offsets and packet drops before trying to tune SLAM.

2. **Frontend LIO**
   - Start with FAST-LIO2-style iterated EKF/direct local-map registration.
   - Fuse two Mid-360 sensors as two measurement sets against one robot state.
   - Do not average two independent LIO odometry streams as the production path.
   - Use wheel odometry as a planar velocity prior and slip diagnostic, not as a hard pose truth.

3. **Submap manager**
   - Convert stable frontend keyframes into overlapping local submaps.
   - Store submap clouds, descriptors, poses, and bounding boxes independently.
   - Publish the active/local map at high rate and global map at low rate.

4. **Backend pose graph**
   - Add between-keyframe/submap factors from frontend odometry.
   - Add loop closure factors only after descriptor and geometric verification gates.
   - Support robust kernels and switchable constraints for bad loops.
   - Keep optimization metrics: iterations, timing, residuals, accepted/rejected loops.

5. **Map persistence and localization**
   - Store maps as submap tiles plus graph metadata, not one monolithic PCD.
   - Load only relevant submaps for localization in large sites.
   - Define a controlled bridge from 3D SLAM pose to the existing 2D Nav2 `/odom`
     and `map -> odom -> base_link` contract after navigation review.

## Outdoor-Specific Risks

- Repetitive structure, open fields, vegetation, and long corridors cause LiDAR
  degeneracy; the frontend must detect weak directions and inflate covariance.
- Dual Mid-360 non-overlap or low-overlap mounting makes extrinsic validation
  harder than ordinary overlapping multi-LiDAR calibration.
- Large maps fail by memory and retrieval strategy before they fail by point
  cloud visualization; submap indexing is required from the first design.
- Dynamic objects should be handled as registration down-weighting/outlier
  rejection, not by deleting raw data before the LIO frontend.
- Existing GOLE 2D localization and AMCL can publish the same TF chain expected
  by Nav2; 3D SLAM must be launched in a mode with explicit TF ownership.

## Validation Datasets

Minimum local dataset ladder:

1. Static dual-Mid-360 plus IMU bag for timestamp, extrinsic, and noise checks.
2. Single-front Mid-360 loop.
3. Single-rear Mid-360 loop.
4. Short dual-Mid-360 loop.
5. Long outdoor loop with repeated structures.
6. Open-field or low-feature route for degeneracy.
7. Rough ground and slope route for roll/pitch and z drift.
8. One-LiDAR dropout route.
9. Wheel-slip route.
10. 30-60 minute memory/runtime route.

Metrics:

- Drift per 100 m and closure error.
- z drift and roll/pitch stability.
- Front/rear LiDAR rate, IMU rate, timestamp skew, packet drops.
- Frontend registration residuals and degeneracy score.
- Backend optimization time, residual distribution, loop acceptance/rejection.
- Map point count, submap count, memory use, and global map publish cost.

## References

- Livox Mid-360 specifications: https://www.livoxtech.com/mid-360/specs
- Livox ROS Driver 2: https://github.com/Livox-SDK/livox_ros_driver2
- FAST-LIO / FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- GLIM: https://github.com/koide3/glim
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- KISS-SLAM: https://github.com/PRBonn/kiss-slam
- hdl_graph_slam: https://github.com/koide3/hdl_graph_slam
- Cartographer algorithm walkthrough:
  https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html
