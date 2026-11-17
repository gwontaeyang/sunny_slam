# Reference Matrix

## Hardware And Driver

| Reference | What Is Used |
|---|---|
| Hesai ROS Driver 2.0 | Multi-device YAML, PointCloud2 layout, receive-vs-sensor timestamp policy, PTP and packet-loss topics |
| Hesai LiDAR SDK 2.0 | JT128/XT parsing and the optional corrected firing-time point type |

Observed Geoje messages match the official ROS wrapper layout:
`x/y/z/intensity/ring/timestamp`, `point_step=26`. The wrapper copies
`LidarPointXYZIRT.timestamp`; this is packet timestamp in the examined SDK
parsers. Corrected channel firing time requires the SDK-specific timestamp
path, not an assumption in the SLAM frontend.

## Frontend

| Reference | Adopted Idea | Current Status |
|---|---|---|
| FAST-LIO2 | Iterated error-state LIO and direct map residuals | Architecture target; not implemented |
| LIO-SAM | Explicit IMU deskew and factor-graph separation | Rotation-only packet deskew implemented |
| CT-ICP | Continuous-time scan representation for motion distortion | Design reference; current implementation remains gyro-only rotational deskew |
| DLIO | Direct LiDAR-inertial odometry with continuous-time deskew | Bias/velocity-aware translational deskew remains pending |
| KISS-ICP | Simple inspectable scan-to-map baseline | GICP baseline implemented |
| X-ICP | Degeneracy and observable-direction reasoning | Surfel Hessian rank and bounded directional pose-graph information implemented |
| LP-ICP | Correspondence-level localizability and soft/hard directional constraints | Follow-up frontend experiment; not implemented |
| Informed, Constrained, Aligned | Field comparison of TSVD, inequality, and Tikhonov degeneracy mitigation | Evidence for a later soft-constrained frontend A/B; not implemented |
| Wildcat | Continuous-time handling for rough motion | Reference for later translation/bias deskew |

## Backend

| Reference | Adopted Idea | Current Status |
|---|---|---|
| Scan Context | Ring/sector place descriptor | Small editable variant implemented |
| LIO-SAM / GTSAM | Factor graph with loop constraints | Batch Ceres SE(3) graph implemented; IMU states and incremental smoothing pending |
| Dynamic Covariance Scaling | Robust loop constraints | Per-loop DCS outer reweighting plus staged Cauchy solve implemented |
| Ceres 3D pose-graph example | Full `6x6` information whitening and fixed gauge pose | Residual whitening and gauge fixing implemented |
| Censi / Bonnabel ICP covariance analyses | Registration uncertainty and fixed-correspondence caveats | Hessian contributes bounded observability shape only; no calibrated covariance claim |
| GLIM | Submap-centric large-scale architecture | Overlapping active/completed submaps implemented; incremental smoother pending |

## Indoor Terrain And Rough Motion

| Reference | Adopted Idea | Current Status |
|---|---|---|
| Vibration-Aware LIO | Separate shock uncertainty from normal scan matching | Discrete shock/recovery map-mutation gate implemented; point covariance weighting pending |
| Robot-Centric Elevation Mapping | Gravity-aligned local height evidence with uncertainty | Gravity-aligned keyframe-local evidence implemented; uncertainty propagation pending |
| GPU Elevation Mapping | Multi-level elevation evidence and traversability products | Design reference for the deferred projection layer |
| Patchwork++ | Region-wise robust ground segmentation | Design reference; current radial profile classifier is intentionally smaller and editable |
| TRAVEL | Gravity-aligned local terrain continuity and slope reasoning | Polar local floor/ramp/step evidence implemented |
| G-VOM | Keep positive/negative obstacle evidence in 3D before projection | Persistent keyframe-reprojected 3D evidence implemented |
| Fast Staircase Detection and Estimation | Require repeated tread/riser geometry for a stair hypothesis | Conservative repeated-riser candidate label implemented; width, yaw, and staircase instance estimation are pending |
| Potential Negative Obstacle Detection by Occlusion Labeling | Missing returns need occlusion context | No-return remains unknown; ray/occlusion layer pending |
| Virtual Surfaces for Negative Obstacle Detection | Confirm risk only after conservative visibility reasoning | Planned downstream negative-obstacle validator |
| HDIF | Learn terrain difficulty from exteroceptive geometry and proprioceptive experience | Causal LiDAR-preview/IMU-traversal memory implemented; learned predictor pending |
| TerraPN | Self-supervised terrain cost from robot-terrain interaction | Traversal response is retained separately from geometry; online learning pending |
| Learning to Reconstruct Safe Terrain | Learn nominal safe geometry and flag departures without manual terrain labels | Reference for later calibrated preview anomaly scoring; current residual is deterministic |
| Navigating by Touch | Use physical interaction as delayed terrain supervision | Stable-to-recovery traversal association implemented; navigation policy learning pending |

## Primary Links

- Hesai ROS Driver 2.0: https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0
- Hesai LiDAR SDK 2.0: https://github.com/HesaiTechnology/HesaiLidar_SDK_2.0
- FAST-LIO2: https://github.com/hku-mars/FAST_LIO
- LIO-SAM: https://github.com/TixiaoShan/LIO-SAM
- CT-ICP: https://arxiv.org/abs/2109.12979
- DLIO: https://arxiv.org/abs/2203.03749
- KISS-ICP: https://github.com/PRBonn/kiss-icp
- GLIM: https://github.com/koide3/glim
- X-ICP: https://arxiv.org/abs/2211.16335
- LP-ICP: https://arxiv.org/abs/2501.02580
- Informed, Constrained, Aligned: https://arxiv.org/abs/2408.11809
- Wildcat: https://arxiv.org/abs/2205.12595
- Dynamic Covariance Scaling: https://doi.org/10.1109/ICRA.2013.6630557
- Ceres 3D pose-graph example: https://github.com/ceres-solver/ceres-solver/tree/master/examples/slam/pose_graph_3d
- An Accurate Closed-form Estimate of ICP's Covariance (Censi): https://censi.science/pub/research/2007-icra-icpcov.pdf
- On the Covariance of ICP-based Scan-matching Techniques (Bonnabel et al.): https://arxiv.org/abs/1410.7632
- Vibration-Aware LIO: https://arxiv.org/abs/2507.04311
- Robot-Centric Elevation Mapping: https://www.research-collection.ethz.ch/items/c14beb8a-7dc9-4413-90f4-ffe814108ab9
- GPU Elevation Mapping: https://arxiv.org/abs/2204.12876
- Patchwork++: https://arxiv.org/abs/2207.11919
- TRAVEL: https://arxiv.org/abs/2206.03190
- G-VOM: https://arxiv.org/abs/2109.13176
- Fast Staircase Detection and Estimation: https://arxiv.org/abs/2211.00610
- Potential Negative Obstacle Detection by Occlusion Labeling: https://www.ri.cmu.edu/publications/potential-negative-obstacle-detection-by-occlusion-labeling/
- Virtual Surfaces for Negative Obstacle Detection: https://arxiv.org/abs/2010.16018
- HDIF: https://arxiv.org/abs/2209.10788
- TerraPN: https://arxiv.org/abs/2202.12873
- Self-Supervised Traversability Prediction by Learning to Reconstruct Safe Terrain: https://arxiv.org/abs/2208.01329
- Navigating by Touch: https://doi.org/10.1007/s10514-021-10013-w
