# Roughness quarantine ownership

The SLAM launch paths must not reuse `amcl_guard` as a roughness scan filter.
`amcl_guard` belongs to the localization runtime, and coupling an experimental
SLAM policy to it can change the production AMCL scan path.

The roughness-aware `slam_toolbox` frontend now owns an opt-in LaserScan
conditioning stage under `slam_2d_core`. It supports observe, drop, throttle,
quarantine, stable-scan warp, and bounded residual-filter experiments. The
default is disabled, and the normal `slam_toolbox` LaserScan path is preserved
when disabled. Do not add these policies to `amcl_guard`, `lidar_merger`, or
shared localization and LiDAR configurations.

Shared PointCloud-to-LaserScan projection must remain a sensor-representation
boundary, not a home for SLAM motion policy; IMU gravity alignment and
roughness-based scan suppression therefore belongs in the opt-in SLAM-owned
frontend. The current frontend receives `sensor_msgs/LaserScan`; it can use IMU
roughness only to classify, quarantine, or condition a scan. It cannot recreate
the deleted z coordinate or perform an exact equivalent of rotating original
3D points before height filtering. Any future 3D gravity alignment or deskew
must be implemented at a point-cloud boundary with per-point timing evidence,
without changing shared sensor or localization nodes.

Stable-scan warp uses an exact-time transform through the configured odometry
frame and applies explicit age, translation, yaw, roll/pitch, beam-count, and
valid-ratio limits. Quarantine requires qualified stable scans to recover and
has a bounded IMU-unavailable fail-open path. Residual filtering is capped by a
maximum removed-beam ratio. All modes remain experimental until bag replay and
new field data show consistent improvement over the post-SLAM reference.

Conditioning runs after the inherited `shouldProcessScan()` motion filter.
Therefore `roughness_scan_conditioning_quarantine_recovery_scan_count` counts
pose-graph scan candidates, and
`roughness_scan_conditioning_quarantine_max_scan_gap_s` bounds the interval
between those candidates rather than the raw LaserScan period. The default
`0.75 s` bound covers the measured moving-candidate p95 of `0.525 s` in the
three representative field bags while still resetting recovery across long
stationary gaps. A `0.20 s` raw-scan-style bound must not be reused here: no
candidate pair in those bags met that interval, causing indefinite quarantine
and unacceptable pose-graph node loss.

The stable-scan and residual-filter source-age defaults are also `0.75 s` for
the same processed-candidate cadence. The IMU-unavailable fail-open clock caps
each positive update step instead of resetting on a sparse candidate, so a
persistent IMU outage eventually opens the gate. Clock rollback still resets
the timer, and invalid timestamps or low-valid-ratio LaserScans remain closed.

The custom executable explicitly stops its worker and upstream publisher
threads and releases `slam_toolbox` helpers before process exit. This breaks
their shared-node ownership cycles and ensures Karto mapper and dataset cleanup
runs while the process-wide sensor registry is still alive.

Historical analysis tools and saved evidence may still name the removed
`amcl_guard`-based experiment so older results remain reproducible. Those
references are not a supported runtime path and must not be used as a current
launch contract.
