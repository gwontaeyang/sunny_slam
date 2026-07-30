# slam_3d_livox_ingest

This package owns the hardware-specific boundary between two Livox Mid-360
`livox_ros_driver2/msg/CustomMsg` streams and the editable GOLE 3D SLAM code.
It does not merge the two sensors. Each accepted packet becomes a separate
canonical `sensor_msgs/msg/PointCloud2` containing:

- `x`, `y`, `z` as `FLOAT32`
- `intensity` as `FLOAT32`
- `offset_time` as `UINT32` nanoseconds relative to `header.stamp`
- `tag`, `line`, and `lidar_id` as `UINT8`

The output header stamp is the Livox `timebase` plus the configured per-sensor
time offset. Livox documents `timebase` as the first-point time. The current
`slam_3d_core` deskew parser therefore reconstructs each absolute point time as
`header.stamp + offset_time * 1e-9`.

`front_clock_domain` and `rear_clock_domain` default to `unverified`. Set them
to the measured hardware/host clock source in commissioning records. A matching
numeric timestamp range alone does not prove that LiDAR and IMU use the same
clock. `front_time_offset_ns` and `rear_time_offset_ns` default to zero and must
only be changed from a measured LiDAR-to-IMU offset.

## Dependency Boundary

`livox_ros_driver2` is intentionally not copied into this repository. With no
driver message package installed, the time-contract library and its tests still
build, but `mid360_ingest_node` is omitted. Install Livox ROS Driver 2 in an
underlay and rebuild this package to enable the node.

Configure the driver for customized messages (`xfer_format=1`). A two-device
deployment also needs independent topics (`multi_topic=1`) and explicit
front/rear topic mapping. The stock `msg_MID360_launch.py` currently selects
custom messages but defaults to one shared LiDAR topic, which is insufficient
for calibrated dual-sensor processing.

## Rejection Contract

Packets are rejected before reaching SLAM when any enabled check fails:

- missing or ROS-unrepresentable `timebase`;
- empty, oversized, or declared/actual point-count mismatch;
- too many non-finite points;
- first offset not near zero;
- decreasing point offsets;
- zero or excessive scan duration;
- non-increasing packet timebase on one sensor.

Diagnostics are published on `/slam_3d/ingest/health`. The latest peer start
skew is a diagnostic observation only; asynchronous front/rear packet arrival
must not be treated as proof of hardware time-sync failure.
