# slam_3d_lidar_ingest

This package owns the hardware-facing LiDAR validation and normalization
boundary for the editable 3D SLAM pipeline. The current deployment profile is
two Hesai JT128 sensors plus one Hesai XT32. It does not geometrically merge
the sensors and it does not perform scan matching. Each accepted input
`sensor_msgs/msg/PointCloud2` remains a separate canonical cloud containing:

- `x`, `y`, `z` as `FLOAT32`
- `intensity` as `FLOAT32`
- `offset_time` as `UINT32` nanoseconds relative to `header.stamp`
- `ring` as `UINT16`
- `sensor_id` as `UINT8`

The output preserves the input `header.frame_id`; this package never relabels
coordinates or applies an extrinsic. `expected_input_frame_id` is a rejection
contract, and all geometry changes remain in `slam_3d_core` behind TF lookup.
The three sensor subscriptions use independent callback groups in a
three-thread executor so one 230,400-point JT128 conversion does not block the
other two streams.

`hesai_imu_normalizer_node` adapts a Hesai built-in IMU for deskew. It validates
the source frame and calibrated clock range, maps the independent IMU clock
into the common SLAM time domain, and applies explicit unit scales before
publishing a canonical `sensor_msgs/msg/Imu`. It does not rotate measurements;
`slam_3d_core` applies the configured IMU-to-base rotation during deskew.

Geoje's XT32 `/hesai/center/lidar_imu` topic is present but empty in all three
large bags. XT32-only replay therefore uses a time-normalized JT128 built-in
IMU as an external angular-rate source. The recorded JT128 values are
quantized in degree/s and g, so the Geoje profile converts them to rad/s and
m/s^2 rather than treating the serialized values as SI.

The input contract observed in `02_samsung_geoje` uses a 26-byte Hesai layout:
`x/y/z/intensity` (`FLOAT32`), `ring` (`UINT16`), and an unaligned
`timestamp` (`FLOAT64`) at byte 18. JT128 scans contain 128 ring IDs and XT32
scans contain 32. The validator reads fields by metadata and `memcpy`; it does
not assume an aligned C struct.

The output start stamp and point offsets satisfy:

```text
point_time = output.header.stamp + offset_time * 1e-9
```

This is the contract consumed by `slam_3d_core` for IMU deskew. In the observed
Geoje bags, however, one `timestamp` value is repeated for 256 consecutive
points. It is packet time, not calibrated channel firing time. The resulting
deskew is therefore packet-level. The diagnostics expose
`timestamp_change_count` and `max_consecutive_equal_timestamps` so this quality
limit remains visible. Extrinsics are applied later in the core through TF,
before the base-frame gyro deskew for each cloud's acquisition interval.

## Clock Policies

The production profile,
`config/hesai_jt128_xt32_ingest_params.yaml`, uses each Hesai sensor header and
rejects a scan when the corrected sensor time differs from the ROS receive
clock by more than 0.25 s. Set all three devices, the host, and the IMU to one
measured clock domain (normally PTP) before enabling SLAM. Per-sensor
`time_offset_ns` values must come from calibration, not visual map tuning.

The recorded Geoje bags do not satisfy that production contract. The two JT128
clocks and the XT32 clock are in different epochs, and neither is consistently
aligned with the recorded IMU clock. For bounded offline investigation only,
`config/geoje_legacy_bag_ingest_params.yaml` uses fixed anchors and a per-sensor
clock scale fitted to `bag record time - observed scan duration`. The fit uses
the bounded offset-190 s replay window and is deterministic across callback
schedules. It preserves packet-relative timing after applying the same clock
scale, but is still only a salvage approximation: bag record time contains
transport and recorder latency. It cannot recover true LiDAR-to-IMU
synchronization from the existing data. Each calibrated profile records its
fit identifier, RMSE, maximum residual, input timestamp validity interval, and
maximum allowed output timestamp step. A scan outside that fitted interval is
rejected instead of silently extrapolating the affine model. The older
`anchored_sensor_time`
policy remains available for exploratory bags but depends on first-callback
timing. The
`ros_receive_time_scan_end` policy remains available for diagnostics, but
re-anchors every callback and is unsuitable when replay processing falls
behind.

## Topics

| Sensor | Input | Canonical output |
|---|---|---|
| front JT128 | `/hesai/front/lidar_points` | `/slam_3d/input/hesai_front/points_timed` |
| rear JT128 | `/hesai/rear/lidar_points` | `/slam_3d/input/hesai_rear/points_timed` |
| center XT32 | `/hesai/center/lidar_points` | `/slam_3d/input/hesai_center/points_timed` |

The optional IMU adapter publishes `/slam_3d/input/hesai_front/imu` from
`/hesai/front/lidar_imu`. Production may select another built-in IMU only after
its clock, units, frame, and sign have been measured.

## Rejection Contract

Packets are rejected before reaching SLAM when any enabled check fails:

- empty, oversized, malformed, big-endian, or truncated cloud;
- missing or wrongly typed Hesai fields;
- too many non-finite points;
- non-finite or decreasing point timestamps;
- zero or excessive scan duration;
- ring IDs outside the configured model range;
- frame IDs or point counts outside the configured sensor profile;
- header/first-point mismatch;
- production clock mismatch;
- calibrated input timestamp outside the fitted validity interval;
- a normalized output timestamp jump above the configured sensor limit;
- a scaled point offset that cannot fit the canonical `UINT32` nanosecond
  field;
- non-increasing normalized scan start on one sensor.

Diagnostics are published on `/slam_3d/ingest/health`. The latest peer start
skew is an observation only; asynchronous callback arrival is not proof of
hardware synchronization.

## Run

Production:

```bash
ros2 launch slam_3d_lidar_ingest hesai_jt128_xt32_ingest.launch.py
```

Legacy Geoje bag analysis:

```bash
ros2 launch slam_3d_lidar_ingest hesai_jt128_xt32_ingest.launch.py \
  use_sim_time:=true \
  params_file:=$(ros2 pkg prefix slam_3d_lidar_ingest)/share/slam_3d_lidar_ingest/config/geoje_legacy_bag_ingest_params.yaml
```
