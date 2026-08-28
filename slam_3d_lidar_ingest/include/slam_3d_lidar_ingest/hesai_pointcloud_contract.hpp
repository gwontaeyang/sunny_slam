#ifndef SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_CONTRACT_HPP_
#define SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_CONTRACT_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "sensor_msgs/msg/point_cloud2.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

enum class CloudContractCode
{
  kValid,
  kEmptyScan,
  kTooFewPoints,
  kTooManyPoints,
  kInvalidDimensions,
  kInvalidDataSize,
  kBigEndianUnsupported,
  kUnexpectedFrameId,
  kMissingRequiredField,
  kUnsupportedFieldLayout,
  kInsufficientFinitePoints,
  kNonFiniteMeasurement,
  kNonFiniteTimestamp,
  kTimestampRegression,
  kZeroScanDuration,
  kScanDurationTooLong,
  kTimestampOffsetOverflow,
  kRingOutOfRange,
  kHeaderTimestampMismatch,
  kMissingHeaderTimestamp,
  kClockMisaligned,
  kClockCalibrationOutOfRange,
  kNonIncreasingOutputStamp,
  kOutputTimestampJump,
  kOutputTimeOutsideRosRange,
};

enum class ClockPolicy
{
  kSensorHeader,
  kRosReceiveTimeScanEnd,
  kAnchoredSensorTime,
  kCalibratedSensorTime,
};

struct HesaiFieldLayout
{
  std::uint32_t x_offset{0};
  std::uint32_t y_offset{0};
  std::uint32_t z_offset{0};
  std::uint32_t intensity_offset{0};
  std::uint32_t ring_offset{0};
  std::uint32_t timestamp_offset{0};
};

struct HesaiCloudContractOptions
{
  std::size_t min_points{1};
  std::size_t max_points{500000};
  std::uint64_t max_scan_duration_ns{200000000};
  std::uint16_t max_ring_id{127};
  double min_finite_point_ratio{0.9};
  double max_header_first_point_delta_s{0.005};
  double timestamp_regression_tolerance_s{1.0e-9};
  bool require_monotonic_timestamps{true};
  bool require_nonzero_duration{true};
  bool require_header_matches_first_point{true};
  bool require_expected_frame{false};
  std::string expected_frame_id;
};

struct HesaiCloudContractResult
{
  CloudContractCode code{CloudContractCode::kValid};
  HesaiFieldLayout layout;
  std::string failed_field;
  std::size_t point_count{0};
  std::size_t finite_point_count{0};
  std::size_t invalid_point_count{0};
  std::size_t nonfinite_timestamp_count{0};
  std::size_t timestamp_regression_count{0};
  std::size_t timestamp_change_count{0};
  std::size_t max_consecutive_equal_timestamps{0};
  std::uint16_t observed_max_ring{0};
  double finite_point_ratio{0.0};
  double header_time_s{0.0};
  double first_point_time_s{0.0};
  double min_point_time_s{0.0};
  double max_point_time_s{0.0};
  double header_first_point_delta_s{0.0};
  std::uint64_t scan_duration_ns{0};

  bool accepted() const
  {
    return code == CloudContractCode::kValid;
  }
};

struct OutputStampResult
{
  CloudContractCode code{CloudContractCode::kValid};
  std::int64_t output_start_ns{0};

  bool accepted() const
  {
    return code == CloudContractCode::kValid;
  }
};

struct PointOffsetResult
{
  CloudContractCode code{CloudContractCode::kValid};
  std::uint32_t output_offset_ns{0};

  bool accepted() const
  {
    return code == CloudContractCode::kValid;
  }
};

const char * cloudContractCodeString(CloudContractCode code);
const char * clockPolicyString(ClockPolicy policy);
std::optional<ClockPolicy> parseClockPolicy(const std::string & value);

HesaiCloudContractResult validateHesaiPointCloud(
  const sensor_msgs::msg::PointCloud2 & message,
  const HesaiCloudContractOptions & options);

OutputStampResult selectOutputStartStamp(
  ClockPolicy policy,
  std::int64_t input_header_ns,
  std::int64_t receive_time_ns,
  std::uint64_t scan_duration_ns,
  std::int64_t configured_offset_ns,
  std::int64_t anchor_input_header_ns = 0,
  std::int64_t anchor_output_start_ns = 0,
  double time_scale = 1.0);

PointOffsetResult selectOutputPointOffset(
  double point_offset_s,
  ClockPolicy policy,
  double time_scale = 1.0);

}  // namespace slam_3d_lidar_ingest
}  // namespace gole

#endif  // SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_CONTRACT_HPP_
