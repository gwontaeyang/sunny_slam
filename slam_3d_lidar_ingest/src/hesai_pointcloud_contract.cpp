#include "slam_3d_lidar_ingest/hesai_pointcloud_contract.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "sensor_msgs/msg/point_field.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

namespace
{

template<typename T>
T readValue(const std::uint8_t * source, const std::uint32_t offset)
{
  T value{};
  std::memcpy(&value, source + offset, sizeof(T));
  return value;
}

const sensor_msgs::msg::PointField * findField(
  const sensor_msgs::msg::PointCloud2 & message,
  const std::string & name)
{
  const auto match = std::find_if(
    message.fields.begin(),
    message.fields.end(),
    [&name](const sensor_msgs::msg::PointField & field) {
      return field.name == name;
    });
  return match == message.fields.end() ? nullptr : &(*match);
}

bool fieldMatches(
  const sensor_msgs::msg::PointField & field,
  const std::uint8_t datatype,
  const std::uint32_t value_size,
  const std::uint32_t point_step)
{
  return field.count == 1 &&
         field.datatype == datatype &&
         field.offset <= point_step &&
         value_size <= point_step - field.offset;
}

std::string lowercase(std::string value)
{
  std::transform(
    value.begin(),
    value.end(),
    value.begin(),
    [](const unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  return value;
}

}  // namespace

const char * cloudContractCodeString(const CloudContractCode code)
{
  switch (code) {
    case CloudContractCode::kValid:
      return "valid";
    case CloudContractCode::kEmptyScan:
      return "empty_scan";
    case CloudContractCode::kTooFewPoints:
      return "too_few_points";
    case CloudContractCode::kTooManyPoints:
      return "too_many_points";
    case CloudContractCode::kInvalidDimensions:
      return "invalid_dimensions";
    case CloudContractCode::kInvalidDataSize:
      return "invalid_data_size";
    case CloudContractCode::kBigEndianUnsupported:
      return "big_endian_unsupported";
    case CloudContractCode::kUnexpectedFrameId:
      return "unexpected_frame_id";
    case CloudContractCode::kMissingRequiredField:
      return "missing_required_field";
    case CloudContractCode::kUnsupportedFieldLayout:
      return "unsupported_field_layout";
    case CloudContractCode::kInsufficientFinitePoints:
      return "insufficient_finite_points";
    case CloudContractCode::kNonFiniteMeasurement:
      return "nonfinite_measurement";
    case CloudContractCode::kNonFiniteTimestamp:
      return "nonfinite_timestamp";
    case CloudContractCode::kTimestampRegression:
      return "timestamp_regression";
    case CloudContractCode::kZeroScanDuration:
      return "zero_scan_duration";
    case CloudContractCode::kScanDurationTooLong:
      return "scan_duration_too_long";
    case CloudContractCode::kTimestampOffsetOverflow:
      return "timestamp_offset_overflow";
    case CloudContractCode::kRingOutOfRange:
      return "ring_out_of_range";
    case CloudContractCode::kHeaderTimestampMismatch:
      return "header_timestamp_mismatch";
    case CloudContractCode::kMissingHeaderTimestamp:
      return "missing_header_timestamp";
    case CloudContractCode::kClockMisaligned:
      return "clock_misaligned";
    case CloudContractCode::kClockCalibrationOutOfRange:
      return "clock_calibration_out_of_range";
    case CloudContractCode::kNonIncreasingOutputStamp:
      return "non_increasing_output_stamp";
    case CloudContractCode::kOutputTimestampJump:
      return "output_timestamp_jump";
    case CloudContractCode::kOutputTimeOutsideRosRange:
      return "output_time_outside_ros_range";
  }
  return "unknown";
}

const char * clockPolicyString(const ClockPolicy policy)
{
  switch (policy) {
    case ClockPolicy::kSensorHeader:
      return "sensor_header";
    case ClockPolicy::kRosReceiveTimeScanEnd:
      return "ros_receive_time_scan_end";
    case ClockPolicy::kAnchoredSensorTime:
      return "anchored_sensor_time";
    case ClockPolicy::kCalibratedSensorTime:
      return "calibrated_sensor_time";
  }
  return "unknown";
}

std::optional<ClockPolicy> parseClockPolicy(const std::string & value)
{
  const auto normalized = lowercase(value);
  if (normalized == "sensor_header" || normalized == "input_header") {
    return ClockPolicy::kSensorHeader;
  }
  if (normalized == "ros_receive_time_scan_end" ||
    normalized == "receive_time_scan_end")
  {
    return ClockPolicy::kRosReceiveTimeScanEnd;
  }
  if (normalized == "anchored_sensor_time" ||
    normalized == "first_receive_anchor")
  {
    return ClockPolicy::kAnchoredSensorTime;
  }
  if (normalized == "calibrated_sensor_time" ||
    normalized == "fixed_calibrated_anchor")
  {
    return ClockPolicy::kCalibratedSensorTime;
  }
  return std::nullopt;
}

HesaiCloudContractResult validateHesaiPointCloud(
  const sensor_msgs::msg::PointCloud2 & message,
  const HesaiCloudContractOptions & options)
{
  HesaiCloudContractResult result;
  const std::uint64_t point_count =
    static_cast<std::uint64_t>(message.width) * message.height;
  if (point_count == 0) {
    result.code = CloudContractCode::kEmptyScan;
    return result;
  }
  if (point_count > std::numeric_limits<std::size_t>::max()) {
    result.code = CloudContractCode::kInvalidDimensions;
    return result;
  }
  result.point_count = static_cast<std::size_t>(point_count);
  if (result.point_count < std::max<std::size_t>(1, options.min_points)) {
    result.code = CloudContractCode::kTooFewPoints;
    return result;
  }
  if (result.point_count > std::max(options.min_points, options.max_points)) {
    result.code = CloudContractCode::kTooManyPoints;
    return result;
  }
  if (message.point_step == 0 || message.row_step == 0) {
    result.code = CloudContractCode::kInvalidDimensions;
    return result;
  }
  const std::uint64_t minimum_row_step =
    static_cast<std::uint64_t>(message.width) * message.point_step;
  const std::uint64_t required_data_size =
    static_cast<std::uint64_t>(message.row_step) * message.height;
  if (message.row_step < minimum_row_step) {
    result.code = CloudContractCode::kInvalidDimensions;
    return result;
  }
  if (required_data_size > message.data.size()) {
    result.code = CloudContractCode::kInvalidDataSize;
    return result;
  }
  if (message.is_bigendian) {
    result.code = CloudContractCode::kBigEndianUnsupported;
    return result;
  }
  if (options.require_expected_frame &&
    message.header.frame_id != options.expected_frame_id)
  {
    result.code = CloudContractCode::kUnexpectedFrameId;
    return result;
  }

  struct RequiredField
  {
    const char * name;
    std::uint8_t datatype;
    std::uint32_t value_size;
    std::uint32_t * destination;
  };
  RequiredField required_fields[] = {
    {"x", sensor_msgs::msg::PointField::FLOAT32, sizeof(float), &result.layout.x_offset},
    {"y", sensor_msgs::msg::PointField::FLOAT32, sizeof(float), &result.layout.y_offset},
    {"z", sensor_msgs::msg::PointField::FLOAT32, sizeof(float), &result.layout.z_offset},
    {
      "intensity", sensor_msgs::msg::PointField::FLOAT32, sizeof(float),
      &result.layout.intensity_offset
    },
    {
      "ring", sensor_msgs::msg::PointField::UINT16, sizeof(std::uint16_t),
      &result.layout.ring_offset
    },
    {
      "timestamp", sensor_msgs::msg::PointField::FLOAT64, sizeof(double),
      &result.layout.timestamp_offset
    },
  };

  for (const auto & required : required_fields) {
    const auto * field = findField(message, required.name);
    if (!field) {
      result.failed_field = required.name;
      result.code = CloudContractCode::kMissingRequiredField;
      return result;
    }
    if (!fieldMatches(
        *field,
        required.datatype,
        required.value_size,
        message.point_step))
    {
      result.failed_field = required.name;
      result.code = CloudContractCode::kUnsupportedFieldLayout;
      return result;
    }
    *required.destination = field->offset;
  }

  bool have_timestamp = false;
  double previous_timestamp_s = 0.0;
  std::size_t current_equal_timestamp_count = 0;
  result.min_point_time_s = std::numeric_limits<double>::infinity();
  result.max_point_time_s = -std::numeric_limits<double>::infinity();

  for (std::uint32_t row = 0; row < message.height; ++row) {
    const auto * row_start = message.data.data() +
      static_cast<std::size_t>(row) * message.row_step;
    for (std::uint32_t column = 0; column < message.width; ++column) {
      const auto * point = row_start +
        static_cast<std::size_t>(column) * message.point_step;
      const float x = readValue<float>(point, result.layout.x_offset);
      const float y = readValue<float>(point, result.layout.y_offset);
      const float z = readValue<float>(point, result.layout.z_offset);
      const std::uint16_t ring =
        readValue<std::uint16_t>(point, result.layout.ring_offset);
      const double timestamp_s =
        readValue<double>(point, result.layout.timestamp_offset);

      if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
        ++result.finite_point_count;
      } else {
        ++result.invalid_point_count;
      }
      result.observed_max_ring = std::max(result.observed_max_ring, ring);

      if (!std::isfinite(timestamp_s)) {
        ++result.nonfinite_timestamp_count;
        continue;
      }
      if (!have_timestamp) {
        result.first_point_time_s = timestamp_s;
        previous_timestamp_s = timestamp_s;
        current_equal_timestamp_count = 1;
        result.max_consecutive_equal_timestamps = 1;
        have_timestamp = true;
      } else {
        if (timestamp_s + std::max(0.0, options.timestamp_regression_tolerance_s) <
          previous_timestamp_s)
        {
          ++result.timestamp_regression_count;
        }
        if (timestamp_s == previous_timestamp_s) {
          ++current_equal_timestamp_count;
        } else {
          ++result.timestamp_change_count;
          current_equal_timestamp_count = 1;
        }
        result.max_consecutive_equal_timestamps = std::max(
          result.max_consecutive_equal_timestamps,
          current_equal_timestamp_count);
        previous_timestamp_s = timestamp_s;
      }
      result.min_point_time_s = std::min(result.min_point_time_s, timestamp_s);
      result.max_point_time_s = std::max(result.max_point_time_s, timestamp_s);
    }
  }

  result.finite_point_ratio =
    static_cast<double>(result.finite_point_count) /
    static_cast<double>(result.point_count);
  if (!have_timestamp || result.nonfinite_timestamp_count > 0) {
    result.code = CloudContractCode::kNonFiniteTimestamp;
    return result;
  }

  const double scan_duration_s =
    result.max_point_time_s - result.min_point_time_s;
  if (!std::isfinite(scan_duration_s) || scan_duration_s < 0.0) {
    result.code = CloudContractCode::kTimestampRegression;
    return result;
  }
  const long double scan_duration_ns =
    static_cast<long double>(scan_duration_s) * 1.0e9L;
  if (scan_duration_ns >
    static_cast<long double>(std::numeric_limits<std::uint32_t>::max()))
  {
    result.code = CloudContractCode::kTimestampOffsetOverflow;
    return result;
  }
  result.scan_duration_ns = static_cast<std::uint64_t>(
    std::llround(scan_duration_ns));

  result.header_time_s =
    static_cast<double>(message.header.stamp.sec) +
    static_cast<double>(message.header.stamp.nanosec) * 1.0e-9;
  result.header_first_point_delta_s =
    result.header_time_s - result.first_point_time_s;

  if (result.finite_point_ratio <
    std::clamp(options.min_finite_point_ratio, 0.0, 1.0))
  {
    result.code = CloudContractCode::kInsufficientFinitePoints;
  } else if (result.observed_max_ring > options.max_ring_id) {
    result.code = CloudContractCode::kRingOutOfRange;
  } else if (
    options.require_monotonic_timestamps &&
    result.timestamp_regression_count > 0)
  {
    result.code = CloudContractCode::kTimestampRegression;
  } else if (
    options.require_nonzero_duration &&
    result.point_count > 1 &&
    result.scan_duration_ns == 0)
  {
    result.code = CloudContractCode::kZeroScanDuration;
  } else if (
    options.max_scan_duration_ns > 0 &&
    result.scan_duration_ns > options.max_scan_duration_ns)
  {
    result.code = CloudContractCode::kScanDurationTooLong;
  } else if (
    options.require_header_matches_first_point &&
    std::abs(result.header_first_point_delta_s) >
    std::max(0.0, options.max_header_first_point_delta_s))
  {
    result.code = CloudContractCode::kHeaderTimestampMismatch;
  }

  return result;
}

OutputStampResult selectOutputStartStamp(
  const ClockPolicy policy,
  const std::int64_t input_header_ns,
  const std::int64_t receive_time_ns,
  const std::uint64_t scan_duration_ns,
  const std::int64_t configured_offset_ns,
  const std::int64_t anchor_input_header_ns,
  const std::int64_t anchor_output_start_ns,
  const double time_scale)
{
  OutputStampResult result;
  long double output_start_ns = 0.0L;
  if (policy == ClockPolicy::kSensorHeader) {
    if (input_header_ns <= 0) {
      result.code = CloudContractCode::kMissingHeaderTimestamp;
      return result;
    }
    output_start_ns =
      static_cast<long double>(input_header_ns) + configured_offset_ns;
  } else if (policy == ClockPolicy::kRosReceiveTimeScanEnd) {
    output_start_ns =
      static_cast<long double>(receive_time_ns) -
      static_cast<long double>(scan_duration_ns) +
      configured_offset_ns;
  } else {
    if (
      input_header_ns <= 0 ||
      anchor_input_header_ns <= 0 ||
      anchor_output_start_ns <= 0)
    {
      result.code = CloudContractCode::kMissingHeaderTimestamp;
      return result;
    }
    const bool calibrated = policy == ClockPolicy::kCalibratedSensorTime;
    if (calibrated && (!std::isfinite(time_scale) || time_scale <= 0.0)) {
      result.code = CloudContractCode::kClockMisaligned;
      return result;
    }
    const long double effective_scale = calibrated ?
      static_cast<long double>(time_scale) : 1.0L;
    output_start_ns =
      static_cast<long double>(anchor_output_start_ns) +
      effective_scale *
      (static_cast<long double>(input_header_ns) -
      static_cast<long double>(anchor_input_header_ns)) +
      configured_offset_ns;
  }

  constexpr std::int64_t kMaxRosTimeNs =
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) *
    1000000000LL + 999999999LL;
  if (!std::isfinite(output_start_ns) ||
    output_start_ns <= 0.0L ||
    output_start_ns > static_cast<long double>(kMaxRosTimeNs))
  {
    result.code = CloudContractCode::kOutputTimeOutsideRosRange;
    return result;
  }

  result.output_start_ns = static_cast<std::int64_t>(output_start_ns);
  return result;
}

PointOffsetResult selectOutputPointOffset(
  const double point_offset_s,
  const ClockPolicy policy,
  const double time_scale)
{
  PointOffsetResult result;
  if (!std::isfinite(point_offset_s) || point_offset_s < 0.0) {
    result.code = CloudContractCode::kTimestampRegression;
    return result;
  }

  const bool calibrated = policy == ClockPolicy::kCalibratedSensorTime;
  if (calibrated && (!std::isfinite(time_scale) || time_scale <= 0.0)) {
    result.code = CloudContractCode::kClockMisaligned;
    return result;
  }
  const long double effective_scale = calibrated ?
    static_cast<long double>(time_scale) : 1.0L;
  const long double rounded_offset_ns = std::round(
    static_cast<long double>(point_offset_s) * effective_scale * 1.0e9L);
  if (!std::isfinite(rounded_offset_ns) ||
    rounded_offset_ns < 0.0L ||
    rounded_offset_ns >
    static_cast<long double>(std::numeric_limits<std::uint32_t>::max()))
  {
    result.code = CloudContractCode::kTimestampOffsetOverflow;
    return result;
  }

  result.output_offset_ns = static_cast<std::uint32_t>(rounded_offset_ns);
  return result;
}

}  // namespace slam_3d_lidar_ingest
}  // namespace gole
