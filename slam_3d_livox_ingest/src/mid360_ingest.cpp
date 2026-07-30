#include "slam_3d_livox_ingest/mid360_ingest.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "sensor_msgs/msg/point_field.hpp"

namespace gole
{
namespace slam_3d_livox_ingest
{

namespace
{

constexpr std::uint32_t kPointStep = 24;

template<typename T>
void writeField(std::uint8_t * destination, const std::uint32_t offset, const T & value)
{
  std::memcpy(destination + offset, &value, sizeof(T));
}

sensor_msgs::msg::PointField makeField(
  std::string name,
  const std::uint32_t offset,
  const std::uint8_t datatype)
{
  sensor_msgs::msg::PointField field;
  field.name = std::move(name);
  field.offset = offset;
  field.datatype = datatype;
  field.count = 1;
  return field;
}

diagnostic_msgs::msg::KeyValue keyValue(std::string key, std::string value)
{
  diagnostic_msgs::msg::KeyValue item;
  item.key = std::move(key);
  item.value = std::move(value);
  return item;
}

template<typename T>
std::string valueString(const T & value)
{
  return std::to_string(value);
}

}  // namespace

Mid360Ingest::Mid360Ingest(const rclcpp::NodeOptions & options)
: Node("mid360_ingest", options)
{
  front_.name = "front";
  rear_.name = "rear";

  const std::string front_input_topic = declare_parameter<std::string>(
    "front_input_topic", "/slam_3d/input/mid360_front/livox");
  const std::string rear_input_topic = declare_parameter<std::string>(
    "rear_input_topic", "/slam_3d/input/mid360_rear/livox");
  const std::string front_output_topic = declare_parameter<std::string>(
    "front_output_topic", "/slam_3d/input/mid360_front/points_timed");
  const std::string rear_output_topic = declare_parameter<std::string>(
    "rear_output_topic", "/slam_3d/input/mid360_rear/points_timed");
  const std::string health_topic = declare_parameter<std::string>(
    "health_topic", "/slam_3d/ingest/health");
  front_.frame_id = declare_parameter<std::string>(
    "front_frame_id", "mid360_front_link");
  rear_.frame_id = declare_parameter<std::string>(
    "rear_frame_id", "mid360_rear_link");
  front_.clock_domain = declare_parameter<std::string>(
    "front_clock_domain", "unverified");
  rear_.clock_domain = declare_parameter<std::string>(
    "rear_clock_domain", "unverified");
  front_.time_offset_ns = declare_parameter<std::int64_t>(
    "front_time_offset_ns", 0);
  rear_.time_offset_ns = declare_parameter<std::int64_t>(
    "rear_time_offset_ns", 0);

  input_depth_ = declare_parameter<int>("input_depth", 10);
  output_depth_ = declare_parameter<int>("output_depth", 5);
  const bool input_reliable = declare_parameter<bool>("input_reliable", false);
  const bool output_reliable = declare_parameter<bool>("output_reliable", false);
  reject_non_increasing_timebase_ = declare_parameter<bool>(
    "reject_non_increasing_timebase", true);
  drop_nonfinite_points_ = declare_parameter<bool>("drop_nonfinite_points", true);
  max_header_timebase_delta_ms_ = declare_parameter<double>(
    "max_header_timebase_delta_ms", 5.0);

  constexpr std::int64_t kMaxCloudPoints =
    static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max() / kPointStep);
  const auto max_points = declare_parameter<std::int64_t>("max_points", 500000);
  contract_options_.max_points = static_cast<std::size_t>(
    std::clamp(max_points, std::int64_t{1}, kMaxCloudPoints));
  const double configured_max_scan_duration_s = declare_parameter<double>(
    "max_scan_duration_s", 0.2);
  const double max_supported_scan_duration_s =
    static_cast<double>(std::numeric_limits<std::uint32_t>::max()) * 1.0e-9;
  const double max_scan_duration_s = std::isfinite(configured_max_scan_duration_s) ?
    std::clamp(
    configured_max_scan_duration_s, 0.0, max_supported_scan_duration_s) :
    0.2;
  contract_options_.max_scan_duration_ns =
    static_cast<std::uint64_t>(max_scan_duration_s * 1.0e9);
  const auto first_point_max_offset_ns = declare_parameter<std::int64_t>(
    "first_point_max_offset_ns", 1000000);
  contract_options_.first_point_max_offset_ns = static_cast<std::uint32_t>(std::clamp(
      first_point_max_offset_ns,
      std::int64_t{0},
      static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())));
  contract_options_.min_finite_point_ratio = declare_parameter<double>(
    "min_finite_point_ratio", 0.9);
  contract_options_.require_point_count_match = declare_parameter<bool>(
    "require_point_count_match", true);
  contract_options_.require_monotonic_offsets = declare_parameter<bool>(
    "require_monotonic_offsets", true);
  contract_options_.require_zero_based_offset = declare_parameter<bool>(
    "require_zero_based_offset", true);
  contract_options_.require_nonzero_duration = declare_parameter<bool>(
    "require_nonzero_duration", true);

  auto input_qos = rclcpp::QoS(
    rclcpp::KeepLast(std::max(1, input_depth_))).durability_volatile();
  if (input_reliable) {
    input_qos.reliable();
  } else {
    input_qos.best_effort();
  }
  auto output_qos = rclcpp::QoS(
    rclcpp::KeepLast(std::max(1, output_depth_))).durability_volatile();
  if (output_reliable) {
    output_qos.reliable();
  } else {
    output_qos.best_effort();
  }

  front_.publisher =
    create_publisher<sensor_msgs::msg::PointCloud2>(front_output_topic, output_qos);
  rear_.publisher =
    create_publisher<sensor_msgs::msg::PointCloud2>(rear_output_topic, output_qos);
  health_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    health_topic, rclcpp::QoS(20).reliable());

  front_.subscription = create_subscription<CustomMsg>(
    front_input_topic,
    input_qos,
    [this](const CustomMsg::SharedPtr message) {
      processMessage(message, front_);
    });
  rear_.subscription = create_subscription<CustomMsg>(
    rear_input_topic,
    input_qos,
    [this](const CustomMsg::SharedPtr message) {
      processMessage(message, rear_);
    });
}

void Mid360Ingest::processMessage(
  const CustomMsg::SharedPtr & message,
  SensorState & sensor)
{
  auto result = validateMid360TimeContract(
    message->timebase,
    message->point_num,
    message->points,
    contract_options_);
  const std::int64_t corrected_timebase_ns = correctedTimebaseNanoseconds(
    message->timebase, sensor.time_offset_ns);
  constexpr std::int64_t kMaxRosTimeNs =
    static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) *
    1000000000LL + 999999999LL;
  if (result.accepted() &&
    (corrected_timebase_ns <= 0 || corrected_timebase_ns > kMaxRosTimeNs))
  {
    result.code = TimeContractCode::kTimebaseOutsideRosRange;
  }

  if (result.accepted() && reject_non_increasing_timebase_ &&
    sensor.last_accepted_corrected_timebase_ns != 0 &&
    corrected_timebase_ns <= sensor.last_accepted_corrected_timebase_ns)
  {
    result.code = TimeContractCode::kNonIncreasingTimebase;
  }

  const std::int64_t header_ns = stampNanoseconds(message->header.stamp);
  const double header_timebase_delta_ms = header_ns > 0 ?
    std::abs(
    static_cast<double>(header_ns) - static_cast<double>(message->timebase)) * 1.0e-6 :
    -1.0;
  const auto & peer = peerOf(sensor);
  const double peer_start_skew_ms = peer.last_accepted_corrected_timebase_ns > 0 ?
    std::abs(
    static_cast<double>(peer.last_accepted_corrected_timebase_ns) -
    static_cast<double>(corrected_timebase_ns)) * 1.0e-6 :
    -1.0;

  if (!result.accepted()) {
    ++sensor.rejected_scan_count;
    publishDiagnostics(
      *message,
      sensor,
      result,
      header_timebase_delta_ms,
      peer_start_skew_ms);
    RCLCPP_WARN(
      get_logger(),
      "Rejected %s Mid-360 scan: %s, declared=%zu actual=%zu duration=%.6fs",
      sensor.name.c_str(),
      timeContractCodeString(result.code),
      result.declared_point_count,
      result.actual_point_count,
      static_cast<double>(result.scan_duration_ns) * 1.0e-9);
    return;
  }

  sensor.publisher->publish(buildCanonicalCloud(*message, sensor, result));
  sensor.last_accepted_timebase_ns = message->timebase;
  sensor.last_accepted_corrected_timebase_ns = corrected_timebase_ns;
  ++sensor.accepted_scan_count;
  publishDiagnostics(
    *message,
    sensor,
    result,
    header_timebase_delta_ms,
    peer_start_skew_ms);
}

sensor_msgs::msg::PointCloud2 Mid360Ingest::buildCanonicalCloud(
  const CustomMsg & message,
  const SensorState & sensor,
  const TimeContractResult & result) const
{
  sensor_msgs::msg::PointCloud2 output;
  const std::int64_t corrected_timebase_ns = correctedTimebaseNanoseconds(
    message.timebase, sensor.time_offset_ns);
  output.header.stamp.sec = static_cast<std::int32_t>(
    corrected_timebase_ns / 1000000000LL);
  output.header.stamp.nanosec =
    static_cast<std::uint32_t>(corrected_timebase_ns % 1000000000LL);
  output.header.frame_id =
    sensor.frame_id.empty() ? message.header.frame_id : sensor.frame_id;
  output.height = 1;
  output.width = static_cast<std::uint32_t>(
    drop_nonfinite_points_ ? result.finite_point_count : result.actual_point_count);
  output.fields = {
    makeField("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    makeField("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    makeField("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    makeField("intensity", 12, sensor_msgs::msg::PointField::FLOAT32),
    makeField("offset_time", 16, sensor_msgs::msg::PointField::UINT32),
    makeField("tag", 20, sensor_msgs::msg::PointField::UINT8),
    makeField("line", 21, sensor_msgs::msg::PointField::UINT8),
    makeField("lidar_id", 22, sensor_msgs::msg::PointField::UINT8),
  };
  output.is_bigendian = false;
  output.point_step = kPointStep;
  output.row_step = output.point_step * output.width;
  output.data.resize(output.row_step);
  output.is_dense = result.invalid_point_count == 0 || drop_nonfinite_points_;

  std::size_t output_index = 0;
  for (const auto & point : message.points) {
    const bool finite =
      std::isfinite(static_cast<double>(point.x)) &&
      std::isfinite(static_cast<double>(point.y)) &&
      std::isfinite(static_cast<double>(point.z));
    if (!finite && drop_nonfinite_points_) {
      continue;
    }

    auto * destination = output.data.data() + output_index * output.point_step;
    const float intensity = static_cast<float>(point.reflectivity);
    writeField(destination, 0, point.x);
    writeField(destination, 4, point.y);
    writeField(destination, 8, point.z);
    writeField(destination, 12, intensity);
    writeField(destination, 16, point.offset_time);
    writeField(destination, 20, point.tag);
    writeField(destination, 21, point.line);
    writeField(destination, 22, message.lidar_id);
    ++output_index;
  }

  return output;
}

void Mid360Ingest::publishDiagnostics(
  const CustomMsg & message,
  const SensorState & sensor,
  const TimeContractResult & result,
  const double header_timebase_delta_ms,
  const double peer_start_skew_ms) const
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = message.header.stamp;
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "slam_3d_livox_ingest/" + sensor.name;
  status.hardware_id = "livox_mid360_" + sensor.name;
  status.level = result.accepted() ?
    diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  status.message = timeContractCodeString(result.code);

  if (result.accepted() && header_timebase_delta_ms >= 0.0 &&
    max_header_timebase_delta_ms_ > 0.0 &&
    header_timebase_delta_ms > max_header_timebase_delta_ms_)
  {
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = "header_timebase_delta";
  }

  status.values = {
    keyValue("contract_status", timeContractCodeString(result.code)),
    keyValue("lidar_id", valueString(static_cast<unsigned int>(message.lidar_id))),
    keyValue("raw_timebase_ns", valueString(message.timebase)),
    keyValue("time_offset_ns", valueString(sensor.time_offset_ns)),
    keyValue(
      "corrected_timebase_ns", valueString(
        correctedTimebaseNanoseconds(message.timebase, sensor.time_offset_ns))),
    keyValue("clock_domain", sensor.clock_domain),
    keyValue("header_timebase_delta_ms", valueString(header_timebase_delta_ms)),
    keyValue("latest_peer_start_skew_ms", valueString(peer_start_skew_ms)),
    keyValue("declared_point_count", valueString(result.declared_point_count)),
    keyValue("actual_point_count", valueString(result.actual_point_count)),
    keyValue("finite_point_count", valueString(result.finite_point_count)),
    keyValue("invalid_point_count", valueString(result.invalid_point_count)),
    keyValue("finite_point_ratio", valueString(result.finite_point_ratio)),
    keyValue("first_offset_ns", valueString(result.first_offset_ns)),
    keyValue("min_offset_ns", valueString(result.min_offset_ns)),
    keyValue("max_offset_ns", valueString(result.max_offset_ns)),
    keyValue(
      "scan_duration_s", valueString(
        static_cast<double>(result.scan_duration_ns) * 1.0e-9)),
    keyValue("offset_regression_count", valueString(result.offset_regression_count)),
    keyValue("accepted_scan_count", valueString(sensor.accepted_scan_count)),
    keyValue("rejected_scan_count", valueString(sensor.rejected_scan_count)),
    keyValue("output_frame", sensor.frame_id.empty() ? message.header.frame_id : sensor.frame_id),
  };
  array.status.push_back(std::move(status));
  health_publisher_->publish(array);
}

std::int64_t Mid360Ingest::stampNanoseconds(
  const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

std::int64_t Mid360Ingest::correctedTimebaseNanoseconds(
  const std::uint64_t timebase_ns,
  const std::int64_t offset_ns)
{
  constexpr auto kInvalid = std::numeric_limits<std::int64_t>::min();
  if (timebase_ns >
    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
  {
    return kInvalid;
  }

  const auto base_ns = static_cast<std::int64_t>(timebase_ns);
  if (offset_ns > 0 &&
    base_ns > std::numeric_limits<std::int64_t>::max() - offset_ns)
  {
    return kInvalid;
  }
  if (offset_ns == std::numeric_limits<std::int64_t>::min()) {
    return kInvalid;
  }
  if (offset_ns < 0 && base_ns < -offset_ns) {
    return kInvalid;
  }
  return base_ns + offset_ns;
}

const Mid360Ingest::SensorState & Mid360Ingest::peerOf(
  const SensorState & sensor) const
{
  return &sensor == &front_ ? rear_ : front_;
}

}  // namespace slam_3d_livox_ingest
}  // namespace gole
