#include "slam_3d_lidar_ingest/hesai_pointcloud_ingest.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "slam_3d_lidar_ingest/point_budget.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

namespace
{

constexpr std::uint32_t kCanonicalPointStep = 24;

template<typename T>
T readField(const std::uint8_t * source, const std::uint32_t offset)
{
  T value{};
  std::memcpy(&value, source + offset, sizeof(T));
  return value;
}

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

HesaiPointCloudIngest::HesaiPointCloudIngest(const rclcpp::NodeOptions & options)
: Node("hesai_pointcloud_ingest", options)
{
  sensors_[0].name = "front";
  sensors_[1].name = "rear";
  sensors_[2].name = "center";

  input_depth_ = declare_parameter<int>("input_depth", 10);
  output_depth_ = declare_parameter<int>("output_depth", 5);
  const auto max_output_points = declare_parameter<std::int64_t>("max_output_points", 0);
  if (max_output_points < 0 || max_output_points == 1 ||
    max_output_points > std::numeric_limits<std::uint32_t>::max() / kCanonicalPointStep)
  {
    throw std::invalid_argument(
            "max_output_points must be 0 (unlimited) or >= 2 within cloud limits");
  }
  max_output_points_ = static_cast<std::size_t>(max_output_points);
  const bool input_reliable = declare_parameter<bool>("input_reliable", true);
  const bool output_reliable = declare_parameter<bool>("output_reliable", true);
  drop_nonfinite_points_ = declare_parameter<bool>("drop_nonfinite_points", true);
  reject_non_increasing_output_stamp_ = declare_parameter<bool>(
    "reject_non_increasing_output_stamp", true);
  reject_clock_misalignment_ = declare_parameter<bool>(
    "reject_clock_misalignment", true);
  max_sensor_header_to_receive_delta_s_ = declare_parameter<double>(
    "max_sensor_header_to_receive_delta_s", 0.25);

  HesaiCloudContractOptions common_options;
  constexpr std::int64_t kMaxCloudPoints =
    static_cast<std::int64_t>(
    std::numeric_limits<std::uint32_t>::max() / kCanonicalPointStep);
  const auto max_points = declare_parameter<std::int64_t>("max_points", 500000);
  common_options.max_points = static_cast<std::size_t>(
    std::clamp(max_points, std::int64_t{1}, kMaxCloudPoints));
  const double configured_max_scan_duration_s = declare_parameter<double>(
    "max_scan_duration_s", 0.2);
  const double max_supported_scan_duration_s =
    static_cast<double>(std::numeric_limits<std::uint32_t>::max()) * 1.0e-9;
  const double max_scan_duration_s = std::isfinite(configured_max_scan_duration_s) ?
    std::clamp(
    configured_max_scan_duration_s, 0.0, max_supported_scan_duration_s) :
    0.2;
  common_options.max_scan_duration_ns = static_cast<std::uint64_t>(
    max_scan_duration_s * 1.0e9);
  common_options.min_finite_point_ratio = declare_parameter<double>(
    "min_finite_point_ratio", 0.9);
  common_options.max_header_first_point_delta_s = declare_parameter<double>(
    "max_header_first_point_delta_s", 0.005);
  common_options.timestamp_regression_tolerance_s = declare_parameter<double>(
    "timestamp_regression_tolerance_s", 1.0e-9);
  common_options.require_monotonic_timestamps = declare_parameter<bool>(
    "require_monotonic_timestamps", true);
  common_options.require_nonzero_duration = declare_parameter<bool>(
    "require_nonzero_duration", true);
  common_options.require_header_matches_first_point = declare_parameter<bool>(
    "require_header_matches_first_point", true);
  for (auto & sensor : sensors_) {
    sensor.contract_options = common_options;
  }

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

  const std::string health_topic = declare_parameter<std::string>(
    "health_topic", "/slam_3d/ingest/health");
  health_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    health_topic, rclcpp::QoS(20).reliable());

  configureSensor(
    0,
    "/hesai/front/lidar_points",
    "/slam_3d/input/hesai_front/points_timed",
    "hesai_front",
    "JT128",
    0,
    127,
    200000,
    260000,
    input_qos,
    output_qos);
  configureSensor(
    1,
    "/hesai/rear/lidar_points",
    "/slam_3d/input/hesai_rear/points_timed",
    "hesai_rear",
    "JT128",
    1,
    127,
    200000,
    260000,
    input_qos,
    output_qos);
  configureSensor(
    2,
    "/hesai/center/lidar_points",
    "/slam_3d/input/hesai_center/points_timed",
    "hesai_center",
    "XT32",
    2,
    31,
    100000,
    160000,
    input_qos,
    output_qos);
}

void HesaiPointCloudIngest::configureSensor(
  const std::size_t sensor_index,
  const std::string & input_topic,
  const std::string & output_topic,
  const std::string & expected_frame_id,
  const std::string & model,
  const std::uint8_t sensor_id,
  const std::uint16_t max_ring_id,
  const std::size_t min_points,
  const std::size_t max_points,
  const rclcpp::QoS & input_qos,
  const rclcpp::QoS & output_qos)
{
  auto & sensor = sensors_.at(sensor_index);
  const std::string prefix = sensor.name + ".";
  const auto configured_input_topic = declare_parameter<std::string>(
    prefix + "input_topic", input_topic);
  const auto configured_output_topic = declare_parameter<std::string>(
    prefix + "output_topic", output_topic);
  sensor.contract_options.expected_frame_id = declare_parameter<std::string>(
    prefix + "expected_input_frame_id", expected_frame_id);
  sensor.contract_options.require_expected_frame = declare_parameter<bool>(
    prefix + "require_expected_input_frame", true);
  sensor.model = declare_parameter<std::string>(prefix + "model", model);
  sensor.clock_domain = declare_parameter<std::string>(
    prefix + "clock_domain", "unverified");
  sensor.time_offset_ns = declare_parameter<std::int64_t>(
    prefix + "time_offset_ns", 0);
  sensor.time_scale = declare_parameter<double>(prefix + "time_scale", 1.0);
  sensor.anchor_input_header_ns = declare_parameter<std::int64_t>(
    prefix + "anchor_input_header_ns", 0);
  sensor.anchor_output_start_ns = declare_parameter<std::int64_t>(
    prefix + "anchor_output_start_ns", 0);
  sensor.calibration_valid_min_input_header_ns = declare_parameter<std::int64_t>(
    prefix + "calibration_valid_min_input_header_ns", 0);
  sensor.calibration_valid_max_input_header_ns = declare_parameter<std::int64_t>(
    prefix + "calibration_valid_max_input_header_ns", 0);
  sensor.max_output_stamp_step_s = declare_parameter<double>(
    prefix + "max_output_stamp_step_s", 0.0);
  sensor.calibration_fit_rmse_ms = declare_parameter<double>(
    prefix + "calibration_fit_rmse_ms", -1.0);
  sensor.calibration_fit_max_abs_ms = declare_parameter<double>(
    prefix + "calibration_fit_max_abs_ms", -1.0);
  sensor.calibration_id = declare_parameter<std::string>(
    prefix + "calibration_id", "none");

  const auto configured_sensor_id = declare_parameter<int>(
    prefix + "sensor_id", sensor_id);
  sensor.sensor_id = static_cast<std::uint8_t>(
    std::clamp(
      configured_sensor_id,
      std::int64_t{0},
      std::int64_t{255}));
  const auto configured_max_ring = declare_parameter<int>(
    prefix + "max_ring_id", max_ring_id);
  sensor.contract_options.max_ring_id = static_cast<std::uint16_t>(
    std::clamp(
      configured_max_ring,
      std::int64_t{0},
      std::int64_t{65535}));
  const auto configured_min_points = declare_parameter<std::int64_t>(
    prefix + "min_points",
    static_cast<std::int64_t>(min_points));
  const auto configured_max_points = declare_parameter<std::int64_t>(
    prefix + "max_points",
    static_cast<std::int64_t>(max_points));
  sensor.contract_options.min_points = static_cast<std::size_t>(
    std::clamp(
      configured_min_points,
      std::int64_t{1},
      static_cast<std::int64_t>(sensor.contract_options.max_points)));
  sensor.contract_options.max_points = static_cast<std::size_t>(
    std::clamp(
      configured_max_points,
      static_cast<std::int64_t>(sensor.contract_options.min_points),
      static_cast<std::int64_t>(sensor.contract_options.max_points)));

  const auto configured_clock_policy = declare_parameter<std::string>(
    prefix + "clock_policy", "sensor_header");
  const auto parsed_policy = parseClockPolicy(configured_clock_policy);
  if (!parsed_policy.has_value()) {
    throw std::invalid_argument(
            "Unsupported " + prefix + "clock_policy: " + configured_clock_policy);
  }
  sensor.clock_policy = *parsed_policy;
  if (
    sensor.clock_policy == ClockPolicy::kCalibratedSensorTime &&
    (!std::isfinite(sensor.time_scale) ||
    sensor.time_scale < 0.5 ||
    sensor.time_scale > 2.0 ||
    sensor.anchor_input_header_ns <= 0 ||
    sensor.anchor_output_start_ns <= 0 ||
    sensor.calibration_valid_min_input_header_ns <= 0 ||
    sensor.calibration_valid_max_input_header_ns <
    sensor.calibration_valid_min_input_header_ns ||
    sensor.anchor_input_header_ns <
    sensor.calibration_valid_min_input_header_ns ||
    sensor.anchor_input_header_ns >
    sensor.calibration_valid_max_input_header_ns ||
    !std::isfinite(sensor.max_output_stamp_step_s) ||
    sensor.max_output_stamp_step_s <= 0.0 ||
    !std::isfinite(sensor.calibration_fit_rmse_ms) ||
    sensor.calibration_fit_rmse_ms < 0.0 ||
    !std::isfinite(sensor.calibration_fit_max_abs_ms) ||
    sensor.calibration_fit_max_abs_ms < sensor.calibration_fit_rmse_ms ||
    sensor.calibration_id.empty() ||
    sensor.calibration_id == "none"))
  {
    throw std::invalid_argument(
            "Invalid calibrated clock parameters for " + sensor.name);
  }
  if (!std::isfinite(sensor.max_output_stamp_step_s) ||
    sensor.max_output_stamp_step_s < 0.0)
  {
    throw std::invalid_argument(
            "Invalid max output stamp step for " + sensor.name);
  }

  sensor.publisher = create_publisher<sensor_msgs::msg::PointCloud2>(
    configured_output_topic, output_qos);
  sensor.callback_group = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions subscription_options;
  subscription_options.callback_group = sensor.callback_group;
  sensor.subscription = create_subscription<sensor_msgs::msg::PointCloud2>(
    configured_input_topic,
    input_qos,
    [this, sensor_index](const sensor_msgs::msg::PointCloud2::SharedPtr message) {
      processMessage(message, sensor_index);
    },
    subscription_options);
}

void HesaiPointCloudIngest::processMessage(
  const sensor_msgs::msg::PointCloud2::SharedPtr & message,
  const std::size_t sensor_index)
{
  auto & sensor = sensors_.at(sensor_index);
  auto result = validateHesaiPointCloud(*message, sensor.contract_options);
  if (result.accepted() && drop_nonfinite_points_ && result.finite_point_count == 0) {
    result.code = CloudContractCode::kInsufficientFinitePoints;
  }
  const std::int64_t receive_time_ns = get_clock()->now().nanoseconds();
  const std::int64_t input_header_ns = stampNanoseconds(message->header.stamp);
  if (
    result.accepted() &&
    sensor.clock_policy == ClockPolicy::kCalibratedSensorTime &&
    (input_header_ns < sensor.calibration_valid_min_input_header_ns ||
    input_header_ns > sensor.calibration_valid_max_input_header_ns))
  {
    result.code = CloudContractCode::kClockCalibrationOutOfRange;
  }
  if (result.accepted()) {
    const auto maximum_point_offset = selectOutputPointOffset(
      result.max_point_time_s - result.first_point_time_s,
      sensor.clock_policy,
      sensor.time_scale);
    if (!maximum_point_offset.accepted()) {
      result.code = maximum_point_offset.code;
    }
  }

  OutputStampResult output_stamp;
  if (
    result.accepted() &&
    sensor.clock_policy == ClockPolicy::kAnchoredSensorTime)
  {
    if (sensor.anchor_output_start_ns == 0) {
      const auto anchor_stamp = selectOutputStartStamp(
        ClockPolicy::kRosReceiveTimeScanEnd,
        input_header_ns,
        receive_time_ns,
        result.scan_duration_ns,
        sensor.time_offset_ns);
      if (anchor_stamp.accepted()) {
        sensor.anchor_input_header_ns = input_header_ns;
        sensor.anchor_output_start_ns = anchor_stamp.output_start_ns;
      } else {
        output_stamp = anchor_stamp;
      }
    }
    if (sensor.anchor_output_start_ns != 0) {
      output_stamp = selectOutputStartStamp(
        sensor.clock_policy,
        input_header_ns,
        receive_time_ns,
        result.scan_duration_ns,
        0,
        sensor.anchor_input_header_ns,
        sensor.anchor_output_start_ns);
    }
  } else {
    output_stamp = selectOutputStartStamp(
      sensor.clock_policy,
      input_header_ns,
      receive_time_ns,
      result.scan_duration_ns,
      sensor.time_offset_ns,
      sensor.anchor_input_header_ns,
      sensor.anchor_output_start_ns,
      sensor.time_scale);
  }

  if (result.accepted() && !output_stamp.accepted()) {
    result.code = output_stamp.code;
  }

  if (result.accepted() &&
    sensor.clock_policy == ClockPolicy::kSensorHeader &&
    reject_clock_misalignment_)
  {
    const long double corrected_header_ns =
      static_cast<long double>(input_header_ns) + sensor.time_offset_ns;
    const double delta_s = static_cast<double>(
      std::abs(static_cast<long double>(receive_time_ns) - corrected_header_ns) * 1.0e-9L);
    if (!std::isfinite(delta_s) ||
      delta_s > std::max(0.0, max_sensor_header_to_receive_delta_s_))
    {
      result.code = CloudContractCode::kClockMisaligned;
    }
  }

  if (result.accepted() &&
    reject_non_increasing_output_stamp_ &&
    sensor.last_output_start_ns.load() != 0 &&
    output_stamp.output_start_ns <= sensor.last_output_start_ns.load())
  {
    result.code = CloudContractCode::kNonIncreasingOutputStamp;
  }
  if (
    result.accepted() &&
    sensor.last_output_start_ns.load() != 0 &&
    sensor.max_output_stamp_step_s > 0.0)
  {
    const long double output_step_ns =
      static_cast<long double>(output_stamp.output_start_ns) -
      static_cast<long double>(sensor.last_output_start_ns.load());
    const long double maximum_step_ns =
      static_cast<long double>(sensor.max_output_stamp_step_s) * 1.0e9L;
    if (!std::isfinite(output_step_ns) ||
      output_step_ns > maximum_step_ns)
    {
      result.code = CloudContractCode::kOutputTimestampJump;
    }
  }

  const double peer_start_skew_ms = peerStartSkewMs(
    sensor, output_stamp.output_start_ns);
  if (!result.accepted()) {
    ++sensor.rejected_scan_count;
    publishDiagnostics(
      *message,
      sensor,
      result,
      receive_time_ns,
      output_stamp.output_start_ns,
      peer_start_skew_ms);
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected %s %s scan: %s field='%s' points=%zu duration=%.6fs "
      "raw_header_to_receive=%.3fs",
      sensor.name.c_str(),
      sensor.model.c_str(),
      cloudContractCodeString(result.code),
      result.failed_field.c_str(),
      result.point_count,
      static_cast<double>(result.scan_duration_ns) * 1.0e-9,
      static_cast<double>(receive_time_ns - input_header_ns) * 1.0e-9);
    return;
  }

  auto output = buildCanonicalCloud(*message, sensor, result, output_stamp.output_start_ns);
  sensor.last_output_point_count = output.width;
  sensor.publisher->publish(output);
  sensor.last_output_start_ns.store(output_stamp.output_start_ns);
  ++sensor.accepted_scan_count;
  publishDiagnostics(
    *message,
    sensor,
    result,
    receive_time_ns,
    output_stamp.output_start_ns,
    peer_start_skew_ms);
}

sensor_msgs::msg::PointCloud2 HesaiPointCloudIngest::buildCanonicalCloud(
  const sensor_msgs::msg::PointCloud2 & message,
  const SensorState & sensor,
  const HesaiCloudContractResult & result,
  const std::int64_t output_start_ns) const
{
  sensor_msgs::msg::PointCloud2 output;
  output.header.stamp.sec = static_cast<std::int32_t>(
    output_start_ns / 1000000000LL);
  output.header.stamp.nanosec = static_cast<std::uint32_t>(
    output_start_ns % 1000000000LL);
  output.header.frame_id = message.header.frame_id;
  output.height = 1;
  const std::size_t available_count = drop_nonfinite_points_ ?
    result.finite_point_count : result.point_count;
  const std::size_t selected_count = max_output_points_ == 0 ? available_count :
    std::min(available_count, max_output_points_);
  output.width = static_cast<std::uint32_t>(selected_count);
  output.fields = {
    makeField("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    makeField("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    makeField("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    makeField("intensity", 12, sensor_msgs::msg::PointField::FLOAT32),
    makeField("offset_time", 16, sensor_msgs::msg::PointField::UINT32),
    makeField("ring", 20, sensor_msgs::msg::PointField::UINT16),
    makeField("sensor_id", 22, sensor_msgs::msg::PointField::UINT8),
  };
  output.is_bigendian = false;
  output.point_step = kCanonicalPointStep;
  output.row_step = output.point_step * output.width;
  output.data.resize(output.row_step);
  output.is_dense = result.invalid_point_count == 0 || drop_nonfinite_points_;

  std::size_t output_index = 0;
  const auto append_point = [&](const std::uint8_t * source) {
      const float x = readField<float>(source, result.layout.x_offset);
      const float y = readField<float>(source, result.layout.y_offset);
      const float z = readField<float>(source, result.layout.z_offset);
      const bool finite = std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
      if (!finite && drop_nonfinite_points_) {
        return;
      }

      const float intensity =
        readField<float>(source, result.layout.intensity_offset);
      const std::uint16_t ring =
        readField<std::uint16_t>(source, result.layout.ring_offset);
      const double timestamp_s =
        readField<double>(source, result.layout.timestamp_offset);
      const auto point_offset = selectOutputPointOffset(
        std::max(0.0, timestamp_s - result.first_point_time_s),
        sensor.clock_policy,
        sensor.time_scale);
      const std::uint32_t point_offset_ns = point_offset.accepted() ?
        point_offset.output_offset_ns :
        std::numeric_limits<std::uint32_t>::max();

      auto * destination = output.data.data() + output_index * output.point_step;
      writeField(destination, 0, x);
      writeField(destination, 4, y);
      writeField(destination, 8, z);
      writeField(destination, 12, intensity);
      writeField(destination, 16, point_offset_ns);
      writeField(destination, 20, ring);
      writeField(destination, 22, sensor.sensor_id);
      ++output_index;
    };
  if (selected_count == available_count) {
    // Keep the uncapped sequential path free of per-point integer divisions.
    for (std::uint32_t row = 0; row < message.height; ++row) {
      const auto * row_start = message.data.data() +
        static_cast<std::size_t>(row) * message.row_step;
      for (std::uint32_t column = 0; column < message.width; ++column) {
        append_point(row_start + static_cast<std::size_t>(column) * message.point_step);
      }
    }
  } else {
    const auto point_at = [&](const std::size_t index) {
        return message.data.data() + (index / message.width) * message.row_step +
               (index % message.width) * message.point_step;
      };
    const auto finite_at = [&](const std::size_t index) {
        const auto * point = point_at(index);
        return std::isfinite(readField<float>(point, result.layout.x_offset)) &&
               std::isfinite(readField<float>(point, result.layout.y_offset)) &&
               std::isfinite(readField<float>(point, result.layout.z_offset));
      };
    std::size_t first = 0;
    std::size_t last = result.point_count - 1;
    if (drop_nonfinite_points_) {
      // There are more finite points than the cap here. Anchor sampling on
      // their extent so invalid edge records cannot remove both scan ends.
      while (!finite_at(first)) {
        ++first;
      }
      while (!finite_at(last)) {
        --last;
      }
    }
    for (std::size_t sample = 0; sample < selected_count; ++sample) {
      const std::size_t index = first + stratifiedPointIndex(
        sample, selected_count, last - first + 1, sensor.accepted_scan_count);
      append_point(point_at(index));
    }
  }

  output.width = static_cast<std::uint32_t>(output_index);
  output.row_step = output.width * output.point_step;
  output.data.resize(output.row_step);

  return output;
}

void HesaiPointCloudIngest::publishDiagnostics(
  const sensor_msgs::msg::PointCloud2 & message,
  const SensorState & sensor,
  const HesaiCloudContractResult & result,
  const std::int64_t receive_time_ns,
  const std::int64_t output_start_ns,
  const double peer_start_skew_ms) const
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp.sec = static_cast<std::int32_t>(
    receive_time_ns / 1000000000LL);
  array.header.stamp.nanosec = static_cast<std::uint32_t>(
    receive_time_ns % 1000000000LL);

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "slam_3d_lidar_ingest/" + sensor.name;
  status.hardware_id = "hesai_" + sensor.model + "_" + sensor.name;
  status.level = result.accepted() ?
    diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  status.message = cloudContractCodeString(result.code);

  const std::int64_t input_header_ns = stampNanoseconds(message.header.stamp);
  double anchor_receive_residual_ms = -1.0;
  if (
    (sensor.clock_policy == ClockPolicy::kAnchoredSensorTime ||
    sensor.clock_policy == ClockPolicy::kCalibratedSensorTime) &&
    sensor.anchor_output_start_ns > 0 &&
    output_start_ns > 0)
  {
    const long double receive_based_start_ns =
      static_cast<long double>(receive_time_ns) -
      static_cast<long double>(result.scan_duration_ns) +
      sensor.time_offset_ns;
    anchor_receive_residual_ms = static_cast<double>(
      (receive_based_start_ns - output_start_ns) * 1.0e-6L);
  }
  status.values = {
    keyValue("sensor_name", sensor.name),
    keyValue("sensor_model", sensor.model),
    keyValue("sensor_id", valueString(static_cast<int>(sensor.sensor_id))),
    keyValue("input_frame_id", message.header.frame_id),
    keyValue(
      "expected_input_frame_id",
      sensor.contract_options.expected_frame_id),
    keyValue("output_frame_id", message.header.frame_id),
    keyValue("clock_domain", sensor.clock_domain),
    keyValue("clock_policy", clockPolicyString(sensor.clock_policy)),
    keyValue("configured_time_offset_ns", valueString(sensor.time_offset_ns)),
    keyValue("configured_time_scale", valueString(sensor.time_scale)),
    keyValue("calibration_id", sensor.calibration_id),
    keyValue(
      "calibration_fit_rmse_ms",
      valueString(sensor.calibration_fit_rmse_ms)),
    keyValue(
      "calibration_fit_max_abs_ms",
      valueString(sensor.calibration_fit_max_abs_ms)),
    keyValue(
      "calibration_valid_min_input_header_ns",
      valueString(sensor.calibration_valid_min_input_header_ns)),
    keyValue(
      "calibration_valid_max_input_header_ns",
      valueString(sensor.calibration_valid_max_input_header_ns)),
    keyValue(
      "max_output_stamp_step_s",
      valueString(sensor.max_output_stamp_step_s)),
    keyValue(
      "time_anchor_source",
      sensor.clock_policy == ClockPolicy::kCalibratedSensorTime ?
      "configured" :
      (sensor.clock_policy == ClockPolicy::kAnchoredSensorTime ?
      "first_receive" : "none")),
    keyValue(
      "time_anchor_initialized",
      sensor.anchor_output_start_ns > 0 ? "true" : "false"),
    keyValue(
      "time_anchor_input_header_ns",
      valueString(sensor.anchor_input_header_ns)),
    keyValue(
      "time_anchor_output_start_ns",
      valueString(sensor.anchor_output_start_ns)),
    keyValue(
      "anchor_receive_residual_ms",
      valueString(anchor_receive_residual_ms)),
    keyValue("input_header_ns", valueString(input_header_ns)),
    keyValue("receive_time_ns", valueString(receive_time_ns)),
    keyValue(
      "input_header_to_receive_s",
      valueString(static_cast<double>(receive_time_ns - input_header_ns) * 1.0e-9)),
    keyValue("output_start_ns", valueString(output_start_ns)),
    keyValue("first_point_time_s", valueString(result.first_point_time_s)),
    keyValue(
      "header_first_point_delta_s",
      valueString(result.header_first_point_delta_s)),
    keyValue(
      "scan_duration_s",
      valueString(static_cast<double>(result.scan_duration_ns) * 1.0e-9)),
    keyValue("point_count", valueString(result.point_count)),
    keyValue("max_output_points", valueString(max_output_points_)),
    keyValue(
      "output_point_count",
      valueString(result.accepted() ? sensor.last_output_point_count : 0)),
    keyValue("finite_point_count", valueString(result.finite_point_count)),
    keyValue("invalid_point_count", valueString(result.invalid_point_count)),
    keyValue("finite_point_ratio", valueString(result.finite_point_ratio)),
    keyValue(
      "timestamp_regression_count",
      valueString(result.timestamp_regression_count)),
    keyValue("timestamp_change_count", valueString(result.timestamp_change_count)),
    keyValue(
      "max_consecutive_equal_timestamps",
      valueString(result.max_consecutive_equal_timestamps)),
    keyValue("configured_min_points", valueString(sensor.contract_options.min_points)),
    keyValue("configured_max_points", valueString(sensor.contract_options.max_points)),
    keyValue("observed_max_ring", valueString(result.observed_max_ring)),
    keyValue("configured_max_ring", valueString(sensor.contract_options.max_ring_id)),
    keyValue("failed_field", result.failed_field),
    keyValue("peer_start_skew_ms", valueString(peer_start_skew_ms)),
    keyValue("accepted_scan_count", valueString(sensor.accepted_scan_count)),
    keyValue("rejected_scan_count", valueString(sensor.rejected_scan_count)),
  };
  array.status.push_back(std::move(status));
  health_publisher_->publish(array);
}

double HesaiPointCloudIngest::peerStartSkewMs(
  const SensorState & sensor,
  const std::int64_t output_start_ns) const
{
  if (output_start_ns <= 0) {
    return -1.0;
  }
  double maximum_skew_ms = -1.0;
  for (const auto & peer : sensors_) {
    const auto peer_start_ns = peer.last_output_start_ns.load();
    if (&peer == &sensor || peer_start_ns <= 0) {
      continue;
    }
    maximum_skew_ms = std::max(
      maximum_skew_ms,
      std::abs(
        static_cast<double>(output_start_ns - peer_start_ns)) * 1.0e-6);
  }
  return maximum_skew_ms;
}

std::int64_t HesaiPointCloudIngest::stampNanoseconds(
  const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<std::int64_t>(stamp.sec) * 1000000000LL +
         static_cast<std::int64_t>(stamp.nanosec);
}

}  // namespace slam_3d_lidar_ingest
}  // namespace gole
