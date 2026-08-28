#include "slam_3d_lidar_ingest/hesai_imu_normalizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

namespace
{

bool finiteVector(const geometry_msgs::msg::Vector3 & value)
{
  return std::isfinite(value.x) &&
         std::isfinite(value.y) &&
         std::isfinite(value.z);
}

void scaleVector(geometry_msgs::msg::Vector3 & value, const double scale)
{
  value.x *= scale;
  value.y *= scale;
  value.z *= scale;
}

void scaleCovariance(std::array<double, 9> & covariance, const double scale)
{
  if (covariance.front() < 0.0) {
    return;
  }
  const double covariance_scale = scale * scale;
  for (auto & value : covariance) {
    value *= covariance_scale;
  }
}

builtin_interfaces::msg::Time timeFromNanoseconds(const std::int64_t stamp_ns)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<std::int32_t>(stamp_ns / 1000000000LL);
  stamp.nanosec = static_cast<std::uint32_t>(stamp_ns % 1000000000LL);
  return stamp;
}

diagnostic_msgs::msg::KeyValue keyValue(
  std::string key,
  std::string value)
{
  diagnostic_msgs::msg::KeyValue item;
  item.key = std::move(key);
  item.value = std::move(value);
  return item;
}

std::string boolString(const bool value)
{
  return value ? "true" : "false";
}

}  // namespace

ImuNormalizationResult normalizeHesaiImu(
  const sensor_msgs::msg::Imu & input,
  const std::int64_t receive_time_ns,
  const ImuNormalizationOptions & options)
{
  ImuNormalizationResult result;
  result.message = input;
  result.input_stamp_ns =
    static_cast<std::int64_t>(input.header.stamp.sec) * 1000000000LL +
    static_cast<std::int64_t>(input.header.stamp.nanosec);

  if (result.input_stamp_ns <= 0) {
    result.code = CloudContractCode::kMissingHeaderTimestamp;
    return result;
  }
  if (options.require_expected_input_frame &&
    input.header.frame_id != options.expected_input_frame_id)
  {
    result.code = CloudContractCode::kUnexpectedFrameId;
    return result;
  }
  if (!finiteVector(input.angular_velocity) ||
    !finiteVector(input.linear_acceleration) ||
    !std::isfinite(options.angular_velocity_scale) ||
    !std::isfinite(options.linear_acceleration_scale))
  {
    result.code = CloudContractCode::kNonFiniteMeasurement;
    return result;
  }
  if (options.clock_policy == ClockPolicy::kCalibratedSensorTime) {
    if (options.calibration_valid_min_input_header_ns > 0 &&
      result.input_stamp_ns < options.calibration_valid_min_input_header_ns)
    {
      result.code = CloudContractCode::kClockCalibrationOutOfRange;
      return result;
    }
    if (options.calibration_valid_max_input_header_ns > 0 &&
      result.input_stamp_ns > options.calibration_valid_max_input_header_ns)
    {
      result.code = CloudContractCode::kClockCalibrationOutOfRange;
      return result;
    }
  }

  const auto output_stamp = selectOutputStartStamp(
    options.clock_policy,
    result.input_stamp_ns,
    receive_time_ns,
    0U,
    options.time_offset_ns,
    options.anchor_input_header_ns,
    options.anchor_output_header_ns,
    options.time_scale);
  if (!output_stamp.accepted()) {
    result.code = output_stamp.code;
    return result;
  }

  result.output_stamp_ns = output_stamp.output_start_ns;
  result.message.header.stamp = timeFromNanoseconds(result.output_stamp_ns);
  if (!options.output_frame_id.empty()) {
    result.message.header.frame_id = options.output_frame_id;
  }
  scaleVector(result.message.angular_velocity, options.angular_velocity_scale);
  scaleVector(result.message.linear_acceleration, options.linear_acceleration_scale);
  scaleCovariance(
    result.message.angular_velocity_covariance,
    options.angular_velocity_scale);
  scaleCovariance(
    result.message.linear_acceleration_covariance,
    options.linear_acceleration_scale);
  return result;
}

HesaiImuNormalizer::HesaiImuNormalizer(const rclcpp::NodeOptions & options)
: Node("hesai_imu_normalizer", options)
{
  const auto input_topic = declare_parameter<std::string>(
    "input_topic",
    "/hesai/front/lidar_imu");
  const auto output_topic = declare_parameter<std::string>(
    "output_topic",
    "/slam_3d/input/hesai_front/imu");
  const auto health_topic = declare_parameter<std::string>(
    "health_topic",
    "/slam_3d/ingest/imu_health");
  options_.expected_input_frame_id = declare_parameter<std::string>(
    "expected_input_frame_id",
    "hesai_front");
  options_.output_frame_id = declare_parameter<std::string>(
    "output_frame_id",
    "hesai_front");
  options_.require_expected_input_frame = declare_parameter<bool>(
    "require_expected_input_frame",
    true);
  clock_domain_ = declare_parameter<std::string>(
    "clock_domain",
    "ptp_required_unverified");
  const auto clock_policy_text = declare_parameter<std::string>(
    "clock_policy",
    "sensor_header");
  const auto clock_policy = parseClockPolicy(clock_policy_text);
  if (!clock_policy.has_value()) {
    throw std::invalid_argument("unsupported clock_policy: " + clock_policy_text);
  }
  options_.clock_policy = *clock_policy;
  options_.time_offset_ns = declare_parameter<std::int64_t>("time_offset_ns", 0);
  options_.time_scale = declare_parameter<double>("time_scale", 1.0);
  options_.anchor_input_header_ns = declare_parameter<std::int64_t>(
    "anchor_input_header_ns",
    0);
  options_.anchor_output_header_ns = declare_parameter<std::int64_t>(
    "anchor_output_header_ns",
    0);
  options_.calibration_valid_min_input_header_ns =
    declare_parameter<std::int64_t>(
    "calibration_valid_min_input_header_ns",
    0);
  options_.calibration_valid_max_input_header_ns =
    declare_parameter<std::int64_t>(
    "calibration_valid_max_input_header_ns",
    0);
  options_.angular_velocity_scale = declare_parameter<double>(
    "angular_velocity_scale",
    1.0);
  options_.linear_acceleration_scale = declare_parameter<double>(
    "linear_acceleration_scale",
    1.0);
  max_output_stamp_step_s_ = declare_parameter<double>(
    "max_output_stamp_step_s",
    0.0);
  reject_non_increasing_output_stamp_ = declare_parameter<bool>(
    "reject_non_increasing_output_stamp",
    true);
  diagnostic_period_s_ = declare_parameter<double>("diagnostic_period_s", 1.0);
  calibration_fit_rmse_ms_ = declare_parameter<double>(
    "calibration_fit_rmse_ms",
    -1.0);
  calibration_fit_max_abs_ms_ = declare_parameter<double>(
    "calibration_fit_max_abs_ms",
    -1.0);
  calibration_id_ = declare_parameter<std::string>(
    "calibration_id",
    "none");
  const int input_depth = declare_parameter<int>("input_depth", 400);
  const int output_depth = declare_parameter<int>("output_depth", 400);
  const bool input_reliable = declare_parameter<bool>("input_reliable", true);
  const bool output_reliable = declare_parameter<bool>("output_reliable", true);

  if (!std::isfinite(options_.time_scale) || options_.time_scale <= 0.0) {
    throw std::invalid_argument("time_scale must be finite and positive");
  }
  if (!std::isfinite(options_.angular_velocity_scale) ||
    !std::isfinite(options_.linear_acceleration_scale))
  {
    throw std::invalid_argument("IMU unit scales must be finite");
  }

  auto input_qos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, input_depth)));
  auto output_qos = rclcpp::QoS(rclcpp::KeepLast(std::max(1, output_depth)));
  if (input_reliable) {
    input_qos.reliable();
  } else {
    input_qos.best_effort();
  }
  if (output_reliable) {
    output_qos.reliable();
  } else {
    output_qos.best_effort();
  }

  publisher_ = create_publisher<sensor_msgs::msg::Imu>(output_topic, output_qos);
  health_publisher_ =
    create_publisher<diagnostic_msgs::msg::DiagnosticArray>(health_topic, 20);
  subscription_ = create_subscription<sensor_msgs::msg::Imu>(
    input_topic,
    input_qos,
    std::bind(
      &HesaiImuNormalizer::processMessage,
      this,
      std::placeholders::_1));
}

void HesaiImuNormalizer::processMessage(
  const sensor_msgs::msg::Imu::SharedPtr message)
{
  const std::int64_t receive_time_ns = now().nanoseconds();
  auto result = normalizeHesaiImu(*message, receive_time_ns, options_);

  if (result.accepted()) {
    const std::int64_t previous_stamp_ns = last_output_stamp_ns_.load();
    if (reject_non_increasing_output_stamp_ &&
      previous_stamp_ns > 0 &&
      result.output_stamp_ns <= previous_stamp_ns)
    {
      result.code = CloudContractCode::kNonIncreasingOutputStamp;
    } else if (
      max_output_stamp_step_s_ > 0.0 &&
      previous_stamp_ns > 0 &&
      static_cast<double>(result.output_stamp_ns - previous_stamp_ns) * 1.0e-9 >
      max_output_stamp_step_s_)
    {
      result.code = CloudContractCode::kOutputTimestampJump;
    }
  }

  if (result.accepted()) {
    last_output_stamp_ns_.store(result.output_stamp_ns);
    ++accepted_count_;
    publisher_->publish(result.message);
  } else {
    ++rejected_count_;
  }

  const std::int64_t last_diagnostic_ns = last_diagnostic_publish_ns_.load();
  const std::int64_t diagnostic_period_ns = static_cast<std::int64_t>(
    std::max(0.0, diagnostic_period_s_) * 1.0e9);
  if (!result.accepted() ||
    diagnostic_period_ns == 0 ||
    last_diagnostic_ns == 0 ||
    receive_time_ns < last_diagnostic_ns ||
    receive_time_ns - last_diagnostic_ns >= diagnostic_period_ns)
  {
    last_diagnostic_publish_ns_.store(receive_time_ns);
    publishDiagnostics(*message, result, receive_time_ns);
  }
}

void HesaiImuNormalizer::publishDiagnostics(
  const sensor_msgs::msg::Imu & input,
  const ImuNormalizationResult & result,
  const std::int64_t receive_time_ns)
{
  diagnostic_msgs::msg::DiagnosticArray diagnostics;
  diagnostics.header.stamp = now();
  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = get_fully_qualified_name() + std::string("/lidar_imu");
  status.hardware_id = input.header.frame_id;
  status.level = result.accepted() ?
    diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::ERROR;
  status.message = cloudContractCodeString(result.code);
  status.values = {
    keyValue("accepted", boolString(result.accepted())),
    keyValue("input_frame_id", input.header.frame_id),
    keyValue("output_frame_id", result.message.header.frame_id),
    keyValue("clock_domain", clock_domain_),
    keyValue("clock_policy", clockPolicyString(options_.clock_policy)),
    keyValue("calibration_id", calibration_id_),
    keyValue("input_stamp_ns", std::to_string(result.input_stamp_ns)),
    keyValue("output_stamp_ns", std::to_string(result.output_stamp_ns)),
    keyValue("receive_time_ns", std::to_string(receive_time_ns)),
    keyValue("time_scale", std::to_string(options_.time_scale)),
    keyValue(
      "angular_velocity_scale",
      std::to_string(options_.angular_velocity_scale)),
    keyValue(
      "linear_acceleration_scale",
      std::to_string(options_.linear_acceleration_scale)),
    keyValue("calibration_fit_rmse_ms", std::to_string(calibration_fit_rmse_ms_)),
    keyValue(
      "calibration_fit_max_abs_ms",
      std::to_string(calibration_fit_max_abs_ms_)),
    keyValue("accepted_count", std::to_string(accepted_count_)),
    keyValue("rejected_count", std::to_string(rejected_count_)),
  };
  diagnostics.status.push_back(std::move(status));
  health_publisher_->publish(diagnostics);
}

}  // namespace slam_3d_lidar_ingest
}  // namespace gole
