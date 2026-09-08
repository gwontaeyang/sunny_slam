#include "slam_2d_core/roughness_scan_conditioner.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace slam_2d_core
{
namespace
{

constexpr double kEpsilon = 1.0e-12;
constexpr double kNanosecondsPerSecond = 1.0e9;

bool finiteVector(const geometry_msgs::msg::Vector3 & value)
{
  return
    std::isfinite(value.x) &&
    std::isfinite(value.y) &&
    std::isfinite(value.z);
}

bool finiteQuaternion(const geometry_msgs::msg::Quaternion & value)
{
  return
    std::isfinite(value.x) &&
    std::isfinite(value.y) &&
    std::isfinite(value.z) &&
    std::isfinite(value.w);
}

bool covarianceUnavailable(const std::array<double, 9> & covariance)
{
  return covariance[0] == -1.0;
}

bool covarianceFinite(const std::array<double, 9> & covariance)
{
  return std::all_of(
    covariance.begin(), covariance.end(),
    [](const double value) {return std::isfinite(value);});
}

bool thresholdEnabled(const double threshold)
{
  return std::isfinite(threshold) && threshold > 0.0;
}

bool validScanGeometry(const sensor_msgs::msg::LaserScan & scan)
{
  return
    !scan.ranges.empty() &&
    std::isfinite(scan.angle_min) &&
    std::isfinite(scan.angle_increment) &&
    std::fabs(scan.angle_increment) > kEpsilon &&
    std::isfinite(scan.range_min) &&
    std::isfinite(scan.range_max) &&
    scan.range_max > scan.range_min;
}

bool validRange(const double range, const sensor_msgs::msg::LaserScan & scan)
{
  return
    std::isfinite(range) &&
    range >= static_cast<double>(scan.range_min) &&
    range <= static_cast<double>(scan.range_max);
}

double percentile(std::vector<double> values, const double fraction)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double clamped = std::clamp(fraction, 0.0, 1.0);
  const double position = clamped * static_cast<double>(values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(position));
  const auto upper = static_cast<std::size_t>(std::ceil(position));
  const double alpha = position - static_cast<double>(lower);
  return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

}  // namespace

ImuTimestampAction classifyImuTimestamp(
  const std::int64_t stamp_ns,
  const std::optional<std::int64_t> & last_stamp_ns,
  const double epoch_reset_threshold_s)
{
  if (
    stamp_ns <= 0 ||
    !std::isfinite(epoch_reset_threshold_s) ||
    epoch_reset_threshold_s < 0.0)
  {
    return ImuTimestampAction::kInvalid;
  }
  if (!last_stamp_ns) {
    return ImuTimestampAction::kFirst;
  }
  if (stamp_ns > *last_stamp_ns) {
    return ImuTimestampAction::kForward;
  }
  if (stamp_ns == *last_stamp_ns) {
    return ImuTimestampAction::kDuplicate;
  }
  const long double rollback_ns =
    static_cast<long double>(*last_stamp_ns) -
    static_cast<long double>(stamp_ns);
  const long double reset_ns =
    static_cast<long double>(epoch_reset_threshold_s) *
    static_cast<long double>(kNanosecondsPerSecond);
  return rollback_ns >= reset_ns ?
         ImuTimestampAction::kResetEpoch :
         ImuTimestampAction::kDropOutOfOrder;
}

bool isValidMessageTimestamp(
  const std::int32_t sec,
  const std::uint32_t nanosec)
{
  return
    sec >= 0 && nanosec < 1000000000U &&
    (sec > 0 || nanosec > 0U);
}

bool shouldDiscardOldestImuSample(
  const std::size_t sample_count,
  const double sample_span_s,
  const double buffer_duration_s)
{
  return
    sample_count > 1U &&
    std::isfinite(sample_span_s) &&
    std::isfinite(buffer_duration_s) &&
    buffer_duration_s >= 0.0 &&
    sample_span_s > buffer_duration_s;
}

ImuQualityRequirements imuQualityRequirements(
  const double tilt_threshold_deg,
  const double gyro_xy_threshold_rad_s,
  const double accel_norm_deviation_threshold_m_s2,
  const double tilt_rate_threshold_deg_s,
  const double accel_norm_rate_threshold_m_s3)
{
  ImuQualityRequirements requirements;
  requirements.orientation =
    thresholdEnabled(tilt_threshold_deg) ||
    thresholdEnabled(tilt_rate_threshold_deg_s);
  requirements.angular_velocity = thresholdEnabled(gyro_xy_threshold_rad_s);
  requirements.linear_acceleration =
    thresholdEnabled(accel_norm_deviation_threshold_m_s2) ||
    thresholdEnabled(accel_norm_rate_threshold_m_s3);
  return requirements;
}

ImuQualityResult validateImuQuality(
  const sensor_msgs::msg::Imu & imu,
  const ImuQualityRequirements & requirements)
{
  if (requirements.orientation) {
    if (covarianceUnavailable(imu.orientation_covariance)) {
      return {false, ImuQualityReason::kOrientationUnavailable};
    }
    if (!covarianceFinite(imu.orientation_covariance)) {
      return {false, ImuQualityReason::kNonfiniteCovariance};
    }
    if (!finiteQuaternion(imu.orientation)) {
      return {false, ImuQualityReason::kNonfiniteOrientation};
    }
    const double norm_squared =
      imu.orientation.x * imu.orientation.x +
      imu.orientation.y * imu.orientation.y +
      imu.orientation.z * imu.orientation.z +
      imu.orientation.w * imu.orientation.w;
    if (!std::isfinite(norm_squared) || norm_squared <= kEpsilon) {
      return {false, ImuQualityReason::kDegenerateOrientation};
    }
  }
  if (requirements.angular_velocity) {
    if (covarianceUnavailable(imu.angular_velocity_covariance)) {
      return {false, ImuQualityReason::kAngularVelocityUnavailable};
    }
    if (!covarianceFinite(imu.angular_velocity_covariance)) {
      return {false, ImuQualityReason::kNonfiniteCovariance};
    }
    if (!finiteVector(imu.angular_velocity)) {
      return {false, ImuQualityReason::kNonfiniteAngularVelocity};
    }
  }
  if (requirements.linear_acceleration) {
    if (covarianceUnavailable(imu.linear_acceleration_covariance)) {
      return {false, ImuQualityReason::kLinearAccelerationUnavailable};
    }
    if (!covarianceFinite(imu.linear_acceleration_covariance)) {
      return {false, ImuQualityReason::kNonfiniteCovariance};
    }
    if (!finiteVector(imu.linear_acceleration)) {
      return {false, ImuQualityReason::kNonfiniteLinearAcceleration};
    }
  }
  return {true, ImuQualityReason::kValid};
}

const char * imuQualityReasonName(const ImuQualityReason reason) noexcept
{
  switch (reason) {
    case ImuQualityReason::kValid:
      return "valid";
    case ImuQualityReason::kOrientationUnavailable:
      return "orientation_unavailable";
    case ImuQualityReason::kAngularVelocityUnavailable:
      return "angular_velocity_unavailable";
    case ImuQualityReason::kLinearAccelerationUnavailable:
      return "linear_acceleration_unavailable";
    case ImuQualityReason::kNonfiniteOrientation:
      return "nonfinite_orientation";
    case ImuQualityReason::kDegenerateOrientation:
      return "degenerate_orientation";
    case ImuQualityReason::kNonfiniteAngularVelocity:
      return "nonfinite_angular_velocity";
    case ImuQualityReason::kNonfiniteLinearAcceleration:
      return "nonfinite_linear_acceleration";
    case ImuQualityReason::kNonfiniteCovariance:
      return "nonfinite_covariance";
  }
  return "unknown";
}

void validateRoughnessRecoveryConfig(const RoughnessRecoveryConfig & config)
{
  if (config.recovery_scan_count == 0U) {
    throw std::invalid_argument("recovery_scan_count must be positive");
  }
  for (const auto & value : {
      std::pair<double, const char *>{config.max_scan_gap_s, "max_scan_gap_s"},
      {config.min_stable_duration_s, "min_stable_duration_s"},
      {config.fault_after_s, "fault_after_s"}
    })
  {
    if (!std::isfinite(value.first) || value.first < 0.0) {
      throw std::invalid_argument(std::string(value.second) + " must be finite and nonnegative");
    }
  }
}

RoughnessRecoveryGate::RoughnessRecoveryGate(RoughnessRecoveryConfig config)
{
  configure(config);
}

void RoughnessRecoveryGate::configure(const RoughnessRecoveryConfig & config)
{
  validateRoughnessRecoveryConfig(config);
  config_ = config;
  reset();
}

void RoughnessRecoveryGate::reset()
{
  state_ = RoughnessRecoveryState::kStable;
  last_sensor_stamp_s_ = 0.0;
  event_start_stamp_s_ = 0.0;
  recovery_start_stamp_s_ = 0.0;
  last_recovery_stamp_s_ = 0.0;
  last_fault_clock_s_ = 0.0;
  fault_duration_s_ = 0.0;
  recovery_scan_count_ = 0U;
  have_sensor_stamp_ = false;
  have_fault_clock_ = false;
  fault_active_ = false;
}

void RoughnessRecoveryGate::resetRecoverySequence()
{
  recovery_start_stamp_s_ = 0.0;
  last_recovery_stamp_s_ = 0.0;
  recovery_scan_count_ = 0U;
}

void RoughnessRecoveryGate::startEvent(
  const double sensor_stamp_s,
  const double fault_clock_s)
{
  const bool new_event = state_ != RoughnessRecoveryState::kQuarantine;
  state_ = RoughnessRecoveryState::kQuarantine;
  resetRecoverySequence();
  if (new_event) {
    event_start_stamp_s_ =
      std::isfinite(sensor_stamp_s) && sensor_stamp_s > 0.0 ?
      sensor_stamp_s : 0.0;
    fault_duration_s_ = 0.0;
    fault_active_ = false;
    last_fault_clock_s_ =
      std::isfinite(fault_clock_s) && fault_clock_s > 0.0 ?
      fault_clock_s : 0.0;
    have_fault_clock_ = last_fault_clock_s_ > 0.0;
  }
}

void RoughnessRecoveryGate::advanceFaultClock(const double fault_clock_s)
{
  if (
    state_ != RoughnessRecoveryState::kQuarantine ||
    !std::isfinite(fault_clock_s) || fault_clock_s <= 0.0)
  {
    return;
  }
  if (!have_fault_clock_) {
    last_fault_clock_s_ = fault_clock_s;
    have_fault_clock_ = true;
    return;
  }
  if (fault_clock_s > last_fault_clock_s_) {
    fault_duration_s_ += fault_clock_s - last_fault_clock_s_;
  }
  last_fault_clock_s_ = fault_clock_s;
}

void RoughnessRecoveryGate::updateFault()
{
  fault_active_ =
    state_ == RoughnessRecoveryState::kQuarantine &&
    fault_duration_s_ >= config_.fault_after_s;
}

RoughnessRecoveryDecision RoughnessRecoveryGate::decision(
  const RoughnessRecoveryReason reason) const
{
  RoughnessRecoveryDecision result;
  result.allow = reason == RoughnessRecoveryReason::kStable;
  result.quarantined = state_ == RoughnessRecoveryState::kQuarantine;
  result.fault_active = fault_active_;
  result.recovery_scan_count = recovery_scan_count_;
  result.stable_duration_s =
    recovery_scan_count_ > 0U &&
    last_recovery_stamp_s_ >= recovery_start_stamp_s_ ?
    last_recovery_stamp_s_ - recovery_start_stamp_s_ : 0.0;
  result.fault_duration_s = fault_duration_s_;
  result.reason = reason;
  return result;
}

RoughnessRecoveryDecision RoughnessRecoveryGate::update(
  const double sensor_stamp_s,
  const bool rough,
  const bool recovery_qualified,
  const double fault_clock_s)
{
  const bool valid_stamp = std::isfinite(sensor_stamp_s) && sensor_stamp_s > 0.0;
  if (!valid_stamp) {
    if (state_ == RoughnessRecoveryState::kStable) {
      startEvent(sensor_stamp_s, fault_clock_s);
    } else {
      resetRecoverySequence();
    }
    advanceFaultClock(fault_clock_s);
    updateFault();
    return decision(RoughnessRecoveryReason::kInvalidTimestamp);
  }

  if (have_sensor_stamp_ && sensor_stamp_s < last_sensor_stamp_s_) {
    if (state_ == RoughnessRecoveryState::kStable) {
      startEvent(sensor_stamp_s, fault_clock_s);
    } else {
      resetRecoverySequence();
    }
    last_sensor_stamp_s_ = sensor_stamp_s;
    advanceFaultClock(fault_clock_s);
    updateFault();
    return decision(RoughnessRecoveryReason::kTimestampRollback);
  }

  if (have_sensor_stamp_ && sensor_stamp_s == last_sensor_stamp_s_) {
    if (rough) {
      if (state_ == RoughnessRecoveryState::kStable) {
        startEvent(sensor_stamp_s, fault_clock_s);
      } else {
        resetRecoverySequence();
      }
    }
    advanceFaultClock(fault_clock_s);
    updateFault();
    return decision(RoughnessRecoveryReason::kDuplicateTimestamp);
  }

  last_sensor_stamp_s_ = sensor_stamp_s;
  have_sensor_stamp_ = true;
  if (rough) {
    if (state_ == RoughnessRecoveryState::kStable) {
      startEvent(sensor_stamp_s, fault_clock_s);
    } else {
      resetRecoverySequence();
    }
    advanceFaultClock(fault_clock_s);
    updateFault();
    return decision(RoughnessRecoveryReason::kRoughEvent);
  }

  if (!recovery_qualified) {
    if (state_ == RoughnessRecoveryState::kQuarantine) {
      resetRecoverySequence();
      advanceFaultClock(fault_clock_s);
      updateFault();
    }
    return decision(RoughnessRecoveryReason::kUnqualified);
  }

  if (state_ == RoughnessRecoveryState::kStable) {
    fault_duration_s_ = 0.0;
    fault_active_ = false;
    have_fault_clock_ = false;
    return decision(RoughnessRecoveryReason::kStable);
  }

  advanceFaultClock(fault_clock_s);
  updateFault();
  if (
    recovery_scan_count_ == 0U ||
    sensor_stamp_s - last_recovery_stamp_s_ > config_.max_scan_gap_s)
  {
    recovery_start_stamp_s_ = sensor_stamp_s;
    recovery_scan_count_ = 1U;
  } else {
    ++recovery_scan_count_;
  }
  last_recovery_stamp_s_ = sensor_stamp_s;
  const double stable_duration_s = sensor_stamp_s - recovery_start_stamp_s_;
  if (
    recovery_scan_count_ >= config_.recovery_scan_count &&
    stable_duration_s >= config_.min_stable_duration_s)
  {
    RoughnessRecoveryDecision result;
    result.allow = true;
    result.recovered = true;
    result.recovery_scan_count = recovery_scan_count_;
    result.stable_duration_s = stable_duration_s;
    result.fault_active = fault_active_;
    result.fault_duration_s = fault_duration_s_;
    result.reason = RoughnessRecoveryReason::kRecovered;
    state_ = RoughnessRecoveryState::kStable;
    resetRecoverySequence();
    fault_duration_s_ = 0.0;
    fault_active_ = false;
    have_fault_clock_ = false;
    return result;
  }
  return decision(RoughnessRecoveryReason::kRecovering);
}

RoughnessRecoveryState RoughnessRecoveryGate::state() const noexcept
{
  return state_;
}

double RoughnessRecoveryGate::eventDuration() const noexcept
{
  if (
    state_ != RoughnessRecoveryState::kQuarantine ||
    event_start_stamp_s_ <= 0.0 ||
    !have_sensor_stamp_ ||
    last_sensor_stamp_s_ < event_start_stamp_s_)
  {
    return 0.0;
  }
  return last_sensor_stamp_s_ - event_start_stamp_s_;
}

ImuUnavailableFailOpenGate::ImuUnavailableFailOpenGate(
  const double fault_after_s,
  const double maximum_clock_step_s)
{
  configure(fault_after_s, maximum_clock_step_s);
}

void ImuUnavailableFailOpenGate::configure(
  const double fault_after_s,
  const double maximum_clock_step_s)
{
  if (
    !std::isfinite(fault_after_s) || fault_after_s < 0.0 ||
    !std::isfinite(maximum_clock_step_s) || maximum_clock_step_s <= 0.0)
  {
    throw std::invalid_argument("invalid IMU unavailable fail-open timing");
  }
  fault_after_s_ = fault_after_s;
  maximum_clock_step_s_ = maximum_clock_step_s;
  reset();
}

void ImuUnavailableFailOpenGate::reset()
{
  last_clock_s_ = 0.0;
  unavailable_duration_s_ = 0.0;
  have_clock_ = false;
}

bool ImuUnavailableFailOpenGate::update(
  const double clock_s,
  const bool imu_available)
{
  if (imu_available) {
    reset();
    return false;
  }
  if (!std::isfinite(clock_s) || clock_s <= 0.0) {
    return false;
  }
  if (!have_clock_) {
    last_clock_s_ = clock_s;
    have_clock_ = true;
    return fault_after_s_ == 0.0;
  }
  const double delta_s = clock_s - last_clock_s_;
  if (delta_s < 0.0) {
    unavailable_duration_s_ = 0.0;
  } else if (delta_s > 0.0) {
    unavailable_duration_s_ += std::min(delta_s, maximum_clock_step_s_);
  }
  last_clock_s_ = clock_s;
  return unavailable_duration_s_ >= fault_after_s_;
}

double ImuUnavailableFailOpenGate::unavailableDurationS() const noexcept
{
  return unavailable_duration_s_;
}

bool shouldFailOpenForUnavailableImu(
  const bool conditioning_enabled,
  const bool quarantine_mode,
  const bool imu_available,
  const bool fail_open_enabled,
  const bool unavailable_fault_active,
  const bool scan_timestamp_valid,
  const bool scan_quality_valid)
{
  return
    conditioning_enabled && quarantine_mode && !imu_available &&
    fail_open_enabled && unavailable_fault_active && scan_timestamp_valid &&
    scan_quality_valid;
}

bool shouldFailClosedForInvalidScanTimestamp(
  const bool conditioning_enabled,
  const bool quarantine_mode,
  const bool scan_timestamp_valid)
{
  return conditioning_enabled && quarantine_mode && !scan_timestamp_valid;
}

LaserScanQuality laserScanQuality(const sensor_msgs::msg::LaserScan & scan)
{
  LaserScanQuality result;
  result.total_beam_count = scan.ranges.size();
  for (const float range : scan.ranges) {
    if (validRange(range, scan)) {
      ++result.valid_beam_count;
    }
  }
  if (result.total_beam_count > 0U) {
    result.valid_ratio =
      static_cast<double>(result.valid_beam_count) /
      static_cast<double>(result.total_beam_count);
  }
  return result;
}

RelativeMotion relativeMotion(
  const geometry_msgs::msg::TransformStamped & transform)
{
  RelativeMotion motion;
  const auto & translation = transform.transform.translation;
  const auto & q_in = transform.transform.rotation;
  if (
    !std::isfinite(translation.x) ||
    !std::isfinite(translation.y) ||
    !std::isfinite(translation.z) ||
    !finiteQuaternion(q_in))
  {
    return motion;
  }
  const double norm = std::sqrt(
    q_in.x * q_in.x + q_in.y * q_in.y +
    q_in.z * q_in.z + q_in.w * q_in.w);
  if (!std::isfinite(norm) || norm <= kEpsilon) {
    return motion;
  }
  const double x = q_in.x / norm;
  const double y = q_in.y / norm;
  const double z = q_in.z / norm;
  const double w = q_in.w / norm;
  const double roll = std::atan2(
    2.0 * (w * x + y * z),
    1.0 - 2.0 * (x * x + y * y));
  const double pitch = std::asin(
    std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  const double yaw = std::atan2(
    2.0 * (w * z + x * y),
    1.0 - 2.0 * (y * y + z * z));
  motion.translation_m = std::hypot(translation.x, translation.y);
  motion.yaw_rad = yaw;
  motion.roll_pitch_rad = std::hypot(roll, pitch);
  motion.finite =
    std::isfinite(motion.translation_m) &&
    std::isfinite(motion.yaw_rad) &&
    std::isfinite(motion.roll_pitch_rad);
  return motion;
}

std::optional<StabilizedScan> warpLaserScan(
  const sensor_msgs::msg::LaserScan & source,
  const sensor_msgs::msg::LaserScan & current_template,
  const geometry_msgs::msg::TransformStamped & current_from_source,
  const std::size_t min_valid_beams)
{
  if (!validScanGeometry(source) || !validScanGeometry(current_template)) {
    return std::nullopt;
  }
  const RelativeMotion motion = relativeMotion(current_from_source);
  if (!motion.finite) {
    return std::nullopt;
  }

  StabilizedScan result;
  result.scan = current_template;
  result.scan.ranges.assign(
    current_template.ranges.size(),
    std::numeric_limits<float>::infinity());
  result.scan.intensities.clear();
  result.scan.time_increment = 0.0F;
  result.scan.scan_time = 0.0F;

  const double cos_yaw = std::cos(motion.yaw_rad);
  const double sin_yaw = std::sin(motion.yaw_rad);
  const auto & translation = current_from_source.transform.translation;
  std::vector<bool> filled(result.scan.ranges.size(), false);
  for (std::size_t index = 0; index < source.ranges.size(); ++index) {
    const double range = source.ranges[index];
    if (!validRange(range, source)) {
      continue;
    }
    const double angle =
      source.angle_min + static_cast<double>(index) * source.angle_increment;
    const double source_x = range * std::cos(angle);
    const double source_y = range * std::sin(angle);
    const double current_x =
      cos_yaw * source_x - sin_yaw * source_y + translation.x;
    const double current_y =
      sin_yaw * source_x + cos_yaw * source_y + translation.y;
    const double current_range = std::hypot(current_x, current_y);
    if (!validRange(current_range, current_template)) {
      continue;
    }
    const double current_angle = std::atan2(current_y, current_x);
    const double floating_bin =
      (current_angle - current_template.angle_min) /
      current_template.angle_increment;
    if (!std::isfinite(floating_bin)) {
      continue;
    }
    const auto output_index = static_cast<long long>(std::llround(floating_bin));
    if (
      output_index < 0 ||
      output_index >= static_cast<long long>(result.scan.ranges.size()))
    {
      continue;
    }
    const std::size_t bin = static_cast<std::size_t>(output_index);
    const double bin_angle =
      current_template.angle_min +
      static_cast<double>(bin) * current_template.angle_increment;
    if (
      std::fabs(current_angle - bin_angle) >
      0.5 * std::fabs(current_template.angle_increment) + 1.0e-9)
    {
      continue;
    }
    result.scan.ranges[bin] = std::min(
      result.scan.ranges[bin], static_cast<float>(current_range));
    if (!filled[bin]) {
      filled[bin] = true;
      ++result.valid_beam_count;
    }
  }
  if (result.valid_beam_count < min_valid_beams) {
    return std::nullopt;
  }
  return result;
}

std::optional<ResidualFilterDirection> parseResidualFilterDirection(
  const std::string & value)
{
  std::string normalized = value;
  std::transform(
    normalized.begin(), normalized.end(), normalized.begin(),
    [](const unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
  if (normalized == "both") {
    return ResidualFilterDirection::kBoth;
  }
  if (normalized == "closer") {
    return ResidualFilterDirection::kCloser;
  }
  if (normalized == "farther") {
    return ResidualFilterDirection::kFarther;
  }
  return std::nullopt;
}

std::optional<ResidualFilteredScan> filterLaserScanByResidual(
  const sensor_msgs::msg::LaserScan & stable_reference,
  const sensor_msgs::msg::LaserScan & current_scan,
  const geometry_msgs::msg::TransformStamped & current_from_reference,
  const double residual_threshold_m,
  const std::size_t neighbor_bins,
  const std::size_t min_overlap_beams,
  const double min_overlap_ratio,
  const std::size_t min_output_valid_beams,
  const ResidualFilterDirection direction,
  const double max_removed_ratio)
{
  if (
    !std::isfinite(residual_threshold_m) || residual_threshold_m < 0.0 ||
    !std::isfinite(min_overlap_ratio) ||
    min_overlap_ratio < 0.0 || min_overlap_ratio > 1.0 ||
    !std::isfinite(max_removed_ratio) ||
    max_removed_ratio < 0.0 || max_removed_ratio > 1.0)
  {
    return std::nullopt;
  }
  const auto warped = warpLaserScan(
    stable_reference, current_scan, current_from_reference, 1U);
  if (!warped) {
    return std::nullopt;
  }

  ResidualFilteredScan result;
  result.scan = current_scan;
  std::vector<double> overlap_residuals;
  std::vector<std::pair<double, std::size_t>> candidates;
  for (std::size_t index = 0; index < current_scan.ranges.size(); ++index) {
    const double current_range = current_scan.ranges[index];
    if (!validRange(current_range, current_scan)) {
      continue;
    }
    ++result.input_valid_beam_count;
    const std::size_t first = index > neighbor_bins ? index - neighbor_bins : 0U;
    const std::size_t last = std::min(
      index + neighbor_bins,
      warped->scan.ranges.empty() ? 0U : warped->scan.ranges.size() - 1U);
    bool have_reference = false;
    double selected_residual = 0.0;
    double selected_abs_residual = std::numeric_limits<double>::infinity();
    for (std::size_t reference_index = first;
      reference_index <= last && reference_index < warped->scan.ranges.size();
      ++reference_index)
    {
      const double reference_range = warped->scan.ranges[reference_index];
      if (!validRange(reference_range, warped->scan)) {
        continue;
      }
      const double residual = current_range - reference_range;
      const double abs_residual = std::fabs(residual);
      if (abs_residual < selected_abs_residual) {
        selected_residual = residual;
        selected_abs_residual = abs_residual;
        have_reference = true;
      }
    }
    if (!have_reference) {
      continue;
    }
    ++result.overlap_beam_count;
    overlap_residuals.push_back(selected_abs_residual);
    const bool direction_matches =
      direction == ResidualFilterDirection::kBoth ||
      (direction == ResidualFilterDirection::kCloser && selected_residual < 0.0) ||
      (direction == ResidualFilterDirection::kFarther && selected_residual > 0.0);
    if (direction_matches && selected_abs_residual > residual_threshold_m) {
      candidates.emplace_back(selected_abs_residual, index);
    }
  }

  const double overlap_ratio = result.input_valid_beam_count > 0U ?
    static_cast<double>(result.overlap_beam_count) /
    static_cast<double>(result.input_valid_beam_count) : 0.0;
  if (
    result.overlap_beam_count < min_overlap_beams ||
    overlap_ratio < min_overlap_ratio)
  {
    return std::nullopt;
  }

  result.candidate_beam_count = candidates.size();
  std::sort(
    candidates.begin(), candidates.end(),
    [](const auto & lhs, const auto & rhs) {
      if (lhs.first != rhs.first) {
        return lhs.first > rhs.first;
      }
      return lhs.second < rhs.second;
    });
  const std::size_t removal_budget = static_cast<std::size_t>(std::floor(
      max_removed_ratio * static_cast<double>(result.input_valid_beam_count) +
      kEpsilon));
  result.removed_beam_count = std::min(removal_budget, candidates.size());
  for (std::size_t index = 0; index < result.removed_beam_count; ++index) {
    result.scan.ranges[candidates[index].second] =
      std::numeric_limits<float>::infinity();
    if (result.scan.intensities.size() == result.scan.ranges.size()) {
      result.scan.intensities[candidates[index].second] = 0.0F;
    }
  }
  result.output_valid_beam_count =
    result.input_valid_beam_count - result.removed_beam_count;
  if (result.output_valid_beam_count < min_output_valid_beams) {
    return std::nullopt;
  }
  result.residual_p50_m = percentile(overlap_residuals, 0.50);
  result.residual_p95_m = percentile(overlap_residuals, 0.95);
  return result;
}

}  // namespace slam_2d_core
