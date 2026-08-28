#ifndef SLAM_2D_CORE__ROUGHNESS_SCAN_CONDITIONER_HPP_
#define SLAM_2D_CORE__ROUGHNESS_SCAN_CONDITIONER_HPP_

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace slam_2d_core
{

enum class ImuTimestampAction
{
  kFirst,
  kForward,
  kDuplicate,
  kDropOutOfOrder,
  kResetEpoch,
  kInvalid,
};

ImuTimestampAction classifyImuTimestamp(
  std::int64_t stamp_ns,
  const std::optional<std::int64_t> & last_stamp_ns,
  double epoch_reset_threshold_s);

bool isValidMessageTimestamp(std::int32_t sec, std::uint32_t nanosec);

bool shouldDiscardOldestImuSample(
  std::size_t sample_count,
  double sample_span_s,
  double buffer_duration_s);

struct ImuQualityRequirements
{
  bool orientation{false};
  bool angular_velocity{false};
  bool linear_acceleration{false};
};

enum class ImuQualityReason
{
  kValid,
  kOrientationUnavailable,
  kAngularVelocityUnavailable,
  kLinearAccelerationUnavailable,
  kNonfiniteOrientation,
  kDegenerateOrientation,
  kNonfiniteAngularVelocity,
  kNonfiniteLinearAcceleration,
  kNonfiniteCovariance,
};

struct ImuQualityResult
{
  bool valid{false};
  ImuQualityReason reason{ImuQualityReason::kValid};
};

ImuQualityRequirements imuQualityRequirements(
  double tilt_threshold_deg,
  double gyro_xy_threshold_rad_s,
  double accel_norm_deviation_threshold_m_s2,
  double tilt_rate_threshold_deg_s,
  double accel_norm_rate_threshold_m_s3);

ImuQualityResult validateImuQuality(
  const sensor_msgs::msg::Imu & imu,
  const ImuQualityRequirements & requirements);

const char * imuQualityReasonName(ImuQualityReason reason) noexcept;

struct RoughnessRecoveryConfig
{
  std::size_t recovery_scan_count{5};
  double max_scan_gap_s{0.75};
  double min_stable_duration_s{0.16};
  double fault_after_s{3.0};
};

void validateRoughnessRecoveryConfig(const RoughnessRecoveryConfig & config);

enum class RoughnessRecoveryState
{
  kStable,
  kQuarantine,
};

enum class RoughnessRecoveryReason
{
  kStable,
  kRoughEvent,
  kRecovering,
  kRecovered,
  kUnqualified,
  kDuplicateTimestamp,
  kTimestampRollback,
  kInvalidTimestamp,
};

struct RoughnessRecoveryDecision
{
  bool allow{false};
  bool quarantined{false};
  bool recovered{false};
  bool fault_active{false};
  std::size_t recovery_scan_count{0};
  double stable_duration_s{0.0};
  double fault_duration_s{0.0};
  RoughnessRecoveryReason reason{RoughnessRecoveryReason::kStable};
};

class RoughnessRecoveryGate
{
public:
  explicit RoughnessRecoveryGate(
    RoughnessRecoveryConfig config = RoughnessRecoveryConfig{});

  void configure(const RoughnessRecoveryConfig & config);
  void reset();
  RoughnessRecoveryDecision update(
    double sensor_stamp_s,
    bool rough,
    bool recovery_qualified,
    double fault_clock_s);
  RoughnessRecoveryState state() const noexcept;
  double eventDuration() const noexcept;

private:
  void resetRecoverySequence();
  void startEvent(double sensor_stamp_s, double fault_clock_s);
  void advanceFaultClock(double fault_clock_s);
  void updateFault();
  RoughnessRecoveryDecision decision(RoughnessRecoveryReason reason) const;

  RoughnessRecoveryConfig config_;
  RoughnessRecoveryState state_{RoughnessRecoveryState::kStable};
  double last_sensor_stamp_s_{0.0};
  double event_start_stamp_s_{0.0};
  double recovery_start_stamp_s_{0.0};
  double last_recovery_stamp_s_{0.0};
  double last_fault_clock_s_{0.0};
  double fault_duration_s_{0.0};
  std::size_t recovery_scan_count_{0};
  bool have_sensor_stamp_{false};
  bool have_fault_clock_{false};
  bool fault_active_{false};
};

class ImuUnavailableFailOpenGate
{
public:
  ImuUnavailableFailOpenGate(
    double fault_after_s = 3.0,
    double maximum_clock_step_s = 0.2);

  void configure(double fault_after_s, double maximum_clock_step_s);
  void reset();
  bool update(double clock_s, bool imu_available);
  double unavailableDurationS() const noexcept;

private:
  double fault_after_s_{3.0};
  double maximum_clock_step_s_{0.2};
  double last_clock_s_{0.0};
  double unavailable_duration_s_{0.0};
  bool have_clock_{false};
};

bool shouldFailOpenForUnavailableImu(
  bool conditioning_enabled,
  bool quarantine_mode,
  bool imu_available,
  bool fail_open_enabled,
  bool unavailable_fault_active,
  bool scan_timestamp_valid,
  bool scan_quality_valid);

bool shouldFailClosedForInvalidScanTimestamp(
  bool conditioning_enabled,
  bool quarantine_mode,
  bool scan_timestamp_valid);

struct LaserScanQuality
{
  std::size_t total_beam_count{0};
  std::size_t valid_beam_count{0};
  double valid_ratio{0.0};
};

LaserScanQuality laserScanQuality(const sensor_msgs::msg::LaserScan & scan);

struct RelativeMotion
{
  double translation_m{0.0};
  double yaw_rad{0.0};
  double roll_pitch_rad{0.0};
  bool finite{false};
};

struct StabilizedScan
{
  sensor_msgs::msg::LaserScan scan;
  std::size_t valid_beam_count{0};
};

RelativeMotion relativeMotion(
  const geometry_msgs::msg::TransformStamped & transform);

std::optional<StabilizedScan> warpLaserScan(
  const sensor_msgs::msg::LaserScan & source,
  const sensor_msgs::msg::LaserScan & current_template,
  const geometry_msgs::msg::TransformStamped & current_from_source,
  std::size_t min_valid_beams);

enum class ResidualFilterDirection
{
  kBoth,
  kCloser,
  kFarther,
};

std::optional<ResidualFilterDirection> parseResidualFilterDirection(
  const std::string & value);

struct ResidualFilteredScan
{
  sensor_msgs::msg::LaserScan scan;
  std::size_t input_valid_beam_count{0};
  std::size_t overlap_beam_count{0};
  std::size_t candidate_beam_count{0};
  std::size_t removed_beam_count{0};
  std::size_t output_valid_beam_count{0};
  double residual_p50_m{0.0};
  double residual_p95_m{0.0};
};

std::optional<ResidualFilteredScan> filterLaserScanByResidual(
  const sensor_msgs::msg::LaserScan & stable_reference,
  const sensor_msgs::msg::LaserScan & current_scan,
  const geometry_msgs::msg::TransformStamped & current_from_reference,
  double residual_threshold_m,
  std::size_t neighbor_bins,
  std::size_t min_overlap_beams,
  double min_overlap_ratio,
  std::size_t min_output_valid_beams,
  ResidualFilterDirection direction,
  double max_removed_ratio);

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_SCAN_CONDITIONER_HPP_
