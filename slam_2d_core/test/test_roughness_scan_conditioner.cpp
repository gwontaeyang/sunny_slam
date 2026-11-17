#include "slam_2d_core/roughness_scan_conditioner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace
{

constexpr double kPi = 3.14159265358979323846;

sensor_msgs::msg::Imu makeImu()
{
  sensor_msgs::msg::Imu imu;
  imu.orientation.w = 1.0;
  imu.linear_acceleration.z = 9.80665;
  return imu;
}

sensor_msgs::msg::LaserScan makeScan(const std::size_t beam_count = 5U)
{
  sensor_msgs::msg::LaserScan scan;
  scan.header.frame_id = "laser";
  scan.angle_min = -static_cast<float>(kPi / 2.0);
  scan.angle_increment = beam_count > 1U ?
    static_cast<float>(kPi / static_cast<double>(beam_count - 1U)) : 0.1F;
  scan.angle_max = scan.angle_min +
    static_cast<float>(beam_count - 1U) * scan.angle_increment;
  scan.range_min = 0.05F;
  scan.range_max = 10.0F;
  scan.ranges.assign(beam_count, std::numeric_limits<float>::infinity());
  return scan;
}

geometry_msgs::msg::TransformStamped makeTransform(
  const double x,
  const double y,
  const double yaw)
{
  geometry_msgs::msg::TransformStamped transform;
  transform.transform.translation.x = x;
  transform.transform.translation.y = y;
  transform.transform.rotation.z = std::sin(yaw / 2.0);
  transform.transform.rotation.w = std::cos(yaw / 2.0);
  return transform;
}

slam_2d_core::RoughnessRecoveryConfig recoveryConfig()
{
  slam_2d_core::RoughnessRecoveryConfig config;
  config.recovery_scan_count = 3U;
  config.max_scan_gap_s = 0.2;
  config.min_stable_duration_s = 0.16;
  config.fault_after_s = 1.0;
  return config;
}

}  // namespace

TEST(ImuQualityPolicy, RequirementsFollowEnabledRoughnessChannels)
{
  const auto requirements = slam_2d_core::imuQualityRequirements(
    2.0, 0.05, 0.35, 5.0, 80.0);
  EXPECT_TRUE(requirements.orientation);
  EXPECT_TRUE(requirements.angular_velocity);
  EXPECT_TRUE(requirements.linear_acceleration);
}

TEST(ImuQualityPolicy, DisabledChannelDoesNotInvalidateSample)
{
  auto imu = makeImu();
  imu.orientation_covariance[0] = -1.0;
  const auto requirements = slam_2d_core::imuQualityRequirements(
    0.0, 0.05, 0.35, 0.0, 80.0);
  EXPECT_TRUE(slam_2d_core::validateImuQuality(imu, requirements).valid);
}

TEST(ImuQualityPolicy, RejectsUnavailableRequiredMeasurement)
{
  auto imu = makeImu();
  imu.angular_velocity_covariance[0] = -1.0;
  const auto result = slam_2d_core::validateImuQuality(
    imu, slam_2d_core::imuQualityRequirements(2.0, 0.05, 0.35, 5.0, 80.0));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(
    slam_2d_core::ImuQualityReason::kAngularVelocityUnavailable,
    result.reason);
}

TEST(ImuQualityPolicy, RejectsNonfiniteRequiredValue)
{
  auto imu = makeImu();
  imu.linear_acceleration.x = std::numeric_limits<double>::quiet_NaN();
  const auto result = slam_2d_core::validateImuQuality(
    imu, slam_2d_core::imuQualityRequirements(2.0, 0.05, 0.35, 5.0, 80.0));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(
    slam_2d_core::ImuQualityReason::kNonfiniteLinearAcceleration,
    result.reason);
}

TEST(ImuQualityPolicy, RejectsDegenerateRequiredQuaternion)
{
  auto imu = makeImu();
  imu.orientation.w = 0.0;
  const auto result = slam_2d_core::validateImuQuality(
    imu, slam_2d_core::imuQualityRequirements(2.0, 0.05, 0.35, 5.0, 80.0));
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(slam_2d_core::ImuQualityReason::kDegenerateOrientation, result.reason);
}

TEST(ImuQualityPolicy, ZeroCovarianceIsAcceptedAsUnknownCovariance)
{
  const auto result = slam_2d_core::validateImuQuality(
    makeImu(),
    slam_2d_core::imuQualityRequirements(2.0, 0.05, 0.35, 5.0, 80.0));
  EXPECT_TRUE(result.valid);
}

TEST(ImuTimestampPolicy, AcceptsFirstAndForwardSamples)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kFirst,
    slam_2d_core::classifyImuTimestamp(100, std::nullopt, 1.0));
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kForward,
    slam_2d_core::classifyImuTimestamp(101, 100, 1.0));
}

TEST(ImuTimestampPolicy, DuplicateReplacesSameStampPayloadWithoutEpochReset)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kDuplicate,
    slam_2d_core::classifyImuTimestamp(100, 100, 1.0));
}

TEST(ImuTimestampPolicy, SmallRollbackIsDroppedAsOutOfOrder)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kDropOutOfOrder,
    slam_2d_core::classifyImuTimestamp(900000000, 1000000000, 1.0));
}

TEST(ImuTimestampPolicy, LargeRollbackStartsNewEpoch)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kResetEpoch,
    slam_2d_core::classifyImuTimestamp(1, 2000000000, 1.0));
}

TEST(ImuTimestampPolicy, RollbackAtThresholdStartsNewEpoch)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kResetEpoch,
    slam_2d_core::classifyImuTimestamp(1000000000, 2000000000, 1.0));
}

TEST(ImuTimestampPolicy, RejectsInvalidStampAndConfiguration)
{
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kInvalid,
    slam_2d_core::classifyImuTimestamp(0, std::nullopt, 1.0));
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kInvalid,
    slam_2d_core::classifyImuTimestamp(1, std::nullopt, -1.0));
  EXPECT_FALSE(slam_2d_core::isValidMessageTimestamp(-1, 0));
  EXPECT_FALSE(slam_2d_core::isValidMessageTimestamp(1, 1000000000U));
}

TEST(ImuTimestampPolicy, PreservesNanosecondOrderingAtEpochScale)
{
  const std::int64_t last = 1780000000000000000LL;
  EXPECT_EQ(
    slam_2d_core::ImuTimestampAction::kForward,
    slam_2d_core::classifyImuTimestamp(last + 1, last, 1.0));
}

TEST(ImuTimestampPolicy, RetainsOneDistinctPredecessorForDuplicateReplacement)
{
  EXPECT_FALSE(slam_2d_core::shouldDiscardOldestImuSample(1U, 2.0, 1.0));
  EXPECT_TRUE(slam_2d_core::shouldDiscardOldestImuSample(2U, 2.0, 1.0));
}

TEST(ImuTimestampPolicy, RejectsTimestampsThatRclcppTimeCannotSafelyConstruct)
{
  EXPECT_FALSE(slam_2d_core::isValidMessageTimestamp(0, 0));
  EXPECT_TRUE(slam_2d_core::isValidMessageTimestamp(0, 1));
}

TEST(RoughnessRecoveryGate, StableScansPassWithoutAnEvent)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  const auto result = gate.update(1.0, false, true, 1.0);
  EXPECT_TRUE(result.allow);
  EXPECT_FALSE(result.quarantined);
}

TEST(RoughnessRecoveryGate, DefaultGapMatchesPoseGraphCandidateCadence)
{
  const slam_2d_core::RoughnessRecoveryConfig config;
  EXPECT_DOUBLE_EQ(0.75, config.max_scan_gap_s);
}

TEST(RoughnessRecoveryGate, RequiresConsecutiveQualifiedScansAfterRoughness)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  EXPECT_FALSE(gate.update(1.0, true, true, 1.0).allow);
  EXPECT_FALSE(gate.update(1.1, false, true, 1.1).allow);
  EXPECT_FALSE(gate.update(1.2, false, true, 1.2).allow);
  const auto recovered = gate.update(1.3, false, true, 1.3);
  EXPECT_TRUE(recovered.allow);
  EXPECT_TRUE(recovered.recovered);
}

TEST(RoughnessRecoveryGate, UnqualifiedScanRestartsRecoveryCount)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  const auto unqualified = gate.update(1.2, false, false, 1.2);
  EXPECT_EQ(0U, unqualified.recovery_scan_count);
  EXPECT_FALSE(gate.update(1.3, false, true, 1.3).allow);
}

TEST(RoughnessRecoveryGate, LongScanGapRestartsRecoveryCount)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  const auto after_gap = gate.update(1.5, false, true, 1.5);
  EXPECT_EQ(1U, after_gap.recovery_scan_count);
}

TEST(RoughnessRecoveryGate, NewRoughSampleRestartsRecovery)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  const auto rough = gate.update(1.2, true, true, 1.2);
  EXPECT_EQ(0U, rough.recovery_scan_count);
  EXPECT_TRUE(rough.quarantined);
}

TEST(RoughnessRecoveryGate, ClockResetStartsFromCurrentEvent)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(10.0, false, true, 10.0);
  const auto rollback = gate.update(1.0, false, true, 10.1);
  EXPECT_FALSE(rollback.allow);
  EXPECT_TRUE(rollback.quarantined);
  EXPECT_EQ(slam_2d_core::RoughnessRecoveryReason::kTimestampRollback, rollback.reason);
}

TEST(RoughnessRecoveryGate, InvalidTimestampCannotCompleteRecovery)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  EXPECT_FALSE(gate.update(std::numeric_limits<double>::quiet_NaN(), false, true, 1.2).allow);
}

TEST(RoughnessRecoveryGate, InvalidTimestampStartsFailClosedRecovery)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  const auto result = gate.update(0.0, false, true, 1.0);
  EXPECT_FALSE(result.allow);
  EXPECT_TRUE(result.quarantined);
}

TEST(RoughnessRecoveryGate, ClockRollbackDuringRecoveryStaysFailClosed)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(2.0, true, true, 2.0);
  gate.update(2.1, false, true, 2.1);
  const auto result = gate.update(1.0, false, true, 2.2);
  EXPECT_FALSE(result.allow);
  EXPECT_TRUE(result.quarantined);
}

TEST(RoughnessRecoveryGate, DuplicateTimestampDoesNotAdvanceRecovery)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  const auto first = gate.update(1.1, false, true, 1.1);
  const auto duplicate = gate.update(1.1, false, true, 1.2);
  EXPECT_EQ(first.recovery_scan_count, duplicate.recovery_scan_count);
  EXPECT_FALSE(duplicate.allow);
}

TEST(RoughnessRecoveryGate, DuplicateTimestampDropsOnlyDuplicateFromStable)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  EXPECT_TRUE(gate.update(1.0, false, true, 1.0).allow);
  EXPECT_FALSE(gate.update(1.0, false, true, 1.1).allow);
  EXPECT_EQ(slam_2d_core::RoughnessRecoveryState::kStable, gate.state());
  EXPECT_TRUE(gate.update(1.1, false, true, 1.2).allow);
}

TEST(RoughnessRecoveryGate, RoughDuplicateStartsQuarantineFromStable)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, false, true, 1.0);
  const auto result = gate.update(1.0, true, true, 1.1);
  EXPECT_FALSE(result.allow);
  EXPECT_TRUE(result.quarantined);
}

TEST(RoughnessRecoveryGate, EventTimestampCannotCountAsFirstRecoveryScan)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  const auto event = gate.update(1.0, true, true, 1.0);
  EXPECT_EQ(0U, event.recovery_scan_count);
}

TEST(RoughnessRecoveryGate, RequiresMinimumStableDuration)
{
  auto config = recoveryConfig();
  config.recovery_scan_count = 2U;
  config.min_stable_duration_s = 0.3;
  slam_2d_core::RoughnessRecoveryGate gate(config);
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  EXPECT_FALSE(gate.update(1.2, false, true, 1.2).allow);
}

TEST(RoughnessRecoveryGate, UnqualifiedImuStateCannotCompleteRecovery)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  gate.update(1.2, false, true, 1.2);
  EXPECT_FALSE(gate.update(1.3, false, false, 1.3).allow);
}

TEST(RoughnessRecoveryGate, StableGateDropsUnqualifiedScanWithoutStartingEvent)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  const auto result = gate.update(1.0, false, false, 1.0);
  EXPECT_FALSE(result.allow);
  EXPECT_FALSE(result.quarantined);
}

TEST(RoughnessRecoveryGate, LongQuarantineRaisesFaultAndRemainsClosed)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 10.0);
  gate.update(1.1, false, false, 10.5);
  const auto result = gate.update(1.2, false, false, 11.1);
  EXPECT_TRUE(result.fault_active);
  EXPECT_FALSE(result.allow);
}

TEST(RoughnessRecoveryGate, FaultClockAdvancesWhenSensorTimestampIsInvalid)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 10.0);
  gate.update(0.0, false, false, 10.5);
  EXPECT_TRUE(gate.update(0.0, false, false, 11.1).fault_active);
}

TEST(RoughnessRecoveryGate, FaultRemainsActiveAcrossSensorClockRollback)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(2.0, true, true, 10.0);
  gate.update(2.1, false, false, 10.6);
  EXPECT_TRUE(gate.update(1.0, false, false, 11.1).fault_active);
}

TEST(RoughnessRecoveryGate, RecoveredGateDropsOnlyDuplicateTimestamp)
{
  slam_2d_core::RoughnessRecoveryGate gate(recoveryConfig());
  gate.update(1.0, true, true, 1.0);
  gate.update(1.1, false, true, 1.1);
  gate.update(1.2, false, true, 1.2);
  EXPECT_TRUE(gate.update(1.3, false, true, 1.3).allow);
  EXPECT_FALSE(gate.update(1.3, false, true, 1.4).allow);
  EXPECT_TRUE(gate.update(1.4, false, true, 1.5).allow);
}

TEST(RoughnessRecoveryGate, RejectsInvalidSafetyDurations)
{
  auto config = recoveryConfig();
  config.max_scan_gap_s = -1.0;
  EXPECT_THROW(slam_2d_core::RoughnessRecoveryGate gate(config), std::invalid_argument);
}

TEST(RoughnessQuarantinePolicy, OpensOnlyAfterUnavailableImuFault)
{
  slam_2d_core::ImuUnavailableFailOpenGate gate(0.3, 0.2);
  EXPECT_FALSE(gate.update(1.0, false));
  EXPECT_FALSE(gate.update(1.1, false));
  EXPECT_FALSE(gate.update(1.2, false));
  const bool fault = gate.update(1.31, false);
  EXPECT_TRUE(fault);
  EXPECT_TRUE(
    slam_2d_core::shouldFailOpenForUnavailableImu(
      true, true, false, true, fault, true, true));
}

TEST(RoughnessQuarantinePolicy, InvalidScanOrEpochResetRemainsClosed)
{
  EXPECT_TRUE(slam_2d_core::shouldFailClosedForInvalidScanTimestamp(true, true, false));
  EXPECT_FALSE(
    slam_2d_core::shouldFailOpenForUnavailableImu(
      true, true, false, true, true, false, true));
}

TEST(RoughnessQuarantinePolicy, LowQualityScanAndDisabledPolicyRemainClosed)
{
  EXPECT_FALSE(
    slam_2d_core::shouldFailOpenForUnavailableImu(
      false, true, false, true, true, true, true));
  EXPECT_FALSE(
    slam_2d_core::shouldFailOpenForUnavailableImu(
      true, false, false, true, true, true, true));
  EXPECT_FALSE(
    slam_2d_core::shouldFailOpenForUnavailableImu(
      true, true, false, true, true, true, false));
}

TEST(RoughnessQuarantinePolicy, ImuUnavailableTimerDoesNotInheritRoughFaultTime)
{
  slam_2d_core::ImuUnavailableFailOpenGate gate(0.3, 0.2);
  gate.update(1.0, false);
  gate.update(1.1, false);
  EXPECT_FALSE(gate.update(1.2, true));
  EXPECT_FALSE(gate.update(1.3, false));
}

TEST(RoughnessQuarantinePolicy, AvailableImuAndClockRollbackResetUnavailableTimer)
{
  slam_2d_core::ImuUnavailableFailOpenGate gate(0.3, 0.2);
  gate.update(1.0, false);
  gate.update(1.1, false);
  EXPECT_FALSE(gate.update(0.5, false));
  EXPECT_DOUBLE_EQ(0.0, gate.unavailableDurationS());
  EXPECT_FALSE(gate.update(0.6, true));
}

TEST(RoughnessQuarantinePolicy, ClockJumpAndFrozenStampCannotOpenGate)
{
  slam_2d_core::ImuUnavailableFailOpenGate gate(0.3, 0.2);
  gate.update(1.0, false);
  EXPECT_FALSE(gate.update(10.0, false));
  EXPECT_DOUBLE_EQ(0.2, gate.unavailableDurationS());
  EXPECT_FALSE(gate.update(10.0, false));
}

TEST(RoughnessQuarantinePolicy, SparsePositiveClockStepsEventuallyOpenGate)
{
  slam_2d_core::ImuUnavailableFailOpenGate gate(0.5, 0.2);
  EXPECT_FALSE(gate.update(1.0, false));
  EXPECT_FALSE(gate.update(2.0, false));
  EXPECT_FALSE(gate.update(3.0, false));
  EXPECT_TRUE(gate.update(4.0, false));
  EXPECT_NEAR(0.6, gate.unavailableDurationS(), 1.0e-12);
}

TEST(RoughnessQuarantinePolicy, RejectsUnsafeNumericConfiguration)
{
  EXPECT_THROW(
    slam_2d_core::ImuUnavailableFailOpenGate gate(-1.0, 0.2),
    std::invalid_argument);
  EXPECT_THROW(
    slam_2d_core::ImuUnavailableFailOpenGate gate(1.0, 0.0),
    std::invalid_argument);
}

TEST(LaserScanStabilizer, IdentityPreservesValidBeam)
{
  auto source = makeScan();
  source.ranges[2] = 1.25F;
  const auto result = slam_2d_core::warpLaserScan(
    source, makeScan(), makeTransform(0.0, 0.0, 0.0), 1U);
  ASSERT_TRUE(result);
  EXPECT_EQ(1U, result->valid_beam_count);
  EXPECT_NEAR(1.25, result->scan.ranges[2], 1.0e-6);
  EXPECT_TRUE(result->scan.intensities.empty());
  EXPECT_FLOAT_EQ(0.0F, result->scan.time_increment);
}

TEST(LaserScanStabilizer, TranslationMovesWallTowardCurrentPose)
{
  auto source = makeScan();
  source.ranges[2] = 1.0F;
  const auto result = slam_2d_core::warpLaserScan(
    source, makeScan(), makeTransform(-0.2, 0.0, 0.0), 1U);
  ASSERT_TRUE(result);
  EXPECT_NEAR(0.8, result->scan.ranges[2], 1.0e-6);
}

TEST(LaserScanStabilizer, YawRotatesBeamIntoCurrentBin)
{
  auto source = makeScan();
  source.ranges[2] = 1.0F;
  const auto result = slam_2d_core::warpLaserScan(
    source, makeScan(), makeTransform(0.0, 0.0, kPi / 2.0), 1U);
  ASSERT_TRUE(result);
  EXPECT_NEAR(1.0, result->scan.ranges[4], 1.0e-6);
}

TEST(LaserScanStabilizer, RejectsInsufficientOutputBeams)
{
  auto source = makeScan();
  source.ranges[2] = 1.0F;
  EXPECT_FALSE(
    slam_2d_core::warpLaserScan(
      source, makeScan(), makeTransform(0.0, 0.0, 0.0), 2U));
}

TEST(LaserScanStabilizer, ResidualFilterPreservesConsistentScan)
{
  auto reference = makeScan(9U);
  reference.ranges.assign(9U, 2.0F);
  const auto result = slam_2d_core::filterLaserScanByResidual(
    reference, reference, makeTransform(0.0, 0.0, 0.0),
    0.08, 0U, 5U, 0.5, 5U,
    slam_2d_core::ResidualFilterDirection::kBoth, 0.5);
  ASSERT_TRUE(result);
  EXPECT_EQ(0U, result->removed_beam_count);
  EXPECT_EQ(9U, result->output_valid_beam_count);
}

TEST(LaserScanStabilizer, ResidualFilterRemovesOnlyDisagreeingBeams)
{
  auto reference = makeScan(9U);
  reference.ranges.assign(9U, 2.0F);
  auto current = reference;
  current.ranges[4] = 1.5F;
  const auto result = slam_2d_core::filterLaserScanByResidual(
    reference, current, makeTransform(0.0, 0.0, 0.0),
    0.08, 0U, 5U, 0.5, 5U,
    slam_2d_core::ResidualFilterDirection::kBoth, 0.5);
  ASSERT_TRUE(result);
  EXPECT_EQ(1U, result->removed_beam_count);
  EXPECT_FALSE(std::isfinite(result->scan.ranges[4]));
  EXPECT_TRUE(std::isfinite(result->scan.ranges[3]));
}

TEST(LaserScanStabilizer, ResidualFilterCanSelectOnlyUnexpectedlyCloserBeams)
{
  auto reference = makeScan(9U);
  reference.ranges.assign(9U, 2.0F);
  auto current = reference;
  current.ranges[2] = 1.5F;
  current.ranges[6] = 2.5F;
  const auto result = slam_2d_core::filterLaserScanByResidual(
    reference, current, makeTransform(0.0, 0.0, 0.0),
    0.08, 0U, 5U, 0.5, 5U,
    slam_2d_core::ResidualFilterDirection::kCloser, 0.5);
  ASSERT_TRUE(result);
  EXPECT_FALSE(std::isfinite(result->scan.ranges[2]));
  EXPECT_TRUE(std::isfinite(result->scan.ranges[6]));
}

TEST(LaserScanStabilizer, ResidualFilterLimitsRemovalToConfiguredBudget)
{
  auto reference = makeScan(10U);
  reference.ranges.assign(10U, 2.0F);
  auto current = reference;
  current.ranges[1] = 1.0F;
  current.ranges[2] = 1.1F;
  current.ranges[3] = 1.2F;
  const auto result = slam_2d_core::filterLaserScanByResidual(
    reference, current, makeTransform(0.0, 0.0, 0.0),
    0.08, 0U, 5U, 0.5, 5U,
    slam_2d_core::ResidualFilterDirection::kCloser, 0.2);
  ASSERT_TRUE(result);
  EXPECT_EQ(2U, result->removed_beam_count);
}

TEST(LaserScanStabilizer, ResidualFilterBreaksEqualResidualTiesByBeamIndex)
{
  auto reference = makeScan(10U);
  reference.ranges.assign(10U, 2.0F);
  auto current = reference;
  current.ranges[2] = 1.0F;
  current.ranges[7] = 1.0F;
  const auto result = slam_2d_core::filterLaserScanByResidual(
    reference, current, makeTransform(0.0, 0.0, 0.0),
    0.08, 0U, 5U, 0.5, 5U,
    slam_2d_core::ResidualFilterDirection::kCloser, 0.1);
  ASSERT_TRUE(result);
  EXPECT_FALSE(std::isfinite(result->scan.ranges[2]));
  EXPECT_TRUE(std::isfinite(result->scan.ranges[7]));
}

TEST(LaserScanStabilizer, ResidualFilterRejectsInsufficientOverlap)
{
  auto reference = makeScan(9U);
  reference.ranges[4] = 2.0F;
  auto current = makeScan(9U);
  current.ranges.assign(9U, 2.0F);
  EXPECT_FALSE(
    slam_2d_core::filterLaserScanByResidual(
      reference, current, makeTransform(0.0, 0.0, 0.0),
      0.08, 0U, 5U, 0.5, 5U,
      slam_2d_core::ResidualFilterDirection::kBoth, 0.5));
}

TEST(LaserScanStabilizer, ReportsPlanarAndOutOfPlaneMotion)
{
  auto transform = makeTransform(0.3, 0.4, 0.2);
  const double roll = 0.04;
  const double pitch = -0.03;
  const double yaw = 0.2;
  const double cr = std::cos(roll / 2.0);
  const double sr = std::sin(roll / 2.0);
  const double cp = std::cos(pitch / 2.0);
  const double sp = std::sin(pitch / 2.0);
  const double cy = std::cos(yaw / 2.0);
  const double sy = std::sin(yaw / 2.0);
  transform.transform.rotation.x = sr * cp * cy - cr * sp * sy;
  transform.transform.rotation.y = cr * sp * cy + sr * cp * sy;
  transform.transform.rotation.z = cr * cp * sy - sr * sp * cy;
  transform.transform.rotation.w = cr * cp * cy + sr * sp * sy;
  const auto motion = slam_2d_core::relativeMotion(transform);
  ASSERT_TRUE(motion.finite);
  EXPECT_NEAR(0.5, motion.translation_m, 1.0e-9);
  EXPECT_NEAR(yaw, motion.yaw_rad, 1.0e-9);
  EXPECT_NEAR(std::hypot(roll, pitch), motion.roll_pitch_rad, 1.0e-9);
}

TEST(LaserScanConditioner, ReportsValidBeamRatio)
{
  auto scan = makeScan(4U);
  scan.ranges[0] = 1.0F;
  scan.ranges[2] = 2.0F;
  const auto quality = slam_2d_core::laserScanQuality(scan);
  EXPECT_EQ(4U, quality.total_beam_count);
  EXPECT_EQ(2U, quality.valid_beam_count);
  EXPECT_DOUBLE_EQ(0.5, quality.valid_ratio);
}
