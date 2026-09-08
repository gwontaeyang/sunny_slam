// Copyright 2026 Gole Robotics

#include <gtest/gtest.h>

#include <limits>

#include "slam_3d_core/imu_shock_policy.hpp"

namespace
{

using gole::slam_3d_core::ImuShockPhase;
using gole::slam_3d_core::ImuShockObservation;
using gole::slam_3d_core::ImuShockPolicyConfig;
using gole::slam_3d_core::ImuShockPolicyState;
using gole::slam_3d_core::updateImuShockPolicy;

TEST(ImuShockPolicy, DisabledPolicyNeverQuarantines)
{
  ImuShockPolicyState state;
  const ImuShockPolicyConfig config{};

  const auto decision = updateImuShockPolicy(
    config, ImuShockObservation::kShock, 10.0, state);

  EXPECT_FALSE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kDisabled);
}

TEST(ImuShockPolicy, HoldsThenRequiresStableRecoveryScans)
{
  ImuShockPolicyConfig config;
  config.enabled = true;
  config.hold_duration_s = 0.5;
  config.recovery_stable_scans = 3;
  ImuShockPolicyState state;

  auto decision = updateImuShockPolicy(config, ImuShockObservation::kShock, 10.0, state);
  ASSERT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.49, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.50, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kRecovering);
  EXPECT_EQ(decision.recovery_stable_scan_count, 1);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.60, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.recovery_stable_scan_count, 2);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.70, state);
  EXPECT_FALSE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kStable);
  EXPECT_TRUE(decision.just_recovered);
}

TEST(ImuShockPolicy, NewEventRestartsRecovery)
{
  ImuShockPolicyConfig config;
  config.enabled = true;
  config.hold_duration_s = 0.2;
  config.recovery_stable_scans = 2;
  ImuShockPolicyState state;

  (void)updateImuShockPolicy(config, ImuShockObservation::kShock, 5.0, state);
  auto decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 5.2, state);
  ASSERT_EQ(decision.phase, ImuShockPhase::kRecovering);

  decision = updateImuShockPolicy(config, ImuShockObservation::kShock, 5.25, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  EXPECT_EQ(decision.recovery_stable_scan_count, 0);
}

TEST(ImuShockPolicy, TimestampRewindFailsClosed)
{
  ImuShockPolicyConfig config;
  config.enabled = true;
  ImuShockPolicyState state;

  (void)updateImuShockPolicy(config, ImuShockObservation::kShock, 10.0, state);
  const auto decision = updateImuShockPolicy(
    config, ImuShockObservation::kStable, 1.0, state);

  EXPECT_TRUE(decision.time_reset);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
}

TEST(ImuShockPolicy, UnknownAndDuplicateObservationsRestartRecoveryHold)
{
  ImuShockPolicyConfig config;
  config.enabled = true;
  config.hold_duration_s = 0.2;
  config.recovery_stable_scans = 2;
  ImuShockPolicyState state;

  (void)updateImuShockPolicy(config, ImuShockObservation::kShock, 5.0, state);
  auto decision = updateImuShockPolicy(config, ImuShockObservation::kUnknown, 5.3, state);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  EXPECT_EQ(decision.recovery_stable_scan_count, 0);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 5.4, state);
  ASSERT_EQ(decision.phase, ImuShockPhase::kRecovering);
  EXPECT_EQ(decision.recovery_stable_scan_count, 1);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 5.4, state);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  EXPECT_EQ(decision.recovery_stable_scan_count, 0);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 5.59, state);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 5.61, state);
  EXPECT_EQ(decision.phase, ImuShockPhase::kRecovering);

  decision = updateImuShockPolicy(config, ImuShockObservation::kUnknown, 5.7, state);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  EXPECT_EQ(decision.recovery_stable_scan_count, 0);
}

TEST(ImuShockPolicy, NonFiniteTimeFailsClosed)
{
  ImuShockPolicyConfig config;
  config.enabled = true;
  config.hold_duration_s = 0.5;
  config.recovery_stable_scans = 2;
  ImuShockPolicyState state;

  auto decision = updateImuShockPolicy(
    config,
    ImuShockObservation::kUnknown,
    std::numeric_limits<double>::quiet_NaN(),
    state);

  EXPECT_TRUE(decision.active);
  EXPECT_TRUE(decision.time_reset);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.0, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);
  EXPECT_DOUBLE_EQ(decision.time_since_event_s, 0.0);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.49, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kShock);

  decision = updateImuShockPolicy(config, ImuShockObservation::kStable, 10.50, state);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.phase, ImuShockPhase::kRecovering);
}

}  // namespace
