#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include "slam_2d_core/adaptive_roughness_memory.hpp"

namespace slam_2d_core
{
namespace
{

AdaptiveRoughnessMemoryObservation observation(
  const double stamp_s,
  const double x_m,
  const double score)
{
  AdaptiveRoughnessMemoryObservation value;
  value.stamp_ns = static_cast<std::int64_t>(stamp_s * 1.0e9);
  value.pose_available = true;
  value.x_m = x_m;
  value.y_m = 0.0;
  value.roughness_available = true;
  value.roughness_score = score;
  return value;
}

TEST(AdaptiveRoughnessMemory, ParsesEveryRequiredMode)
{
  for (const auto * name : {
        "intervention_off",
        "legacy_fixed_memory",
        "adaptive_memory_observe",
        "adaptive_memory_only",
        "adaptive_quality_aware"})
  {
    RoughnessConstraintMode mode;
    ASSERT_TRUE(parseRoughnessConstraintMode(name, mode));
    EXPECT_EQ(name, std::string(roughnessConstraintModeName(mode)));
  }
  RoughnessConstraintMode mode;
  EXPECT_FALSE(parseRoughnessConstraintMode("unknown", mode));
}

TEST(AdaptiveRoughnessMemory, DecaysMonotonicallyWithTimeAndDistance)
{
  AdaptiveRoughnessMemory memory;
  const auto initial = memory.update(observation(1.0, 0.0, 5.5));
  const auto later = memory.update(observation(1.4, 0.05, 1.5));
  const auto farther = memory.update(observation(1.8, 0.15, 1.5));
  ASSERT_TRUE(initial.active);
  EXPECT_GT(initial.strength, later.strength);
  EXPECT_GT(later.strength, farther.strength);
  EXPECT_GE(later.elapsed_s, initial.elapsed_s);
  EXPECT_GE(farther.distance_m, later.distance_m);
}

TEST(AdaptiveRoughnessMemory, StrongerEventGetsLongerMemory)
{
  AdaptiveRoughnessMemory weak;
  AdaptiveRoughnessMemory strong;
  const auto weak_state = weak.update(observation(1.0, 0.0, 2.75));
  const auto strong_state = strong.update(observation(1.0, 0.0, 5.5));
  EXPECT_LT(weak_state.strength, strong_state.strength);
  EXPECT_LT(weak_state.effective_duration_s, strong_state.effective_duration_s);
  EXPECT_LT(weak_state.effective_distance_m, strong_state.effective_distance_m);
}

TEST(AdaptiveRoughnessMemory, StableRecoveryAcceleratesDecayAndResets)
{
  AdaptiveRoughnessMemory memory;
  const auto initial = memory.update(observation(1.0, 0.0, 5.5));
  const auto stable = memory.update(observation(1.4, 0.0, 0.5));
  const auto reset = memory.update(observation(5.0, 0.0, 0.5));
  EXPECT_GT(initial.strength, stable.strength);
  EXPECT_GT(stable.stable_recovery_s, 0.0);
  EXPECT_FALSE(reset.active);
  EXPECT_DOUBLE_EQ(reset.strength, 0.0);
}

TEST(AdaptiveRoughnessMemory, WeakRetriggerDoesNotRechargePastMaximumIntensity)
{
  AdaptiveRoughnessMemory memory;
  ASSERT_TRUE(memory.update(observation(1.0, 0.0, 5.5)).active);
  const auto decayed = memory.update(observation(1.5, 0.0, 1.5));
  const auto retriggered = memory.update(observation(1.6, 0.0, 2.75));
  EXPECT_GT(decayed.strength, 0.5);
  EXPECT_LE(retriggered.strength, decayed.strength);
  EXPECT_LT(retriggered.event_intensity, 1.0);
}

TEST(AdaptiveRoughnessMemory, DeadbandBreaksContinuousStableRecovery)
{
  AdaptiveRoughnessMemory memory;
  ASSERT_TRUE(memory.update(observation(1.0, 0.0, 5.5)).active);
  const auto stable = memory.update(observation(1.2, 0.0, 0.5));
  ASSERT_GT(stable.stable_recovery_s, 0.0);
  const auto deadband = memory.update(observation(1.3, 0.0, 1.5));
  EXPECT_DOUBLE_EQ(deadband.stable_recovery_s, 0.0);
  const auto stable_again = memory.update(observation(1.4, 0.0, 0.5));
  EXPECT_NEAR(stable_again.stable_recovery_s, 0.1, 1.0e-9);
}

TEST(AdaptiveRoughnessMemory, TimestampRollbackResetsState)
{
  AdaptiveRoughnessMemory memory;
  ASSERT_TRUE(memory.update(observation(2.0, 0.0, 5.5)).active);
  const auto rollback = memory.update(observation(1.0, 0.0, 1.0));
  EXPECT_FALSE(rollback.available);
  EXPECT_TRUE(rollback.timestamp_reset);
  EXPECT_FALSE(rollback.active);
  const auto restarted = memory.update(observation(3.0, 0.0, 1.0));
  EXPECT_TRUE(restarted.available);
  EXPECT_FALSE(restarted.active);
}

TEST(AdaptiveRoughnessMemory, MissingImuOrPoseIsUnavailableForFailOpen)
{
  AdaptiveRoughnessMemory memory;
  auto missing_imu = observation(1.0, 0.0, 5.5);
  missing_imu.roughness_available = false;
  EXPECT_FALSE(memory.update(missing_imu).available);

  auto missing_pose = observation(2.0, 0.0, 5.5);
  missing_pose.pose_available = false;
  EXPECT_FALSE(memory.update(missing_pose).available);
}

TEST(AdaptiveRoughnessMemory, RollbackAcrossUnavailableObservationResetsState)
{
  AdaptiveRoughnessMemory memory;
  ASSERT_TRUE(memory.update(observation(100.0, 0.0, 5.5)).active);
  auto unavailable = observation(200.0, 0.0, 1.0);
  unavailable.roughness_available = false;
  EXPECT_FALSE(memory.update(unavailable).available);

  const auto rollback = memory.update(observation(150.0, 0.0, 1.0));
  EXPECT_TRUE(rollback.timestamp_reset);
  EXPECT_FALSE(rollback.available);
  EXPECT_FALSE(rollback.active);
}

TEST(AdaptiveRoughnessMemory, InvalidParametersAreRejected)
{
  AdaptiveRoughnessMemoryConfig config;
  config.maximum_distance_m = -1.0;
  std::string reason;
  EXPECT_FALSE(validateAdaptiveRoughnessMemoryConfig(config, &reason));
  EXPECT_FALSE(reason.empty());
  EXPECT_THROW(
      {
        AdaptiveRoughnessMemory memory(config);
        (void)memory;
      },
    std::invalid_argument);

  config = AdaptiveRoughnessMemoryConfig{};
  config.trigger_score = 0.0;
  EXPECT_FALSE(validateAdaptiveRoughnessMemoryConfig(config, &reason));
  config = AdaptiveRoughnessMemoryConfig{};
  config.minimum_duration_s = 0.0;
  EXPECT_FALSE(validateAdaptiveRoughnessMemoryConfig(config, &reason));
  config = AdaptiveRoughnessMemoryConfig{};
  config.minimum_distance_m = 0.0;
  EXPECT_FALSE(validateAdaptiveRoughnessMemoryConfig(config, &reason));
}

NearChainQualityConfig calibratedQualityConfig()
{
  NearChainQualityConfig config;
  config.calibrated = true;
  return config;
}

NearChainQualityObservation poorCandidate()
{
  NearChainQualityObservation observation;
  observation.available = true;
  observation.normalized_response = 0.08;
  observation.translation_disagreement_m = 0.10;
  observation.yaw_disagreement_rad = 0.08726646259971647;
  observation.xy_covariance_max_eigenvalue_m2 = 0.04;
  observation.yaw_covariance_rad2 = 0.030461741978670857;
  return observation;
}

TEST(NearChainQuality, RequiresCalibrationAndCompleteEvidence)
{
  NearChainQualityConfig uncalibrated;
  EXPECT_FALSE(evaluateNearChainQuality(poorCandidate(), uncalibrated).available);
  auto incomplete = poorCandidate();
  incomplete.yaw_covariance_rad2 = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(
    evaluateNearChainQuality(incomplete, calibratedQualityConfig()).available);
}

TEST(NearChainQuality, HighQualityConstraintHasZeroRisk)
{
  auto candidate = poorCandidate();
  candidate.normalized_response = 0.30;
  candidate.translation_disagreement_m = 0.0;
  candidate.yaw_disagreement_rad = 0.0;
  candidate.xy_covariance_max_eigenvalue_m2 = 0.0;
  candidate.yaw_covariance_rad2 = 0.0;
  const auto quality = evaluateNearChainQuality(
    candidate, calibratedQualityConfig());
  ASSERT_TRUE(quality.available);
  EXPECT_DOUBLE_EQ(quality.risk, 0.0);
}

TEST(NearChainQuality, WeakResponseAloneHasZeroRisk)
{
  auto candidate = poorCandidate();
  candidate.translation_disagreement_m = 0.0;
  candidate.yaw_disagreement_rad = 0.0;
  candidate.xy_covariance_max_eigenvalue_m2 = 0.0;
  candidate.yaw_covariance_rad2 = 0.0;
  const auto quality = evaluateNearChainQuality(
    candidate, calibratedQualityConfig());
  ASSERT_TRUE(quality.available);
  EXPECT_DOUBLE_EQ(quality.response_risk, 1.0);
  EXPECT_DOUBLE_EQ(quality.disagreement_risk, 0.0);
  EXPECT_DOUBLE_EQ(quality.covariance_risk, 0.0);
  EXPECT_DOUBLE_EQ(quality.risk, 0.0);
}

TEST(AdaptiveConstraintDecision, QualityAwareModeKeepsHighQualityConstraint)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.active = true;
  memory.strength = 1.0;
  NearChainQuality quality;
  quality.available = true;
  quality.risk = 0.0;
  const auto decision = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveQualityAware,
    memory,
    quality,
    AdaptiveConstraintPolicyConfig{});
  EXPECT_TRUE(decision.evidence_available);
  EXPECT_FALSE(decision.intervention_applied);
  EXPECT_EQ(RoughnessConstraintAction::kAccepted, decision.action);
}

TEST(AdaptiveConstraintDecision, ZeroDownweightBoundaryKeepsZeroRiskConstraint)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.strength = 1.0;
  NearChainQuality quality;
  quality.available = true;
  quality.risk = 0.0;
  AdaptiveConstraintPolicyConfig config;
  config.downweight_risk = 0.0;
  const auto decision = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveQualityAware,
    memory,
    quality,
    config);
  EXPECT_FALSE(decision.intervention_applied);
  EXPECT_EQ(RoughnessConstraintAction::kAccepted, decision.action);
}

TEST(AdaptiveConstraintDecision, QualityAwareModeFailsOpenWithoutQuality)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.active = true;
  memory.strength = 1.0;
  const auto decision = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveQualityAware,
    memory,
    NearChainQuality{},
    AdaptiveConstraintPolicyConfig{});
  EXPECT_TRUE(decision.fail_open);
  EXPECT_FALSE(decision.intervention_applied);
  EXPECT_EQ(RoughnessConstraintAction::kAccepted, decision.action);
}

TEST(AdaptiveConstraintDecision, ObserveAndOffModesNeverIntervene)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.strength = 1.0;
  NearChainQuality quality;
  quality.available = true;
  quality.risk = 1.0;
  for (const auto mode : {
        RoughnessConstraintMode::kInterventionOff,
        RoughnessConstraintMode::kAdaptiveMemoryObserve})
  {
    const auto decision = adaptiveConstraintDecision(
      mode, memory, quality, AdaptiveConstraintPolicyConfig{});
    EXPECT_FALSE(decision.intervention_applied);
    EXPECT_EQ(RoughnessConstraintAction::kAccepted, decision.action);
  }
}

TEST(AdaptiveConstraintDecision, MemoryOnlyModeRemainsExplicitAblation)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.strength = 1.0;
  const auto decision = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveMemoryOnly,
    memory,
    NearChainQuality{},
    AdaptiveConstraintPolicyConfig{});
  EXPECT_TRUE(decision.intervention_applied);
  EXPECT_EQ(RoughnessConstraintAction::kRejected, decision.action);
}

TEST(AdaptiveConstraintDecision, IsDeterministicForIdenticalInput)
{
  AdaptiveRoughnessMemoryState memory;
  memory.available = true;
  memory.strength = 0.7;
  NearChainQuality quality;
  quality.available = true;
  quality.risk = 0.8;
  const auto first = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveQualityAware,
    memory,
    quality,
    AdaptiveConstraintPolicyConfig{});
  const auto second = adaptiveConstraintDecision(
    RoughnessConstraintMode::kAdaptiveQualityAware,
    memory,
    quality,
    AdaptiveConstraintPolicyConfig{});
  EXPECT_EQ(first.action, second.action);
  EXPECT_DOUBLE_EQ(first.combined_risk, second.combined_risk);
  EXPECT_DOUBLE_EQ(first.information_scale, second.information_scale);
}

TEST(DynamicResponseThreshold, SeparatesRejectedFromPreservedCandidates)
{
  const std::vector<DynamicResponseThresholdCandidate> candidates{
    {true, 0.10, RoughnessConstraintAction::kRejected},
    {true, 0.20, RoughnessConstraintAction::kAccepted},
    {true, 0.30, RoughnessConstraintAction::kDownweighted},
  };
  const auto decision = deriveDynamicResponseThreshold(
    0.08, 1.0, 1.0e-6, candidates);
  ASSERT_TRUE(decision.available);
  EXPECT_TRUE(decision.intervention_required);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_GT(decision.requested_threshold, 0.10);
  EXPECT_LT(decision.requested_threshold, 0.20);
}

TEST(DynamicResponseThreshold, FailsOpenWhenResponsesOverlap)
{
  const std::vector<DynamicResponseThresholdCandidate> candidates{
    {true, 0.20, RoughnessConstraintAction::kRejected},
    {true, 0.10, RoughnessConstraintAction::kAccepted},
  };
  const auto decision = deriveDynamicResponseThreshold(
    0.08, 1.0, 1.0e-6, candidates);
  EXPECT_TRUE(decision.available);
  EXPECT_TRUE(decision.fail_open);
  EXPECT_FALSE(decision.intervention_required);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(DynamicResponseThreshold, LeavesBaselineWhenNoCandidateNeedsRejection)
{
  const std::vector<DynamicResponseThresholdCandidate> candidates{
    {true, 0.10, RoughnessConstraintAction::kAccepted},
    {true, 0.20, RoughnessConstraintAction::kDownweighted},
  };
  const auto decision = deriveDynamicResponseThreshold(
    0.08, 1.0, 1.0e-6, candidates);
  EXPECT_TRUE(decision.available);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_FALSE(decision.intervention_required);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(DynamicResponseThreshold, RejectsInvalidParameters)
{
  const auto decision = deriveDynamicResponseThreshold(
    0.08, 0.05, 1.0e-6, {});
  EXPECT_FALSE(decision.available);
  EXPECT_TRUE(decision.fail_open);
}

TEST(AdaptiveConstraintDecision, RejectsZeroRejectRiskConfiguration)
{
  AdaptiveConstraintPolicyConfig config;
  config.downweight_risk = 0.0;
  config.reject_risk = 0.0;
  std::string reason;
  EXPECT_FALSE(validateAdaptiveConstraintPolicyConfig(config, &reason));
  EXPECT_FALSE(reason.empty());
}

}  // namespace
}  // namespace slam_2d_core
