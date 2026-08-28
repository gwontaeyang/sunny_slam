#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "slam_2d_core/roughness_innovation_policy.hpp"

namespace slam_2d_core
{
namespace
{

RoughnessNearChainGateInput nominalNearChainInput()
{
  RoughnessNearChainGateInput input;
  input.enabled = true;
  input.scan_stamp_ns = 10'000'000'000;
  input.last_scan_stamp_ns = 9'900'000'000;
  input.imu_window_complete = true;
  input.rough_sample_count = 0;
  input.roughness_score = 1.0;
  input.roughness_score_gate = 2.75;
  input.minimum_rough_samples = 1;
  input.recovery_duration_s = 2.0;
  input.baseline_link_match_threshold = 0.08;
  input.gated_link_match_threshold = 1.0;
  return input;
}

}  // namespace

TEST(RoughnessInnovationPolicy, UsesMaximumNormalizedImuChannel)
{
  const RoughnessSignal signal{
    1.0,
    0.10,
    0.175,
    15.0,
    40.0,
  };
  const RoughnessThresholds thresholds;

  EXPECT_DOUBLE_EQ(normalizedRoughnessScore(signal, thresholds), 3.0);
}

TEST(RoughnessInnovationPolicy, RequiresCompleteImuWindow)
{
  const auto decision = roughnessInnovationDecision(
    4.0, false, true, 2.75, 0.20, 0.20, 0.04, 0.05);

  EXPECT_FALSE(decision.rough);
  EXPECT_FALSE(decision.reject);
}

TEST(RoughnessInnovationPolicy, RejectsOnlyRoughLargeInnovation)
{
  const auto rough_small = roughnessInnovationDecision(
    3.0, true, true, 2.75, 0.01, 0.01, 0.04, 0.05);
  const auto smooth_large = roughnessInnovationDecision(
    1.0, true, true, 2.75, 0.20, 0.20, 0.04, 0.05);
  const auto rough_translation = roughnessInnovationDecision(
    3.0, true, true, 2.75, 0.04, 0.01, 0.04, 0.05);
  const auto rough_yaw = roughnessInnovationDecision(
    3.0, true, true, 2.75, 0.01, -0.05, 0.04, 0.05);

  EXPECT_FALSE(rough_small.reject);
  EXPECT_FALSE(smooth_large.reject);
  EXPECT_TRUE(rough_translation.reject);
  EXPECT_TRUE(rough_yaw.reject);
}

TEST(RoughnessInnovationPolicy, FailsOpenForNonfiniteEvidence)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto decision = roughnessInnovationDecision(
    nan, true, true, 2.75, nan, nan, 0.04, 0.05);

  EXPECT_FALSE(decision.rough);
  EXPECT_FALSE(decision.translation_large);
  EXPECT_FALSE(decision.yaw_large);
  EXPECT_FALSE(decision.reject);
}

TEST(RoughnessInnovationPolicy, RequiresSustainedRoughness)
{
  const auto decision = roughnessInnovationDecision(
    4.0, true, false, 2.75, 0.20, 0.20, 0.04, 0.05);

  EXPECT_FALSE(decision.rough);
  EXPECT_FALSE(decision.reject);
}

TEST(RoughnessPostConfirmationPolicy, ConfirmsTranslationOnlyWithAllEvidence)
{
  const auto decision = roughnessPostConfirmationDecision(
    true, false,
    0.40, 0.006, 0.001,
    0.50, 0.003, 0.001,
    0.03, 0.0,
    0.03, 0.017453292519943295,
    0.05, 1.5);

  EXPECT_TRUE(decision.response_improved);
  EXPECT_TRUE(decision.translation_covariance_improved);
  EXPECT_TRUE(decision.translation_inconsistent);
  EXPECT_TRUE(decision.translation_confirmed_transient);
  EXPECT_TRUE(decision.confirmed_transient);
  EXPECT_DOUBLE_EQ(decision.response_gain, 0.10);
  EXPECT_DOUBLE_EQ(decision.translation_covariance_improvement_ratio, 2.0);
}

TEST(RoughnessPostConfirmationPolicy, ConfirmsYawAcrossWrappedSign)
{
  const auto decision = roughnessPostConfirmationDecision(
    false, true,
    0.40, 0.006, 0.003,
    0.50, 0.006, 0.001,
    0.0, -0.02,
    0.03, 0.017453292519943295,
    0.05, 1.5);

  EXPECT_TRUE(decision.yaw_inconsistent);
  EXPECT_TRUE(decision.yaw_covariance_improved);
  EXPECT_TRUE(decision.yaw_confirmed_transient);
  EXPECT_TRUE(decision.confirmed_transient);
}

TEST(RoughnessPostConfirmationPolicy, FailsOpenWithoutConfidenceImprovement)
{
  const auto weak_response = roughnessPostConfirmationDecision(
    true, false,
    0.40, 0.006, 0.001,
    0.44, 0.003, 0.001,
    0.04, 0.0,
    0.03, 0.017453292519943295,
    0.05, 1.5);
  const auto weak_covariance = roughnessPostConfirmationDecision(
    true, false,
    0.40, 0.006, 0.001,
    0.50, 0.005, 0.001,
    0.04, 0.0,
    0.03, 0.017453292519943295,
    0.05, 1.5);

  EXPECT_FALSE(weak_response.confirmed_transient);
  EXPECT_FALSE(weak_covariance.confirmed_transient);
}

TEST(RoughnessPostConfirmationPolicy, FailsOpenForNonfiniteEvidence)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto decision = roughnessPostConfirmationDecision(
    true, true,
    0.40, nan, nan,
    0.50, nan, nan,
    nan, nan,
    0.03, 0.017453292519943295,
    0.05, 1.5);

  EXPECT_FALSE(decision.translation_inconsistent);
  EXPECT_FALSE(decision.yaw_inconsistent);
  EXPECT_FALSE(decision.confirmed_transient);
}

TEST(RoughnessPostConfirmationPolicy, DoesNotConfirmAnInitiallySmallChannel)
{
  const auto decision = roughnessPostConfirmationDecision(
    false, false,
    0.40, 0.006, 0.003,
    0.50, 0.003, 0.001,
    0.10, 0.10,
    0.03, 0.017453292519943295,
    0.05, 1.5);

  EXPECT_FALSE(decision.translation_inconsistent);
  EXPECT_FALSE(decision.yaw_inconsistent);
  EXPECT_FALSE(decision.confirmed_transient);
}

TEST(RoughnessNearChainGatePolicy, DisabledUsesBaselineAndClearsHold)
{
  auto input = nominalNearChainInput();
  input.enabled = false;
  input.recovery_until_stamp_ns = 12'000'000'000;
  input.rough_sample_count = 3;
  input.roughness_score = 4.0;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kDisabled);
  EXPECT_FALSE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(decision.next_last_scan_stamp_ns, input.scan_stamp_ns);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 0);
  EXPECT_DOUBLE_EQ(decision.baseline_threshold, 0.08);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(RoughnessNearChainGatePolicy, CompleteStableScanIsNormal)
{
  const auto input = nominalNearChainInput();

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kNormal);
  EXPECT_FALSE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 0);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason), "normal");
}

TEST(RoughnessNearChainGatePolicy, RoughScanStartsAndExtendsRecoveryHold)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 2;
  input.roughness_score = 3.0;
  input.recovery_until_stamp_ns = 11'000'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRough);
  EXPECT_TRUE(decision.active);
  EXPECT_TRUE(decision.rough_trigger);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 1.0);
  EXPECT_STREQ(roughnessNearChainGateReasonName(decision.reason), "rough");
}

TEST(RoughnessNearChainGatePolicy, RoughScanNeverShortensExistingHold)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 1;
  input.roughness_score = 2.75;
  input.recovery_until_stamp_ns = 13'000'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRough);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 13'000'000'000);
}

TEST(RoughnessNearChainGatePolicy, StableScanInsideHoldUsesRecoveryGate)
{
  auto input = nominalNearChainInput();
  input.recovery_until_stamp_ns = input.scan_stamp_ns;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRecovery);
  EXPECT_TRUE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(
    decision.next_recovery_until_stamp_ns,
    input.recovery_until_stamp_ns);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 1.0);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason), "recovery");
}

TEST(RoughnessNearChainGatePolicy, ActiveGateStartsLoopClosingCooldown)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 1;
  input.roughness_score = 3.0;
  input.loop_closing_suppression_enabled = true;
  input.loop_closing_cooldown_s = 1.5;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_TRUE(decision.active);
  EXPECT_TRUE(decision.suppress_loop_closing);
  EXPECT_FALSE(decision.loop_closing_cooldown);
  EXPECT_EQ(
    decision.next_loop_closing_suppression_until_stamp_ns,
    11'500'000'000);
}

TEST(RoughnessNearChainGatePolicy, LoopClosingCooldownOutlivesThresholdGate)
{
  auto input = nominalNearChainInput();
  input.loop_closing_suppression_enabled = true;
  input.loop_closing_cooldown_s = 1.5;
  input.loop_closing_suppression_until_stamp_ns = 10'500'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kNormal);
  EXPECT_FALSE(decision.active);
  EXPECT_TRUE(decision.suppress_loop_closing);
  EXPECT_TRUE(decision.loop_closing_cooldown);
  EXPECT_EQ(
    decision.next_loop_closing_suppression_until_stamp_ns,
    input.loop_closing_suppression_until_stamp_ns);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(RoughnessNearChainGatePolicy, ExpiredLoopClosingCooldownClears)
{
  auto input = nominalNearChainInput();
  input.loop_closing_suppression_enabled = true;
  input.loop_closing_cooldown_s = 1.5;
  input.loop_closing_suppression_until_stamp_ns = input.scan_stamp_ns - 1;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_FALSE(decision.active);
  EXPECT_FALSE(decision.suppress_loop_closing);
  EXPECT_FALSE(decision.loop_closing_cooldown);
  EXPECT_EQ(decision.next_loop_closing_suppression_until_stamp_ns, 0);
}

TEST(RoughnessNearChainGatePolicy, SpatialRevisitPrecedesIncompleteWindow)
{
  auto input = nominalNearChainInput();
  input.imu_window_complete = false;
  input.spatial_revisit = true;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kSpatialRevisit);
  EXPECT_TRUE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 1.0);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason),
    "spatial_revisit");
}

TEST(RoughnessNearChainGatePolicy, CurrentRoughnessPrecedesSpatialRevisit)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 1;
  input.roughness_score = 3.0;
  input.spatial_revisit = true;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRough);
  EXPECT_TRUE(decision.active);
  EXPECT_TRUE(decision.rough_trigger);
}

TEST(RoughnessNearChainGatePolicy, SpatialPassagePrecedesIncompleteWindow)
{
  auto input = nominalNearChainInput();
  input.imu_window_complete = false;
  input.spatial_passage = true;
  input.extend_recovery_on_spatial_trigger = true;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kSpatialPassage);
  EXPECT_TRUE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason),
    "spatial_passage");
}

TEST(RoughnessNearChainGatePolicy, SpatialRevisitCanExtendRecoveryHold)
{
  auto input = nominalNearChainInput();
  input.spatial_revisit = true;
  input.extend_recovery_on_spatial_trigger = true;
  input.recovery_until_stamp_ns = 11'000'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kSpatialRevisit);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
}

TEST(RoughnessNearChainGatePolicy, SpatialRevisitPreservesLegacyHoldByDefault)
{
  auto input = nominalNearChainInput();
  input.spatial_revisit = true;
  input.recovery_until_stamp_ns = 11'000'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kSpatialRevisit);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 11'000'000'000);
}

TEST(RoughnessNearChainGatePolicy, ExpiredHoldReturnsToNormal)
{
  auto input = nominalNearChainInput();
  input.recovery_until_stamp_ns = input.scan_stamp_ns - 1;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kNormal);
  EXPECT_FALSE(decision.active);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 0);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(RoughnessNearChainGatePolicy, IncompleteWindowFailsOpenAndPreservesHold)
{
  auto input = nominalNearChainInput();
  input.imu_window_complete = false;
  input.rough_sample_count = 3;
  input.roughness_score = 4.0;
  input.recovery_until_stamp_ns = 12'000'000'000;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kIncomplete);
  EXPECT_FALSE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason), "incomplete");
}

TEST(RoughnessNearChainGatePolicy, PassageRecoveryProtectsIncompleteWindow)
{
  auto input = nominalNearChainInput();
  input.imu_window_complete = false;
  input.recovery_until_stamp_ns = 12'000'000'000;
  input.extend_recovery_on_spatial_trigger = true;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRecovery);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
}

TEST(RoughnessNearChainGatePolicy, PassageProtectsInvalidImuEvidence)
{
  auto input = nominalNearChainInput();
  input.roughness_score = std::numeric_limits<double>::quiet_NaN();
  input.spatial_passage = true;
  input.extend_recovery_on_spatial_trigger = true;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kSpatialPassage);
  EXPECT_TRUE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
}

TEST(RoughnessNearChainGatePolicy, TimestampRollbackResetsHoldAndFailsOpen)
{
  auto input = nominalNearChainInput();
  input.last_scan_stamp_ns = input.scan_stamp_ns + 1;
  input.recovery_until_stamp_ns = 12'000'000'000;
  input.rough_sample_count = 3;
  input.roughness_score = 4.0;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(
    decision.reason,
    RoughnessNearChainGateReason::kTimestampRollback);
  EXPECT_FALSE(decision.active);
  EXPECT_FALSE(decision.rough_trigger);
  EXPECT_EQ(decision.next_last_scan_stamp_ns, input.scan_stamp_ns);
  EXPECT_EQ(decision.next_recovery_until_stamp_ns, 0);
  EXPECT_EQ(decision.next_loop_closing_suppression_until_stamp_ns, 0);
  EXPECT_FALSE(decision.suppress_loop_closing);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(decision.reason),
    "timestamp_rollback");
}

TEST(RoughnessNearChainGatePolicy, NonfiniteRoughnessEvidenceFailsOpen)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const bool nan_score : {false, true}) {
    auto input = nominalNearChainInput();
    input.recovery_until_stamp_ns = 12'000'000'000;
    if (nan_score) {
      input.roughness_score = nan;
    } else {
      input.roughness_score_gate = nan;
    }

    const auto decision = roughnessNearChainGateDecision(input);

    EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kInvalidInput);
    EXPECT_FALSE(decision.active);
    EXPECT_FALSE(decision.rough_trigger);
    EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
    EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
  }
}

TEST(RoughnessNearChainGatePolicy, InvalidDurationFailsOpen)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double duration : {-1.0, nan}) {
    auto input = nominalNearChainInput();
    input.rough_sample_count = 1;
    input.roughness_score = 3.0;
    input.recovery_until_stamp_ns = 12'000'000'000;
    input.recovery_duration_s = duration;

    const auto decision = roughnessNearChainGateDecision(input);

    EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kInvalidInput);
    EXPECT_FALSE(decision.active);
    EXPECT_FALSE(decision.rough_trigger);
    EXPECT_EQ(decision.next_recovery_until_stamp_ns, 12'000'000'000);
    EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
  }
}

TEST(RoughnessNearChainGatePolicy, InvalidLoopClosingCooldownFailsOpen)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (const double duration : {-1.0, nan}) {
    auto input = nominalNearChainInput();
    input.rough_sample_count = 1;
    input.roughness_score = 3.0;
    input.loop_closing_suppression_enabled = true;
    input.loop_closing_cooldown_s = duration;

    const auto decision = roughnessNearChainGateDecision(input);

    EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kInvalidInput);
    EXPECT_FALSE(decision.active);
    EXPECT_FALSE(decision.suppress_loop_closing);
    EXPECT_EQ(decision.next_loop_closing_suppression_until_stamp_ns, 0);
  }
}

TEST(RoughnessNearChainGatePolicy, GatedThresholdCannotLowerBaseline)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 1;
  input.roughness_score = 3.0;
  input.gated_link_match_threshold = 0.01;

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_TRUE(decision.active);
  EXPECT_DOUBLE_EQ(decision.baseline_threshold, 0.08);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(RoughnessNearChainGatePolicy, NonfiniteGatedThresholdUsesBaseline)
{
  auto input = nominalNearChainInput();
  input.rough_sample_count = 1;
  input.roughness_score = 3.0;
  input.gated_link_match_threshold =
    std::numeric_limits<double>::quiet_NaN();

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRough);
  EXPECT_TRUE(decision.active);
  EXPECT_DOUBLE_EQ(decision.requested_threshold, 0.08);
}

TEST(RoughnessNearChainGatePolicy, RecoveryDurationAdditionSaturates)
{
  auto input = nominalNearChainInput();
  input.scan_stamp_ns = std::numeric_limits<std::int64_t>::max() - 10;
  input.last_scan_stamp_ns = input.scan_stamp_ns - 1;
  input.rough_sample_count = 1;
  input.roughness_score = 3.0;
  input.recovery_duration_s = std::numeric_limits<double>::max();

  const auto decision = roughnessNearChainGateDecision(input);

  EXPECT_EQ(decision.reason, RoughnessNearChainGateReason::kRough);
  EXPECT_TRUE(decision.active);
  EXPECT_EQ(
    decision.next_recovery_until_stamp_ns,
    std::numeric_limits<std::int64_t>::max());
}

TEST(RoughnessSpatialRevisitPolicy, MatchesOldNearbyOppositeHeadingZone)
{
  const std::vector<RoughnessSpatialZone> zones{
    {-3.780, 10.820, 0.0, 10'000'000'000},
    {4.0, 5.0, 0.0, 10'000'000'000},
  };

  const auto decision = roughnessSpatialRevisitDecision(
    true,
    -4.146,
    10.815,
    3.10,
    60'000'000'000,
    zones,
    0.6,
    20.0,
    2.2);

  EXPECT_TRUE(decision.input_valid);
  EXPECT_TRUE(decision.matched);
  EXPECT_EQ(decision.matched_zone_index, 0U);
  EXPECT_NEAR(decision.nearest_distance_m, 0.366, 1.0e-4);
  EXPECT_NEAR(decision.heading_delta_rad, 3.10, 1.0e-9);
  EXPECT_EQ(decision.source_rough_stamp_ns, 10'000'000'000);
}

TEST(RoughnessSpatialRevisitPolicy, RejectsSamePassAndSameHeading)
{
  const std::vector<RoughnessSpatialZone> zones{
    {1.0, 2.0, 0.1, 50'000'000'000},
  };

  const auto same_pass = roughnessSpatialRevisitDecision(
    true, 1.1, 2.0, 3.0, 55'000'000'000, zones, 0.6, 20.0, 2.2);
  const auto same_heading = roughnessSpatialRevisitDecision(
    true, 1.1, 2.0, 0.2, 80'000'000'000, zones, 0.6, 20.0, 2.2);

  EXPECT_TRUE(same_pass.input_valid);
  EXPECT_FALSE(same_pass.matched);
  EXPECT_TRUE(std::isnan(same_pass.nearest_distance_m));
  EXPECT_TRUE(same_heading.input_valid);
  EXPECT_FALSE(same_heading.matched);
  EXPECT_TRUE(std::isnan(same_heading.nearest_distance_m));
  EXPECT_TRUE(std::isnan(same_heading.heading_delta_rad));
}

TEST(RoughnessSpatialRevisitPolicy, SameHeadingZoneDoesNotHideValidZone)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 10'000'000'000},
    {0.2, 0.0, 3.0, 10'000'000'000},
  };

  const auto decision = roughnessSpatialRevisitDecision(
    true, 0.05, 0.0, 0.1, 40'000'000'000, zones, 0.6, 20.0, 2.2);

  EXPECT_TRUE(decision.matched);
  EXPECT_EQ(decision.matched_zone_index, 1U);
  EXPECT_NEAR(decision.nearest_distance_m, 0.15, 1.0e-9);
}

TEST(RoughnessSpatialRevisitPolicy, UsesWrappedHeadingDifference)
{
  const double pi = std::acos(-1.0);
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, pi - 0.1, 10'000'000'000},
  };

  const auto decision = roughnessSpatialRevisitDecision(
    true, 0.0, 0.0, -pi + 0.1, 40'000'000'000, zones, 0.2, 20.0, 0.15);

  EXPECT_TRUE(decision.matched);
  EXPECT_NEAR(decision.heading_delta_rad, 0.2, 1.0e-9);
}

TEST(RoughnessSpatialRevisitPolicy, InvalidConfigurationFailsOpen)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 10'000'000'000},
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();

  const auto disabled = roughnessSpatialRevisitDecision(
    false, 0.0, 0.0, 3.0, 40'000'000'000, zones, 0.2, 20.0, 2.2);
  const auto invalid = roughnessSpatialRevisitDecision(
    true, nan, 0.0, 3.0, 40'000'000'000, zones, 0.2, 20.0, 2.2);

  EXPECT_FALSE(disabled.input_valid);
  EXPECT_FALSE(disabled.matched);
  EXPECT_FALSE(invalid.input_valid);
  EXPECT_FALSE(invalid.matched);
}

TEST(RoughnessSpatialPassagePolicy, MatchesRecentNearbyZoneWithoutHeadingGate)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 3.0, 8'000'000'000},
    {0.5, 0.0, -3.0, 9'000'000'000},
  };

  const auto decision = roughnessSpatialPassageDecision(
    true, 0.45, 0.0, 10'000'000'000, zones, 0.55, 3.5);

  EXPECT_TRUE(decision.input_valid);
  EXPECT_TRUE(decision.matched);
  EXPECT_EQ(decision.matched_zone_index, 1U);
  EXPECT_NEAR(decision.nearest_distance_m, 0.05, 1.0e-9);
  EXPECT_NEAR(decision.source_age_s, 1.0, 1.0e-9);
  EXPECT_EQ(decision.source_rough_stamp_ns, 9'000'000'000);
}

TEST(RoughnessSpatialPassagePolicy, RejectsExpiredAndDistantZones)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 5'000'000'000},
    {2.0, 0.0, 0.0, 9'000'000'000},
  };

  const auto decision = roughnessSpatialPassageDecision(
    true, 0.0, 0.0, 10'000'000'000, zones, 0.55, 3.5);

  EXPECT_TRUE(decision.input_valid);
  EXPECT_FALSE(decision.matched);
  EXPECT_EQ(decision.matched_zone_index, 1U);
  EXPECT_DOUBLE_EQ(decision.nearest_distance_m, 2.0);
  EXPECT_DOUBLE_EQ(decision.source_age_s, 1.0);
}

TEST(RoughnessSpatialPassagePolicy, UsesLatestRoughStampForPassageAge)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 1'000'000'000, 9'000'000'000},
  };

  const auto decision = roughnessSpatialPassageDecision(
    true, 0.1, 0.0, 10'000'000'000, zones, 0.55, 3.5);

  EXPECT_TRUE(decision.matched);
  EXPECT_DOUBLE_EQ(decision.source_age_s, 1.0);
  EXPECT_EQ(decision.source_rough_stamp_ns, 9'000'000'000);
}

TEST(RoughnessSpatialPassagePolicy, RadiusAndMaximumAgeAreInclusive)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 1'000'000'000, 6'500'000'000},
  };

  const auto decision = roughnessSpatialPassageDecision(
    true, 0.55, 0.0, 10'000'000'000, zones, 0.55, 3.5);

  EXPECT_TRUE(decision.matched);
  EXPECT_DOUBLE_EQ(decision.nearest_distance_m, 0.55);
  EXPECT_DOUBLE_EQ(decision.source_age_s, 3.5);
}

TEST(RoughnessSpatialPassagePolicy, RejectsEqualOrFutureLatestStamp)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 1'000'000'000, 10'000'000'000},
    {0.0, 0.0, 0.0, 1'000'000'000, 11'000'000'000},
  };

  const auto decision = roughnessSpatialPassageDecision(
    true, 0.0, 0.0, 10'000'000'000, zones, 0.55, 3.5);

  EXPECT_TRUE(decision.input_valid);
  EXPECT_FALSE(decision.matched);
  EXPECT_TRUE(std::isnan(decision.nearest_distance_m));
}

TEST(RoughnessSpatialPassagePolicy, InvalidConfigurationFailsOpen)
{
  const std::vector<RoughnessSpatialZone> zones{
    {0.0, 0.0, 0.0, 9'000'000'000},
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();

  const auto disabled = roughnessSpatialPassageDecision(
    false, 0.0, 0.0, 10'000'000'000, zones, 0.55, 3.5);
  const auto invalid_radius = roughnessSpatialPassageDecision(
    true, 0.0, 0.0, 10'000'000'000, zones, -0.1, 3.5);
  const auto invalid_age = roughnessSpatialPassageDecision(
    true, 0.0, 0.0, 10'000'000'000, zones, 0.55, nan);

  EXPECT_FALSE(disabled.input_valid);
  EXPECT_FALSE(invalid_radius.input_valid);
  EXPECT_FALSE(invalid_age.input_valid);
  EXPECT_FALSE(disabled.matched);
  EXPECT_FALSE(invalid_radius.matched);
  EXPECT_FALSE(invalid_age.matched);
}

TEST(RoughnessPoseGraphAnchorPolicy, RefreshUsesLatestCorrectedPose)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {11, 3, {1.0, 2.0, 0.1, 10'000'000'000, 11'000'000'000}},
  };
  const std::vector<RoughnessCorrectedScanPose> corrected{
    {11, 3, -4.0, 8.0, 3.0},
  };

  const auto result = refreshRoughnessPoseGraphAnchors(anchors, corrected);

  ASSERT_EQ(anchors.size(), 1U);
  ASSERT_EQ(result.zones.size(), 1U);
  EXPECT_DOUBLE_EQ(anchors[0].zone.x_m, -4.0);
  EXPECT_DOUBLE_EQ(anchors[0].zone.y_m, 8.0);
  EXPECT_DOUBLE_EQ(anchors[0].zone.heading_rad, 3.0);
  EXPECT_EQ(anchors[0].zone.rough_stamp_ns, 10'000'000'000);
  EXPECT_EQ(anchors[0].zone.last_rough_stamp_ns, 11'000'000'000);
  EXPECT_DOUBLE_EQ(result.zones[0].x_m, -4.0);
  EXPECT_EQ(result.invalid_anchor_count, 0U);
  EXPECT_EQ(result.missing_scan_count, 0U);
  EXPECT_EQ(result.invalid_corrected_pose_count, 0U);
}

TEST(RoughnessPoseGraphAnchorPolicy, RefreshRemovesStaleAndInvalidAnchors)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {10, 1, {0.0, 0.0, 0.0, 1'000'000'000, 1'000'000'000}},
    {20, 2, {1.0, 0.0, 0.0, 2'000'000'000, 2'000'000'000}},
    {-1, 3, {2.0, 0.0, 0.0, 3'000'000'000, 3'000'000'000}},
  };
  const std::vector<RoughnessCorrectedScanPose> corrected{
    {10, 1, 5.0, 6.0, 0.5},
  };

  const auto result = refreshRoughnessPoseGraphAnchors(anchors, corrected);

  ASSERT_EQ(anchors.size(), 1U);
  EXPECT_EQ(anchors[0].scan_unique_id, 10);
  EXPECT_EQ(result.zones.size(), 1U);
  EXPECT_EQ(result.invalid_anchor_count, 1U);
  EXPECT_EQ(result.missing_scan_count, 1U);
  EXPECT_EQ(result.invalid_corrected_pose_count, 0U);
}

TEST(RoughnessPoseGraphAnchorPolicy, RefreshRejectsNonfiniteCorrectedPose)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {10, 1, {0.0, 0.0, 0.0, 1'000'000'000, 1'000'000'000}},
  };
  const std::vector<RoughnessCorrectedScanPose> corrected{
    {10, 1, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
  };

  const auto result = refreshRoughnessPoseGraphAnchors(anchors, corrected);

  EXPECT_TRUE(anchors.empty());
  EXPECT_TRUE(result.zones.empty());
  EXPECT_EQ(result.invalid_corrected_pose_count, 1U);
}

TEST(RoughnessPoseGraphAnchorPolicy, SamePassMergeRetainsFirstScanIdentity)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {10, 1, {0.0, 0.0, 0.0, 1'000'000'000, 1'000'000'000}},
  };
  const RoughnessPoseGraphAnchor current{
    11, 2, {0.05, 0.0, 0.1, 2'000'000'000, 2'000'000'000}};

  const auto result = rememberRoughnessPoseGraphAnchor(
    anchors, current, 0.10, 20.0, 200);

  ASSERT_EQ(anchors.size(), 1U);
  EXPECT_TRUE(result.accepted);
  EXPECT_TRUE(result.merged);
  EXPECT_FALSE(result.inserted);
  EXPECT_EQ(anchors[0].scan_unique_id, 10);
  EXPECT_EQ(anchors[0].scan_state_id, 1);
  EXPECT_DOUBLE_EQ(anchors[0].zone.x_m, 0.0);
  EXPECT_EQ(anchors[0].zone.rough_stamp_ns, 1'000'000'000);
  EXPECT_EQ(anchors[0].zone.last_rough_stamp_ns, 2'000'000'000);
}

TEST(RoughnessPoseGraphAnchorPolicy, CapacityEvictsOldestLatestObservation)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {10, 1, {0.0, 0.0, 0.0, 1'000'000'000, 4'000'000'000}},
    {20, 2, {2.0, 0.0, 0.0, 2'000'000'000, 8'000'000'000}},
  };
  const RoughnessPoseGraphAnchor current{
    30, 3, {4.0, 0.0, 0.0, 30'000'000'000, 30'000'000'000}};

  const auto result = rememberRoughnessPoseGraphAnchor(
    anchors, current, 0.10, 20.0, 2);

  ASSERT_EQ(anchors.size(), 2U);
  EXPECT_TRUE(result.accepted);
  EXPECT_TRUE(result.inserted);
  EXPECT_TRUE(result.evicted);
  EXPECT_EQ(anchors[0].scan_unique_id, 20);
  EXPECT_EQ(anchors[1].scan_unique_id, 30);
}

TEST(RoughnessPoseGraphAnchorPolicy, InvalidAnchorFailsOpenWithoutMutation)
{
  std::vector<RoughnessPoseGraphAnchor> anchors{
    {10, 1, {0.0, 0.0, 0.0, 1'000'000'000, 1'000'000'000}},
  };
  const RoughnessPoseGraphAnchor invalid{
    -1, 2, {1.0, 0.0, 0.0, 2'000'000'000, 2'000'000'000}};

  const auto result = rememberRoughnessPoseGraphAnchor(
    anchors, invalid, 0.10, 20.0, 2);

  EXPECT_FALSE(result.accepted);
  ASSERT_EQ(anchors.size(), 1U);
  EXPECT_EQ(anchors[0].scan_unique_id, 10);
}

TEST(RoughnessNearChainGatePolicy, UnknownReasonHasStableString)
{
  EXPECT_STREQ(
    roughnessNearChainGateReasonName(
      static_cast<RoughnessNearChainGateReason>(999)),
    "unknown");
}

}  // namespace slam_2d_core
