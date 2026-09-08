#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "slam_2d_core/near_chain_rotation_consistency.hpp"

namespace slam_2d_core
{
namespace
{

NearChainRotationConsistencyConfig testConfig()
{
  NearChainRotationConsistencyConfig config;
  config.enabled = true;
  config.history_size = 8U;
  config.minimum_history_batches = 3U;
  config.outlier_sigma = 4.0;
  config.extreme_yaw_sigma = 8.0;
  config.yaw_scale_floor_rad = 0.01;
  config.translation_scale_floor_m = 0.01;
  config.weighted_disagreement_scale_floor = 0.10;
  config.chain_overlap_trigger = 0.75;
  config.maximum_repeat_state_gap = 2;
  config.stable_recovery_batches = 3U;
  return config;
}

NearChainRotationCandidateObservation candidate(
  const double yaw_rad = 0.01,
  const double translation_m = 0.02,
  const double weighted_disagreement_sq = 0.20,
  std::vector<std::int32_t> chain = {1, 2, 3, 4})
{
  NearChainRotationCandidateObservation value;
  value.eligible = true;
  value.normalized_response = 0.25;
  value.translation_disagreement_m = translation_m;
  value.yaw_disagreement_rad = yaw_rad;
  value.covariance_weighted_disagreement_sq = weighted_disagreement_sq;
  value.chain_state_ids = std::move(chain);
  return value;
}

NearChainRotationBatchObservation batch(
  const std::int64_t stamp,
  const std::int32_t target,
  NearChainRotationCandidateObservation value = candidate(),
  const bool memory_available = true,
  const double memory_strength = 0.0)
{
  NearChainRotationBatchObservation observation;
  observation.stamp_ns = stamp;
  observation.target_state_id = target;
  observation.memory_available = memory_available;
  observation.memory_strength = memory_strength;
  observation.candidates.push_back(std::move(value));
  return observation;
}

void warmUp(NearChainRotationConsistency & policy)
{
  for (std::int32_t index = 0; index < 3; ++index) {
    policy.evaluate(batch(1'000 + index, index), false);
  }
}

NearChainRotationCandidateObservation outlier(
  std::vector<std::int32_t> chain = {1, 2, 3, 4, 5, 6, 7, 8})
{
  return candidate(0.30, 0.60, 100.0, std::move(chain));
}

TEST(NearChainRotationConsistency, ValidatesParameters)
{
  auto config = testConfig();
  std::string reason;
  EXPECT_TRUE(validateNearChainRotationConsistencyConfig(config, &reason));
  EXPECT_TRUE(reason.empty());

  config.minimum_history_batches = config.history_size + 1U;
  EXPECT_FALSE(validateNearChainRotationConsistencyConfig(config, &reason));
  EXPECT_FALSE(reason.empty());
  EXPECT_THROW(
      {
        NearChainRotationConsistency policy(config);
        (void)policy;
      },
    std::invalid_argument);

  config = testConfig();
  config.chain_overlap_trigger = 1.1;
  EXPECT_FALSE(validateNearChainRotationConsistencyConfig(config));
  config = testConfig();
  config.extreme_yaw_sigma = config.outlier_sigma - 0.1;
  EXPECT_FALSE(validateNearChainRotationConsistencyConfig(config));
}

TEST(NearChainRotationConsistency, DisabledPolicyDoesNotBuildState)
{
  auto config = testConfig();
  config.enabled = false;
  NearChainRotationConsistency policy(config);
  const auto decision = policy.evaluate(batch(1'000, 0), true);
  EXPECT_FALSE(decision.available);
  ASSERT_EQ(1U, decision.candidates.size());
  EXPECT_FALSE(decision.candidates.front().intervention_reject);
}

TEST(NearChainRotationConsistency, WarmupFailsOpenDuringActiveMemory)
{
  NearChainRotationConsistency policy(testConfig());
  const auto first = policy.evaluate(batch(1'000, 0, candidate(), true, 1.0), true);
  EXPECT_FALSE(first.model_available);
  EXPECT_TRUE(first.fail_open);
  EXPECT_FALSE(first.candidates.front().intervention_reject);

  policy.evaluate(batch(1'001, 1), false);
  policy.evaluate(batch(1'002, 2), false);
  const auto ready = policy.evaluate(batch(1'003, 3, candidate(), true, 1.0), true);
  EXPECT_TRUE(ready.model_available);
  EXPECT_FALSE(ready.fail_open);
}

TEST(NearChainRotationConsistency, HighQualityCandidateSurvivesMemoryAlone)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  const auto decision = policy.evaluate(
    batch(2'000, 10, candidate(), true, 1.0), true);
  ASSERT_TRUE(decision.model_available);
  EXPECT_TRUE(decision.memory_trigger);
  ASSERT_EQ(1U, decision.candidates.size());
  EXPECT_FALSE(decision.candidates.front().candidate_risky);
  EXPECT_FALSE(decision.candidates.front().proposed_reject);
  EXPECT_FALSE(decision.candidates.front().intervention_reject);
}

TEST(NearChainRotationConsistency, RoughnessPriorCanRejectExtremeCandidate)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  const auto decision = policy.evaluate(
    batch(2'000, 10, outlier(), true, 0.8), true);
  ASSERT_TRUE(decision.model_available);
  EXPECT_TRUE(decision.memory_trigger);
  EXPECT_TRUE(decision.latch_active_after);
  ASSERT_EQ(1U, decision.candidates.size());
  EXPECT_TRUE(decision.candidates.front().candidate_risky);
  EXPECT_TRUE(decision.candidates.front().proposed_reject);
  EXPECT_TRUE(decision.candidates.front().intervention_reject);
}

TEST(NearChainRotationConsistency, RepeatedOverlappingOutlierTriggersWithoutActiveMemory)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  const auto first = policy.evaluate(
    batch(2'000, 10, outlier(), true, 0.0), true);
  EXPECT_TRUE(first.candidates.front().candidate_risky);
  EXPECT_FALSE(first.candidates.front().proposed_reject);

  const auto repeated = policy.evaluate(
    batch(2'001, 11, outlier({2, 3, 4, 5, 6, 7, 8, 9}), true, 0.0), true);
  EXPECT_TRUE(repeated.repeated_chain_trigger);
  EXPECT_GT(repeated.candidates.front().prior_chain_overlap, 0.75);
  EXPECT_TRUE(repeated.candidates.front().intervention_reject);
  EXPECT_TRUE(repeated.latch_active_after);
}

TEST(NearChainRotationConsistency, UnrelatedChainDoesNotTriggerRepeat)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  policy.evaluate(batch(2'000, 10, outlier(), true, 0.0), true);
  const auto unrelated = policy.evaluate(
    batch(2'001, 11, outlier({20, 21, 22, 23}), true, 0.0), true);
  EXPECT_FALSE(unrelated.repeated_chain_trigger);
  EXPECT_FALSE(unrelated.candidates.front().proposed_reject);
}

TEST(NearChainRotationConsistency, MissingImuFailsOpenWhileLatched)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  ASSERT_TRUE(
    policy.evaluate(batch(2'000, 10, outlier(), true, 1.0), true)
    .latch_active_after);
  const auto missing = policy.evaluate(
    batch(2'001, 11, outlier(), false, 0.0), true);
  EXPECT_TRUE(missing.candidates.front().proposed_reject);
  EXPECT_TRUE(missing.candidates.front().fail_open);
  EXPECT_FALSE(missing.candidates.front().intervention_reject);
}

TEST(NearChainRotationConsistency, MissingCandidateQualityFailsOpen)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  ASSERT_TRUE(
    policy.evaluate(batch(2'000, 10, outlier(), true, 1.0), true)
    .latch_active_after);
  auto incomplete = candidate();
  incomplete.yaw_disagreement_rad = std::numeric_limits<double>::quiet_NaN();
  const auto missing = policy.evaluate(
    batch(2'001, 11, incomplete, true, 0.0), true);
  EXPECT_TRUE(missing.candidates.front().fail_open);
  EXPECT_FALSE(missing.candidates.front().intervention_reject);
}

TEST(NearChainRotationConsistency, StableBatchesReleaseLatch)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  ASSERT_TRUE(
    policy.evaluate(batch(2'000, 10, outlier(), true, 1.0), true)
    .latch_active_after);
  EXPECT_TRUE(policy.evaluate(batch(2'001, 11), true).latch_active_after);
  EXPECT_TRUE(policy.evaluate(batch(2'002, 12), true).latch_active_after);
  EXPECT_FALSE(policy.evaluate(batch(2'003, 13), true).latch_active_after);
}

TEST(NearChainRotationConsistency, TimestampRollbackResetsLearnedModel)
{
  NearChainRotationConsistency policy(testConfig());
  warmUp(policy);
  ASSERT_TRUE(policy.evaluate(batch(2'000, 10), false).model_available);
  const auto rollback = policy.evaluate(batch(900, 0), true);
  EXPECT_TRUE(rollback.timestamp_reset);
  EXPECT_FALSE(rollback.model_available);
  EXPECT_FALSE(rollback.latch_active_after);
}

TEST(NearChainRotationConsistency, IdenticalInputSequenceIsDeterministic)
{
  NearChainRotationConsistency first(testConfig());
  NearChainRotationConsistency second(testConfig());
  for (std::int32_t index = 0; index < 3; ++index) {
    first.evaluate(batch(1'000 + index, index), false);
    second.evaluate(batch(1'000 + index, index), false);
  }
  const auto first_result = first.evaluate(
    batch(2'000, 10, outlier(), true, 0.9), true);
  const auto second_result = second.evaluate(
    batch(2'000, 10, outlier(), true, 0.9), true);
  EXPECT_EQ(first_result.model_available, second_result.model_available);
  EXPECT_EQ(first_result.memory_trigger, second_result.memory_trigger);
  EXPECT_EQ(
    first_result.candidates.front().intervention_reject,
    second_result.candidates.front().intervention_reject);
  EXPECT_DOUBLE_EQ(
    first_result.yaw_limit.limit, second_result.yaw_limit.limit);
}

}  // namespace
}  // namespace slam_2d_core
