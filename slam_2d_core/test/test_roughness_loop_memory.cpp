#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "slam_2d_core/roughness_loop_memory.hpp"

namespace slam_2d_core
{
namespace
{

RoughnessScanMemorySample sample(
  const std::int32_t unique_id,
  const double strength,
  const std::uint64_t generation = 4U)
{
  RoughnessScanMemorySample value;
  value.mapper_generation = generation;
  value.state_id = unique_id;
  value.unique_id = unique_id;
  value.stamp_ns = 1000 + unique_id;
  value.available = true;
  value.roughness_score_available = true;
  value.roughness_score = strength * 5.5;
  value.memory_strength = strength;
  value.effective_duration_s = 0.5 + strength * 3.0;
  value.effective_distance_m = 0.1 + strength * 0.45;
  return value;
}

TEST(RoughnessLoopMemory, RetainsPlacePriorAfterTransientMemoryDecays)
{
  RoughnessLoopMemory memory;
  memory.reset(4U);
  ASSERT_TRUE(memory.record(sample(10, 0.8)));
  ASSERT_TRUE(memory.record(sample(11, 0.3)));
  ASSERT_TRUE(memory.record(sample(20, 0.0)));

  const auto prior = memory.evaluate(4U, {10, 11}, 11, 20);

  ASSERT_TRUE(prior.available);
  EXPECT_EQ(prior.source_peak_unique_id, 10);
  EXPECT_DOUBLE_EQ(prior.source_peak_strength, 0.8);
  EXPECT_DOUBLE_EQ(prior.actual_source_strength, 0.3);
  EXPECT_DOUBLE_EQ(prior.target_strength, 0.0);
  EXPECT_DOUBLE_EQ(prior.combined_strength, 0.8);
}

TEST(RoughnessLoopMemory, CombinesSourceAndTargetWithoutExceedingUnitRange)
{
  RoughnessLoopMemory memory;
  memory.reset(4U);
  ASSERT_TRUE(memory.record(sample(1, 0.75)));
  ASSERT_TRUE(memory.record(sample(2, 0.6)));

  const auto prior = memory.evaluate(4U, {1}, 1, 2);

  ASSERT_TRUE(prior.available);
  EXPECT_NEAR(prior.combined_strength, 0.9, 1.0e-12);
}

TEST(RoughnessLoopMemory, MissingSourceOrTargetFailsOpenAsUnavailable)
{
  RoughnessLoopMemory memory;
  memory.reset(4U);
  ASSERT_TRUE(memory.record(sample(1, 0.8)));

  EXPECT_FALSE(memory.evaluate(4U, {1}, 1, 99).available);
  EXPECT_FALSE(memory.evaluate(4U, {99}, 99, 1).available);
  EXPECT_FALSE(memory.evaluate(4U, {}, 1, 1).available);
}

TEST(RoughnessLoopMemory, GenerationResetPreventsCrossMapReuse)
{
  RoughnessLoopMemory memory;
  memory.reset(4U);
  ASSERT_TRUE(memory.record(sample(1, 0.8)));

  memory.reset(5U);

  EXPECT_EQ(memory.size(), 0U);
  EXPECT_EQ(memory.find(4U, 1), nullptr);
  EXPECT_FALSE(memory.record(sample(2, 0.5, 4U)));
  EXPECT_TRUE(memory.record(sample(2, 0.5, 5U)));
}

TEST(RoughnessLoopMemory, InvalidEvidenceIsRejectedWithoutMutation)
{
  RoughnessLoopMemory memory;
  memory.reset(4U);
  auto invalid = sample(1, 0.5);
  invalid.memory_strength = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(memory.record(invalid));
  EXPECT_EQ(memory.size(), 0U);
}

TEST(RoughnessLoopMemory, CapacityEvictsOldestScanDeterministically)
{
  RoughnessLoopMemory memory(2U);
  memory.reset(4U);
  ASSERT_TRUE(memory.record(sample(1, 0.1)));
  ASSERT_TRUE(memory.record(sample(2, 0.2)));
  ASSERT_TRUE(memory.record(sample(3, 0.3)));

  EXPECT_EQ(memory.find(4U, 1), nullptr);
  EXPECT_NE(memory.find(4U, 2), nullptr);
  EXPECT_NE(memory.find(4U, 3), nullptr);
  EXPECT_EQ(memory.size(), 2U);
}

TEST(RoughnessLoopMemory, RejectsZeroCapacity)
{
  EXPECT_THROW(RoughnessLoopMemory memory(0U), std::invalid_argument);
}

}  // namespace
}  // namespace slam_2d_core
