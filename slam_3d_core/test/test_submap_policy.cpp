#include <gtest/gtest.h>

#include "slam_3d_core/submap_policy.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace
{

TEST(SubmapPolicy, ClosesAtKeyframeLimit)
{
  const SubmapPolicyConfig config{20, 12.0, 10};
  EXPECT_EQ(submapClosureReason(19, 11.0, config), SubmapClosureReason::kNone);
  EXPECT_EQ(
    submapClosureReason(20, 11.0, config),
    SubmapClosureReason::kKeyframeLimit);
}

TEST(SubmapPolicy, ClosesAtTravelLimit)
{
  const SubmapPolicyConfig config{20, 12.0, 10};
  EXPECT_EQ(
    submapClosureReason(8, 12.0, config),
    SubmapClosureReason::kTravelLimit);
}

TEST(SubmapPolicy, KeyframeLimitHasDeterministicPriority)
{
  const SubmapPolicyConfig config{20, 12.0, 10};
  EXPECT_EQ(
    submapClosureReason(20, 12.0, config),
    SubmapClosureReason::kKeyframeLimit);
}

TEST(SubmapPolicy, OverlapAlwaysLeavesRoomForProgress)
{
  EXPECT_EQ(boundedSubmapOverlap(20, 10), 10U);
  EXPECT_EQ(submapOverlapStartIndex(20, 10), 10U);
  EXPECT_EQ(boundedSubmapOverlap(5, 99), 4U);
  EXPECT_EQ(submapOverlapStartIndex(5, 99), 1U);
  EXPECT_EQ(boundedSubmapOverlap(1, 1), 0U);
  EXPECT_EQ(submapOverlapStartIndex(1, 1), 1U);
}

}  // namespace
}  // namespace slam_3d_core
}  // namespace gole
