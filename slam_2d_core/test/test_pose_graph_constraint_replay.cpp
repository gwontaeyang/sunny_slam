#include "slam_2d_core/pose_graph_constraint_replay.hpp"

#include <array>
#include <sstream>
#include <string>

#include "gtest/gtest.h"

namespace slam_2d_core
{
namespace
{

constexpr const char * kHeader =
  "group_label,edge_label,source_state_id,target_state_id,relative_x_m,relative_y_m,"
  "relative_yaw_rad,covariance_xx,covariance_xy,covariance_xyaw,covariance_yy,"
  "covariance_yyaw,covariance_yawyaw\n";

TEST(PoseGraphConstraintReplay, ParsesValidatedGroup)
{
  std::istringstream input(
    std::string(kHeader) +
    "event4,pre,137,224,-10.6,-28.3,1.78,0.12,0,0,0.12,0,0.001\n" +
    "event4,post,139,226,-10.5,-28.5,1.79,0.14,0,0,0.14,0,0.002\n");
  const auto result = parsePoseGraphReplayConstraints(input);

  ASSERT_TRUE(result.errors.empty());
  ASSERT_EQ(result.constraints.size(), 2U);
  EXPECT_EQ(result.constraints[0].group_label, "event4");
  EXPECT_EQ(result.constraints[1].edge_label, "post");
  EXPECT_DOUBLE_EQ(result.constraints[0].relative_yaw_rad, 1.78);
}

TEST(PoseGraphConstraintReplay, RejectsDuplicateReversePairAtomically)
{
  std::istringstream input(
    std::string(kHeader) +
    "group,first,3,8,0,0,0,1,0,0,1,0,1\n" +
    "group,reverse,8,3,0,0,0,1,0,0,1,0,1\n");
  const auto result = parsePoseGraphReplayConstraints(input);

  EXPECT_FALSE(result.errors.empty());
  EXPECT_TRUE(result.constraints.empty());
}

TEST(PoseGraphConstraintReplay, RejectsNonFiniteAndNonPositiveCovariance)
{
  std::istringstream non_finite(
    std::string(kHeader) + "group,edge,1,2,nan,0,0,1,0,0,1,0,1\n");
  EXPECT_FALSE(parsePoseGraphReplayConstraints(non_finite).errors.empty());

  std::istringstream singular(
    std::string(kHeader) + "group,edge,1,2,0,0,0,1,1,0,1,0,1\n");
  EXPECT_FALSE(parsePoseGraphReplayConstraints(singular).errors.empty());
}

TEST(PoseGraphConstraintReplay, PositiveDefiniteCheckIncludesCrossTerms)
{
  EXPECT_TRUE(isPositiveDefiniteCovariance({2.0, 0.1, 0.2, 3.0, 0.1, 1.0}));
  EXPECT_FALSE(isPositiveDefiniteCovariance({1.0, 0.0, 2.0, 1.0, 0.0, 1.0}));
}

TEST(PoseGraphConstraintReplay, RequiresExactHeaderAndAtLeastOneConstraint)
{
  std::istringstream wrong_header("source,target\n1,2\n");
  EXPECT_FALSE(parsePoseGraphReplayConstraints(wrong_header).errors.empty());

  std::istringstream empty{std::string(kHeader)};
  EXPECT_FALSE(parsePoseGraphReplayConstraints(empty).errors.empty());
}

}  // namespace
}  // namespace slam_2d_core
