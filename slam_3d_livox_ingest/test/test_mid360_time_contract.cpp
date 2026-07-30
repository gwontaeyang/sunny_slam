#include <cstdint>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "slam_3d_livox_ingest/mid360_time_contract.hpp"

namespace
{

using gole::slam_3d_livox_ingest::TimeContractCode;
using gole::slam_3d_livox_ingest::TimeContractOptions;
using gole::slam_3d_livox_ingest::timeContractCodeString;
using gole::slam_3d_livox_ingest::validateMid360TimeContract;

struct TestPoint
{
  std::uint32_t offset_time;
  float x;
  float y;
  float z;
};

std::vector<TestPoint> validPoints()
{
  return {
    {0, 1.0F, 0.0F, 0.0F},
    {50000000, 2.0F, 0.0F, 0.0F},
    {100000000, 3.0F, 0.0F, 0.0F},
  };
}

TEST(Mid360TimeContract, AcceptsOrderedNanosecondOffsets)
{
  const auto points = validPoints();
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    static_cast<std::uint32_t>(points.size()),
    points,
    TimeContractOptions{});

  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.code, TimeContractCode::kValid);
  EXPECT_EQ(result.finite_point_count, 3U);
  EXPECT_EQ(result.scan_duration_ns, 100000000U);
  EXPECT_DOUBLE_EQ(result.finite_point_ratio, 1.0);
  EXPECT_STREQ(timeContractCodeString(result.code), "valid");
}

TEST(Mid360TimeContract, RejectsDeclaredPointCountMismatch)
{
  const auto points = validPoints();
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    4,
    points,
    TimeContractOptions{});

  EXPECT_EQ(result.code, TimeContractCode::kPointCountMismatch);
}

TEST(Mid360TimeContract, RejectsOffsetRegression)
{
  auto points = validPoints();
  points[2].offset_time = 10000000;
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    static_cast<std::uint32_t>(points.size()),
    points,
    TimeContractOptions{});

  EXPECT_EQ(result.code, TimeContractCode::kOffsetRegression);
  EXPECT_EQ(result.offset_regression_count, 1U);
}

TEST(Mid360TimeContract, RejectsBrokenConstantPointTimes)
{
  auto points = validPoints();
  for (auto & point : points) {
    point.offset_time = 0;
  }
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    static_cast<std::uint32_t>(points.size()),
    points,
    TimeContractOptions{});

  EXPECT_EQ(result.code, TimeContractCode::kZeroScanDuration);
}

TEST(Mid360TimeContract, RejectsExcessiveScanDuration)
{
  auto points = validPoints();
  points.back().offset_time = 250000000;
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    static_cast<std::uint32_t>(points.size()),
    points,
    TimeContractOptions{});

  EXPECT_EQ(result.code, TimeContractCode::kScanDurationTooLong);
}

TEST(Mid360TimeContract, RejectsInsufficientFinitePointRatio)
{
  auto points = validPoints();
  points[1].x = std::numeric_limits<float>::quiet_NaN();
  TimeContractOptions options;
  options.min_finite_point_ratio = 0.9;
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    static_cast<std::uint32_t>(points.size()),
    points,
    options);

  EXPECT_EQ(result.code, TimeContractCode::kInsufficientFinitePoints);
  EXPECT_EQ(result.invalid_point_count, 1U);
}

TEST(Mid360TimeContract, RejectsTimebaseOutsideBuiltinTimeRange)
{
  const auto points = validPoints();
  constexpr std::uint64_t outside_ros_time =
    static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) *
    1000000000ULL + 1000000000ULL;
  const auto result = validateMid360TimeContract(
    outside_ros_time,
    static_cast<std::uint32_t>(points.size()),
    points,
    TimeContractOptions{});

  EXPECT_EQ(result.code, TimeContractCode::kTimebaseOutsideRosRange);
}

TEST(Mid360TimeContract, DisabledStrictChecksRemainExplicit)
{
  auto points = validPoints();
  points[0].offset_time = 2000000;
  points[2].offset_time = 1000000;
  TimeContractOptions options;
  options.require_point_count_match = false;
  options.require_monotonic_offsets = false;
  options.require_zero_based_offset = false;
  options.require_nonzero_duration = false;
  const auto result = validateMid360TimeContract(
    10000000000ULL,
    99,
    points,
    options);

  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.offset_regression_count, 1U);
}

}  // namespace
