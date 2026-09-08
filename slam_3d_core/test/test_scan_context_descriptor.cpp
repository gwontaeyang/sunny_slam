#include "slam_3d_core/scan_context_descriptor.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

namespace
{

TEST(ScanContextDescriptor, IdenticalDescriptorsHaveZeroDistance)
{
  const std::vector<float> descriptor{
    1.0F, 2.0F, 3.0F, 4.0F,
    2.0F, 1.0F, 4.0F, 3.0F};

  const auto match = gole::slam_3d_core::matchScanContextDescriptors(
    descriptor, descriptor, 4);

  EXPECT_NEAR(match.distance, 0.0, 1.0e-12);
  EXPECT_EQ(match.best_shift, 0);
  EXPECT_EQ(match.compared_sectors, 4);
}

TEST(ScanContextDescriptor, RecoversCircularYawShift)
{
  const std::vector<float> query{
    1.0F, 2.0F, 3.0F, 4.0F,
    4.0F, 1.0F, 2.0F, 3.0F};
  const std::vector<float> candidate{
    2.0F, 3.0F, 4.0F, 1.0F,
    1.0F, 2.0F, 3.0F, 4.0F};

  const auto match = gole::slam_3d_core::matchScanContextDescriptors(
    query, candidate, 4);

  EXPECT_NEAR(match.distance, 0.0, 1.0e-12);
  EXPECT_EQ(match.best_shift, 1);
}

TEST(ScanContextDescriptor, ColumnCosineIgnoresPerSectorDensityScale)
{
  const std::vector<float> query{
    1.0F, 2.0F, 3.0F,
    2.0F, 4.0F, 6.0F};
  const std::vector<float> density_scaled{
    5.0F, 0.5F, 12.0F,
    10.0F, 1.0F, 24.0F};

  const auto match = gole::slam_3d_core::matchScanContextDescriptors(
    query, density_scaled, 3);

  EXPECT_NEAR(match.distance, 0.0, 1.0e-12);
  EXPECT_EQ(match.compared_sectors, 3);
}

TEST(ScanContextDescriptor, EmptyOrMismatchedInputFailsClosed)
{
  const auto empty = gole::slam_3d_core::matchScanContextDescriptors({}, {}, 60);
  const auto mismatched = gole::slam_3d_core::matchScanContextDescriptors(
    std::vector<float>{1.0F, 2.0F},
    std::vector<float>{1.0F},
    2);

  EXPECT_FALSE(std::isfinite(empty.distance));
  EXPECT_FALSE(std::isfinite(mismatched.distance));
}

}  // namespace
