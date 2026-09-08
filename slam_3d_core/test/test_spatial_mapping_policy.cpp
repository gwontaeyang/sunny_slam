#include <cmath>

#include <gtest/gtest.h>

#include "slam_3d_core/spatial_mapping_policy.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace
{

Eigen::Isometry3d pose(
  const double x,
  const double y,
  const double z,
  const double roll_deg,
  const double pitch_deg,
  const double yaw_deg)
{
  constexpr double degrees_to_radians = 3.14159265358979323846 / 180.0;
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.translation() = Eigen::Vector3d(x, y, z);
  result.linear() =
    (Eigen::AngleAxisd(yaw_deg * degrees_to_radians, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch_deg * degrees_to_radians, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll_deg * degrees_to_radians, Eigen::Vector3d::UnitX()))
    .toRotationMatrix();
  return result;
}

TEST(SpatialMappingPolicy, ParsesSupportedModesAndRejectsUnknownMode)
{
  ASSERT_TRUE(parseMappingSpatialMode("planar").has_value());
  EXPECT_EQ(*parseMappingSpatialMode("planar"), MappingSpatialMode::kPlanar);
  EXPECT_EQ(*parseMappingSpatialMode("se3"), MappingSpatialMode::kSe3);
  EXPECT_EQ(*parseMappingSpatialMode("floor_aware"), MappingSpatialMode::kFloorAware);
  EXPECT_FALSE(parseMappingSpatialMode("indoor").has_value());
}

TEST(SpatialMappingPolicy, PlanarModePreservesXyAndYawOnlyKeyframeRule)
{
  SpatialMappingPolicyConfig config;
  config.mode = MappingSpatialMode::kPlanar;
  config.keyframe_translation_m = 0.8;
  config.keyframe_yaw_deg = 8.0;
  config.keyframe_vertical_translation_m = 0.2;
  config.keyframe_rotation_3d_deg = 5.0;

  EXPECT_FALSE(shouldAddSpatialKeyframe(pose(0, 0, 2, 30, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_TRUE(shouldAddSpatialKeyframe(pose(0.8, 0, 0, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_TRUE(shouldAddSpatialKeyframe(pose(0, 0, 0, 0, 0, 8), pose(0, 0, 0, 0, 0, 0), config));
}

TEST(SpatialMappingPolicy, Se3ModeCreatesKeyframesForVerticalAndRollMotion)
{
  SpatialMappingPolicyConfig config;
  config.mode = MappingSpatialMode::kSe3;
  config.keyframe_translation_m = 0.8;
  config.keyframe_yaw_deg = 8.0;
  config.keyframe_vertical_translation_m = 0.2;
  config.keyframe_rotation_3d_deg = 6.0;

  EXPECT_TRUE(shouldAddSpatialKeyframe(pose(0, 0, 0.21, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_TRUE(shouldAddSpatialKeyframe(pose(0, 0, 0, 6.1, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_FALSE(
    shouldAddSpatialKeyframe(
      pose(0.1, 0, 0.1, 2, 0, 1), pose(0, 0, 0, 0, 0, 0),
      config));
}

TEST(SpatialMappingPolicy, FloorAwareWindowRejectsAnotherFloorAtSameXy)
{
  SpatialMappingPolicyConfig config;
  config.mode = MappingSpatialMode::kFloorAware;
  config.local_map_radius_m = 15.0;
  config.local_map_vertical_radius_m = 2.0;

  EXPECT_TRUE(
    poseWithinLocalMapWindow(
      pose(10, 0, 1.5, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_FALSE(
    poseWithinLocalMapWindow(
      pose(1, 0, 3.5, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_FALSE(
    poseWithinLocalMapWindow(
      pose(16, 0, 0, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
}

TEST(SpatialMappingPolicy, Se3WindowUsesSphericalDistance)
{
  SpatialMappingPolicyConfig config;
  config.mode = MappingSpatialMode::kSe3;
  config.local_map_radius_m = 3.0;
  config.local_map_vertical_radius_m = 2.95;

  EXPECT_TRUE(
    poseWithinLocalMapWindow(
      pose(0, 0, 2.9, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_FALSE(
    poseWithinLocalMapWindow(
      pose(0, 0, 3.1, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
  EXPECT_FALSE(
    poseWithinLocalMapWindow(
      pose(0, 0, 2.96, 0, 0, 0), pose(0, 0, 0, 0, 0, 0), config));
}

TEST(SpatialMappingPolicy, AccumulatedTravelClosesAnOutAndBackSubmap)
{
  const std::vector<Eigen::Isometry3d> poses = {
    pose(0, 0, 0, 0, 0, 0),
    pose(5, 0, 0, 0, 0, 0),
    pose(0, 0, 0, 0, 0, 180)};

  EXPECT_NEAR(
    accumulatedSpatialTravel(poses, MappingSpatialMode::kPlanar),
    10.0,
    1.0e-12);
  EXPECT_NEAR(
    spatialTravelDistance(poses.back(), poses.front(), MappingSpatialMode::kPlanar),
    0.0,
    1.0e-12);
}

}  // namespace
}  // namespace slam_3d_core
}  // namespace gole
