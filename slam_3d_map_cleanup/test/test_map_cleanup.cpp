#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

#include "slam_3d_map_cleanup/map_cleanup.hpp"

namespace gole
{
namespace slam_3d_map_cleanup
{
namespace
{

TEST(MapCleanup, DefaultsMatchIndoorBaseline)
{
  const TemporalStaticMapConfig config;
  EXPECT_TRUE(config.enabled);
  EXPECT_DOUBLE_EQ(config.voxel_size_m, 0.22);
  EXPECT_EQ(config.min_hit_observations, 3);
  EXPECT_DOUBLE_EQ(config.min_observation_span_s, 1.0);
  EXPECT_DOUBLE_EQ(config.min_viewpoint_distance_m, 0.6);
  EXPECT_TRUE(config.ray_clearing_enabled);
  EXPECT_DOUBLE_EQ(config.ray_endpoint_margin_m, 0.45);
  EXPECT_DOUBLE_EQ(config.max_ray_length_m, 30.0);
  EXPECT_EQ(config.max_rays_per_observation, 2000);
  EXPECT_TRUE(config.prior_free_enabled);
  EXPECT_EQ(config.min_prior_free_observations, 4);
  EXPECT_EQ(config.max_prior_free_observations, 8);
  EXPECT_DOUBLE_EQ(config.prior_free_max_sensor_vertical_drop_m, 0.9);
  EXPECT_DOUBLE_EQ(config.promotion_min_hit_ratio, 0.35);
  EXPECT_DOUBLE_EQ(config.retention_min_hit_ratio, 0.3);
  EXPECT_EQ(config.min_miss_observations_for_demotion, 4);
  EXPECT_TRUE(validTemporalStaticMapConfig(config));
}

TEST(MapCleanup, TransformsSensorOriginAndEndpointFromBase)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::AngleAxisd(std::acos(-1.0) / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  pose.translation() = Eigen::Vector3d(10.0, 20.0, 3.0);
  const TemporalStaticMapSensorView base{
    Eigen::Vector3d(1.0, 2.0, 0.5), {Eigen::Vector3d(4.0, 5.0, 1.0)}};
  const auto world = projectSensorView(pose, base, MapZFilter{}, 0.22);
  EXPECT_TRUE(world.sensor_origin.isApprox(Eigen::Vector3d(8.0, 21.0, 3.5)));
  ASSERT_EQ(world.endpoints.size(), 1U);
  EXPECT_TRUE(world.endpoints.front().isApprox(Eigen::Vector3d(5.0, 24.0, 4.0)));
}

TEST(MapCleanup, AppliesInclusiveWorldZFilterAfterRotation)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::AngleAxisd(std::acos(-1.0) / 2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  pose.translation() = Eigen::Vector3d(0.0, 0.0, 2.0);
  const TemporalStaticMapSensorView base{Eigen::Vector3d::Zero(),
    {Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(2.0, 0.0, 0.0),
      Eigen::Vector3d(-1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)}};
  const auto world = projectSensorView(pose, base, MapZFilter{true, 0.0, 1.0}, 0.22);
  ASSERT_EQ(world.endpoints.size(), 2U);
  EXPECT_NEAR(world.endpoints[0].z(), 1.0, 1.0e-12);
  EXPECT_NEAR(world.endpoints[1].z(), 0.0, 1.0e-12);
  EXPECT_DOUBLE_EQ(world.sensor_origin.z(), 2.0);
}

TEST(MapCleanup, UsesPostLoopPosesOnFreshProjection)
{
  TemporalStaticMapConfig config;
  config.ray_clearing_enabled = false;
  TemporalStaticMapAccumulator before(config);
  TemporalStaticMapAccumulator after(config);
  const TemporalStaticMapSensorView base{Eigen::Vector3d(1.0, 0.0, 0.0),
    {Eigen::Vector3d(2.1, 0.1, 0.1)}};
  Eigen::Isometry3d optimized_pose = Eigen::Isometry3d::Identity();
  optimized_pose.translation().x() = 10.0;
  for (int index = 0; index < 3; ++index) {
    before.addObservation(index, {projectSensorView(
          Eigen::Isometry3d::Identity(), base, MapZFilter{}, config.voxel_size_m)});
    after.addObservation(index, {projectSensorView(
          optimized_pose, base, MapZFilter{}, config.voxel_size_m)});
  }
  const auto old_points = before.points(TemporalStaticMapSelection::kConfirmed);
  const auto new_points = after.points(TemporalStaticMapSelection::kConfirmed);
  ASSERT_EQ(old_points.size(), 1U);
  ASSERT_EQ(new_points.size(), 1U);
  EXPECT_NEAR(old_points.front().position.x(), 2.1, 1.0e-12);
  EXPECT_NEAR(new_points.front().position.x(), 12.1, 1.0e-12);
  EXPECT_EQ(new_points.front().hit_observations, 3);
}

TEST(MapCleanup, TransformedOriginControlsRayContradictions)
{
  TemporalStaticMapConfig config;
  config.voxel_size_m = 0.5;
  config.min_hit_observations = 1;
  config.min_observation_span_s = 0.0;
  config.min_viewpoint_distance_m = 0.0;
  config.prior_free_enabled = false;
  TemporalStaticMapAccumulator accumulator(config);
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(10.0, 20.0, 0.0);
  pose.linear() = Eigen::AngleAxisd(std::acos(-1.0) / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const TemporalStaticMapSensorView first{Eigen::Vector3d(1.1, 0.1, 0.1),
    {Eigen::Vector3d(3.1, 0.1, 0.1)}};
  const TemporalStaticMapSensorView second{first.sensor_origin,
    {Eigen::Vector3d(5.1, 0.1, 0.1)}};
  accumulator.addObservation(0.0, {projectSensorView(pose, first, MapZFilter{}, 0.5)});
  accumulator.addObservation(1.0, {projectSensorView(pose, second, MapZFilter{}, 0.5)});
  EXPECT_EQ(accumulator.stats().ray_miss_updates, 1U);
}

TEST(MapCleanup, EmptyAfterZFilterIsValidButEmptyInputIsNot)
{
  const TemporalStaticMapSensorView view{Eigen::Vector3d::Zero(), {Eigen::Vector3d(1.0, 0.0, 2.0)}};
  const auto projected = projectSensorView(
    Eigen::Isometry3d::Identity(), view, MapZFilter{true, 0.0, 1.0}, 0.22);
  EXPECT_TRUE(projected.endpoints.empty());
  EXPECT_THROW(projectSensorView(Eigen::Isometry3d::Identity(),
      TemporalStaticMapSensorView{}, MapZFilter{}, 0.22), std::runtime_error);
}

TEST(MapCleanup, RejectsInvalidTransformOriginPointAndBounds)
{
  auto pose = Eigen::Isometry3d::Identity();
  TemporalStaticMapSensorView base{Eigen::Vector3d::Zero(), {Eigen::Vector3d(1.0, 0.0, 0.0)}};
  pose.linear()(0, 0) = 2.0;
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{}, 0.22), std::runtime_error);
  pose = Eigen::Isometry3d::Identity();
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{true, 2.0, 1.0}, 0.22), std::runtime_error);
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{}, 0.0), std::runtime_error);
  base.sensor_origin.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{}, 0.22), std::runtime_error);
  base.sensor_origin.x() = 0.0;
  base.endpoints.front().x() = std::numeric_limits<double>::infinity();
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{}, 0.22), std::runtime_error);
  base.endpoints.front().x() = 1.0e30;
  EXPECT_THROW(projectSensorView(pose, base, MapZFilter{}, 0.22), std::runtime_error);
}

}  // namespace
}  // namespace slam_3d_map_cleanup
}  // namespace gole
