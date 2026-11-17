#include <gtest/gtest.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "slam_3d_core/temporal_static_map.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace
{

TemporalStaticMapConfig testConfig()
{
  TemporalStaticMapConfig config;
  config.enabled = true;
  config.voxel_size_m = 0.5;
  config.min_hit_observations = 3;
  config.min_observation_span_s = 1.0;
  config.min_viewpoint_distance_m = 0.5;
  config.ray_clearing_enabled = false;
  config.ray_endpoint_margin_m = 0.4;
  config.max_ray_length_m = 10.0;
  config.max_rays_per_observation = 100;
  config.prior_free_enabled = true;
  config.min_prior_free_observations = 4;
  config.max_prior_free_observations = 8;
  config.prior_free_max_sensor_vertical_drop_m = 0.0;
  config.promotion_min_hit_ratio = 0.5;
  config.retention_min_hit_ratio = 0.3;
  config.min_miss_observations_for_demotion = 2;
  return config;
}

TEST(TemporalStaticMap, RejectsInvalidConfiguration)
{
  auto config = testConfig();
  config.voxel_size_m = 0.0;
  EXPECT_FALSE(validTemporalStaticMapConfig(config));
  EXPECT_THROW(TemporalStaticMapAccumulator accumulator(config), std::invalid_argument);

  config = testConfig();
  config.retention_min_hit_ratio = config.promotion_min_hit_ratio + 0.1;
  EXPECT_FALSE(validTemporalStaticMapConfig(config));

  config = testConfig();
  config.min_prior_free_observations = 0;
  EXPECT_FALSE(validTemporalStaticMapConfig(config));

  config = testConfig();
  config.min_prior_free_observations = 5;
  config.max_prior_free_observations = 4;
  EXPECT_FALSE(validTemporalStaticMapConfig(config));

  config = testConfig();
  config.prior_free_max_sensor_vertical_drop_m = -0.1;
  EXPECT_FALSE(validTemporalStaticMapConfig(config));
}

TEST(TemporalStaticMap, CountsOneMergedKeyframeOnlyOncePerVoxel)
{
  TemporalStaticMapAccumulator accumulator(testConfig());
  accumulator.addObservation(
    0.0,
    std::vector<TemporalStaticMapSensorView>{
          {Eigen::Vector3d(0.5, 0.0, 0.0), {Eigen::Vector3d(2.01, 0.0, 0.0)}},
          {Eigen::Vector3d(-0.5, 0.0, 0.0), {Eigen::Vector3d(2.20, 0.1, 0.1)}}});

  const auto candidates = accumulator.points(TemporalStaticMapSelection::kAll);
  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(candidates.front().hit_observations, 1);
  EXPECT_DOUBLE_EQ(candidates.front().viewpoint_distance_m, 0.0);
  EXPECT_FALSE(candidates.front().confirmed);
  EXPECT_EQ(accumulator.stats().endpoint_hit_updates, 1U);
}

TEST(TemporalStaticMap, PromotesTimeOrViewpointSeparatedStaticEvidence)
{
  TemporalStaticMapAccumulator accumulator(testConfig());
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d(0.0, 0.5 * index, 0.0),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }

  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  ASSERT_EQ(confirmed.size(), 1U);
  EXPECT_EQ(confirmed.front().hit_observations, 3);
  EXPECT_DOUBLE_EQ(confirmed.front().observation_span_s, 2.0);
  EXPECT_GE(confirmed.front().viewpoint_distance_m, 1.0);
  EXPECT_EQ(accumulator.stats().promotions, 1U);
}

TEST(TemporalStaticMap, LeavesAOnePassMovingTrailUnconfirmed)
{
  TemporalStaticMapAccumulator accumulator(testConfig());
  for (int index = 0; index < 6; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d(static_cast<double>(index), 0.0, 0.0),
      {Eigen::Vector3d(2.1 + index, 1.1, 0.1)});
  }

  EXPECT_TRUE(accumulator.points(TemporalStaticMapSelection::kConfirmed).empty());
  EXPECT_EQ(accumulator.points(TemporalStaticMapSelection::kRejected).size(), 6U);
}

TEST(TemporalStaticMap, RayMissesDemotePreviouslyConfirmedVoxel)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  TemporalStaticMapAccumulator accumulator(config);
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d(0.0, 0.2 * index, 0.0),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }
  ASSERT_EQ(accumulator.points(TemporalStaticMapSelection::kConfirmed).size(), 1U);

  for (int index = 0; index < 8; ++index) {
    accumulator.addObservation(
      3.0 + index,
      Eigen::Vector3d(0.0, 0.1, 0.0),
      {Eigen::Vector3d(4.1 + 0.5 * index, 0.1, 0.1)});
  }

  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  const auto rejected = accumulator.points(TemporalStaticMapSelection::kRejected);
  EXPECT_TRUE(confirmed.empty());
  ASSERT_EQ(rejected.size(), 9U);
  EXPECT_GE(accumulator.stats().ray_miss_updates, 8U);
  EXPECT_EQ(accumulator.stats().demotions, 1U);
}

TEST(TemporalStaticMap, PriorFreeEvidencePreventsPromotionAfterLongKnownFreeHistory)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  TemporalStaticMapAccumulator accumulator(config);
  for (int index = 0; index < 12; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(5.1, 0.1, 0.1)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      12.0 + index,
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }

  const auto candidates = accumulator.points(TemporalStaticMapSelection::kAll);
  const auto target = std::find_if(
    candidates.begin(), candidates.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target, candidates.end());
  EXPECT_EQ(target->hit_observations, 3);
  EXPECT_EQ(target->miss_observations, config.max_prior_free_observations);
  EXPECT_FALSE(target->confirmed);
  EXPECT_GT(accumulator.stats().prior_free_voxel_count, 0U);
  EXPECT_GT(accumulator.stats().prior_free_update_count, 0U);
  EXPECT_EQ(accumulator.stats().prior_free_seeded_candidate_count, 1U);
}

TEST(TemporalStaticMap, PersistentSurfaceRecoversFromCappedPriorFreeEvidence)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.max_prior_free_observations = 4;
  TemporalStaticMapAccumulator accumulator(config);
  for (int index = 0; index < 4; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(5.1, 0.1, 0.1)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      4.0 + index,
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }
  const auto before_recovery = accumulator.points(TemporalStaticMapSelection::kAll);
  const auto target_before_recovery = std::find_if(
    before_recovery.begin(), before_recovery.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target_before_recovery, before_recovery.end());
  EXPECT_FALSE(target_before_recovery->confirmed);

  accumulator.addObservation(
    7.0,
    Eigen::Vector3d::Zero(),
    {Eigen::Vector3d(2.1, 0.1, 0.1)});
  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  const auto target = std::find_if(
    confirmed.begin(), confirmed.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target, confirmed.end());
  EXPECT_EQ(target->hit_observations, 4);
  EXPECT_EQ(target->miss_observations, 4);
  EXPECT_DOUBLE_EQ(target->hit_ratio, config.promotion_min_hit_ratio);
}

TEST(TemporalStaticMap, SubthresholdPriorFreeEvidenceDoesNotBlockPromotion)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  TemporalStaticMapAccumulator accumulator(config);
  for (int index = 0; index < config.min_prior_free_observations - 1; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(5.1, 0.1, 0.1)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      3.0 + index,
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }

  const auto candidates = accumulator.points(TemporalStaticMapSelection::kAll);
  const auto target = std::find_if(
    candidates.begin(), candidates.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target, candidates.end());
  EXPECT_EQ(target->miss_observations, 0);
  EXPECT_TRUE(target->confirmed);
  EXPECT_EQ(accumulator.stats().prior_free_seeded_candidate_count, 0U);
}

TEST(TemporalStaticMap, CountsPriorFreeVoxelOnceAcrossMultipleRaysPerObservation)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.min_prior_free_observations = 1;
  TemporalStaticMapAccumulator accumulator(config);
  accumulator.addObservation(
    0.0,
    Eigen::Vector3d::Zero(),
    {Eigen::Vector3d(4.1, 0.1, 0.1), Eigen::Vector3d(5.1, 0.1, 0.1)});
  accumulator.addObservation(
    1.0,
    Eigen::Vector3d::Zero(),
    {Eigen::Vector3d(2.1, 0.1, 0.1)});

  const auto candidates = accumulator.points(TemporalStaticMapSelection::kAll);
  const auto target = std::find_if(
    candidates.begin(), candidates.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target, candidates.end());
  EXPECT_EQ(target->hit_observations, 1);
  EXPECT_EQ(target->miss_observations, 1);
  EXPECT_EQ(accumulator.stats().prior_free_seeded_candidate_count, 1U);
}

TEST(TemporalStaticMap, DisabledPriorFreeDoesNotSeedFirstHitMisses)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.prior_free_enabled = false;
  TemporalStaticMapAccumulator accumulator(config);
  for (int index = 0; index < 4; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(5.1, 0.1, 0.1)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      4.0 + index,
      Eigen::Vector3d::Zero(),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }

  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  const auto target = std::find_if(
    confirmed.begin(), confirmed.end(),
    [](const auto & point) {return std::abs(point.position.x() - 2.1) < 1.0e-12;});
  ASSERT_NE(target, confirmed.end());
  EXPECT_EQ(target->miss_observations, 0);
  EXPECT_EQ(accumulator.stats().prior_free_voxel_count, 0U);
  EXPECT_EQ(accumulator.stats().prior_free_seeded_candidate_count, 0U);
}

TEST(TemporalStaticMap, PriorFreeVerticalBandProtectsGrazingFloorSurface)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.prior_free_max_sensor_vertical_drop_m = 0.8;
  TemporalStaticMapAccumulator accumulator(config);
  const Eigen::Vector3d sensor_origin(0.0, 0.0, 1.3);
  for (int index = 0; index < 4; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      sensor_origin,
      {Eigen::Vector3d(10.0, 0.1, 0.1)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      4.0 + index,
      sensor_origin,
      {Eigen::Vector3d(8.1, 0.1, 0.2)});
  }

  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  const auto target = std::find_if(
    confirmed.begin(), confirmed.end(),
    [](const auto & point) {return std::abs(point.position.x() - 8.1) < 1.0e-12;});
  ASSERT_NE(target, confirmed.end());
  EXPECT_EQ(target->miss_observations, 0);
}

TEST(TemporalStaticMap, PriorFreeVerticalBandStillRejectsUpperBodyHeight)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.prior_free_max_sensor_vertical_drop_m = 0.8;
  TemporalStaticMapAccumulator accumulator(config);
  const Eigen::Vector3d sensor_origin(0.0, 0.0, 1.3);
  for (int index = 0; index < 4; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      sensor_origin,
      {Eigen::Vector3d(10.0, 0.1, 0.7)});
  }
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      4.0 + index,
      sensor_origin,
      {Eigen::Vector3d(8.1, 0.1, 0.8)});
  }

  const auto rejected = accumulator.points(TemporalStaticMapSelection::kRejected);
  const auto target = std::find_if(
    rejected.begin(), rejected.end(),
    [](const auto & point) {return std::abs(point.position.x() - 8.1) < 1.0e-12;});
  ASSERT_NE(target, rejected.end());
  EXPECT_EQ(target->miss_observations, 4);
  EXPECT_EQ(accumulator.stats().prior_free_seeded_candidate_count, 1U);
}

TEST(TemporalStaticMap, VerticalBandDoesNotDisableMissesForExistingFloorCandidate)
{
  auto config = testConfig();
  config.ray_clearing_enabled = true;
  config.prior_free_max_sensor_vertical_drop_m = 0.8;
  TemporalStaticMapAccumulator accumulator(config);
  const Eigen::Vector3d sensor_origin(0.0, 0.0, 1.3);
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      sensor_origin,
      {Eigen::Vector3d(8.1, 0.1, 0.2)});
  }
  ASSERT_EQ(accumulator.points(TemporalStaticMapSelection::kConfirmed).size(), 1U);

  for (int index = 0; index < 8; ++index) {
    accumulator.addObservation(
      3.0 + index,
      sensor_origin,
      {Eigen::Vector3d(10.1, 0.1, 0.1)});
  }

  const auto rejected = accumulator.points(TemporalStaticMapSelection::kRejected);
  const auto target = std::find_if(
    rejected.begin(), rejected.end(),
    [](const auto & point) {return std::abs(point.position.x() - 8.1) < 1.0e-12;});
  ASSERT_NE(target, rejected.end());
  EXPECT_GE(target->miss_observations, config.min_miss_observations_for_demotion);
  EXPECT_EQ(accumulator.stats().demotions, 1U);
}

TEST(TemporalStaticMap, SensorViewOrderDoesNotChangeMergedPointEvidence)
{
  auto config = testConfig();
  config.min_observation_span_s = 100.0;
  config.min_viewpoint_distance_m = 0.75;
  const auto accumulate = [&config](const bool reverse_views) {
      TemporalStaticMapAccumulator accumulator(config);
      for (int index = 0; index < 3; ++index) {
        const double y = 0.4 * index;
        std::vector<TemporalStaticMapSensorView> views{
          {Eigen::Vector3d(1.0, y, 0.0), {Eigen::Vector3d(2.01, 0.0, 0.0)}},
          {Eigen::Vector3d(-1.0, y, 0.0), {Eigen::Vector3d(2.20, 0.1, 0.1)}}};
        if (reverse_views) {
          std::reverse(views.begin(), views.end());
        }
        accumulator.addObservation(static_cast<double>(index), views);
      }
      return accumulator.points(TemporalStaticMapSelection::kAll);
    };

  const auto forward = accumulate(false);
  const auto reversed = accumulate(true);
  ASSERT_EQ(forward.size(), 1U);
  ASSERT_EQ(reversed.size(), 1U);
  EXPECT_EQ(forward.front().hit_observations, 3);
  EXPECT_EQ(forward.front().hit_observations, reversed.front().hit_observations);
  EXPECT_EQ(forward.front().miss_observations, reversed.front().miss_observations);
  EXPECT_DOUBLE_EQ(forward.front().position.x(), reversed.front().position.x());
  EXPECT_DOUBLE_EQ(forward.front().position.y(), reversed.front().position.y());
  EXPECT_DOUBLE_EQ(forward.front().position.z(), reversed.front().position.z());
  EXPECT_DOUBLE_EQ(
    forward.front().viewpoint_distance_m,
    reversed.front().viewpoint_distance_m);
  EXPECT_NEAR(forward.front().viewpoint_distance_m, 0.8, 1.0e-12);
  EXPECT_EQ(forward.front().confirmed, reversed.front().confirmed);
  EXPECT_TRUE(forward.front().confirmed);
}

TEST(TemporalStaticMap, ResetSupportsPoseGraphReprojectionRebuild)
{
  TemporalStaticMapAccumulator accumulator(testConfig());
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d(0.0, 0.5 * index, 0.0),
      {Eigen::Vector3d(2.1, 0.1, 0.1)});
  }
  ASSERT_EQ(accumulator.points(TemporalStaticMapSelection::kConfirmed).size(), 1U);

  accumulator.reset();
  for (int index = 0; index < 3; ++index) {
    accumulator.addObservation(
      static_cast<double>(index),
      Eigen::Vector3d(1.0, 0.5 * index, 0.0),
      {Eigen::Vector3d(3.1, 0.1, 0.1)});
  }

  const auto confirmed = accumulator.points(TemporalStaticMapSelection::kConfirmed);
  ASSERT_EQ(confirmed.size(), 1U);
  EXPECT_NEAR(confirmed.front().position.x(), 3.1, 1.0e-12);
  EXPECT_EQ(accumulator.stats().observations, 3U);
}

}  // namespace
}  // namespace slam_3d_core
}  // namespace gole
