// Copyright 2026 Gole Robotics

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Geometry>

#include "slam_3d_core/indoor_terrain_evidence.hpp"

namespace
{

using gole::slam_3d_core::IndoorTerrainConfig;
using gole::slam_3d_core::TerrainAccumulatorConfig;
using gole::slam_3d_core::TerrainEvidenceAccumulator;
using gole::slam_3d_core::TerrainLabel;
using gole::slam_3d_core::TerrainSupportState;
using gole::slam_3d_core::classifyIndoorTerrain;
using gole::slam_3d_core::gravityAlignmentRotation;
using gole::slam_3d_core::terrainLabelColor;
using gole::slam_3d_core::terrainSupportColor;

IndoorTerrainConfig testConfig()
{
  IndoorTerrainConfig config;
  config.enabled = true;
  config.azimuth_bins = 36;
  config.radial_bin_size_m = 0.5;
  config.min_range_m = 0.5;
  config.max_range_m = 8.0;
  config.seed_max_range_m = 2.0;
  config.min_points_per_bin = 2;
  config.floor_quantile = 0.2;
  config.max_seed_height_delta_m = 0.45;
  config.ground_band_m = 0.06;
  config.max_adjacent_gap_m = 0.55;
  config.floor_max_slope_deg = 4.0;
  config.ramp_min_slope_deg = 5.0;
  config.ramp_max_slope_deg = 22.0;
  config.ramp_min_run_bins = 3;
  config.step_min_height_m = 0.08;
  config.step_max_height_m = 0.30;
  config.drop_min_height_m = 0.12;
  config.drop_max_height_m = 0.60;
  config.transition_support_bins = 2;
  return config;
}

void appendRadialBin(
  std::vector<Eigen::Vector3d> & points,
  const int radial_index,
  const double z_m)
{
  const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.5;
  points.emplace_back(x_m, 0.01, z_m);
  points.emplace_back(x_m, 0.02, z_m + 0.005);
  points.emplace_back(x_m, 0.03, z_m + 0.01);
}

void appendRadialBinAtResolution(
  std::vector<Eigen::Vector3d> & points,
  const int radial_index,
  const double radial_bin_size_m,
  const double z_m)
{
  const double x_m =
    (static_cast<double>(radial_index) + 0.5) * radial_bin_size_m;
  points.emplace_back(x_m, 0.001, z_m);
  points.emplace_back(x_m, 0.002, z_m + 0.001);
  points.emplace_back(x_m, 0.003, z_m + 0.002);
}

void appendPlanarRampRay(
  std::vector<Eigen::Vector3d> & points,
  const double azimuth_deg,
  const double slope_rad)
{
  const double azimuth_rad = azimuth_deg * M_PI / 180.0;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    const double radius_m = (static_cast<double>(radial_index) + 0.5) * 0.5;
    for (const double lateral_jitter_m : {-0.008, 0.0, 0.008}) {
      const double x_m = radius_m * std::cos(azimuth_rad) -
        lateral_jitter_m * std::sin(azimuth_rad);
      const double y_m = radius_m * std::sin(azimuth_rad) +
        lateral_jitter_m * std::cos(azimuth_rad);
      points.emplace_back(x_m, y_m, std::tan(slope_rad) * (x_m - 0.75));
    }
  }
}

std::size_t countLabel(
  const std::vector<TerrainLabel> & labels,
  const TerrainLabel wanted)
{
  return static_cast<std::size_t>(std::count(labels.begin(), labels.end(), wanted));
}

TEST(IndoorTerrainEvidence, FlatProfileStaysFloor)
{
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, 0.0);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    testConfig());

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_GT(countLabel(result.labels, TerrainLabel::kFloor), 20U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kRamp), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
}

TEST(IndoorTerrainEvidence, GravityAlignmentMakesTiltedFloorInvariant)
{
  std::vector<Eigen::Vector3d> level_points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(level_points, radial_index, 0.0);
  }
  const Eigen::Matrix3d base_R_gravity =
    Eigen::AngleAxisd(10.0 * M_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d up_base = base_R_gravity * Eigen::Vector3d::UnitZ();
  const auto gravity_R_base = gravityAlignmentRotation(up_base);
  ASSERT_TRUE(gravity_R_base.has_value());

  std::vector<Eigen::Vector3d> tilted_points;
  tilted_points.reserve(level_points.size());
  for (const auto & point : level_points) {
    tilted_points.push_back(*gravity_R_base * (base_R_gravity * point));
  }
  const auto result = classifyIndoorTerrain(
    tilted_points,
    Eigen::Vector3d::Zero(),
    testConfig());

  EXPECT_GT(countLabel(result.labels, TerrainLabel::kFloor), 20U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kRamp), 0U);
}

TEST(IndoorTerrainEvidence, CeilingOnlyCloudCannotSeedFloor)
{
  auto config = testConfig();
  config.max_profile_height_above_base_m = 0.5;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, 2.0);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_FALSE(result.stats.seed_valid);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kFloor), 0U);
}

TEST(IndoorTerrainEvidence, DispersedSeedCandidatesDoNotForceFloorSeed)
{
  auto config = testConfig();
  config.seed_height_cluster_tolerance_m = 0.10;
  config.seed_min_cluster_bins = 3;
  std::vector<Eigen::Vector3d> points;
  const std::vector<double> heights{-0.40, 0.0, 0.40};
  for (std::size_t index = 0U; index < heights.size(); ++index) {
    appendRadialBin(points, static_cast<int>(index) + 1, heights[index]);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_FALSE(result.stats.seed_valid);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kFloor), 0U);
}

TEST(IndoorTerrainEvidence, SlidingSeedClusterIsFound)
{
  auto config = testConfig();
  config.seed_reference_height_m = 0.14;
  config.seed_max_range_m = 2.5;
  config.max_seed_height_delta_m = 0.5;
  config.seed_height_cluster_tolerance_m = 0.10;
  config.seed_min_cluster_bins = 3;
  std::vector<Eigen::Vector3d> points;
  const std::vector<double> heights{0.0, 0.09, 0.18, 0.19};
  for (std::size_t index = 0U; index < heights.size(); ++index) {
    appendRadialBin(points, static_cast<int>(index) + 1, heights[index]);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_NEAR(result.stats.seed_floor_z_m, 0.09, 0.02);
}

TEST(IndoorTerrainEvidence, ShelfAboveAbsoluteSeedGateCannotJoinFloorProfile)
{
  auto config = testConfig();
  config.max_profile_height_above_base_m = 0.35;
  config.max_seed_height_delta_m = 0.45;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, 0.0);
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.5;
    points.emplace_back(x_m, 0.60, 0.40);
    points.emplace_back(x_m, 0.61, 0.405);
    points.emplace_back(x_m, 0.62, 0.41);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  ASSERT_TRUE(result.stats.seed_valid);
  for (std::size_t index = 0U; index < points.size(); ++index) {
    if (points[index].z() >= 0.35) {
      EXPECT_EQ(result.labels[index], TerrainLabel::kUnknown);
    }
  }
}

TEST(IndoorTerrainEvidence, CoherentInclineIsRamp)
{
  auto config = testConfig();
  config.max_profile_height_above_base_m = 0.8;
  config.seed_height_cluster_tolerance_m = 0.20;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 10.0 * M_PI / 180.0;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.5;
    appendRadialBin(points, radial_index, std::tan(slope_rad) * (x_m - 0.75));
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    config);

  EXPECT_GT(countLabel(result.labels, TerrainLabel::kRamp), 20U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
  EXPECT_LT(result.stats.max_roughness_score, 0.1);
}

TEST(IndoorTerrainEvidence, WidePlanarInclinePassesGravityAlignedPlaneValidation)
{
  auto config = testConfig();
  config.max_profile_height_above_base_m = 0.9;
  config.seed_height_cluster_tolerance_m = 0.20;
  config.ramp_plane_validation_enabled = true;
  config.ramp_plane_neighborhood_radius_m = 0.90;
  config.ramp_plane_min_points = 12;
  config.ramp_plane_min_length_m = 0.50;
  config.ramp_plane_min_width_m = 0.20;
  config.ramp_plane_max_rmse_m = 0.02;
  config.ramp_plane_max_outlier_distance_m = 0.04;
  config.ramp_plane_min_inlier_ratio = 0.80;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 10.0 * M_PI / 180.0;
  for (const double azimuth_deg : {-12.0, -4.0, 4.0, 12.0}) {
    appendPlanarRampRay(points, azimuth_deg, slope_rad);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d::Zero(),
    config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_GT(result.stats.ramp_plane_candidate_points, 40U);
  EXPECT_GT(result.stats.ramp_plane_validated_points, 30U);
  EXPECT_GT(countLabel(result.labels, TerrainLabel::kRamp), 30U);
  EXPECT_EQ(
    result.stats.ramp_plane_candidate_points,
    result.stats.ramp_plane_validated_points + result.stats.ramp_plane_rejected_points);
  EXPECT_LE(result.stats.max_accepted_ramp_plane_rmse_m, 0.02);
}

TEST(IndoorTerrainEvidence, RadialOnlyInclineFailsPlaneWidthValidation)
{
  auto config = testConfig();
  config.max_profile_height_above_base_m = 0.8;
  config.seed_height_cluster_tolerance_m = 0.20;
  config.ramp_plane_validation_enabled = true;
  config.ramp_plane_neighborhood_radius_m = 0.90;
  config.ramp_plane_min_points = 9;
  config.ramp_plane_min_length_m = 0.50;
  config.ramp_plane_min_width_m = 0.20;
  config.ramp_plane_max_rmse_m = 0.05;
  config.ramp_plane_max_outlier_distance_m = 0.08;
  config.ramp_plane_min_inlier_ratio = 0.70;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 10.0 * M_PI / 180.0;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.5;
    appendRadialBin(points, radial_index, std::tan(slope_rad) * (x_m - 0.75));
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d::Zero(),
    config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_GT(result.stats.ramp_plane_candidate_points, 20U);
  EXPECT_EQ(result.stats.ramp_plane_validated_points, 0U);
  EXPECT_EQ(
    result.stats.ramp_plane_rejected_points,
    result.stats.ramp_plane_candidate_points);
  EXPECT_GT(result.stats.ramp_plane_rejected_width, 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kRamp), 0U);
}

TEST(IndoorTerrainEvidence, SemanticColorsAreStableAndDistinct)
{
  EXPECT_EQ(terrainLabelColor(TerrainLabel::kFloor), (std::array<std::uint8_t, 3U>{105, 190, 105}));
  EXPECT_EQ(terrainLabelColor(TerrainLabel::kRamp), (std::array<std::uint8_t, 3U>{45, 165, 255}));
  EXPECT_EQ(
    terrainLabelColor(TerrainLabel::kPositiveStep),
    (std::array<std::uint8_t, 3U>{255, 165, 35}));
  EXPECT_EQ(
    terrainLabelColor(TerrainLabel::kObservedDrop),
    (std::array<std::uint8_t, 3U>{235, 55, 55}));
  EXPECT_EQ(
    terrainLabelColor(TerrainLabel::kStaircase),
    (std::array<std::uint8_t, 3U>{190, 90, 245}));
}

TEST(IndoorTerrainEvidence, SupportConfidenceColorsRemainDistinct)
{
  EXPECT_EQ(
    terrainSupportColor(TerrainLabel::kFloor, TerrainSupportState::kConfirmedSupport),
    (std::array<std::uint8_t, 3U>{40, 170, 70}));
  EXPECT_EQ(
    terrainSupportColor(TerrainLabel::kFloor, TerrainSupportState::kProbableSupport),
    (std::array<std::uint8_t, 3U>{175, 225, 85}));
  EXPECT_EQ(
    terrainSupportColor(TerrainLabel::kUnknown, TerrainSupportState::kProbableHazard),
    (std::array<std::uint8_t, 3U>{245, 205, 55}));
  EXPECT_EQ(
    terrainSupportColor(TerrainLabel::kObservedDrop, TerrainSupportState::kConfirmedHazard),
    terrainLabelColor(TerrainLabel::kObservedDrop));
}

TEST(IndoorTerrainEvidence, SubStepBumpProducesAdvanceRoughnessPreview)
{
  auto config = testConfig();
  config.roughness_preview_enabled = true;
  config.roughness_min_height_m = 0.025;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, radial_index == 6 ? 0.04 : 0.0);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
  EXPECT_GE(result.stats.max_roughness_score, 1.5);
  EXPECT_GT(result.stats.roughness_preview_points, 0U);
}

TEST(IndoorTerrainEvidence, ShortInclineAtStairResolutionIsNotRamp)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.10;
  config.max_adjacent_gap_m = 0.35;
  config.ramp_min_run_bins = 3;
  config.ramp_min_run_length_m = 0.50;
  config.min_range_m = 0.5;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 10.0 * M_PI / 180.0;
  for (int radial_index = 5; radial_index <= 7; ++radial_index) {
    const double run_m = static_cast<double>(radial_index - 5) * 0.10;
    appendRadialBinAtResolution(
      points, radial_index, config.radial_bin_size_m, std::tan(slope_rad) * run_m);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kRamp), 0U);
}

TEST(IndoorTerrainEvidence, LongInclineAtStairResolutionIsRamp)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.10;
  config.max_adjacent_gap_m = 0.35;
  config.ramp_min_run_bins = 3;
  config.ramp_min_run_length_m = 0.50;
  config.min_range_m = 0.5;
  config.seed_height_cluster_tolerance_m = 0.15;
  config.max_profile_height_above_base_m = 0.8;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 10.0 * M_PI / 180.0;
  for (int radial_index = 5; radial_index <= 12; ++radial_index) {
    const double run_m = static_cast<double>(radial_index - 5) * 0.10;
    appendRadialBinAtResolution(
      points, radial_index, config.radial_bin_size_m, std::tan(slope_rad) * run_m);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_GT(countLabel(result.labels, TerrainLabel::kRamp), 15U);
}

TEST(IndoorTerrainEvidence, SupportedUpwardDiscontinuityIsPositiveStep)
{
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, radial_index < 6 ? 0.0 : 0.16);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    testConfig());

  EXPECT_GT(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
}

TEST(IndoorTerrainEvidence, DirectlyObservedLowerFloorIsDropEvidence)
{
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 10; ++radial_index) {
    appendRadialBin(points, radial_index, radial_index < 6 ? 0.0 : -0.20);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    testConfig());

  EXPECT_GT(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
}

TEST(IndoorTerrainEvidence, RepeatedAscendingRisersAreStaircase)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.25;
  config.max_adjacent_gap_m = 0.30;
  config.seed_max_range_m = 1.25;
  config.staircase_min_risers = 3;
  config.staircase_max_profile_height_above_base_m = 2.0;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 2; radial_index <= 10; ++radial_index) {
    const int ascent_count = std::max(0, (radial_index - 3) / 2);
    const double z_m = 0.18 * static_cast<double>(ascent_count);
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.25;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_NEAR(result.stats.seed_floor_z_m, 0.0, 0.02);
  EXPECT_GE(countLabel(result.labels, TerrainLabel::kStaircase), 9U);
  EXPECT_EQ(
    result.stats.staircase_points,
    countLabel(result.labels, TerrainLabel::kStaircase));
}

TEST(IndoorTerrainEvidence, RepeatedDescendingRisersKeepCurrentFloorSeed)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.25;
  config.max_adjacent_gap_m = 0.30;
  config.seed_max_range_m = 3.5;
  config.floor_quantile = 0.05;
  config.seed_reference_height_m = 0.0;
  config.seed_height_cluster_tolerance_m = 0.08;
  config.seed_min_cluster_bins = 3;
  config.staircase_min_risers = 3;
  config.staircase_max_profile_height_above_base_m = 2.0;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 2; radial_index <= 16; ++radial_index) {
    const int descent_count = std::min(3, std::max(0, (radial_index - 3) / 2));
    const double z_m = -0.18 * static_cast<double>(descent_count);
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.25;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_NEAR(result.stats.seed_floor_z_m, 0.0, 0.02);
  EXPECT_GE(countLabel(result.labels, TerrainLabel::kStaircase), 9U);
}

TEST(IndoorTerrainEvidence, ShortTreadsRemainResolvableAtIndoorProfileResolution)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.10;
  config.max_adjacent_gap_m = 0.15;
  config.seed_max_range_m = 0.8;
  config.transition_support_bins = 2;
  config.staircase_min_risers = 3;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 4; radial_index <= 18; ++radial_index) {
    const int ascent_count = std::min(3, std::max(0, (radial_index - 6) / 3));
    const double z_m = 0.17 * static_cast<double>(ascent_count);
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.10;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  ASSERT_TRUE(result.stats.seed_valid);
  EXPECT_GE(countLabel(result.labels, TerrainLabel::kStaircase), 18U);
}

TEST(IndoorTerrainEvidence, UnsupportedRiserSequenceIsNotStaircase)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.10;
  config.max_adjacent_gap_m = 0.35;
  config.seed_max_range_m = 0.8;
  config.transition_support_bins = 2;
  config.staircase_min_risers = 3;
  std::vector<Eigen::Vector3d> points;
  const std::vector<double> levels{0.0, 0.17, 0.34, 0.51};
  for (int radial_index = 4; radial_index <= 7; ++radial_index) {
    const double z_m = levels[static_cast<std::size_t>(radial_index - 4)];
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.10;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
}

TEST(IndoorTerrainEvidence, TwoRisersRemainIndividualSteps)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.25;
  config.max_adjacent_gap_m = 0.30;
  config.seed_max_range_m = 1.25;
  config.staircase_min_risers = 3;
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 2; radial_index <= 8; ++radial_index) {
    const int ascent_count = std::min(2, std::max(0, (radial_index - 3) / 2));
    const double z_m = 0.16 * static_cast<double>(ascent_count);
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.25;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
  EXPECT_GT(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
}

TEST(IndoorTerrainEvidence, IrregularRepeatedRisersAreNotStaircase)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.25;
  config.max_adjacent_gap_m = 0.30;
  config.seed_max_range_m = 1.25;
  config.staircase_min_risers = 3;
  config.staircase_max_riser_height_deviation_m = 0.03;
  const std::vector<double> levels{0.0, 0.13, 0.36, 0.50};
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 2; radial_index <= 10; ++radial_index) {
    const int level_index = std::min(3, std::max(0, (radial_index - 3) / 2));
    const double z_m = levels[static_cast<std::size_t>(level_index)];
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.25;
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
}

TEST(IndoorTerrainEvidence, InvalidDiscontinuityBreaksStaircaseRun)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.10;
  config.max_adjacent_gap_m = 0.15;
  config.seed_max_range_m = 0.8;
  config.transition_support_bins = 2;
  config.staircase_min_risers = 3;
  std::vector<Eigen::Vector3d> points;
  const std::vector<double> levels{
    0.0, 0.0, 0.18, 0.18, 0.63, 0.63, 0.81, 0.81, 0.99, 0.99};
  for (std::size_t index = 0U; index < levels.size(); ++index) {
    const int radial_index = static_cast<int>(index) + 4;
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.10;
    points.emplace_back(x_m, 0.01, levels[index]);
    points.emplace_back(x_m, 0.02, levels[index] + 0.005);
    points.emplace_back(x_m, 0.03, levels[index] + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
}

TEST(IndoorTerrainEvidence, SteepContinuousInclineIsNotStaircase)
{
  auto config = testConfig();
  config.radial_bin_size_m = 0.25;
  config.max_adjacent_gap_m = 0.30;
  config.seed_max_range_m = 1.0;
  config.max_profile_height_above_base_m = 2.0;
  config.staircase_max_profile_height_above_base_m = 2.0;
  std::vector<Eigen::Vector3d> points;
  constexpr double slope_rad = 25.0 * M_PI / 180.0;
  for (int radial_index = 2; radial_index <= 10; ++radial_index) {
    const double x_m = (static_cast<double>(radial_index) + 0.5) * 0.25;
    const double z_m = std::tan(slope_rad) * (x_m - 0.625);
    points.emplace_back(x_m, 0.01, z_m);
    points.emplace_back(x_m, 0.02, z_m + 0.005);
    points.emplace_back(x_m, 0.03, z_m + 0.01);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
}

TEST(IndoorTerrainEvidence, MissingReturnsAloneNeverBecomeDropEvidence)
{
  std::vector<Eigen::Vector3d> points;
  for (int radial_index = 1; radial_index <= 4; ++radial_index) {
    appendRadialBin(points, radial_index, 0.0);
  }
  for (int radial_index = 8; radial_index <= 11; ++radial_index) {
    appendRadialBin(points, radial_index, -0.20);
  }

  const auto result = classifyIndoorTerrain(
    points,
    Eigen::Vector3d(0.0, 0.0, 1.0),
    testConfig());

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
}

TEST(IndoorTerrainEvidence, SteepWallLikeProfileRemainsUnknown)
{
  auto config = testConfig();
  std::vector<Eigen::Vector3d> points;
  for (int radial = 1; radial <= 5; ++radial) {
    const double height_m = static_cast<double>(radial - 1) * 0.35;
    appendRadialBin(points, radial, height_m);
  }

  const auto result = classifyIndoorTerrain(points, Eigen::Vector3d::Zero(), config);

  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kFloor), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kRamp), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kPositiveStep), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kObservedDrop), 0U);
  EXPECT_EQ(countLabel(result.labels, TerrainLabel::kStaircase), 0U);
}

TEST(TerrainEvidenceAccumulator, RequiresIndependentViewpoints)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.5;
  TerrainEvidenceAccumulator accumulator(config);
  const std::vector<Eigen::Vector3d> points{Eigen::Vector3d(2.0, 0.0, 0.0)};
  const std::vector<TerrainLabel> labels{TerrainLabel::kObservedDrop};

  accumulator.addObservation(points, labels, Eigen::Vector3d(0.0, 0.0, 1.0));
  accumulator.addObservation(points, labels, Eigen::Vector3d(0.1, 0.0, 1.0));
  EXPECT_TRUE(accumulator.confirmedEvidence().empty());

  accumulator.addObservation(points, labels, Eigen::Vector3d(0.6, 0.0, 1.0));
  const auto confirmed = accumulator.confirmedEvidence();
  ASSERT_EQ(confirmed.size(), 1U);
  EXPECT_EQ(confirmed.front().label, TerrainLabel::kObservedDrop);
  EXPECT_EQ(confirmed.front().confirmations, 2);
}

TEST(TerrainEvidenceAccumulator, ContradictoryEvidenceCanRevokeHazard)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.min_label_confidence = 0.67;
  TerrainEvidenceAccumulator accumulator(config);
  const std::vector<Eigen::Vector3d> points{Eigen::Vector3d(2.0, 0.0, 0.0)};

  accumulator.addObservation(
    points, {TerrainLabel::kObservedDrop}, Eigen::Vector3d(0.0, 0.0, 0.0));
  accumulator.addObservation(
    points, {TerrainLabel::kObservedDrop}, Eigen::Vector3d(0.3, 0.0, 0.0));
  ASSERT_EQ(accumulator.confirmedEvidence().size(), 1U);

  accumulator.addObservation(
    points, {TerrainLabel::kFloor}, Eigen::Vector3d(0.6, 0.0, 0.0));
  accumulator.addObservation(
    points, {TerrainLabel::kFloor}, Eigen::Vector3d(0.9, 0.0, 0.0));
  accumulator.addObservation(
    points, {TerrainLabel::kFloor}, Eigen::Vector3d(1.2, 0.0, 0.0));
  accumulator.addObservation(
    points, {TerrainLabel::kFloor}, Eigen::Vector3d(1.5, 0.0, 0.0));
  accumulator.addObservation(
    points, {TerrainLabel::kFloor}, Eigen::Vector3d(1.8, 0.0, 0.0));

  const auto evidence = accumulator.confirmedEvidence();
  ASSERT_EQ(evidence.size(), 1U);
  EXPECT_EQ(evidence.front().label, TerrainLabel::kFloor);
  EXPECT_EQ(evidence.front().observations, 7);
  EXPECT_NEAR(evidence.front().confidence, 5.0 / 7.0, 1.0e-12);
  EXPECT_EQ(accumulator.stats().conflicted_voxels, 1U);
}

TEST(TerrainEvidenceAccumulator, AmbiguousViewpointCastsNoVote)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.5;
  config.min_confirmations = 1;
  config.min_viewpoint_distance_m = 0.25;
  config.min_view_label_confidence = 0.6;
  config.min_label_confidence = 0.5;
  TerrainEvidenceAccumulator accumulator(config);

  accumulator.addObservation(
    {Eigen::Vector3d(2.01, 0.01, 0.01), Eigen::Vector3d(2.02, 0.02, 0.02)},
    {TerrainLabel::kFloor, TerrainLabel::kObservedDrop},
    Eigen::Vector3d::Zero());

  const auto evidence = accumulator.confirmedEvidence();
  EXPECT_TRUE(evidence.empty());
  EXPECT_EQ(accumulator.stats().conflicted_voxels, 0U);
  EXPECT_EQ(accumulator.stats().ambiguous_view_observations, 1U);
}

TEST(TerrainEvidenceAccumulator, DominantLabelCastsOneVotePerViewpoint)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.5;
  config.min_confirmations = 1;
  config.min_viewpoint_distance_m = 0.25;
  config.min_view_label_confidence = 0.6;
  config.min_label_confidence = 0.5;
  TerrainEvidenceAccumulator accumulator(config);

  accumulator.addObservation(
    {
      Eigen::Vector3d(2.01, 0.01, 0.01),
      Eigen::Vector3d(2.02, 0.02, 0.02),
      Eigen::Vector3d(2.03, 0.03, 0.03)},
    {TerrainLabel::kFloor, TerrainLabel::kObservedDrop, TerrainLabel::kObservedDrop},
    Eigen::Vector3d::Zero());

  const auto evidence = accumulator.confirmedEvidence();
  ASSERT_EQ(evidence.size(), 1U);
  EXPECT_EQ(evidence.front().label, TerrainLabel::kObservedDrop);
  EXPECT_EQ(evidence.front().confirmations, 1);
  EXPECT_EQ(evidence.front().observations, 1);
  EXPECT_EQ(accumulator.stats().ambiguous_view_observations, 0U);
}

TEST(TerrainEvidenceAccumulator, PreservesMeanGeometricRoughnessByLabel)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.5;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  TerrainEvidenceAccumulator accumulator(config);
  const std::vector<Eigen::Vector3d> points{Eigen::Vector3d(2.0, 0.0, 0.0)};
  const std::vector<TerrainLabel> labels{TerrainLabel::kFloor};

  accumulator.addObservation(
    points, labels, {1.5}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    points, labels, {2.5}, Eigen::Vector3d(0.3, 0.0, 0.0));

  const auto evidence = accumulator.confirmedEvidence();
  ASSERT_EQ(evidence.size(), 1U);
  EXPECT_EQ(evidence.front().label, TerrainLabel::kFloor);
  EXPECT_NEAR(evidence.front().geometric_roughness_score, 2.0, 1.0e-12);
}

TEST(TerrainEvidenceAccumulator, ConnectedSingleViewFloorBecomesProbableSupport)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.min_label_confidence = 0.67;
  config.probable_support_min_confidence = 0.60;
  config.probable_support_max_roughness_score = 1.5;
  config.probable_support_require_confirmed_connectivity = true;
  TerrainEvidenceAccumulator accumulator(config);

  const std::vector<Eigen::Vector3d> seed{Eigen::Vector3d(0.05, 0.05, 0.0)};
  accumulator.addObservation(seed, {TerrainLabel::kFloor}, {0.0}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    seed, {TerrainLabel::kFloor}, {0.0}, Eigen::Vector3d(0.3, 0.0, 0.0));
  accumulator.addObservation(
    {Eigen::Vector3d(0.25, 0.05, 0.0)},
    {TerrainLabel::kFloor},
    {0.0},
    Eigen::Vector3d(0.6, 0.0, 0.0));

  const auto support = accumulator.supportEvidence();
  ASSERT_EQ(support.size(), 2U);
  const auto probable = std::find_if(
    support.begin(), support.end(), [](const auto & point) {
      return point.state == TerrainSupportState::kProbableSupport;
    });
  ASSERT_NE(probable, support.end());
  EXPECT_EQ(probable->label, TerrainLabel::kFloor);
  EXPECT_EQ(probable->confirmations, 1);
  EXPECT_DOUBLE_EQ(probable->support_probability, 1.0);
  EXPECT_GT(probable->geometric_traversability_score, 0.0);
  EXPECT_LT(probable->geometric_traversability_score, 1.0);
  EXPECT_FALSE(probable->connectivity_rejected);
  EXPECT_EQ(accumulator.stats().probable_support_voxels, 1U);
  EXPECT_EQ(accumulator.stats().confirmed_support_voxels, 1U);
}

TEST(TerrainEvidenceAccumulator, ConnectivityWorksAcrossNegativeBucketBoundary)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.1;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.probable_support_max_anchor_distance_m = 0.45;
  config.probable_support_max_neighbor_height_delta_m = 0.16;
  config.probable_support_require_confirmed_connectivity = true;
  TerrainEvidenceAccumulator accumulator(config);

  const std::vector<Eigen::Vector3d> anchor{Eigen::Vector3d(0.41, 0.01, 0.0)};
  accumulator.addObservation(anchor, {TerrainLabel::kFloor}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    anchor, {TerrainLabel::kFloor}, Eigen::Vector3d(0.3, 0.0, 0.0));
  accumulator.addObservation(
    {Eigen::Vector3d(-0.01, 0.01, 0.0)},
    {TerrainLabel::kFloor},
    Eigen::Vector3d(0.6, 0.0, 0.0));

  const auto support = accumulator.supportEvidence();
  const auto probable = std::find_if(
    support.begin(), support.end(), [](const auto & point) {
      return point.state == TerrainSupportState::kProbableSupport;
    });
  ASSERT_NE(probable, support.end());
  EXPECT_FALSE(probable->connectivity_rejected);
}

TEST(TerrainEvidenceAccumulator, SupportConnectivityRebuildScalesToIndoorMap)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.1;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.probable_support_max_anchor_distance_m = 0.45;
  config.probable_support_require_confirmed_connectivity = true;
  TerrainEvidenceAccumulator accumulator(config);

  std::vector<Eigen::Vector3d> anchors;
  std::vector<Eigen::Vector3d> probable_points;
  anchors.reserve(10000U);
  probable_points.reserve(10000U);
  for (int x = 0; x < 100; ++x) {
    for (int y = 0; y < 100; ++y) {
      const Eigen::Vector3d anchor(
        0.02 + 0.6 * static_cast<double>(x),
        0.02 + 0.6 * static_cast<double>(y),
        0.0);
      anchors.push_back(anchor);
      probable_points.push_back(anchor + Eigen::Vector3d(0.2, 0.0, 0.0));
    }
  }
  const std::vector<TerrainLabel> labels(anchors.size(), TerrainLabel::kFloor);

  accumulator.addObservation(anchors, labels, Eigen::Vector3d::Zero());
  accumulator.addObservation(anchors, labels, Eigen::Vector3d(0.3, 0.0, 0.0));
  accumulator.addObservation(
    probable_points, labels, Eigen::Vector3d(0.6, 0.0, 0.0));

  const auto stats = accumulator.stats();
  EXPECT_EQ(stats.confirmed_support_voxels, 10000U);
  EXPECT_EQ(stats.probable_support_voxels, 10000U);
  EXPECT_EQ(stats.isolated_probable_support_voxels, 0U);
  EXPECT_LT(stats.support_rebuild_runtime_ms, 1000.0);
}

TEST(TerrainEvidenceAccumulator, IsolatedSingleViewFloorRemainsUnresolved)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.probable_support_require_confirmed_connectivity = true;
  TerrainEvidenceAccumulator accumulator(config);

  const std::vector<Eigen::Vector3d> seed{Eigen::Vector3d(0.05, 0.05, 0.0)};
  accumulator.addObservation(seed, {TerrainLabel::kFloor}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    seed, {TerrainLabel::kFloor}, Eigen::Vector3d(0.3, 0.0, 0.0));
  accumulator.addObservation(
    {Eigen::Vector3d(2.05, 0.05, 0.0)},
    {TerrainLabel::kFloor},
    Eigen::Vector3d(0.6, 0.0, 0.0));

  const auto support = accumulator.supportEvidence();
  const auto isolated = std::find_if(
    support.begin(), support.end(), [](const auto & point) {
      return point.connectivity_rejected;
    });
  ASSERT_NE(isolated, support.end());
  EXPECT_EQ(isolated->state, TerrainSupportState::kUnresolved);
  EXPECT_DOUBLE_EQ(isolated->geometric_traversability_score, 0.0);
  EXPECT_EQ(accumulator.stats().isolated_probable_support_voxels, 1U);
}

TEST(TerrainEvidenceAccumulator, FloorRampSemanticAmbiguityKeepsSupportVote)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.min_view_label_confidence = 0.67;
  config.min_label_confidence = 0.67;
  config.probable_support_min_confidence = 0.60;
  config.probable_support_require_confirmed_connectivity = false;
  TerrainEvidenceAccumulator accumulator(config);

  accumulator.addObservation(
    {Eigen::Vector3d(0.05, 0.05, 0.0), Eigen::Vector3d(0.06, 0.05, 0.0)},
    {TerrainLabel::kFloor, TerrainLabel::kRamp},
    {0.1, 0.1},
    Eigen::Vector3d::Zero());

  EXPECT_TRUE(accumulator.evidence().empty());
  const auto support = accumulator.supportEvidence();
  ASSERT_EQ(support.size(), 1U);
  EXPECT_EQ(support.front().state, TerrainSupportState::kProbableSupport);
  EXPECT_EQ(support.front().label, TerrainLabel::kUnknown);
  EXPECT_DOUBLE_EQ(support.front().support_probability, 1.0);
  EXPECT_EQ(accumulator.stats().ambiguous_view_observations, 1U);
}

TEST(TerrainEvidenceAccumulator, SupportHazardConflictNeverBecomesTraversable)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.probable_support_require_confirmed_connectivity = false;
  TerrainEvidenceAccumulator accumulator(config);
  const std::vector<Eigen::Vector3d> point{Eigen::Vector3d(0.05, 0.05, 0.0)};

  accumulator.addObservation(point, {TerrainLabel::kFloor}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    point, {TerrainLabel::kPositiveStep}, Eigen::Vector3d(0.3, 0.0, 0.0));

  const auto support = accumulator.supportEvidence();
  ASSERT_EQ(support.size(), 1U);
  EXPECT_EQ(support.front().state, TerrainSupportState::kUnresolved);
  EXPECT_DOUBLE_EQ(support.front().support_probability, 0.5);
  EXPECT_DOUBLE_EQ(support.front().geometric_traversability_score, 0.0);
}

TEST(TerrainEvidenceAccumulator, MinorityHazardVoteVetoesProbableSupport)
{
  TerrainAccumulatorConfig config;
  config.voxel_size_m = 0.2;
  config.min_confirmations = 2;
  config.min_viewpoint_distance_m = 0.25;
  config.probable_support_require_confirmed_connectivity = false;
  TerrainEvidenceAccumulator accumulator(config);
  const std::vector<Eigen::Vector3d> point{Eigen::Vector3d(0.05, 0.05, 0.0)};

  accumulator.addObservation(point, {TerrainLabel::kFloor}, Eigen::Vector3d::Zero());
  accumulator.addObservation(
    point, {TerrainLabel::kPositiveStep}, Eigen::Vector3d(0.3, 0.0, 0.0));
  accumulator.addObservation(
    point, {TerrainLabel::kFloor}, Eigen::Vector3d(0.6, 0.0, 0.0));
  accumulator.addObservation(
    point, {TerrainLabel::kFloor}, Eigen::Vector3d(0.9, 0.0, 0.0));

  const auto support = accumulator.supportEvidence();
  ASSERT_EQ(support.size(), 1U);
  EXPECT_EQ(support.front().state, TerrainSupportState::kUnresolved);
  EXPECT_NEAR(support.front().support_probability, 0.75, 1.0e-12);
  EXPECT_DOUBLE_EQ(support.front().geometric_traversability_score, 0.0);
}

}  // namespace
