// Copyright 2026 Gole Robotics

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "slam_3d_core/voxel_surfel_registration.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace
{

using Registration = VoxelSurfelRegistration;
using CloudT = Registration::CloudT;
using PointT = Registration::PointT;
using Vector6d = Registration::Vector6d;

PointT point(const Eigen::Vector3d & value)
{
  return PointT(
    static_cast<float>(value.x()),
    static_cast<float>(value.y()),
    static_cast<float>(value.z()));
}

void finalizeCloud(CloudT & cloud)
{
  cloud.width = static_cast<std::uint32_t>(cloud.size());
  cloud.height = 1U;
  cloud.is_dense = true;
}

Eigen::Matrix3d rpy(double roll, double pitch, double yaw)
{
  return
    (
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
}

Eigen::Isometry3d pose(
  double roll,
  double pitch,
  double yaw,
  const Eigen::Vector3d & translation)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = rpy(roll, pitch, yaw);
  result.translation() = translation;
  return result;
}

double rotationError(
  const Eigen::Isometry3d & estimate,
  const Eigen::Isometry3d & truth)
{
  return Eigen::AngleAxisd(estimate.linear() * truth.linear().transpose()).angle();
}

double yaw(const Eigen::Isometry3d & value)
{
  return std::atan2(value.linear()(1, 0), value.linear()(0, 0));
}

CloudT transformedToSource(
  const CloudT & target,
  const Eigen::Isometry3d & target_T_source)
{
  CloudT source;
  source.reserve(target.size());
  const Eigen::Isometry3d source_T_target = target_T_source.inverse();
  for (const auto & target_point : target.points) {
    source.push_back(
      point(
        source_T_target *
        Eigen::Vector3d(target_point.x, target_point.y, target_point.z)));
  }
  finalizeCloud(source);
  return source;
}

void addHorizontalPlane(
  CloudT & cloud,
  double z,
  double minimum_x,
  double maximum_x,
  double minimum_y,
  double maximum_y,
  double spacing)
{
  for (double x = minimum_x; x <= maximum_x + 1.0e-9; x += spacing) {
    for (double y = minimum_y; y <= maximum_y + 1.0e-9; y += spacing) {
      cloud.push_back(point(Eigen::Vector3d(x, y, z)));
    }
  }
}

void addXPlane(
  CloudT & cloud,
  double x,
  double minimum_y,
  double maximum_y,
  double minimum_z,
  double maximum_z,
  double spacing)
{
  for (double y = minimum_y; y <= maximum_y + 1.0e-9; y += spacing) {
    for (double z = minimum_z; z <= maximum_z + 1.0e-9; z += spacing) {
      cloud.push_back(point(Eigen::Vector3d(x, y, z)));
    }
  }
}

void addYPlane(
  CloudT & cloud,
  double y,
  double minimum_x,
  double maximum_x,
  double minimum_z,
  double maximum_z,
  double spacing)
{
  for (double x = minimum_x; x <= maximum_x + 1.0e-9; x += spacing) {
    for (double z = minimum_z; z <= maximum_z + 1.0e-9; z += spacing) {
      cloud.push_back(point(Eigen::Vector3d(x, y, z)));
    }
  }
}

CloudT threeOrthogonalPlanes(double offset_x = 0.0, double offset_y = 0.0, double offset_z = 0.0)
{
  CloudT cloud;
  constexpr double kMinimum = -2.4;
  constexpr double kMaximum = 2.4;
  constexpr double kSpacing = 0.16;
  addHorizontalPlane(
    cloud,
    offset_z,
    offset_x + kMinimum,
    offset_x + kMaximum,
    offset_y + kMinimum,
    offset_y + kMaximum,
    kSpacing);
  addXPlane(
    cloud,
    offset_x,
    offset_y + kMinimum,
    offset_y + kMaximum,
    offset_z + kMinimum,
    offset_z + kMaximum,
    kSpacing);
  addYPlane(
    cloud,
    offset_y,
    offset_x + kMinimum,
    offset_x + kMaximum,
    offset_z + kMinimum,
    offset_z + kMaximum,
    kSpacing);
  finalizeCloud(cloud);
  return cloud;
}

Registration::Config baseConfig()
{
  Registration::Config config;
  config.voxel_size = 0.8;
  config.min_points_per_voxel = 8U;
  config.max_planarity_ratio = 0.08;
  config.min_tangent_variance = 1.0e-3;
  config.max_plane_distance = 0.45;
  config.max_center_distance = 1.35;
  config.min_correspondences = 80U;
  config.min_correspondence_ratio = 0.55;
  config.huber_delta = 0.08;
  config.max_iterations = 35;
  config.max_backtracking_steps = 10;
  config.backtracking_factor = 0.5;
  config.rotation_convergence_rad = 1.0e-7;
  config.translation_convergence_m = 1.0e-7;
  config.relative_cost_convergence = 1.0e-10;
  config.max_rotation_step_rad = 0.15;
  config.max_translation_step_m = 0.30;
  config.rotation_scale = 1.0;
  config.relative_eigen_threshold = 1.0e-5;
  config.min_observable_rank = 6;
  config.max_quality_rmse = 0.04;
  config.max_quality_p95 = 0.08;
  return config;
}

TEST(VoxelSurfelRegistrationMath, AnalyticJacobianMatchesCentralFiniteDifference)
{
  const Eigen::Vector3d source_point(1.2, -0.7, 0.4);
  const Eigen::Vector3d centroid(-0.3, 0.8, 1.1);
  const Eigen::Vector3d normal = Eigen::Vector3d(0.2, -0.5, 0.84).normalized();
  const Eigen::Isometry3d transform =
    pose(0.17, -0.11, 0.23, Eigen::Vector3d(0.4, -0.2, 0.7));

  const Registration::RowVector6d analytic =
    Registration::pointToPlaneJacobian(
    source_point, transform, normal);

  constexpr double kEpsilon = 1.0e-7;
  for (int axis = 0; axis < 6; ++axis) {
    Vector6d positive = Vector6d::Zero();
    Vector6d negative = Vector6d::Zero();
    positive[axis] = kEpsilon;
    negative[axis] = -kEpsilon;
    const double positive_residual = Registration::pointToPlaneResidual(
      source_point,
      Registration::applyIncrement(transform, positive),
      centroid,
      normal);
    const double negative_residual = Registration::pointToPlaneResidual(
      source_point,
      Registration::applyIncrement(transform, negative),
      centroid,
      normal);
    const double numerical =
      (positive_residual - negative_residual) / (2.0 * kEpsilon);
    EXPECT_NEAR(analytic[axis], numerical, 2.0e-8) << "axis=" << axis;
  }
}

TEST(VoxelSurfelRegistrationMath, AppliesRotationBeforeIndependentTranslation)
{
  const Eigen::Isometry3d initial =
    pose(0.1, -0.2, 0.3, Eigen::Vector3d(1.0, 2.0, 3.0));
  Vector6d increment = Vector6d::Zero();
  increment << 0.02, -0.03, 0.01, 0.4, -0.5, 0.6;

  const Eigen::Isometry3d updated =
    Registration::applyIncrement(initial, increment);
  const double angle = increment.head<3>().norm();
  const Eigen::Matrix3d increment_rotation =
    Eigen::AngleAxisd(angle, increment.head<3>() / angle).toRotationMatrix();

  EXPECT_TRUE(
    updated.linear().isApprox(increment_rotation * initial.linear(), 1.0e-12));
  EXPECT_TRUE(
    updated.translation().isApprox(
      initial.translation() + increment.tail<3>(), 1.0e-12));
}

TEST(VoxelSurfelRegistrationDegeneracy, SingleFloorHasRankThreeAndSuppressesXYYaw)
{
  CloudT target;
  addHorizontalPlane(target, 0.0, -3.0, 3.0, -3.0, 3.0, 0.14);
  finalizeCloud(target);

  const Eigen::Isometry3d truth =
    pose(0.045, -0.035, 0.12, Eigen::Vector3d(0.24, -0.18, 0.20));
  const CloudT source = transformedToSource(target, truth);

  Registration::Config config = baseConfig();
  config.min_observable_rank = 3;
  config.relative_eigen_threshold = 1.0e-4;
  config.max_quality_rmse = 0.02;
  config.max_quality_p95 = 0.03;
  const Registration registration(config);

  const Registration::Result result =
    registration.align(source, target, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.accepted)
    << result.reason << " rmse=" << result.rmse
    << " p95=" << result.p95_abs_residual
    << " rank=" << result.observable_rank
    << " t=" << result.target_T_source.translation().transpose();
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.observable_rank, 3);
  EXPECT_LT(result.rmse, 2.0e-3);
  EXPECT_NEAR(result.target_T_source.translation().x(), 0.0, 1.0e-8);
  EXPECT_NEAR(result.target_T_source.translation().y(), 0.0, 1.0e-8);
  EXPECT_NEAR(yaw(result.target_T_source), 0.0, 2.0e-3);
  EXPECT_GT(result.weak_rotation_axes.z(), 0.95);
  EXPECT_GT(result.weak_translation_axes.x(), 0.95);
  EXPECT_GT(result.weak_translation_axes.y(), 0.95);
  EXPECT_LT(result.weak_translation_axes.z(), 0.05);
}

TEST(VoxelSurfelRegistrationRecovery, ThreeOrthogonalPlanesRecoverFullRankTransform)
{
  const CloudT target = threeOrthogonalPlanes();
  const Eigen::Isometry3d truth =
    pose(0.035, -0.028, 0.048, Eigen::Vector3d(0.12, -0.09, 0.10));
  const CloudT source = transformedToSource(target, truth);
  const Registration registration(baseConfig());

  const Registration::Result result =
    registration.align(source, target, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.accepted)
    << result.reason << " rmse=" << result.rmse
    << " p95=" << result.p95_abs_residual
    << " rank=" << result.observable_rank
    << " t=" << result.target_T_source.translation().transpose();
  EXPECT_TRUE(result.converged);
  EXPECT_EQ(result.observable_rank, 6);
  EXPECT_LT((result.target_T_source.translation() - truth.translation()).norm(), 0.006);
  EXPECT_LT(rotationError(result.target_T_source, truth), 0.003);
  EXPECT_LT(result.rmse, 0.003);
  EXPECT_LT(result.p95_abs_residual, 0.006);
  EXPECT_LT(result.weak_state_vector.maxCoeff(), 0.05);
  EXPECT_GT(result.hessian_condition, 1.0);
  EXPECT_GT(result.total_time_ms, 0.0);
}

TEST(VoxelSurfelRegistrationDegeneracy, CorridorSuppressesLongitudinalTranslation)
{
  CloudT target;
  addHorizontalPlane(target, 0.0, -8.0, 8.0, -1.2, 1.2, 0.16);
  addYPlane(target, -2.0, -8.0, 8.0, 0.8, 3.2, 0.16);
  addYPlane(target, 2.0, -8.0, 8.0, 0.8, 3.2, 0.16);
  finalizeCloud(target);

  const Eigen::Isometry3d truth =
    pose(0.025, -0.018, 0.035, Eigen::Vector3d(0.28, -0.12, 0.10));
  const CloudT source = transformedToSource(target, truth);

  Registration::Config config = baseConfig();
  config.min_observable_rank = 5;
  config.relative_eigen_threshold = 1.0e-4;
  const Registration registration(config);
  const Registration::Result result =
    registration.align(source, target, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.accepted)
    << result.reason << " rmse=" << result.rmse
    << " p95=" << result.p95_abs_residual
    << " rank=" << result.observable_rank
    << " t=" << result.target_T_source.translation().transpose();
  EXPECT_EQ(result.observable_rank, 5);
  EXPECT_NEAR(result.target_T_source.translation().x(), 0.0, 1.0e-8);
  EXPECT_NEAR(result.target_T_source.translation().y(), truth.translation().y(), 0.006);
  EXPECT_NEAR(result.target_T_source.translation().z(), truth.translation().z(), 0.006);
  EXPECT_LT(rotationError(result.target_T_source, truth), 0.004);
  EXPECT_GT(result.weak_translation_axes.x(), 0.95);
  EXPECT_LT(result.weak_translation_axes.y(), 0.05);
  EXPECT_LT(result.weak_translation_axes.z(), 0.05);
}

TEST(VoxelSurfelRegistrationRobustness, HuberLimitsThirtyPercentDeterministicOutliers)
{
  const CloudT target = threeOrthogonalPlanes();
  const Eigen::Isometry3d truth =
    pose(0.025, -0.020, 0.030, Eigen::Vector3d(0.08, -0.06, 0.07));
  CloudT source = transformedToSource(target, truth);

  std::array<std::vector<Eigen::Vector3d>, 3> plane_samples;
  for (const auto & target_point : target.points) {
    if (
      std::abs(target_point.z) < 1.0e-6 &&
      std::abs(target_point.x) > 0.8 &&
      std::abs(target_point.y) > 0.8)
    {
      plane_samples[2].emplace_back(target_point.x, target_point.y, target_point.z);
    } else if (
      std::abs(target_point.x) < 1.0e-6 &&
      std::abs(target_point.y) > 0.8 &&
      std::abs(target_point.z) > 0.8)
    {
      plane_samples[0].emplace_back(target_point.x, target_point.y, target_point.z);
    } else if (
      std::abs(target_point.y) < 1.0e-6 &&
      std::abs(target_point.x) > 0.8 &&
      std::abs(target_point.z) > 0.8)
    {
      plane_samples[1].emplace_back(target_point.x, target_point.y, target_point.z);
    }
  }
  for (const auto & samples : plane_samples) {
    ASSERT_FALSE(samples.empty());
  }

  const std::size_t inlier_count = source.size();
  const std::size_t outlier_count =
    static_cast<std::size_t>(
    std::ceil(static_cast<double>(inlier_count) * 3.0 / 7.0));
  const Eigen::Isometry3d source_T_target = truth.inverse();
  for (std::size_t index = 0; index < outlier_count; ++index) {
    const std::size_t axis = index % plane_samples.size();
    const auto & samples = plane_samples[axis];
    Eigen::Vector3d outlier_target =
      samples[(index / plane_samples.size()) % samples.size()];
    outlier_target[static_cast<Eigen::Index>(axis)] += 0.22;
    source.push_back(point(source_T_target * outlier_target));
  }
  finalizeCloud(source);

  Registration::Config robust_config = baseConfig();
  robust_config.huber_delta = 0.025;
  robust_config.max_quality_rmse = 0.16;
  robust_config.max_quality_p95 = 0.30;
  const Registration robust_registration(robust_config);
  const Registration::Result robust =
    robust_registration.align(source, target, Eigen::Isometry3d::Identity());

  Registration::Config quadratic_config = robust_config;
  quadratic_config.huber_delta = 10.0;
  quadratic_config.max_quality_rmse = 1.0;
  quadratic_config.max_quality_p95 = 1.0;
  const Registration quadratic_registration(quadratic_config);
  const Registration::Result quadratic =
    quadratic_registration.align(source, target, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(robust.accepted)
    << robust.reason << " rmse=" << robust.rmse
    << " p95=" << robust.p95_abs_residual
    << " rank=" << robust.observable_rank
    << " t=" << robust.target_T_source.translation().transpose();
  ASSERT_TRUE(quadratic.accepted) << quadratic.reason;
  const double robust_translation_error =
    (robust.target_T_source.translation() - truth.translation()).norm();
  const double quadratic_translation_error =
    (quadratic.target_T_source.translation() - truth.translation()).norm();
  EXPECT_LT(robust_translation_error, 0.035);
  EXPECT_LT(rotationError(robust.target_T_source, truth), 0.01);
  EXPECT_GT(robust.robust_downweight_ratio, 0.20);
  EXPECT_LT(robust_translation_error, 0.55 * quadratic_translation_error);
}

TEST(VoxelSurfelRegistrationFailure, SparseTargetProducesNoSurfels)
{
  CloudT target;
  for (int index = 0; index < 20; ++index) {
    target.push_back(point(Eigen::Vector3d(2.0 * index, 0.0, 0.0)));
  }
  finalizeCloud(target);
  const CloudT source = target;

  Registration::Config config = baseConfig();
  config.voxel_size = 0.5;
  config.min_points_per_voxel = 6U;
  const Registration::Result result =
    Registration(config).align(source, target, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "no_planar_surfels");
  EXPECT_EQ(result.planar_surfels, 0U);
}

TEST(VoxelSurfelRegistrationFailure, IsotropicVoxelIsRejectedAsNonplanar)
{
  CloudT target;
  for (double x = 0.1; x <= 0.9; x += 0.1) {
    for (double y = 0.1; y <= 0.9; y += 0.1) {
      for (double z = 0.1; z <= 0.9; z += 0.1) {
        target.push_back(point(Eigen::Vector3d(x, y, z)));
      }
    }
  }
  finalizeCloud(target);

  Registration::Config config = baseConfig();
  config.voxel_size = 2.0;
  config.max_planarity_ratio = 0.10;
  const Registration::Result result =
    Registration(config).align(target, target, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "no_planar_surfels");
}

TEST(VoxelSurfelRegistrationFailure, AllNanSourceFailsWithoutNumericalOutput)
{
  CloudT target;
  addHorizontalPlane(target, 0.0, -2.0, 2.0, -2.0, 2.0, 0.14);
  finalizeCloud(target);

  CloudT source;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int index = 0; index < 100; ++index) {
    source.push_back(PointT(nan, nan, nan));
  }
  finalizeCloud(source);
  source.is_dense = false;

  Registration::Config config = baseConfig();
  config.min_observable_rank = 3;
  const Registration::Result result =
    Registration(config).align(source, target, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "empty_source");
  EXPECT_EQ(result.valid_source_points, 0U);
  EXPECT_TRUE(result.target_T_source.matrix().array().isFinite().all());
}

TEST(VoxelSurfelRegistrationVoxelHash, NegativeCoordinatesDoNotAliasAcrossZero)
{
  CloudT target;
  for (double x = 0.1; x <= 0.9; x += 0.1) {
    for (double y = 0.1; y <= 0.9; y += 0.1) {
      target.push_back(point(Eigen::Vector3d(x, y, -0.6)));
      target.push_back(point(Eigen::Vector3d(x, y, 0.6)));
    }
  }
  finalizeCloud(target);

  Registration::Config config = baseConfig();
  config.voxel_size = 1.0;
  config.min_points_per_voxel = 8U;
  config.min_tangent_variance = 1.0e-3;
  config.min_correspondences = 80U;
  config.min_correspondence_ratio = 0.90;
  config.min_observable_rank = 3;
  config.relative_eigen_threshold = 1.0e-4;
  config.max_plane_distance = 0.2;
  config.max_quality_rmse = 0.01;
  config.max_quality_p95 = 0.01;
  const Registration::Result result =
    Registration(config).align(target, target, Eigen::Isometry3d::Identity());

  ASSERT_TRUE(result.accepted) << result.reason;
  EXPECT_EQ(result.planar_surfels, 2U);
  EXPECT_EQ(result.observable_rank, 3);
  EXPECT_LT(result.rmse, 1.0e-8);
}

TEST(VoxelSurfelRegistrationFailure, RejectsNegativeVoxelSize)
{
  const CloudT target = threeOrthogonalPlanes();
  Registration::Config config = baseConfig();
  config.voxel_size = -0.5;

  const Registration::Result result =
    Registration(config).align(target, target, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason, "invalid_config");
}

}  // namespace
}  // namespace slam_3d_core
}  // namespace gole
