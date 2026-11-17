// Copyright 2026 Gole Robotics

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <cmath>

#include "slam_3d_core/registration_information.hpp"

namespace
{

using gole::slam_3d_core::RegistrationInformationConfig;
using gole::slam_3d_core::RegistrationMatrix6d;
using gole::slam_3d_core::RegistrationVector6d;
using gole::slam_3d_core::registrationHessianToPoseGraphInformation;

RegistrationInformationConfig config()
{
  RegistrationInformationConfig value;
  value.relative_eigen_threshold = 1.0e-3;
  value.strong_eigenvalue_ratio = 5.0e-2;
  value.weak_information_ratio = 2.0e-2;
  return value;
}

RegistrationVector6d unitBaseline()
{
  return RegistrationVector6d::Ones();
}

Eigen::Isometry3d yawPose(const double yaw_rad)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return pose;
}

Eigen::Matrix3d rotationExp(const Eigen::Vector3d & tangent)
{
  const double angle = tangent.norm();
  if (angle < 1.0e-15) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(angle, tangent / angle).toRotationMatrix();
}

Eigen::Vector3d rotationLog(const Eigen::Matrix3d & rotation)
{
  const Eigen::AngleAxisd angle_axis(rotation);
  return angle_axis.angle() * angle_axis.axis();
}

}  // namespace

TEST(RegistrationInformation, FullRankIsotropicHessianRetainsConfiguredBaseline)
{
  const auto result = registrationHessianToPoseGraphInformation(
    RegistrationMatrix6d::Identity(),
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    config());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_EQ(result.observable_rank, 6);
  EXPECT_NEAR(result.observable_condition, 1.0, 1.0e-12);
  EXPECT_TRUE(result.information_matrix.isApprox(RegistrationMatrix6d::Identity(), 1.0e-12));
}

TEST(RegistrationInformation, SinglePlaneWeakensOnlyUnobservableGraphDirections)
{
  // Absolute point-to-plane Hessian order is [rx, ry, rz, tx, ty, tz].
  RegistrationMatrix6d map_hessian = RegistrationMatrix6d::Zero();
  map_hessian.diagonal() << 10.0, 10.0, 0.0, 0.0, 0.0, 10.0;

  const auto result = registrationHessianToPoseGraphInformation(
    map_hessian,
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    config());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_EQ(result.observable_rank, 3);
  EXPECT_NEAR(result.information_matrix(0, 0), 0.02, 1.0e-12);
  EXPECT_NEAR(result.information_matrix(1, 1), 0.02, 1.0e-12);
  EXPECT_NEAR(result.information_matrix(2, 2), 1.0, 1.0e-12);
  EXPECT_NEAR(result.information_matrix(3, 3), 1.0, 1.0e-12);
  EXPECT_NEAR(result.information_matrix(4, 4), 1.0, 1.0e-12);
  EXPECT_NEAR(result.information_matrix(5, 5), 0.02, 1.0e-12);
}

TEST(RegistrationInformation, RotatesMapObservabilityIntoTheFromFrame)
{
  RegistrationMatrix6d map_hessian = RegistrationMatrix6d::Zero();
  map_hessian.diagonal() << 100.0, 100.0, 100.0, 100.0, 0.0, 0.0;

  const auto result = registrationHessianToPoseGraphInformation(
    map_hessian,
    yawPose(M_PI_2),
    unitBaseline(),
    config());

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_GT(result.information_matrix(1, 1), 0.95);
  EXPECT_LT(result.information_matrix(0, 0), 0.05);
  EXPECT_LT(result.information_matrix(2, 2), 0.05);
}

TEST(RegistrationInformation, EndpointLeverArmPreservesTranslationRotationCoupling)
{
  RegistrationMatrix6d map_hessian = RegistrationMatrix6d::Zero();
  map_hessian.diagonal().tail<3>().setConstant(100.0);

  const auto result = registrationHessianToPoseGraphInformation(
    map_hessian,
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    config(),
    Eigen::Vector3d(1.0, 0.0, 0.0));

  ASSERT_TRUE(result.valid) << result.failure_reason;
  EXPECT_EQ(result.observable_rank, 3);
  EXPECT_GT(std::abs(result.information_matrix(1, 5)), 0.1);
  EXPECT_GT(std::abs(result.information_matrix(2, 4)), 0.1);
  EXPECT_TRUE(
    result.information_matrix.isApprox(
      result.information_matrix.transpose(), 1.0e-12));
}

TEST(RegistrationInformation, DirectionalShapeNeverExceedsConfiguredBaseline)
{
  RegistrationMatrix6d map_hessian = RegistrationMatrix6d::Identity();
  map_hessian(0, 3) = 0.35;
  map_hessian(3, 0) = 0.35;
  map_hessian(1, 5) = -0.2;
  map_hessian(5, 1) = -0.2;
  RegistrationVector6d configured;
  configured << 2.0, 3.0, 8.0, 1.5, 2.5, 4.0;

  const auto result = registrationHessianToPoseGraphInformation(
    map_hessian,
    yawPose(0.37),
    configured,
    config(),
    Eigen::Vector3d(0.8, -0.3, 0.2));

  ASSERT_TRUE(result.valid) << result.failure_reason;
  const RegistrationMatrix6d remaining_information =
    configured.asDiagonal().toDenseMatrix() - result.information_matrix;
  const Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> remaining_solver(
    0.5 * (remaining_information + remaining_information.transpose()));
  ASSERT_EQ(remaining_solver.info(), Eigen::Success);
  EXPECT_GE(remaining_solver.eigenvalues().minCoeff(), -1.0e-12);
  const Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> output_solver(
    result.information_matrix);
  ASSERT_EQ(output_solver.info(), Eigen::Success);
  EXPECT_GT(output_solver.eigenvalues().minCoeff(), 0.0);
}

TEST(RegistrationInformation, FrameAndLeverTransportMatchesFiniteDifference)
{
  Eigen::Isometry3d map_T_from = Eigen::Isometry3d::Identity();
  map_T_from.translation() = Eigen::Vector3d(4.0, -2.0, 0.7);
  map_T_from.linear() =
    (Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.17, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.09, Eigen::Vector3d::UnitX())).toRotationMatrix();
  Eigen::Isometry3d from_T_endpoint = Eigen::Isometry3d::Identity();
  from_T_endpoint.translation() = Eigen::Vector3d(1.3, -0.4, 0.2);
  from_T_endpoint.linear() =
    Eigen::AngleAxisd(-0.23, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  Eigen::Isometry3d endpoint_T_origin = Eigen::Isometry3d::Identity();
  endpoint_T_origin.translation() = Eigen::Vector3d(0.8, 0.15, -0.05);

  const Eigen::Isometry3d map_T_endpoint = map_T_from * from_T_endpoint;
  const Eigen::Isometry3d map_T_origin = map_T_endpoint * endpoint_T_origin;
  const Eigen::Vector3d lever_map =
    map_T_endpoint.linear() * endpoint_T_origin.translation();

  RegistrationMatrix6d numerical_jacobian = RegistrationMatrix6d::Zero();
  constexpr double kEpsilon = 1.0e-7;
  auto origin_increment = [&](const RegistrationVector6d & graph_residual) {
      Eigen::Isometry3d perturbed_endpoint = map_T_endpoint;
      perturbed_endpoint.translation() +=
        map_T_from.linear() * graph_residual.head<3>();
      perturbed_endpoint.linear() =
        map_T_from.linear() * rotationExp(-graph_residual.tail<3>()) *
        from_T_endpoint.linear();
      const Eigen::Isometry3d perturbed_origin = perturbed_endpoint * endpoint_T_origin;
      RegistrationVector6d increment;
      increment.head<3>() = rotationLog(
        perturbed_origin.linear() * map_T_origin.linear().transpose());
      increment.tail<3>() = perturbed_origin.translation() - map_T_origin.translation();
      return increment;
    };
  for (int column = 0; column < 6; ++column) {
    RegistrationVector6d positive = RegistrationVector6d::Zero();
    RegistrationVector6d negative = RegistrationVector6d::Zero();
    positive[column] = kEpsilon;
    negative[column] = -kEpsilon;
    numerical_jacobian.col(column) =
      (origin_increment(positive) - origin_increment(negative)) /
      (2.0 * kEpsilon);
  }

  RegistrationMatrix6d map_hessian = RegistrationMatrix6d::Identity();
  map_hessian(0, 3) = 0.17;
  map_hessian(3, 0) = 0.17;
  map_hessian(2, 5) = -0.11;
  map_hessian(5, 2) = -0.11;
  const auto result = registrationHessianToPoseGraphInformation(
    map_hessian,
    map_T_from,
    unitBaseline(),
    config(),
    lever_map);

  ASSERT_TRUE(result.valid) << result.failure_reason;
  const RegistrationMatrix6d expected =
    numerical_jacobian.transpose() * map_hessian * numerical_jacobian;
  EXPECT_TRUE(result.graph_residual_hessian.isApprox(expected, 2.0e-7)) <<
    "transport error=" << (result.graph_residual_hessian - expected).norm();
}

TEST(RegistrationInformation, RejectsIndefiniteOrInvalidConfiguration)
{
  RegistrationMatrix6d indefinite = RegistrationMatrix6d::Identity();
  indefinite(0, 0) = -1.0;
  auto result = registrationHessianToPoseGraphInformation(
    indefinite,
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    config());
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "invalid_map_pose_hessian");

  RegistrationInformationConfig invalid_config = config();
  invalid_config.strong_eigenvalue_ratio =
    0.5 * invalid_config.relative_eigen_threshold;
  result = registrationHessianToPoseGraphInformation(
    RegistrationMatrix6d::Identity(),
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    invalid_config);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "invalid_config");

  invalid_config = config();
  invalid_config.weak_information_ratio = 0.0;
  result = registrationHessianToPoseGraphInformation(
    RegistrationMatrix6d::Identity(),
    Eigen::Isometry3d::Identity(),
    unitBaseline(),
    invalid_config);
  EXPECT_FALSE(result.valid);
  EXPECT_EQ(result.failure_reason, "invalid_config");
}
