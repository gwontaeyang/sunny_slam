// Copyright 2026 Gole Robotics

#include <cmath>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "slam_3d_core/pose_graph_optimizer.hpp"

namespace
{

using gole::slam_3d_core::PoseGraphConstraint;
using gole::slam_3d_core::PoseGraphOptimizationConfig;
using gole::slam_3d_core::PoseGraphOptimizer;

Eigen::Isometry3d poseRpy(
  const double x,
  const double y,
  const double z,
  const double roll_rad,
  const double pitch_rad,
  const double yaw_rad)
{
  Eigen::Isometry3d value = Eigen::Isometry3d::Identity();
  value.translation() = Eigen::Vector3d(x, y, z);
  value.linear() = (
    Eigen::AngleAxisd(yaw_rad, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch_rad, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll_rad, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return value;
}

Eigen::Isometry3d pose(
  const double x,
  const double y,
  const double z,
  const double yaw_rad)
{
  return poseRpy(x, y, z, 0.0, 0.0, yaw_rad);
}

PoseGraphConstraint constraint(
  const int from,
  const int to,
  const Eigen::Isometry3d & measurement,
  const double translation_information,
  const double rotation_information,
  const bool loop_closure)
{
  PoseGraphConstraint edge;
  edge.from_index = from;
  edge.to_index = to;
  edge.from_T_to = measurement;
  edge.information_matrix.setZero();
  edge.information_matrix.diagonal() <<
    translation_information,
    translation_information,
    translation_information,
    rotation_information,
    rotation_information,
    rotation_information;
  edge.loop_closure = loop_closure;
  return edge;
}

TEST(PoseGraphOptimizer, ChiSquaredUsesFullCorrelatedInformation)
{
  const std::vector<Eigen::Isometry3d> poses{
    pose(0.0, 0.0, 0.0, 0.0),
    pose(1.0, 2.0, 0.0, 0.0)};
  PoseGraphConstraint edge = constraint(
    0, 1, pose(0.0, 0.0, 0.0, 0.0), 1.0, 1.0, false);
  edge.information_matrix(0, 0) = 2.0;
  edge.information_matrix(1, 1) = 3.0;
  edge.information_matrix(0, 1) = 0.5;
  edge.information_matrix(1, 0) = 0.5;

  EXPECT_NEAR(
    gole::slam_3d_core::poseGraphConstraintChiSquared(edge, poses),
    16.0,
    1.0e-12);
}

TEST(PoseGraphOptimizer, SolveUsesFullCorrelatedInformation)
{
  const std::vector<Eigen::Isometry3d> initial_poses{
    pose(0.0, 0.0, 0.0, 0.0),
    pose(0.5, 1.0, 0.0, 0.0)};
  PoseGraphConstraint origin_measurement = constraint(
    0, 1, pose(0.0, 0.0, 0.0, 0.0), 1.0, 10.0, false);
  PoseGraphConstraint correlated_measurement = constraint(
    0, 1, pose(1.0, 2.0, 0.0, 0.0), 1.0, 10.0, false);
  correlated_measurement.information_matrix(0, 0) = 2.0;
  correlated_measurement.information_matrix(1, 1) = 3.0;
  correlated_measurement.information_matrix(0, 1) = 0.5;
  correlated_measurement.information_matrix(1, 0) = 0.5;

  PoseGraphOptimizationConfig config;
  config.max_iterations_per_stage = 30;
  config.robust_scale_factors = {1.0};
  config.robust_loop_enabled = false;
  config.dcs_enabled = false;
  config.function_tolerance = 1.0e-12;
  config.gradient_tolerance = 1.0e-12;
  config.parameter_tolerance = 1.0e-12;
  const auto result = PoseGraphOptimizer(config).optimize(
    initial_poses, {origin_measurement, correlated_measurement});

  ASSERT_TRUE(result.success) << result.failure_reason;
  // (I + H) x = H [1, 2]^T for H=[[2,.5],[.5,3]].
  EXPECT_NEAR(result.poses.back().translation().x(), 8.75 / 11.75, 1.0e-6);
  EXPECT_NEAR(result.poses.back().translation().y(), 18.0 / 11.75, 1.0e-6);
  EXPECT_NEAR(result.poses.back().translation().z(), 0.0, 1.0e-8);
}

double rotationErrorRad(
  const Eigen::Isometry3d & lhs,
  const Eigen::Isometry3d & rhs)
{
  Eigen::Quaterniond error(lhs.linear().transpose() * rhs.linear());
  error.normalize();
  return Eigen::AngleAxisd(error).angle();
}

}  // namespace

TEST(PoseGraphOptimizer, ConstraintResidualUsesTheFromPoseFrame)
{
  std::vector<Eigen::Isometry3d> poses{
    pose(2.0, -1.0, 0.5, M_PI_2),
    pose(2.0, 1.0, 0.7, M_PI_2 + 0.1)};
  const Eigen::Isometry3d measurement = poses.front().inverse() * poses.back();
  const PoseGraphConstraint edge = constraint(0, 1, measurement, 3.0, 2.0, false);

  EXPECT_NEAR(
    gole::slam_3d_core::poseGraphConstraintChiSquared(edge, poses),
    0.0,
    1.0e-12);
}

TEST(PoseGraphOptimizer, DynamicCovarianceScalingMatchesPublishedClosedForm)
{
  EXPECT_DOUBLE_EQ(gole::slam_3d_core::dynamicCovarianceScale(0.0, 1.0), 1.0);
  EXPECT_DOUBLE_EQ(gole::slam_3d_core::dynamicCovarianceScale(1.0, 1.0), 1.0);
  EXPECT_NEAR(gole::slam_3d_core::dynamicCovarianceScale(3.0, 1.0), 0.5, 1.0e-12);
  EXPECT_NEAR(gole::slam_3d_core::dynamicCovarianceScale(9.0, 1.0), 0.2, 1.0e-12);
  EXPECT_DOUBLE_EQ(gole::slam_3d_core::dynamicCovarianceScale(INFINITY, 1.0), 0.0);
}

TEST(PoseGraphOptimizer, MultiStageSe3SolveDistributesAClosureCorrection)
{
  constexpr int kLastIndex = 10;
  std::vector<Eigen::Isometry3d> initial_poses;
  for (int index = 0; index <= kLastIndex; ++index) {
    initial_poses.push_back(
      poseRpy(
        static_cast<double>(index),
        0.0,
        0.10 * static_cast<double>(index),
        0.004 * static_cast<double>(index),
        -0.006 * static_cast<double>(index),
        0.01 * static_cast<double>(index)));
  }

  std::vector<PoseGraphConstraint> constraints;
  for (int index = 0; index < kLastIndex; ++index) {
    constraints.push_back(
      constraint(
        index,
        index + 1,
        initial_poses[static_cast<std::size_t>(index)].inverse() *
        initial_poses[static_cast<std::size_t>(index + 1)],
        1.0,
        1.0,
        false));
  }
  constraints.push_back(
    constraint(0, kLastIndex, pose(10.0, 0.0, 0.0, 0.0), 4.0, 4.0, true));

  PoseGraphOptimizationConfig config;
  config.max_iterations_per_stage = 25;
  config.robust_scale_factors = {4.0, 2.0, 1.0};
  config.dcs_enabled = true;
  config.dcs_start_stage = 0;
  config.dcs_phi = 5.0;
  PoseGraphOptimizer optimizer(config);

  const double initial_loop_error =
    gole::slam_3d_core::poseGraphConstraintChiSquared(constraints.back(), initial_poses);
  const auto result = optimizer.optimize(initial_poses, constraints);

  ASSERT_TRUE(result.success) << result.failure_reason;
  ASSERT_EQ(result.stages.size(), 3U);
  ASSERT_EQ(result.poses.size(), initial_poses.size());
  EXPECT_EQ(result.dcs_downweighted_loops, 0);
  for (const auto & stage : result.stages) {
    // Costs are reevaluated from normalized output poses for every backend.
    EXPECT_LE(stage.final_cost, stage.initial_cost + 1.0e-12);
  }
  EXPECT_LT(
    gole::slam_3d_core::poseGraphConstraintChiSquared(constraints.back(), result.poses),
    0.25 * initial_loop_error);
  EXPECT_LT(std::abs(result.poses.back().translation().z()), 0.35);
  EXPECT_LT(rotationErrorRad(result.poses.back(), pose(10.0, 0.0, 0.0, 0.0)), 0.04);
  EXPECT_NEAR(
    (result.poses.front().inverse() * initial_poses.front()).translation().norm(), 0.0, 1.0e-12);
  EXPECT_NEAR(rotationErrorRad(result.poses.front(), initial_poses.front()), 0.0, 1.0e-12);
}

TEST(PoseGraphOptimizer, RejectsDcsStartStageOutsideTheAnnealingSchedule)
{
  PoseGraphOptimizationConfig config;
  config.robust_scale_factors = {4.0, 2.0, 1.0};
  config.dcs_enabled = true;
  config.dcs_start_stage = 3;

  EXPECT_FALSE(gole::slam_3d_core::validPoseGraphOptimizationConfig(config));
  EXPECT_THROW(PoseGraphOptimizer{config}, std::invalid_argument);
}

TEST(PoseGraphOptimizer, DcsSuppressesAnInconsistentLoop)
{
  constexpr int kLastIndex = 12;
  std::vector<Eigen::Isometry3d> initial_poses;
  for (int index = 0; index <= kLastIndex; ++index) {
    initial_poses.push_back(pose(static_cast<double>(index), 0.0, 0.0, 0.0));
  }

  std::vector<PoseGraphConstraint> constraints;
  for (int index = 0; index < kLastIndex; ++index) {
    constraints.push_back(
      constraint(index, index + 1, pose(1.0, 0.0, 0.0, 0.0), 10.0, 10.0, false));
  }
  constraints.push_back(
    constraint(0, kLastIndex, pose(12.0, 10.0, 0.0, 0.0), 1.0, 1.0, true));

  PoseGraphOptimizationConfig config;
  config.max_iterations_per_stage = 15;
  config.robust_scale_factors = {4.0, 2.0, 1.0};
  config.robust_loop_enabled = false;
  config.dcs_enabled = true;
  config.dcs_start_stage = 0;
  config.dcs_phi = 1.0;
  config.dcs_min_scale = 0.02;
  const auto result = PoseGraphOptimizer(config).optimize(initial_poses, constraints);

  ASSERT_TRUE(result.success) << result.failure_reason;
  EXPECT_EQ(result.dcs_downweighted_loops, 1);
  EXPECT_LT(result.min_dcs_scale, 0.05);
  EXPECT_LT(std::abs(result.poses.back().translation().y()), 0.25);
}

TEST(PoseGraphOptimizer, RejectsDisconnectedGraphTransactionally)
{
  const std::vector<Eigen::Isometry3d> initial_poses{
    pose(0.0, 0.0, 0.0, 0.0),
    pose(1.0, 0.0, 0.0, 0.0),
    pose(5.0, 0.0, 0.0, 0.0)};
  const std::vector<PoseGraphConstraint> constraints{
    constraint(0, 1, pose(1.0, 0.0, 0.0, 0.0), 1.0, 1.0, false)};

  const auto result = PoseGraphOptimizer().optimize(initial_poses, constraints);

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.failure_reason, "disconnected_graph");
  ASSERT_EQ(result.poses.size(), initial_poses.size());
  EXPECT_NEAR(
    (result.poses.back().translation() - initial_poses.back().translation()).norm(),
    0.0,
    1.0e-12);
}

TEST(PoseGraphOptimizer, RejectsIndefiniteFullInformationTransactionally)
{
  const std::vector<Eigen::Isometry3d> initial_poses{
    pose(0.0, 0.0, 0.0, 0.0),
    pose(1.0, 0.0, 0.0, 0.0)};
  PoseGraphConstraint edge = constraint(
    0, 1, pose(1.0, 0.0, 0.0, 0.0), 1.0, 1.0, false);
  edge.information_matrix(0, 0) = -1.0;

  const auto result = PoseGraphOptimizer().optimize(initial_poses, {edge});

  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.failure_reason, "invalid_constraint");
  ASSERT_EQ(result.poses.size(), initial_poses.size());
  EXPECT_TRUE(result.poses.back().matrix().isApprox(initial_poses.back().matrix(), 1.0e-12));
}
