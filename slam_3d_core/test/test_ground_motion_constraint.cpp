#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "slam_3d_core/ground_motion_constraint.hpp"

namespace
{

using gole::slam_3d_core::GroundMotionConstraintConfig;
using gole::slam_3d_core::constrainGroundVehicleMotion;
using gole::slam_3d_core::constrainGroundVehiclePoseSequence;

Eigen::Isometry3d makeReference()
{
  Eigen::Isometry3d reference = Eigen::Isometry3d::Identity();
  reference.translation() = Eigen::Vector3d(4.0, -2.0, 1.5);
  reference.linear() =
    (
    Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(-0.15, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitX())).toRotationMatrix();
  return reference;
}

TEST(GroundMotionConstraint, DisabledLeavesCandidateUnchanged)
{
  const Eigen::Isometry3d reference = makeReference();
  Eigen::Isometry3d candidate = reference;
  candidate.translation() += Eigen::Vector3d(1.0, 2.0, 3.0);

  const auto result = constrainGroundVehicleMotion(
    candidate, reference, GroundMotionConstraintConfig{}, false);

  EXPECT_FALSE(result.evaluated);
  EXPECT_FALSE(result.applied);
  EXPECT_EQ(result.status, "disabled");
  EXPECT_TRUE(result.pose.matrix().isApprox(candidate.matrix(), 1.0e-12));
}

TEST(GroundMotionConstraint, HardConstraintRemovesOnlyBodyVerticalMotion)
{
  const Eigen::Isometry3d reference = makeReference();
  const Eigen::Vector3d body_delta(1.2, -0.3, 0.45);
  Eigen::Isometry3d candidate = reference;
  candidate.translation() += reference.linear() * body_delta;
  candidate.linear() =
    candidate.linear() *
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  GroundMotionConstraintConfig config;
  config.enabled = true;
  const auto result = constrainGroundVehicleMotion(
    candidate, reference, config, false);
  const Eigen::Vector3d constrained_body_delta =
    reference.linear().transpose() *
    (result.pose.translation() - reference.translation());

  EXPECT_TRUE(result.evaluated);
  EXPECT_TRUE(result.applied);
  EXPECT_EQ(result.status, "nominal");
  EXPECT_NEAR(result.raw_body_z_step_m, 0.45, 1.0e-12);
  EXPECT_NEAR(result.constrained_body_z_step_m, 0.0, 1.0e-12);
  EXPECT_NEAR(constrained_body_delta.x(), body_delta.x(), 1.0e-12);
  EXPECT_NEAR(constrained_body_delta.y(), body_delta.y(), 1.0e-12);
  EXPECT_NEAR(constrained_body_delta.z(), 0.0, 1.0e-12);
  EXPECT_TRUE(result.pose.linear().isApprox(candidate.linear(), 1.0e-12));
}

TEST(GroundMotionConstraint, BodyForwardMotionPreservesMapElevationOnSlope)
{
  Eigen::Isometry3d reference = Eigen::Isometry3d::Identity();
  reference.linear() =
    Eigen::AngleAxisd(-10.0 * M_PI / 180.0, Eigen::Vector3d::UnitY())
    .toRotationMatrix();
  Eigen::Isometry3d candidate = reference;
  candidate.translation() = reference.linear() * Eigen::Vector3d(2.0, 0.0, 0.0);

  GroundMotionConstraintConfig config;
  config.enabled = true;
  const auto result = constrainGroundVehicleMotion(
    candidate, reference, config, false);

  EXPECT_FALSE(result.applied);
  EXPECT_NEAR(result.pose.translation().z(), candidate.translation().z(), 1.0e-12);
  EXPECT_GT(std::abs(result.pose.translation().z()), 0.3);
}

TEST(GroundMotionConstraint, RoughnessUsesOnlyTheConfiguredRelaxedLimit)
{
  const Eigen::Isometry3d reference = makeReference();
  Eigen::Isometry3d candidate = reference;
  candidate.translation() += reference.linear() * Eigen::Vector3d(0.5, 0.1, -0.08);

  GroundMotionConstraintConfig config;
  config.enabled = true;
  config.nominal_max_body_z_step_m = 0.01;
  config.roughness_max_body_z_step_m = 0.03;
  const auto nominal = constrainGroundVehicleMotion(
    candidate, reference, config, false);
  const auto rough = constrainGroundVehicleMotion(
    candidate, reference, config, true);

  EXPECT_NEAR(nominal.constrained_body_z_step_m, -0.01, 1.0e-12);
  EXPECT_FALSE(nominal.roughness_relaxed);
  EXPECT_EQ(nominal.status, "nominal");
  EXPECT_NEAR(rough.constrained_body_z_step_m, -0.03, 1.0e-12);
  EXPECT_TRUE(rough.roughness_relaxed);
  EXPECT_EQ(rough.status, "roughness_relaxed");
}

TEST(GroundMotionConstraint, SequenceRestoresBodyVerticalInvariantAfterPlanarCorrections)
{
  GroundMotionConstraintConfig config;
  config.enabled = true;
  config.nominal_max_body_z_step_m = 0.01;
  config.roughness_max_body_z_step_m = 0.03;

  std::vector<Eigen::Isometry3d> poses(4, Eigen::Isometry3d::Identity());
  for (std::size_t index = 0; index < poses.size(); ++index) {
    poses[index].linear() =
      (
      Eigen::AngleAxisd(0.05 * static_cast<double>(index), Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(-0.12, Eigen::Vector3d::UnitY())).toRotationMatrix();
    poses[index].translation() =
      Eigen::Vector3d(
      2.0 * static_cast<double>(index),
      0.3 * static_cast<double>(index),
      0.0);
  }

  const auto stats = constrainGroundVehiclePoseSequence(poses, config, true);

  ASSERT_TRUE(stats.valid);
  EXPECT_EQ(stats.evaluated_edges, 3);
  EXPECT_GT(stats.applied_edges, 0);
  EXPECT_GT(stats.max_raw_body_z_step_m, config.roughness_max_body_z_step_m);
  EXPECT_LE(
    stats.max_constrained_body_z_step_m,
    config.roughness_max_body_z_step_m + 1.0e-12);
  for (std::size_t index = 1; index < poses.size(); ++index) {
    const Eigen::Vector3d body_delta =
      poses[index - 1U].linear().transpose() *
      (poses[index].translation() - poses[index - 1U].translation());
    EXPECT_LE(std::abs(body_delta.z()), config.roughness_max_body_z_step_m + 1.0e-12);
  }
}

TEST(GroundMotionConstraint, SequenceFailsClosedOnInvalidPose)
{
  GroundMotionConstraintConfig config;
  config.enabled = true;
  std::vector<Eigen::Isometry3d> poses(2, Eigen::Isometry3d::Identity());
  poses.back().translation().x() = std::numeric_limits<double>::quiet_NaN();

  const auto stats = constrainGroundVehiclePoseSequence(poses, config, false);

  EXPECT_FALSE(stats.valid);
  EXPECT_EQ(stats.evaluated_edges, 0);
}

}  // namespace
