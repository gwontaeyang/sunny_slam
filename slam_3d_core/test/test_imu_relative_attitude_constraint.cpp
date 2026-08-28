#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include <Eigen/Geometry>

#include "slam_3d_core/imu_relative_attitude_constraint.hpp"

namespace
{

using gole::slam_3d_core::ImuRelativeAttitudeConstraintConfig;
using gole::slam_3d_core::constrainImuRelativeRollPitch;
using gole::slam_3d_core::rollPitchYawFromRotation;

Eigen::Matrix3d rotationFromRpyDegrees(
  const double roll_deg,
  const double pitch_deg,
  const double yaw_deg)
{
  constexpr double kDegreesToRadians = M_PI / 180.0;
  return (
    Eigen::AngleAxisd(yaw_deg * kDegreesToRadians, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(pitch_deg * kDegreesToRadians, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(roll_deg * kDegreesToRadians, Eigen::Vector3d::UnitX()))
         .toRotationMatrix();
}

TEST(ImuRelativeAttitudeConstraint, DisabledLeavesCandidateUnchanged)
{
  Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
  candidate.translation() = Eigen::Vector3d(1.0, 2.0, 3.0);
  candidate.linear() = rotationFromRpyDegrees(7.0, -4.0, 30.0);

  const auto result = constrainImuRelativeRollPitch(
    candidate,
    rotationFromRpyDegrees(2.0, 3.0, 10.0),
    rotationFromRpyDegrees(-5.0, 8.0, 60.0),
    ImuRelativeAttitudeConstraintConfig{});

  EXPECT_FALSE(result.evaluated);
  EXPECT_FALSE(result.applied);
  EXPECT_EQ(result.status, "disabled");
  EXPECT_TRUE(result.pose.matrix().isApprox(candidate.matrix(), 1.0e-12));
}

TEST(ImuRelativeAttitudeConstraint, UsesInitialImuOrientationAsMapGauge)
{
  const Eigen::Matrix3d initial = rotationFromRpyDegrees(-3.0, 4.0, 20.0);
  const Eigen::Matrix3d relative = rotationFromRpyDegrees(6.0, -8.0, 35.0);
  const Eigen::Matrix3d current = initial * relative;
  Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
  candidate.translation() = Eigen::Vector3d(4.0, -2.0, 0.7);
  candidate.linear() = rotationFromRpyDegrees(25.0, 18.0, -70.0);

  ImuRelativeAttitudeConstraintConfig config;
  config.enabled = true;
  const auto result = constrainImuRelativeRollPitch(candidate, initial, current, config);
  const Eigen::Vector3d constrained_rpy = rollPitchYawFromRotation(result.pose.linear());

  EXPECT_TRUE(result.evaluated);
  EXPECT_TRUE(result.applied);
  EXPECT_EQ(result.status, "applied");
  EXPECT_NEAR(result.target_roll_deg, 6.0, 1.0e-9);
  EXPECT_NEAR(result.target_pitch_deg, -8.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.x() * 180.0 / M_PI, 6.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.y() * 180.0 / M_PI, -8.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.z() * 180.0 / M_PI, -70.0, 1.0e-9);
  EXPECT_TRUE(result.pose.translation().isApprox(candidate.translation(), 1.0e-12));
}

TEST(ImuRelativeAttitudeConstraint, BoundsEachPerScanTiltCorrection)
{
  Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();
  candidate.linear() = rotationFromRpyDegrees(-10.0, 12.0, 45.0);
  ImuRelativeAttitudeConstraintConfig config;
  config.enabled = true;
  config.max_roll_pitch_correction_deg = 3.0;

  const auto result = constrainImuRelativeRollPitch(
    candidate,
    Eigen::Matrix3d::Identity(),
    rotationFromRpyDegrees(8.0, -9.0, 120.0),
    config);
  const Eigen::Vector3d constrained_rpy = rollPitchYawFromRotation(result.pose.linear());

  EXPECT_NEAR(result.raw_roll_error_deg, 18.0, 1.0e-9);
  EXPECT_NEAR(result.raw_pitch_error_deg, -21.0, 1.0e-9);
  EXPECT_NEAR(result.applied_roll_correction_deg, 3.0, 1.0e-9);
  EXPECT_NEAR(result.applied_pitch_correction_deg, -3.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.x() * 180.0 / M_PI, -7.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.y() * 180.0 / M_PI, 9.0, 1.0e-9);
  EXPECT_NEAR(constrained_rpy.z() * 180.0 / M_PI, 45.0, 1.0e-9);
}

TEST(ImuRelativeAttitudeConstraint, InvalidOrientationFailsClosed)
{
  Eigen::Matrix3d invalid = Eigen::Matrix3d::Identity();
  invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
  ImuRelativeAttitudeConstraintConfig config;
  config.enabled = true;
  const Eigen::Isometry3d candidate = Eigen::Isometry3d::Identity();

  const auto result = constrainImuRelativeRollPitch(
    candidate,
    Eigen::Matrix3d::Identity(),
    invalid,
    config);

  EXPECT_TRUE(result.evaluated);
  EXPECT_FALSE(result.applied);
  EXPECT_EQ(result.status, "invalid_orientation");
  EXPECT_TRUE(result.pose.matrix().isApprox(candidate.matrix(), 1.0e-12));
}

}  // namespace
