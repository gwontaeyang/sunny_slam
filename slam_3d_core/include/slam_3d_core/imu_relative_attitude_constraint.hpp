#ifndef SLAM_3D_CORE__IMU_RELATIVE_ATTITUDE_CONSTRAINT_HPP_
#define SLAM_3D_CORE__IMU_RELATIVE_ATTITUDE_CONSTRAINT_HPP_

#include <algorithm>
#include <cmath>
#include <string>

#include <Eigen/Geometry>

#include "slam_3d_core/rigid_transform.hpp"

namespace gole
{
namespace slam_3d_core
{

struct ImuRelativeAttitudeConstraintConfig
{
  bool enabled{false};
  double max_roll_pitch_correction_deg{0.0};
};

struct ImuRelativeAttitudeConstraintResult
{
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
  bool evaluated{false};
  bool applied{false};
  double target_roll_deg{0.0};
  double target_pitch_deg{0.0};
  double raw_roll_error_deg{0.0};
  double raw_pitch_error_deg{0.0};
  double applied_roll_correction_deg{0.0};
  double applied_pitch_correction_deg{0.0};
  std::string status{"disabled"};
};

inline double wrapAttitudeAngle(const double angle_rad) noexcept
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

inline Eigen::Vector3d rollPitchYawFromRotation(const Eigen::Matrix3d & rotation)
{
  return Eigen::Vector3d(
    std::atan2(rotation(2, 1), rotation(2, 2)),
    std::atan2(
      -rotation(2, 0),
      std::hypot(rotation(2, 1), rotation(2, 2))),
    std::atan2(rotation(1, 0), rotation(0, 0)));
}

inline ImuRelativeAttitudeConstraintResult constrainImuRelativeRollPitch(
  const Eigen::Isometry3d & candidate,
  const Eigen::Matrix3d & initial_world_R_base,
  const Eigen::Matrix3d & current_world_R_base,
  const ImuRelativeAttitudeConstraintConfig & config)
{
  ImuRelativeAttitudeConstraintResult result;
  result.pose = projectToRigidIsometry(candidate);
  if (!config.enabled) {
    return result;
  }

  result.evaluated = true;
  if (!result.pose.matrix().allFinite() || !initial_world_R_base.allFinite() ||
    !current_world_R_base.allFinite())
  {
    result.status = "invalid_orientation";
    return result;
  }

  Eigen::Quaterniond initial_rotation(initial_world_R_base);
  Eigen::Quaterniond current_rotation(current_world_R_base);
  if (!initial_rotation.coeffs().allFinite() || !current_rotation.coeffs().allFinite() ||
    initial_rotation.norm() <= 1.0e-12 || current_rotation.norm() <= 1.0e-12)
  {
    result.status = "invalid_orientation";
    return result;
  }
  initial_rotation.normalize();
  current_rotation.normalize();

  const Eigen::Matrix3d map_R_base_imu =
    initial_rotation.toRotationMatrix().transpose() *
    current_rotation.toRotationMatrix();
  const Eigen::Vector3d target_rpy = rollPitchYawFromRotation(map_R_base_imu);
  const Eigen::Vector3d candidate_rpy = rollPitchYawFromRotation(result.pose.linear());
  const double raw_roll_error = wrapAttitudeAngle(target_rpy.x() - candidate_rpy.x());
  const double raw_pitch_error = wrapAttitudeAngle(target_rpy.y() - candidate_rpy.y());

  const double max_correction_rad =
    std::max(0.0, config.max_roll_pitch_correction_deg) * M_PI / 180.0;
  const double applied_roll = max_correction_rad > 0.0 ?
    std::clamp(raw_roll_error, -max_correction_rad, max_correction_rad) :
    raw_roll_error;
  const double applied_pitch = max_correction_rad > 0.0 ?
    std::clamp(raw_pitch_error, -max_correction_rad, max_correction_rad) :
    raw_pitch_error;

  const double constrained_roll = candidate_rpy.x() + applied_roll;
  const double constrained_pitch = candidate_rpy.y() + applied_pitch;
  const double preserved_yaw = candidate_rpy.z();
  result.pose.linear() =
    (
    Eigen::AngleAxisd(preserved_yaw, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(constrained_pitch, Eigen::Vector3d::UnitY()) *
    Eigen::AngleAxisd(constrained_roll, Eigen::Vector3d::UnitX())).toRotationMatrix();
  result.pose = projectToRigidIsometry(result.pose);

  constexpr double kRadiansToDegrees = 180.0 / M_PI;
  result.target_roll_deg = target_rpy.x() * kRadiansToDegrees;
  result.target_pitch_deg = target_rpy.y() * kRadiansToDegrees;
  result.raw_roll_error_deg = raw_roll_error * kRadiansToDegrees;
  result.raw_pitch_error_deg = raw_pitch_error * kRadiansToDegrees;
  result.applied_roll_correction_deg = applied_roll * kRadiansToDegrees;
  result.applied_pitch_correction_deg = applied_pitch * kRadiansToDegrees;
  result.applied =
    std::abs(applied_roll) > 1.0e-12 || std::abs(applied_pitch) > 1.0e-12;
  result.status = result.applied ? "applied" : "within_tolerance";
  return result;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__IMU_RELATIVE_ATTITUDE_CONSTRAINT_HPP_
