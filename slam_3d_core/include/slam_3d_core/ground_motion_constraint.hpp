#ifndef SLAM_3D_CORE__GROUND_MOTION_CONSTRAINT_HPP_
#define SLAM_3D_CORE__GROUND_MOTION_CONSTRAINT_HPP_

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "slam_3d_core/rigid_transform.hpp"

namespace gole
{
namespace slam_3d_core
{

struct GroundMotionConstraintConfig
{
  bool enabled{false};
  double nominal_max_body_z_step_m{0.0};
  double roughness_max_body_z_step_m{0.0};
};

struct GroundMotionConstraintResult
{
  Eigen::Isometry3d pose{Eigen::Isometry3d::Identity()};
  bool evaluated{false};
  bool applied{false};
  bool roughness_relaxed{false};
  double raw_body_z_step_m{0.0};
  double constrained_body_z_step_m{0.0};
  double max_body_z_step_m{0.0};
  std::string status{"disabled"};
};

struct GroundMotionSequenceConstraintStats
{
  bool valid{true};
  int evaluated_edges{0};
  int applied_edges{0};
  double max_raw_body_z_step_m{0.0};
  double max_constrained_body_z_step_m{0.0};
};

inline GroundMotionConstraintResult constrainGroundVehicleMotion(
  const Eigen::Isometry3d & candidate,
  const Eigen::Isometry3d & reference,
  const GroundMotionConstraintConfig & config,
  const bool roughness_recent)
{
  GroundMotionConstraintResult result;
  result.pose = projectToRigidIsometry(candidate);
  if (!config.enabled) {
    return result;
  }

  result.evaluated = true;
  const Eigen::Isometry3d rigid_reference = projectToRigidIsometry(reference);
  if (!result.pose.matrix().allFinite() || !rigid_reference.matrix().allFinite()) {
    result.status = "invalid_transform";
    return result;
  }

  const double nominal_limit =
    std::max(0.0, config.nominal_max_body_z_step_m);
  const double roughness_limit =
    std::max(nominal_limit, config.roughness_max_body_z_step_m);
  result.roughness_relaxed = roughness_recent && roughness_limit > nominal_limit;
  result.max_body_z_step_m =
    result.roughness_relaxed ? roughness_limit : nominal_limit;

  Eigen::Vector3d body_delta =
    rigid_reference.linear().transpose() *
    (result.pose.translation() - rigid_reference.translation());
  result.raw_body_z_step_m = body_delta.z();
  body_delta.z() = std::clamp(
    body_delta.z(),
    -result.max_body_z_step_m,
    result.max_body_z_step_m);
  result.constrained_body_z_step_m = body_delta.z();
  result.pose.translation() =
    rigid_reference.translation() + rigid_reference.linear() * body_delta;
  result.pose = projectToRigidIsometry(result.pose);
  result.applied =
    std::abs(result.raw_body_z_step_m - result.constrained_body_z_step_m) >
    1.0e-12;
  result.status = result.roughness_relaxed ? "roughness_relaxed" : "nominal";
  return result;
}

inline GroundMotionSequenceConstraintStats constrainGroundVehiclePoseSequence(
  std::vector<Eigen::Isometry3d> & poses,
  const GroundMotionConstraintConfig & config,
  const bool use_relaxed_limit)
{
  GroundMotionSequenceConstraintStats stats;
  if (!config.enabled || poses.size() < 2U) {
    return stats;
  }

  for (std::size_t index = 1; index < poses.size(); ++index) {
    const auto result = constrainGroundVehicleMotion(
      poses[index], poses[index - 1U], config, use_relaxed_limit);
    if (!result.evaluated || result.status == "invalid_transform" ||
      !result.pose.matrix().allFinite())
    {
      stats.valid = false;
      return stats;
    }
    ++stats.evaluated_edges;
    stats.applied_edges += result.applied ? 1 : 0;
    stats.max_raw_body_z_step_m = std::max(
      stats.max_raw_body_z_step_m,
      std::abs(result.raw_body_z_step_m));
    stats.max_constrained_body_z_step_m = std::max(
      stats.max_constrained_body_z_step_m,
      std::abs(result.constrained_body_z_step_m));
    poses[index] = result.pose;
  }
  return stats;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__GROUND_MOTION_CONSTRAINT_HPP_
