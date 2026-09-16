// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__POSE_GRAPH_BACKEND_HPP_
#define SLAM_3D_CORE__POSE_GRAPH_BACKEND_HPP_

#include "slam_3d_core/pose_graph_optimizer.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace detail
{

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// All adapters use the existing translation-first residual, not the library's
// native SE(3) Logmap convention. Canonicalize the error for correlated information.
template<typename T>
Eigen::Matrix<T, 6, 1> relativePoseResidual(
  const Eigen::Matrix<T, 3, 1> & from_position,
  const Eigen::Quaternion<T> & from_rotation,
  const Eigen::Matrix<T, 3, 1> & to_position,
  const Eigen::Quaternion<T> & to_rotation,
  const Eigen::Isometry3d & measurement)
{
  Eigen::Quaterniond measured_rotation(measurement.linear());
  measured_rotation.normalize();
  const Eigen::Quaternion<T> estimated_rotation = from_rotation.conjugate() * to_rotation;
  Eigen::Quaternion<T> error = measured_rotation.template cast<T>() *
    estimated_rotation.conjugate();
  if (error.w() < T(0)) {
    error.coeffs() *= T(-1);
  }
  Eigen::Matrix<T, 6, 1> residual;
  residual.template head<3>() = from_rotation.conjugate() * (to_position - from_position) -
    measurement.translation().template cast<T>();
  residual.template tail<3>() = T(2) * error.vec();
  return residual;
}

inline Vector6d relativePoseResidual(
  const Eigen::Isometry3d & from, const Eigen::Isometry3d & to,
  const Eigen::Isometry3d & measurement)
{
  return relativePoseResidual<double>(
    from.translation(), Eigen::Quaterniond(from.linear()).normalized(),
    to.translation(), Eigen::Quaterniond(to.linear()).normalized(), measurement);
}

struct StageSolveResult
{
  bool success{false};
  std::vector<Eigen::Isometry3d> poses;
  PoseGraphOptimizationStageStats stats;
  std::string failure_reason;
};

// sqrt_information already includes this stage's DCS scale. Each adapter
// applies block Cauchy loss only to loop edges, and fixes pose zero exactly.
StageSolveResult solveCeresStage(
  const PoseGraphOptimizationConfig & config,
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information, double cauchy_scale);

StageSolveResult solveG2oStage(
  const PoseGraphOptimizationConfig & config,
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information, double cauchy_scale);

StageSolveResult solveGtsamStage(
  const PoseGraphOptimizationConfig & config,
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information, double cauchy_scale);

}  // namespace detail
}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__POSE_GRAPH_BACKEND_HPP_
