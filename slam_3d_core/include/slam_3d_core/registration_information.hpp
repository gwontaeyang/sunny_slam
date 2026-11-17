// Copyright 2026 Gole Robotics

#ifndef SLAM_3D_CORE__REGISTRATION_INFORMATION_HPP_
#define SLAM_3D_CORE__REGISTRATION_INFORMATION_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <limits>
#include <string>

namespace gole
{
namespace slam_3d_core
{

using RegistrationMatrix6d = Eigen::Matrix<double, 6, 6>;
using RegistrationVector6d = Eigen::Matrix<double, 6, 1>;

struct RegistrationInformationConfig
{
  // Eigenvalues below this fraction of the largest normalized eigenvalue are
  // reported as geometrically unobservable.
  double relative_eigen_threshold{1.0e-3};
  // Directions at or above this ratio retain the configured graph weight.
  // Intermediate directions are scaled continuously.
  double strong_eigenvalue_ratio{5.0e-2};
  // Keep a bounded positive floor so a sequence of degenerate registrations
  // does not make the graph numerically disconnected.
  double weak_information_ratio{2.0e-2};
};

struct RegistrationInformationResult
{
  bool valid{false};
  RegistrationMatrix6d information_matrix{RegistrationMatrix6d::Zero()};
  // Hessian transported to graph residual order [tx, ty, tz, rx, ry, rz].
  RegistrationMatrix6d graph_residual_hessian{RegistrationMatrix6d::Zero()};
  std::array<double, 6> normalized_hessian_eigenvalues{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 6> information_gains{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  int observable_rank{0};
  double observable_condition{std::numeric_limits<double>::infinity()};
  double minimum_information_gain{0.0};
  std::string failure_reason;
};

bool validRegistrationInformationConfig(
  const RegistrationInformationConfig & config) noexcept;

RegistrationMatrix6d configuredDiagonalInformation(
  const RegistrationVector6d & information_diagonal);

// Converts a robust point-to-plane Hessian for an absolute pose perturbation
// [drot_map, dtrans_map] into the exact pose-graph residual convention
// [dtrans_from, drot_from]. The result uses the configured diagonal information
// as an upper bound; the Hessian contributes observability shape, not an
// uncalibrated point-count-dependent covariance magnitude.
RegistrationInformationResult registrationHessianToPoseGraphInformation(
  const RegistrationMatrix6d & map_pose_hessian_rotation_translation,
  const Eigen::Isometry3d & map_T_from,
  const RegistrationVector6d & configured_information_diagonal,
  const RegistrationInformationConfig & config,
  const Eigen::Vector3d & endpoint_to_registration_origin_map = Eigen::Vector3d::Zero());

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__REGISTRATION_INFORMATION_HPP_
