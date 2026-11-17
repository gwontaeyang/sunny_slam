// Copyright 2026 Gole Robotics

#include "slam_3d_core/registration_information.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>

namespace gole
{
namespace slam_3d_core
{
namespace
{

bool finiteRigidPose(const Eigen::Isometry3d & pose)
{
  if (!pose.matrix().allFinite()) {
    return false;
  }
  const Eigen::Matrix3d orthogonality =
    pose.linear().transpose() * pose.linear() - Eigen::Matrix3d::Identity();
  return orthogonality.norm() <= 1.0e-6 &&
         std::abs(pose.linear().determinant() - 1.0) <= 1.0e-6;
}

bool symmetricPositiveSemidefinite(
  const RegistrationMatrix6d & matrix,
  Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> & solver)
{
  if (!matrix.allFinite()) {
    return false;
  }
  const double matrix_scale = std::max(1.0, matrix.norm());
  if ((matrix - matrix.transpose()).norm() > 1.0e-8 * matrix_scale) {
    return false;
  }
  solver.compute(0.5 * (matrix + matrix.transpose()));
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return false;
  }
  const double maximum = solver.eigenvalues().cwiseAbs().maxCoeff();
  const double negative_tolerance = 1.0e-10 * std::max(1.0, maximum);
  return solver.eigenvalues().minCoeff() >= -negative_tolerance;
}

}  // namespace

bool validRegistrationInformationConfig(
  const RegistrationInformationConfig & config) noexcept
{
  return std::isfinite(config.relative_eigen_threshold) &&
         config.relative_eigen_threshold > 0.0 &&
         config.relative_eigen_threshold <= 1.0 &&
         std::isfinite(config.strong_eigenvalue_ratio) &&
         config.strong_eigenvalue_ratio >= config.relative_eigen_threshold &&
         config.strong_eigenvalue_ratio <= 1.0 &&
         std::isfinite(config.weak_information_ratio) &&
         config.weak_information_ratio > 0.0 &&
         config.weak_information_ratio <= 1.0;
}

RegistrationMatrix6d configuredDiagonalInformation(
  const RegistrationVector6d & information_diagonal)
{
  return information_diagonal.asDiagonal();
}

RegistrationInformationResult registrationHessianToPoseGraphInformation(
  const RegistrationMatrix6d & map_pose_hessian_rotation_translation,
  const Eigen::Isometry3d & map_T_from,
  const RegistrationVector6d & configured_information_diagonal,
  const RegistrationInformationConfig & config,
  const Eigen::Vector3d & endpoint_to_registration_origin_map)
{
  RegistrationInformationResult result;
  auto fail = [&result](const std::string & reason) {
      result.valid = false;
      result.failure_reason = reason;
      return result;
    };

  if (!validRegistrationInformationConfig(config)) {
    return fail("invalid_config");
  }
  if (!finiteRigidPose(map_T_from)) {
    return fail("invalid_reference_pose");
  }
  if (!endpoint_to_registration_origin_map.allFinite()) {
    return fail("invalid_registration_origin_offset");
  }
  if (!configured_information_diagonal.allFinite() ||
    (configured_information_diagonal.array() <= 0.0).any())
  {
    return fail("invalid_configured_information");
  }

  Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> map_hessian_solver;
  if (!symmetricPositiveSemidefinite(
      map_pose_hessian_rotation_translation, map_hessian_solver))
  {
    return fail("invalid_map_pose_hessian");
  }
  if (map_hessian_solver.eigenvalues().maxCoeff() <= 0.0) {
    return fail("zero_map_pose_hessian");
  }

  // Registration uses independent left rotation and map translation updates:
  //   R_to' = Exp(drot_map) R_to, t_to' = t_to + dtrans_map.
  // For the graph residual used by PoseGraphOptimizer, a positive rotational
  // residual applies Exp(-drot_from) to from_R_to. Therefore
  //   drot_map = -map_R_from * drot_from
  //   dtrans_map = map_R_from * dtrans_from.
  RegistrationMatrix6d map_increment_from_graph_residual =
    RegistrationMatrix6d::Zero();
  map_increment_from_graph_residual.block<3, 3>(0, 3) = -map_T_from.linear();
  map_increment_from_graph_residual.block<3, 3>(3, 0) = map_T_from.linear();
  Eigen::Matrix3d lever_skew;
  lever_skew <<
    0.0, -endpoint_to_registration_origin_map.z(), endpoint_to_registration_origin_map.y(),
    endpoint_to_registration_origin_map.z(), 0.0, -endpoint_to_registration_origin_map.x(),
    -endpoint_to_registration_origin_map.y(), endpoint_to_registration_origin_map.x(), 0.0;
  map_increment_from_graph_residual.block<3, 3>(3, 3) =
    lever_skew * map_T_from.linear();
  result.graph_residual_hessian =
    map_increment_from_graph_residual.transpose() *
    (0.5 * (map_pose_hessian_rotation_translation +
    map_pose_hessian_rotation_translation.transpose())) *
    map_increment_from_graph_residual;
  result.graph_residual_hessian =
    0.5 * (result.graph_residual_hessian + result.graph_residual_hessian.transpose());

  // Whiten parameters by the configured graph information. This removes the
  // arbitrary meter/radian scale while preserving the established DCS/Cauchy
  // tuning. The registration Hessian can reduce weak directions but cannot
  // make any generalized direction stronger than the configured baseline.
  const RegistrationVector6d sqrt_configured =
    configured_information_diagonal.array().sqrt().matrix();
  const RegistrationVector6d inverse_sqrt_configured =
    sqrt_configured.array().inverse().matrix();
  const RegistrationMatrix6d inverse_scale = inverse_sqrt_configured.asDiagonal();
  RegistrationMatrix6d normalized_hessian =
    inverse_scale * result.graph_residual_hessian * inverse_scale;
  normalized_hessian = 0.5 * (normalized_hessian + normalized_hessian.transpose());

  Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> normalized_solver;
  if (!symmetricPositiveSemidefinite(normalized_hessian, normalized_solver)) {
    return fail("invalid_normalized_hessian");
  }
  const RegistrationVector6d eigenvalues =
    normalized_solver.eigenvalues().cwiseMax(0.0);
  const double maximum_eigenvalue = eigenvalues.maxCoeff();
  if (!std::isfinite(maximum_eigenvalue) || maximum_eigenvalue <= 0.0) {
    return fail("zero_normalized_hessian");
  }

  const double observable_threshold =
    maximum_eigenvalue * config.relative_eigen_threshold;
  const double strong_threshold =
    maximum_eigenvalue * config.strong_eigenvalue_ratio;
  double minimum_observable_eigenvalue = maximum_eigenvalue;
  RegistrationVector6d gains = RegistrationVector6d::Zero();
  for (int index = 0; index < 6; ++index) {
    result.normalized_hessian_eigenvalues[static_cast<std::size_t>(index)] =
      eigenvalues[index];
    if (eigenvalues[index] > observable_threshold) {
      ++result.observable_rank;
      minimum_observable_eigenvalue = std::min(
        minimum_observable_eigenvalue, eigenvalues[index]);
    }
    gains[index] = std::clamp(
      eigenvalues[index] / std::max(strong_threshold, 1.0e-15),
      config.weak_information_ratio,
      1.0);
    result.information_gains[static_cast<std::size_t>(index)] = gains[index];
  }
  if (result.observable_rank > 0) {
    result.observable_condition =
      maximum_eigenvalue / minimum_observable_eigenvalue;
  }
  result.minimum_information_gain = gains.minCoeff();

  const RegistrationMatrix6d normalized_information =
    normalized_solver.eigenvectors() * gains.asDiagonal() *
    normalized_solver.eigenvectors().transpose();
  const RegistrationMatrix6d configured_scale = sqrt_configured.asDiagonal();
  result.information_matrix =
    configured_scale * normalized_information * configured_scale;
  result.information_matrix =
    0.5 * (result.information_matrix + result.information_matrix.transpose());

  Eigen::SelfAdjointEigenSolver<RegistrationMatrix6d> information_solver;
  if (!symmetricPositiveSemidefinite(result.information_matrix, information_solver) ||
    information_solver.eigenvalues().maxCoeff() <= 0.0)
  {
    return fail("invalid_output_information");
  }

  result.valid = true;
  return result;
}

}  // namespace slam_3d_core
}  // namespace gole
