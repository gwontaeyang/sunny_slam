// Copyright 2026 Gole Robotics

#include "slam_3d_core/pose_graph_optimizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>

#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include <Eigen/Eigenvalues>

#include "slam_3d_core/rigid_transform.hpp"

namespace gole
{
namespace slam_3d_core
{

namespace
{

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

struct PoseParameters
{
  std::array<double, 3> position{};
  // Eigen and ceres::EigenQuaternionParameterization use [x, y, z, w].
  std::array<double, 4> quaternion{};
};

Eigen::Quaterniond canonicalQuaternion(const Eigen::Matrix3d & rotation)
{
  Eigen::Quaterniond quaternion(rotation);
  quaternion.normalize();
  if (quaternion.w() < 0.0) {
    quaternion.coeffs() *= -1.0;
  }
  return quaternion;
}

PoseParameters toParameters(const Eigen::Isometry3d & pose)
{
  const Eigen::Quaterniond quaternion = canonicalQuaternion(pose.linear());
  PoseParameters parameters;
  parameters.position = {
    pose.translation().x(), pose.translation().y(), pose.translation().z()};
  parameters.quaternion = {
    quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
  return parameters;
}

Eigen::Isometry3d fromParameters(const PoseParameters & parameters)
{
  Eigen::Map<const Eigen::Quaterniond> quaternion(parameters.quaternion.data());
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(
    parameters.position[0], parameters.position[1], parameters.position[2]);
  pose.linear() = quaternion.normalized().toRotationMatrix();
  return projectToRigidIsometry(pose);
}

std::vector<Eigen::Isometry3d> parametersToPoses(
  const std::vector<PoseParameters> & parameters)
{
  std::vector<Eigen::Isometry3d> poses;
  poses.reserve(parameters.size());
  for (const auto & pose_parameters : parameters) {
    poses.push_back(fromParameters(pose_parameters));
  }
  return poses;
}

bool finitePose(const Eigen::Isometry3d & pose)
{
  return pose.matrix().allFinite() &&
         pose.translation().allFinite() &&
         std::isfinite(Eigen::Quaterniond(pose.linear()).norm());
}

bool validInformationMatrix(const Matrix6d & information)
{
  if (!information.allFinite()) {
    return false;
  }
  const double scale = std::max(1.0, information.norm());
  if ((information - information.transpose()).norm() > 1.0e-9 * scale) {
    return false;
  }

  const Matrix6d symmetric = 0.5 * (information + information.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6d> solver(symmetric);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return false;
  }
  const double maximum = solver.eigenvalues().maxCoeff();
  const double tolerance = 1.0e-10 * std::max(1.0, std::abs(maximum));
  return maximum > tolerance && solver.eigenvalues().minCoeff() >= -tolerance;
}

Matrix6d symmetricSquareRoot(const Matrix6d & information)
{
  const Matrix6d symmetric = 0.5 * (information + information.transpose());
  const Eigen::SelfAdjointEigenSolver<Matrix6d> solver(symmetric);
  const Eigen::Matrix<double, 6, 1> square_root_eigenvalues =
    solver.eigenvalues().cwiseMax(0.0).cwiseSqrt();
  return solver.eigenvectors() * square_root_eigenvalues.asDiagonal() *
         solver.eigenvectors().transpose();
}

bool validConstraint(
  const PoseGraphConstraint & constraint,
  const std::size_t pose_count)
{
  return constraint.from_index >= 0 && constraint.to_index >= 0 &&
         constraint.from_index != constraint.to_index &&
         constraint.from_index < static_cast<int>(pose_count) &&
         constraint.to_index < static_cast<int>(pose_count) &&
         finitePose(constraint.from_T_to) &&
         validInformationMatrix(constraint.information_matrix);
}

bool connectedToGauge(
  const std::size_t pose_count,
  const std::vector<PoseGraphConstraint> & constraints)
{
  std::vector<std::vector<int>> adjacency(pose_count);
  for (const auto & constraint : constraints) {
    adjacency[static_cast<std::size_t>(constraint.from_index)].push_back(
      constraint.to_index);
    adjacency[static_cast<std::size_t>(constraint.to_index)].push_back(
      constraint.from_index);
  }

  std::vector<bool> visited(pose_count, false);
  std::queue<int> queue;
  queue.push(0);
  visited.front() = true;
  while (!queue.empty()) {
    const int index = queue.front();
    queue.pop();
    for (const int neighbor : adjacency[static_cast<std::size_t>(index)]) {
      if (!visited[static_cast<std::size_t>(neighbor)]) {
        visited[static_cast<std::size_t>(neighbor)] = true;
        queue.push(neighbor);
      }
    }
  }
  return std::all_of(visited.begin(), visited.end(), [](const bool value) {return value;});
}

class RelativePoseResidual
{
public:
  RelativePoseResidual(
    const Eigen::Isometry3d & measured_from_T_to,
    const Matrix6d & sqrt_information)
  : measured_translation_(measured_from_T_to.translation()),
    measured_rotation_(canonicalQuaternion(measured_from_T_to.linear())),
    sqrt_information_(sqrt_information)
  {
  }

  template<typename T>
  bool operator()(
    const T * const from_position_ptr,
    const T * const from_quaternion_ptr,
    const T * const to_position_ptr,
    const T * const to_quaternion_ptr,
    T * residual_ptr) const
  {
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> from_position(from_position_ptr);
    const Eigen::Map<const Eigen::Quaternion<T>> from_rotation(from_quaternion_ptr);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> to_position(to_position_ptr);
    const Eigen::Map<const Eigen::Quaternion<T>> to_rotation(to_quaternion_ptr);

    const Eigen::Quaternion<T> inverse_from_rotation = from_rotation.conjugate();
    const Eigen::Matrix<T, 3, 1> estimated_translation =
      inverse_from_rotation * (to_position - from_position);
    const Eigen::Quaternion<T> estimated_rotation = inverse_from_rotation * to_rotation;
    const Eigen::Quaternion<T> rotation_error =
      measured_rotation_.template cast<T>() * estimated_rotation.conjugate();

    Eigen::Matrix<T, 6, 1> raw_residual;
    raw_residual.template head<3>() =
      estimated_translation - measured_translation_.template cast<T>();
    // This is the standard multiplicative SO(3) tangent approximation used
    // by the official Ceres 3D pose-graph example.
    raw_residual.template tail<3>() = T(2.0) * rotation_error.vec();
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual = sqrt_information_.template cast<T>() * raw_residual;
    return true;
  }

private:
  Eigen::Vector3d measured_translation_;
  Eigen::Quaterniond measured_rotation_;
  Matrix6d sqrt_information_;
};

std::vector<double> computeDcsScales(
  const PoseGraphOptimizationConfig & config,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Eigen::Isometry3d> & poses,
  int & downweighted_loops,
  double & min_scale,
  double & mean_scale)
{
  std::vector<double> scales(constraints.size(), 1.0);
  downweighted_loops = 0;
  min_scale = 1.0;
  double scale_sum = 0.0;
  int loop_count = 0;
  for (std::size_t index = 0; index < constraints.size(); ++index) {
    if (!constraints[index].loop_closure) {
      continue;
    }
    ++loop_count;
    const double chi_squared = poseGraphConstraintChiSquared(constraints[index], poses);
    const double scale = std::clamp(
      dynamicCovarianceScale(chi_squared, config.dcs_phi),
      config.dcs_min_scale,
      1.0);
    scales[index] = scale;
    scale_sum += scale;
    min_scale = std::min(min_scale, scale);
    if (scale < 0.999) {
      ++downweighted_loops;
    }
  }
  mean_scale = loop_count > 0 ? scale_sum / static_cast<double>(loop_count) : 1.0;
  return scales;
}

}  // namespace

bool validPoseGraphOptimizationConfig(
  const PoseGraphOptimizationConfig & config) noexcept
{
  if (config.max_iterations_per_stage <= 0 ||
    config.robust_scale_factors.empty() ||
    !std::isfinite(config.loop_cauchy_scale) || config.loop_cauchy_scale <= 0.0 ||
    config.dcs_start_stage < 0 ||
    !std::isfinite(config.dcs_phi) || config.dcs_phi <= 0.0 ||
    !std::isfinite(config.dcs_min_scale) || config.dcs_min_scale < 0.0 ||
    config.dcs_min_scale > 1.0 ||
    !std::isfinite(config.function_tolerance) || config.function_tolerance < 0.0 ||
    !std::isfinite(config.gradient_tolerance) || config.gradient_tolerance < 0.0 ||
    !std::isfinite(config.parameter_tolerance) || config.parameter_tolerance < 0.0 ||
    config.num_threads <= 0)
  {
    return false;
  }
  if (config.dcs_enabled &&
    config.dcs_start_stage >= static_cast<int>(config.robust_scale_factors.size()))
  {
    return false;
  }
  return std::all_of(
    config.robust_scale_factors.begin(), config.robust_scale_factors.end(),
    [](const double scale) {return std::isfinite(scale) && scale > 0.0;});
}

double poseGraphConstraintChiSquared(
  const PoseGraphConstraint & constraint,
  const std::vector<Eigen::Isometry3d> & poses) noexcept
{
  if (!validConstraint(constraint, poses.size())) {
    return std::numeric_limits<double>::infinity();
  }
  const auto & from = poses[static_cast<std::size_t>(constraint.from_index)];
  const auto & to = poses[static_cast<std::size_t>(constraint.to_index)];
  const Eigen::Quaterniond from_rotation = canonicalQuaternion(from.linear());
  const Eigen::Quaterniond to_rotation = canonicalQuaternion(to.linear());
  const Eigen::Quaterniond measured_rotation =
    canonicalQuaternion(constraint.from_T_to.linear());
  const Eigen::Vector3d estimated_translation =
    from_rotation.conjugate() * (to.translation() - from.translation());
  const Eigen::Quaterniond estimated_rotation = from_rotation.conjugate() * to_rotation;
  Eigen::Quaterniond rotation_error = measured_rotation * estimated_rotation.conjugate();
  if (rotation_error.w() < 0.0) {
    rotation_error.coeffs() *= -1.0;
  }

  Vector6d residual;
  residual.head<3>() = estimated_translation - constraint.from_T_to.translation();
  residual.tail<3>() = 2.0 * rotation_error.vec();
  const double chi_squared =
    residual.dot(constraint.information_matrix * residual);
  return std::isfinite(chi_squared) ? chi_squared :
         std::numeric_limits<double>::infinity();
}

double dynamicCovarianceScale(const double chi_squared, const double phi) noexcept
{
  if (!std::isfinite(chi_squared) || chi_squared < 0.0 ||
    !std::isfinite(phi) || phi <= 0.0)
  {
    return 0.0;
  }
  return std::min(1.0, (2.0 * phi) / (phi + chi_squared));
}

PoseGraphOptimizer::PoseGraphOptimizer(PoseGraphOptimizationConfig config)
: config_(std::move(config))
{
  if (!validPoseGraphOptimizationConfig(config_)) {
    throw std::invalid_argument("invalid pose graph optimization config");
  }
}

const PoseGraphOptimizationConfig & PoseGraphOptimizer::config() const noexcept
{
  return config_;
}

void PoseGraphOptimizer::setConfig(const PoseGraphOptimizationConfig & config)
{
  if (!validPoseGraphOptimizationConfig(config)) {
    throw std::invalid_argument("invalid pose graph optimization config");
  }
  config_ = config;
}

PoseGraphOptimizationResult PoseGraphOptimizer::optimize(
  const std::vector<Eigen::Isometry3d> & initial_poses,
  const std::vector<PoseGraphConstraint> & constraints) const
{
  const auto optimization_started = std::chrono::steady_clock::now();
  PoseGraphOptimizationResult result;
  result.poses = initial_poses;
  result.final_dcs_scales.assign(constraints.size(), 1.0);
  auto fail = [&](const std::string & reason) {
      result.success = false;
      result.failure_reason = reason;
      result.poses = initial_poses;
      result.runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - optimization_started).count();
      return result;
    };

  if (!validPoseGraphOptimizationConfig(config_)) {
    return fail("invalid_config");
  }
  if (initial_poses.size() < 2U || constraints.empty()) {
    return fail("insufficient_graph");
  }
  if (!std::all_of(initial_poses.begin(), initial_poses.end(), finitePose)) {
    return fail("nonfinite_initial_pose");
  }
  if (!std::all_of(
      constraints.begin(), constraints.end(),
      [&](const PoseGraphConstraint & constraint) {
        return validConstraint(constraint, initial_poses.size());
      }))
  {
    return fail("invalid_constraint");
  }
  if (!connectedToGauge(initial_poses.size(), constraints)) {
    return fail("disconnected_graph");
  }

  std::vector<PoseParameters> parameters;
  parameters.reserve(initial_poses.size());
  for (const auto & pose : initial_poses) {
    parameters.push_back(toParameters(projectToRigidIsometry(pose)));
  }
  std::vector<double> dcs_scales(constraints.size(), 1.0);

  for (std::size_t stage_index = 0;
    stage_index < config_.robust_scale_factors.size(); ++stage_index)
  {
    const auto stage_started = std::chrono::steady_clock::now();
    PoseGraphOptimizationStageStats stage;
    stage.stage_index = static_cast<int>(stage_index);
    stage.robust_scale_factor = config_.robust_scale_factors[stage_index];
    stage.cauchy_scale = config_.loop_cauchy_scale * stage.robust_scale_factor;

    const std::vector<Eigen::Isometry3d> current_poses = parametersToPoses(parameters);
    if (config_.dcs_enabled &&
      static_cast<int>(stage_index) >= config_.dcs_start_stage)
    {
      dcs_scales = computeDcsScales(
        config_, constraints, current_poses,
        stage.dcs_downweighted_loops,
        stage.min_dcs_scale,
        stage.mean_dcs_scale);
    }

    ceres::Problem problem;
    for (std::size_t index = 0; index < parameters.size(); ++index) {
      problem.AddParameterBlock(parameters[index].position.data(), 3);
      problem.AddParameterBlock(
        parameters[index].quaternion.data(), 4,
        new ceres::EigenQuaternionParameterization());
    }
    problem.SetParameterBlockConstant(parameters.front().position.data());
    problem.SetParameterBlockConstant(parameters.front().quaternion.data());

    for (std::size_t index = 0; index < constraints.size(); ++index) {
      const auto & constraint = constraints[index];
      Matrix6d sqrt_information = symmetricSquareRoot(
        constraint.information_matrix);
      if (constraint.loop_closure && config_.dcs_enabled) {
        // DCS scales the information matrix by s^2, so its square root by s.
        sqrt_information *= dcs_scales[index];
      }
      auto * cost_function =
        new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 3, 4, 3, 4>(
        new RelativePoseResidual(constraint.from_T_to, sqrt_information));
      ceres::LossFunction * loss_function = nullptr;
      if (constraint.loop_closure && config_.robust_loop_enabled) {
        loss_function = new ceres::CauchyLoss(stage.cauchy_scale);
      }
      problem.AddResidualBlock(
        cost_function,
        loss_function,
        parameters[static_cast<std::size_t>(constraint.from_index)].position.data(),
        parameters[static_cast<std::size_t>(constraint.from_index)].quaternion.data(),
        parameters[static_cast<std::size_t>(constraint.to_index)].position.data(),
        parameters[static_cast<std::size_t>(constraint.to_index)].quaternion.data());
    }

    ceres::Solver::Options options;
    options.max_num_iterations = config_.max_iterations_per_stage;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.function_tolerance = config_.function_tolerance;
    options.gradient_tolerance = config_.gradient_tolerance;
    options.parameter_tolerance = config_.parameter_tolerance;
    options.num_threads = config_.num_threads;
    options.use_nonmonotonic_steps = config_.use_nonmonotonic_steps;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    stage.iterations = static_cast<int>(summary.iterations.size());
    stage.successful_steps = summary.num_successful_steps;
    stage.unsuccessful_steps = summary.num_unsuccessful_steps;
    stage.initial_cost = summary.initial_cost;
    stage.final_cost = summary.final_cost;
    if (!summary.iterations.empty()) {
      stage.final_gradient_max_norm = summary.iterations.back().gradient_max_norm;
    }
    stage.termination_type = ceres::TerminationTypeToString(summary.termination_type);
    stage.brief_report = summary.BriefReport();
    stage.runtime_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - stage_started).count();
    result.stages.push_back(stage);
    result.total_iterations += stage.iterations;
    result.successful_steps += std::max(0, stage.successful_steps);
    result.unsuccessful_steps += std::max(0, stage.unsuccessful_steps);
    if (stage_index == 0U) {
      result.initial_cost = stage.initial_cost;
    }
    result.final_cost = stage.final_cost;
    result.final_gradient_max_norm = stage.final_gradient_max_norm;
    result.termination_type = stage.termination_type;

    if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost)) {
      return fail("ceres_stage_" + std::to_string(stage_index) + "_failed");
    }
  }

  result.poses = parametersToPoses(parameters);
  if (!std::all_of(result.poses.begin(), result.poses.end(), finitePose)) {
    return fail("nonfinite_optimized_pose");
  }
  if (config_.dcs_enabled) {
    double mean_scale = 1.0;
    result.final_dcs_scales = computeDcsScales(
      config_, constraints, result.poses,
      result.dcs_downweighted_loops,
      result.min_dcs_scale,
      mean_scale);
  }
  result.success = true;
  result.runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - optimization_started).count();
  return result;
}

}  // namespace slam_3d_core
}  // namespace gole
