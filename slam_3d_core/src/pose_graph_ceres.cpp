// Copyright 2026 Gole Robotics

#include "pose_graph_backend.hpp"

#include <array>
#include <cmath>

#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

namespace gole
{
namespace slam_3d_core
{
namespace detail
{
namespace
{

struct PoseParameters
{
  std::array<double, 3> position{};
  // Eigen quaternion storage is [x, y, z, w].
  std::array<double, 4> quaternion{};
};

class RelativePoseCost
{
public:
  RelativePoseCost(const Eigen::Isometry3d & measurement, const Matrix6d & sqrt_information)
  : measurement_(measurement), sqrt_information_(sqrt_information) {}

  template<typename T>
  bool operator()(
    const T * const from_position, const T * const from_quaternion,
    const T * const to_position, const T * const to_quaternion, T * residual_ptr) const
  {
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(residual_ptr);
    residual = sqrt_information_.template cast<T>() * relativePoseResidual<T>(
      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(from_position),
      Eigen::Map<const Eigen::Quaternion<T>>(from_quaternion),
      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(to_position),
      Eigen::Map<const Eigen::Quaternion<T>>(to_quaternion), measurement_);
    return true;
  }

private:
  Eigen::Isometry3d measurement_;
  Matrix6d sqrt_information_;
};

}  // namespace

StageSolveResult solveCeresStage(
  const PoseGraphOptimizationConfig & config,
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information, const double cauchy_scale)
{
  StageSolveResult result;
  result.poses = poses;
  std::vector<PoseParameters> parameters(poses.size());
  for (std::size_t index = 0; index < poses.size(); ++index) {
    auto quaternion = Eigen::Quaterniond(poses[index].linear()).normalized();
    if (quaternion.w() < 0.0) {
      quaternion.coeffs() *= -1.0;
    }
    const auto position = poses[index].translation();
    parameters[index].position = {position.x(), position.y(), position.z()};
    parameters[index].quaternion = {
      quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()};
  }
  ceres::Problem problem;
  for (auto & parameter : parameters) {
    problem.AddParameterBlock(parameter.position.data(), 3);
    problem.AddParameterBlock(
      parameter.quaternion.data(), 4, new ceres::EigenQuaternionParameterization());
  }
  problem.SetParameterBlockConstant(parameters.front().position.data());
  problem.SetParameterBlockConstant(parameters.front().quaternion.data());
  for (std::size_t index = 0; index < constraints.size(); ++index) {
    const auto & edge = constraints[index];
    auto * cost = new ceres::AutoDiffCostFunction<RelativePoseCost, 6, 3, 4, 3, 4>(
      new RelativePoseCost(edge.from_T_to, sqrt_information[index]));
    ceres::LossFunction * loss = nullptr;
    if (edge.loop_closure && config.robust_loop_enabled) {
      loss = new ceres::CauchyLoss(cauchy_scale);
    }
    auto & from = parameters[static_cast<std::size_t>(edge.from_index)];
    auto & to = parameters[static_cast<std::size_t>(edge.to_index)];
    problem.AddResidualBlock(
      cost, loss, from.position.data(), from.quaternion.data(),
      to.position.data(), to.quaternion.data());
  }
  ceres::Solver::Options options;
  options.max_num_iterations = config.max_iterations_per_stage;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.function_tolerance = config.function_tolerance;
  options.gradient_tolerance = config.gradient_tolerance;
  options.parameter_tolerance = config.parameter_tolerance;
  options.num_threads = config.num_threads;
  options.use_nonmonotonic_steps = config.use_nonmonotonic_steps;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  result.stats.iterations = static_cast<int>(summary.iterations.size());
  result.stats.successful_steps = summary.num_successful_steps;
  result.stats.unsuccessful_steps = summary.num_unsuccessful_steps;
  result.stats.step_counts_available = true;
  result.stats.initial_cost = summary.initial_cost;
  result.stats.final_cost = summary.final_cost;
  if (!summary.iterations.empty()) {
    result.stats.final_gradient_max_norm = summary.iterations.back().gradient_max_norm;
    result.stats.gradient_available = true;
  }
  result.stats.termination_type = ceres::TerminationTypeToString(summary.termination_type);
  result.stats.brief_report = summary.BriefReport();
  if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost)) {
    result.failure_reason = summary.message;
    return result;
  }
  // Preserve the gauge bit-for-bit; it never participated in the solve.
  for (std::size_t index = 1; index < parameters.size(); ++index) {
    result.poses[index].translation() = Eigen::Map<const Eigen::Vector3d>(
      parameters[index].position.data());
    result.poses[index].linear() = Eigen::Map<const Eigen::Quaterniond>(
      parameters[index].quaternion.data()).normalized().toRotationMatrix();
  }
  result.success = true;
  return result;
}

}  // namespace detail
}  // namespace slam_3d_core
}  // namespace gole
