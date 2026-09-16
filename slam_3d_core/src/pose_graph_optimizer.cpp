// Copyright 2026 Gole Robotics

#include "slam_3d_core/pose_graph_optimizer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <stdexcept>
#include <utility>

#include <Eigen/Eigenvalues>

#include "pose_graph_backend.hpp"
#include "slam_3d_core/rigid_transform.hpp"

namespace gole
{
namespace slam_3d_core
{
namespace
{

using detail::Matrix6d;

bool finitePose(const Eigen::Isometry3d & pose)
{
  return pose.matrix().allFinite() &&
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
  return solver.eigenvectors() * solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal() *
         solver.eigenvectors().transpose();
}

bool validConstraint(const PoseGraphConstraint & constraint, const std::size_t pose_count)
{
  return constraint.from_index >= 0 && constraint.to_index >= 0 &&
         constraint.from_index != constraint.to_index &&
         static_cast<std::size_t>(constraint.from_index) < pose_count &&
         static_cast<std::size_t>(constraint.to_index) < pose_count &&
         finitePose(constraint.from_T_to) && validInformationMatrix(constraint.information_matrix);
}

bool connectedToGauge(
  const std::size_t pose_count, const std::vector<PoseGraphConstraint> & constraints)
{
  std::vector<std::vector<int>> adjacency(pose_count);
  for (const auto & constraint : constraints) {
    adjacency[static_cast<std::size_t>(constraint.from_index)].push_back(constraint.to_index);
    adjacency[static_cast<std::size_t>(constraint.to_index)].push_back(constraint.from_index);
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

std::vector<double> computeDcsScales(
  const PoseGraphOptimizationConfig & config,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Eigen::Isometry3d> & poses,
  int & downweighted_loops, double & min_scale, double & mean_scale)
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
      dynamicCovarianceScale(chi_squared, config.dcs_phi), config.dcs_min_scale, 1.0);
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

// Report the same half-rho objective across libraries (g2o reports full rho).
double stageObjective(
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information,
  const bool robust_loop_enabled, const double cauchy_scale)
{
  double cost = 0.0;
  for (std::size_t index = 0; index < constraints.size(); ++index) {
    const auto & edge = constraints[index];
    const double squared = (sqrt_information[index] * detail::relativePoseResidual(
        poses[static_cast<std::size_t>(edge.from_index)],
        poses[static_cast<std::size_t>(edge.to_index)], edge.from_T_to)).squaredNorm();
    const double scale_squared = cauchy_scale * cauchy_scale;
    cost += 0.5 * (robust_loop_enabled && edge.loop_closure ?
      scale_squared * std::log1p(squared / scale_squared) : squared);
  }
  return cost;
}

}  // namespace

std::vector<std::string> availablePoseGraphSolvers()
{
  std::vector<std::string> solvers{"ceres_se3"};
#ifdef SLAM_3D_HAS_G2O
  solvers.emplace_back("g2o_se3");
#endif
#ifdef SLAM_3D_HAS_GTSAM
  solvers.emplace_back("gtsam_se3");
#endif
  return solvers;
}

bool poseGraphSolverAvailable(const std::string & backend) noexcept
{
  if (backend == "ceres_se3") {
    return true;
  }
#ifdef SLAM_3D_HAS_G2O
  if (backend == "g2o_se3") {
    return true;
  }
#endif
#ifdef SLAM_3D_HAS_GTSAM
  if (backend == "gtsam_se3") {
    return true;
  }
#endif
  return false;
}

bool validPoseGraphOptimizationConfig(const PoseGraphOptimizationConfig & config) noexcept
{
  if ((config.backend != "ceres_se3" && config.backend != "g2o_se3" &&
    config.backend != "gtsam_se3") || config.max_iterations_per_stage <= 0 ||
    config.robust_scale_factors.empty() ||
    !std::isfinite(config.loop_cauchy_scale) || config.loop_cauchy_scale <= 0.0 ||
    config.dcs_start_stage < 0 || !std::isfinite(config.dcs_phi) || config.dcs_phi <= 0.0 ||
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
    [&](const double scale) {
      const double squared = std::pow(scale * config.loop_cauchy_scale, 2);
      return std::isfinite(scale) && scale > 0.0 && std::isfinite(squared) && squared > 0.0;
    });
}

double poseGraphConstraintChiSquared(
  const PoseGraphConstraint & constraint,
  const std::vector<Eigen::Isometry3d> & poses) noexcept
{
  if (!validConstraint(constraint, poses.size())) {
    return std::numeric_limits<double>::infinity();
  }
  const auto residual = detail::relativePoseResidual(
    poses[static_cast<std::size_t>(constraint.from_index)],
    poses[static_cast<std::size_t>(constraint.to_index)], constraint.from_T_to);
  const double chi_squared = residual.dot(constraint.information_matrix * residual);
  return std::isfinite(chi_squared) ? chi_squared : std::numeric_limits<double>::infinity();
}

double dynamicCovarianceScale(const double chi_squared, const double phi) noexcept
{
  if (!std::isfinite(chi_squared) || chi_squared < 0.0 || !std::isfinite(phi) || phi <= 0.0) {
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
  result.backend = config_.backend;
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
  if (!poseGraphSolverAvailable(config_.backend)) {
    return fail("backend_unavailable:" + config_.backend);
  }
  if (initial_poses.size() < 2U || constraints.empty()) {
    return fail("insufficient_graph");
  }
  if (!std::all_of(initial_poses.begin(), initial_poses.end(), finitePose)) {
    return fail("nonfinite_initial_pose");
  }
  if (!std::all_of(
      constraints.begin(), constraints.end(),
      [&](const PoseGraphConstraint & edge) {return validConstraint(edge, initial_poses.size());}))
  {
    return fail("invalid_constraint");
  }
  if (!connectedToGauge(initial_poses.size(), constraints)) {
    return fail("disconnected_graph");
  }
  std::vector<Eigen::Isometry3d> current_poses;
  for (const auto & pose : initial_poses) {
    current_poses.push_back(projectToRigidIsometry(pose));
  }
  std::vector<Matrix6d> base_sqrt_information;
  for (const auto & edge : constraints) {
    base_sqrt_information.push_back(symmetricSquareRoot(edge.information_matrix));
  }
  std::vector<double> dcs_scales(constraints.size(), 1.0);
  for (std::size_t index = 0; index < config_.robust_scale_factors.size(); ++index) {
    const auto stage_started = std::chrono::steady_clock::now();
    PoseGraphOptimizationStageStats stage;
    stage.stage_index = static_cast<int>(index);
    stage.robust_scale_factor = config_.robust_scale_factors[index];
    stage.cauchy_scale = config_.loop_cauchy_scale * stage.robust_scale_factor;
    if (config_.dcs_enabled && static_cast<int>(index) >= config_.dcs_start_stage) {
      dcs_scales = computeDcsScales(
        config_, constraints, current_poses, stage.dcs_downweighted_loops,
        stage.min_dcs_scale, stage.mean_dcs_scale);
    }
    auto sqrt_information = base_sqrt_information;
    for (std::size_t edge = 0; edge < constraints.size(); ++edge) {
      sqrt_information[edge] *= dcs_scales[edge];
    }
    stage.initial_cost = stageObjective(
      current_poses, constraints, sqrt_information, config_.robust_loop_enabled,
      stage.cauchy_scale);
    if (!std::isfinite(stage.initial_cost)) {
      return fail("nonfinite_initial_cost");
    }
    detail::StageSolveResult solved;
    try {
      if (config_.backend == "ceres_se3") {
        solved = detail::solveCeresStage(
          config_, current_poses, constraints, sqrt_information, stage.cauchy_scale);
      }
#ifdef SLAM_3D_HAS_G2O
      else if (config_.backend == "g2o_se3") {
        solved = detail::solveG2oStage(
          config_, current_poses, constraints, sqrt_information, stage.cauchy_scale);
      }
#endif
#ifdef SLAM_3D_HAS_GTSAM
      else if (config_.backend == "gtsam_se3") {
        solved = detail::solveGtsamStage(
          config_, current_poses, constraints, sqrt_information, stage.cauchy_scale);
      }
#endif
    } catch (const std::exception & error) {
      return fail(config_.backend + "_exception:" + error.what());
    }
    stage.iterations = solved.stats.iterations;
    stage.successful_steps = solved.stats.successful_steps;
    stage.unsuccessful_steps = solved.stats.unsuccessful_steps;
    stage.step_counts_available = solved.stats.step_counts_available;
    stage.gradient_available = solved.stats.gradient_available;
    stage.final_gradient_max_norm = solved.stats.final_gradient_max_norm;
    stage.termination_type = solved.stats.termination_type;
    stage.brief_report = solved.stats.brief_report;
    const bool finite_solution = solved.poses.size() == current_poses.size() &&
      std::all_of(solved.poses.begin(), solved.poses.end(), finitePose);
    if (!solved.success) {
      // Adapters roll poses back on failure. Preserve their last native cost
      // (all normalized to half-rho), not a cost evaluated on rollback poses.
      stage.final_cost = solved.stats.final_cost;
    } else if (finite_solution) {
      stage.final_cost = stageObjective(
        solved.poses, constraints, sqrt_information, config_.robust_loop_enabled,
        stage.cauchy_scale);
    }
    stage.runtime_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - stage_started).count();
    result.stages.push_back(stage);
    result.total_iterations += stage.iterations;
    result.successful_steps += std::max(0, stage.successful_steps);
    result.unsuccessful_steps += std::max(0, stage.unsuccessful_steps);
    if (index == 0U) {
      result.initial_cost = stage.initial_cost;
    }
    result.final_cost = stage.final_cost;
    result.final_gradient_max_norm = stage.final_gradient_max_norm;
    result.termination_type = stage.termination_type;
    if (!solved.success || !finite_solution || !std::isfinite(stage.final_cost)) {
      return fail(
        config_.backend + "_stage_" + std::to_string(index) +
        "_failed:" + solved.failure_reason);
    }
    if (stage.final_cost > stage.initial_cost + 1.0e-8 * std::max(1.0, stage.initial_cost)) {
      return fail(config_.backend + "_cost_increased");
    }
    if (!solved.poses.front().matrix().isApprox(current_poses.front().matrix(), 1.0e-10)) {
      return fail(config_.backend + "_gauge_changed");
    }
    current_poses = std::move(solved.poses);
  }
  result.poses = std::move(current_poses);
  if (config_.dcs_enabled) {
    double mean_scale = 1.0;
    result.final_dcs_scales = computeDcsScales(
      config_, constraints, result.poses, result.dcs_downweighted_loops,
      result.min_dcs_scale, mean_scale);
  }
  result.success = true;
  result.runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - optimization_started).count();
  return result;
}

}  // namespace slam_3d_core
}  // namespace gole
