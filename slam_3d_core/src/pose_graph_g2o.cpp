// Copyright 2026 Gole Robotics

#include "pose_graph_backend.hpp"

#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace gole
{
namespace slam_3d_core
{
namespace detail
{
namespace
{

class RelativePoseEdge final
  : public g2o::BaseBinaryEdge<6, Eigen::Isometry3d, g2o::VertexSE3, g2o::VertexSE3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativePoseEdge(const Eigen::Isometry3d & measurement, const Matrix6d & sqrt_information)
  : sqrt_information_(sqrt_information)
  {
    setMeasurement(measurement);
    setInformation(Matrix6d::Identity());
  }

  void computeError() override
  {
    _error = residual(vertexXn<0>()->estimate(), vertexXn<1>()->estimate());
  }

  void linearizeOplus() override
  {
    // Correctness-first central differences in VertexSE3's right MQT chart.
    // Perturb copies to avoid mutating shared vertices/caches during linearization.
    constexpr double delta = 1.0e-6;
    const Eigen::Isometry3d from = vertexXn<0>()->estimate();
    const Eigen::Isometry3d to = vertexXn<1>()->estimate();
    for (int column = 0; column < 6; ++column) {
      Vector6d update = Vector6d::Zero();
      update[column] = delta;
      const Eigen::Isometry3d plus = g2o::internal::fromVectorMQT(update);
      update[column] = -delta;
      const Eigen::Isometry3d minus = g2o::internal::fromVectorMQT(update);
      if (!vertexXn<0>()->fixed()) {
        _jacobianOplusXi.col(column) =
          (residual(from * plus, to) - residual(from * minus, to)) / (2.0 * delta);
      }
      if (!vertexXn<1>()->fixed()) {
        _jacobianOplusXj.col(column) =
          (residual(from, to * plus) - residual(from, to * minus)) / (2.0 * delta);
      }
    }
  }

  bool read(std::istream &) override {return false;}
  bool write(std::ostream &) const override {return false;}

private:
  Vector6d residual(const Eigen::Isometry3d & from, const Eigen::Isometry3d & to) const
  {
    return sqrt_information_ * relativePoseResidual(from, to, _measurement);
  }

  Matrix6d sqrt_information_;
};

// SparseOptimizer::optimize() returns an iteration count, not the LM status.
// Observe the real status and only claim convergence for an explicit cost test.
class ObservedLevenberg final : public g2o::OptimizationAlgorithmLevenberg
{
public:
  ObservedLevenberg(std::unique_ptr<g2o::Solver> solver, double function_tolerance)
  : g2o::OptimizationAlgorithmLevenberg(std::move(solver)),
    function_tolerance_(function_tolerance)
  {}

  SolverResult solve(int iteration, bool online = false) override
  {
    optimizer()->computeActiveErrors();
    const double before = optimizer()->activeRobustChi2();
    ++iterations;
    native_result = g2o::OptimizationAlgorithmLevenberg::solve(iteration, online);
    optimizer()->computeActiveErrors();
    const double after = optimizer()->activeRobustChi2();
    if (!std::isfinite(after) || after > before) {
      invalid_cost = true;
      return Fail;
    }
    if (native_result == OK &&
      (after == 0.0 || (before > after && function_tolerance_ > 0.0 &&
      (before - after) / before <= function_tolerance_)))
    {
      cost_converged = true;
      return Terminate;
    }
    return native_result;
  }

  int iterations{0};
  SolverResult native_result{Fail};
  bool cost_converged{false};
  bool invalid_cost{false};

private:
  double function_tolerance_;
};

}  // namespace

StageSolveResult solveG2oStage(
  const PoseGraphOptimizationConfig & config,
  const std::vector<Eigen::Isometry3d> & poses,
  const std::vector<PoseGraphConstraint> & constraints,
  const std::vector<Matrix6d> & sqrt_information, double cauchy_scale)
{
  const auto start = std::chrono::steady_clock::now();
  StageSolveResult result;
  result.poses = poses;
  result.stats.cauchy_scale = cauchy_scale;
  result.stats.gradient_available = false;
  result.stats.step_counts_available = false;
  std::string reason;
  try {
    if (poses.size() < 2 || constraints.empty() ||
      sqrt_information.size() != constraints.size() || config.max_iterations_per_stage <= 0)
    {
      throw std::invalid_argument("invalid g2o stage dimensions or iteration limit");
    }
    g2o::SparseOptimizer optimizer;
    using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>>;
    auto linear_solver = std::make_unique<g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>>();
    auto block_solver = std::make_unique<BlockSolver>(std::move(linear_solver));
    auto algorithm = std::make_unique<ObservedLevenberg>(
      std::move(block_solver), config.function_tolerance);
    auto * observed = algorithm.get();
    optimizer.setAlgorithm(algorithm.release());
    optimizer.setVerbose(false);
    for (std::size_t i = 0; i < poses.size(); ++i) {
      auto vertex = std::make_unique<g2o::VertexSE3>();
      vertex->setId(static_cast<int>(i));
      vertex->setEstimate(poses[i]);
      vertex->setFixed(i == 0);
      if (!optimizer.addVertex(vertex.get())) {
        throw std::runtime_error("g2o could not add a pose vertex");
      }
      vertex.release();
    }
    for (std::size_t i = 0; i < constraints.size(); ++i) {
      const auto & constraint = constraints[i];
      if (constraint.from_index < 0 || constraint.to_index < 0 ||
        static_cast<std::size_t>(constraint.from_index) >= poses.size() ||
        static_cast<std::size_t>(constraint.to_index) >= poses.size() ||
        constraint.from_index == constraint.to_index)
      {
        throw std::invalid_argument("invalid g2o constraint endpoints");
      }
      auto edge = std::make_unique<RelativePoseEdge>(constraint.from_T_to, sqrt_information[i]);
      edge->setVertex(0, optimizer.vertex(constraint.from_index));
      edge->setVertex(1, optimizer.vertex(constraint.to_index));
      if (config.robust_loop_enabled && constraint.loop_closure) {
        if (!std::isfinite(cauchy_scale) || cauchy_scale <= 0.0) {
          throw std::invalid_argument("invalid g2o Cauchy scale");
        }
        auto kernel = std::make_unique<g2o::RobustKernelCauchy>();
        kernel->setDelta(cauchy_scale);
        edge->setRobustKernel(kernel.release());
      }
      if (!optimizer.addEdge(edge.get())) {
        throw std::runtime_error("g2o could not add a relative pose edge");
      }
      edge.release();
    }
    if (!optimizer.initializeOptimization()) {
      throw std::runtime_error("g2o initializeOptimization failed");
    }
    optimizer.computeActiveErrors();
    result.stats.initial_cost = 0.5 * optimizer.activeRobustChi2();
    if (!std::isfinite(result.stats.initial_cost)) {
      throw std::runtime_error("nonfinite g2o initial cost");
    }
    if (result.stats.initial_cost == 0.0) {
      result.stats.termination_type = "CONVERGENCE";
      reason = "zero initial cost";
    } else {
      const int optimized = optimizer.optimize(config.max_iterations_per_stage);
      result.stats.iterations = observed->iterations;
      optimizer.computeActiveErrors();
      result.stats.final_cost = 0.5 * optimizer.activeRobustChi2();
      if (optimized < 0 || observed->iterations == 0 || observed->invalid_cost ||
        observed->native_result == g2o::OptimizationAlgorithm::Fail)
      {
        throw std::runtime_error("g2o LM failed or produced a nonfinite/increasing cost");
      }
      result.stats.termination_type = observed->cost_converged ? "CONVERGENCE" : "NO_CONVERGENCE";
      if (observed->cost_converged) {
        reason = "relative cost decrease tolerance (or zero cost)";
      } else if (observed->native_result == g2o::OptimizationAlgorithm::Terminate) {
        reason = "native LM Terminate; stationarity not verified";
      } else {
        reason = "iteration limit";
      }
    }
    optimizer.computeActiveErrors();
    result.stats.final_cost = 0.5 * optimizer.activeRobustChi2();
    auto solved = poses;
    for (std::size_t i = 1; i < poses.size(); ++i) {
      solved[i] = static_cast<const g2o::VertexSE3 *>(
        optimizer.vertex(static_cast<int>(i)))->estimate();
      if (!solved[i].matrix().allFinite()) {
        throw std::runtime_error("nonfinite g2o output pose");
      }
    }
    result.poses = std::move(solved);
    result.success = true;
  } catch (const std::exception & error) {
    result.failure_reason = error.what();
    result.stats.termination_type = "FAILURE";
    reason = result.failure_reason;
  } catch (...) {
    result.failure_reason = "unknown g2o exception";
    result.stats.termination_type = "FAILURE";
    reason = result.failure_reason;
  }
  result.stats.runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  std::ostringstream report;
  report << "g2o LM/Eigen: " << reason << "; iterations=" << result.stats.iterations
         << "; native half robust chi2=" << result.stats.initial_cost << " -> "
         << result.stats.final_cost
         << "; central differences (MQT delta=1e-6); function_tolerance tests relative cost"
         << ", not identical Ceres stopping"
         << "; gradient_tolerance/parameter_tolerance unsupported; gradient/step counts unavailable"
         << "; num_threads/use_nonmonotonic_steps not applied";
  result.stats.brief_report = report.str();
  return result;
}

}  // namespace detail
}  // namespace slam_3d_core
}  // namespace gole
