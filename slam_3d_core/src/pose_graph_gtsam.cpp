// Copyright 2026 Gole Robotics

#include "pose_graph_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <boost/make_shared.hpp>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/linear/VectorValues.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

namespace gole
{
namespace slam_3d_core
{
namespace detail
{
namespace
{

gtsam::Pose3 toGtsamPose(const Eigen::Isometry3d & pose)
{
  return gtsam::Pose3(gtsam::Rot3(pose.linear()), pose.translation());
}

Eigen::Isometry3d fromGtsamPose(const gtsam::Pose3 & pose)
{
  Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
  result.linear() = pose.rotation().matrix();
  result.translation() = pose.translation();
  return result;
}

// The factor returns the already whitened, translation-first quaternion error.
// Unit noise preserves correlated/rank-deficient information without inversion.
class RelativePoseFactor final : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
public:
  RelativePoseFactor(
    const PoseGraphConstraint & constraint, const Matrix6d & sqrt_information,
    const gtsam::SharedNoiseModel & noise)
  : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
      noise, constraint.from_index, constraint.to_index),
    measurement_(constraint.from_T_to), sqrt_information_(sqrt_information)
  {}

  gtsam::Vector evaluateError(
    const gtsam::Pose3 & from, const gtsam::Pose3 & to,
    boost::optional<gtsam::Matrix &> from_jacobian = boost::none,
    boost::optional<gtsam::Matrix &> to_jacobian = boost::none) const override
  {
    const auto residual = [this](const gtsam::Pose3 & a, const gtsam::Pose3 & b) -> Vector6d {
        return sqrt_information_ * relativePoseResidual(
          fromGtsamPose(a), fromGtsamPose(b), measurement_);
      };
    // Correctness-first central differences use the installed Pose3 retract chart,
    // not a BetweenFactor Logmap Jacobian for a different residual convention.
    constexpr double delta = 1.0e-6;
    if (from_jacobian) {
      *from_jacobian = gtsam::numericalDerivative21<Vector6d, gtsam::Pose3, gtsam::Pose3>(
        residual, from, to, delta);
    }
    if (to_jacobian) {
      *to_jacobian = gtsam::numericalDerivative22<Vector6d, gtsam::Pose3, gtsam::Pose3>(
        residual, from, to, delta);
    }
    return residual(from, to);
  }

private:
  Eigen::Isometry3d measurement_;
  Matrix6d sqrt_information_;
};

class GaugeRelativePoseFactor final : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
  GaugeRelativePoseFactor(
    const PoseGraphConstraint & constraint, const Matrix6d & sqrt_information,
    const Eigen::Isometry3d & gauge, const gtsam::SharedNoiseModel & noise)
  : gtsam::NoiseModelFactor1<gtsam::Pose3>(
      noise, constraint.from_index == 0 ? constraint.to_index : constraint.from_index),
    measurement_(constraint.from_T_to), sqrt_information_(sqrt_information),
    gauge_(gauge), gauge_is_from_(constraint.from_index == 0)
  {}

  gtsam::Vector evaluateError(
    const gtsam::Pose3 & pose,
    boost::optional<gtsam::Matrix &> jacobian = boost::none) const override
  {
    const auto residual = [this](const gtsam::Pose3 & variable) -> Vector6d {
        const Eigen::Isometry3d other = fromGtsamPose(variable);
        return sqrt_information_ * (gauge_is_from_ ?
               relativePoseResidual(gauge_, other, measurement_) :
               relativePoseResidual(other, gauge_, measurement_));
      };
    if (jacobian) {
      *jacobian = gtsam::numericalDerivative11<Vector6d, gtsam::Pose3>(residual, pose, 1.0e-6);
    }
    return residual(pose);
  }

private:
  Eigen::Isometry3d measurement_;
  Matrix6d sqrt_information_;
  Eigen::Isometry3d gauge_;
  bool gauge_is_from_;
};

double gradientMaxNorm(const gtsam::NonlinearFactorGraph & graph, const gtsam::Values & values)
{
  // Undamped J^T r includes robust reweighting, but no artificial LM priors.
  // Pose3's local right-retract chart is [rx, ry, rz, tx, ty, tz], not the
  // translation-first residual or Ceres's ambient quaternion parameterization.
  const auto gradient = graph.linearize(values)->gradientAtZero();
  if (gradient.size() != values.size()) {
    throw std::runtime_error("incomplete GTSAM linearization gradient");
  }
  double maximum = 0.0;
  for (const auto & entry : gradient) {
    if (!values.exists(entry.first) || entry.second.size() != 6 || !entry.second.allFinite()) {
      throw std::runtime_error("invalid GTSAM linearization gradient");
    }
    maximum = std::max(maximum, entry.second.cwiseAbs().maxCoeff());
  }
  return maximum;
}

}  // namespace

StageSolveResult solveGtsamStage(
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
  double final_lambda = std::numeric_limits<double>::quiet_NaN();
  std::string reason;
  try {
    if (poses.size() < 2 || constraints.empty() ||
      sqrt_information.size() != constraints.size() || config.max_iterations_per_stage <= 0)
    {
      throw std::invalid_argument("invalid GTSAM stage dimensions or iteration limit");
    }
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    // Pose zero is a constant inside unary factors, not an optimized variable.
    // This fixes any world-frame gauge exactly without a huge/constrained prior.
    for (std::size_t i = 1; i < poses.size(); ++i) {
      values.insert(i, toGtsamPose(poses[i]));
    }
    for (std::size_t i = 0; i < constraints.size(); ++i) {
      const auto & constraint = constraints[i];
      if (constraint.from_index < 0 || constraint.to_index < 0 ||
        static_cast<std::size_t>(constraint.from_index) >= poses.size() ||
        static_cast<std::size_t>(constraint.to_index) >= poses.size() ||
        constraint.from_index == constraint.to_index)
      {
        throw std::invalid_argument("invalid GTSAM constraint endpoints");
      }
      gtsam::SharedNoiseModel noise = gtsam::noiseModel::Unit::Create(6);
      if (config.robust_loop_enabled && constraint.loop_closure) {
        if (!std::isfinite(cauchy_scale) || cauchy_scale <= 0.0) {
          throw std::invalid_argument("invalid GTSAM Cauchy scale");
        }
        noise = gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(
            cauchy_scale, gtsam::noiseModel::mEstimator::Base::Block), noise);
      }
      if (constraint.from_index == 0 || constraint.to_index == 0) {
        graph.push_back(
          boost::make_shared<GaugeRelativePoseFactor>(
            constraint, sqrt_information[i], poses.front(), noise));
      } else {
        graph.push_back(
          boost::make_shared<RelativePoseFactor>(
            constraint, sqrt_information[i], noise));
      }
    }
    gtsam::LevenbergMarquardtParams params;
    params.maxIterations = static_cast<std::size_t>(config.max_iterations_per_stage);
    params.relativeErrorTol = config.function_tolerance;
    params.absoluteErrorTol = 0.0;
    params.errorTol = 0.0;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    result.stats.initial_cost = optimizer.error();
    result.stats.final_cost = optimizer.error();
    final_lambda = optimizer.lambda();
    if (!std::isfinite(result.stats.initial_cost)) {
      throw std::runtime_error("nonfinite GTSAM initial cost");
    }
    result.stats.termination_type = "NO_CONVERGENCE";
    reason = "iteration limit";
    const auto update_gradient = [&]() {
        result.stats.gradient_available = false;
        result.stats.final_gradient_max_norm = std::numeric_limits<double>::infinity();
        result.stats.final_gradient_max_norm = gradientMaxNorm(graph, optimizer.values());
        result.stats.gradient_available = true;
      };
    update_gradient();
    if (result.stats.final_gradient_max_norm <= config.gradient_tolerance) {
      result.stats.termination_type = "CONVERGENCE";
      reason = "initial GTSAM local gradient tolerance";
    } else {
      for (int iteration = 0; iteration < config.max_iterations_per_stage; ++iteration) {
        const double before = optimizer.error();
        const std::size_t previous_iterations = optimizer.iterations();
        result.stats.gradient_available = false;
        result.stats.final_gradient_max_norm = std::numeric_limits<double>::infinity();
        optimizer.iterate();
        result.stats.iterations = static_cast<int>(optimizer.iterations());
        result.stats.final_cost = optimizer.error();
        final_lambda = optimizer.lambda();
        if (!std::isfinite(result.stats.final_cost) || result.stats.final_cost > before) {
          throw std::runtime_error("GTSAM LM produced a nonfinite/increasing cost");
        }
        update_gradient();
        if (!std::isfinite(final_lambda)) {
          throw std::runtime_error("nonfinite GTSAM LM damping");
        }
        // At floating-point stationarity LM can exhaust its damping range while
        // trying to decrease a near-zero cost. Only the undamped gradient may
        // justify accepting that state; unchanged cost alone is not evidence.
        if (result.stats.final_gradient_max_norm <= config.gradient_tolerance) {
          result.stats.termination_type = "CONVERGENCE";
          reason = final_lambda >= params.lambdaUpperBound ?
            "GTSAM local gradient tolerance after LM damping exhaustion" :
            "GTSAM local gradient tolerance";
          break;
        }
        if (final_lambda >= params.lambdaUpperBound) {
          throw std::runtime_error("GTSAM LM exhausted its damping range");
        }
        if (optimizer.iterations() == previous_iterations) {
          reason = "native LM stopped without accepted update; gradient tolerance not met";
          break;
        }
        if (result.stats.final_cost == 0.0 ||
          (config.function_tolerance > 0.0 &&
          gtsam::checkConvergence(params, before, result.stats.final_cost)))
        {
          result.stats.termination_type = "CONVERGENCE";
          reason = "GTSAM relative error tolerance (or zero cost)";
          break;
        }
      }
    }
    auto solved = poses;
    for (std::size_t i = 1; i < poses.size(); ++i) {
      solved[i] = fromGtsamPose(optimizer.values().at<gtsam::Pose3>(i));
      if (!solved[i].matrix().allFinite()) {
        throw std::runtime_error("nonfinite GTSAM output pose");
      }
    }
    result.poses = std::move(solved);
    result.success = true;
  } catch (const std::exception & error) {
    result.failure_reason = error.what();
    result.stats.termination_type = "FAILURE";
    reason = result.failure_reason;
  } catch (...) {
    result.failure_reason = "unknown GTSAM exception";
    result.stats.termination_type = "FAILURE";
    reason = result.failure_reason;
  }
  result.stats.runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
  std::ostringstream report;
  report << "GTSAM LM: " << reason << "; native iterations=" << result.stats.iterations
         << "; native graph.error=" << result.stats.initial_cost << " -> "
         << result.stats.final_cost
         << "; native lambda=" << final_lambda
         << "; central differences (Pose3 retract delta=1e-6)"
         << "; function_tolerance mapped to GTSAM relativeErrorTol, not identical Ceres stopping"
         << "; gradient_tolerance=" << config.gradient_tolerance
         << " on undamped local right-retract [rx,ry,rz,tx,ty,tz] gradient infinity norm="
         << result.stats.final_gradient_max_norm
         << "; parameter_tolerance unsupported; step counts unavailable"
         << "; num_threads/use_nonmonotonic_steps not applied";
  result.stats.brief_report = report.str();
  return result;
}

}  // namespace detail
}  // namespace slam_3d_core
}  // namespace gole
