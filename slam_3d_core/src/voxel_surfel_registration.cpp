// Copyright 2026 Gole Robotics

#include "slam_3d_core/voxel_surfel_registration.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gole
{
namespace slam_3d_core
{
namespace
{

using Clock = std::chrono::steady_clock;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using RowVector6d = VoxelSurfelRegistration::RowVector6d;
using Vector6d = VoxelSurfelRegistration::Vector6d;

struct VoxelKey
{
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const
  {
    const auto mix = [](std::size_t seed, std::size_t value) {
        value += 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
        return seed ^ value;
      };

    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed = mix(seed, std::hash<std::int64_t>{}(key.y));
    return mix(seed, std::hash<std::int64_t>{}(key.z));
  }
};

struct VoxelAccumulator
{
  std::size_t count{0};
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d sum_outer{Eigen::Matrix3d::Zero()};

  void add(const Eigen::Vector3d & point)
  {
    ++count;
    sum += point;
    sum_outer.noalias() += point * point.transpose();
  }
};

struct Surfel
{
  Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
};

using AccumulatorMap =
  std::unordered_map<VoxelKey, VoxelAccumulator, VoxelKeyHash>;
using SurfelMap = std::unordered_map<VoxelKey, Surfel, VoxelKeyHash>;

struct TargetModel
{
  SurfelMap surfels;
  std::size_t voxel_count{0};
};

struct Correspondence
{
  Eigen::Vector3d source_point{Eigen::Vector3d::Zero()};
  const Surfel * surfel{nullptr};
  double residual{0.0};
  RowVector6d jacobian{RowVector6d::Zero()};
};

struct CorrespondenceSet
{
  std::vector<Correspondence> values;
  std::size_t valid_source_points{0};
};

struct ResidualStatistics
{
  double rmse{0.0};
  double median_abs{0.0};
  double p95_abs{0.0};
  double robust_cost{0.0};
  double downweight_ratio{0.0};
};

struct LinearSystem
{
  Matrix6d hessian{Matrix6d::Zero()};
  Vector6d gradient{Vector6d::Zero()};
  std::array<double, 6> eigenvalues{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  int observable_rank{0};
  double condition{0.0};
  Vector6d weak_state{Vector6d::Zero()};
  Vector6d normalized_step{Vector6d::Zero()};
  bool valid{false};
};

double elapsedMilliseconds(const Clock::time_point & start, const Clock::time_point & end)
{
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool isFinite(const Eigen::Vector3d & value)
{
  return value.array().isFinite().all();
}

bool isFinite(const Eigen::Isometry3d & transform)
{
  return transform.matrix().array().isFinite().all();
}

bool isFinitePoint(const pcl::PointXYZ & point)
{
  return
    std::isfinite(static_cast<double>(point.x)) &&
    std::isfinite(static_cast<double>(point.y)) &&
    std::isfinite(static_cast<double>(point.z));
}

Eigen::Matrix3d skew(const Eigen::Vector3d & value)
{
  Eigen::Matrix3d result;
  result <<
    0.0, -value.z(), value.y(),
    value.z(), 0.0, -value.x(),
    -value.y(), value.x(), 0.0;
  return result;
}

Eigen::Matrix3d rotationExp(const Eigen::Vector3d & rotation_vector)
{
  const double angle = rotation_vector.norm();
  if (angle < 1.0e-12) {
    return Eigen::Matrix3d::Identity();
  }
  return Eigen::AngleAxisd(angle, rotation_vector / angle).toRotationMatrix();
}

std::optional<VoxelKey> voxelKey(const Eigen::Vector3d & point, double voxel_size)
{
  if (!isFinite(point) || !std::isfinite(voxel_size) || voxel_size <= 0.0) {
    return std::nullopt;
  }

  const Eigen::Array3d scaled = point.array() / voxel_size;
  constexpr double kMinIndex =
    static_cast<double>(std::numeric_limits<std::int64_t>::min()) * 0.5;
  constexpr double kMaxIndex =
    static_cast<double>(std::numeric_limits<std::int64_t>::max()) * 0.5;
  if ((scaled < kMinIndex).any() || (scaled > kMaxIndex).any()) {
    return std::nullopt;
  }

  return VoxelKey{
    static_cast<std::int64_t>(std::floor(scaled.x())),
    static_cast<std::int64_t>(std::floor(scaled.y())),
    static_cast<std::int64_t>(std::floor(scaled.z()))};
}

bool validConfig(const VoxelSurfelRegistration::Config & config)
{
  const auto positive_finite = [](double value) {
      return std::isfinite(value) && value > 0.0;
    };
  const auto nonnegative_finite = [](double value) {
      return std::isfinite(value) && value >= 0.0;
    };

  return
    positive_finite(config.voxel_size) &&
    config.min_points_per_voxel >= 3U &&
    nonnegative_finite(config.max_planarity_ratio) &&
    config.max_planarity_ratio < 1.0 &&
    nonnegative_finite(config.min_tangent_variance) &&
    positive_finite(config.max_plane_distance) &&
    positive_finite(config.max_center_distance) &&
    config.min_correspondences > 0U &&
    positive_finite(config.min_correspondence_ratio) &&
    config.min_correspondence_ratio <= 1.0 &&
    positive_finite(config.huber_delta) &&
    config.max_iterations > 0 &&
    config.max_backtracking_steps >= 0 &&
    positive_finite(config.backtracking_factor) &&
    config.backtracking_factor < 1.0 &&
    nonnegative_finite(config.rotation_convergence_rad) &&
    nonnegative_finite(config.translation_convergence_m) &&
    nonnegative_finite(config.relative_cost_convergence) &&
    positive_finite(config.max_rotation_step_rad) &&
    positive_finite(config.max_translation_step_m) &&
    positive_finite(config.rotation_scale) &&
    positive_finite(config.relative_eigen_threshold) &&
    config.relative_eigen_threshold < 1.0 &&
    config.min_observable_rank >= 1 &&
    config.min_observable_rank <= 6 &&
    positive_finite(config.max_quality_rmse) &&
    positive_finite(config.max_quality_p95);
}

TargetModel buildTargetModel(
  const VoxelSurfelRegistration::CloudT & target,
  const VoxelSurfelRegistration::Config & config)
{
  AccumulatorMap accumulators;
  accumulators.reserve(target.size());

  for (const auto & point : target.points) {
    if (!isFinitePoint(point)) {
      continue;
    }
    const Eigen::Vector3d value(point.x, point.y, point.z);
    const auto key = voxelKey(value, config.voxel_size);
    if (!key.has_value()) {
      continue;
    }
    accumulators[*key].add(value);
  }

  TargetModel model;
  model.voxel_count = accumulators.size();
  model.surfels.reserve(accumulators.size());
  for (const auto & entry : accumulators) {
    const auto & accumulator = entry.second;
    if (accumulator.count < config.min_points_per_voxel) {
      continue;
    }

    const double inverse_count = 1.0 / static_cast<double>(accumulator.count);
    const Eigen::Vector3d centroid = accumulator.sum * inverse_count;
    Eigen::Matrix3d covariance =
      accumulator.sum_outer * inverse_count - centroid * centroid.transpose();
    covariance = 0.5 * (covariance + covariance.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
    if (solver.info() != Eigen::Success) {
      continue;
    }

    Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseMax(0.0);
    const double middle_eigenvalue = eigenvalues.y();
    const double planarity_ratio =
      eigenvalues.x() / std::max(middle_eigenvalue, 1.0e-15);
    if (
      middle_eigenvalue < config.min_tangent_variance ||
      planarity_ratio > config.max_planarity_ratio)
    {
      continue;
    }

    Eigen::Vector3d normal = solver.eigenvectors().col(0);
    if (!isFinite(normal) || normal.squaredNorm() < 0.5) {
      continue;
    }
    normal.normalize();
    model.surfels.emplace(entry.first, Surfel{centroid, normal});
  }

  return model;
}

CorrespondenceSet findCorrespondences(
  const VoxelSurfelRegistration::CloudT & source,
  const SurfelMap & target_surfels,
  const Eigen::Isometry3d & target_T_source,
  const VoxelSurfelRegistration::Config & config)
{
  CorrespondenceSet result;
  result.values.reserve(source.size());

  for (const auto & point : source.points) {
    if (!isFinitePoint(point)) {
      continue;
    }
    ++result.valid_source_points;

    const Eigen::Vector3d source_point(point.x, point.y, point.z);
    const Eigen::Vector3d transformed = target_T_source * source_point;
    const auto center_key = voxelKey(transformed, config.voxel_size);
    if (!center_key.has_value()) {
      continue;
    }

    const Surfel * best_surfel = nullptr;
    double best_abs_plane_distance = std::numeric_limits<double>::infinity();
    double best_center_distance = std::numeric_limits<double>::infinity();
    double best_residual = 0.0;

    for (std::int64_t dx = -1; dx <= 1; ++dx) {
      for (std::int64_t dy = -1; dy <= 1; ++dy) {
        for (std::int64_t dz = -1; dz <= 1; ++dz) {
          const VoxelKey candidate_key{
            center_key->x + dx,
            center_key->y + dy,
            center_key->z + dz};
          const auto iterator = target_surfels.find(candidate_key);
          if (iterator == target_surfels.end()) {
            continue;
          }

          const Surfel & surfel = iterator->second;
          const Eigen::Vector3d offset = transformed - surfel.centroid;
          const double center_distance = offset.norm();
          if (center_distance > config.max_center_distance) {
            continue;
          }
          const double residual = surfel.normal.dot(offset);
          const double abs_plane_distance = std::abs(residual);
          if (abs_plane_distance > config.max_plane_distance) {
            continue;
          }

          if (
            abs_plane_distance < best_abs_plane_distance ||
            (abs_plane_distance == best_abs_plane_distance &&
            center_distance < best_center_distance))
          {
            best_surfel = &surfel;
            best_abs_plane_distance = abs_plane_distance;
            best_center_distance = center_distance;
            best_residual = residual;
          }
        }
      }
    }

    if (best_surfel == nullptr) {
      continue;
    }

    result.values.push_back(
      Correspondence{
            source_point,
            best_surfel,
            best_residual,
            VoxelSurfelRegistration::pointToPlaneJacobian(
              source_point, target_T_source, best_surfel->normal)});
  }

  return result;
}

double huberLoss(double residual, double delta)
{
  const double absolute_residual = std::abs(residual);
  if (absolute_residual <= delta) {
    return 0.5 * residual * residual;
  }
  return delta * (absolute_residual - 0.5 * delta);
}

double huberWeight(double residual, double delta)
{
  const double absolute_residual = std::abs(residual);
  if (absolute_residual <= delta || absolute_residual < 1.0e-15) {
    return 1.0;
  }
  return delta / absolute_residual;
}

double percentileSorted(const std::vector<double> & sorted_values, double percentile)
{
  if (sorted_values.empty()) {
    return 0.0;
  }
  const double index = percentile * static_cast<double>(sorted_values.size() - 1U);
  const std::size_t lower = static_cast<std::size_t>(std::floor(index));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
  if (lower == upper) {
    return sorted_values[lower];
  }
  const double fraction = index - static_cast<double>(lower);
  return
    (1.0 - fraction) * sorted_values[lower] +
    fraction * sorted_values[upper];
}

ResidualStatistics residualStatistics(
  const std::vector<Correspondence> & correspondences,
  double huber_delta)
{
  ResidualStatistics result;
  if (correspondences.empty()) {
    return result;
  }

  std::vector<double> absolute_residuals;
  absolute_residuals.reserve(correspondences.size());
  double squared_sum = 0.0;
  double robust_cost_sum = 0.0;
  std::size_t downweighted = 0;
  for (const auto & correspondence : correspondences) {
    const double residual = correspondence.residual;
    const double absolute_residual = std::abs(residual);
    absolute_residuals.push_back(absolute_residual);
    squared_sum += residual * residual;
    robust_cost_sum += huberLoss(residual, huber_delta);
    if (absolute_residual > huber_delta) {
      ++downweighted;
    }
  }
  std::sort(absolute_residuals.begin(), absolute_residuals.end());

  const double inverse_count = 1.0 / static_cast<double>(correspondences.size());
  result.rmse = std::sqrt(squared_sum * inverse_count);
  result.median_abs = percentileSorted(absolute_residuals, 0.5);
  result.p95_abs = percentileSorted(absolute_residuals, 0.95);
  result.robust_cost = robust_cost_sum * inverse_count;
  result.downweight_ratio = static_cast<double>(downweighted) * inverse_count;
  return result;
}

LinearSystem buildLinearSystem(
  const std::vector<Correspondence> & correspondences,
  const VoxelSurfelRegistration::Config & config)
{
  LinearSystem system;
  for (const auto & correspondence : correspondences) {
    RowVector6d normalized_jacobian = correspondence.jacobian;
    normalized_jacobian.template head<3>() /= config.rotation_scale;
    const double weight = huberWeight(correspondence.residual, config.huber_delta);
    system.hessian.noalias() +=
      weight * normalized_jacobian.transpose() * normalized_jacobian;
    system.gradient.noalias() +=
      weight * normalized_jacobian.transpose() * correspondence.residual;
  }
  system.hessian = 0.5 * (system.hessian + system.hessian.transpose());

  const Eigen::SelfAdjointEigenSolver<Matrix6d> eigen_solver(system.hessian);
  if (eigen_solver.info() != Eigen::Success) {
    return system;
  }

  const Eigen::Matrix<double, 6, 1> eigenvalues =
    eigen_solver.eigenvalues().cwiseMax(0.0);
  const Matrix6d eigenvectors = eigen_solver.eigenvectors();
  for (int index = 0; index < 6; ++index) {
    system.eigenvalues[static_cast<std::size_t>(index)] = eigenvalues[index];
  }

  const double maximum_eigenvalue = eigenvalues.maxCoeff();
  if (!std::isfinite(maximum_eigenvalue) || maximum_eigenvalue <= 0.0) {
    return system;
  }
  const double observable_threshold =
    std::max(
    maximum_eigenvalue * config.relative_eigen_threshold,
    maximum_eigenvalue * 1.0e-12);

  double minimum_observable_eigenvalue = maximum_eigenvalue;
  for (int index = 0; index < 6; ++index) {
    const Vector6d direction = eigenvectors.col(index);
    if (eigenvalues[index] > observable_threshold) {
      ++system.observable_rank;
      minimum_observable_eigenvalue =
        std::min(minimum_observable_eigenvalue, eigenvalues[index]);
      system.normalized_step.noalias() -=
        direction * direction.dot(system.gradient) / eigenvalues[index];
    } else {
      system.weak_state.array() += direction.array().square();
    }
  }

  if (system.observable_rank > 0) {
    system.condition = maximum_eigenvalue / minimum_observable_eigenvalue;
  }
  system.valid =
    system.normalized_step.array().isFinite().all() &&
    system.weak_state.array().isFinite().all();
  return system;
}

Vector6d physicalStep(const LinearSystem & system, double rotation_scale)
{
  Vector6d result = system.normalized_step;
  result.template head<3>() /= rotation_scale;
  return result;
}

Vector6d clampStep(
  const Vector6d & step,
  const VoxelSurfelRegistration::Config & config)
{
  Vector6d result = step;
  const double rotation_norm = result.template head<3>().norm();
  if (rotation_norm > config.max_rotation_step_rad) {
    result.template head<3>() *= config.max_rotation_step_rad / rotation_norm;
  }
  const double translation_norm = result.template tail<3>().norm();
  if (translation_norm > config.max_translation_step_m) {
    result.template tail<3>() *= config.max_translation_step_m / translation_norm;
  }
  return result;
}

double fixedCorrespondenceCost(
  const std::vector<Correspondence> & correspondences,
  const Eigen::Isometry3d & target_T_source,
  double huber_delta)
{
  if (correspondences.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double cost = 0.0;
  for (const auto & correspondence : correspondences) {
    const double residual = VoxelSurfelRegistration::pointToPlaneResidual(
      correspondence.source_point,
      target_T_source,
      correspondence.surfel->centroid,
      correspondence.surfel->normal);
    cost += huberLoss(residual, huber_delta);
  }
  return cost / static_cast<double>(correspondences.size());
}

void updateResultDiagnostics(
  VoxelSurfelRegistration::Result & result,
  const CorrespondenceSet & correspondence_set,
  const VoxelSurfelRegistration::Config & config,
  const LinearSystem * linear_system)
{
  result.valid_source_points = correspondence_set.valid_source_points;
  result.correspondences = correspondence_set.values.size();
  if (result.valid_source_points > 0U) {
    result.correspondence_ratio =
      static_cast<double>(result.correspondences) /
      static_cast<double>(result.valid_source_points);
  }

  const ResidualStatistics statistics =
    residualStatistics(correspondence_set.values, config.huber_delta);
  result.rmse = statistics.rmse;
  result.median_abs_residual = statistics.median_abs;
  result.p95_abs_residual = statistics.p95_abs;
  result.robust_cost = statistics.robust_cost;
  result.robust_downweight_ratio = statistics.downweight_ratio;

  if (linear_system == nullptr) {
    return;
  }
  result.hessian_eigenvalues = linear_system->eigenvalues;
  result.observable_rank = linear_system->observable_rank;
  result.hessian_condition = linear_system->condition;
  result.weak_state_vector = linear_system->weak_state;
  result.weak_rotation_axes = linear_system->weak_state.template head<3>();
  result.weak_translation_axes = linear_system->weak_state.template tail<3>();
}

}  // namespace

VoxelSurfelRegistration::VoxelSurfelRegistration()
: config_(Config{})
{
}

VoxelSurfelRegistration::VoxelSurfelRegistration(const Config & config)
: config_(config)
{
}

const VoxelSurfelRegistration::Config & VoxelSurfelRegistration::config() const
{
  return config_;
}

Eigen::Isometry3d VoxelSurfelRegistration::applyIncrement(
  const Eigen::Isometry3d & target_T_source,
  const Vector6d & increment)
{
  Eigen::Isometry3d result = target_T_source;
  result.linear() =
    rotationExp(increment.template head<3>()) * target_T_source.linear();
  result.translation() =
    target_T_source.translation() + increment.template tail<3>();
  return result;
}

double VoxelSurfelRegistration::pointToPlaneResidual(
  const Eigen::Vector3d & source_point,
  const Eigen::Isometry3d & target_T_source,
  const Eigen::Vector3d & target_centroid,
  const Eigen::Vector3d & target_normal)
{
  return target_normal.dot(target_T_source * source_point - target_centroid);
}

VoxelSurfelRegistration::RowVector6d VoxelSurfelRegistration::pointToPlaneJacobian(
  const Eigen::Vector3d & source_point,
  const Eigen::Isometry3d & target_T_source,
  const Eigen::Vector3d & target_normal)
{
  const Eigen::Vector3d rotated_source = target_T_source.linear() * source_point;
  RowVector6d jacobian;
  jacobian.template head<3>() =
    target_normal.transpose() * (-skew(rotated_source));
  jacobian.template tail<3>() = target_normal.transpose();
  return jacobian;
}

VoxelSurfelRegistration::Result VoxelSurfelRegistration::align(
  const CloudT & source,
  const CloudT & target,
  const Eigen::Isometry3d & initial_guess) const
{
  const Clock::time_point total_start = Clock::now();
  Result result;
  result.target_T_source = initial_guess;

  if (!validConfig(config_)) {
    result.reason = "invalid_config";
    result.total_time_ms = elapsedMilliseconds(total_start, Clock::now());
    return result;
  }
  if (!isFinite(initial_guess)) {
    result.reason = "invalid_initial_guess";
    result.total_time_ms = elapsedMilliseconds(total_start, Clock::now());
    return result;
  }

  const Clock::time_point target_build_start = Clock::now();
  const TargetModel target_model = buildTargetModel(target, config_);
  const Clock::time_point target_build_end = Clock::now();
  result.target_voxels = target_model.voxel_count;
  result.planar_surfels = target_model.surfels.size();
  result.target_build_time_ms =
    elapsedMilliseconds(target_build_start, target_build_end);

  if (source.empty()) {
    result.reason = "empty_source";
    result.total_time_ms = elapsedMilliseconds(total_start, Clock::now());
    return result;
  }
  if (target_model.voxel_count == 0U) {
    result.reason = "empty_target";
    result.total_time_ms = elapsedMilliseconds(total_start, Clock::now());
    return result;
  }
  if (target_model.surfels.empty()) {
    result.reason = "no_planar_surfels";
    result.total_time_ms = elapsedMilliseconds(total_start, Clock::now());
    return result;
  }

  const Clock::time_point optimization_start = Clock::now();
  std::string terminal_reason{"accepted_max_iterations"};
  bool fatal_optimizer_failure = false;

  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const CorrespondenceSet correspondence_set =
      findCorrespondences(source, target_model.surfels, result.target_T_source, config_);
    result.valid_source_points = correspondence_set.valid_source_points;
    result.correspondences = correspondence_set.values.size();
    result.correspondence_ratio =
      correspondence_set.valid_source_points > 0U ?
      static_cast<double>(correspondence_set.values.size()) /
      static_cast<double>(correspondence_set.valid_source_points) :
      0.0;

    if (correspondence_set.valid_source_points == 0U) {
      terminal_reason = "empty_source";
      fatal_optimizer_failure = true;
      break;
    }
    if (correspondence_set.values.size() < config_.min_correspondences) {
      terminal_reason = "insufficient_correspondences";
      fatal_optimizer_failure = true;
      break;
    }
    if (result.correspondence_ratio < config_.min_correspondence_ratio) {
      terminal_reason = "insufficient_correspondence_ratio";
      fatal_optimizer_failure = true;
      break;
    }

    const LinearSystem system = buildLinearSystem(correspondence_set.values, config_);
    if (!system.valid) {
      terminal_reason = "numerical_failure";
      fatal_optimizer_failure = true;
      break;
    }
    if (system.observable_rank < config_.min_observable_rank) {
      terminal_reason = "insufficient_observable_rank";
      fatal_optimizer_failure = true;
      break;
    }

    Vector6d step = clampStep(physicalStep(system, config_.rotation_scale), config_);
    ++result.iterations;
    if (
      step.template head<3>().norm() <= config_.rotation_convergence_rad &&
      step.template tail<3>().norm() <= config_.translation_convergence_m)
    {
      result.converged = true;
      terminal_reason = "converged_step";
      break;
    }

    const double previous_cost =
      fixedCorrespondenceCost(
      correspondence_set.values, result.target_T_source, config_.huber_delta);
    bool accepted_step = false;
    double accepted_cost = previous_cost;
    double scale = 1.0;
    for (int backtracking = 0;
      backtracking <= config_.max_backtracking_steps;
      ++backtracking)
    {
      const Eigen::Isometry3d candidate =
        applyIncrement(result.target_T_source, scale * step);
      const double candidate_cost =
        fixedCorrespondenceCost(
        correspondence_set.values, candidate, config_.huber_delta);
      if (std::isfinite(candidate_cost) && candidate_cost < previous_cost) {
        result.target_T_source = candidate;
        accepted_cost = candidate_cost;
        accepted_step = true;
        break;
      }
      scale *= config_.backtracking_factor;
    }

    if (!accepted_step) {
      terminal_reason = "line_search_failed";
      fatal_optimizer_failure = true;
      break;
    }

    const double relative_cost_change =
      (previous_cost - accepted_cost) / std::max(previous_cost, 1.0e-15);
    if (relative_cost_change <= config_.relative_cost_convergence) {
      result.converged = true;
      terminal_reason = "converged_cost";
      break;
    }
  }

  const CorrespondenceSet final_correspondences =
    findCorrespondences(source, target_model.surfels, result.target_T_source, config_);
  std::optional<LinearSystem> final_system;
  if (!final_correspondences.values.empty()) {
    final_system = buildLinearSystem(final_correspondences.values, config_);
  }
  updateResultDiagnostics(
    result,
    final_correspondences,
    config_,
    final_system.has_value() ? &(*final_system) : nullptr);

  if (!fatal_optimizer_failure) {
    if (result.valid_source_points == 0U) {
      terminal_reason = "empty_source";
    } else if (result.correspondences < config_.min_correspondences) {
      terminal_reason = "insufficient_correspondences";
    } else if (result.correspondence_ratio < config_.min_correspondence_ratio) {
      terminal_reason = "insufficient_correspondence_ratio";
    } else if (
      !final_system.has_value() || !final_system->valid ||
      result.observable_rank < config_.min_observable_rank)
    {
      terminal_reason = "insufficient_observable_rank";
    } else if (!std::isfinite(result.rmse) || result.rmse > config_.max_quality_rmse) {
      terminal_reason = "quality_rmse";
    } else if (
      !std::isfinite(result.p95_abs_residual) ||
      result.p95_abs_residual > config_.max_quality_p95)
    {
      terminal_reason = "quality_p95";
    } else {
      result.accepted = true;
    }
  }

  result.reason = terminal_reason;
  const Clock::time_point optimization_end = Clock::now();
  result.optimization_time_ms =
    elapsedMilliseconds(optimization_start, optimization_end);
  result.total_time_ms = elapsedMilliseconds(total_start, optimization_end);
  return result;
}

}  // namespace slam_3d_core
}  // namespace gole
