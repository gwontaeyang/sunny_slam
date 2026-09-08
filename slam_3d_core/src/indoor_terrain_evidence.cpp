// Copyright 2026 Gole Robotics

#include "slam_3d_core/indoor_terrain_evidence.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace gole
{
namespace slam_3d_core
{

namespace
{

bool isSupportLabel(const TerrainLabel label)
{
  return label == TerrainLabel::kFloor || label == TerrainLabel::kRamp;
}

bool isHazardLabel(const TerrainLabel label)
{
  return label == TerrainLabel::kPositiveStep ||
         label == TerrainLabel::kObservedDrop ||
         label == TerrainLabel::kStaircase;
}

struct PolarBin
{
  std::vector<std::size_t> point_indices;
  double floor_z_m{0.0};
  bool valid{false};
};

struct ProfileNode
{
  int radial_index{0};
  const PolarBin * bin{nullptr};
  TerrainLabel label{TerrainLabel::kUnknown};
  double roughness_score{0.0};
};

struct StairTransition
{
  std::size_t profile_index{0U};
  int direction{0};
  double riser_height_m{0.0};
};

struct PlaneGridKey
{
  std::int64_t x{0};
  std::int64_t y{0};

  bool operator==(const PlaneGridKey & other) const
  {
    return x == other.x && y == other.y;
  }
};

struct PlaneGridKeyHash
{
  std::size_t operator()(const PlaneGridKey & key) const
  {
    const auto x_hash = std::hash<std::int64_t>{}(key.x);
    const auto y_hash = std::hash<std::int64_t>{}(key.y);
    return x_hash ^ (y_hash + 0x9e3779b9U + (x_hash << 6U) + (x_hash >> 2U));
  }
};

struct PlaneEstimate
{
  bool valid{false};
  Eigen::Vector3d centroid{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d axis_a{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d axis_b{Eigen::Vector3d::UnitY()};
};

struct RampPlaneFit
{
  enum class RejectionReason
  {
    kNone,
    kInsufficientPoints,
    kPlaneEstimation,
    kInlierSupport,
    kLength,
    kWidth,
    kSlope,
    kRmse,
  };

  bool accepted{false};
  RejectionReason rejection_reason{RejectionReason::kInsufficientPoints};
  double slope_deg{0.0};
  double rmse_m{0.0};
  double length_m{0.0};
  double width_m{0.0};
  double inlier_ratio{0.0};
};

PlaneGridKey planeGridKey(const Eigen::Vector3d & point, const double cell_size_m)
{
  return PlaneGridKey{
    static_cast<std::int64_t>(std::floor(point.x() / cell_size_m)),
    static_cast<std::int64_t>(std::floor(point.y() / cell_size_m))};
}

PlaneEstimate estimatePlane(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<std::size_t> & indices)
{
  PlaneEstimate estimate;
  if (indices.size() < 3U) {
    return estimate;
  }

  for (const std::size_t index : indices) {
    if (index >= points.size() || !points[index].allFinite()) {
      return estimate;
    }
    estimate.centroid += points[index];
  }
  estimate.centroid /= static_cast<double>(indices.size());

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const std::size_t index : indices) {
    const Eigen::Vector3d centered = points[index] - estimate.centroid;
    covariance.noalias() += centered * centered.transpose();
  }
  covariance /= static_cast<double>(indices.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite() ||
    !solver.eigenvectors().allFinite())
  {
    return estimate;
  }
  estimate.normal = solver.eigenvectors().col(0).normalized();
  if (estimate.normal.z() < 0.0) {
    estimate.normal = -estimate.normal;
  }
  estimate.axis_a = solver.eigenvectors().col(2).normalized();
  estimate.axis_b = solver.eigenvectors().col(1).normalized();
  estimate.valid = estimate.normal.allFinite() && estimate.axis_a.allFinite() &&
    estimate.axis_b.allFinite();
  return estimate;
}

RampPlaneFit fitRampPlane(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<std::size_t> & neighborhood,
  const IndoorTerrainConfig & config)
{
  RampPlaneFit fit;
  const std::size_t minimum_points = static_cast<std::size_t>(
    std::max(3, config.ramp_plane_min_points));
  if (neighborhood.size() < minimum_points) {
    return fit;
  }

  const PlaneEstimate initial = estimatePlane(points, neighborhood);
  if (!initial.valid) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kPlaneEstimation;
    return fit;
  }
  const double maximum_outlier_distance_m = std::max(
    1.0e-4, config.ramp_plane_max_outlier_distance_m);
  std::vector<std::size_t> inliers;
  inliers.reserve(neighborhood.size());
  for (const std::size_t index : neighborhood) {
    const double residual_m = std::abs(
      initial.normal.dot(points[index] - initial.centroid));
    if (std::isfinite(residual_m) && residual_m <= maximum_outlier_distance_m) {
      inliers.push_back(index);
    }
  }
  fit.inlier_ratio = static_cast<double>(inliers.size()) /
    static_cast<double>(neighborhood.size());
  if (inliers.size() < minimum_points ||
    fit.inlier_ratio < std::clamp(config.ramp_plane_min_inlier_ratio, 0.0, 1.0))
  {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kInlierSupport;
    return fit;
  }

  const PlaneEstimate refined = estimatePlane(points, inliers);
  if (!refined.valid) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kPlaneEstimation;
    return fit;
  }

  double squared_residual_sum = 0.0;
  double axis_a_min = std::numeric_limits<double>::infinity();
  double axis_a_max = -std::numeric_limits<double>::infinity();
  double axis_b_min = std::numeric_limits<double>::infinity();
  double axis_b_max = -std::numeric_limits<double>::infinity();
  for (const std::size_t index : inliers) {
    const Eigen::Vector3d centered = points[index] - refined.centroid;
    const double residual_m = refined.normal.dot(centered);
    squared_residual_sum += residual_m * residual_m;
    const double axis_a = refined.axis_a.dot(centered);
    const double axis_b = refined.axis_b.dot(centered);
    axis_a_min = std::min(axis_a_min, axis_a);
    axis_a_max = std::max(axis_a_max, axis_a);
    axis_b_min = std::min(axis_b_min, axis_b);
    axis_b_max = std::max(axis_b_max, axis_b);
  }

  fit.rmse_m = std::sqrt(
    squared_residual_sum / static_cast<double>(inliers.size()));
  const double extent_a_m = axis_a_max - axis_a_min;
  const double extent_b_m = axis_b_max - axis_b_min;
  fit.length_m = std::max(extent_a_m, extent_b_m);
  fit.width_m = std::min(extent_a_m, extent_b_m);
  fit.slope_deg = std::acos(std::clamp(refined.normal.z(), -1.0, 1.0)) *
    180.0 / M_PI;

  const double minimum_slope_deg = std::max(
    std::max(0.0, config.floor_max_slope_deg),
    std::max(0.0, config.ramp_min_slope_deg));
  const double maximum_slope_deg = std::max(
    minimum_slope_deg, config.ramp_max_slope_deg);
  if (fit.length_m < std::max(0.0, config.ramp_plane_min_length_m)) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kLength;
  } else if (fit.width_m < std::max(0.0, config.ramp_plane_min_width_m)) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kWidth;
  } else if (fit.slope_deg < minimum_slope_deg || fit.slope_deg > maximum_slope_deg) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kSlope;
  } else if (fit.rmse_m > std::max(0.0, config.ramp_plane_max_rmse_m)) {
    fit.rejection_reason = RampPlaneFit::RejectionReason::kRmse;
  } else {
    fit.accepted = true;
    fit.rejection_reason = RampPlaneFit::RejectionReason::kNone;
  }
  return fit;
}

void validateRampPlanes(
  const std::vector<Eigen::Vector3d> & points,
  const IndoorTerrainConfig & config,
  TerrainClassificationResult & result)
{
  if (!config.ramp_plane_validation_enabled) {
    return;
  }

  std::vector<std::size_t> ramp_candidates;
  ramp_candidates.reserve(result.labels.size());
  const double radius_m = std::max(1.0e-3, config.ramp_plane_neighborhood_radius_m);
  const double radius_squared_m = radius_m * radius_m;
  std::unordered_map<PlaneGridKey, std::vector<std::size_t>, PlaneGridKeyHash> surface_grid;
  for (std::size_t index = 0U; index < result.labels.size(); ++index) {
    const TerrainLabel label = result.labels[index];
    if (label == TerrainLabel::kRamp) {
      ramp_candidates.push_back(index);
    }
    if ((label == TerrainLabel::kFloor || label == TerrainLabel::kRamp) &&
      index < points.size() && points[index].allFinite())
    {
      surface_grid[planeGridKey(points[index], radius_m)].push_back(index);
    }
  }

  result.stats.ramp_plane_candidate_points = ramp_candidates.size();
  for (const std::size_t candidate_index : ramp_candidates) {
    ++result.stats.ramp_plane_fits_attempted;
    const Eigen::Vector3d & candidate = points[candidate_index];
    const PlaneGridKey center = planeGridKey(candidate, radius_m);
    std::vector<std::size_t> neighborhood;
    for (std::int64_t x_offset = -1; x_offset <= 1; ++x_offset) {
      for (std::int64_t y_offset = -1; y_offset <= 1; ++y_offset) {
        const auto found = surface_grid.find(
          PlaneGridKey{center.x + x_offset, center.y + y_offset});
        if (found == surface_grid.end()) {
          continue;
        }
        for (const std::size_t index : found->second) {
          const Eigen::Vector3d delta = points[index] - candidate;
          if (delta.head<2>().squaredNorm() <= radius_squared_m) {
            neighborhood.push_back(index);
          }
        }
      }
    }

    const RampPlaneFit fit = fitRampPlane(points, neighborhood, config);
    if (fit.accepted) {
      ++result.stats.ramp_plane_validated_points;
      ++result.stats.ramp_plane_fits_accepted;
      result.stats.max_accepted_ramp_plane_rmse_m = std::max(
        result.stats.max_accepted_ramp_plane_rmse_m,
        fit.rmse_m);
      continue;
    }
    result.labels[candidate_index] = TerrainLabel::kUnknown;
    ++result.stats.ramp_plane_rejected_points;
    switch (fit.rejection_reason) {
      case RampPlaneFit::RejectionReason::kInsufficientPoints:
        ++result.stats.ramp_plane_rejected_insufficient_points;
        break;
      case RampPlaneFit::RejectionReason::kPlaneEstimation:
        ++result.stats.ramp_plane_rejected_plane_estimation;
        break;
      case RampPlaneFit::RejectionReason::kInlierSupport:
        ++result.stats.ramp_plane_rejected_inlier_support;
        break;
      case RampPlaneFit::RejectionReason::kLength:
        ++result.stats.ramp_plane_rejected_length;
        break;
      case RampPlaneFit::RejectionReason::kWidth:
        ++result.stats.ramp_plane_rejected_width;
        break;
      case RampPlaneFit::RejectionReason::kSlope:
        ++result.stats.ramp_plane_rejected_slope;
        break;
      case RampPlaneFit::RejectionReason::kRmse:
        ++result.stats.ramp_plane_rejected_rmse;
        break;
      case RampPlaneFit::RejectionReason::kNone:
        ++result.stats.ramp_plane_rejected_plane_estimation;
        break;
    }
  }
}

double quantile(std::vector<double> values, const double requested_quantile)
{
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double q = std::clamp(requested_quantile, 0.0, 1.0);
  const std::size_t index = static_cast<std::size_t>(
    std::floor(q * static_cast<double>(values.size() - 1U)));
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return values[index];
}

double selectSeedFloorHeight(
  std::vector<double> candidates,
  const IndoorTerrainConfig & config)
{
  if (candidates.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(candidates.begin(), candidates.end());
  const double reference_height_m = config.seed_reference_height_m;
  const double maximum_reference_delta_m = std::max(0.0, config.max_seed_height_delta_m);
  const double cluster_tolerance_m = std::max(
    1.0e-3, config.seed_height_cluster_tolerance_m);
  const int minimum_cluster_bins = std::max(1, config.seed_min_cluster_bins);

  struct CandidateCluster
  {
    std::size_t begin{0U};
    std::size_t end{0U};
    double center_m{0.0};
    double reference_distance_m{0.0};
  };
  std::optional<CandidateCluster> best;
  for (std::size_t begin = 0U; begin < candidates.size(); ++begin) {
    if (std::abs(candidates[begin] - reference_height_m) > maximum_reference_delta_m) {
      continue;
    }
    std::size_t end = begin + 1U;
    while (end < candidates.size() &&
      candidates[end] - candidates[begin] <= cluster_tolerance_m)
    {
      ++end;
    }
    std::vector<double> cluster(candidates.begin() + begin, candidates.begin() + end);
    const double center_m = quantile(cluster, 0.5);
    const CandidateCluster candidate{
      begin, end, center_m, std::abs(center_m - reference_height_m)};
    const std::size_t candidate_size = candidate.end - candidate.begin;
    const std::size_t best_size = best.has_value() ? best->end - best->begin : 0U;
    if (static_cast<int>(candidate_size) >= minimum_cluster_bins &&
      (!best.has_value() ||
      candidate.reference_distance_m < best->reference_distance_m ||
      (candidate.reference_distance_m == best->reference_distance_m &&
      candidate_size > best_size)))
    {
      best = candidate;
    }
  }
  if (!best.has_value()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::vector<double> selected(
    candidates.begin() + best->begin, candidates.begin() + best->end);
  return quantile(std::move(selected), config.floor_quantile);
}

std::size_t labelIndex(const TerrainLabel label)
{
  return static_cast<std::size_t>(label);
}

bool levelHasSupport(
  const std::vector<ProfileNode> & profile,
  const std::size_t start_index,
  const int direction,
  const int required_bins,
  const double height_tolerance_m,
  const double radial_bin_size_m,
  const double max_adjacent_gap_m)
{
  if (profile.empty() || start_index >= profile.size()) {
    return false;
  }
  const int required = std::max(1, required_bins);
  const double reference_z = profile[start_index].bin->floor_z_m;
  int supported = 1;
  std::size_t index = start_index;
  while (supported < required) {
    if ((direction < 0 && index == 0U) ||
      (direction > 0 && index + 1U >= profile.size()))
    {
      break;
    }
    const std::size_t next_index = direction < 0 ? index - 1U : index + 1U;
    const double radial_gap_m = static_cast<double>(
      std::abs(
        profile[next_index].radial_index - profile[index].radial_index)) *
      radial_bin_size_m;
    if (radial_gap_m > max_adjacent_gap_m ||
      std::abs(profile[next_index].bin->floor_z_m - reference_z) > height_tolerance_m)
    {
      break;
    }
    ++supported;
    index = next_index;
  }
  return supported >= required;
}

std::vector<std::vector<ProfileNode>> buildContinuousProfiles(
  const std::vector<const PolarBin *> & radial_bins,
  const double seed_floor_z_m,
  const IndoorTerrainConfig & config)
{
  std::vector<std::vector<ProfileNode>> profiles;
  std::vector<ProfileNode> active;
  const double radial_bin_size_m = std::max(1.0e-3, config.radial_bin_size_m);
  const double max_adjacent_gap_m = std::max(radial_bin_size_m, config.max_adjacent_gap_m);
  const double max_transition_height_m = std::max(
    std::max(
      std::max(config.step_max_height_m, config.drop_max_height_m),
      config.staircase_max_riser_height_m),
    std::max(config.ground_band_m, 1.0e-3));
  const double max_profile_height_m = config.staircase_enabled ?
    std::max(
    config.max_profile_height_above_base_m,
    config.staircase_max_profile_height_above_base_m) :
    config.max_profile_height_above_base_m;

  for (std::size_t radial_index = 0; radial_index < radial_bins.size(); ++radial_index) {
    const PolarBin * bin = radial_bins[radial_index];
    if (bin == nullptr || !bin->valid) {
      continue;
    }
    if (bin->floor_z_m > std::max(0.0, max_profile_height_m)) {
      if (!active.empty()) {
        profiles.push_back(std::move(active));
        active.clear();
      }
      continue;
    }

    if (active.empty()) {
      if (std::abs(bin->floor_z_m - seed_floor_z_m) <=
        std::max(0.0, config.max_seed_height_delta_m))
      {
        active.push_back(
          ProfileNode{static_cast<int>(radial_index), bin, TerrainLabel::kUnknown});
      }
      continue;
    }

    const ProfileNode & previous = active.back();
    const double radial_gap_m = static_cast<double>(
      static_cast<int>(radial_index) - previous.radial_index) * radial_bin_size_m;
    const double height_delta_m = bin->floor_z_m - previous.bin->floor_z_m;
    if (radial_gap_m <= max_adjacent_gap_m &&
      std::abs(height_delta_m) <= max_transition_height_m)
    {
      active.push_back(
        ProfileNode{static_cast<int>(radial_index), bin, TerrainLabel::kUnknown});
      continue;
    }

    if (!active.empty()) {
      profiles.push_back(std::move(active));
      active.clear();
    }
    if (std::abs(bin->floor_z_m - seed_floor_z_m) <=
      std::max(0.0, config.max_seed_height_delta_m))
    {
      active.push_back(
        ProfileNode{static_cast<int>(radial_index), bin, TerrainLabel::kUnknown});
    }
  }

  if (!active.empty()) {
    profiles.push_back(std::move(active));
  }
  return profiles;
}

void classifyStaircases(
  std::vector<ProfileNode> & profile,
  const IndoorTerrainConfig & config)
{
  const int minimum_risers = std::max(2, config.staircase_min_risers);
  if (!config.staircase_enabled || profile.size() < static_cast<std::size_t>(minimum_risers + 1)) {
    return;
  }

  const double radial_bin_size_m = std::max(1.0e-3, config.radial_bin_size_m);
  const double quantization_tolerance_m = 0.5 * radial_bin_size_m;
  const double minimum_tread_depth_m = std::max(
    radial_bin_size_m,
    config.staircase_min_tread_depth_m - quantization_tolerance_m);
  const double maximum_tread_depth_m = std::max(
    minimum_tread_depth_m,
    config.staircase_max_tread_depth_m + quantization_tolerance_m);
  const double minimum_riser_height_m = std::max(0.0, config.staircase_min_riser_height_m);
  const double maximum_riser_height_m = std::max(
    minimum_riser_height_m, config.staircase_max_riser_height_m);
  const double level_support_tolerance_m = std::max(
    1.0e-3,
    std::min(config.ground_band_m * 1.5, minimum_riser_height_m * 0.5));
  const int minimum_level_support_bins = std::max(2, config.transition_support_bins);

  std::vector<StairTransition> transitions;
  transitions.reserve(profile.size());
  for (std::size_t index = 1U; index < profile.size(); ++index) {
    const double radial_gap_m = static_cast<double>(
      profile[index].radial_index - profile[index - 1U].radial_index) *
      radial_bin_size_m;
    const double height_delta_m =
      profile[index].bin->floor_z_m - profile[index - 1U].bin->floor_z_m;
    const double absolute_height_m = std::abs(height_delta_m);
    if (radial_gap_m > config.max_adjacent_gap_m) {
      // A run cannot bridge an invalid discontinuity or missing profile span.
      transitions.push_back(StairTransition{index, 0, 0.0});
      continue;
    }
    if (absolute_height_m <= level_support_tolerance_m) {
      continue;
    }
    if (absolute_height_m < minimum_riser_height_m ||
      absolute_height_m > maximum_riser_height_m)
    {
      transitions.push_back(StairTransition{index, 0, 0.0});
      continue;
    }
    const bool supported_before = levelHasSupport(
      profile,
      index - 1U,
      -1,
      minimum_level_support_bins,
      level_support_tolerance_m,
      radial_bin_size_m,
      config.max_adjacent_gap_m);
    const bool supported_after = levelHasSupport(
      profile,
      index,
      1,
      minimum_level_support_bins,
      level_support_tolerance_m,
      radial_bin_size_m,
      config.max_adjacent_gap_m);
    if (!supported_before || !supported_after) {
      transitions.push_back(StairTransition{index, 0, 0.0});
      continue;
    }
    transitions.push_back(
      StairTransition{index, height_delta_m > 0.0 ? 1 : -1, absolute_height_m});
  }

  const auto classify_run =
    [&profile, &config, radial_bin_size_m, minimum_risers](
    const std::vector<StairTransition> & run) {
      if (run.size() < static_cast<std::size_t>(minimum_risers)) {
        return;
      }
      std::vector<double> riser_heights;
      std::vector<double> tread_depths;
      riser_heights.reserve(run.size());
      tread_depths.reserve(run.size() - 1U);
      for (std::size_t index = 0U; index < run.size(); ++index) {
        riser_heights.push_back(run[index].riser_height_m);
        if (index > 0U) {
          tread_depths.push_back(
            static_cast<double>(
              profile[run[index].profile_index].radial_index -
              profile[run[index - 1U].profile_index].radial_index) *
            radial_bin_size_m);
        }
      }
      const double median_riser_height_m = quantile(riser_heights, 0.5);
      const double median_tread_depth_m = quantile(tread_depths, 0.5);
      const bool risers_consistent = std::all_of(
        riser_heights.begin(), riser_heights.end(),
        [&config, median_riser_height_m](const double value) {
          return std::abs(value - median_riser_height_m) <=
          std::max(0.0, config.staircase_max_riser_height_deviation_m);
        });
      const bool treads_consistent = std::all_of(
        tread_depths.begin(), tread_depths.end(),
        [&config, median_tread_depth_m](const double value) {
          return std::abs(value - median_tread_depth_m) <=
          std::max(0.0, config.staircase_max_tread_depth_deviation_m);
        });
      if (!risers_consistent || !treads_consistent) {
        return;
      }

      const std::size_t first_source_index = run.front().profile_index - 1U;
      const std::size_t last_target_index = run.back().profile_index;
      const double radial_run_m = static_cast<double>(
        profile[last_target_index].radial_index -
        profile[first_source_index].radial_index) * radial_bin_size_m;
      const double height_run_m = std::abs(
        profile[last_target_index].bin->floor_z_m -
        profile[first_source_index].bin->floor_z_m);
      if (radial_run_m <= 0.0) {
        return;
      }
      const double staircase_slope_deg =
        std::atan2(height_run_m, radial_run_m) * 180.0 / M_PI;
      if (staircase_slope_deg < std::max(0.0, config.staircase_min_slope_deg) ||
        staircase_slope_deg > std::max(
          config.staircase_min_slope_deg, config.staircase_max_slope_deg))
      {
        return;
      }

      for (std::size_t profile_index = run.front().profile_index;
        profile_index <= last_target_index; ++profile_index)
      {
        profile[profile_index].label = TerrainLabel::kStaircase;
      }
    };

  std::vector<StairTransition> active_run;
  for (const StairTransition & transition : transitions) {
    if (transition.direction == 0) {
      classify_run(active_run);
      active_run.clear();
      continue;
    }
    bool extends_active_run = active_run.empty();
    if (!active_run.empty() && transition.direction == active_run.back().direction) {
      const double tread_depth_m = static_cast<double>(
        profile[transition.profile_index].radial_index -
        profile[active_run.back().profile_index].radial_index) * radial_bin_size_m;
      extends_active_run =
        tread_depth_m >= minimum_tread_depth_m && tread_depth_m <= maximum_tread_depth_m;
    }
    if (!extends_active_run) {
      classify_run(active_run);
      active_run.clear();
    }
    active_run.push_back(transition);
  }
  classify_run(active_run);
}

void classifyProfile(
  std::vector<ProfileNode> & profile,
  const IndoorTerrainConfig & config)
{
  if (profile.size() < 2U) {
    return;
  }

  const double radial_bin_size_m = std::max(1.0e-3, config.radial_bin_size_m);
  const double ramp_min_slope_deg = std::max(
    std::max(0.0, config.floor_max_slope_deg),
    std::max(0.0, config.ramp_min_slope_deg));
  const double ramp_max_slope_deg = std::max(ramp_min_slope_deg, config.ramp_max_slope_deg);
  const int minimum_ramp_transitions = std::max(1, config.ramp_min_run_bins - 1);
  const double minimum_ramp_run_length_m = std::max(0.0, config.ramp_min_run_length_m);

  std::vector<double> slope_deg(profile.size() - 1U, 0.0);
  for (std::size_t index = 1U; index < profile.size(); ++index) {
    const double radial_gap_m = static_cast<double>(
      profile[index].radial_index - profile[index - 1U].radial_index) * radial_bin_size_m;
    const double height_delta_m =
      profile[index].bin->floor_z_m - profile[index - 1U].bin->floor_z_m;
    slope_deg[index - 1U] = std::atan2(height_delta_m, radial_gap_m) * 180.0 / M_PI;
  }

  const double floor_max_slope_deg = std::max(0.0, config.floor_max_slope_deg);
  for (std::size_t transition = 0U; transition < slope_deg.size(); ++transition) {
    const bool below_general_height_gate =
      profile[transition].bin->floor_z_m <=
      std::max(0.0, config.max_profile_height_above_base_m) &&
      profile[transition + 1U].bin->floor_z_m <=
      std::max(0.0, config.max_profile_height_above_base_m);
    if (below_general_height_gate &&
      std::abs(slope_deg[transition]) <= floor_max_slope_deg)
    {
      profile[transition].label = TerrainLabel::kFloor;
      profile[transition + 1U].label = TerrainLabel::kFloor;
    }
  }

  std::size_t transition_index = 0U;
  while (transition_index < slope_deg.size()) {
    const double slope = slope_deg[transition_index];
    const int sign = slope > 0.0 ? 1 : (slope < 0.0 ? -1 : 0);
    const double absolute_slope = std::abs(slope);
    if (sign == 0 || absolute_slope < ramp_min_slope_deg ||
      absolute_slope > ramp_max_slope_deg)
    {
      ++transition_index;
      continue;
    }

    std::size_t run_end = transition_index + 1U;
    while (run_end < slope_deg.size()) {
      const double candidate = slope_deg[run_end];
      const int candidate_sign = candidate > 0.0 ? 1 : (candidate < 0.0 ? -1 : 0);
      const double candidate_absolute = std::abs(candidate);
      if (candidate_sign != sign || candidate_absolute < ramp_min_slope_deg ||
        candidate_absolute > ramp_max_slope_deg)
      {
        break;
      }
      ++run_end;
    }

    const int run_transition_count = static_cast<int>(run_end - transition_index);
    const double run_length_m = static_cast<double>(
      profile[run_end].radial_index - profile[transition_index].radial_index) *
      radial_bin_size_m;
    if (run_transition_count >= minimum_ramp_transitions &&
      run_length_m >= minimum_ramp_run_length_m)
    {
      for (std::size_t node_index = transition_index;
        node_index <= run_end && node_index < profile.size(); ++node_index)
      {
        if (profile[node_index].bin->floor_z_m <=
          std::max(0.0, config.max_profile_height_above_base_m))
        {
          profile[node_index].label = TerrainLabel::kRamp;
        }
      }
    }
    transition_index = run_end;
  }

  const double support_tolerance_m = std::max(
    1.0e-3,
    std::min(config.ground_band_m * 1.5, config.step_min_height_m * 0.75));
  for (std::size_t index = 1U; index < profile.size(); ++index) {
    if (profile[index - 1U].label == TerrainLabel::kRamp ||
      profile[index].label == TerrainLabel::kRamp)
    {
      continue;
    }
    const double height_delta_m =
      profile[index].bin->floor_z_m - profile[index - 1U].bin->floor_z_m;
    if (profile[index - 1U].bin->floor_z_m >
      std::max(0.0, config.max_profile_height_above_base_m) ||
      profile[index].bin->floor_z_m >
      std::max(0.0, config.max_profile_height_above_base_m))
    {
      continue;
    }
    const bool supported_before = levelHasSupport(
      profile,
      index - 1U,
      -1,
      config.transition_support_bins,
      support_tolerance_m,
      radial_bin_size_m,
      config.max_adjacent_gap_m);
    const bool supported_after = levelHasSupport(
      profile,
      index,
      1,
      config.transition_support_bins,
      support_tolerance_m,
      radial_bin_size_m,
      config.max_adjacent_gap_m);
    if (!supported_before || !supported_after) {
      continue;
    }

    if (height_delta_m >= std::max(0.0, config.step_min_height_m) &&
      height_delta_m <= std::max(config.step_min_height_m, config.step_max_height_m))
    {
      profile[index].label = TerrainLabel::kPositiveStep;
    } else if (-height_delta_m >= std::max(0.0, config.drop_min_height_m) &&
      -height_delta_m <= std::max(config.drop_min_height_m, config.drop_max_height_m))
    {
      profile[index].label = TerrainLabel::kObservedDrop;
    }
  }

  classifyStaircases(profile, config);
}

void estimateProfileRoughness(
  std::vector<ProfileNode> & profile,
  const IndoorTerrainConfig & config)
{
  if (!config.roughness_preview_enabled || profile.size() < 3U) {
    return;
  }
  const double radial_bin_size_m = std::max(1.0e-3, config.radial_bin_size_m);
  const double minimum_height_m = std::max(1.0e-6, config.roughness_min_height_m);
  const double maximum_score = std::max(1.0, config.roughness_max_score);

  for (std::size_t index = 1U; index + 1U < profile.size(); ++index) {
    const double left_range_m =
      static_cast<double>(profile[index - 1U].radial_index) * radial_bin_size_m;
    const double center_range_m =
      static_cast<double>(profile[index].radial_index) * radial_bin_size_m;
    const double right_range_m =
      static_cast<double>(profile[index + 1U].radial_index) * radial_bin_size_m;
    const double left_gap_m = center_range_m - left_range_m;
    const double right_gap_m = right_range_m - center_range_m;
    const double span_m = right_range_m - left_range_m;
    if (left_gap_m <= 0.0 || right_gap_m <= 0.0 || span_m <= 0.0 ||
      left_gap_m > config.max_adjacent_gap_m ||
      right_gap_m > config.max_adjacent_gap_m)
    {
      continue;
    }

    const double interpolation_ratio = (center_range_m - left_range_m) / span_m;
    const double interpolated_height_m =
      profile[index - 1U].bin->floor_z_m +
      interpolation_ratio *
      (profile[index + 1U].bin->floor_z_m - profile[index - 1U].bin->floor_z_m);
    const double high_frequency_height_m = std::abs(
      profile[index].bin->floor_z_m - interpolated_height_m);
    profile[index].roughness_score = std::clamp(
      high_frequency_height_m / minimum_height_m,
      0.0,
      maximum_score);
  }
}

}  // namespace

const char * terrainLabelName(const TerrainLabel label)
{
  switch (label) {
    case TerrainLabel::kUnknown:
      return "unknown";
    case TerrainLabel::kFloor:
      return "floor";
    case TerrainLabel::kRamp:
      return "ramp";
    case TerrainLabel::kPositiveStep:
      return "positive_step";
    case TerrainLabel::kObservedDrop:
      return "observed_drop";
    case TerrainLabel::kStaircase:
      return "staircase";
  }
  return "unknown";
}

float terrainLabelValue(const TerrainLabel label)
{
  return static_cast<float>(static_cast<std::uint8_t>(label));
}

std::array<std::uint8_t, 3U> terrainLabelColor(const TerrainLabel label)
{
  switch (label) {
    case TerrainLabel::kFloor:
      return {105U, 190U, 105U};
    case TerrainLabel::kRamp:
      return {45U, 165U, 255U};
    case TerrainLabel::kPositiveStep:
      return {255U, 165U, 35U};
    case TerrainLabel::kObservedDrop:
      return {235U, 55U, 55U};
    case TerrainLabel::kStaircase:
      return {190U, 90U, 245U};
    case TerrainLabel::kUnknown:
      return {145U, 145U, 145U};
  }
  return {145U, 145U, 145U};
}

const char * terrainSupportStateName(const TerrainSupportState state)
{
  switch (state) {
    case TerrainSupportState::kUnresolved:
      return "unresolved";
    case TerrainSupportState::kProbableSupport:
      return "probable_support";
    case TerrainSupportState::kConfirmedSupport:
      return "confirmed_support";
    case TerrainSupportState::kProbableHazard:
      return "probable_hazard";
    case TerrainSupportState::kConfirmedHazard:
      return "confirmed_hazard";
  }
  return "unresolved";
}

std::array<std::uint8_t, 3U> terrainSupportColor(
  const TerrainLabel label,
  const TerrainSupportState state)
{
  switch (state) {
    case TerrainSupportState::kConfirmedSupport:
      if (label == TerrainLabel::kRamp) {
        return {45U, 165U, 255U};
      }
      if (label == TerrainLabel::kFloor) {
        return {40U, 170U, 70U};
      }
      return {45U, 175U, 155U};
    case TerrainSupportState::kProbableSupport:
      if (label == TerrainLabel::kRamp) {
        return {85U, 215U, 225U};
      }
      if (label == TerrainLabel::kFloor) {
        return {175U, 225U, 85U};
      }
      return {135U, 220U, 160U};
    case TerrainSupportState::kConfirmedHazard:
      return terrainLabelColor(label);
    case TerrainSupportState::kProbableHazard:
      return {245U, 205U, 55U};
    case TerrainSupportState::kUnresolved:
      return {145U, 145U, 145U};
  }
  return {145U, 145U, 145U};
}

std::optional<Eigen::Matrix3d> gravityAlignmentRotation(
  const Eigen::Vector3d & up_in_source_frame)
{
  if (!up_in_source_frame.allFinite() || up_in_source_frame.norm() <= 1.0e-6) {
    return std::nullopt;
  }

  const Eigen::Vector3d up = up_in_source_frame.normalized();
  Eigen::Vector3d forward = Eigen::Vector3d::UnitX() - up * up.x();
  if (forward.norm() <= 1.0e-6) {
    forward = Eigen::Vector3d::UnitY() - up * up.y();
  }
  if (forward.norm() <= 1.0e-6) {
    return std::nullopt;
  }
  forward.normalize();
  const Eigen::Vector3d left = up.cross(forward).normalized();

  Eigen::Matrix3d gravity_R_source;
  gravity_R_source.row(0) = forward.transpose();
  gravity_R_source.row(1) = left.transpose();
  gravity_R_source.row(2) = up.transpose();
  return gravity_R_source;
}

TerrainClassificationResult classifyIndoorTerrain(
  const std::vector<Eigen::Vector3d> & points_gravity_aligned,
  const Eigen::Vector3d & sensor_origin,
  const IndoorTerrainConfig & config)
{
  TerrainClassificationResult result;
  result.labels.assign(points_gravity_aligned.size(), TerrainLabel::kUnknown);
  result.roughness_scores.assign(points_gravity_aligned.size(), 0.0);
  result.stats.input_points = points_gravity_aligned.size();
  if (!config.enabled || points_gravity_aligned.empty() || !sensor_origin.allFinite()) {
    return result;
  }
  result.stats.evaluated = true;

  const int azimuth_bin_count = std::max(8, config.azimuth_bins);
  const double radial_bin_size_m = std::max(1.0e-3, config.radial_bin_size_m);
  const double min_range_m = std::max(0.0, config.min_range_m);
  const double max_range_m = std::max(min_range_m + radial_bin_size_m, config.max_range_m);
  const int radial_bin_count = std::max(
    1,
    static_cast<int>(std::ceil(max_range_m / radial_bin_size_m)));
  std::vector<PolarBin> bins(
    static_cast<std::size_t>(azimuth_bin_count * radial_bin_count));

  for (std::size_t point_index = 0U;
    point_index < points_gravity_aligned.size(); ++point_index)
  {
    const Eigen::Vector3d & point = points_gravity_aligned[point_index];
    if (!point.allFinite()) {
      continue;
    }
    const Eigen::Vector3d relative = point - sensor_origin;
    const double range_xy_m = std::hypot(relative.x(), relative.y());
    if (range_xy_m < min_range_m || range_xy_m > max_range_m) {
      continue;
    }
    double azimuth = std::atan2(relative.y(), relative.x());
    if (azimuth < 0.0) {
      azimuth += 2.0 * M_PI;
    }
    const int azimuth_index = std::min(
      azimuth_bin_count - 1,
      static_cast<int>(
        std::floor(azimuth / (2.0 * M_PI) * static_cast<double>(azimuth_bin_count))));
    const int radial_index = std::min(
      radial_bin_count - 1,
      static_cast<int>(std::floor(range_xy_m / radial_bin_size_m)));
    bins[static_cast<std::size_t>(azimuth_index * radial_bin_count + radial_index)]
    .point_indices.push_back(point_index);
  }

  std::vector<double> seed_candidates;
  const int minimum_points = std::max(1, config.min_points_per_bin);
  const double seed_max_range_m = std::clamp(
    config.seed_max_range_m,
    min_range_m,
    max_range_m);
  for (int azimuth_index = 0; azimuth_index < azimuth_bin_count; ++azimuth_index) {
    for (int radial_index = 0; radial_index < radial_bin_count; ++radial_index) {
      PolarBin & bin = bins[static_cast<std::size_t>(
            azimuth_index * radial_bin_count + radial_index)];
      if (static_cast<int>(bin.point_indices.size()) < minimum_points) {
        continue;
      }
      std::vector<double> heights;
      heights.reserve(bin.point_indices.size());
      for (const std::size_t point_index : bin.point_indices) {
        heights.push_back(points_gravity_aligned[point_index].z());
      }
      bin.floor_z_m = quantile(std::move(heights), config.floor_quantile);
      bin.valid = std::isfinite(bin.floor_z_m);
      const double radial_center_m =
        (static_cast<double>(radial_index) + 0.5) * radial_bin_size_m;
      if (bin.valid && radial_center_m >= min_range_m &&
        radial_center_m <= seed_max_range_m &&
        bin.floor_z_m <= std::max(0.0, config.max_profile_height_above_base_m) &&
        std::abs(bin.floor_z_m - config.seed_reference_height_m) <=
        std::max(0.0, config.max_seed_height_delta_m))
      {
        seed_candidates.push_back(bin.floor_z_m);
      }
    }
  }

  const double seed_floor_z_m = selectSeedFloorHeight(seed_candidates, config);
  if (!std::isfinite(seed_floor_z_m)) {
    return result;
  }
  result.stats.seed_valid = true;
  result.stats.seed_floor_z_m = seed_floor_z_m;

  for (int azimuth_index = 0; azimuth_index < azimuth_bin_count; ++azimuth_index) {
    std::vector<const PolarBin *> radial_bins(static_cast<std::size_t>(radial_bin_count), nullptr);
    for (int radial_index = 0; radial_index < radial_bin_count; ++radial_index) {
      const PolarBin & bin = bins[static_cast<std::size_t>(
            azimuth_index * radial_bin_count + radial_index)];
      if (bin.valid) {
        radial_bins[static_cast<std::size_t>(radial_index)] = &bin;
      }
    }

    auto profiles = buildContinuousProfiles(
      radial_bins,
      seed_floor_z_m,
      config);
    for (auto & profile : profiles) {
      classifyProfile(profile, config);
      estimateProfileRoughness(profile, config);
      result.stats.profile_bins += profile.size();
      for (const ProfileNode & node : profile) {
        for (const std::size_t point_index : node.bin->point_indices) {
          if (std::abs(
              points_gravity_aligned[point_index].z() - node.bin->floor_z_m) >
            std::max(1.0e-3, config.ground_band_m))
          {
            continue;
          }
          result.labels[point_index] = node.label;
          result.roughness_scores[point_index] = node.roughness_score;
        }
      }
    }
  }

  validateRampPlanes(points_gravity_aligned, config, result);

  for (const TerrainLabel label : result.labels) {
    switch (label) {
      case TerrainLabel::kFloor:
        ++result.stats.floor_points;
        break;
      case TerrainLabel::kRamp:
        ++result.stats.ramp_points;
        break;
      case TerrainLabel::kPositiveStep:
        ++result.stats.positive_step_points;
        break;
      case TerrainLabel::kObservedDrop:
        ++result.stats.observed_drop_points;
        break;
      case TerrainLabel::kStaircase:
        ++result.stats.staircase_points;
        break;
      case TerrainLabel::kUnknown:
        break;
    }
  }
  for (const double score : result.roughness_scores) {
    if (!std::isfinite(score)) {
      continue;
    }
    result.stats.max_roughness_score = std::max(
      result.stats.max_roughness_score,
      score);
    if (score >= 1.0) {
      ++result.stats.roughness_preview_points;
    }
  }
  return result;
}

TerrainEvidenceAccumulator::TerrainEvidenceAccumulator(
  const TerrainAccumulatorConfig & config)
{
  configure(config);
}

void TerrainEvidenceAccumulator::configure(const TerrainAccumulatorConfig & config)
{
  config_ = config;
  config_.voxel_size_m = std::max(1.0e-3, config_.voxel_size_m);
  config_.min_confirmations = std::max(1, config_.min_confirmations);
  config_.min_viewpoint_distance_m = std::max(0.0, config_.min_viewpoint_distance_m);
  config_.min_view_label_confidence = std::clamp(
    config_.min_view_label_confidence, 0.5, 1.0);
  config_.min_label_confidence = std::clamp(
    config_.min_label_confidence,
    std::nextafter(0.5, 1.0),
    1.0);
  config_.probable_support_min_confidence = std::clamp(
    config_.probable_support_min_confidence,
    std::nextafter(0.5, 1.0),
    config_.min_label_confidence);
  config_.probable_support_max_roughness_score = std::max(
    0.0, config_.probable_support_max_roughness_score);
  config_.probable_support_max_anchor_distance_m = std::max(
    0.0, config_.probable_support_max_anchor_distance_m);
  config_.probable_support_max_neighbor_height_delta_m = std::max(
    0.0, config_.probable_support_max_neighbor_height_delta_m);
  config_.ramp_traversability_scale = std::clamp(
    config_.ramp_traversability_scale, 0.0, 1.0);
  clear();
}

void TerrainEvidenceAccumulator::clear()
{
  voxels_.clear();
  ambiguous_view_observation_count_ = 0U;
  cached_support_evidence_.clear();
  cached_stats_ = TerrainAccumulatorStats{};
}

std::size_t TerrainEvidenceAccumulator::VoxelKeyHash::operator()(const VoxelKey & key) const
{
  const auto mix = [](std::uint64_t value) {
      value ^= value >> 30U;
      value *= 0xbf58476d1ce4e5b9ULL;
      value ^= value >> 27U;
      value *= 0x94d049bb133111ebULL;
      value ^= value >> 31U;
      return value;
    };
  const std::uint64_t x = mix(static_cast<std::uint64_t>(key.x));
  const std::uint64_t y = mix(static_cast<std::uint64_t>(key.y) + 0x9e3779b97f4a7c15ULL);
  const std::uint64_t z = mix(static_cast<std::uint64_t>(key.z) + 0x243f6a8885a308d3ULL);
  return static_cast<std::size_t>(x ^ (y << 1U) ^ (z << 2U));
}

TerrainEvidenceAccumulator::VoxelKey TerrainEvidenceAccumulator::voxelKey(
  const Eigen::Vector3d & point) const
{
  return VoxelKey{
    static_cast<std::int64_t>(std::floor(point.x() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.y() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.z() / config_.voxel_size_m))};
}

void TerrainEvidenceAccumulator::addObservation(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<TerrainLabel> & labels,
  const Eigen::Vector3d & viewpoint)
{
  addObservation(points, labels, std::vector<double>(points.size(), 0.0), viewpoint);
}

void TerrainEvidenceAccumulator::addObservation(
  const std::vector<Eigen::Vector3d> & points,
  const std::vector<TerrainLabel> & labels,
  const std::vector<double> & roughness_scores,
  const Eigen::Vector3d & viewpoint)
{
  if (points.size() != labels.size() || points.size() != roughness_scores.size() ||
    !viewpoint.allFinite())
  {
    return;
  }

  struct LocalEvidence
  {
    std::array<Eigen::Vector3d, kTerrainLabelCount> position_sum;
    std::array<double, kTerrainLabelCount> roughness_score_sum{};
    std::array<int, kTerrainLabelCount> point_count{};

    LocalEvidence()
    {
      for (auto & value : position_sum) {
        value.setZero();
      }
    }
  };
  std::unordered_map<VoxelKey, LocalEvidence, VoxelKeyHash> local_voxels;
  for (std::size_t index = 0U; index < points.size(); ++index) {
    const TerrainLabel label = labels[index];
    const std::size_t label_index = labelIndex(label);
    if (label == TerrainLabel::kUnknown || label_index >= kTerrainLabelCount ||
      !points[index].allFinite())
    {
      continue;
    }
    if (!std::isfinite(roughness_scores[index]) || roughness_scores[index] < 0.0) {
      continue;
    }
    LocalEvidence & local = local_voxels[voxelKey(points[index])];
    local.position_sum[label_index] += points[index];
    local.roughness_score_sum[label_index] += roughness_scores[index];
    ++local.point_count[label_index];
  }

  for (const auto & entry : local_voxels) {
    const auto existing_voxel = voxels_.find(entry.first);
    bool independent_viewpoint = true;
    if (existing_voxel != voxels_.end()) {
      independent_viewpoint = std::all_of(
        existing_voxel->second.viewpoints.begin(),
        existing_voxel->second.viewpoints.end(),
        [this, &viewpoint](const Eigen::Vector3d & previous_viewpoint) {
          return (viewpoint - previous_viewpoint).norm() >=
          config_.min_viewpoint_distance_m;
        });
    }
    if (!independent_viewpoint) {
      continue;
    }
    const std::array<TerrainLabel, 5> tie_break_priority{
      TerrainLabel::kStaircase,
      TerrainLabel::kObservedDrop,
      TerrainLabel::kPositiveStep,
      TerrainLabel::kRamp,
      TerrainLabel::kFloor};
    TerrainLabel observed_label = TerrainLabel::kUnknown;
    int observed_point_count = 0;
    int total_point_count = 0;
    int support_point_count = 0;
    int hazard_point_count = 0;
    Eigen::Vector3d support_position_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d hazard_position_sum = Eigen::Vector3d::Zero();
    double support_roughness_sum = 0.0;
    double hazard_roughness_sum = 0.0;
    for (std::size_t index = 1U; index < kTerrainLabelCount; ++index) {
      const TerrainLabel label = static_cast<TerrainLabel>(index);
      const int point_count = entry.second.point_count[index];
      total_point_count += point_count;
      if (isSupportLabel(label)) {
        support_point_count += point_count;
        support_position_sum += entry.second.position_sum[index];
        support_roughness_sum += entry.second.roughness_score_sum[index];
      } else if (isHazardLabel(label)) {
        hazard_point_count += point_count;
        hazard_position_sum += entry.second.position_sum[index];
        hazard_roughness_sum += entry.second.roughness_score_sum[index];
      }
    }
    for (const TerrainLabel label : tie_break_priority) {
      const int point_count = entry.second.point_count[labelIndex(label)];
      if (point_count > observed_point_count) {
        observed_label = label;
        observed_point_count = point_count;
      }
    }
    const double view_label_confidence = total_point_count > 0 ?
      static_cast<double>(observed_point_count) / static_cast<double>(total_point_count) : 0.0;
    const bool semantic_vote_valid = observed_label != TerrainLabel::kUnknown &&
      observed_point_count > 0 &&
      view_label_confidence >= config_.min_view_label_confidence;
    const int group_point_count = std::max(support_point_count, hazard_point_count);
    const double view_group_confidence = total_point_count > 0 ?
      static_cast<double>(group_point_count) / static_cast<double>(total_point_count) : 0.0;
    const bool support_group_wins = support_point_count > hazard_point_count;
    const bool hazard_group_wins = hazard_point_count > support_point_count;
    const bool group_vote_valid = group_point_count > 0 &&
      (support_group_wins || hazard_group_wins) &&
      view_group_confidence >= config_.probable_support_min_confidence;
    if (!semantic_vote_valid) {
      ++ambiguous_view_observation_count_;
    }
    if (!semantic_vote_valid && !group_vote_valid) {
      continue;
    }

    VoxelEvidence & voxel = voxels_[entry.first];
    if (semantic_vote_valid) {
      LabelEvidence & evidence = voxel.labels[labelIndex(observed_label)];
      evidence.position_sum +=
        entry.second.position_sum[labelIndex(observed_label)] /
        static_cast<double>(observed_point_count);
      evidence.roughness_score_sum +=
        entry.second.roughness_score_sum[labelIndex(observed_label)] /
        static_cast<double>(observed_point_count);
      ++evidence.confirmations;
      evidence.viewpoints.push_back(viewpoint);
    }
    if (group_vote_valid) {
      LabelEvidence & group = support_group_wins ? voxel.support : voxel.hazard;
      const Eigen::Vector3d & position_sum = support_group_wins ?
        support_position_sum : hazard_position_sum;
      const double roughness_sum = support_group_wins ?
        support_roughness_sum : hazard_roughness_sum;
      group.position_sum += position_sum / static_cast<double>(group_point_count);
      group.roughness_score_sum += roughness_sum / static_cast<double>(group_point_count);
      ++group.confirmations;
      group.viewpoints.push_back(viewpoint);
    }
    voxel.viewpoints.push_back(viewpoint);
  }
  rebuildStatsCache();
}

TerrainLabel TerrainEvidenceAccumulator::dominantLabel(const VoxelEvidence & evidence) const
{
  const std::array<TerrainLabel, 5> priority{
    TerrainLabel::kStaircase,
    TerrainLabel::kObservedDrop,
    TerrainLabel::kPositiveStep,
    TerrainLabel::kRamp,
    TerrainLabel::kFloor};
  TerrainLabel winner = TerrainLabel::kUnknown;
  int winner_confirmations = 0;
  for (const TerrainLabel label : priority) {
    const int confirmations = evidence.labels[labelIndex(label)].confirmations;
    if (confirmations > winner_confirmations) {
      winner = label;
      winner_confirmations = confirmations;
    }
  }
  return winner;
}

TerrainLabel TerrainEvidenceAccumulator::confirmedLabel(
  const VoxelEvidence & evidence) const
{
  const TerrainLabel winner = dominantLabel(evidence);
  const int winner_confirmations = winner == TerrainLabel::kUnknown ? 0 :
    evidence.labels[labelIndex(winner)].confirmations;
  int total_confirmations = 0;
  for (std::size_t index = 1U; index < kTerrainLabelCount; ++index) {
    total_confirmations += evidence.labels[index].confirmations;
  }
  if (winner_confirmations < config_.min_confirmations || total_confirmations <= 0) {
    return TerrainLabel::kUnknown;
  }
  const double confidence =
    static_cast<double>(winner_confirmations) / static_cast<double>(total_confirmations);
  return confidence >= config_.min_label_confidence ? winner : TerrainLabel::kUnknown;
}

std::vector<TerrainEvidencePoint> TerrainEvidenceAccumulator::evidence() const
{
  std::vector<TerrainEvidencePoint> output;
  output.reserve(voxels_.size());
  for (const auto & entry : voxels_) {
    const TerrainLabel label = dominantLabel(entry.second);
    if (label == TerrainLabel::kUnknown) {
      continue;
    }
    const LabelEvidence & label_evidence = entry.second.labels[labelIndex(label)];
    int observations = 0;
    for (std::size_t index = 1U; index < kTerrainLabelCount; ++index) {
      observations += entry.second.labels[index].confirmations;
    }
    if (label_evidence.confirmations <= 0 || observations <= 0) {
      continue;
    }
    const double confidence =
      static_cast<double>(label_evidence.confirmations) /
      static_cast<double>(observations);
    output.push_back(
      TerrainEvidencePoint{
          label_evidence.position_sum /
          static_cast<double>(label_evidence.confirmations),
          label,
          label_evidence.confirmations,
          observations,
          confidence,
          label_evidence.roughness_score_sum /
          static_cast<double>(label_evidence.confirmations),
          confirmedLabel(entry.second) == label});
  }
  std::sort(
    output.begin(),
    output.end(),
    [](const TerrainEvidencePoint & lhs, const TerrainEvidencePoint & rhs) {
      if (lhs.position.x() != rhs.position.x()) {
        return lhs.position.x() < rhs.position.x();
      }
      if (lhs.position.y() != rhs.position.y()) {
        return lhs.position.y() < rhs.position.y();
      }
      return lhs.position.z() < rhs.position.z();
    });
  return output;
}

std::vector<ConfirmedTerrainPoint> TerrainEvidenceAccumulator::confirmedEvidence() const
{
  std::vector<ConfirmedTerrainPoint> output;
  const auto all_evidence = evidence();
  output.reserve(all_evidence.size());
  for (const auto & item : all_evidence) {
    if (!item.confirmed) {
      continue;
    }
    output.push_back(
      ConfirmedTerrainPoint{
          item.position,
          item.label,
          item.confirmations,
          item.observations,
          item.confidence,
          item.geometric_roughness_score});
  }
  return output;
}

std::vector<TerrainSupportPoint> TerrainEvidenceAccumulator::supportEvidence() const
{
  return cached_support_evidence_;
}

std::vector<TerrainSupportPoint> TerrainEvidenceAccumulator::buildSupportEvidence() const
{
  struct Candidate
  {
    VoxelKey key;
    TerrainSupportPoint point;
  };

  const auto dominant_group_label = [](const VoxelEvidence & voxel, const bool support) {
      const std::array<TerrainLabel, 3> support_priority{
        TerrainLabel::kRamp,
        TerrainLabel::kFloor,
        TerrainLabel::kUnknown};
      const std::array<TerrainLabel, 3> hazard_priority{
        TerrainLabel::kStaircase,
        TerrainLabel::kObservedDrop,
        TerrainLabel::kPositiveStep};
      TerrainLabel winner = TerrainLabel::kUnknown;
      int winner_count = 0;
      bool tied = false;
      const auto consider = [&](const TerrainLabel label) {
          if (label == TerrainLabel::kUnknown || isSupportLabel(label) != support) {
            return;
          }
          const int count = voxel.labels[labelIndex(label)].confirmations;
          if (count > winner_count) {
            winner = label;
            winner_count = count;
            tied = false;
          } else if (count > 0 && count == winner_count) {
            tied = true;
          }
        };
      if (support) {
        for (const TerrainLabel label : support_priority) {
          consider(label);
        }
      } else {
        for (const TerrainLabel label : hazard_priority) {
          consider(label);
        }
      }
      return tied ? TerrainLabel::kUnknown : winner;
    };

  std::vector<Candidate> candidates;
  candidates.reserve(voxels_.size());
  for (const auto & entry : voxels_) {
    const VoxelEvidence & voxel = entry.second;
    const int support_observations = voxel.support.confirmations;
    const int hazard_observations = voxel.hazard.confirmations;
    const int observations = support_observations + hazard_observations;
    if (observations <= 0) {
      continue;
    }

    const bool support_wins = support_observations > hazard_observations;
    const bool hazard_wins = hazard_observations > support_observations;
    const LabelEvidence * selected_group = nullptr;
    int confirmations = 0;
    if (support_wins) {
      selected_group = &voxel.support;
      confirmations = support_observations;
    } else if (hazard_wins) {
      selected_group = &voxel.hazard;
      confirmations = hazard_observations;
    }

    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double roughness_score = 0.0;
    if (selected_group != nullptr && confirmations > 0) {
      position = selected_group->position_sum / static_cast<double>(confirmations);
      roughness_score =
        selected_group->roughness_score_sum / static_cast<double>(confirmations);
    } else {
      position = (voxel.support.position_sum + voxel.hazard.position_sum) /
        static_cast<double>(observations);
      roughness_score =
        (voxel.support.roughness_score_sum + voxel.hazard.roughness_score_sum) /
        static_cast<double>(observations);
    }
    if (!position.allFinite() || !std::isfinite(roughness_score)) {
      continue;
    }

    TerrainLabel label = TerrainLabel::kUnknown;
    if (support_wins) {
      label = dominant_group_label(voxel, true);
    } else if (hazard_wins) {
      label = dominant_group_label(voxel, false);
    }
    int semantic_observations = 0;
    int semantic_confirmations = 0;
    for (std::size_t index = 1U; index < kTerrainLabelCount; ++index) {
      semantic_observations += voxel.labels[index].confirmations;
    }
    if (label != TerrainLabel::kUnknown) {
      semantic_confirmations = voxel.labels[labelIndex(label)].confirmations;
    }
    const double label_confidence = semantic_observations > 0 ?
      static_cast<double>(semantic_confirmations) /
      static_cast<double>(semantic_observations) : 0.0;
    const double support_probability =
      static_cast<double>(support_observations) / static_cast<double>(observations);
    const double group_confidence =
      static_cast<double>(confirmations) / static_cast<double>(observations);

    TerrainSupportState state = TerrainSupportState::kUnresolved;
    if (support_wins && hazard_observations == 0 &&
      confirmations >= config_.min_confirmations &&
      group_confidence >= config_.min_label_confidence)
    {
      state = TerrainSupportState::kConfirmedSupport;
    } else if (support_wins && hazard_observations == 0 &&
      group_confidence >= config_.probable_support_min_confidence &&
      roughness_score <= config_.probable_support_max_roughness_score)
    {
      state = TerrainSupportState::kProbableSupport;
    } else if (hazard_wins && confirmations >= config_.min_confirmations &&
      group_confidence >= config_.min_label_confidence)
    {
      state = TerrainSupportState::kConfirmedHazard;
    } else if (hazard_wins &&
      group_confidence >= config_.probable_support_min_confidence)
    {
      state = TerrainSupportState::kProbableHazard;
    }

    double geometric_traversability_score = 0.0;
    if (state == TerrainSupportState::kProbableSupport ||
      state == TerrainSupportState::kConfirmedSupport)
    {
      const double maturity = std::clamp(
        static_cast<double>(support_observations) /
        static_cast<double>(config_.min_confirmations),
        0.0,
        1.0);
      const double roughness_scale = std::max(
        config_.probable_support_max_roughness_score, 1.0e-6);
      const double roughness_factor =
        1.0 / (1.0 + roughness_score / roughness_scale);
      const double surface_factor = label == TerrainLabel::kRamp ?
        config_.ramp_traversability_scale : 1.0;
      geometric_traversability_score = std::clamp(
        support_probability * maturity * roughness_factor * surface_factor,
        0.0,
        1.0);
    }

    candidates.push_back(
      Candidate{
          entry.first,
          TerrainSupportPoint{
            position,
            label,
            state,
            confirmations,
            observations,
            label_confidence,
            support_probability,
            geometric_traversability_score,
            roughness_score,
            false}});
  }

  if (config_.probable_support_require_confirmed_connectivity) {
    // Index confirmed anchors in XY buckets sized to the connectivity radius.
    // A probable point can only be within range of anchors in its own bucket
    // or one of the eight adjacent buckets. This keeps rebuilding linear in
    // map size instead of probing a 3D voxel cube for every probable point.
    const double bucket_size_m = config_.probable_support_max_anchor_distance_m;
    std::unordered_map<PlaneGridKey, std::vector<std::size_t>, PlaneGridKeyHash>
    confirmed_buckets;
    confirmed_buckets.reserve(candidates.size());
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
      if (candidates[index].point.state == TerrainSupportState::kConfirmedSupport) {
        confirmed_buckets[planeGridKey(candidates[index].point.position, bucket_size_m)]
        .push_back(index);
      }
    }

    const double max_xy_distance_squared =
      config_.probable_support_max_anchor_distance_m *
      config_.probable_support_max_anchor_distance_m;
    for (Candidate & candidate : candidates) {
      if (candidate.point.state != TerrainSupportState::kProbableSupport) {
        continue;
      }
      bool connected = false;
      const PlaneGridKey candidate_bucket =
        planeGridKey(candidate.point.position, bucket_size_m);
      for (std::int64_t dx = -1; dx <= 1 && !connected; ++dx) {
        for (std::int64_t dy = -1; dy <= 1 && !connected; ++dy) {
          const auto bucket = confirmed_buckets.find(
            PlaneGridKey{candidate_bucket.x + dx, candidate_bucket.y + dy});
          if (bucket == confirmed_buckets.end()) {
            continue;
          }
          for (const std::size_t anchor_index : bucket->second) {
            const Eigen::Vector3d delta =
              candidates[anchor_index].point.position - candidate.point.position;
            if (delta.head<2>().squaredNorm() <= max_xy_distance_squared &&
              std::abs(delta.z()) <=
              config_.probable_support_max_neighbor_height_delta_m)
            {
              connected = true;
              break;
            }
          }
        }
      }
      if (!connected) {
        candidate.point.state = TerrainSupportState::kUnresolved;
        candidate.point.geometric_traversability_score = 0.0;
        candidate.point.connectivity_rejected = true;
      }
    }
  }

  std::vector<TerrainSupportPoint> output;
  output.reserve(candidates.size());
  for (auto & candidate : candidates) {
    output.push_back(std::move(candidate.point));
  }
  std::sort(
    output.begin(),
    output.end(),
    [](const TerrainSupportPoint & lhs, const TerrainSupportPoint & rhs) {
      if (lhs.position.x() != rhs.position.x()) {
        return lhs.position.x() < rhs.position.x();
      }
      if (lhs.position.y() != rhs.position.y()) {
        return lhs.position.y() < rhs.position.y();
      }
      return lhs.position.z() < rhs.position.z();
    });
  return output;
}

void TerrainEvidenceAccumulator::rebuildStatsCache()
{
  const auto support_rebuild_start = std::chrono::steady_clock::now();
  TerrainAccumulatorStats output;
  output.candidate_voxels = voxels_.size();
  output.ambiguous_view_observations = ambiguous_view_observation_count_;
  for (const auto & entry : voxels_) {
    int labels_with_observations = 0;
    for (std::size_t index = 1U; index < kTerrainLabelCount; ++index) {
      if (entry.second.labels[index].confirmations > 0) {
        ++labels_with_observations;
      }
    }
    if (labels_with_observations > 1) {
      ++output.conflicted_voxels;
    }
    const TerrainLabel label = confirmedLabel(entry.second);
    if (label == TerrainLabel::kUnknown) {
      continue;
    }
    ++output.confirmed_voxels;
    ++output.confirmed_by_label[labelIndex(label)];
  }
  cached_support_evidence_ = buildSupportEvidence();
  output.support_rebuild_runtime_ms =
    std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - support_rebuild_start).count();
  for (const auto & point : cached_support_evidence_) {
    switch (point.state) {
      case TerrainSupportState::kProbableSupport:
        ++output.probable_support_voxels;
        break;
      case TerrainSupportState::kConfirmedSupport:
        ++output.confirmed_support_voxels;
        break;
      case TerrainSupportState::kProbableHazard:
        ++output.probable_hazard_voxels;
        break;
      case TerrainSupportState::kConfirmedHazard:
        ++output.confirmed_hazard_voxels;
        break;
      case TerrainSupportState::kUnresolved:
        ++output.unresolved_voxels;
        if (point.connectivity_rejected) {
          ++output.isolated_probable_support_voxels;
        }
        break;
    }
  }
  cached_stats_ = output;
}

TerrainAccumulatorStats TerrainEvidenceAccumulator::stats() const
{
  return cached_stats_;
}

}  // namespace slam_3d_core
}  // namespace gole
