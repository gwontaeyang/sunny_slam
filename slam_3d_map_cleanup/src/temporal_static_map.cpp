#include "slam_3d_map_cleanup/temporal_static_map.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace gole
{
namespace slam_3d_map_cleanup
{
namespace
{

constexpr double kDirectionEpsilon = 1.0e-12;

bool finiteNonnegative(const double value)
{
  return std::isfinite(value) && value >= 0.0;
}

std::size_t mixHash(const std::size_t seed, const std::size_t value)
{
  return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U));
}

}  // namespace

bool validTemporalStaticMapConfig(const TemporalStaticMapConfig & config)
{
  return
    std::isfinite(config.voxel_size_m) && config.voxel_size_m > 0.0 &&
    config.min_hit_observations >= 1 &&
    finiteNonnegative(config.min_observation_span_s) &&
    finiteNonnegative(config.min_viewpoint_distance_m) &&
    finiteNonnegative(config.ray_endpoint_margin_m) &&
    std::isfinite(config.max_ray_length_m) && config.max_ray_length_m > 0.0 &&
    config.max_rays_per_observation >= 0 &&
    config.min_prior_free_observations >= 1 &&
    config.max_prior_free_observations >= config.min_prior_free_observations &&
    finiteNonnegative(config.prior_free_max_sensor_vertical_drop_m) &&
    std::isfinite(config.promotion_min_hit_ratio) &&
    config.promotion_min_hit_ratio > 0.0 && config.promotion_min_hit_ratio <= 1.0 &&
    std::isfinite(config.retention_min_hit_ratio) &&
    config.retention_min_hit_ratio >= 0.0 &&
    config.retention_min_hit_ratio <= config.promotion_min_hit_ratio &&
    config.min_miss_observations_for_demotion >= 1;
}

TemporalStaticMapAccumulator::TemporalStaticMapAccumulator(
  const TemporalStaticMapConfig & config)
{
  setConfig(config);
}

void TemporalStaticMapAccumulator::setConfig(const TemporalStaticMapConfig & config)
{
  if (!validTemporalStaticMapConfig(config)) {
    throw std::invalid_argument("invalid temporal static map configuration");
  }
  config_ = config;
  reset();
}

void TemporalStaticMapAccumulator::reset()
{
  voxels_.clear();
  prior_free_voxels_.clear();
  stats_ = TemporalStaticMapStats{};
}

bool TemporalStaticMapAccumulator::VoxelKey::operator==(const VoxelKey & other) const
{
  return x == other.x && y == other.y && z == other.z;
}

bool TemporalStaticMapAccumulator::VoxelKey::operator<(const VoxelKey & other) const
{
  return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
}

std::size_t TemporalStaticMapAccumulator::VoxelKeyHash::operator()(
  const VoxelKey & key) const
{
  std::size_t seed = std::hash<std::int64_t>{}(key.x);
  seed = mixHash(seed, std::hash<std::int64_t>{}(key.y));
  return mixHash(seed, std::hash<std::int64_t>{}(key.z));
}

TemporalStaticMapAccumulator::VoxelKey TemporalStaticMapAccumulator::voxelKey(
  const Eigen::Vector3d & point) const
{
  return VoxelKey{
    static_cast<std::int64_t>(std::floor(point.x() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.y() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.z() / config_.voxel_size_m))};
}

double TemporalStaticMapAccumulator::hitRatio(const VoxelEvidence & evidence) const
{
  const int total = evidence.hit_observations + evidence.miss_observations;
  if (total <= 0) {
    return 0.0;
  }
  return static_cast<double>(evidence.hit_observations) / static_cast<double>(total);
}

bool TemporalStaticMapAccumulator::matureForPromotion(
  const VoxelEvidence & evidence) const
{
  if (evidence.hit_observations < config_.min_hit_observations) {
    return false;
  }
  const bool no_separation_gate =
    config_.min_observation_span_s <= 0.0 &&
    config_.min_viewpoint_distance_m <= 0.0;
  const bool time_separated =
    evidence.last_stamp_s - evidence.first_stamp_s >= config_.min_observation_span_s;
  const bool viewpoint_separated =
    evidence.max_viewpoint_distance_m >= config_.min_viewpoint_distance_m;
  return no_separation_gate || time_separated || viewpoint_separated;
}

void TemporalStaticMapAccumulator::updateEvidenceState(VoxelEvidence & evidence)
{
  const bool was_confirmed = evidence.confirmed;
  if (!evidence.confirmed) {
    evidence.confirmed =
      matureForPromotion(evidence) &&
      hitRatio(evidence) >= config_.promotion_min_hit_ratio;
  } else if (
    evidence.misses_since_last_hit >= config_.min_miss_observations_for_demotion &&
    hitRatio(evidence) < config_.retention_min_hit_ratio)
  {
    evidence.confirmed = false;
  }

  const bool was_contradicted = evidence.contradicted;
  evidence.contradicted =
    evidence.miss_observations > 0 &&
    hitRatio(evidence) < config_.promotion_min_hit_ratio;
  if (evidence.confirmed != was_confirmed) {
    if (evidence.confirmed) {
      ++stats_.promotions;
      ++stats_.confirmed_voxels;
    } else {
      ++stats_.demotions;
      --stats_.confirmed_voxels;
    }
  }
  if (evidence.contradicted != was_contradicted) {
    if (evidence.contradicted) {
      ++stats_.contradicted_voxels;
    } else {
      --stats_.contradicted_voxels;
    }
  }
  stats_.rejected_voxels = stats_.candidate_voxels - stats_.confirmed_voxels;
}

void TemporalStaticMapAccumulator::applyRayMisses(
  const Eigen::Vector3d & sensor_origin,
  const Eigen::Vector3d & endpoint,
  const LocalEndpointMap & endpoints_in_observation,
  const std::size_t observation_id)
{
  const Eigen::Vector3d delta = endpoint - sensor_origin;
  const double ray_length_m = delta.norm();
  if (!sensor_origin.allFinite() || !endpoint.allFinite() ||
    !std::isfinite(ray_length_m) || ray_length_m <= kDirectionEpsilon)
  {
    return;
  }

  const double free_length_m = std::min(
    std::max(0.0, ray_length_m - config_.ray_endpoint_margin_m),
    config_.max_ray_length_m);
  if (free_length_m <= kDirectionEpsilon) {
    return;
  }

  const Eigen::Vector3d direction = delta / ray_length_m;
  VoxelKey current = voxelKey(sensor_origin);
  const Eigen::Vector3d free_endpoint = sensor_origin + direction * free_length_m;
  const VoxelKey final_key = voxelKey(free_endpoint);

  std::array<int, 3> step{0, 0, 0};
  std::array<double, 3> next_boundary_distance{
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};
  std::array<double, 3> voxel_crossing_distance{
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::infinity()};

  const auto initialize_axis = [&](const int axis, const std::int64_t index) {
      const double component = direction[axis];
      if (std::abs(component) <= kDirectionEpsilon) {
        return;
      }
      step[axis] = component > 0.0 ? 1 : -1;
      const double boundary = static_cast<double>(
        step[axis] > 0 ? index + 1 : index) * config_.voxel_size_m;
      next_boundary_distance[axis] =
        std::max(0.0, (boundary - sensor_origin[axis]) / component);
      voxel_crossing_distance[axis] = config_.voxel_size_m / std::abs(component);
    };
  initialize_axis(0, current.x);
  initialize_axis(1, current.y);
  initialize_axis(2, current.z);

  const std::size_t maximum_cells = static_cast<std::size_t>(
    std::ceil(3.0 * free_length_m / config_.voxel_size_m)) + 4U;
  double travelled_m = 0.0;
  for (std::size_t iteration = 0U; iteration < maximum_cells; ++iteration) {
    ++stats_.ray_voxel_checks;
    if (endpoints_in_observation.find(current) == endpoints_in_observation.end()) {
      const auto candidate = voxels_.find(current);
      if (candidate != voxels_.end()) {
        auto & evidence = candidate->second;
        if (evidence.last_miss_observation != observation_id) {
          evidence.last_miss_observation = observation_id;
          ++evidence.miss_observations;
          ++evidence.misses_since_last_hit;
          ++stats_.ray_miss_updates;
          updateEvidenceState(evidence);
        }
      } else if (config_.prior_free_enabled) {
        const double voxel_center_z =
          (static_cast<double>(current.z) + 0.5) * config_.voxel_size_m;
        const bool below_prior_free_band =
          config_.prior_free_max_sensor_vertical_drop_m > 0.0 &&
          sensor_origin.z() - voxel_center_z >
          config_.prior_free_max_sensor_vertical_drop_m;
        if (!below_prior_free_band) {
          const auto insertion = prior_free_voxels_.try_emplace(current);
          auto & prior_free = insertion.first->second;
          if (insertion.second) {
            stats_.prior_free_voxel_count = prior_free_voxels_.size();
          }
          if (prior_free.last_observation != observation_id) {
            prior_free.last_observation = observation_id;
            if (prior_free.observations < config_.max_prior_free_observations) {
              ++prior_free.observations;
              ++stats_.prior_free_update_count;
            }
          }
        }
      }
    }

    if (current == final_key) {
      break;
    }
    const double next_distance = std::min(
      next_boundary_distance[0],
      std::min(next_boundary_distance[1], next_boundary_distance[2]));
    if (!std::isfinite(next_distance) || next_distance > free_length_m) {
      break;
    }
    travelled_m = next_distance;
    for (int axis = 0; axis < 3; ++axis) {
      if (std::abs(next_boundary_distance[axis] - travelled_m) > 1.0e-10) {
        continue;
      }
      if (axis == 0) {
        current.x += step[axis];
      } else if (axis == 1) {
        current.y += step[axis];
      } else {
        current.z += step[axis];
      }
      next_boundary_distance[axis] += voxel_crossing_distance[axis];
    }
  }
}

void TemporalStaticMapAccumulator::addObservation(
  const double stamp_s,
  const Eigen::Vector3d & sensor_origin,
  const std::vector<Eigen::Vector3d> & endpoints)
{
  addObservation(
    stamp_s,
    std::vector<TemporalStaticMapSensorView>{{sensor_origin, endpoints}});
}

void TemporalStaticMapAccumulator::addObservation(
  const double stamp_s,
  const std::vector<TemporalStaticMapSensorView> & sensor_views)
{
  if (!config_.enabled) {
    return;
  }
  const auto started = std::chrono::steady_clock::now();
  ++stats_.observations;
  const std::size_t observation_id = stats_.observations;

  LocalEndpointMap local_endpoints;
  std::size_t total_input_points = 0U;
  for (const auto & view : sensor_views) {
    total_input_points += view.endpoints.size();
  }
  stats_.input_points += total_input_points;
  local_endpoints.reserve(total_input_points);
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> rays;
  rays.reserve(total_input_points);
  std::vector<const TemporalStaticMapSensorView *> ordered_sensor_views;
  ordered_sensor_views.reserve(sensor_views.size());
  for (const auto & view : sensor_views) {
    ordered_sensor_views.push_back(&view);
  }
  std::sort(
    ordered_sensor_views.begin(), ordered_sensor_views.end(),
    [](const auto * lhs, const auto * rhs) {
      const bool lhs_finite = lhs->sensor_origin.allFinite();
      const bool rhs_finite = rhs->sensor_origin.allFinite();
      if (lhs_finite != rhs_finite) {
        return lhs_finite;
      }
      if (!lhs_finite) {
        return lhs->endpoints.size() < rhs->endpoints.size();
      }
      const auto lhs_origin = std::make_tuple(
        lhs->sensor_origin.x(), lhs->sensor_origin.y(), lhs->sensor_origin.z());
      const auto rhs_origin = std::make_tuple(
        rhs->sensor_origin.x(), rhs->sensor_origin.y(), rhs->sensor_origin.z());
      return lhs_origin < rhs_origin;
    });
  for (const auto * view : ordered_sensor_views) {
    LocalEndpointMap view_endpoints;
    view_endpoints.reserve(view->endpoints.size());
    for (const auto & endpoint : view->endpoints) {
      if (!endpoint.allFinite()) {
        continue;
      }
      auto & local = view_endpoints[voxelKey(endpoint)];
      local.sum += endpoint;
      ++local.point_count;
    }
    for (const auto & item : view_endpoints) {
      if (item.second.point_count <= 0) {
        continue;
      }
      const Eigen::Vector3d centroid = item.second.sum /
        static_cast<double>(item.second.point_count);
      auto & merged = local_endpoints[item.first];
      merged.sum += item.second.sum;
      merged.point_count += item.second.point_count;
      if (view->sensor_origin.allFinite()) {
        merged.viewpoint_sum += view->sensor_origin;
        ++merged.viewpoint_count;
      }
      if (view->sensor_origin.allFinite()) {
        rays.emplace_back(view->sensor_origin, centroid);
      }
    }
  }

  std::vector<std::pair<VoxelKey, Eigen::Vector3d>> ordered_endpoints;
  ordered_endpoints.reserve(local_endpoints.size());
  for (const auto & item : local_endpoints) {
    if (item.second.point_count <= 0) {
      continue;
    }
    ordered_endpoints.emplace_back(
      item.first,
      item.second.sum / static_cast<double>(item.second.point_count));
  }
  std::sort(
    ordered_endpoints.begin(), ordered_endpoints.end(),
    [](const auto & lhs, const auto & rhs) {return lhs.first < rhs.first;});

  for (const auto & item : ordered_endpoints) {
    const auto local = local_endpoints.find(item.first);
    const bool have_viewpoint =
      local != local_endpoints.end() && local->second.viewpoint_count > 0;
    Eigen::Vector3d viewpoint = Eigen::Vector3d::Zero();
    if (have_viewpoint) {
      viewpoint = local->second.viewpoint_sum /
        static_cast<double>(local->second.viewpoint_count);
    }
    const bool new_voxel = voxels_.find(item.first) == voxels_.end();
    int prior_free_misses = 0;
    if (new_voxel) {
      const auto prior_free = prior_free_voxels_.find(item.first);
      if (prior_free != prior_free_voxels_.end()) {
        if (prior_free->second.observations >= config_.min_prior_free_observations) {
          prior_free_misses = std::min(
            prior_free->second.observations,
            config_.max_prior_free_observations);
          ++stats_.prior_free_seeded_candidate_count;
        }
        prior_free_voxels_.erase(prior_free);
        stats_.prior_free_voxel_count = prior_free_voxels_.size();
      }
    }
    auto & evidence = voxels_[item.first];
    if (new_voxel) {
      ++stats_.candidate_voxels;
      ++stats_.rejected_voxels;
      evidence.miss_observations = prior_free_misses;
      evidence.first_stamp_s = std::isfinite(stamp_s) ? stamp_s : 0.0;
      evidence.last_stamp_s = evidence.first_stamp_s;
      if (have_viewpoint) {
        evidence.first_viewpoint = viewpoint;
        evidence.have_first_viewpoint = true;
      }
    }
    evidence.position_sum += item.second;
    ++evidence.hit_observations;
    evidence.misses_since_last_hit = 0;
    if (std::isfinite(stamp_s)) {
      evidence.first_stamp_s = std::min(evidence.first_stamp_s, stamp_s);
      evidence.last_stamp_s = std::max(evidence.last_stamp_s, stamp_s);
    }
    if (have_viewpoint && !evidence.have_first_viewpoint) {
      evidence.first_viewpoint = viewpoint;
      evidence.have_first_viewpoint = true;
    } else if (have_viewpoint) {
      evidence.max_viewpoint_distance_m = std::max(
        evidence.max_viewpoint_distance_m,
        (viewpoint - evidence.first_viewpoint).norm());
    }
    ++stats_.endpoint_hit_updates;
    updateEvidenceState(evidence);
  }

  if (config_.ray_clearing_enabled && config_.max_rays_per_observation > 0 &&
    !rays.empty())
  {
    std::sort(
      rays.begin(), rays.end(),
      [](const auto & lhs, const auto & rhs) {
        return std::make_tuple(
          lhs.second.x(), lhs.second.y(), lhs.second.z(),
          lhs.first.x(), lhs.first.y(), lhs.first.z()) <
        std::make_tuple(
          rhs.second.x(), rhs.second.y(), rhs.second.z(),
          rhs.first.x(), rhs.first.y(), rhs.first.z());
      });
    const std::size_t maximum_rays = static_cast<std::size_t>(
      config_.max_rays_per_observation);
    const std::size_t ray_count = std::min(maximum_rays, rays.size());
    for (std::size_t ray_index = 0U; ray_index < ray_count; ++ray_index) {
      const std::size_t selected_index = std::min(
        rays.size() - 1U,
        ray_index * rays.size() / ray_count);
      applyRayMisses(
        rays[selected_index].first,
        rays[selected_index].second,
        local_endpoints,
        observation_id);
    }
  }

  stats_.last_update_runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - started).count();
  stats_.cumulative_update_runtime_ms += stats_.last_update_runtime_ms;
}

std::vector<TemporalStaticMapPoint> TemporalStaticMapAccumulator::points(
  const TemporalStaticMapSelection selection) const
{
  std::vector<std::pair<VoxelKey, TemporalStaticMapPoint>> ordered;
  ordered.reserve(voxels_.size());
  for (const auto & item : voxels_) {
    const auto & evidence = item.second;
    const bool selected =
      selection == TemporalStaticMapSelection::kAll ||
      (selection == TemporalStaticMapSelection::kConfirmed && evidence.confirmed) ||
      (selection == TemporalStaticMapSelection::kRejected && !evidence.confirmed);
    if (!selected || evidence.hit_observations <= 0) {
      continue;
    }
    TemporalStaticMapPoint point;
    point.position = evidence.position_sum /
      static_cast<double>(evidence.hit_observations);
    point.hit_observations = evidence.hit_observations;
    point.miss_observations = evidence.miss_observations;
    point.observation_span_s = std::max(
      0.0, evidence.last_stamp_s - evidence.first_stamp_s);
    point.viewpoint_distance_m = evidence.max_viewpoint_distance_m;
    point.hit_ratio = hitRatio(evidence);
    point.confirmed = evidence.confirmed;
    ordered.emplace_back(item.first, point);
  }
  std::sort(
    ordered.begin(), ordered.end(),
    [](const auto & lhs, const auto & rhs) {return lhs.first < rhs.first;});

  std::vector<TemporalStaticMapPoint> output;
  output.reserve(ordered.size());
  for (auto & item : ordered) {
    output.push_back(std::move(item.second));
  }
  return output;
}

const TemporalStaticMapStats & TemporalStaticMapAccumulator::stats() const
{
  return stats_;
}

const TemporalStaticMapConfig & TemporalStaticMapAccumulator::config() const
{
  return config_;
}

}  // namespace slam_3d_map_cleanup
}  // namespace gole
