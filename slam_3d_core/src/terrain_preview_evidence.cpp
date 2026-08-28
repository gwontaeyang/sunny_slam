// Copyright 2026 Gole Robotics

#include "slam_3d_core/terrain_preview_evidence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace gole
{
namespace slam_3d_core
{

namespace
{

constexpr long double kNanosecondsPerSecond = 1.0e9L;

bool finiteEvidence(const TerrainEvidencePoint & evidence)
{
  return evidence.position.allFinite() &&
         evidence.label != TerrainLabel::kUnknown &&
         evidence.confirmations > 0 &&
         evidence.observations >= evidence.confirmations &&
         std::isfinite(evidence.confidence) &&
         std::isfinite(evidence.geometric_roughness_score);
}

}  // namespace

const char * terrainPreviewStateName(const TerrainPreviewState state)
{
  switch (state) {
    case TerrainPreviewState::kTentative:
      return "tentative";
    case TerrainPreviewState::kGeometryConfirmed:
      return "geometry_confirmed";
    case TerrainPreviewState::kTraversalConfirmed:
      return "traversal_confirmed";
    case TerrainPreviewState::kUnsafeTraversalConflict:
      return "unsafe_traversal_conflict";
  }
  return "tentative";
}

const char * terrainTraversalResponseName(const TerrainTraversalResponse response)
{
  switch (response) {
    case TerrainTraversalResponse::kUnknown:
      return "unknown";
    case TerrainTraversalResponse::kStable:
      return "stable";
    case TerrainTraversalResponse::kShock:
      return "shock";
  }
  return "unknown";
}

float terrainPreviewValue(const TerrainLabel label, const TerrainPreviewState state)
{
  return static_cast<float>(
    10U * static_cast<std::uint8_t>(label) + static_cast<std::uint8_t>(state));
}

TerrainPreviewTracker::TerrainPreviewTracker(const TerrainPreviewConfig & config)
{
  configure(config);
}

void TerrainPreviewTracker::configure(const TerrainPreviewConfig & config)
{
  config_ = config;
  config_.voxel_size_m = std::max(1.0e-3, config_.voxel_size_m);
  config_.association_radius_m = std::max(0.0, config_.association_radius_m);
  config_.exit_radius_m = std::max(
    config_.association_radius_m,
    config_.exit_radius_m);
  config_.max_vertical_delta_m = std::max(0.0, config_.max_vertical_delta_m);
  config_.max_sample_gap_s = std::max(0.0, config_.max_sample_gap_s);
  config_.max_swept_segment_length_m = std::max(
    0.0,
    config_.max_swept_segment_length_m);
  config_.min_ahead_distance_m = std::max(0.0, config_.min_ahead_distance_m);
  config_.max_lateral_distance_m = std::max(0.0, config_.max_lateral_distance_m);
  config_.min_label_confidence = std::clamp(
    config_.min_label_confidence,
    std::nextafter(0.5, 1.0),
    1.0);
  config_.roughness_score_gate = std::max(1.0e-6, config_.roughness_score_gate);
  config_.ramp_shock_prior = std::max(0.0, config_.ramp_shock_prior);
  config_.positive_step_shock_prior = std::max(0.0, config_.positive_step_shock_prior);
  config_.min_passage_samples = std::max(1, config_.min_passage_samples);
  config_.min_passage_duration_s = std::max(0.0, config_.min_passage_duration_s);
  clear();
}

void TerrainPreviewTracker::clear()
{
  landmarks_.clear();
  active_passage_keys_.clear();
  previous_traversal_sample_.reset();
  traversal_sample_count_ = 0U;
  invalid_sample_count_ = 0U;
  continuity_break_count_ = 0U;
  cached_landmark_stats_ = TerrainPreviewStats{};
}

std::size_t TerrainPreviewTracker::VoxelKeyHash::operator()(const VoxelKey & key) const
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

TerrainPreviewTracker::VoxelKey TerrainPreviewTracker::voxelKey(
  const Eigen::Vector3d & point) const
{
  return VoxelKey{
    static_cast<std::int64_t>(std::floor(point.x() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.y() / config_.voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.z() / config_.voxel_size_m))};
}

void TerrainPreviewTracker::updateGeometry(
  const std::vector<TerrainEvidencePoint> & evidence,
  const std::int64_t observation_stamp_ns,
  const Eigen::Vector3d & viewpoint,
  const Eigen::Vector3d & view_forward)
{
  if (!config_.enabled || observation_stamp_ns <= 0 || !viewpoint.allFinite() ||
    !view_forward.allFinite())
  {
    return;
  }
  Eigen::Vector2d forward_xy(view_forward.x(), view_forward.y());
  if (forward_xy.norm() > 1.0e-6) {
    forward_xy.normalize();
  } else {
    forward_xy.setZero();
  }

  std::unordered_set<VoxelKey, VoxelKeyHash> observed_keys;
  observed_keys.reserve(evidence.size());
  for (const auto & item : evidence) {
    if (!finiteEvidence(item)) {
      continue;
    }
    // Flat floor voxels are not landmarks. Keep floor only when the radial
    // profile contains a high-frequency height residual worth previewing.
    if (item.label == TerrainLabel::kFloor &&
      item.geometric_roughness_score < config_.roughness_score_gate)
    {
      continue;
    }
    const VoxelKey key = voxelKey(item.position);
    observed_keys.insert(key);
    const auto insertion = landmarks_.try_emplace(key);
    Landmark & landmark = insertion.first->second;
    const bool label_changed = !insertion.second && landmark.geometry.label != item.label;
    if (insertion.second || label_changed) {
      landmark.first_observed_stamp_ns = observation_stamp_ns;
      landmark.first_preview_distance_m = std::hypot(
        item.position.x() - viewpoint.x(),
        item.position.y() - viewpoint.y());
      const Eigen::Vector2d relative_xy(
        item.position.x() - viewpoint.x(),
        item.position.y() - viewpoint.y());
      landmark.first_preview_ahead_distance_m = relative_xy.dot(forward_xy);
      const double lateral_distance_m = forward_xy.squaredNorm() > 0.0 ?
        (relative_xy - landmark.first_preview_ahead_distance_m * forward_xy).norm() :
        std::numeric_limits<double>::infinity();
      landmark.first_seen_ahead = forward_xy.squaredNorm() > 0.0 &&
        landmark.first_preview_ahead_distance_m >= config_.min_ahead_distance_m &&
        (config_.max_lateral_distance_m <= 0.0 ||
        lateral_distance_m <= config_.max_lateral_distance_m);
    }
    landmark.geometry = item;
    if (label_changed) {
      // A directional step/drop label can flip when viewed from the opposite
      // side. Do not carry robot-response semantics across that identity
      // change; a future direction-invariant ledge primitive will replace it.
      resetTraversalEvidence(landmark);
      active_passage_keys_.erase(key);
    }
  }
  for (auto iterator = landmarks_.begin(); iterator != landmarks_.end(); ) {
    if (observed_keys.find(iterator->first) == observed_keys.end()) {
      active_passage_keys_.erase(iterator->first);
      iterator = landmarks_.erase(iterator);
    } else {
      ++iterator;
    }
  }
  rebuildStatsCache();
}

TerrainTraversalUpdate TerrainPreviewTracker::addTraversalSample(
  const TerrainTraversalSample & sample)
{
  TerrainTraversalUpdate update;
  if (!config_.enabled || sample.stamp_ns <= 0 || !sample.position.allFinite() ||
    !std::isfinite(sample.roughness_score) || sample.roughness_score < 0.0)
  {
    ++invalid_sample_count_;
    return update;
  }
  if (previous_traversal_sample_.has_value() &&
    sample.stamp_ns <= previous_traversal_sample_->stamp_ns)
  {
    ++invalid_sample_count_;
    return update;
  }

  if (sample.continuity_break_before) {
    for (const VoxelKey & key : active_passage_keys_) {
      const auto landmark = landmarks_.find(key);
      if (landmark != landmarks_.end()) {
        landmark->second.passage = PassageState{};
      }
    }
    active_passage_keys_.clear();
    previous_traversal_sample_.reset();
    ++continuity_break_count_;
  }

  const std::optional<TerrainTraversalSample> previous_sample = previous_traversal_sample_;
  previous_traversal_sample_ = sample;
  ++traversal_sample_count_;
  update.accepted = true;
  const TerrainTraversalSample * previous_sample_ptr = previous_sample.has_value() ?
    &(*previous_sample) : nullptr;
  auto candidate_keys = nearbyLandmarkKeys(sample, previous_sample_ptr);
  candidate_keys.insert(active_passage_keys_.begin(), active_passage_keys_.end());
  for (const VoxelKey & key : candidate_keys) {
    const auto landmark = landmarks_.find(key);
    if (landmark == landmarks_.end()) {
      active_passage_keys_.erase(key);
      continue;
    }
    update.evidence_changed = updateLandmarkPassage(
      landmark->second, sample, previous_sample_ptr, update.finalized_passages) ||
      update.evidence_changed;
    if (landmark->second.passage.active) {
      active_passage_keys_.insert(key);
    } else {
      active_passage_keys_.erase(key);
    }
  }
  if (update.evidence_changed) {
    rebuildStatsCache();
  }
  return update;
}

bool TerrainPreviewTracker::segmentIsUsable(
  const TerrainTraversalSample & sample,
  const TerrainTraversalSample * previous_sample) const
{
  if (previous_sample == nullptr) {
    return false;
  }
  const double sample_gap_s = static_cast<double>(
    static_cast<long double>(sample.stamp_ns - previous_sample->stamp_ns) /
    kNanosecondsPerSecond);
  const double segment_length_m = (sample.position - previous_sample->position).norm();
  return sample_gap_s >= 0.0 &&
         (config_.max_sample_gap_s <= 0.0 || sample_gap_s <= config_.max_sample_gap_s) &&
         (config_.max_swept_segment_length_m <= 0.0 ||
         segment_length_m <= config_.max_swept_segment_length_m);
}

std::unordered_set<TerrainPreviewTracker::VoxelKey, TerrainPreviewTracker::VoxelKeyHash>
TerrainPreviewTracker::nearbyLandmarkKeys(
  const TerrainTraversalSample & sample,
  const TerrainTraversalSample * previous_sample) const
{
  const bool use_segment = segmentIsUsable(sample, previous_sample);
  const Eigen::Vector3d start = use_segment ? previous_sample->position : sample.position;
  const Eigen::Vector3d minimum = start.cwiseMin(sample.position) - Eigen::Vector3d(
    config_.association_radius_m,
    config_.association_radius_m,
    config_.max_vertical_delta_m);
  const Eigen::Vector3d maximum = start.cwiseMax(sample.position) + Eigen::Vector3d(
    config_.association_radius_m,
    config_.association_radius_m,
    config_.max_vertical_delta_m);
  const VoxelKey min_key = voxelKey(minimum);
  const VoxelKey max_key = voxelKey(maximum);

  std::unordered_set<VoxelKey, VoxelKeyHash> keys;
  for (std::int64_t x = min_key.x; x <= max_key.x; ++x) {
    for (std::int64_t y = min_key.y; y <= max_key.y; ++y) {
      for (std::int64_t z = min_key.z; z <= max_key.z; ++z) {
        const VoxelKey key{x, y, z};
        if (landmarks_.find(key) != landmarks_.end()) {
          keys.insert(key);
        }
      }
    }
  }
  return keys;
}

bool TerrainPreviewTracker::isUnsafeForTraversal(const TerrainLabel label) const
{
  return label == TerrainLabel::kObservedDrop || label == TerrainLabel::kStaircase;
}

double TerrainPreviewTracker::predictedShockScore(const Landmark & landmark) const
{
  double label_prior = 0.0;
  if (landmark.geometry.label == TerrainLabel::kRamp) {
    label_prior = config_.ramp_shock_prior;
  } else if (
    landmark.geometry.label == TerrainLabel::kPositiveStep ||
    landmark.geometry.label == TerrainLabel::kStaircase)
  {
    label_prior = config_.positive_step_shock_prior;
  }
  return std::max(label_prior, landmark.geometry.geometric_roughness_score);
}

TerrainPreviewState TerrainPreviewTracker::previewState(const Landmark & landmark) const
{
  if (landmark.unsafe_traversal_conflicts > 0) {
    return TerrainPreviewState::kUnsafeTraversalConflict;
  }
  if (landmark.traversal_confirmations > 0) {
    return TerrainPreviewState::kTraversalConfirmed;
  }
  if (landmark.geometry.confirmed) {
    return TerrainPreviewState::kGeometryConfirmed;
  }
  return TerrainPreviewState::kTentative;
}

void TerrainPreviewTracker::resetTraversalEvidence(Landmark & landmark) const
{
  landmark.first_traversed_stamp_ns = 0;
  landmark.first_preview_lead_time_s = 0.0;
  landmark.traversal_confirmations = 0;
  landmark.shock_passages = 0;
  landmark.stable_passages = 0;
  landmark.unsafe_traversal_conflicts = 0;
  landmark.passage = PassageState{};
}

bool TerrainPreviewTracker::updateLandmarkPassage(
  Landmark & landmark,
  const TerrainTraversalSample & sample,
  const TerrainTraversalSample * previous_sample,
  std::size_t & finalized_passages) const
{
  Eigen::Vector3d closest_position = sample.position;
  if (segmentIsUsable(sample, previous_sample)) {
    const Eigen::Vector2d segment_xy =
      sample.position.head<2>() - previous_sample->position.head<2>();
    if (segment_xy.squaredNorm() > 1.0e-12) {
      const Eigen::Vector2d candidate_offset =
        landmark.geometry.position.head<2>() - previous_sample->position.head<2>();
      const double ratio = std::clamp(
        candidate_offset.dot(segment_xy) / segment_xy.squaredNorm(),
        0.0,
        1.0);
      closest_position = previous_sample->position +
        ratio * (sample.position - previous_sample->position);
    }
  }
  const double horizontal_distance_m = (
    closest_position.head<2>() - landmark.geometry.position.head<2>()).norm();
  const double vertical_distance_m = std::abs(
    closest_position.z() - landmark.geometry.position.z());
  const double endpoint_horizontal_distance_m = (
    sample.position.head<2>() - landmark.geometry.position.head<2>()).norm();
  const double endpoint_vertical_distance_m = std::abs(
    sample.position.z() - landmark.geometry.position.z());
  const bool response_at_candidate =
    endpoint_horizontal_distance_m <= config_.association_radius_m &&
    endpoint_vertical_distance_m <= config_.max_vertical_delta_m;

  bool changed = false;
  if (landmark.passage.active && config_.max_sample_gap_s > 0.0) {
    const double sample_gap_s = static_cast<double>(
      static_cast<long double>(sample.stamp_ns - landmark.passage.last_stamp_ns) /
      kNanosecondsPerSecond);
    if (sample_gap_s > config_.max_sample_gap_s) {
      const bool finalized = finalizePassage(landmark);
      changed = finalized || changed;
      finalized_passages += finalized ? 1U : 0U;
    }
  }

  const bool near = horizontal_distance_m <= config_.association_radius_m &&
    vertical_distance_m <= config_.max_vertical_delta_m;
  if (near) {
    if (!landmark.passage.active) {
      landmark.passage = PassageState{};
      landmark.passage.active = true;
      const bool swept_segment = segmentIsUsable(sample, previous_sample);
      landmark.passage.enter_stamp_ns = swept_segment ?
        previous_sample->stamp_ns : sample.stamp_ns;
      landmark.passage.sample_count = swept_segment ? 2 : 1;
    } else {
      ++landmark.passage.sample_count;
    }
    landmark.passage.last_stamp_ns = sample.stamp_ns;
    landmark.passage.saw_stable = landmark.passage.saw_stable ||
      sample.response == TerrainTraversalResponse::kStable;
    landmark.passage.saw_shock = landmark.passage.saw_shock ||
      (sample.response == TerrainTraversalResponse::kShock && response_at_candidate);
    if (sample.response != TerrainTraversalResponse::kShock || response_at_candidate) {
      landmark.passage.peak_roughness_score = std::max(
        landmark.passage.peak_roughness_score,
        sample.roughness_score);
    }
    return changed;
  }

  const bool outside_exit = horizontal_distance_m >= config_.exit_radius_m ||
    vertical_distance_m > config_.max_vertical_delta_m;
  if (landmark.passage.active && outside_exit) {
    const bool finalized = finalizePassage(landmark);
    changed = finalized || changed;
    finalized_passages += finalized ? 1U : 0U;
  }
  return changed;
}

bool TerrainPreviewTracker::finalizePassage(Landmark & landmark) const
{
  const PassageState passage = landmark.passage;
  landmark.passage = PassageState{};
  if (!passage.active || passage.sample_count < config_.min_passage_samples ||
    (!passage.saw_stable && !passage.saw_shock))
  {
    return false;
  }
  const double duration_s = static_cast<double>(
    static_cast<long double>(passage.last_stamp_ns - passage.enter_stamp_ns) /
    kNanosecondsPerSecond);
  if (duration_s < config_.min_passage_duration_s ||
    landmark.geometry.confidence < config_.min_label_confidence)
  {
    return false;
  }

  if (isUnsafeForTraversal(landmark.geometry.label)) {
    ++landmark.unsafe_traversal_conflicts;
  } else {
    ++landmark.traversal_confirmations;
    if (passage.saw_shock) {
      ++landmark.shock_passages;
    } else {
      ++landmark.stable_passages;
    }
  }
  if (landmark.first_traversed_stamp_ns == 0) {
    landmark.first_traversed_stamp_ns = passage.enter_stamp_ns;
    if (landmark.first_seen_ahead && landmark.first_observed_stamp_ns > 0 &&
      passage.enter_stamp_ns >= landmark.first_observed_stamp_ns)
    {
      landmark.first_preview_lead_time_s = static_cast<double>(
        static_cast<long double>(
          passage.enter_stamp_ns - landmark.first_observed_stamp_ns) /
        kNanosecondsPerSecond);
    }
  }
  return true;
}

std::vector<TerrainPreviewPoint> TerrainPreviewTracker::previewEvidence() const
{
  std::vector<TerrainPreviewPoint> output;
  if (!config_.enabled) {
    return output;
  }
  output.reserve(landmarks_.size());
  for (const auto & entry : landmarks_) {
    const Landmark & landmark = entry.second;
    const int response_passages = landmark.shock_passages + landmark.stable_passages;
    output.push_back(
      TerrainPreviewPoint{
          landmark.geometry.position,
          landmark.geometry.label,
          previewState(landmark),
          landmark.geometry.confirmations,
          landmark.geometry.observations,
          landmark.geometry.confidence,
          landmark.geometry.geometric_roughness_score,
          predictedShockScore(landmark),
          landmark.traversal_confirmations,
          landmark.shock_passages,
          landmark.stable_passages,
          landmark.unsafe_traversal_conflicts,
          response_passages > 0 ?
          static_cast<double>(landmark.shock_passages) /
          static_cast<double>(response_passages) : 0.0,
          landmark.first_observed_stamp_ns,
          landmark.first_traversed_stamp_ns,
          landmark.first_preview_distance_m,
          landmark.first_preview_ahead_distance_m,
          landmark.first_seen_ahead,
          landmark.first_preview_lead_time_s});
  }
  std::sort(
    output.begin(),
    output.end(),
    [](const TerrainPreviewPoint & lhs, const TerrainPreviewPoint & rhs) {
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

void TerrainPreviewTracker::rebuildStatsCache()
{
  TerrainPreviewStats output;
  output.landmarks = landmarks_.size();
  double lead_time_sum_s = 0.0;
  std::size_t lead_time_count = 0U;
  for (const auto & entry : landmarks_) {
    const Landmark & landmark = entry.second;
    switch (previewState(landmark)) {
      case TerrainPreviewState::kTentative:
        ++output.tentative_landmarks;
        break;
      case TerrainPreviewState::kGeometryConfirmed:
        ++output.geometry_confirmed_landmarks;
        break;
      case TerrainPreviewState::kTraversalConfirmed:
        ++output.traversal_confirmed_landmarks;
        break;
      case TerrainPreviewState::kUnsafeTraversalConflict:
        ++output.unsafe_traversal_conflict_landmarks;
        break;
    }
    if (predictedShockScore(landmark) >= config_.roughness_score_gate) {
      ++output.roughness_preview_landmarks;
    }
    if (landmark.first_seen_ahead) {
      ++output.ahead_preview_landmarks;
    }
    output.shock_passages += static_cast<std::size_t>(landmark.shock_passages);
    output.stable_passages += static_cast<std::size_t>(landmark.stable_passages);
    output.unsafe_traversal_conflicts +=
      static_cast<std::size_t>(landmark.unsafe_traversal_conflicts);
    output.finalized_passages += static_cast<std::size_t>(
      landmark.traversal_confirmations + landmark.unsafe_traversal_conflicts);
    if (landmark.first_traversed_stamp_ns > 0 && landmark.first_seen_ahead) {
      lead_time_sum_s += landmark.first_preview_lead_time_s;
      output.max_preview_lead_time_s = std::max(
        output.max_preview_lead_time_s,
        landmark.first_preview_lead_time_s);
      ++lead_time_count;
    }
  }
  output.previewed_before_traversal_landmarks = lead_time_count;
  output.mean_preview_lead_time_s = lead_time_count > 0U ?
    lead_time_sum_s / static_cast<double>(lead_time_count) : 0.0;
  cached_landmark_stats_ = output;
}

TerrainPreviewStats TerrainPreviewTracker::stats() const
{
  TerrainPreviewStats output = cached_landmark_stats_;
  output.active_passages = active_passage_keys_.size();
  output.traversal_samples = traversal_sample_count_;
  output.invalid_samples = invalid_sample_count_;
  output.continuity_breaks = continuity_break_count_;
  return output;
}

}  // namespace slam_3d_core
}  // namespace gole
