#include "slam_2d_core/roughness_place_recognition.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace slam_2d_core
{
namespace
{

constexpr double kDistanceEpsilonM = 1.0e-6;
constexpr double kValueEpsilon = 1.0e-12;
constexpr double kTwoPiRad = 6.28318530717958647692;

bool finitePose(const RoughnessPlaceObservation & observation)
{
  return
    observation.stamp_ns > 0 &&
    observation.scan_state_id >= 0 &&
    observation.scan_unique_id >= 0 &&
    std::isfinite(observation.odom_x_m) &&
    std::isfinite(observation.odom_y_m) &&
    std::isfinite(observation.odom_heading_rad) &&
    std::isfinite(observation.corrected_x_m) &&
    std::isfinite(observation.corrected_y_m) &&
    std::isfinite(observation.corrected_heading_rad);
}

bool hasFiniteFeature(const RoughnessPlaceObservation & observation)
{
  return std::any_of(
    observation.features.begin(), observation.features.end(),
    [](const double value) {return std::isfinite(value);});
}

double normalizedFeature(const double value, const double scale)
{
  if (
    !std::isfinite(value) || value < 0.0 ||
    !std::isfinite(scale) || scale <= 0.0)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // Compress isolated shocks while preserving ordering and relative shape.
  return std::log1p(value / scale);
}

struct SpatialSample
{
  double distance_m{0.0};
  RoughnessPlaceFeatures features{};
};

std::vector<SpatialSample> spatialSamples(
  const std::vector<RoughnessPlaceObservation> & observations)
{
  std::vector<SpatialSample> samples;
  if (observations.empty()) {
    return samples;
  }
  samples.reserve(observations.size());
  samples.push_back({0.0, observations.front().features});
  for (std::size_t index = 1U; index < observations.size(); ++index) {
    const auto & previous = observations[index - 1U];
    const auto & current = observations[index];
    const double step_m = std::hypot(
      current.odom_x_m - previous.odom_x_m,
      current.odom_y_m - previous.odom_y_m);
    if (!std::isfinite(step_m)) {
      return {};
    }
    const double distance_m = samples.back().distance_m + step_m;
    if (step_m <= kDistanceEpsilonM) {
      for (std::size_t channel = 0U;
        channel < kRoughnessPlaceFeatureCount; ++channel)
      {
        const double value = current.features[channel];
        if (std::isfinite(value)) {
          samples.back().features[channel] = value;
        }
      }
      continue;
    }
    samples.push_back({distance_m, current.features});
  }
  return samples;
}

double interpolateChannel(
  const std::vector<SpatialSample> & samples,
  const std::size_t channel,
  const double target_distance_m)
{
  std::size_t lower = samples.size();
  std::size_t upper = samples.size();
  for (std::size_t index = 0U; index < samples.size(); ++index) {
    if (!std::isfinite(samples[index].features[channel])) {
      continue;
    }
    if (samples[index].distance_m <= target_distance_m) {
      lower = index;
    }
    if (samples[index].distance_m >= target_distance_m) {
      upper = index;
      break;
    }
  }
  if (lower == samples.size() && upper == samples.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (lower == samples.size()) {
    return samples[upper].features[channel];
  }
  if (upper == samples.size()) {
    return samples[lower].features[channel];
  }
  if (lower == upper) {
    return samples[lower].features[channel];
  }
  const double span_m =
    samples[upper].distance_m - samples[lower].distance_m;
  if (span_m <= kDistanceEpsilonM) {
    return samples[lower].features[channel];
  }
  const double ratio = std::clamp(
    (target_distance_m - samples[lower].distance_m) / span_m,
    0.0,
    1.0);
  return
    samples[lower].features[channel] * (1.0 - ratio) +
    samples[upper].features[channel] * ratio;
}

struct DirectionalMatch
{
  bool valid{false};
  double score{std::numeric_limits<double>::quiet_NaN()};
  double shape_score{std::numeric_limits<double>::quiet_NaN()};
  double amplitude_score{std::numeric_limits<double>::quiet_NaN()};
  double length_score{std::numeric_limits<double>::quiet_NaN()};
  std::size_t compared_channel_count{0U};
};

DirectionalMatch directionalMatch(
  const RoughnessPlaceDescriptor & query,
  const RoughnessPlaceDescriptor & reference,
  const bool reverse_reference)
{
  DirectionalMatch result;
  if (!query.valid() || !reference.valid() || query.bin_count != reference.bin_count) {
    return result;
  }

  double shape_sum = 0.0;
  double amplitude_sum = 0.0;
  for (std::size_t channel = 0U;
    channel < kRoughnessPlaceFeatureCount; ++channel)
  {
    if (!query.channel_valid[channel] || !reference.channel_valid[channel]) {
      continue;
    }
    double query_mean = 0.0;
    double reference_mean = 0.0;
    for (std::size_t bin = 0U; bin < query.bin_count; ++bin) {
      const std::size_t reference_bin = reverse_reference ?
        reference.bin_count - 1U - bin : bin;
      query_mean += query.value(bin, channel);
      reference_mean += reference.value(reference_bin, channel);
    }
    query_mean /= static_cast<double>(query.bin_count);
    reference_mean /= static_cast<double>(reference.bin_count);

    double numerator = 0.0;
    double query_energy = 0.0;
    double reference_energy = 0.0;
    double channel_amplitude = 0.0;
    for (std::size_t bin = 0U; bin < query.bin_count; ++bin) {
      const std::size_t reference_bin = reverse_reference ?
        reference.bin_count - 1U - bin : bin;
      const double query_value = query.value(bin, channel);
      const double reference_value = reference.value(reference_bin, channel);
      const double query_centered = query_value - query_mean;
      const double reference_centered = reference_value - reference_mean;
      numerator += query_centered * reference_centered;
      query_energy += query_centered * query_centered;
      reference_energy += reference_centered * reference_centered;
      const double relative_difference = std::fabs(query_value - reference_value) /
        (std::fabs(query_value) + std::fabs(reference_value) + 0.25);
      channel_amplitude += 1.0 - std::clamp(relative_difference, 0.0, 1.0);
    }
    channel_amplitude /= static_cast<double>(query.bin_count);

    double channel_shape = channel_amplitude;
    if (query_energy > kValueEpsilon && reference_energy > kValueEpsilon) {
      const double correlation = std::clamp(
        numerator / std::sqrt(query_energy * reference_energy),
        -1.0,
        1.0);
      channel_shape = 0.5 * (correlation + 1.0);
    } else if (
      (query_energy > kValueEpsilon) !=
      (reference_energy > kValueEpsilon))
    {
      channel_shape = 0.0;
    }
    shape_sum += channel_shape;
    amplitude_sum += channel_amplitude;
    ++result.compared_channel_count;
  }
  if (result.compared_channel_count == 0U) {
    return result;
  }

  result.shape_score =
    shape_sum / static_cast<double>(result.compared_channel_count);
  result.amplitude_score =
    amplitude_sum / static_cast<double>(result.compared_channel_count);
  result.length_score = std::exp(
    -std::fabs(std::log(query.path_length_m / reference.path_length_m)));
  result.score =
    (result.shape_score + result.amplitude_score + result.length_score) / 3.0;
  result.valid =
    std::isfinite(result.score) && result.score >= 0.0 && result.score <= 1.0;
  return result;
}

bool finitePlacePose(const RoughnessPlacePose2D & pose) noexcept
{
  return
    std::isfinite(pose.x_m) && std::isfinite(pose.y_m) &&
    std::isfinite(pose.yaw_rad);
}

double normalizeAngle(const double angle_rad) noexcept
{
  return std::remainder(angle_rad, kTwoPiRad);
}

RoughnessPlacePose2D relativeMotion(
  const RoughnessPlacePose2D & pre,
  const RoughnessPlacePose2D & post) noexcept
{
  const double delta_x_m = post.x_m - pre.x_m;
  const double delta_y_m = post.y_m - pre.y_m;
  const double cosine = std::cos(pre.yaw_rad);
  const double sine = std::sin(pre.yaw_rad);
  return {
    cosine * delta_x_m + sine * delta_y_m,
    -sine * delta_x_m + cosine * delta_y_m,
    normalizeAngle(post.yaw_rad - pre.yaw_rad),
  };
}

}  // namespace

RoughnessPlaceRelativeMotionConsistency roughnessPlaceRelativeMotionConsistency(
  const RoughnessPlacePose2D & first_pre,
  const RoughnessPlacePose2D & first_post,
  const RoughnessPlacePose2D & second_pre,
  const RoughnessPlacePose2D & second_post) noexcept
{
  RoughnessPlaceRelativeMotionConsistency result;
  if (
    !finitePlacePose(first_pre) || !finitePlacePose(first_post) ||
    !finitePlacePose(second_pre) || !finitePlacePose(second_post))
  {
    return result;
  }
  const RoughnessPlacePose2D first_motion = relativeMotion(
    first_pre, first_post);
  const RoughnessPlacePose2D second_motion = relativeMotion(
    second_pre, second_post);
  result.translation_error_m = std::hypot(
    first_motion.x_m - second_motion.x_m,
    first_motion.y_m - second_motion.y_m);
  result.yaw_error_rad = std::fabs(
    normalizeAngle(first_motion.yaw_rad - second_motion.yaw_rad));
  result.available =
    std::isfinite(result.translation_error_m) &&
    std::isfinite(result.yaw_error_rad);
  return result;
}

RoughnessPlaceFeatures normalizedRoughnessPlaceFeatures(
  const double roughness_score,
  const RoughnessSignal & signal,
  const RoughnessThresholds & thresholds)
{
  RoughnessPlaceFeatures features;
  features.fill(std::numeric_limits<double>::quiet_NaN());
  features[0] = normalizedFeature(roughness_score, 1.0);
  features[1] = normalizedFeature(signal.tilt_deg, thresholds.tilt_deg);
  features[2] = normalizedFeature(
    signal.gyro_xy_rad_s, thresholds.gyro_xy_rad_s);
  features[3] = normalizedFeature(
    signal.accel_norm_deviation_m_s2,
    thresholds.accel_norm_deviation_m_s2);
  features[4] = normalizedFeature(
    signal.tilt_rate_deg_s, thresholds.tilt_rate_deg_s);
  features[5] = normalizedFeature(
    signal.accel_norm_rate_m_s3,
    thresholds.accel_norm_rate_m_s3);
  return features;
}

bool RoughnessPlaceDescriptor::valid() const noexcept
{
  return
    bin_count >= 2U &&
    source_observation_count >= 3U &&
    valid_channel_count > 0U &&
    std::isfinite(path_length_m) && path_length_m > kDistanceEpsilonM &&
    values.size() == bin_count * kRoughnessPlaceFeatureCount;
}

double RoughnessPlaceDescriptor::value(
  const std::size_t bin,
  const std::size_t channel) const noexcept
{
  if (
    bin >= bin_count || channel >= kRoughnessPlaceFeatureCount ||
    values.size() != bin_count * kRoughnessPlaceFeatureCount)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return values[bin * kRoughnessPlaceFeatureCount + channel];
}

RoughnessPlaceDescriptor buildRoughnessPlaceDescriptor(
  const std::vector<RoughnessPlaceObservation> & observations,
  const std::size_t bin_count)
{
  RoughnessPlaceDescriptor descriptor;
  descriptor.bin_count = bin_count;
  descriptor.source_observation_count = observations.size();
  if (bin_count < 2U || observations.size() < 3U) {
    return descriptor;
  }
  if (!std::all_of(observations.begin(), observations.end(), finitePose)) {
    return descriptor;
  }

  const std::vector<SpatialSample> samples = spatialSamples(observations);
  if (samples.size() < 2U) {
    return descriptor;
  }
  descriptor.path_length_m = samples.back().distance_m;
  if (
    !std::isfinite(descriptor.path_length_m) ||
    descriptor.path_length_m <= kDistanceEpsilonM)
  {
    return descriptor;
  }

  for (std::size_t channel = 0U;
    channel < kRoughnessPlaceFeatureCount; ++channel)
  {
    const std::size_t finite_count = static_cast<std::size_t>(std::count_if(
        samples.begin(), samples.end(),
        [channel](const SpatialSample & sample) {
          return std::isfinite(sample.features[channel]);
        }));
    descriptor.channel_valid[channel] = finite_count >= 2U;
    if (descriptor.channel_valid[channel]) {
      ++descriptor.valid_channel_count;
    }
  }
  if (descriptor.valid_channel_count == 0U) {
    return descriptor;
  }

  descriptor.values.assign(
    bin_count * kRoughnessPlaceFeatureCount,
    std::numeric_limits<double>::quiet_NaN());
  for (std::size_t bin = 0U; bin < bin_count; ++bin) {
    const double target_distance_m = descriptor.path_length_m *
      static_cast<double>(bin) / static_cast<double>(bin_count - 1U);
    for (std::size_t channel = 0U;
      channel < kRoughnessPlaceFeatureCount; ++channel)
    {
      if (!descriptor.channel_valid[channel]) {
        continue;
      }
      descriptor.values[bin * kRoughnessPlaceFeatureCount + channel] =
        interpolateChannel(samples, channel, target_distance_m);
    }
  }
  return descriptor;
}

const char * roughnessPlaceTrackerActionName(
  const RoughnessPlaceTrackerAction action) noexcept
{
  switch (action) {
    case RoughnessPlaceTrackerAction::kNone:
      return "none";
    case RoughnessPlaceTrackerAction::kStableAnchorUpdated:
      return "stable_anchor_updated";
    case RoughnessPlaceTrackerAction::kStarted:
      return "started";
    case RoughnessPlaceTrackerAction::kExtended:
      return "extended";
    case RoughnessPlaceTrackerAction::kFinalized:
      return "finalized";
    case RoughnessPlaceTrackerAction::kIgnoredWithoutPreAnchor:
      return "ignored_without_pre_anchor";
    case RoughnessPlaceTrackerAction::kIgnoredInvalid:
      return "ignored_invalid";
    case RoughnessPlaceTrackerAction::kTimestampReset:
      return "timestamp_reset";
  }
  return "unknown";
}

RoughnessPlaceEventTracker::RoughnessPlaceEventTracker(
  const std::size_t descriptor_bin_count,
  const std::size_t maximum_observations_per_event)
{
  configure(descriptor_bin_count, maximum_observations_per_event);
}

void RoughnessPlaceEventTracker::configure(
  const std::size_t descriptor_bin_count,
  const std::size_t maximum_observations_per_event)
{
  descriptor_bin_count_ = std::max(descriptor_bin_count, std::size_t{2});
  maximum_observations_per_event_ = std::max(
    maximum_observations_per_event, std::size_t{3});
  reset();
}

void RoughnessPlaceEventTracker::reset()
{
  next_event_id_ = 1U;
  last_stamp_ns_ = 0;
  last_stable_anchor_.reset();
  active_observations_.clear();
  active_rough_observation_count_ = 0U;
}

void RoughnessPlaceEventTracker::appendActiveObservation(
  const RoughnessPlaceObservation & observation)
{
  if (active_observations_.size() < maximum_observations_per_event_) {
    active_observations_.push_back(observation);
    return;
  }
  // Preserve both boundaries and replace the latest interior sample. This is
  // only a memory bound; event matching remains distance-normalized.
  if (active_observations_.size() > 1U) {
    active_observations_.back() = observation;
  }
}

RoughnessPlaceTrackerUpdate RoughnessPlaceEventTracker::update(
  const RoughnessPlaceObservation & observation)
{
  RoughnessPlaceTrackerUpdate update;
  if (last_stamp_ns_ != 0 && observation.stamp_ns < last_stamp_ns_) {
    reset();
    update.action = RoughnessPlaceTrackerAction::kTimestampReset;
    return update;
  }
  if (observation.stamp_ns == last_stamp_ns_ && last_stamp_ns_ != 0) {
    update.action = RoughnessPlaceTrackerAction::kIgnoredInvalid;
    return update;
  }
  last_stamp_ns_ = observation.stamp_ns;
  if (
    !observation.quality_valid || !finitePose(observation) ||
    !hasFiniteFeature(observation))
  {
    update.action = RoughnessPlaceTrackerAction::kIgnoredInvalid;
    return update;
  }

  if (observation.rough) {
    if (active_observations_.empty()) {
      if (!last_stable_anchor_) {
        update.action = RoughnessPlaceTrackerAction::kIgnoredWithoutPreAnchor;
        return update;
      }
      active_observations_.push_back(*last_stable_anchor_);
      appendActiveObservation(observation);
      active_rough_observation_count_ = 1U;
      update.action = RoughnessPlaceTrackerAction::kStarted;
      return update;
    }
    appendActiveObservation(observation);
    ++active_rough_observation_count_;
    update.action = RoughnessPlaceTrackerAction::kExtended;
    return update;
  }

  if (active_observations_.empty()) {
    last_stable_anchor_ = observation;
    update.action = RoughnessPlaceTrackerAction::kStableAnchorUpdated;
    return update;
  }

  appendActiveObservation(observation);
  RoughnessPlaceEvent event;
  event.event_id = next_event_id_++;
  event.start_stamp_ns = active_observations_.front().stamp_ns;
  event.end_stamp_ns = observation.stamp_ns;
  event.pre_anchor = active_observations_.front();
  event.post_anchor = observation;
  event.pre_anchor_state_id = event.pre_anchor.scan_state_id;
  event.post_anchor_state_id = event.post_anchor.scan_state_id;
  event.rough_observation_count = active_rough_observation_count_;
  event.descriptor = buildRoughnessPlaceDescriptor(
    active_observations_, descriptor_bin_count_);
  active_observations_.clear();
  active_rough_observation_count_ = 0U;
  last_stable_anchor_ = observation;
  update.action = RoughnessPlaceTrackerAction::kFinalized;
  update.finalized_event = std::move(event);
  return update;
}

bool RoughnessPlaceEventTracker::active() const noexcept
{
  return !active_observations_.empty();
}

const char * roughnessPlaceMatchDirectionName(
  const RoughnessPlaceMatchDirection direction) noexcept
{
  switch (direction) {
    case RoughnessPlaceMatchDirection::kForward:
      return "forward";
    case RoughnessPlaceMatchDirection::kReverse:
      return "reverse";
  }
  return "unknown";
}

RoughnessPlaceAnchorPairing roughnessPlaceAnchorPairing(
  const RoughnessPlaceEvent & reference,
  const RoughnessPlaceEvent & query,
  const RoughnessPlaceMatchDirection direction) noexcept
{
  RoughnessPlaceAnchorPairing pairing;
  pairing.reference_pre_state_id = reference.pre_anchor_state_id;
  pairing.reference_post_state_id = reference.post_anchor_state_id;
  if (direction == RoughnessPlaceMatchDirection::kReverse) {
    pairing.query_for_reference_pre_state_id = query.post_anchor_state_id;
    pairing.query_for_reference_post_state_id = query.pre_anchor_state_id;
  } else {
    pairing.query_for_reference_pre_state_id = query.pre_anchor_state_id;
    pairing.query_for_reference_post_state_id = query.post_anchor_state_id;
  }
  pairing.valid =
    pairing.reference_pre_state_id >= 0 &&
    pairing.query_for_reference_pre_state_id >= 0 &&
    pairing.reference_post_state_id >= 0 &&
    pairing.query_for_reference_post_state_id >= 0;
  return pairing;
}

bool RoughnessPlaceConstraintPassage::valid() const noexcept
{
  return
    reference_event_id > 0U && query_event_id > 0U &&
    reference_first_state_id >= 0 &&
    reference_last_state_id >= reference_first_state_id &&
    query_first_state_id >= 0 && query_last_state_id >= query_first_state_id;
}

RoughnessPlaceConstraintPassage roughnessPlaceConstraintPassage(
  const RoughnessPlaceEvent & reference,
  const RoughnessPlaceEvent & query) noexcept
{
  RoughnessPlaceConstraintPassage passage;
  passage.reference_event_id = reference.event_id;
  passage.query_event_id = query.event_id;
  passage.reference_first_state_id = std::min(
    reference.pre_anchor_state_id, reference.post_anchor_state_id);
  passage.reference_last_state_id = std::max(
    reference.pre_anchor_state_id, reference.post_anchor_state_id);
  passage.query_first_state_id = std::min(
    query.pre_anchor_state_id, query.post_anchor_state_id);
  passage.query_last_state_id = std::max(
    query.pre_anchor_state_id, query.post_anchor_state_id);
  return passage;
}

RoughnessPlacePassageCorrelation correlateRoughnessPlaceConstraintPassages(
  const RoughnessPlaceConstraintPassage & left,
  const RoughnessPlaceConstraintPassage & right) noexcept
{
  RoughnessPlacePassageCorrelation result;
  if (!left.valid() || !right.valid()) {
    return result;
  }
  const auto intervening_state_count = [](
    const std::int32_t left_first,
    const std::int32_t left_last,
    const std::int32_t right_first,
    const std::int32_t right_last) -> std::size_t {
      if (left_last < right_first) {
        return static_cast<std::size_t>(
          static_cast<std::int64_t>(right_first) -
          static_cast<std::int64_t>(left_last) - 1);
      }
      if (right_last < left_first) {
        return static_cast<std::size_t>(
          static_cast<std::int64_t>(left_first) -
          static_cast<std::int64_t>(right_last) - 1);
      }
      return 0U;
    };
  result.valid = true;
  result.reference_intervening_state_count = intervening_state_count(
    left.reference_first_state_id,
    left.reference_last_state_id,
    right.reference_first_state_id,
    right.reference_last_state_id);
  result.query_intervening_state_count = intervening_state_count(
    left.query_first_state_id,
    left.query_last_state_id,
    right.query_first_state_id,
    right.query_last_state_id);
  result.correlated =
    result.reference_intervening_state_count == 0U &&
    result.query_intervening_state_count == 0U;
  return result;
}

RoughnessPlaceDescriptorMatch matchRoughnessPlaceDescriptors(
  const RoughnessPlaceDescriptor & query,
  const RoughnessPlaceDescriptor & reference)
{
  const DirectionalMatch forward = directionalMatch(query, reference, false);
  const DirectionalMatch reverse = directionalMatch(query, reference, true);
  const bool use_reverse =
    reverse.valid && (!forward.valid || reverse.score > forward.score);
  const DirectionalMatch & selected = use_reverse ? reverse : forward;
  RoughnessPlaceDescriptorMatch result;
  if (!selected.valid) {
    return result;
  }
  result.valid = true;
  result.direction = use_reverse ?
    RoughnessPlaceMatchDirection::kReverse :
    RoughnessPlaceMatchDirection::kForward;
  result.score = selected.score;
  result.shape_score = selected.shape_score;
  result.amplitude_score = selected.amplitude_score;
  result.length_score = selected.length_score;
  result.compared_channel_count = selected.compared_channel_count;
  return result;
}

RoughnessPlaceCandidate bestRoughnessPlaceCandidate(
  const RoughnessPlaceDescriptor & query,
  const std::vector<RoughnessPlaceEvent> & references)
{
  RoughnessPlaceCandidate result;
  double second_best = -1.0;
  for (std::size_t index = 0U; index < references.size(); ++index) {
    const RoughnessPlaceDescriptorMatch match =
      matchRoughnessPlaceDescriptors(query, references[index].descriptor);
    if (!match.valid) {
      continue;
    }
    if (!result.valid || match.score > result.match.score) {
      if (result.valid) {
        second_best = std::max(second_best, result.match.score);
      }
      result.valid = true;
      result.event_index = index;
      result.event_id = references[index].event_id;
      result.match = match;
    } else {
      second_best = std::max(second_best, match.score);
    }
  }
  if (!result.valid) {
    return result;
  }
  if (second_best >= 0.0) {
    result.second_best_score = second_best;
    result.score_margin = result.match.score - second_best;
  }
  return result;
}

std::vector<RoughnessPlaceCandidate> rankedRoughnessPlaceCandidates(
  const RoughnessPlaceEvent & query,
  const std::vector<RoughnessPlaceEvent> & references,
  const std::size_t maximum_candidates,
  const double minimum_event_separation_s,
  const std::size_t minimum_state_id_separation)
{
  std::vector<RoughnessPlaceCandidate> candidates;
  if (
    maximum_candidates == 0U || !query.descriptor.valid() ||
    query.start_stamp_ns <= 0 ||
    !std::isfinite(minimum_event_separation_s) ||
    minimum_event_separation_s < 0.0)
  {
    return candidates;
  }

  for (std::size_t index = 0U; index < references.size(); ++index) {
    const RoughnessPlaceEvent & reference = references[index];
    if (reference.end_stamp_ns <= 0 || reference.end_stamp_ns > query.start_stamp_ns) {
      continue;
    }
    const double event_separation_s =
      static_cast<double>(query.start_stamp_ns - reference.end_stamp_ns) *
      1.0e-9;
    if (event_separation_s < minimum_event_separation_s) {
      continue;
    }

    const RoughnessPlaceDescriptorMatch match =
      matchRoughnessPlaceDescriptors(query.descriptor, reference.descriptor);
    if (!match.valid) {
      continue;
    }
    const RoughnessPlaceAnchorPairing pairing = roughnessPlaceAnchorPairing(
      reference, query, match.direction);
    if (!pairing.valid) {
      continue;
    }
    const std::size_t pre_separation = static_cast<std::size_t>(std::abs(
        static_cast<std::int64_t>(pairing.reference_pre_state_id) -
        static_cast<std::int64_t>(pairing.query_for_reference_pre_state_id)));
    const std::size_t post_separation = static_cast<std::size_t>(std::abs(
        static_cast<std::int64_t>(pairing.reference_post_state_id) -
        static_cast<std::int64_t>(pairing.query_for_reference_post_state_id)));
    const std::size_t state_separation = std::min(pre_separation, post_separation);
    if (state_separation < minimum_state_id_separation) {
      continue;
    }

    RoughnessPlaceCandidate candidate;
    candidate.valid = true;
    candidate.event_index = index;
    candidate.event_id = reference.event_id;
    candidate.match = match;
    candidate.event_separation_s = event_separation_s;
    candidate.minimum_state_id_separation = state_separation;
    candidates.push_back(std::move(candidate));
  }

  std::stable_sort(
    candidates.begin(), candidates.end(),
    [](const RoughnessPlaceCandidate & left, const RoughnessPlaceCandidate & right) {
      if (left.match.score != right.match.score) {
        return left.match.score > right.match.score;
      }
      return left.event_index < right.event_index;
    });
  if (candidates.size() > maximum_candidates) {
    candidates.resize(maximum_candidates);
  }
  if (candidates.size() > 1U) {
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
      const std::size_t comparison_index = index == 0U ? 1U : 0U;
      candidates[index].second_best_score =
        candidates[comparison_index].match.score;
      candidates[index].score_margin =
        candidates[index].match.score -
        candidates[index].second_best_score;
    }
  }
  return candidates;
}

bool validRoughnessPlaceConstraintThresholds(
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept
{
  const auto finite_positive = [](const double value) {
      return std::isfinite(value) && value > 0.0;
    };
  return
    std::isfinite(thresholds.minimum_fine_response) &&
    thresholds.minimum_fine_response >= 0.0 &&
    thresholds.minimum_fine_response <= 1.0 &&
    finite_positive(
    thresholds.maximum_proposal_disagreement_translation_m) &&
    finite_positive(thresholds.maximum_proposal_disagreement_yaw_rad) &&
    finite_positive(thresholds.maximum_fine_xy_covariance_m2) &&
    finite_positive(thresholds.maximum_fine_yaw_covariance_rad2) &&
    finite_positive(thresholds.maximum_bidirectional_translation_error_m) &&
    finite_positive(thresholds.maximum_bidirectional_yaw_error_rad) &&
    finite_positive(thresholds.maximum_pair_alignment_translation_error_m) &&
    finite_positive(thresholds.maximum_pair_alignment_yaw_error_rad);
}

const char * roughnessPlaceConstraintDecisionReasonName(
  const RoughnessPlaceConstraintDecisionReason reason) noexcept
{
  switch (reason) {
    case RoughnessPlaceConstraintDecisionReason::kAccepted:
      return "accepted";
    case RoughnessPlaceConstraintDecisionReason::kMissingEvidence:
      return "missing_evidence";
    case RoughnessPlaceConstraintDecisionReason::kFineResponse:
      return "fine_response";
    case RoughnessPlaceConstraintDecisionReason::kProposalDisagreement:
      return "proposal_disagreement";
    case RoughnessPlaceConstraintDecisionReason::kFineCovariance:
      return "fine_covariance";
    case RoughnessPlaceConstraintDecisionReason::kBidirectionalConsistency:
      return "bidirectional_consistency";
    case RoughnessPlaceConstraintDecisionReason::kPairConsistency:
      return "pair_consistency";
    case RoughnessPlaceConstraintDecisionReason::kInvalidThresholds:
      return "invalid_thresholds";
  }
  return "unknown";
}

RoughnessPlaceConstraintDecision evaluateRoughnessPlaceConstraint(
  const RoughnessPlacePairMatchEvidence & evidence,
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept
{
  RoughnessPlaceConstraintDecision decision;
  if (!validRoughnessPlaceConstraintThresholds(thresholds)) {
    decision.reason = RoughnessPlaceConstraintDecisionReason::kInvalidThresholds;
    return decision;
  }
  const auto directional_complete = [](const RoughnessPlaceDirectionalMatchEvidence & match) {
      return
        match.available && std::isfinite(match.fine_response) &&
        std::isfinite(match.fine_xy_covariance_max_eigenvalue_m2) &&
        match.fine_xy_covariance_max_eigenvalue_m2 >= 0.0 &&
        std::isfinite(match.fine_yaw_covariance_rad2) &&
        match.fine_yaw_covariance_rad2 >= 0.0 &&
        std::isfinite(match.proposal_disagreement_translation_m) &&
        match.proposal_disagreement_translation_m >= 0.0 &&
        std::isfinite(match.proposal_disagreement_yaw_rad) &&
        match.proposal_disagreement_yaw_rad >= 0.0;
    };
  const auto anchor_complete = [&directional_complete](
    const RoughnessPlaceAnchorMatchEvidence & anchor) {
      return
        anchor.available && directional_complete(anchor.forward) &&
        directional_complete(anchor.reverse) &&
        std::isfinite(anchor.bidirectional_translation_error_m) &&
        anchor.bidirectional_translation_error_m >= 0.0 &&
        std::isfinite(anchor.bidirectional_yaw_error_rad) &&
        anchor.bidirectional_yaw_error_rad >= 0.0;
    };
  if (
    !evidence.available || !anchor_complete(evidence.pre_anchor) ||
    !anchor_complete(evidence.post_anchor) ||
    !std::isfinite(evidence.forward_alignment_translation_error_m) ||
    !std::isfinite(evidence.forward_alignment_yaw_error_rad) ||
    !std::isfinite(evidence.reverse_alignment_translation_error_m) ||
    !std::isfinite(evidence.reverse_alignment_yaw_error_rad))
  {
    decision.reason = RoughnessPlaceConstraintDecisionReason::kMissingEvidence;
    return decision;
  }

  // Preserve pair-level diagnostics even when an earlier quality gate rejects
  // the candidate. The values are observational and do not alter gate order.
  decision.maximum_pair_alignment_translation_error_m = std::max(
    evidence.forward_alignment_translation_error_m,
    evidence.reverse_alignment_translation_error_m);
  decision.maximum_pair_alignment_yaw_error_rad = std::max(
    evidence.forward_alignment_yaw_error_rad,
    evidence.reverse_alignment_yaw_error_rad);

  const std::array<const RoughnessPlaceAnchorMatchEvidence *, 2U> anchors{
    &evidence.pre_anchor, &evidence.post_anchor};
  for (const auto * anchor : anchors) {
    if (
      anchor->forward.fine_response < thresholds.minimum_fine_response ||
      anchor->reverse.fine_response < thresholds.minimum_fine_response)
    {
      decision.fail_open = false;
      decision.reason = RoughnessPlaceConstraintDecisionReason::kFineResponse;
      return decision;
    }
    if (
      anchor->forward.proposal_disagreement_translation_m >
      thresholds.maximum_proposal_disagreement_translation_m ||
      anchor->reverse.proposal_disagreement_translation_m >
      thresholds.maximum_proposal_disagreement_translation_m ||
      anchor->forward.proposal_disagreement_yaw_rad >
      thresholds.maximum_proposal_disagreement_yaw_rad ||
      anchor->reverse.proposal_disagreement_yaw_rad >
      thresholds.maximum_proposal_disagreement_yaw_rad)
    {
      decision.fail_open = false;
      decision.reason =
        RoughnessPlaceConstraintDecisionReason::kProposalDisagreement;
      return decision;
    }
    if (
      anchor->forward.fine_xy_covariance_max_eigenvalue_m2 >
      thresholds.maximum_fine_xy_covariance_m2 ||
      anchor->reverse.fine_xy_covariance_max_eigenvalue_m2 >
      thresholds.maximum_fine_xy_covariance_m2 ||
      anchor->forward.fine_yaw_covariance_rad2 >
      thresholds.maximum_fine_yaw_covariance_rad2 ||
      anchor->reverse.fine_yaw_covariance_rad2 >
      thresholds.maximum_fine_yaw_covariance_rad2)
    {
      decision.fail_open = false;
      decision.reason = RoughnessPlaceConstraintDecisionReason::kFineCovariance;
      return decision;
    }
    if (
      anchor->bidirectional_translation_error_m >
      thresholds.maximum_bidirectional_translation_error_m ||
      anchor->bidirectional_yaw_error_rad >
      thresholds.maximum_bidirectional_yaw_error_rad)
    {
      decision.fail_open = false;
      decision.reason =
        RoughnessPlaceConstraintDecisionReason::kBidirectionalConsistency;
      return decision;
    }
  }

  if (
    decision.maximum_pair_alignment_translation_error_m >
    thresholds.maximum_pair_alignment_translation_error_m ||
    decision.maximum_pair_alignment_yaw_error_rad >
    thresholds.maximum_pair_alignment_yaw_error_rad)
  {
    decision.fail_open = false;
    decision.reason = RoughnessPlaceConstraintDecisionReason::kPairConsistency;
    return decision;
  }

  const double pre_min_response = std::min(
    evidence.pre_anchor.forward.fine_response,
    evidence.pre_anchor.reverse.fine_response);
  const double post_min_response = std::min(
    evidence.post_anchor.forward.fine_response,
    evidence.post_anchor.reverse.fine_response);
  decision.selected_anchor_index = post_min_response > pre_min_response ? 1U : 0U;
  decision.selected_minimum_fine_response = std::max(
    pre_min_response, post_min_response);
  decision.accepted = true;
  decision.fail_open = false;
  decision.reason = RoughnessPlaceConstraintDecisionReason::kAccepted;
  return decision;
}

RoughnessPlaceConstraintQuality roughnessPlaceConstraintQuality(
  const RoughnessPlacePairMatchEvidence & evidence,
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept
{
  RoughnessPlaceConstraintQuality quality;
  if (!validRoughnessPlaceConstraintThresholds(thresholds)) {
    return quality;
  }
  const auto directional_complete = [](const RoughnessPlaceDirectionalMatchEvidence & match) {
      return
        match.available && std::isfinite(match.fine_response) &&
        std::isfinite(match.fine_xy_covariance_max_eigenvalue_m2) &&
        match.fine_xy_covariance_max_eigenvalue_m2 >= 0.0 &&
        std::isfinite(match.fine_yaw_covariance_rad2) &&
        match.fine_yaw_covariance_rad2 >= 0.0 &&
        std::isfinite(match.proposal_disagreement_translation_m) &&
        match.proposal_disagreement_translation_m >= 0.0 &&
        std::isfinite(match.proposal_disagreement_yaw_rad) &&
        match.proposal_disagreement_yaw_rad >= 0.0;
    };
  const auto anchor_complete = [&directional_complete](
    const RoughnessPlaceAnchorMatchEvidence & anchor) {
      return
        anchor.available && directional_complete(anchor.forward) &&
        directional_complete(anchor.reverse) &&
        std::isfinite(anchor.bidirectional_translation_error_m) &&
        anchor.bidirectional_translation_error_m >= 0.0 &&
        std::isfinite(anchor.bidirectional_yaw_error_rad) &&
        anchor.bidirectional_yaw_error_rad >= 0.0;
    };
  if (
    !evidence.available || !anchor_complete(evidence.pre_anchor) ||
    !anchor_complete(evidence.post_anchor) ||
    !std::isfinite(evidence.forward_alignment_translation_error_m) ||
    evidence.forward_alignment_translation_error_m < 0.0 ||
    !std::isfinite(evidence.forward_alignment_yaw_error_rad) ||
    evidence.forward_alignment_yaw_error_rad < 0.0 ||
    !std::isfinite(evidence.reverse_alignment_translation_error_m) ||
    evidence.reverse_alignment_translation_error_m < 0.0 ||
    !std::isfinite(evidence.reverse_alignment_yaw_error_rad) ||
    evidence.reverse_alignment_yaw_error_rad < 0.0)
  {
    return quality;
  }

  const std::array<const RoughnessPlaceAnchorMatchEvidence *, 2U> anchors{
    &evidence.pre_anchor, &evidence.post_anchor};
  quality.minimum_fine_response = std::numeric_limits<double>::infinity();
  double maximum_gate_utilization = 0.0;
  const auto include_ratio = [&maximum_gate_utilization](
    const double value, const double maximum) {
      maximum_gate_utilization = std::max(
        maximum_gate_utilization, value / maximum);
    };
  for (const auto * anchor : anchors) {
    for (const auto * direction : {&anchor->forward, &anchor->reverse}) {
      quality.minimum_fine_response = std::min(
        quality.minimum_fine_response, direction->fine_response);
      include_ratio(
        direction->proposal_disagreement_translation_m,
        thresholds.maximum_proposal_disagreement_translation_m);
      include_ratio(
        direction->proposal_disagreement_yaw_rad,
        thresholds.maximum_proposal_disagreement_yaw_rad);
      include_ratio(
        direction->fine_xy_covariance_max_eigenvalue_m2,
        thresholds.maximum_fine_xy_covariance_m2);
      include_ratio(
        direction->fine_yaw_covariance_rad2,
        thresholds.maximum_fine_yaw_covariance_rad2);
    }
    include_ratio(
      anchor->bidirectional_translation_error_m,
      thresholds.maximum_bidirectional_translation_error_m);
    include_ratio(
      anchor->bidirectional_yaw_error_rad,
      thresholds.maximum_bidirectional_yaw_error_rad);
  }
  include_ratio(
    evidence.forward_alignment_translation_error_m,
    thresholds.maximum_pair_alignment_translation_error_m);
  include_ratio(
    evidence.forward_alignment_yaw_error_rad,
    thresholds.maximum_pair_alignment_yaw_error_rad);
  include_ratio(
    evidence.reverse_alignment_translation_error_m,
    thresholds.maximum_pair_alignment_translation_error_m);
  include_ratio(
    evidence.reverse_alignment_yaw_error_rad,
    thresholds.maximum_pair_alignment_yaw_error_rad);
  const double response_denominator = std::max(
    quality.minimum_fine_response, kValueEpsilon);
  include_ratio(thresholds.minimum_fine_response, response_denominator);
  if (!std::isfinite(maximum_gate_utilization)) {
    return RoughnessPlaceConstraintQuality{};
  }
  quality.available = true;
  quality.maximum_gate_utilization = maximum_gate_utilization;
  return quality;
}

const char * roughnessPlaceConstraintHypothesisSelectionReasonName(
  const RoughnessPlaceConstraintHypothesisSelectionReason reason) noexcept
{
  switch (reason) {
    case RoughnessPlaceConstraintHypothesisSelectionReason::kMissingQuality:
      return "missing_quality";
    case RoughnessPlaceConstraintHypothesisSelectionReason::kNoAcceptedHypothesis:
      return "no_accepted_hypothesis";
    case RoughnessPlaceConstraintHypothesisSelectionReason::kOnlyAcceptedHypothesis:
      return "only_accepted_hypothesis";
    case RoughnessPlaceConstraintHypothesisSelectionReason::kLowerGateUtilization:
      return "lower_gate_utilization";
    case RoughnessPlaceConstraintHypothesisSelectionReason::kHigherFineResponse:
      return "higher_fine_response";
    case RoughnessPlaceConstraintHypothesisSelectionReason::kPreferredDirectionTie:
      return "preferred_direction_tie";
  }
  return "unknown";
}

RoughnessPlaceConstraintHypothesisSelection
selectRoughnessPlaceConstraintHypothesis(
  const RoughnessPlaceConstraintHypothesis & preferred,
  const RoughnessPlaceConstraintHypothesis & alternate) noexcept
{
  RoughnessPlaceConstraintHypothesisSelection selection;
  const auto quality_complete = [](const RoughnessPlaceConstraintHypothesis & hypothesis) {
      return
        hypothesis.quality.available &&
        std::isfinite(hypothesis.quality.maximum_gate_utilization) &&
        std::isfinite(hypothesis.quality.minimum_fine_response);
    };
  if (!quality_complete(preferred) || !quality_complete(alternate)) {
    return selection;
  }

  if (preferred.decision.accepted != alternate.decision.accepted) {
    selection.selected = true;
    selection.fail_open = false;
    selection.selected_index = preferred.decision.accepted ? 0U : 1U;
    selection.direction_overridden = selection.selected_index == 1U;
    selection.reason =
      RoughnessPlaceConstraintHypothesisSelectionReason::kOnlyAcceptedHypothesis;
    return selection;
  }
  if (!preferred.decision.accepted) {
    selection.fail_open =
      preferred.decision.fail_open || alternate.decision.fail_open;
    selection.reason =
      RoughnessPlaceConstraintHypothesisSelectionReason::kNoAcceptedHypothesis;
    return selection;
  }

  selection.selected = true;
  selection.fail_open = false;
  if (
    preferred.quality.maximum_gate_utilization !=
    alternate.quality.maximum_gate_utilization)
  {
    selection.selected_index =
      alternate.quality.maximum_gate_utilization <
      preferred.quality.maximum_gate_utilization ? 1U : 0U;
    selection.reason =
      RoughnessPlaceConstraintHypothesisSelectionReason::kLowerGateUtilization;
  } else if (
    preferred.quality.minimum_fine_response !=
    alternate.quality.minimum_fine_response)
  {
    selection.selected_index =
      alternate.quality.minimum_fine_response >
      preferred.quality.minimum_fine_response ? 1U : 0U;
    selection.reason =
      RoughnessPlaceConstraintHypothesisSelectionReason::kHigherFineResponse;
  } else {
    selection.selected_index = 0U;
    selection.reason =
      RoughnessPlaceConstraintHypothesisSelectionReason::kPreferredDirectionTie;
  }
  selection.direction_overridden = selection.selected_index == 1U;
  return selection;
}

}  // namespace slam_2d_core
