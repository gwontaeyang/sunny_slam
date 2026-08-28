#include "slam_2d_core/adaptive_roughness_memory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slam_2d_core
{
namespace
{

constexpr double kNanosecondsPerSecond = 1.0e9;

void setReason(std::string * reason, const char * value) noexcept
{
  if (reason) {
    *reason = value;
  }
}

bool finiteNonnegative(const double value) noexcept
{
  return std::isfinite(value) && value >= 0.0;
}

double unitRamp(
  const double value,
  const double low,
  const double high) noexcept
{
  return std::clamp((value - low) / (high - low), 0.0, 1.0);
}

}  // namespace

const char * roughnessConstraintModeName(
  const RoughnessConstraintMode mode) noexcept
{
  switch (mode) {
    case RoughnessConstraintMode::kInterventionOff:
      return "intervention_off";
    case RoughnessConstraintMode::kLegacyFixedMemory:
      return "legacy_fixed_memory";
    case RoughnessConstraintMode::kAdaptiveMemoryObserve:
      return "adaptive_memory_observe";
    case RoughnessConstraintMode::kAdaptiveMemoryOnly:
      return "adaptive_memory_only";
    case RoughnessConstraintMode::kAdaptiveQualityAware:
      return "adaptive_quality_aware";
  }
  return "intervention_off";
}

bool parseRoughnessConstraintMode(
  const std::string & name,
  RoughnessConstraintMode & mode) noexcept
{
  if (name == "intervention_off") {
    mode = RoughnessConstraintMode::kInterventionOff;
  } else if (name == "legacy_fixed_memory") {
    mode = RoughnessConstraintMode::kLegacyFixedMemory;
  } else if (name == "adaptive_memory_observe") {
    mode = RoughnessConstraintMode::kAdaptiveMemoryObserve;
  } else if (name == "adaptive_memory_only") {
    mode = RoughnessConstraintMode::kAdaptiveMemoryOnly;
  } else if (name == "adaptive_quality_aware") {
    mode = RoughnessConstraintMode::kAdaptiveQualityAware;
  } else {
    return false;
  }
  return true;
}

bool validateAdaptiveRoughnessMemoryConfig(
  const AdaptiveRoughnessMemoryConfig & config,
  std::string * reason) noexcept
{
  if (!std::isfinite(config.trigger_score) || config.trigger_score <= 0.0) {
    setReason(reason, "trigger_score must be finite and positive");
    return false;
  }
  if (
    !std::isfinite(config.saturation_score) ||
    config.saturation_score <= config.trigger_score)
  {
    setReason(reason, "saturation_score must be finite and greater than trigger_score");
    return false;
  }
  if (
    !finiteNonnegative(config.recovery_score) ||
    config.recovery_score >= config.trigger_score)
  {
    setReason(reason, "recovery_score must be below trigger_score");
    return false;
  }
  if (
    !std::isfinite(config.minimum_duration_s) ||
    config.minimum_duration_s <= 0.0 ||
    !std::isfinite(config.maximum_duration_s) ||
    config.maximum_duration_s <= 0.0 ||
    config.minimum_duration_s > config.maximum_duration_s)
  {
    setReason(reason, "duration bounds are invalid");
    return false;
  }
  if (
    !std::isfinite(config.minimum_distance_m) ||
    config.minimum_distance_m <= 0.0 ||
    !std::isfinite(config.maximum_distance_m) ||
    config.maximum_distance_m <= 0.0 ||
    config.minimum_distance_m > config.maximum_distance_m)
  {
    setReason(reason, "distance bounds are invalid");
    return false;
  }
  if (
    !std::isfinite(config.stable_recovery_time_constant_s) ||
    config.stable_recovery_time_constant_s <= 0.0)
  {
    setReason(reason, "stable_recovery_time_constant_s must be positive");
    return false;
  }
  setReason(reason, "");
  return true;
}

AdaptiveRoughnessMemory::AdaptiveRoughnessMemory(
  AdaptiveRoughnessMemoryConfig config)
{
  configure(config);
}

void AdaptiveRoughnessMemory::configure(
  const AdaptiveRoughnessMemoryConfig & config)
{
  std::string reason;
  if (!validateAdaptiveRoughnessMemoryConfig(config, &reason)) {
    throw std::invalid_argument(reason);
  }
  config_ = config;
  reset();
}

void AdaptiveRoughnessMemory::reset() noexcept
{
  have_last_stamp_ = false;
  have_last_observation_ = false;
  event_active_ = false;
  last_seen_stamp_ns_ = 0;
  last_stamp_ns_ = 0;
  event_stamp_ns_ = 0;
  last_x_m_ = 0.0;
  last_y_m_ = 0.0;
  event_intensity_ = 0.0;
  event_distance_m_ = 0.0;
  stable_recovery_s_ = 0.0;
}

AdaptiveRoughnessMemoryState AdaptiveRoughnessMemory::unavailableState(
  const bool timestamp_reset) const noexcept
{
  AdaptiveRoughnessMemoryState state;
  state.timestamp_reset = timestamp_reset;
  state.active = event_active_;
  state.event_intensity = event_intensity_;
  state.distance_m = event_distance_m_;
  state.stable_recovery_s = stable_recovery_s_;
  return state;
}

double AdaptiveRoughnessMemory::normalizedEventIntensity(
  const double roughness_score) const noexcept
{
  return std::clamp(
    roughness_score / config_.saturation_score,
    0.0,
    1.0);
}

AdaptiveRoughnessMemoryState AdaptiveRoughnessMemory::update(
  const AdaptiveRoughnessMemoryObservation & observation)
{
  if (
    observation.stamp_ns > 0 && have_last_stamp_ &&
    observation.stamp_ns < last_seen_stamp_ns_)
  {
    reset();
    return unavailableState(true);
  }
  if (observation.stamp_ns > 0) {
    have_last_stamp_ = true;
    last_seen_stamp_ns_ = observation.stamp_ns;
  }

  const bool input_valid =
    observation.stamp_ns > 0 && observation.pose_available &&
    std::isfinite(observation.x_m) && std::isfinite(observation.y_m) &&
    observation.roughness_available &&
    std::isfinite(observation.roughness_score) &&
    observation.roughness_score >= 0.0;
  if (!input_valid) {
    return unavailableState(false);
  }

  double delta_s = 0.0;
  if (have_last_observation_) {
    delta_s = static_cast<double>(observation.stamp_ns - last_stamp_ns_) /
      kNanosecondsPerSecond;
    event_distance_m_ += std::hypot(
      observation.x_m - last_x_m_,
      observation.y_m - last_y_m_);
  }

  double prior_strength = 0.0;
  if (event_active_) {
    const double prior_elapsed_s =
      static_cast<double>(observation.stamp_ns - event_stamp_ns_) /
      kNanosecondsPerSecond;
    const double prior_duration_s = config_.minimum_duration_s +
      event_intensity_ *
      (config_.maximum_duration_s - config_.minimum_duration_s);
    const double prior_distance_m = config_.minimum_distance_m +
      event_intensity_ *
      (config_.maximum_distance_m - config_.minimum_distance_m);
    const double time_decay = std::clamp(
      1.0 - prior_elapsed_s / prior_duration_s, 0.0, 1.0);
    const double distance_decay = std::clamp(
      1.0 - event_distance_m_ / prior_distance_m, 0.0, 1.0);
    const double recovery_decay = std::exp(
      -stable_recovery_s_ / config_.stable_recovery_time_constant_s);
    prior_strength = std::clamp(
      event_intensity_ * time_decay * distance_decay * recovery_decay,
      0.0,
      1.0);
  }

  have_last_observation_ = true;
  last_stamp_ns_ = observation.stamp_ns;
  last_x_m_ = observation.x_m;
  last_y_m_ = observation.y_m;

  AdaptiveRoughnessMemoryState state;
  state.available = true;
  state.rough_trigger =
    observation.roughness_score >= config_.trigger_score;
  if (state.rough_trigger) {
    event_active_ = true;
    event_stamp_ns_ = observation.stamp_ns;
    event_distance_m_ = 0.0;
    stable_recovery_s_ = 0.0;
    event_intensity_ = std::max(
      prior_strength,
      normalizedEventIntensity(observation.roughness_score));
  } else if (
    event_active_ &&
    observation.roughness_score <= config_.recovery_score)
  {
    stable_recovery_s_ += delta_s;
  } else if (event_active_) {
    // Recovery must be continuous; a deadband sample breaks the stable run.
    stable_recovery_s_ = 0.0;
  }

  state.event_intensity = event_intensity_;
  state.elapsed_s = event_active_ ?
    static_cast<double>(observation.stamp_ns - event_stamp_ns_) /
    kNanosecondsPerSecond : 0.0;
  state.distance_m = event_distance_m_;
  state.stable_recovery_s = stable_recovery_s_;
  state.effective_duration_s = config_.minimum_duration_s +
    event_intensity_ *
    (config_.maximum_duration_s - config_.minimum_duration_s);
  state.effective_distance_m = config_.minimum_distance_m +
    event_intensity_ *
    (config_.maximum_distance_m - config_.minimum_distance_m);

  if (!event_active_) {
    return state;
  }

  const double time_decay = std::clamp(
    1.0 - state.elapsed_s / state.effective_duration_s,
    0.0,
    1.0);
  const double distance_decay = std::clamp(
    1.0 - state.distance_m / state.effective_distance_m,
    0.0,
    1.0);
  const double recovery_decay = std::exp(
    -state.stable_recovery_s / config_.stable_recovery_time_constant_s);
  state.strength = std::clamp(
    event_intensity_ * time_decay * distance_decay * recovery_decay,
    0.0,
    1.0);
  state.active = state.strength > 0.0;
  if (!state.active) {
    event_active_ = false;
    event_intensity_ = 0.0;
    event_distance_m_ = 0.0;
    stable_recovery_s_ = 0.0;
  }
  return state;
}

bool validateNearChainQualityConfig(
  const NearChainQualityConfig & config,
  std::string * reason) noexcept
{
  if (!config.calibrated) {
    setReason(reason, "quality calibration is disabled");
    return false;
  }
  if (
    !finiteNonnegative(config.response_bad) ||
    !std::isfinite(config.response_good) ||
    config.response_good <= config.response_bad)
  {
    setReason(reason, "response bounds are invalid");
    return false;
  }
  const auto valid_bounds = [](const double good, const double bad) {
      return finiteNonnegative(good) && std::isfinite(bad) && bad > good;
    };
  if (!valid_bounds(
      config.translation_disagreement_good_m,
      config.translation_disagreement_bad_m))
  {
    setReason(reason, "translation disagreement bounds are invalid");
    return false;
  }
  if (!valid_bounds(
      config.yaw_disagreement_good_rad,
      config.yaw_disagreement_bad_rad))
  {
    setReason(reason, "yaw disagreement bounds are invalid");
    return false;
  }
  if (!valid_bounds(
      config.xy_covariance_good_m2,
      config.xy_covariance_bad_m2))
  {
    setReason(reason, "XY covariance bounds are invalid");
    return false;
  }
  if (!valid_bounds(
      config.yaw_covariance_good_rad2,
      config.yaw_covariance_bad_rad2))
  {
    setReason(reason, "yaw covariance bounds are invalid");
    return false;
  }
  setReason(reason, "");
  return true;
}

NearChainQuality evaluateNearChainQuality(
  const NearChainQualityObservation & observation,
  const NearChainQualityConfig & config) noexcept
{
  NearChainQuality quality;
  if (!observation.available || !validateNearChainQualityConfig(config)) {
    return quality;
  }
  if (
    !finiteNonnegative(observation.normalized_response) ||
    !finiteNonnegative(observation.translation_disagreement_m) ||
    !finiteNonnegative(observation.yaw_disagreement_rad) ||
    !finiteNonnegative(observation.xy_covariance_max_eigenvalue_m2) ||
    !finiteNonnegative(observation.yaw_covariance_rad2))
  {
    return quality;
  }

  quality.response_risk = 1.0 - unitRamp(
    observation.normalized_response,
    config.response_bad,
    config.response_good);
  quality.disagreement_risk = std::max(
    unitRamp(
      observation.translation_disagreement_m,
      config.translation_disagreement_good_m,
      config.translation_disagreement_bad_m),
    unitRamp(
      std::fabs(observation.yaw_disagreement_rad),
      config.yaw_disagreement_good_rad,
      config.yaw_disagreement_bad_rad));
  quality.covariance_risk = std::max(
    unitRamp(
      observation.xy_covariance_max_eigenvalue_m2,
      config.xy_covariance_good_m2,
      config.xy_covariance_bad_m2),
    unitRamp(
      observation.yaw_covariance_rad2,
      config.yaw_covariance_good_rad2,
      config.yaw_covariance_bad_rad2));

  // A weak response alone is not enough: independent geometric or uncertainty
  // evidence must also indicate that this candidate is risky.
  quality.risk = quality.response_risk * std::max(
    quality.disagreement_risk,
    quality.covariance_risk);
  quality.available = true;
  return quality;
}

const char * roughnessConstraintActionName(
  const RoughnessConstraintAction action) noexcept
{
  switch (action) {
    case RoughnessConstraintAction::kAccepted:
      return "accepted";
    case RoughnessConstraintAction::kDownweighted:
      return "downweighted";
    case RoughnessConstraintAction::kRejected:
      return "rejected";
  }
  return "accepted";
}

bool validateAdaptiveConstraintPolicyConfig(
  const AdaptiveConstraintPolicyConfig & config,
  std::string * reason) noexcept
{
  if (
    !finiteNonnegative(config.downweight_risk) ||
    !std::isfinite(config.reject_risk) ||
    config.reject_risk <= 0.0 ||
    config.reject_risk > 1.0 ||
    config.downweight_risk > config.reject_risk)
  {
    setReason(reason, "risk thresholds must satisfy 0 <= downweight <= reject <= 1");
    return false;
  }
  if (
    !std::isfinite(config.minimum_information_scale) ||
    config.minimum_information_scale <= 0.0 ||
    config.minimum_information_scale > 1.0)
  {
    setReason(reason, "minimum_information_scale must be in (0, 1]");
    return false;
  }
  setReason(reason, "");
  return true;
}

AdaptiveConstraintDecision adaptiveConstraintDecision(
  const RoughnessConstraintMode mode,
  const AdaptiveRoughnessMemoryState & memory,
  const NearChainQuality & quality,
  const AdaptiveConstraintPolicyConfig & config) noexcept
{
  AdaptiveConstraintDecision decision;
  decision.memory_strength = memory.strength;
  decision.quality_risk = quality.risk;

  if (
    mode == RoughnessConstraintMode::kInterventionOff ||
    mode == RoughnessConstraintMode::kLegacyFixedMemory ||
    mode == RoughnessConstraintMode::kAdaptiveMemoryObserve)
  {
    return decision;
  }
  if (!validateAdaptiveConstraintPolicyConfig(config) || !memory.available) {
    decision.fail_open = true;
    return decision;
  }

  if (mode == RoughnessConstraintMode::kAdaptiveQualityAware) {
    if (!quality.available) {
      decision.fail_open = true;
      return decision;
    }
    decision.quality_risk = quality.risk;
  } else {
    decision.quality_risk = 1.0;
  }

  decision.evidence_available = true;
  decision.combined_risk = std::clamp(
    memory.strength * decision.quality_risk,
    0.0,
    1.0);
  if (decision.combined_risk <= config.downweight_risk) {
    return decision;
  }

  decision.intervention_applied = true;
  if (decision.combined_risk >= config.reject_risk) {
    decision.action = RoughnessConstraintAction::kRejected;
    decision.information_scale = 0.0;
  } else {
    decision.action = RoughnessConstraintAction::kDownweighted;
    const double span = std::max(
      config.reject_risk - config.downweight_risk,
      1.0e-12);
    const double progress = std::clamp(
      (decision.combined_risk - config.downweight_risk) / span,
      0.0,
      1.0);
    decision.information_scale = 1.0 -
      progress * (1.0 - config.minimum_information_scale);
  }
  return decision;
}

DynamicResponseThresholdDecision deriveDynamicResponseThreshold(
  const double baseline_threshold,
  const double maximum_threshold,
  const double acceptance_tolerance,
  const std::vector<DynamicResponseThresholdCandidate> & candidates) noexcept
{
  DynamicResponseThresholdDecision decision;
  decision.baseline_threshold = baseline_threshold;
  decision.requested_threshold = baseline_threshold;
  if (
    !finiteNonnegative(baseline_threshold) ||
    !std::isfinite(maximum_threshold) ||
    maximum_threshold < baseline_threshold ||
    !finiteNonnegative(acceptance_tolerance))
  {
    decision.fail_open = true;
    return decision;
  }

  double highest_rejected = -std::numeric_limits<double>::infinity();
  double lowest_preserved = std::numeric_limits<double>::infinity();
  for (const auto & candidate : candidates) {
    if (
      !candidate.eligible ||
      !finiteNonnegative(candidate.normalized_response) ||
      candidate.normalized_response <= baseline_threshold - acceptance_tolerance)
    {
      continue;
    }
    if (candidate.action == RoughnessConstraintAction::kRejected) {
      highest_rejected = std::max(
        highest_rejected, candidate.normalized_response);
      ++decision.rejected_candidate_count;
    } else {
      lowest_preserved = std::min(
        lowest_preserved, candidate.normalized_response);
      ++decision.preserved_candidate_count;
    }
  }

  decision.available = true;
  if (decision.rejected_candidate_count == 0U) {
    return decision;
  }
  decision.highest_rejected_response = highest_rejected;
  if (decision.preserved_candidate_count != 0U) {
    decision.lowest_preserved_response = lowest_preserved;
    if (
      highest_rejected + 2.0 * acceptance_tolerance >=
      lowest_preserved)
    {
      decision.fail_open = true;
      return decision;
    }
    decision.requested_threshold =
      0.5 * (highest_rejected + lowest_preserved);
  } else {
    decision.requested_threshold =
      highest_rejected + 2.0 * acceptance_tolerance;
  }

  if (
    !std::isfinite(decision.requested_threshold) ||
    decision.requested_threshold > maximum_threshold ||
    decision.requested_threshold <= baseline_threshold)
  {
    decision.requested_threshold = baseline_threshold;
    decision.fail_open = true;
    return decision;
  }
  decision.intervention_required = true;
  return decision;
}

}  // namespace slam_2d_core
