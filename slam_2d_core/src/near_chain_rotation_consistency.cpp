#include "slam_2d_core/near_chain_rotation_consistency.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace slam_2d_core
{
namespace
{

constexpr double kMadToSigma = 1.4826;

void setReason(std::string * reason, const char * value) noexcept
{
  if (reason) {
    *reason = value;
  }
}

bool finitePositive(const double value) noexcept
{
  return std::isfinite(value) && value > 0.0;
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if (values.size() % 2U != 0U) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1U] + values[middle]);
}

NearChainRobustLimit robustLimit(
  const std::deque<double> & values,
  const std::size_t minimum_samples,
  const double sigma,
  const double scale_floor,
  const double extreme_sigma)
{
  NearChainRobustLimit result;
  std::vector<double> finite_values;
  finite_values.reserve(values.size());
  for (const double value : values) {
    if (std::isfinite(value)) {
      finite_values.push_back(value);
    }
  }
  if (finite_values.size() < minimum_samples) {
    return result;
  }
  result.median = median(finite_values);
  for (double & value : finite_values) {
    value = std::fabs(value - result.median);
  }
  result.scale = std::max(kMadToSigma * median(finite_values), scale_floor);
  result.limit = result.median + sigma * result.scale;
  result.extreme_limit = result.median + extreme_sigma * result.scale;
  result.available =
    std::isfinite(result.median) && std::isfinite(result.scale) &&
    std::isfinite(result.limit) && std::isfinite(result.extreme_limit);
  return result;
}

double chainOverlap(
  const std::vector<std::int32_t> & first,
  const std::vector<std::int32_t> & second)
{
  if (first.empty() || second.empty()) {
    return 0.0;
  }
  const std::unordered_set<std::int32_t> first_set(first.begin(), first.end());
  const std::unordered_set<std::int32_t> second_set(second.begin(), second.end());
  std::size_t intersection = 0U;
  for (const std::int32_t value : first_set) {
    intersection += second_set.count(value) != 0U ? 1U : 0U;
  }
  const std::size_t union_size =
    first_set.size() + second_set.size() - intersection;
  return union_size == 0U ? 0.0 :
         static_cast<double>(intersection) / static_cast<double>(union_size);
}

double maximumChainOverlap(
  const std::vector<std::int32_t> & chain,
  const std::vector<std::vector<std::int32_t>> & prior_chains)
{
  double maximum = 0.0;
  for (const auto & prior : prior_chains) {
    maximum = std::max(maximum, chainOverlap(chain, prior));
  }
  return maximum;
}

struct CandidateFeatures
{
  bool available{false};
  double yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double translation_m{std::numeric_limits<double>::quiet_NaN()};
  double weighted_disagreement{std::numeric_limits<double>::quiet_NaN()};
};

CandidateFeatures candidateFeatures(
  const NearChainRotationCandidateObservation & observation) noexcept
{
  CandidateFeatures features;
  features.yaw_rad = std::fabs(observation.yaw_disagreement_rad);
  features.translation_m = observation.translation_disagreement_m;
  if (
    std::isfinite(observation.covariance_weighted_disagreement_sq) &&
    observation.covariance_weighted_disagreement_sq >= 0.0)
  {
    features.weighted_disagreement = std::log1p(
      observation.covariance_weighted_disagreement_sq);
  }
  features.available =
    observation.eligible && std::isfinite(observation.normalized_response) &&
    observation.normalized_response >= 0.0 &&
    std::isfinite(features.yaw_rad) &&
    std::isfinite(features.translation_m) && features.translation_m >= 0.0 &&
    std::isfinite(features.weighted_disagreement);
  return features;
}

void appendBounded(
  std::deque<double> & values,
  const double value,
  const std::size_t maximum_size)
{
  values.push_back(value);
  while (values.size() > maximum_size) {
    values.pop_front();
  }
}

}  // namespace

bool validateNearChainRotationConsistencyConfig(
  const NearChainRotationConsistencyConfig & config,
  std::string * reason) noexcept
{
  if (
    config.history_size == 0U || config.minimum_history_batches == 0U ||
    config.minimum_history_batches > config.history_size)
  {
    setReason(reason, "history sizes must satisfy 1 <= minimum <= capacity");
    return false;
  }
  if (
    !finitePositive(config.outlier_sigma) ||
    !std::isfinite(config.extreme_yaw_sigma) ||
    config.extreme_yaw_sigma < config.outlier_sigma)
  {
    setReason(reason, "sigma bounds are invalid");
    return false;
  }
  if (
    !finitePositive(config.yaw_scale_floor_rad) ||
    !finitePositive(config.translation_scale_floor_m) ||
    !finitePositive(config.weighted_disagreement_scale_floor))
  {
    setReason(reason, "robust scale floors must be finite and positive");
    return false;
  }
  if (
    !std::isfinite(config.chain_overlap_trigger) ||
    config.chain_overlap_trigger < 0.0 || config.chain_overlap_trigger > 1.0)
  {
    setReason(reason, "chain overlap trigger must be in [0, 1]");
    return false;
  }
  if (
    config.maximum_repeat_state_gap < 1 ||
    config.stable_recovery_batches == 0U)
  {
    setReason(reason, "repeat gap and recovery batch count must be positive");
    return false;
  }
  setReason(reason, "");
  return true;
}

NearChainRotationConsistency::NearChainRotationConsistency(
  NearChainRotationConsistencyConfig config)
{
  configure(config);
}

void NearChainRotationConsistency::configure(
  const NearChainRotationConsistencyConfig & config)
{
  std::string reason;
  if (!validateNearChainRotationConsistencyConfig(config, &reason)) {
    throw std::invalid_argument(
            "Invalid near-chain rotation consistency config: " + reason);
  }
  config_ = config;
  reset();
}

NearChainRotationBatchDecision NearChainRotationConsistency::evaluate(
  const NearChainRotationBatchObservation & observation,
  const bool intervention_enabled)
{
  NearChainRotationBatchDecision result;
  result.candidates.resize(observation.candidates.size());
  if (!config_.enabled) {
    return result;
  }
  result.available = observation.stamp_ns > 0 && observation.target_state_id >= 0;
  if (!result.available) {
    result.fail_open = intervention_enabled;
    for (auto & candidate : result.candidates) {
      candidate.fail_open = intervention_enabled;
    }
    return result;
  }

  if (have_last_stamp_ && observation.stamp_ns < last_stamp_ns_) {
    reset();
    result.timestamp_reset = true;
  }
  have_last_stamp_ = true;
  last_stamp_ns_ = observation.stamp_ns;

  result.latch_active_before = latch_active_;
  result.history_count = std::min(
    {yaw_history_rad_.size(), translation_history_m_.size(),
      weighted_disagreement_history_.size()});
  result.yaw_limit = robustLimit(
    yaw_history_rad_, config_.minimum_history_batches,
    config_.outlier_sigma, config_.yaw_scale_floor_rad,
    config_.extreme_yaw_sigma);
  result.translation_limit = robustLimit(
    translation_history_m_, config_.minimum_history_batches,
    config_.outlier_sigma, config_.translation_scale_floor_m,
    config_.outlier_sigma);
  result.weighted_disagreement_limit = robustLimit(
    weighted_disagreement_history_, config_.minimum_history_batches,
    config_.outlier_sigma, config_.weighted_disagreement_scale_floor,
    config_.outlier_sigma);
  result.model_available =
    result.yaw_limit.available && result.translation_limit.available &&
    result.weighted_disagreement_limit.available;

  const bool memory_evidence_available =
    observation.memory_available &&
    std::isfinite(observation.memory_strength) &&
    observation.memory_strength >= 0.0 && observation.memory_strength <= 1.0;
  result.memory_trigger =
    memory_evidence_available && observation.memory_strength > 0.0;

  std::vector<CandidateFeatures> features;
  std::vector<bool> risky;
  features.reserve(observation.candidates.size());
  risky.reserve(observation.candidates.size());
  bool any_risky = false;
  for (std::size_t index = 0; index < observation.candidates.size(); ++index) {
    const auto & candidate = observation.candidates[index];
    const CandidateFeatures values = candidateFeatures(candidate);
    features.push_back(values);
    auto & decision = result.candidates[index];
    decision.evidence_available = values.available && result.model_available;
    decision.absolute_yaw_disagreement_rad = values.yaw_rad;
    decision.translation_disagreement_m = values.translation_m;
    decision.log1p_covariance_weighted_disagreement =
      values.weighted_disagreement;
    decision.prior_chain_overlap = maximumChainOverlap(
      candidate.chain_state_ids, previous_risky_chains_);
    const bool yaw_outlier =
      decision.evidence_available && values.yaw_rad > result.yaw_limit.limit;
    const bool yaw_extreme =
      decision.evidence_available &&
      values.yaw_rad > result.yaw_limit.extreme_limit;
    const bool translation_outlier =
      decision.evidence_available &&
      values.translation_m > result.translation_limit.limit;
    const bool weighted_outlier =
      decision.evidence_available &&
      values.weighted_disagreement >
      result.weighted_disagreement_limit.limit;
    decision.candidate_risky =
      weighted_outlier && (yaw_extreme || (yaw_outlier && translation_outlier));
    risky.push_back(decision.candidate_risky);
    any_risky = any_risky || decision.candidate_risky;
  }

  const std::int32_t repeat_state_gap =
    observation.target_state_id - previous_risky_target_state_id_;
  if (
    previous_risky_target_state_id_ >= 0 && repeat_state_gap > 0 &&
    repeat_state_gap <= config_.maximum_repeat_state_gap)
  {
    for (std::size_t index = 0; index < result.candidates.size(); ++index) {
      if (
        result.candidates[index].candidate_risky &&
        result.candidates[index].prior_chain_overlap >=
        config_.chain_overlap_trigger)
      {
        result.repeated_chain_trigger = true;
        break;
      }
    }
  }

  const bool event_trigger =
    any_risky &&
    (latch_active_ || result.memory_trigger || result.repeated_chain_trigger);
  for (std::size_t index = 0; index < result.candidates.size(); ++index) {
    const auto & candidate = observation.candidates[index];
    auto & decision = result.candidates[index];
    decision.proposed_reject = decision.candidate_risky && event_trigger;
    const bool active_missing_quality =
      intervention_enabled && candidate.eligible &&
      (latch_active_ || result.memory_trigger) &&
      (!result.model_available || !features[index].available);
    decision.fail_open =
      active_missing_quality ||
      (intervention_enabled && decision.proposed_reject &&
      !memory_evidence_available);
    decision.intervention_reject =
      intervention_enabled && decision.proposed_reject &&
      memory_evidence_available && !decision.fail_open;
    result.fail_open = result.fail_open || decision.fail_open;
  }

  if (any_risky) {
    previous_risky_target_state_id_ = observation.target_state_id;
    previous_risky_chains_.clear();
    for (std::size_t index = 0; index < observation.candidates.size(); ++index) {
      if (risky[index]) {
        previous_risky_chains_.push_back(
          observation.candidates[index].chain_state_ids);
      }
    }
    stable_batch_count_ = 0U;
    if (event_trigger && memory_evidence_available) {
      latch_active_ = true;
    }
  } else {
    previous_risky_target_state_id_ = -1;
    previous_risky_chains_.clear();
    if (latch_active_) {
      ++stable_batch_count_;
      if (stable_batch_count_ >= config_.stable_recovery_batches) {
        latch_active_ = false;
        stable_batch_count_ = 0U;
      }
    }
  }
  result.latch_active_after = latch_active_;

  std::vector<double> stable_yaw;
  std::vector<double> stable_translation;
  std::vector<double> stable_weighted;
  for (std::size_t index = 0; index < features.size(); ++index) {
    if (features[index].available && !risky[index]) {
      stable_yaw.push_back(features[index].yaw_rad);
      stable_translation.push_back(features[index].translation_m);
      stable_weighted.push_back(features[index].weighted_disagreement);
    }
  }
  if (!stable_yaw.empty()) {
    appendBounded(
      yaw_history_rad_, median(std::move(stable_yaw)), config_.history_size);
    appendBounded(
      translation_history_m_, median(std::move(stable_translation)),
      config_.history_size);
    appendBounded(
      weighted_disagreement_history_, median(std::move(stable_weighted)),
      config_.history_size);
  }
  return result;
}

void NearChainRotationConsistency::reset() noexcept
{
  yaw_history_rad_.clear();
  translation_history_m_.clear();
  weighted_disagreement_history_.clear();
  have_last_stamp_ = false;
  last_stamp_ns_ = 0;
  previous_risky_target_state_id_ = -1;
  previous_risky_chains_.clear();
  latch_active_ = false;
  stable_batch_count_ = 0U;
}

}  // namespace slam_2d_core
