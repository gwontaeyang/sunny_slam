#ifndef SLAM_2D_CORE__ADAPTIVE_ROUGHNESS_MEMORY_HPP_
#define SLAM_2D_CORE__ADAPTIVE_ROUGHNESS_MEMORY_HPP_

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace slam_2d_core
{

enum class RoughnessConstraintMode
{
  kInterventionOff,
  kLegacyFixedMemory,
  kAdaptiveMemoryObserve,
  kAdaptiveMemoryOnly,
  kAdaptiveQualityAware,
};

const char * roughnessConstraintModeName(RoughnessConstraintMode mode) noexcept;

bool parseRoughnessConstraintMode(
  const std::string & name,
  RoughnessConstraintMode & mode) noexcept;

struct AdaptiveRoughnessMemoryConfig
{
  double trigger_score{2.75};
  double saturation_score{5.5};
  double recovery_score{1.0};
  double minimum_duration_s{0.5};
  double maximum_duration_s{3.5};
  double minimum_distance_m{0.10};
  double maximum_distance_m{0.55};
  double stable_recovery_time_constant_s{0.5};
};

bool validateAdaptiveRoughnessMemoryConfig(
  const AdaptiveRoughnessMemoryConfig & config,
  std::string * reason = nullptr) noexcept;

struct AdaptiveRoughnessMemoryObservation
{
  std::int64_t stamp_ns{0};
  bool pose_available{false};
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  bool roughness_available{false};
  double roughness_score{std::numeric_limits<double>::quiet_NaN()};
};

struct AdaptiveRoughnessMemoryState
{
  bool available{false};
  bool active{false};
  bool rough_trigger{false};
  bool timestamp_reset{false};
  double strength{0.0};
  double event_intensity{0.0};
  double elapsed_s{0.0};
  double distance_m{0.0};
  double stable_recovery_s{0.0};
  double effective_duration_s{0.0};
  double effective_distance_m{0.0};
};

class AdaptiveRoughnessMemory
{
public:
  explicit AdaptiveRoughnessMemory(
    AdaptiveRoughnessMemoryConfig config = AdaptiveRoughnessMemoryConfig{});

  void configure(const AdaptiveRoughnessMemoryConfig & config);
  AdaptiveRoughnessMemoryState update(
    const AdaptiveRoughnessMemoryObservation & observation);

  void reset() noexcept;
  const AdaptiveRoughnessMemoryConfig & config() const noexcept {return config_;}

private:
  AdaptiveRoughnessMemoryState unavailableState(bool timestamp_reset) const noexcept;
  double normalizedEventIntensity(double roughness_score) const noexcept;

  AdaptiveRoughnessMemoryConfig config_;
  bool have_last_stamp_{false};
  bool have_last_observation_{false};
  bool event_active_{false};
  std::int64_t last_seen_stamp_ns_{0};
  std::int64_t last_stamp_ns_{0};
  std::int64_t event_stamp_ns_{0};
  double last_x_m_{0.0};
  double last_y_m_{0.0};
  double event_intensity_{0.0};
  double event_distance_m_{0.0};
  double stable_recovery_s_{0.0};
};

struct NearChainQualityConfig
{
  bool calibrated{false};
  double response_bad{0.08};
  double response_good{0.30};
  double translation_disagreement_good_m{0.02};
  double translation_disagreement_bad_m{0.10};
  double yaw_disagreement_good_rad{0.017453292519943295};
  double yaw_disagreement_bad_rad{0.08726646259971647};
  double xy_covariance_good_m2{0.0025};
  double xy_covariance_bad_m2{0.04};
  double yaw_covariance_good_rad2{0.00030461741978670857};
  double yaw_covariance_bad_rad2{0.030461741978670857};
};

bool validateNearChainQualityConfig(
  const NearChainQualityConfig & config,
  std::string * reason = nullptr) noexcept;

struct NearChainQualityObservation
{
  bool available{false};
  double normalized_response{std::numeric_limits<double>::quiet_NaN()};
  double translation_disagreement_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_disagreement_rad{std::numeric_limits<double>::quiet_NaN()};
  double xy_covariance_max_eigenvalue_m2{std::numeric_limits<double>::quiet_NaN()};
  double yaw_covariance_rad2{std::numeric_limits<double>::quiet_NaN()};
};

struct NearChainQuality
{
  bool available{false};
  double response_risk{0.0};
  double disagreement_risk{0.0};
  double covariance_risk{0.0};
  double risk{0.0};
};

NearChainQuality evaluateNearChainQuality(
  const NearChainQualityObservation & observation,
  const NearChainQualityConfig & config) noexcept;

enum class RoughnessConstraintAction
{
  kAccepted,
  kDownweighted,
  kRejected,
};

const char * roughnessConstraintActionName(
  RoughnessConstraintAction action) noexcept;

struct AdaptiveConstraintPolicyConfig
{
  double downweight_risk{0.35};
  double reject_risk{0.80};
  double minimum_information_scale{0.10};
};

bool validateAdaptiveConstraintPolicyConfig(
  const AdaptiveConstraintPolicyConfig & config,
  std::string * reason = nullptr) noexcept;

struct AdaptiveConstraintDecision
{
  bool evidence_available{false};
  bool intervention_applied{false};
  bool fail_open{false};
  double memory_strength{0.0};
  double quality_risk{0.0};
  double combined_risk{0.0};
  double information_scale{1.0};
  RoughnessConstraintAction action{RoughnessConstraintAction::kAccepted};
};

AdaptiveConstraintDecision adaptiveConstraintDecision(
  RoughnessConstraintMode mode,
  const AdaptiveRoughnessMemoryState & memory,
  const NearChainQuality & quality,
  const AdaptiveConstraintPolicyConfig & config) noexcept;

struct DynamicResponseThresholdCandidate
{
  bool eligible{false};
  double normalized_response{std::numeric_limits<double>::quiet_NaN()};
  RoughnessConstraintAction action{RoughnessConstraintAction::kAccepted};
};

struct DynamicResponseThresholdDecision
{
  bool available{false};
  bool intervention_required{false};
  bool fail_open{false};
  double baseline_threshold{std::numeric_limits<double>::quiet_NaN()};
  double requested_threshold{std::numeric_limits<double>::quiet_NaN()};
  double highest_rejected_response{std::numeric_limits<double>::quiet_NaN()};
  double lowest_preserved_response{std::numeric_limits<double>::quiet_NaN()};
  std::size_t rejected_candidate_count{0};
  std::size_t preserved_candidate_count{0};
};

DynamicResponseThresholdDecision deriveDynamicResponseThreshold(
  double baseline_threshold,
  double maximum_threshold,
  double acceptance_tolerance,
  const std::vector<DynamicResponseThresholdCandidate> & candidates) noexcept;

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ADAPTIVE_ROUGHNESS_MEMORY_HPP_
