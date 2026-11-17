#ifndef SLAM_2D_CORE__NEAR_CHAIN_ROTATION_CONSISTENCY_HPP_
#define SLAM_2D_CORE__NEAR_CHAIN_ROTATION_CONSISTENCY_HPP_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace slam_2d_core
{

struct NearChainRotationConsistencyConfig
{
  bool enabled{false};
  std::size_t history_size{64};
  std::size_t minimum_history_batches{8};
  double outlier_sigma{6.0};
  double extreme_yaw_sigma{12.0};
  double yaw_scale_floor_rad{0.006981317007977318};
  double translation_scale_floor_m{0.03};
  double weighted_disagreement_scale_floor{0.25};
  double chain_overlap_trigger{0.75};
  std::int32_t maximum_repeat_state_gap{2};
  std::size_t stable_recovery_batches{3};
};

bool validateNearChainRotationConsistencyConfig(
  const NearChainRotationConsistencyConfig & config,
  std::string * reason = nullptr) noexcept;

struct NearChainRotationCandidateObservation
{
  bool eligible{false};
  double normalized_response{std::numeric_limits<double>::quiet_NaN()};
  double translation_disagreement_m{
    std::numeric_limits<double>::quiet_NaN()};
  double yaw_disagreement_rad{std::numeric_limits<double>::quiet_NaN()};
  double covariance_weighted_disagreement_sq{
    std::numeric_limits<double>::quiet_NaN()};
  std::vector<std::int32_t> chain_state_ids;
};

struct NearChainRotationBatchObservation
{
  std::int64_t stamp_ns{0};
  std::int32_t target_state_id{-1};
  bool memory_available{false};
  double memory_strength{0.0};
  std::vector<NearChainRotationCandidateObservation> candidates;
};

struct NearChainRobustLimit
{
  bool available{false};
  double median{std::numeric_limits<double>::quiet_NaN()};
  double scale{std::numeric_limits<double>::quiet_NaN()};
  double limit{std::numeric_limits<double>::quiet_NaN()};
  double extreme_limit{std::numeric_limits<double>::quiet_NaN()};
};

struct NearChainRotationCandidateDecision
{
  bool evidence_available{false};
  bool candidate_risky{false};
  bool proposed_reject{false};
  bool intervention_reject{false};
  bool fail_open{false};
  double absolute_yaw_disagreement_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double translation_disagreement_m{
    std::numeric_limits<double>::quiet_NaN()};
  double log1p_covariance_weighted_disagreement{
    std::numeric_limits<double>::quiet_NaN()};
  double prior_chain_overlap{0.0};
};

struct NearChainRotationBatchDecision
{
  bool available{false};
  bool timestamp_reset{false};
  bool model_available{false};
  bool memory_trigger{false};
  bool repeated_chain_trigger{false};
  bool latch_active_before{false};
  bool latch_active_after{false};
  bool fail_open{false};
  std::size_t history_count{0};
  NearChainRobustLimit yaw_limit;
  NearChainRobustLimit translation_limit;
  NearChainRobustLimit weighted_disagreement_limit;
  std::vector<NearChainRotationCandidateDecision> candidates;
};

class NearChainRotationConsistency
{
public:
  explicit NearChainRotationConsistency(
    NearChainRotationConsistencyConfig config =
    NearChainRotationConsistencyConfig{});

  void configure(const NearChainRotationConsistencyConfig & config);
  NearChainRotationBatchDecision evaluate(
    const NearChainRotationBatchObservation & observation,
    bool intervention_enabled);
  void reset() noexcept;

  const NearChainRotationConsistencyConfig & config() const noexcept
  {
    return config_;
  }

private:
  NearChainRotationConsistencyConfig config_;
  std::deque<double> yaw_history_rad_;
  std::deque<double> translation_history_m_;
  std::deque<double> weighted_disagreement_history_;
  bool have_last_stamp_{false};
  std::int64_t last_stamp_ns_{0};
  std::int32_t previous_risky_target_state_id_{-1};
  std::vector<std::vector<std::int32_t>> previous_risky_chains_;
  bool latch_active_{false};
  std::size_t stable_batch_count_{0};
};

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__NEAR_CHAIN_ROTATION_CONSISTENCY_HPP_
