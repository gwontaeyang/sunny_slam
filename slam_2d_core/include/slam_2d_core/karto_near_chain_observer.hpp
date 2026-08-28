#ifndef SLAM_2D_CORE__KARTO_NEAR_CHAIN_OBSERVER_HPP_
#define SLAM_2D_CORE__KARTO_NEAR_CHAIN_OBSERVER_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "slam_2d_core/adaptive_roughness_memory.hpp"
#include "slam_2d_core/karto_scan_matcher_vector.hpp"

namespace slam_2d_core
{

struct KartoNearChainCandidate
{
  std::size_t candidate_index{0};
  std::string sensor_name;
  std::vector<std::int32_t> chain_state_ids;
  std::vector<std::int32_t> chain_unique_ids;
  std::int32_t source_state_id{-1};
  std::int32_t source_unique_id{-1};
  std::int32_t target_state_id{-1};
  std::int32_t target_unique_id{-1};
  double normalized_response{std::numeric_limits<double>::quiet_NaN()};
  karto::Pose2 matched_sensor_pose;
  karto::Pose2 matched_target_pose;
  karto::Pose2 sequential_matched_target_pose;
  karto::Pose2 relative_transform;
  karto::Matrix3 covariance;
  karto::Matrix3 factor_covariance;
  double translation_disagreement_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_disagreement_rad{std::numeric_limits<double>::quiet_NaN()};
  double xy_covariance_max_eigenvalue_m2{
    std::numeric_limits<double>::quiet_NaN()};
  double covariance_weighted_disagreement_sq{
    std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_x_m{std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_y_m{std::numeric_limits<double>::quiet_NaN()};
  double pre_residual_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_x_m{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_y_m{std::numeric_limits<double>::quiet_NaN()};
  double post_residual_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  double post_covariance_weighted_residual_sq{
    std::numeric_limits<double>::quiet_NaN()};
  double source_distance_m{std::numeric_limits<double>::quiet_NaN()};
  bool chain_size_accepted{false};
  bool baseline_response_accepted{false};
  bool applied_response_accepted{false};
  bool distance_accepted{false};
  bool prelinked_by_sequential_or_running{false};
  bool baseline_pose_fusion_used{false};
  bool actual_edge_present{false};
  bool outcome_parity{false};

  NearChainQualityObservation qualityObservation() const noexcept;
};

struct KartoNearChainObservationBatch
{
  bool available{false};
  bool target_id_prediction_reliable{false};
  bool candidate_endpoints_unique{true};
  bool new_edge_accounting_complete{false};
  bool outcome_parity{false};
  std::string failure_reason;
  std::int32_t predicted_target_state_id{-1};
  std::int32_t predicted_target_unique_id{-1};
  double baseline_response_threshold{
    std::numeric_limits<double>::quiet_NaN()};
  double applied_response_threshold{
    std::numeric_limits<double>::quiet_NaN()};
  std::size_t eligible_chain_count{0};
  std::size_t baseline_pose_fusion_count{0};
  std::size_t applied_pose_fusion_count{0};
  std::size_t actual_candidate_edge_count{0};
  std::size_t new_edge_count{0};
  std::size_t classified_new_edge_count{0};
  std::size_t unexpected_new_edge_count{0};
  std::vector<std::int32_t> prelinked_source_unique_ids;
  std::vector<KartoNearChainCandidate> candidates;
};

struct NearChainEdgeEndpoint
{
  std::int32_t source_unique_id{-1};
  std::int32_t target_unique_id{-1};
  bool link_info_available{false};
};

struct NearChainEdgeAccounting
{
  bool complete{false};
  bool candidate_endpoints_unique{false};
  std::size_t classified_edge_count{0};
  std::size_t candidate_edge_count{0};
  std::size_t unexpected_edge_count{0};
};

NearChainEdgeAccounting evaluateNearChainEdgeAccounting(
  std::int32_t target_unique_id,
  const std::vector<std::int32_t> & candidate_source_unique_ids,
  const std::vector<std::int32_t> & prelinked_source_unique_ids,
  const std::vector<std::int32_t> & externally_classified_source_unique_ids,
  const std::vector<NearChainEdgeEndpoint> & new_edges) noexcept;

KartoNearChainObservationBatch observeKartoNearChainCandidates(
  karto::Mapper * mapper,
  karto::LocalizedRangeScan * current_scan,
  karto::LocalizedRangeScan * previous_scan,
  const karto::LocalizedRangeScanVector & running_scans);

void updateKartoNearChainCandidateOutcomes(
  KartoNearChainObservationBatch & batch,
  karto::Mapper * mapper,
  const karto::LocalizedRangeScan * processed_scan,
  double applied_response_threshold,
  std::size_t edge_count_before,
  const std::vector<std::int32_t> & externally_classified_source_unique_ids)
noexcept;

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__KARTO_NEAR_CHAIN_OBSERVER_HPP_
