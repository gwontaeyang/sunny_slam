#ifndef SLAM_2D_CORE__ROUGHNESS_PLACE_RECOGNITION_HPP_
#define SLAM_2D_CORE__ROUGHNESS_PLACE_RECOGNITION_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "slam_2d_core/roughness_innovation_policy.hpp"

namespace slam_2d_core
{

constexpr std::size_t kRoughnessPlaceFeatureCount = 6U;
using RoughnessPlaceFeatures =
  std::array<double, kRoughnessPlaceFeatureCount>;

RoughnessPlaceFeatures normalizedRoughnessPlaceFeatures(
  double roughness_score,
  const RoughnessSignal & signal,
  const RoughnessThresholds & thresholds);

struct RoughnessPlaceObservation
{
  std::int64_t stamp_ns{0};
  std::int32_t scan_state_id{-1};
  std::int32_t scan_unique_id{-1};
  double odom_x_m{std::numeric_limits<double>::quiet_NaN()};
  double odom_y_m{std::numeric_limits<double>::quiet_NaN()};
  double odom_heading_rad{std::numeric_limits<double>::quiet_NaN()};
  double corrected_x_m{std::numeric_limits<double>::quiet_NaN()};
  double corrected_y_m{std::numeric_limits<double>::quiet_NaN()};
  double corrected_heading_rad{std::numeric_limits<double>::quiet_NaN()};
  RoughnessPlaceFeatures features{};
  bool quality_valid{false};
  bool rough{false};
};

struct RoughnessPlaceDescriptor
{
  std::size_t bin_count{0};
  std::size_t source_observation_count{0};
  std::size_t valid_channel_count{0};
  double path_length_m{std::numeric_limits<double>::quiet_NaN()};
  std::array<bool, kRoughnessPlaceFeatureCount> channel_valid{};
  // Bin-major storage: values[bin * feature_count + channel].
  std::vector<double> values;

  bool valid() const noexcept;
  double value(std::size_t bin, std::size_t channel) const noexcept;
};

RoughnessPlaceDescriptor buildRoughnessPlaceDescriptor(
  const std::vector<RoughnessPlaceObservation> & observations,
  std::size_t bin_count);

struct RoughnessPlaceEvent
{
  std::uint64_t event_id{0};
  std::int64_t start_stamp_ns{0};
  std::int64_t end_stamp_ns{0};
  std::int32_t pre_anchor_state_id{-1};
  std::int32_t post_anchor_state_id{-1};
  std::size_t rough_observation_count{0};
  RoughnessPlaceObservation pre_anchor;
  RoughnessPlaceObservation post_anchor;
  RoughnessPlaceDescriptor descriptor;
};

enum class RoughnessPlaceTrackerAction
{
  kNone,
  kStableAnchorUpdated,
  kStarted,
  kExtended,
  kFinalized,
  kIgnoredWithoutPreAnchor,
  kIgnoredInvalid,
  kTimestampReset,
};

const char * roughnessPlaceTrackerActionName(
  RoughnessPlaceTrackerAction action) noexcept;

struct RoughnessPlaceTrackerUpdate
{
  RoughnessPlaceTrackerAction action{RoughnessPlaceTrackerAction::kNone};
  std::optional<RoughnessPlaceEvent> finalized_event;
};

class RoughnessPlaceEventTracker
{
public:
  explicit RoughnessPlaceEventTracker(
    std::size_t descriptor_bin_count = 16U,
    std::size_t maximum_observations_per_event = 512U);

  void configure(
    std::size_t descriptor_bin_count,
    std::size_t maximum_observations_per_event);
  void reset();
  RoughnessPlaceTrackerUpdate update(
    const RoughnessPlaceObservation & observation);
  bool active() const noexcept;

private:
  void appendActiveObservation(const RoughnessPlaceObservation & observation);

  std::size_t descriptor_bin_count_{16U};
  std::size_t maximum_observations_per_event_{512U};
  std::uint64_t next_event_id_{1U};
  std::int64_t last_stamp_ns_{0};
  std::optional<RoughnessPlaceObservation> last_stable_anchor_;
  std::vector<RoughnessPlaceObservation> active_observations_;
  std::size_t active_rough_observation_count_{0U};
};

enum class RoughnessPlaceMatchDirection
{
  kForward,
  kReverse,
};

const char * roughnessPlaceMatchDirectionName(
  RoughnessPlaceMatchDirection direction) noexcept;

struct RoughnessPlaceAnchorPairing
{
  bool valid{false};
  std::int32_t reference_pre_state_id{-1};
  std::int32_t query_for_reference_pre_state_id{-1};
  std::int32_t reference_post_state_id{-1};
  std::int32_t query_for_reference_post_state_id{-1};
};

RoughnessPlaceAnchorPairing roughnessPlaceAnchorPairing(
  const RoughnessPlaceEvent & reference,
  const RoughnessPlaceEvent & query,
  RoughnessPlaceMatchDirection direction) noexcept;

struct RoughnessPlaceConstraintPassage
{
  std::uint64_t reference_event_id{0U};
  std::uint64_t query_event_id{0U};
  std::int32_t reference_first_state_id{-1};
  std::int32_t reference_last_state_id{-1};
  std::int32_t query_first_state_id{-1};
  std::int32_t query_last_state_id{-1};

  bool valid() const noexcept;
};

RoughnessPlaceConstraintPassage roughnessPlaceConstraintPassage(
  const RoughnessPlaceEvent & reference,
  const RoughnessPlaceEvent & query) noexcept;

struct RoughnessPlacePassageCorrelation
{
  bool valid{false};
  bool correlated{false};
  std::size_t reference_intervening_state_count{0U};
  std::size_t query_intervening_state_count{0U};
};

RoughnessPlacePassageCorrelation correlateRoughnessPlaceConstraintPassages(
  const RoughnessPlaceConstraintPassage & left,
  const RoughnessPlaceConstraintPassage & right) noexcept;

struct RoughnessPlaceDescriptorMatch
{
  bool valid{false};
  RoughnessPlaceMatchDirection direction{
    RoughnessPlaceMatchDirection::kForward};
  double score{std::numeric_limits<double>::quiet_NaN()};
  double shape_score{std::numeric_limits<double>::quiet_NaN()};
  double amplitude_score{std::numeric_limits<double>::quiet_NaN()};
  double length_score{std::numeric_limits<double>::quiet_NaN()};
  std::size_t compared_channel_count{0U};
};

RoughnessPlaceDescriptorMatch matchRoughnessPlaceDescriptors(
  const RoughnessPlaceDescriptor & query,
  const RoughnessPlaceDescriptor & reference);

struct RoughnessPlaceCandidate
{
  bool valid{false};
  std::size_t event_index{std::numeric_limits<std::size_t>::max()};
  std::uint64_t event_id{0U};
  RoughnessPlaceDescriptorMatch match;
  double event_separation_s{std::numeric_limits<double>::quiet_NaN()};
  std::size_t minimum_state_id_separation{0U};
  double second_best_score{std::numeric_limits<double>::quiet_NaN()};
  double score_margin{std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceCandidate bestRoughnessPlaceCandidate(
  const RoughnessPlaceDescriptor & query,
  const std::vector<RoughnessPlaceEvent> & references);

std::vector<RoughnessPlaceCandidate> rankedRoughnessPlaceCandidates(
  const RoughnessPlaceEvent & query,
  const std::vector<RoughnessPlaceEvent> & references,
  std::size_t maximum_candidates,
  double minimum_event_separation_s,
  std::size_t minimum_state_id_separation);

struct RoughnessPlacePose2D
{
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_rad{std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessPlaceRelativeMotionConsistency
{
  bool available{false};
  double translation_error_m{std::numeric_limits<double>::quiet_NaN()};
  double yaw_error_rad{std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceRelativeMotionConsistency roughnessPlaceRelativeMotionConsistency(
  const RoughnessPlacePose2D & first_pre,
  const RoughnessPlacePose2D & first_post,
  const RoughnessPlacePose2D & second_pre,
  const RoughnessPlacePose2D & second_post) noexcept;

struct RoughnessPlaceDirectionalMatchEvidence
{
  bool available{false};
  double fine_response{std::numeric_limits<double>::quiet_NaN()};
  double fine_xy_covariance_max_eigenvalue_m2{
    std::numeric_limits<double>::quiet_NaN()};
  double fine_yaw_covariance_rad2{
    std::numeric_limits<double>::quiet_NaN()};
  double proposal_disagreement_translation_m{
    std::numeric_limits<double>::quiet_NaN()};
  double proposal_disagreement_yaw_rad{
    std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessPlaceAnchorMatchEvidence
{
  bool available{false};
  RoughnessPlaceDirectionalMatchEvidence forward;
  RoughnessPlaceDirectionalMatchEvidence reverse;
  double bidirectional_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double bidirectional_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessPlacePairMatchEvidence
{
  bool available{false};
  RoughnessPlaceAnchorMatchEvidence pre_anchor;
  RoughnessPlaceAnchorMatchEvidence post_anchor;
  double forward_alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double forward_alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
  double reverse_alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double reverse_alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessPlaceConstraintThresholds
{
  double minimum_fine_response{0.30};
  double maximum_proposal_disagreement_translation_m{0.30};
  double maximum_proposal_disagreement_yaw_rad{0.08726646259971647};
  double maximum_fine_xy_covariance_m2{0.04};
  double maximum_fine_yaw_covariance_rad2{0.007615435494667714};
  double maximum_bidirectional_translation_error_m{0.10};
  double maximum_bidirectional_yaw_error_rad{0.03490658503988659};
  double maximum_pair_alignment_translation_error_m{0.50};
  double maximum_pair_alignment_yaw_error_rad{0.05235987755982989};
};

bool validRoughnessPlaceConstraintThresholds(
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept;

enum class RoughnessPlaceConstraintDecisionReason
{
  kAccepted,
  kMissingEvidence,
  kFineResponse,
  kProposalDisagreement,
  kFineCovariance,
  kBidirectionalConsistency,
  kPairConsistency,
  kInvalidThresholds,
};

const char * roughnessPlaceConstraintDecisionReasonName(
  RoughnessPlaceConstraintDecisionReason reason) noexcept;

struct RoughnessPlaceConstraintDecision
{
  bool accepted{false};
  bool fail_open{true};
  RoughnessPlaceConstraintDecisionReason reason{
    RoughnessPlaceConstraintDecisionReason::kMissingEvidence};
  std::size_t selected_anchor_index{0U};
  double selected_minimum_fine_response{
    std::numeric_limits<double>::quiet_NaN()};
  double maximum_pair_alignment_translation_error_m{
    std::numeric_limits<double>::quiet_NaN()};
  double maximum_pair_alignment_yaw_error_rad{
    std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceConstraintDecision evaluateRoughnessPlaceConstraint(
  const RoughnessPlacePairMatchEvidence & evidence,
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept;

struct RoughnessPlaceConstraintQuality
{
  bool available{false};
  double maximum_gate_utilization{
    std::numeric_limits<double>::quiet_NaN()};
  double minimum_fine_response{
    std::numeric_limits<double>::quiet_NaN()};
};

RoughnessPlaceConstraintQuality roughnessPlaceConstraintQuality(
  const RoughnessPlacePairMatchEvidence & evidence,
  const RoughnessPlaceConstraintThresholds & thresholds) noexcept;

struct RoughnessPlaceConstraintHypothesis
{
  RoughnessPlaceMatchDirection direction{
    RoughnessPlaceMatchDirection::kForward};
  RoughnessPlaceConstraintDecision decision;
  RoughnessPlaceConstraintQuality quality;
};

enum class RoughnessPlaceConstraintHypothesisSelectionReason
{
  kMissingQuality,
  kNoAcceptedHypothesis,
  kOnlyAcceptedHypothesis,
  kLowerGateUtilization,
  kHigherFineResponse,
  kPreferredDirectionTie,
};

const char * roughnessPlaceConstraintHypothesisSelectionReasonName(
  RoughnessPlaceConstraintHypothesisSelectionReason reason) noexcept;

struct RoughnessPlaceConstraintHypothesisSelection
{
  bool selected{false};
  bool fail_open{true};
  std::size_t selected_index{0U};
  bool direction_overridden{false};
  RoughnessPlaceConstraintHypothesisSelectionReason reason{
    RoughnessPlaceConstraintHypothesisSelectionReason::kMissingQuality};
};

// The first hypothesis is the descriptor-preferred direction. Missing quality
// in either direction fails open so the observer cannot inject a one-sided edge.
RoughnessPlaceConstraintHypothesisSelection
selectRoughnessPlaceConstraintHypothesis(
  const RoughnessPlaceConstraintHypothesis & preferred,
  const RoughnessPlaceConstraintHypothesis & alternate) noexcept;

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_PLACE_RECOGNITION_HPP_
