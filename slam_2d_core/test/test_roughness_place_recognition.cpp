#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "slam_2d_core/roughness_place_recognition.hpp"

namespace slam_2d_core
{
namespace
{

RoughnessPlaceObservation observation(
  const std::int64_t stamp_ns,
  const std::int32_t state_id,
  const double distance_m,
  const double feature,
  const bool rough)
{
  RoughnessPlaceObservation value;
  value.stamp_ns = stamp_ns;
  value.scan_state_id = state_id;
  value.scan_unique_id = state_id;
  value.odom_x_m = distance_m;
  value.odom_y_m = 0.0;
  value.odom_heading_rad = 0.0;
  value.corrected_x_m = distance_m;
  value.corrected_y_m = 0.0;
  value.corrected_heading_rad = 0.0;
  value.features.fill(feature);
  value.quality_valid = true;
  value.rough = rough;
  return value;
}

RoughnessPlaceDescriptor descriptor(
  const std::vector<double> & distances,
  const std::vector<double> & values,
  const std::size_t bins = 17U)
{
  std::vector<RoughnessPlaceObservation> observations;
  for (std::size_t index = 0U; index < distances.size(); ++index) {
    observations.push_back(
      observation(
        static_cast<std::int64_t>(index + 1U) * 1'000'000'000LL,
        static_cast<std::int32_t>(index),
        distances[index],
        values[index],
        values[index] > 0.5));
  }
  return buildRoughnessPlaceDescriptor(observations, bins);
}

RoughnessPlaceEvent placeEvent(
  const std::uint64_t event_id,
  const std::int64_t start_stamp_ns,
  const std::int64_t end_stamp_ns,
  const std::int32_t pre_state_id,
  const std::int32_t post_state_id,
  RoughnessPlaceDescriptor event_descriptor)
{
  RoughnessPlaceEvent event;
  event.event_id = event_id;
  event.start_stamp_ns = start_stamp_ns;
  event.end_stamp_ns = end_stamp_ns;
  event.pre_anchor_state_id = pre_state_id;
  event.post_anchor_state_id = post_state_id;
  event.descriptor = std::move(event_descriptor);
  return event;
}

RoughnessPlaceDirectionalMatchEvidence directionalEvidence(
  const double response = 0.45,
  const double translation_m = 0.05,
  const double yaw_rad = 0.01,
  const double xy_covariance_m2 = 0.01,
  const double yaw_covariance_rad2 = 0.001)
{
  RoughnessPlaceDirectionalMatchEvidence evidence;
  evidence.available = true;
  evidence.fine_response = response;
  evidence.fine_xy_covariance_max_eigenvalue_m2 = xy_covariance_m2;
  evidence.fine_yaw_covariance_rad2 = yaw_covariance_rad2;
  evidence.proposal_disagreement_translation_m = translation_m;
  evidence.proposal_disagreement_yaw_rad = yaw_rad;
  return evidence;
}

RoughnessPlacePairMatchEvidence goodPairEvidence()
{
  RoughnessPlacePairMatchEvidence evidence;
  evidence.available = true;
  evidence.pre_anchor.available = true;
  evidence.pre_anchor.forward = directionalEvidence(0.46);
  evidence.pre_anchor.reverse = directionalEvidence(0.44);
  evidence.pre_anchor.bidirectional_translation_error_m = 0.01;
  evidence.pre_anchor.bidirectional_yaw_error_rad = 0.005;
  evidence.post_anchor.available = true;
  evidence.post_anchor.forward = directionalEvidence(0.48);
  evidence.post_anchor.reverse = directionalEvidence(0.47);
  evidence.post_anchor.bidirectional_translation_error_m = 0.02;
  evidence.post_anchor.bidirectional_yaw_error_rad = 0.006;
  evidence.forward_alignment_translation_error_m = 0.28;
  evidence.forward_alignment_yaw_error_rad = 0.015;
  evidence.reverse_alignment_translation_error_m = 0.33;
  evidence.reverse_alignment_yaw_error_rad = 0.019;
  return evidence;
}

RoughnessPlacePose2D applyLocalMotion(
  const RoughnessPlacePose2D & pose,
  const double x_m,
  const double y_m,
  const double yaw_rad)
{
  const double cosine = std::cos(pose.yaw_rad);
  const double sine = std::sin(pose.yaw_rad);
  return {
    pose.x_m + cosine * x_m - sine * y_m,
    pose.y_m + sine * x_m + cosine * y_m,
    pose.yaw_rad + yaw_rad,
  };
}

}  // namespace

TEST(RoughnessPlaceRelativeMotionConsistency, IsInvariantToIndependentMapFrames)
{
  const RoughnessPlacePose2D first_pre{10.0, -4.0, 1.2};
  const RoughnessPlacePose2D second_pre{-8.0, 12.0, -2.0};
  const RoughnessPlacePose2D first_post = applyLocalMotion(
    first_pre, 0.30, 0.10, 0.04);
  const RoughnessPlacePose2D second_post = applyLocalMotion(
    second_pre, 0.30, 0.10, 0.04);

  const auto consistency = roughnessPlaceRelativeMotionConsistency(
    first_pre, first_post, second_pre, second_post);

  ASSERT_TRUE(consistency.available);
  EXPECT_NEAR(consistency.translation_error_m, 0.0, 1.0e-12);
  EXPECT_NEAR(consistency.yaw_error_rad, 0.0, 1.0e-12);
}

TEST(RoughnessPlaceRelativeMotionConsistency, MeasuresLocalPassageDisagreement)
{
  const RoughnessPlacePose2D first_pre{3.0, 5.0, 0.7};
  const RoughnessPlacePose2D second_pre{-2.0, 1.0, -1.4};
  const RoughnessPlacePose2D first_post = applyLocalMotion(
    first_pre, 0.30, 0.10, 0.04);
  const RoughnessPlacePose2D second_post = applyLocalMotion(
    second_pre, 0.33, 0.14, -0.01);

  const auto consistency = roughnessPlaceRelativeMotionConsistency(
    first_pre, first_post, second_pre, second_post);

  ASSERT_TRUE(consistency.available);
  EXPECT_NEAR(consistency.translation_error_m, 0.05, 1.0e-12);
  EXPECT_NEAR(consistency.yaw_error_rad, 0.05, 1.0e-12);
}

TEST(RoughnessPlaceRelativeMotionConsistency, NormalizesYawAcrossPi)
{
  constexpr double kPi = 3.14159265358979323846;
  const RoughnessPlacePose2D first_pre{0.0, 0.0, kPi - 0.01};
  const RoughnessPlacePose2D second_pre{1.0, 2.0, -kPi + 0.02};
  const RoughnessPlacePose2D first_post = applyLocalMotion(
    first_pre, 0.2, 0.0, 0.03);
  const RoughnessPlacePose2D second_post = applyLocalMotion(
    second_pre, 0.2, 0.0, 0.03);

  const auto consistency = roughnessPlaceRelativeMotionConsistency(
    first_pre, first_post, second_pre, second_post);

  ASSERT_TRUE(consistency.available);
  EXPECT_NEAR(consistency.translation_error_m, 0.0, 1.0e-12);
  EXPECT_NEAR(consistency.yaw_error_rad, 0.0, 1.0e-12);
}

TEST(RoughnessPlaceRelativeMotionConsistency, MissingPoseFailsOpen)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto consistency = roughnessPlaceRelativeMotionConsistency(
    {0.0, 0.0, 0.0}, {0.1, 0.0, 0.0}, {nan, 0.0, 0.0}, {0.1, 0.0, 0.0});

  EXPECT_FALSE(consistency.available);
  EXPECT_TRUE(std::isnan(consistency.translation_error_m));
  EXPECT_TRUE(std::isnan(consistency.yaw_error_rad));
}

TEST(RoughnessPlaceFeatures, UsesThresholdNormalizedLogCompression)
{
  RoughnessSignal signal;
  signal.tilt_deg = 2.0;
  signal.gyro_xy_rad_s = 0.10;
  signal.accel_norm_deviation_m_s2 = 0.35;
  signal.tilt_rate_deg_s = 10.0;
  signal.accel_norm_rate_m_s3 = 80.0;
  const RoughnessThresholds thresholds;

  const auto features = normalizedRoughnessPlaceFeatures(
    3.0, signal, thresholds);

  EXPECT_DOUBLE_EQ(features[0], std::log(4.0));
  EXPECT_DOUBLE_EQ(features[1], std::log(2.0));
  EXPECT_DOUBLE_EQ(features[2], std::log(3.0));
  EXPECT_DOUBLE_EQ(features[3], std::log(2.0));
  EXPECT_DOUBLE_EQ(features[4], std::log(3.0));
  EXPECT_DOUBLE_EQ(features[5], std::log(2.0));
}

TEST(RoughnessPlaceFeatures, DisabledOrInvalidChannelRemainsUnavailable)
{
  RoughnessSignal signal;
  signal.tilt_deg = std::numeric_limits<double>::quiet_NaN();
  RoughnessThresholds thresholds;
  thresholds.gyro_xy_rad_s = 0.0;

  const auto features = normalizedRoughnessPlaceFeatures(
    1.0, signal, thresholds);

  EXPECT_TRUE(std::isnan(features[1]));
  EXPECT_TRUE(std::isnan(features[2]));
  EXPECT_TRUE(std::isfinite(features[0]));
}

TEST(RoughnessPlaceDescriptor, IsInvariantToSamplingDensityOnDistanceAxis)
{
  const auto sparse = descriptor(
    {0.0, 0.5, 1.0},
    {0.0, 1.0, 0.0});
  const auto dense = descriptor(
    {0.0, 0.25, 0.5, 0.75, 1.0},
    {0.0, 0.5, 1.0, 0.5, 0.0});

  ASSERT_TRUE(sparse.valid());
  ASSERT_TRUE(dense.valid());
  const auto match = matchRoughnessPlaceDescriptors(sparse, dense);
  ASSERT_TRUE(match.valid);
  EXPECT_EQ(match.direction, RoughnessPlaceMatchDirection::kForward);
  EXPECT_NEAR(match.score, 1.0, 1.0e-12);
  EXPECT_NEAR(match.shape_score, 1.0, 1.0e-12);
  EXPECT_NEAR(match.amplitude_score, 1.0, 1.0e-12);
  EXPECT_NEAR(match.length_score, 1.0, 1.0e-12);
}

TEST(RoughnessPlaceDescriptor, SelectsReverseTraversalForAsymmetricProfile)
{
  const auto reference = descriptor(
    {0.0, 0.3, 0.7, 1.0},
    {0.0, 0.2, 1.0, 0.0});
  const auto reverse = descriptor(
    {0.0, 0.3, 0.7, 1.0},
    {0.0, 1.0, 0.2, 0.0});

  const auto match = matchRoughnessPlaceDescriptors(reverse, reference);

  ASSERT_TRUE(match.valid);
  EXPECT_EQ(match.direction, RoughnessPlaceMatchDirection::kReverse);
  EXPECT_NEAR(match.score, 1.0, 1.0e-12);
}

TEST(RoughnessPlaceAnchorPairing, PreservesPreAndPostForForwardTraversal)
{
  RoughnessPlaceEvent reference;
  reference.pre_anchor_state_id = 10;
  reference.post_anchor_state_id = 20;
  RoughnessPlaceEvent query;
  query.pre_anchor_state_id = 110;
  query.post_anchor_state_id = 120;

  const auto pairing = roughnessPlaceAnchorPairing(
    reference, query, RoughnessPlaceMatchDirection::kForward);

  ASSERT_TRUE(pairing.valid);
  EXPECT_EQ(pairing.reference_pre_state_id, 10);
  EXPECT_EQ(pairing.query_for_reference_pre_state_id, 110);
  EXPECT_EQ(pairing.reference_post_state_id, 20);
  EXPECT_EQ(pairing.query_for_reference_post_state_id, 120);
}

TEST(RoughnessPlaceAnchorPairing, SwapsQueryBoundariesForReverseTraversal)
{
  RoughnessPlaceEvent reference;
  reference.pre_anchor_state_id = 10;
  reference.post_anchor_state_id = 20;
  RoughnessPlaceEvent query;
  query.pre_anchor_state_id = 110;
  query.post_anchor_state_id = 120;

  const auto pairing = roughnessPlaceAnchorPairing(
    reference, query, RoughnessPlaceMatchDirection::kReverse);

  ASSERT_TRUE(pairing.valid);
  EXPECT_EQ(pairing.reference_pre_state_id, 10);
  EXPECT_EQ(pairing.query_for_reference_pre_state_id, 120);
  EXPECT_EQ(pairing.reference_post_state_id, 20);
  EXPECT_EQ(pairing.query_for_reference_post_state_id, 110);
}

TEST(RoughnessPlaceAnchorPairing, MissingBoundaryFailsOpen)
{
  RoughnessPlaceEvent reference;
  reference.pre_anchor_state_id = 10;
  reference.post_anchor_state_id = -1;
  RoughnessPlaceEvent query;
  query.pre_anchor_state_id = 110;
  query.post_anchor_state_id = 120;

  EXPECT_FALSE(
    roughnessPlaceAnchorPairing(
      reference, query, RoughnessPlaceMatchDirection::kForward).valid);
}

TEST(RoughnessPlacePassageCorrelation, AdjacentLocalChainsAreCorrelated)
{
  const RoughnessPlaceConstraintPassage first{
    2U, 4U, 137, 139, 224, 226};
  const RoughnessPlaceConstraintPassage second{
    3U, 5U, 140, 142, 227, 229};

  const auto correlation = correlateRoughnessPlaceConstraintPassages(
    first, second);

  ASSERT_TRUE(correlation.valid);
  EXPECT_TRUE(correlation.correlated);
  EXPECT_EQ(correlation.reference_intervening_state_count, 0U);
  EXPECT_EQ(correlation.query_intervening_state_count, 0U);
}

TEST(RoughnessPlacePassageCorrelation, RequiresBothChainsToBeAdjacent)
{
  const RoughnessPlaceConstraintPassage first{
    2U, 4U, 137, 139, 224, 226};
  const RoughnessPlaceConstraintPassage independent_query{
    3U, 6U, 140, 142, 240, 242};

  const auto correlation = correlateRoughnessPlaceConstraintPassages(
    first, independent_query);

  ASSERT_TRUE(correlation.valid);
  EXPECT_FALSE(correlation.correlated);
  EXPECT_EQ(correlation.reference_intervening_state_count, 0U);
  EXPECT_EQ(correlation.query_intervening_state_count, 13U);
}

TEST(RoughnessPlacePassageCorrelation, InvalidPassageFailsOpen)
{
  RoughnessPlaceConstraintPassage invalid;
  const RoughnessPlaceConstraintPassage valid{
    2U, 4U, 137, 139, 224, 226};

  const auto correlation = correlateRoughnessPlaceConstraintPassages(
    invalid, valid);

  EXPECT_FALSE(correlation.valid);
  EXPECT_FALSE(correlation.correlated);
}

TEST(RoughnessPlaceDescriptor, RejectsStationaryEvent)
{
  const auto stationary = descriptor(
    {0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0});

  EXPECT_FALSE(stationary.valid());
}

TEST(RoughnessPlaceEventTracker, RequiresStableAnchorsOnBothSides)
{
  RoughnessPlaceEventTracker tracker(17U, 32U);

  const auto no_pre = tracker.update(
    observation(1'000'000'000LL, 1, 0.2, 1.0, true));
  EXPECT_EQ(
    no_pre.action,
    RoughnessPlaceTrackerAction::kIgnoredWithoutPreAnchor);

  EXPECT_EQ(
    tracker.update(observation(2'000'000'000LL, 2, 0.0, 0.0, false)).action,
    RoughnessPlaceTrackerAction::kStableAnchorUpdated);
  EXPECT_EQ(
    tracker.update(observation(3'000'000'000LL, 3, 0.5, 1.0, true)).action,
    RoughnessPlaceTrackerAction::kStarted);
  EXPECT_TRUE(tracker.active());
  const auto finalized = tracker.update(
    observation(4'000'000'000LL, 4, 1.0, 0.0, false));

  ASSERT_EQ(finalized.action, RoughnessPlaceTrackerAction::kFinalized);
  ASSERT_TRUE(finalized.finalized_event.has_value());
  EXPECT_EQ(finalized.finalized_event->event_id, 1U);
  EXPECT_EQ(finalized.finalized_event->pre_anchor_state_id, 2);
  EXPECT_EQ(finalized.finalized_event->post_anchor_state_id, 4);
  EXPECT_EQ(finalized.finalized_event->rough_observation_count, 1U);
  EXPECT_TRUE(finalized.finalized_event->descriptor.valid());
  EXPECT_NEAR(finalized.finalized_event->descriptor.path_length_m, 1.0, 1.0e-12);
  EXPECT_FALSE(tracker.active());
}

TEST(RoughnessPlaceEventTracker, TimestampRollbackDropsProvisionalEvent)
{
  RoughnessPlaceEventTracker tracker;
  tracker.update(observation(2'000'000'000LL, 1, 0.0, 0.0, false));
  tracker.update(observation(3'000'000'000LL, 2, 0.5, 1.0, true));
  ASSERT_TRUE(tracker.active());

  const auto reset = tracker.update(
    observation(1'000'000'000LL, 3, 1.0, 0.0, false));

  EXPECT_EQ(reset.action, RoughnessPlaceTrackerAction::kTimestampReset);
  EXPECT_FALSE(tracker.active());
}

TEST(RoughnessPlaceCandidate, ReportsBestAndMarginWithoutAcceptanceThreshold)
{
  RoughnessPlaceEvent exact;
  exact.event_id = 7U;
  exact.descriptor = descriptor(
    {0.0, 0.5, 1.0},
    {0.0, 1.0, 0.0});
  RoughnessPlaceEvent different;
  different.event_id = 8U;
  different.descriptor = descriptor(
    {0.0, 0.5, 1.0},
    {0.0, 0.2, 0.0});
  const auto query = descriptor(
    {0.0, 0.25, 0.5, 0.75, 1.0},
    {0.0, 0.5, 1.0, 0.5, 0.0});

  const auto candidate = bestRoughnessPlaceCandidate(
    query, {different, exact});

  ASSERT_TRUE(candidate.valid);
  EXPECT_EQ(candidate.event_id, 7U);
  EXPECT_EQ(candidate.event_index, 1U);
  EXPECT_NEAR(candidate.match.score, 1.0, 1.0e-12);
  EXPECT_TRUE(std::isfinite(candidate.second_best_score));
  EXPECT_GT(candidate.score_margin, 0.0);
}

TEST(RoughnessPlaceCandidate, RankedSearchFiltersRecentAndNearGraphEvents)
{
  const auto exact_descriptor = descriptor(
    {0.0, 0.5, 1.0}, {0.0, 1.0, 0.0});
  const auto weaker_descriptor = descriptor(
    {0.0, 0.5, 1.0}, {0.0, 0.7, 0.0});
  const RoughnessPlaceEvent old_exact = placeEvent(
    1U, 1'000'000'000LL, 2'000'000'000LL, 10, 12,
    exact_descriptor);
  const RoughnessPlaceEvent old_weaker = placeEvent(
    2U, 3'000'000'000LL, 4'000'000'000LL, 20, 22,
    weaker_descriptor);
  const RoughnessPlaceEvent recent_exact = placeEvent(
    3U, 27'000'000'000LL, 28'000'000'000LL, 190, 192,
    exact_descriptor);
  const RoughnessPlaceEvent query = placeEvent(
    4U, 30'000'000'000LL, 31'000'000'000LL, 200, 202,
    exact_descriptor);

  const auto ranked = rankedRoughnessPlaceCandidates(
    query, {old_weaker, recent_exact, old_exact}, 5U, 20.0, 20U);

  ASSERT_EQ(ranked.size(), 2U);
  EXPECT_EQ(ranked[0].event_id, 1U);
  EXPECT_EQ(ranked[1].event_id, 2U);
  EXPECT_NEAR(ranked[0].event_separation_s, 28.0, 1.0e-12);
  EXPECT_EQ(ranked[0].minimum_state_id_separation, 190U);
  EXPECT_GT(ranked[0].score_margin, 0.0);
}

TEST(RoughnessPlaceCandidate, RankedSearchHonorsMaximumCandidateCount)
{
  const auto exact_descriptor = descriptor(
    {0.0, 0.5, 1.0}, {0.0, 1.0, 0.0});
  const RoughnessPlaceEvent query = placeEvent(
    3U, 30'000'000'000LL, 31'000'000'000LL, 200, 202,
    exact_descriptor);
  const auto ranked = rankedRoughnessPlaceCandidates(
    query,
    {
      placeEvent(1U, 1'000'000'000LL, 2'000'000'000LL, 10, 12, exact_descriptor),
      placeEvent(2U, 3'000'000'000LL, 4'000'000'000LL, 20, 22, exact_descriptor),
    },
    1U, 20.0, 20U);

  ASSERT_EQ(ranked.size(), 1U);
  EXPECT_EQ(ranked[0].event_id, 1U);
}

TEST(RoughnessPlaceConstraint, AcceptsOnlyCompleteTwoAnchorAgreement)
{
  const auto decision = evaluateRoughnessPlaceConstraint(
    goodPairEvidence(), RoughnessPlaceConstraintThresholds{});

  EXPECT_TRUE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason, RoughnessPlaceConstraintDecisionReason::kAccepted);
  EXPECT_EQ(decision.selected_anchor_index, 1U);
  EXPECT_NEAR(
    decision.maximum_pair_alignment_translation_error_m, 0.33, 1.0e-12);
}

TEST(RoughnessPlaceConstraint, MissingQualityFailsOpen)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.post_anchor.reverse.available = false;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_TRUE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kMissingEvidence);
}

TEST(RoughnessPlaceConstraint, RejectsWeakDirectionalScanMatch)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.pre_anchor.reverse.fine_response = 0.15;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kFineResponse);
  EXPECT_NEAR(
    decision.maximum_pair_alignment_translation_error_m, 0.33, 1.0e-12);
  EXPECT_NEAR(
    decision.maximum_pair_alignment_yaw_error_rad, 0.019, 1.0e-12);
}

TEST(RoughnessPlaceConstraint, RejectsProposalDisagreement)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.post_anchor.forward.proposal_disagreement_translation_m = 0.31;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kProposalDisagreement);
}

TEST(RoughnessPlaceConstraint, RejectsLargeFineCovariance)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.pre_anchor.forward.fine_xy_covariance_max_eigenvalue_m2 = 0.05;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kFineCovariance);
}

TEST(RoughnessPlaceConstraint, RejectsBidirectionalDisagreement)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.pre_anchor.bidirectional_yaw_error_rad = 0.04;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kBidirectionalConsistency);
}

TEST(RoughnessPlaceConstraint, RejectsPairwiseBranchDisagreement)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.reverse_alignment_translation_error_m = 3.8;
  evidence.reverse_alignment_yaw_error_rad = 0.23;

  const auto decision = evaluateRoughnessPlaceConstraint(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(decision.accepted);
  EXPECT_FALSE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kPairConsistency);
}

TEST(RoughnessPlaceConstraint, InvalidThresholdsFailOpen)
{
  RoughnessPlaceConstraintThresholds thresholds;
  thresholds.minimum_fine_response = 1.1;

  const auto decision = evaluateRoughnessPlaceConstraint(
    goodPairEvidence(), thresholds);

  EXPECT_FALSE(decision.accepted);
  EXPECT_TRUE(decision.fail_open);
  EXPECT_EQ(
    decision.reason,
    RoughnessPlaceConstraintDecisionReason::kInvalidThresholds);
}

TEST(RoughnessPlaceConstraintQuality, ReportsWorstNormalizedGateUtilization)
{
  const RoughnessPlaceConstraintThresholds thresholds;
  const auto quality = roughnessPlaceConstraintQuality(
    goodPairEvidence(), thresholds);

  ASSERT_TRUE(quality.available);
  EXPECT_NEAR(quality.minimum_fine_response, 0.44, 1.0e-12);
  EXPECT_NEAR(
    quality.maximum_gate_utilization,
    thresholds.minimum_fine_response / 0.44,
    1.0e-12);
}

TEST(RoughnessPlaceConstraintQuality, WorseningAnyGateCannotReduceRisk)
{
  const RoughnessPlaceConstraintThresholds thresholds;
  const auto baseline = roughnessPlaceConstraintQuality(
    goodPairEvidence(), thresholds);
  ASSERT_TRUE(baseline.available);

  RoughnessPlacePairMatchEvidence worse = goodPairEvidence();
  worse.post_anchor.forward.proposal_disagreement_translation_m = 0.29;
  const auto worse_quality = roughnessPlaceConstraintQuality(worse, thresholds);

  ASSERT_TRUE(worse_quality.available);
  EXPECT_GE(
    worse_quality.maximum_gate_utilization,
    baseline.maximum_gate_utilization);
  EXPECT_NEAR(worse_quality.maximum_gate_utilization, 0.29 / 0.30, 1.0e-12);
}

TEST(RoughnessPlaceConstraintQuality, MissingEvidenceIsUnavailable)
{
  RoughnessPlacePairMatchEvidence evidence = goodPairEvidence();
  evidence.pre_anchor.forward.available = false;

  const auto quality = roughnessPlaceConstraintQuality(
    evidence, RoughnessPlaceConstraintThresholds{});

  EXPECT_FALSE(quality.available);
}

TEST(RoughnessPlaceConstraintHypothesis, MissingAlternateQualityFailsOpen)
{
  const RoughnessPlaceConstraintThresholds thresholds;
  RoughnessPlaceConstraintHypothesis preferred;
  preferred.direction = RoughnessPlaceMatchDirection::kReverse;
  preferred.decision = evaluateRoughnessPlaceConstraint(
    goodPairEvidence(), thresholds);
  preferred.quality = roughnessPlaceConstraintQuality(
    goodPairEvidence(), thresholds);
  RoughnessPlaceConstraintHypothesis alternate;
  alternate.direction = RoughnessPlaceMatchDirection::kForward;

  const auto selection = selectRoughnessPlaceConstraintHypothesis(
    preferred, alternate);

  EXPECT_FALSE(selection.selected);
  EXPECT_TRUE(selection.fail_open);
  EXPECT_EQ(
    selection.reason,
    RoughnessPlaceConstraintHypothesisSelectionReason::kMissingQuality);
}

TEST(RoughnessPlaceConstraintHypothesis, AcceptedLowerRiskDirectionOverridesDescriptor)
{
  const RoughnessPlaceConstraintThresholds thresholds;
  RoughnessPlacePairMatchEvidence preferred_evidence = goodPairEvidence();
  preferred_evidence.forward_alignment_translation_error_m = 0.45;
  RoughnessPlacePairMatchEvidence alternate_evidence = goodPairEvidence();
  alternate_evidence.forward_alignment_translation_error_m = 0.10;
  alternate_evidence.reverse_alignment_translation_error_m = 0.12;
  RoughnessPlaceConstraintHypothesis preferred;
  preferred.direction = RoughnessPlaceMatchDirection::kReverse;
  preferred.decision = evaluateRoughnessPlaceConstraint(
    preferred_evidence, thresholds);
  preferred.quality = roughnessPlaceConstraintQuality(
    preferred_evidence, thresholds);
  RoughnessPlaceConstraintHypothesis alternate;
  alternate.direction = RoughnessPlaceMatchDirection::kForward;
  alternate.decision = evaluateRoughnessPlaceConstraint(
    alternate_evidence, thresholds);
  alternate.quality = roughnessPlaceConstraintQuality(
    alternate_evidence, thresholds);

  const auto selection = selectRoughnessPlaceConstraintHypothesis(
    preferred, alternate);

  ASSERT_TRUE(selection.selected);
  EXPECT_FALSE(selection.fail_open);
  EXPECT_EQ(selection.selected_index, 1U);
  EXPECT_TRUE(selection.direction_overridden);
  EXPECT_EQ(
    selection.reason,
    RoughnessPlaceConstraintHypothesisSelectionReason::kLowerGateUtilization);
}

TEST(RoughnessPlaceConstraintHypothesis, ExactTieDeterministicallyKeepsPreferred)
{
  const RoughnessPlaceConstraintThresholds thresholds;
  RoughnessPlaceConstraintHypothesis preferred;
  preferred.direction = RoughnessPlaceMatchDirection::kReverse;
  preferred.decision = evaluateRoughnessPlaceConstraint(
    goodPairEvidence(), thresholds);
  preferred.quality = roughnessPlaceConstraintQuality(
    goodPairEvidence(), thresholds);
  RoughnessPlaceConstraintHypothesis alternate = preferred;
  alternate.direction = RoughnessPlaceMatchDirection::kForward;

  const auto first = selectRoughnessPlaceConstraintHypothesis(
    preferred, alternate);
  const auto second = selectRoughnessPlaceConstraintHypothesis(
    preferred, alternate);

  ASSERT_TRUE(first.selected);
  EXPECT_EQ(first.selected_index, 0U);
  EXPECT_FALSE(first.direction_overridden);
  EXPECT_EQ(first.selected_index, second.selected_index);
  EXPECT_EQ(
    first.reason,
    RoughnessPlaceConstraintHypothesisSelectionReason::kPreferredDirectionTie);
}

}  // namespace slam_2d_core
