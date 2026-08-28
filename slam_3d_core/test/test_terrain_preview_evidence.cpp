// Copyright 2026 Gole Robotics

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "slam_3d_core/terrain_preview_evidence.hpp"

namespace
{

using gole::slam_3d_core::TerrainEvidencePoint;
using gole::slam_3d_core::TerrainLabel;
using gole::slam_3d_core::TerrainPreviewConfig;
using gole::slam_3d_core::TerrainPreviewState;
using gole::slam_3d_core::TerrainPreviewTracker;
using gole::slam_3d_core::TerrainTraversalResponse;
using gole::slam_3d_core::TerrainTraversalSample;

constexpr std::int64_t kSecondNs = 1000000000LL;

TerrainPreviewConfig testConfig()
{
  TerrainPreviewConfig config;
  config.enabled = true;
  config.voxel_size_m = 0.20;
  config.association_radius_m = 0.25;
  config.exit_radius_m = 0.50;
  config.max_vertical_delta_m = 0.30;
  config.max_sample_gap_s = 2.0;
  config.max_swept_segment_length_m = 3.0;
  config.min_ahead_distance_m = 0.50;
  config.max_lateral_distance_m = 0.75;
  config.min_label_confidence = 0.67;
  config.roughness_score_gate = 1.0;
  config.min_passage_samples = 2;
  config.min_passage_duration_s = 0.05;
  return config;
}

TerrainEvidencePoint evidence(
  const TerrainLabel label,
  const bool confirmed = true,
  const double roughness = 0.0)
{
  TerrainEvidencePoint item;
  item.position = Eigen::Vector3d::Zero();
  item.label = label;
  item.confirmations = confirmed ? 2 : 1;
  item.observations = item.confirmations;
  item.confidence = 1.0;
  item.geometric_roughness_score = roughness;
  item.confirmed = confirmed;
  return item;
}

TerrainTraversalSample sample(
  const double stamp_s,
  const double x_m,
  const TerrainTraversalResponse response,
  const bool continuity_break_before = false)
{
  return TerrainTraversalSample{
    static_cast<std::int64_t>(stamp_s * static_cast<double>(kSecondNs)),
    Eigen::Vector3d(x_m, 0.0, 0.0),
    response,
    response == TerrainTraversalResponse::kShock ? 1.5 : 0.2,
    continuity_break_before};
}

void addCrossing(
  TerrainPreviewTracker & tracker,
  const TerrainTraversalResponse response)
{
  EXPECT_TRUE(
    tracker.addTraversalSample(sample(2.0, -1.0, TerrainTraversalResponse::kStable))
    .accepted);
  EXPECT_TRUE(tracker.addTraversalSample(sample(3.0, 1.0, response)).accepted);
  const auto exit_update = tracker.addTraversalSample(
    sample(4.0, 1.1, TerrainTraversalResponse::kStable));
  EXPECT_TRUE(exit_update.accepted);
  EXPECT_EQ(exit_update.finalized_passages, 1U);
}

TEST(TerrainPreviewEvidence, GeometryStateSeparatesTentativeAndConfirmed)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep, false)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());
  ASSERT_EQ(tracker.previewEvidence().size(), 1U);
  EXPECT_EQ(tracker.previewEvidence().front().state, TerrainPreviewState::kTentative);

  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep, true)},
    2 * kSecondNs,
    Eigen::Vector3d(-1.5, 0.0, 0.0),
    Eigen::Vector3d::UnitX());
  ASSERT_EQ(tracker.previewEvidence().size(), 1U);
  EXPECT_EQ(
    tracker.previewEvidence().front().state,
    TerrainPreviewState::kGeometryConfirmed);
}

TEST(TerrainPreviewEvidence, SweptStablePassageConfirmsPreviouslySeenCandidate)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  addCrossing(tracker, TerrainTraversalResponse::kStable);

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kTraversalConfirmed);
  EXPECT_EQ(preview.front().stable_passages, 1);
  EXPECT_EQ(preview.front().shock_passages, 0);
  EXPECT_TRUE(preview.front().first_seen_ahead);
  EXPECT_NEAR(preview.front().first_preview_ahead_distance_m, 2.0, 1.0e-12);
  EXPECT_NEAR(preview.front().first_preview_lead_time_s, 1.0, 1.0e-12);
}

TEST(TerrainPreviewEvidence, SweptShockPassageRetainsRobotResponse)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  EXPECT_TRUE(
    tracker.addTraversalSample(sample(2.0, -1.0, TerrainTraversalResponse::kStable))
    .accepted);
  EXPECT_TRUE(
    tracker.addTraversalSample(sample(3.0, 0.0, TerrainTraversalResponse::kShock))
    .accepted);
  EXPECT_TRUE(
    tracker.addTraversalSample(sample(4.0, 1.0, TerrainTraversalResponse::kStable))
    .accepted);
  const auto exit_update = tracker.addTraversalSample(
    sample(5.0, 1.1, TerrainTraversalResponse::kStable));
  EXPECT_EQ(exit_update.finalized_passages, 1U);

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kTraversalConfirmed);
  EXPECT_EQ(preview.front().shock_passages, 1);
  EXPECT_DOUBLE_EQ(preview.front().empirical_shock_probability, 1.0);
}

TEST(TerrainPreviewEvidence, ShockResponseDoesNotBindAlongEntireSweptSegment)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  EXPECT_TRUE(
    tracker.addTraversalSample(sample(2.0, -1.0, TerrainTraversalResponse::kStable))
    .accepted);
  EXPECT_TRUE(
    tracker.addTraversalSample(sample(3.0, 1.0, TerrainTraversalResponse::kShock))
    .accepted);
  const auto exit_update = tracker.addTraversalSample(
    sample(4.0, 1.1, TerrainTraversalResponse::kStable));

  EXPECT_EQ(exit_update.finalized_passages, 0U);
  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kGeometryConfirmed);
  EXPECT_EQ(preview.front().shock_passages, 0);
}

TEST(TerrainPreviewEvidence, ContinuityBreakDoesNotBridgeQuarantinedMotion)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  EXPECT_TRUE(
    tracker.addTraversalSample(sample(2.0, -1.0, TerrainTraversalResponse::kStable))
    .accepted);
  EXPECT_TRUE(
    tracker.addTraversalSample(
      sample(3.0, 1.0, TerrainTraversalResponse::kStable, true)).accepted);
  const auto exit_update = tracker.addTraversalSample(
    sample(4.0, 1.1, TerrainTraversalResponse::kStable));

  EXPECT_EQ(exit_update.finalized_passages, 0U);
  ASSERT_EQ(tracker.previewEvidence().size(), 1U);
  EXPECT_EQ(
    tracker.previewEvidence().front().state,
    TerrainPreviewState::kGeometryConfirmed);
  EXPECT_EQ(tracker.stats().continuity_breaks, 1U);
}

TEST(TerrainPreviewEvidence, UnknownImuResponseCannotConfirmTraversal)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  EXPECT_TRUE(
    tracker.addTraversalSample(
      sample(2.0, -1.0, TerrainTraversalResponse::kUnknown)).accepted);
  EXPECT_TRUE(
    tracker.addTraversalSample(
      sample(3.0, 1.0, TerrainTraversalResponse::kUnknown)).accepted);
  const auto exit_update = tracker.addTraversalSample(
    sample(4.0, 1.1, TerrainTraversalResponse::kUnknown));

  EXPECT_EQ(exit_update.finalized_passages, 0U);
  ASSERT_EQ(tracker.previewEvidence().size(), 1U);
  EXPECT_EQ(
    tracker.previewEvidence().front().state,
    TerrainPreviewState::kGeometryConfirmed);
}

TEST(TerrainPreviewEvidence, DropTraversalIsSafetyConflictNotConfirmation)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kObservedDrop)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  addCrossing(tracker, TerrainTraversalResponse::kStable);

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kUnsafeTraversalConflict);
  EXPECT_EQ(preview.front().traversal_confirmations, 0);
  EXPECT_EQ(preview.front().unsafe_traversal_conflicts, 1);
}

TEST(TerrainPreviewEvidence, DirectionalLabelFlipClearsOldTraversalSemantics)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());
  addCrossing(tracker, TerrainTraversalResponse::kStable);
  ASSERT_EQ(
    tracker.previewEvidence().front().state,
    TerrainPreviewState::kTraversalConfirmed);

  tracker.updateGeometry(
    {evidence(TerrainLabel::kObservedDrop)},
    5 * kSecondNs,
    Eigen::Vector3d(2.0, 0.0, 0.0),
    -Eigen::Vector3d::UnitX());

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().label, TerrainLabel::kObservedDrop);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kGeometryConfirmed);
  EXPECT_EQ(preview.front().traversal_confirmations, 0);
  EXPECT_EQ(preview.front().first_observed_stamp_ns, 5 * kSecondNs);
  EXPECT_NEAR(preview.front().first_preview_ahead_distance_m, 2.0, 1.0e-12);
}

TEST(TerrainPreviewEvidence, FlatFloorIsFilteredButRoughFloorIsPreviewed)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kFloor, true, 0.5)},
    kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());
  EXPECT_TRUE(tracker.previewEvidence().empty());

  tracker.updateGeometry(
    {evidence(TerrainLabel::kFloor, true, 1.5)},
    2 * kSecondNs,
    Eigen::Vector3d(-2.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());
  ASSERT_EQ(tracker.previewEvidence().size(), 1U);
  EXPECT_DOUBLE_EQ(tracker.previewEvidence().front().predicted_shock_score, 1.5);
  EXPECT_EQ(tracker.stats().roughness_preview_landmarks, 1U);
}

TEST(TerrainPreviewEvidence, DuplicateAndRewoundSamplesAreRejected)
{
  TerrainPreviewTracker tracker(testConfig());
  EXPECT_TRUE(
    tracker.addTraversalSample(
      sample(2.0, -1.0, TerrainTraversalResponse::kStable)).accepted);
  EXPECT_FALSE(
    tracker.addTraversalSample(
      sample(2.0, -0.5, TerrainTraversalResponse::kStable)).accepted);
  EXPECT_FALSE(
    tracker.addTraversalSample(
      sample(1.0, -0.5, TerrainTraversalResponse::kStable)).accepted);
  EXPECT_EQ(tracker.stats().traversal_samples, 1U);
  EXPECT_EQ(tracker.stats().invalid_samples, 2U);
}

TEST(TerrainPreviewEvidence, TraversalFirstSeenBehindDoesNotCountAsPreviewLead)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(1.0, 0.0, 0.0),
    Eigen::Vector3d::UnitX());

  addCrossing(tracker, TerrainTraversalResponse::kStable);

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kTraversalConfirmed);
  EXPECT_FALSE(preview.front().first_seen_ahead);
  EXPECT_EQ(tracker.stats().previewed_before_traversal_landmarks, 0U);
  EXPECT_DOUBLE_EQ(tracker.stats().mean_preview_lead_time_s, 0.0);
}

TEST(TerrainPreviewEvidence, TraversalFirstSeenOutsideForwardCorridorIsNotPreviewLead)
{
  TerrainPreviewTracker tracker(testConfig());
  tracker.updateGeometry(
    {evidence(TerrainLabel::kPositiveStep)},
    kSecondNs,
    Eigen::Vector3d(-2.0, -2.0, 0.0),
    Eigen::Vector3d::UnitX());

  addCrossing(tracker, TerrainTraversalResponse::kStable);

  const auto preview = tracker.previewEvidence();
  ASSERT_EQ(preview.size(), 1U);
  EXPECT_EQ(preview.front().state, TerrainPreviewState::kTraversalConfirmed);
  EXPECT_GT(preview.front().first_preview_ahead_distance_m, 0.0);
  EXPECT_FALSE(preview.front().first_seen_ahead);
  EXPECT_EQ(tracker.stats().previewed_before_traversal_landmarks, 0U);
}

}  // namespace
