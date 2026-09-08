#include "slam_3d_core/registration_policy.hpp"

#include "gtest/gtest.h"

namespace
{

gole::slam_3d_core::MappingCloudRefinementConfig mappingRefinementConfig()
{
  gole::slam_3d_core::MappingCloudRefinementConfig config;
  config.enabled = true;
  config.min_source_points = 200;
  config.min_target_points = 1000;
  config.min_inlier_ratio = 0.18;
  config.min_target_coverage_ratio = 0.10;
  config.max_inlier_rmse_m = 0.30;
  config.max_translation_m = 0.75;
  config.max_rotation_deg = 5.0;
  config.min_observable_rank = 6;
  config.max_hessian_condition = 20000.0;
  config.min_score_improvement_ratio = 0.05;
  config.max_primary_score_regression_ratio = 0.05;
  return config;
}

gole::slam_3d_core::MappingCloudRefinementMetrics mappingRefinementMetrics()
{
  gole::slam_3d_core::MappingCloudRefinementMetrics metrics;
  metrics.registration_accepted = true;
  metrics.source_points = 900;
  metrics.target_points = 12000;
  metrics.before_inlier_ratio = 0.22;
  metrics.before_inlier_rmse_m = 0.24;
  metrics.after_inlier_ratio = 0.29;
  metrics.after_inlier_rmse_m = 0.16;
  metrics.before_target_coverage_ratio = 0.14;
  metrics.after_target_coverage_ratio = 0.18;
  metrics.correction_translation_m = 0.42;
  metrics.correction_rotation_deg = 2.1;
  metrics.observable_rank = 6;
  metrics.hessian_condition = 8000.0;
  metrics.primary_before_inlier_ratio = 0.62;
  metrics.primary_before_inlier_rmse_m = 0.20;
  metrics.primary_after_inlier_ratio = 0.61;
  metrics.primary_after_inlier_rmse_m = 0.20;
  return metrics;
}

gole::slam_3d_core::MappingCloudVerifiedCoarseConfig verifiedCoarseConfig()
{
  gole::slam_3d_core::MappingCloudVerifiedCoarseConfig config;
  config.enabled = true;
  config.min_inlier_ratio = 0.90;
  config.max_inlier_rmse_m = 0.20;
  config.min_target_coverage_ratio = 0.70;
  config.max_score_regression_ratio = 0.03;
  return config;
}

gole::slam_3d_core::PoseGraphMinimumIntervalOverrideConfig intervalOverrideConfig()
{
  gole::slam_3d_core::PoseGraphMinimumIntervalOverrideConfig config;
  config.enabled = true;
  config.max_descriptor_distance = 0.12;
  config.min_inlier_ratio = 0.90;
  config.max_inlier_rmse_m = 0.15;
  config.min_target_coverage_ratio = 0.50;
  return config;
}

gole::slam_3d_core::PoseGraphMinimumIntervalMetrics intervalMetrics()
{
  gole::slam_3d_core::PoseGraphMinimumIntervalMetrics metrics;
  metrics.have_previous_correction = true;
  metrics.elapsed_since_previous_correction_s = 22.0;
  metrics.mapping_refinement_verified = true;
  metrics.descriptor_distance = 0.09;
  metrics.inlier_ratio = 0.98;
  metrics.inlier_rmse_m = 0.05;
  metrics.target_coverage_ratio = 0.66;
  return metrics;
}

gole::slam_3d_core::PoseGraphDuplicateSubmapPairOverrideConfig duplicatePairConfig()
{
  gole::slam_3d_core::PoseGraphDuplicateSubmapPairOverrideConfig config;
  config.enabled = true;
  config.max_edges = 2;
  return config;
}

TEST(RegistrationPolicy, PoseGraphMinimumIntervalAllowsInactiveGate)
{
  auto metrics = intervalMetrics();
  metrics.have_previous_correction = false;

  const auto decision = gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
    30.0, intervalOverrideConfig(), metrics);

  EXPECT_TRUE(decision.allow);
  EXPECT_FALSE(decision.interval_active);
  EXPECT_FALSE(decision.override_applied);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kInactive);
}

TEST(RegistrationPolicy, PoseGraphMinimumIntervalRejectsWhenOverrideDisabled)
{
  auto config = intervalOverrideConfig();
  config.enabled = false;

  const auto decision = gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
    30.0, config, intervalMetrics());

  EXPECT_FALSE(decision.allow);
  EXPECT_TRUE(decision.interval_active);
  EXPECT_FALSE(decision.override_applied);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kOverrideDisabled);
}

TEST(RegistrationPolicy, PoseGraphMinimumIntervalAllowsConfirmedHighConfidenceLoop)
{
  const auto decision = gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
    30.0, intervalOverrideConfig(), intervalMetrics());

  EXPECT_TRUE(decision.allow);
  EXPECT_TRUE(decision.interval_active);
  EXPECT_TRUE(decision.override_applied);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kOverrideAccepted);
}

TEST(RegistrationPolicy, HighConfidenceConstraintUsesIndependentGeometryChecks)
{
  const auto config = intervalOverrideConfig();
  const auto metrics = intervalMetrics();

  EXPECT_TRUE(gole::slam_3d_core::isHighConfidencePoseGraphConstraint(config, metrics));

  auto no_refinement = metrics;
  no_refinement.mapping_refinement_verified = false;
  EXPECT_FALSE(
    gole::slam_3d_core::isHighConfidencePoseGraphConstraint(config, no_refinement));

  auto low_coverage = metrics;
  low_coverage.target_coverage_ratio = 0.49;
  EXPECT_FALSE(
    gole::slam_3d_core::isHighConfidencePoseGraphConstraint(config, low_coverage));
}

TEST(RegistrationPolicy, HighConfidenceConstraintAcceptsMeasuredReturnQuality)
{
  auto metrics = intervalMetrics();
  metrics.descriptor_distance = 0.0567289;
  metrics.inlier_ratio = 0.980683;
  metrics.inlier_rmse_m = 0.115993;
  metrics.target_coverage_ratio = 0.849799;

  EXPECT_TRUE(
    gole::slam_3d_core::isHighConfidencePoseGraphConstraint(
      intervalOverrideConfig(), metrics));
}

TEST(RegistrationPolicy, PoseGraphMinimumIntervalRequiresEveryIndependentQualityGate)
{
  const auto config = intervalOverrideConfig();

  auto no_refinement = intervalMetrics();
  no_refinement.mapping_refinement_verified = false;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
      30.0, config, no_refinement).reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kMappingRefinementRequired);

  auto distant_descriptor = intervalMetrics();
  distant_descriptor.descriptor_distance = 0.13;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
      30.0, config, distant_descriptor).reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kDescriptorDistance);

  auto low_inlier = intervalMetrics();
  low_inlier.inlier_ratio = 0.89;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
      30.0, config, low_inlier).reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kLowInlierRatio);

  auto high_rmse = intervalMetrics();
  high_rmse.inlier_rmse_m = 0.16;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
      30.0, config, high_rmse).reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kHighInlierRmse);

  auto low_coverage = intervalMetrics();
  low_coverage.target_coverage_ratio = 0.49;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
      30.0, config, low_coverage).reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kLowTargetCoverage);
}

TEST(RegistrationPolicy, PoseGraphMinimumIntervalDoesNotOverrideClockRewind)
{
  auto metrics = intervalMetrics();
  metrics.elapsed_since_previous_correction_s = -0.1;

  const auto decision = gole::slam_3d_core::evaluatePoseGraphMinimumInterval(
    30.0, intervalOverrideConfig(), metrics);

  EXPECT_FALSE(decision.allow);
  EXPECT_FALSE(decision.override_applied);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphMinimumIntervalReason::kInvalidElapsedTime);
}

TEST(RegistrationPolicy, DuplicateSubmapPairAllowsFirstObservation)
{
  gole::slam_3d_core::PoseGraphDuplicateSubmapPairMetrics metrics;

  const auto decision = gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
    duplicatePairConfig(), metrics);

  EXPECT_TRUE(decision.allow);
  EXPECT_FALSE(decision.override_applied);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kFirstObservation);
}

TEST(RegistrationPolicy, DuplicateSubmapPairRequiresExplicitHighConfidenceOverride)
{
  gole::slam_3d_core::PoseGraphDuplicateSubmapPairMetrics metrics;
  metrics.accepted_edges = 1;

  auto disabled_config = duplicatePairConfig();
  disabled_config.enabled = false;
  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
      disabled_config, metrics).reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kOverrideDisabled);

  EXPECT_EQ(
    gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
      duplicatePairConfig(), metrics).reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kHighConfidenceRequired);
}

TEST(RegistrationPolicy, DuplicateSubmapPairRejectsRepeatedTargetAnchor)
{
  gole::slam_3d_core::PoseGraphDuplicateSubmapPairMetrics metrics;
  metrics.accepted_edges = 1;
  metrics.high_confidence_interval_override = true;
  metrics.target_anchor_already_used = true;

  const auto decision = gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
    duplicatePairConfig(), metrics);

  EXPECT_FALSE(decision.allow);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kTargetAnchorAlreadyUsed);
}

TEST(RegistrationPolicy, DuplicateSubmapPairCapsDistinctConfirmedAnchors)
{
  gole::slam_3d_core::PoseGraphDuplicateSubmapPairMetrics metrics;
  metrics.accepted_edges = 1;
  metrics.high_confidence_interval_override = true;

  const auto accepted = gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
    duplicatePairConfig(), metrics);
  EXPECT_TRUE(accepted.allow);
  EXPECT_TRUE(accepted.override_applied);
  EXPECT_EQ(
    accepted.reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kOverrideAccepted);

  metrics.accepted_edges = 2;
  const auto capped = gole::slam_3d_core::evaluatePoseGraphDuplicateSubmapPair(
    duplicatePairConfig(), metrics);
  EXPECT_FALSE(capped.allow);
  EXPECT_EQ(
    capped.reason,
    gole::slam_3d_core::PoseGraphDuplicateSubmapPairReason::kEdgeLimit);
}

TEST(RegistrationPolicy, SameLayerConstraintKeepsExistingVerticalRecoveryMode)
{
  const auto selection = gole::slam_3d_core::selectPoseGraphSameLayerConstraint(
    true, 3.0, false, false, 1.5);

  EXPECT_TRUE(selection.enabled);
  EXPECT_DOUBLE_EQ(selection.max_horizontal_distance_m, 3.0);
  EXPECT_EQ(
    selection.mode,
    gole::slam_3d_core::PoseGraphSameLayerConstraintMode::
    kRegistrationOnlyVerticalRecovery);
}

TEST(RegistrationPolicy, SameLayerConstraintRequiresExplicitSingleFloorReturnMode)
{
  const auto disabled = gole::slam_3d_core::selectPoseGraphSameLayerConstraint(
    false, 3.0, true, false, 1.5);
  const auto enabled = gole::slam_3d_core::selectPoseGraphSameLayerConstraint(
    false, 3.0, true, true, 1.5);

  EXPECT_FALSE(disabled.enabled);
  EXPECT_TRUE(enabled.enabled);
  EXPECT_DOUBLE_EQ(enabled.max_horizontal_distance_m, 1.5);
  EXPECT_EQ(
    enabled.mode,
    gole::slam_3d_core::PoseGraphSameLayerConstraintMode::
    kHighConfidenceSingleFloorReturn);
}

TEST(RegistrationPolicy, SameLayerConstraintPrioritizesBoundedSingleFloorReturnGate)
{
  const auto selection = gole::slam_3d_core::selectPoseGraphSameLayerConstraint(
    true, 3.0, true, true, 1.25);

  EXPECT_TRUE(selection.enabled);
  EXPECT_DOUBLE_EQ(selection.max_horizontal_distance_m, 1.25);
  EXPECT_EQ(
    selection.mode,
    gole::slam_3d_core::PoseGraphSameLayerConstraintMode::
    kHighConfidenceSingleFloorReturn);
}

TEST(RegistrationPolicy, SameLayerVerticalWeightOnlyStrengthensConfirmedSingleFloorReturn)
{
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePoseGraphVerticalTranslationWeight(2.0, 4.0, true),
    8.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePoseGraphVerticalTranslationWeight(2.0, 4.0, false),
    2.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePoseGraphVerticalTranslationWeight(2.0, 0.5, true),
    2.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePoseGraphVerticalTranslationWeight(
      2.0, std::numeric_limits<double>::quiet_NaN(), true),
    2.0);
}

TEST(RegistrationPolicy, WeightedTranslationErrorUsesMapAxes)
{
  const Eigen::Vector3d expected_map_translation(4.0, -3.0, 1.0);
  const Eigen::Vector3d actual_map_translation(5.0, -1.0, 4.0);

  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::weightedMapTranslationSquaredError(
      expected_map_translation, actual_map_translation, 2.0, 8.0),
    82.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::weightedMapTranslationSquaredError(
      expected_map_translation, actual_map_translation, -1.0, 8.0),
    72.0);
  EXPECT_TRUE(
    std::isinf(
      gole::slam_3d_core::weightedMapTranslationSquaredError(
        Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN()),
        actual_map_translation, 2.0, 8.0)));
}

TEST(RegistrationPolicy, SameLayerReferenceUsesNearestPoseInsideMatchedSubmap)
{
  const std::vector<gole::slam_3d_core::SameLayerReferenceCandidate> candidates{
    {37, 10.0, 0.0, 4.2},
    {35, 1.1, 1.1, 0.3},
    {36, 1.8, 1.5, 0.4}};

  const auto selection = gole::slam_3d_core::selectNearestSameLayerReference(
    1.0, 1.0, 3.0, candidates);

  ASSERT_TRUE(selection.found);
  EXPECT_TRUE(selection.within_max_distance);
  EXPECT_EQ(selection.keyframe_index, 35);
  EXPECT_NEAR(selection.horizontal_distance_m, std::sqrt(0.02), 1.0e-12);
  EXPECT_DOUBLE_EQ(selection.z_m, 0.3);
}

TEST(RegistrationPolicy, SameLayerReferenceReportsNearestPoseOutsideGate)
{
  const std::vector<gole::slam_3d_core::SameLayerReferenceCandidate> candidates{
    {8, 5.0, 0.0, 1.0},
    {9, 8.0, 0.0, 2.0}};

  const auto selection = gole::slam_3d_core::selectNearestSameLayerReference(
    0.0, 0.0, 3.0, candidates);

  ASSERT_TRUE(selection.found);
  EXPECT_FALSE(selection.within_max_distance);
  EXPECT_EQ(selection.keyframe_index, 8);
  EXPECT_DOUBLE_EQ(selection.horizontal_distance_m, 5.0);
}

TEST(RegistrationPolicy, SameLayerReferenceIgnoresInvalidCandidatesAndDisabledGate)
{
  const std::vector<gole::slam_3d_core::SameLayerReferenceCandidate> candidates{
    {-1, 0.0, 0.0, 0.0},
    {4, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
    {5, 0.2, 0.1, 0.7}};

  const auto selection = gole::slam_3d_core::selectNearestSameLayerReference(
    0.0, 0.0, 0.0, candidates);

  ASSERT_TRUE(selection.found);
  EXPECT_FALSE(selection.within_max_distance);
  EXPECT_EQ(selection.keyframe_index, 5);
}

TEST(RegistrationPolicy, SameLayerZCorrectionPreservesKeyframeToCurrentVerticalMotion)
{
  Eigen::Isometry3d matched_current_pose = Eigen::Isometry3d::Identity();
  matched_current_pose.translation() = Eigen::Vector3d(2.7, -7.9, -4.7);
  Eigen::Isometry3d graph_target_pose = Eigen::Isometry3d::Identity();
  graph_target_pose.translation() = Eigen::Vector3d(2.1, -12.1, -4.8);
  graph_target_pose.linear() =
    Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto correction =
    gole::slam_3d_core::transportSameLayerZCorrectionToGraphTarget(
    matched_current_pose, -0.4, graph_target_pose);

  ASSERT_TRUE(correction.valid);
  EXPECT_NEAR(correction.requested_z_correction_m, 4.3, 1.0e-12);
  EXPECT_NEAR(correction.graph_target_pose.translation().z(), -0.5, 1.0e-12);
  EXPECT_DOUBLE_EQ(
    correction.graph_target_pose.translation().x(),
    graph_target_pose.translation().x());
  EXPECT_DOUBLE_EQ(
    correction.graph_target_pose.translation().y(),
    graph_target_pose.translation().y());
  EXPECT_TRUE(
    correction.graph_target_pose.linear().isApprox(
      graph_target_pose.linear(), 1.0e-12));
  EXPECT_NEAR(
    correction.graph_target_pose.translation().z() - (-0.4),
    graph_target_pose.translation().z() - matched_current_pose.translation().z(),
    1.0e-12);
}

TEST(RegistrationPolicy, SameLayerPoseGraphSeedDistributesZByHorizontalTravel)
{
  std::vector<Eigen::Isometry3d> poses(5, Eigen::Isometry3d::Identity());
  poses[0].translation() = Eigen::Vector3d(0.0, 0.0, 0.0);
  poses[1].translation() = Eigen::Vector3d(1.0, 0.0, -0.1);
  poses[2].translation() = Eigen::Vector3d(2.0, 0.0, -1.1);
  poses[3].translation() = Eigen::Vector3d(4.0, 0.0, -2.1);
  poses[4].translation() = Eigen::Vector3d(7.0, 0.0, -4.1);
  poses[4].linear() =
    Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const auto original = poses;

  const auto stats = gole::slam_3d_core::seedSameLayerPoseGraphZCorrection(
    poses, 1, -0.1);

  ASSERT_TRUE(stats.valid);
  EXPECT_TRUE(stats.applied);
  EXPECT_NEAR(stats.requested_endpoint_correction_m, 4.0, 1.0e-12);
  EXPECT_NEAR(stats.horizontal_path_length_m, 6.0, 1.0e-12);
  EXPECT_NEAR(stats.max_z_increment_m, 2.0, 1.0e-12);
  EXPECT_TRUE(poses[0].matrix().isApprox(original[0].matrix(), 1.0e-12));
  EXPECT_TRUE(poses[1].matrix().isApprox(original[1].matrix(), 1.0e-12));
  EXPECT_NEAR(poses[2].translation().z(), -1.1 + 4.0 / 6.0, 1.0e-12);
  EXPECT_NEAR(poses[3].translation().z(), -2.1 + 2.0, 1.0e-12);
  EXPECT_NEAR(poses[4].translation().z(), -0.1, 1.0e-12);
  EXPECT_DOUBLE_EQ(poses[4].translation().x(), original[4].translation().x());
  EXPECT_DOUBLE_EQ(poses[4].translation().y(), original[4].translation().y());
  EXPECT_TRUE(poses[4].linear().isApprox(original[4].linear(), 1.0e-12));
}

TEST(RegistrationPolicy, SameLayerPoseGraphSeedRejectsInvalidAnchorWithoutMutation)
{
  std::vector<Eigen::Isometry3d> poses(2, Eigen::Isometry3d::Identity());
  const auto original = poses;

  const auto stats = gole::slam_3d_core::seedSameLayerPoseGraphZCorrection(
    poses, 1, 0.0);

  EXPECT_FALSE(stats.valid);
  EXPECT_FALSE(stats.applied);
  EXPECT_TRUE(poses[0].matrix().isApprox(original[0].matrix(), 1.0e-12));
  EXPECT_TRUE(poses[1].matrix().isApprox(original[1].matrix(), 1.0e-12));
}

TEST(RegistrationPolicy, SameLayerPoseGraphSeedFallsBackToNodeIndexWithoutHorizontalTravel)
{
  std::vector<Eigen::Isometry3d> poses(4, Eigen::Isometry3d::Identity());
  poses[1].translation().z() = -1.0;
  poses[2].translation().z() = -2.0;
  poses[3].translation().z() = -3.0;

  const auto stats = gole::slam_3d_core::seedSameLayerPoseGraphZCorrection(
    poses, 0, 0.0);

  ASSERT_TRUE(stats.valid);
  EXPECT_TRUE(stats.applied);
  EXPECT_DOUBLE_EQ(stats.horizontal_path_length_m, 0.0);
  EXPECT_NEAR(stats.max_z_increment_m, 1.0, 1.0e-12);
  for (const auto & pose : poses) {
    EXPECT_NEAR(pose.translation().z(), 0.0, 1.0e-12);
  }
}

TEST(RegistrationPolicy, SameLayerPoseGraphSeedRejectsNonfinitePoseBeforeMutation)
{
  std::vector<Eigen::Isometry3d> poses(3, Eigen::Isometry3d::Identity());
  poses[1].translation().x() = std::numeric_limits<double>::quiet_NaN();
  poses[2].translation() = Eigen::Vector3d(2.0, 0.0, -2.0);
  const Eigen::Isometry3d first_before = poses[0];
  const Eigen::Isometry3d last_before = poses[2];

  const auto stats = gole::slam_3d_core::seedSameLayerPoseGraphZCorrection(
    poses, 0, 0.0);

  EXPECT_FALSE(stats.valid);
  EXPECT_FALSE(stats.applied);
  EXPECT_TRUE(poses[0].matrix().isApprox(first_before.matrix(), 1.0e-12));
  EXPECT_TRUE(poses[2].matrix().isApprox(last_before.matrix(), 1.0e-12));
}

TEST(RegistrationPolicy, DistributedPoseGraphSeedClosesEndpointAndPreservesAnchor)
{
  std::vector<Eigen::Isometry3d> poses(5, Eigen::Isometry3d::Identity());
  for (std::size_t index = 0; index < poses.size(); ++index) {
    poses[index].translation() = Eigen::Vector3d(
      static_cast<double>(index) * 2.0,
      static_cast<double>(index) * 0.5,
      static_cast<double>(index) * 0.1);
    poses[index].linear() = Eigen::AngleAxisd(
      static_cast<double>(index) * 0.05,
      Eigen::Vector3d::UnitZ()).toRotationMatrix();
  }
  const auto original = poses;
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  target.translation() = Eigen::Vector3d(2.2, 0.4, 0.05);
  target.linear() = Eigen::AngleAxisd(-0.02, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto stats = gole::slam_3d_core::seedDistributedPoseGraphCorrection(
    poses, 1, target);

  ASSERT_TRUE(stats.valid);
  EXPECT_TRUE(stats.applied);
  EXPECT_TRUE(poses[0].matrix().isApprox(original[0].matrix(), 1.0e-12));
  EXPECT_TRUE(poses[1].matrix().isApprox(original[1].matrix(), 1.0e-12));
  EXPECT_TRUE(poses.back().matrix().isApprox(target.matrix(), 1.0e-8));
  EXPECT_GT(stats.path_length_m, 0.0);
  EXPECT_GT(stats.endpoint_correction_translation_m, 0.0);
  EXPECT_GT(stats.max_relative_translation_error_m, 0.0);
  EXPECT_LT(stats.max_relative_translation_error_m, stats.endpoint_correction_translation_m);
}

TEST(RegistrationPolicy, DistributedPoseGraphSeedIsTransactionalOnInvalidInput)
{
  std::vector<Eigen::Isometry3d> poses(3, Eigen::Isometry3d::Identity());
  poses[1].translation().x() = std::numeric_limits<double>::quiet_NaN();
  const auto original = poses;

  const auto stats = gole::slam_3d_core::seedDistributedPoseGraphCorrection(
    poses, 0, Eigen::Isometry3d::Identity());

  EXPECT_FALSE(stats.valid);
  EXPECT_FALSE(stats.applied);
  EXPECT_TRUE(poses[0].matrix().isApprox(original[0].matrix(), 1.0e-12));
  EXPECT_TRUE(poses[2].matrix().isApprox(original[2].matrix(), 1.0e-12));
}

TEST(RegistrationPolicy, DistributedPoseGraphSeedReportsExactEndpointAsNoOp)
{
  std::vector<Eigen::Isometry3d> poses(3, Eigen::Isometry3d::Identity());
  poses[1].translation() = Eigen::Vector3d(1.0, 0.5, -0.1);
  poses[2].translation() = Eigen::Vector3d(2.0, 1.0, -0.2);
  poses[2].linear() = Eigen::AngleAxisd(
    0.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const auto original = poses;

  const auto stats = gole::slam_3d_core::seedDistributedPoseGraphCorrection(
    poses, 0, poses.back());

  ASSERT_TRUE(stats.valid);
  EXPECT_FALSE(stats.applied);
  EXPECT_NEAR(stats.endpoint_correction_translation_m, 0.0, 1.0e-12);
  EXPECT_NEAR(stats.endpoint_correction_rotation_deg, 0.0, 1.0e-10);
  for (std::size_t index = 0; index < poses.size(); ++index) {
    EXPECT_TRUE(poses[index].matrix().isApprox(original[index].matrix(), 1.0e-12));
  }
}

TEST(RegistrationPolicy, DistributedPoseGraphSeedFallsBackToNodeIndexWithoutTravel)
{
  std::vector<Eigen::Isometry3d> poses(3, Eigen::Isometry3d::Identity());
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  target.translation() = Eigen::Vector3d(3.0, 0.0, 0.0);
  target.linear() = Eigen::AngleAxisd(
    M_PI_2, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto stats = gole::slam_3d_core::seedDistributedPoseGraphCorrection(
    poses, 0, target);

  ASSERT_TRUE(stats.valid);
  EXPECT_TRUE(stats.applied);
  EXPECT_DOUBLE_EQ(stats.path_length_m, 0.0);
  EXPECT_TRUE(
    poses.front().matrix().isApprox(
      Eigen::Isometry3d::Identity().matrix(), 1.0e-12));
  EXPECT_NEAR(poses[1].translation().x(), 1.5, 1.0e-12);
  EXPECT_NEAR(
    Eigen::AngleAxisd(poses[1].linear()).angle(), M_PI_4, 1.0e-12);
  EXPECT_TRUE(poses.back().matrix().isApprox(target.matrix(), 1.0e-8));
}

TEST(RegistrationPolicy, VerticalGateKeepsNormalCandidateWithoutGroundSupport)
{
  const auto decision =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    1.8, 2.0, 5.0, false, false);

  EXPECT_TRUE(decision.eligible);
  EXPECT_FALSE(decision.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(decision.active_limit_m, 2.0);
}

TEST(RegistrationPolicy, VerticalGateRequiresGroundSupportOnBothScansForRecovery)
{
  const auto missing_current =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    3.8, 2.0, 5.0, false, true);
  const auto missing_candidate =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    3.8, 2.0, 5.0, true, false);
  const auto supported =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    3.8, 2.0, 5.0, true, true);

  EXPECT_FALSE(missing_current.eligible);
  EXPECT_FALSE(missing_candidate.eligible);
  EXPECT_TRUE(supported.eligible);
  EXPECT_TRUE(supported.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(supported.active_limit_m, 5.0);
}

TEST(RegistrationPolicy, VerticalGateRecoveryRemainsBounded)
{
  const auto decision =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    5.1, 2.0, 5.0, true, true);

  EXPECT_FALSE(decision.eligible);
  EXPECT_FALSE(decision.used_registration_only_recovery);
}

TEST(RegistrationPolicy, DisabledNormalVerticalGateRemainsUnbounded)
{
  const auto decision =
    gole::slam_3d_core::evaluatePlaceRecognitionVerticalGate(
    20.0, 0.0, 5.0, false, false);

  EXPECT_TRUE(decision.eligible);
  EXPECT_FALSE(decision.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(decision.active_limit_m, 0.0);
}

TEST(RegistrationPolicy, GroundRecoveryRelaxesConsistencyOnlyForConsecutiveRecoveryMatches)
{
  const auto both_recovery =
    gole::slam_3d_core::evaluatePlaceRecognitionConsistencyTranslationLimit(
    0.75, 1.10, true, true);
  const auto current_only =
    gole::slam_3d_core::evaluatePlaceRecognitionConsistencyTranslationLimit(
    0.75, 1.10, true, false);
  const auto previous_only =
    gole::slam_3d_core::evaluatePlaceRecognitionConsistencyTranslationLimit(
    0.75, 1.10, false, true);

  EXPECT_TRUE(both_recovery.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(both_recovery.active_limit_m, 1.10);
  EXPECT_FALSE(current_only.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(current_only.active_limit_m, 0.75);
  EXPECT_FALSE(previous_only.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(previous_only.active_limit_m, 0.75);
}

TEST(RegistrationPolicy, GroundRecoveryNeverTightensOrChangesUnlimitedConsistency)
{
  const auto tighter_recovery =
    gole::slam_3d_core::evaluatePlaceRecognitionConsistencyTranslationLimit(
    0.75, 0.50, true, true);
  const auto unlimited_normal =
    gole::slam_3d_core::evaluatePlaceRecognitionConsistencyTranslationLimit(
    0.0, 1.10, true, true);

  EXPECT_FALSE(tighter_recovery.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(tighter_recovery.active_limit_m, 0.75);
  EXPECT_FALSE(unlimited_normal.used_registration_only_recovery);
  EXPECT_DOUBLE_EQ(unlimited_normal.active_limit_m, 0.0);
}

TEST(RegistrationPolicy, GroundRecoveryContinuityAllowsOneAdjacentSubmapWithinBoundedGap)
{
  EXPECT_TRUE(
    gole::slam_3d_core::placeRecognitionRecoveryConsistencyContinuous(
      true, true, 7, 6, 7.5, 8.0));
  EXPECT_FALSE(
    gole::slam_3d_core::placeRecognitionRecoveryConsistencyContinuous(
      true, true, 7, 5, 7.5, 8.0));
  EXPECT_FALSE(
    gole::slam_3d_core::placeRecognitionRecoveryConsistencyContinuous(
      true, true, 7, 6, 8.1, 8.0));
  EXPECT_FALSE(
    gole::slam_3d_core::placeRecognitionRecoveryConsistencyContinuous(
      true, false, 7, 7, 1.0, 8.0));
}

TEST(RegistrationPolicy, GroundRecoveryTranslationJumpLimitIsConditional)
{
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionTranslationJumpLimit(
      4.0, 5.0, false), 4.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionTranslationJumpLimit(
      4.0, 5.0, true), 5.0);
}

TEST(RegistrationPolicy, MappingRefinementAlignmentLimitIsConditional)
{
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionAlignmentCorrectionLimit(
      0.8, 3.5, 4.0, false, false),
    0.8);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionAlignmentCorrectionLimit(
      0.8, 3.5, 4.0, true, false),
    3.5);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionAlignmentCorrectionLimit(
      0.8, 3.5, 4.0, false, true),
    4.0);
  EXPECT_DOUBLE_EQ(
    gole::slam_3d_core::effectivePlaceRecognitionAlignmentCorrectionLimit(
      0.8, 5.0, 4.0, true, true),
    5.0);
}

TEST(RegistrationPolicy, GapTranslationGateRelaxesOnlyAfterQualifiedGap)
{
  gole::slam_3d_core::GapAwareTranslationGateConfig config;
  config.enabled = true;
  config.base_max_translation_m = 1.5;
  config.min_gap_s = 0.5;
  config.max_speed_mps = 1.5;
  config.max_translation_m = 3.0;
  config.min_inlier_ratio = 0.5;
  config.max_inlier_rmse_m = 0.3;

  const auto decision = gole::slam_3d_core::computeGapAwareTranslationGate(
    config, 2.3, 0.78, 0.21);

  EXPECT_TRUE(decision.gap_relaxation_applied);
  EXPECT_DOUBLE_EQ(decision.max_translation_m, 3.0);
}

TEST(RegistrationPolicy, GapTranslationGateKeepsBaseLimitAtNormalCadence)
{
  gole::slam_3d_core::GapAwareTranslationGateConfig config;
  config.enabled = true;
  config.base_max_translation_m = 1.5;
  config.min_gap_s = 0.5;
  config.max_speed_mps = 1.5;
  config.max_translation_m = 3.0;
  config.min_inlier_ratio = 0.5;
  config.max_inlier_rmse_m = 0.3;

  const auto decision = gole::slam_3d_core::computeGapAwareTranslationGate(
    config, 0.1, 0.90, 0.20);

  EXPECT_FALSE(decision.gap_relaxation_applied);
  EXPECT_DOUBLE_EQ(decision.max_translation_m, 1.5);
}

TEST(RegistrationPolicy, GapTranslationGateRequiresStrongOverlap)
{
  gole::slam_3d_core::GapAwareTranslationGateConfig config;
  config.enabled = true;
  config.base_max_translation_m = 1.5;
  config.min_gap_s = 0.5;
  config.max_speed_mps = 1.5;
  config.max_translation_m = 3.0;
  config.min_inlier_ratio = 0.5;
  config.max_inlier_rmse_m = 0.3;

  const auto weak_inliers = gole::slam_3d_core::computeGapAwareTranslationGate(
    config, 2.3, 0.40, 0.20);
  const auto high_rmse = gole::slam_3d_core::computeGapAwareTranslationGate(
    config, 2.3, 0.80, 0.35);

  EXPECT_FALSE(weak_inliers.gap_relaxation_applied);
  EXPECT_FALSE(high_rmse.gap_relaxation_applied);
  EXPECT_DOUBLE_EQ(weak_inliers.max_translation_m, 1.5);
  EXPECT_DOUBLE_EQ(high_rmse.max_translation_m, 1.5);
}

TEST(RegistrationPolicy, GapRecoverySelectsAcceptedAlternativeAfterPrimaryFailure)
{
  const gole::slam_3d_core::RegistrationHypothesisQuality primary{
    false, 0.02, 0.95};
  const gole::slam_3d_core::RegistrationHypothesisQuality alternative{
    true, 0.04, 0.80};

  const auto decision = gole::slam_3d_core::selectRegistrationHypothesis(
    primary, alternative, 0.02);

  EXPECT_TRUE(decision.select_alternative);
  EXPECT_FALSE(std::isfinite(decision.primary_score));
  EXPECT_TRUE(std::isfinite(decision.alternative_score));
}

TEST(RegistrationPolicy, GapRecoveryRequiresConfiguredScoreImprovement)
{
  const gole::slam_3d_core::RegistrationHypothesisQuality primary{
    true, 0.030, 0.90};
  const gole::slam_3d_core::RegistrationHypothesisQuality marginal_alternative{
    true, 0.0297, 0.90};
  const gole::slam_3d_core::RegistrationHypothesisQuality strong_alternative{
    true, 0.027, 0.90};

  EXPECT_FALSE(
    gole::slam_3d_core::selectRegistrationHypothesis(
      primary, marginal_alternative, 0.02).select_alternative);
  EXPECT_TRUE(
    gole::slam_3d_core::selectRegistrationHypothesis(
      primary, strong_alternative, 0.02).select_alternative);
}

TEST(RegistrationPolicy, GapRecoveryNeverSelectsRejectedOrNonfiniteAlternative)
{
  const gole::slam_3d_core::RegistrationHypothesisQuality primary{
    true, 0.030, 0.90};
  const gole::slam_3d_core::RegistrationHypothesisQuality rejected{
    false, 0.010, 0.99};
  const gole::slam_3d_core::RegistrationHypothesisQuality nonfinite{
    true, std::numeric_limits<double>::quiet_NaN(), 0.99};

  EXPECT_FALSE(
    gole::slam_3d_core::selectRegistrationHypothesis(
      primary, rejected, 0.0).select_alternative);
  EXPECT_FALSE(
    gole::slam_3d_core::selectRegistrationHypothesis(
      primary, nonfinite, 0.0).select_alternative);
}

TEST(RegistrationPolicy, MappingCloudRefinementAcceptsBoundedObservableImprovement)
{
  const auto decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), mappingRefinementMetrics());

  EXPECT_TRUE(decision.apply);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kAccepted);
  EXPECT_LT(decision.after_score, decision.before_score);
}

TEST(RegistrationPolicy, MappingCloudRefinementRejectsSparseOrDegenerateEvidence)
{
  auto sparse = mappingRefinementMetrics();
  sparse.source_points = 199;
  const auto sparse_decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), sparse);

  auto degenerate = mappingRefinementMetrics();
  degenerate.observable_rank = 5;
  const auto degenerate_decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), degenerate);

  EXPECT_FALSE(sparse_decision.apply);
  EXPECT_EQ(
    sparse_decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kInsufficientSourcePoints);
  EXPECT_FALSE(degenerate_decision.apply);
  EXPECT_EQ(
    degenerate_decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kLowObservableRank);
}

TEST(RegistrationPolicy, MappingCloudRefinementRejectsIllConditionedSolution)
{
  auto metrics = mappingRefinementMetrics();
  metrics.hessian_condition = 25000.0;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);

  EXPECT_FALSE(decision.apply);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kIllConditioned);
}

TEST(RegistrationPolicy, MappingCloudRefinementRejectsSmallRepeatedTargetPatch)
{
  auto metrics = mappingRefinementMetrics();
  metrics.after_target_coverage_ratio = 0.04;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);

  EXPECT_FALSE(decision.apply);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kLowTargetCoverage);
}

TEST(RegistrationPolicy, MappingCloudRefinementRequiresActualOverlapImprovement)
{
  auto metrics = mappingRefinementMetrics();
  metrics.after_inlier_ratio = metrics.before_inlier_ratio;
  metrics.after_inlier_rmse_m = 0.235;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);

  EXPECT_FALSE(decision.apply);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kInsufficientImprovement);
}

TEST(RegistrationPolicy, MappingCloudRefinementPreservesPrimaryRegistrationEvidence)
{
  auto metrics = mappingRefinementMetrics();
  metrics.primary_after_inlier_ratio = 0.45;
  metrics.primary_after_inlier_rmse_m = 0.28;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);

  EXPECT_FALSE(decision.apply);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kPrimaryRegistrationRegression);
}

TEST(RegistrationPolicy, VerifiedCoarseAcceptsStationaryHighOverlapReturn)
{
  auto metrics = mappingRefinementMetrics();
  metrics.before_inlier_ratio = 0.989529;
  metrics.before_inlier_rmse_m = 0.177307;
  metrics.after_inlier_ratio = 0.989674;
  metrics.after_inlier_rmse_m = 0.178216;
  metrics.before_target_coverage_ratio = 0.789759;
  metrics.after_target_coverage_ratio = 0.792771;
  metrics.correction_translation_m = 0.01;
  metrics.correction_rotation_deg = 0.1;
  metrics.hessian_condition = 93.5;
  metrics.primary_before_inlier_ratio = 0.979058;
  metrics.primary_before_inlier_rmse_m = 0.115967;
  metrics.primary_after_inlier_ratio = 0.978;
  metrics.primary_after_inlier_rmse_m = 0.117;

  const auto refinement_decision =
    gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);
  const auto coarse_decision =
    gole::slam_3d_core::evaluateMappingCloudVerifiedCoarse(
    verifiedCoarseConfig(), mappingRefinementConfig(), refinement_decision, metrics);

  EXPECT_FALSE(refinement_decision.apply);
  EXPECT_EQ(
    refinement_decision.reason,
    gole::slam_3d_core::MappingCloudRefinementReason::kInsufficientImprovement);
  EXPECT_TRUE(coarse_decision.accept);
  EXPECT_EQ(
    coarse_decision.reason,
    gole::slam_3d_core::MappingCloudVerifiedCoarseReason::kAccepted);
}

TEST(RegistrationPolicy, VerifiedCoarseRejectsWeakAbsoluteOverlap)
{
  auto metrics = mappingRefinementMetrics();
  metrics.before_inlier_ratio = 0.95;
  metrics.before_inlier_rmse_m = 0.31;
  metrics.after_inlier_ratio = 0.95;
  metrics.after_inlier_rmse_m = 0.31;
  metrics.before_target_coverage_ratio = 0.80;
  metrics.after_target_coverage_ratio = 0.80;
  gole::slam_3d_core::MappingCloudRefinementDecision refinement_decision;
  refinement_decision.reason =
    gole::slam_3d_core::MappingCloudRefinementReason::kInsufficientImprovement;
  refinement_decision.before_score = 0.10;
  refinement_decision.after_score = 0.10;
  refinement_decision.primary_before_score = 0.02;
  refinement_decision.primary_after_score = 0.02;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudVerifiedCoarse(
    verifiedCoarseConfig(), mappingRefinementConfig(), refinement_decision, metrics);

  EXPECT_FALSE(decision.accept);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudVerifiedCoarseReason::kHighInlierRmse);
}

TEST(RegistrationPolicy, VerifiedCoarseCannotOverrideOtherRefinementFailures)
{
  const auto metrics = mappingRefinementMetrics();
  auto refinement_decision = gole::slam_3d_core::evaluateMappingCloudRefinement(
    mappingRefinementConfig(), metrics);
  refinement_decision.reason =
    gole::slam_3d_core::MappingCloudRefinementReason::kHighInlierRmse;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudVerifiedCoarse(
    verifiedCoarseConfig(), mappingRefinementConfig(), refinement_decision, metrics);

  EXPECT_FALSE(decision.accept);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudVerifiedCoarseReason::
    kRefinementDidNotStopAtInsufficientImprovement);
}

TEST(RegistrationPolicy, VerifiedCoarseRejectsMappingScoreRegression)
{
  auto metrics = mappingRefinementMetrics();
  metrics.before_inlier_ratio = 0.98;
  metrics.before_inlier_rmse_m = 0.10;
  metrics.after_inlier_ratio = 0.98;
  metrics.after_inlier_rmse_m = 0.11;
  metrics.before_target_coverage_ratio = 0.80;
  metrics.after_target_coverage_ratio = 0.80;
  gole::slam_3d_core::MappingCloudRefinementDecision refinement_decision;
  refinement_decision.reason =
    gole::slam_3d_core::MappingCloudRefinementReason::kInsufficientImprovement;
  refinement_decision.before_score = 0.010;
  refinement_decision.after_score = 0.011;
  refinement_decision.primary_before_score = 0.010;
  refinement_decision.primary_after_score = 0.010;

  const auto decision = gole::slam_3d_core::evaluateMappingCloudVerifiedCoarse(
    verifiedCoarseConfig(), mappingRefinementConfig(), refinement_decision, metrics);

  EXPECT_FALSE(decision.accept);
  EXPECT_EQ(
    decision.reason,
    gole::slam_3d_core::MappingCloudVerifiedCoarseReason::kMappingScoreRegression);
}

TEST(RegistrationPolicy, ValidatesRelocalizationEvenWhenLocalRegistrationWasRejected)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldValidateAcceptedPlaceRecognition(
      true,
      true,
      true));
}

TEST(RegistrationPolicy, DoesNotValidateWithoutAnAcceptedPlaceResult)
{
  EXPECT_FALSE(
    gole::slam_3d_core::shouldValidateAcceptedPlaceRecognition(
      false,
      true,
      true));
}

TEST(RegistrationPolicy, RestoresPrePlaceResultWhenBackendRejectsLoop)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, true, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, true, true));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, false, false, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(false, false, true, false));
}

TEST(RegistrationPolicy, RestoresShockCandidateBeforeGraphWithoutRequiredValidation)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldRestorePrePlaceResult(true, true, false, false));
}

TEST(RegistrationPolicy, DoesNotMutatePoseGraphForShockQuarantinedPlaceCandidate)
{
  EXPECT_FALSE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(true, true));
  EXPECT_TRUE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(true, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldAttemptPoseGraphCorrection(false, false));
}

TEST(RegistrationPolicy, PreservesOnlyVerifiedLargeLoopMeasurement)
{
  EXPECT_TRUE(
    gole::slam_3d_core::shouldPreserveVerifiedLargeLoopMeasurement(true, true));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldPreserveVerifiedLargeLoopMeasurement(true, false));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldPreserveVerifiedLargeLoopMeasurement(false, true));
  EXPECT_FALSE(
    gole::slam_3d_core::shouldPreserveVerifiedLargeLoopMeasurement(false, false));
}

TEST(RegistrationPolicy, PlacePoseMetricsUseBoundedCurrentPrediction)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  prediction.translation() = Eigen::Vector3d(25.0, -4.0, 1.0);

  Eigen::Isometry3d bounded_place_pose = prediction;
  bounded_place_pose.translation() += Eigen::Vector3d(0.4, -0.2, 0.1);
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    5.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()) * prediction.linear();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_TRUE(metrics.finite);
  EXPECT_NEAR(metrics.prediction_jump_translation_m, std::sqrt(0.21), 1.0e-9);
  EXPECT_NEAR(metrics.prediction_jump_rotation_deg, 5.0, 1.0e-9);
}

TEST(RegistrationPolicy, PlacePoseMetricsExposeGlobalMapCorrection)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  prediction.translation() = Eigen::Vector3d(30.0, 0.0, 0.0);

  Eigen::Isometry3d bounded_place_pose = prediction;
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    10.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()) * prediction.linear();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_TRUE(metrics.finite);
  EXPECT_NEAR(metrics.prediction_jump_translation_m, 0.0, 1.0e-9);
  EXPECT_GT(metrics.map_correction_translation_m, 5.0);
  EXPECT_NEAR(metrics.map_correction_rotation_deg, 10.0, 1.0e-9);
  EXPECT_TRUE(
    gole::slam_3d_core::placePredictionJumpWithinLimits(metrics, 1.2, 20.0));
}

TEST(RegistrationPolicy, PlacePredictionJumpRejectsLargeBoundedRotation)
{
  Eigen::Isometry3d prediction = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d bounded_place_pose = Eigen::Isometry3d::Identity();
  bounded_place_pose.linear() =
    Eigen::AngleAxisd(
    65.0 * 3.14159265358979323846 / 180.0,
    Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const auto metrics = gole::slam_3d_core::computePlaceRecognitionPoseMetrics(
    prediction, bounded_place_pose);

  EXPECT_FALSE(
    gole::slam_3d_core::placePredictionJumpWithinLimits(metrics, 8.0, 60.0));
}

TEST(RegistrationPolicy, CurrentScanLoopMeasurementIsTransportedToGraphEndpoint)
{
  Eigen::Isometry3d keyframe_pose = Eigen::Isometry3d::Identity();
  keyframe_pose.translation() = Eigen::Vector3d(8.0, -3.0, 0.5);
  keyframe_pose.linear() =
    Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  Eigen::Isometry3d keyframe_T_current = Eigen::Isometry3d::Identity();
  keyframe_T_current.translation() = Eigen::Vector3d(0.7, 0.1, 0.05);
  keyframe_T_current.linear() =
    Eigen::AngleAxisd(-0.04, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Isometry3d current_reference_pose =
    keyframe_pose * keyframe_T_current;

  Eigen::Isometry3d current_place_pose = current_reference_pose;
  const Eigen::Isometry3d place_correction =
    Eigen::Translation3d(-0.4, 0.2, 0.1) *
    Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitZ());
  current_place_pose = place_correction * current_place_pose;

  const Eigen::Isometry3d transported_keyframe_pose =
    gole::slam_3d_core::transportCurrentPlaceMatchToKeyframe(
    current_place_pose,
    current_reference_pose,
    keyframe_pose);
  const Eigen::Isometry3d reconstructed_current_pose =
    transported_keyframe_pose * keyframe_T_current;

  EXPECT_TRUE(
    reconstructed_current_pose.matrix().isApprox(
      current_place_pose.matrix(),
      1.0e-12));
}

TEST(RigidTransform, ProjectsAccumulatedScaleBackOntoSe3)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  transform.translation() = Eigen::Vector3d(3.0, -2.0, 0.5);
  transform.linear() =
    1.025 *
    Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  const Eigen::Isometry3d rigid =
    gole::slam_3d_core::projectToRigidIsometry(transform);

  EXPECT_NEAR(rigid.linear().determinant(), 1.0, 1.0e-12);
  EXPECT_NEAR(
    (rigid.linear().transpose() * rigid.linear() - Eigen::Matrix3d::Identity()).norm(),
    0.0,
    1.0e-12);
  EXPECT_TRUE(rigid.translation().isApprox(transform.translation(), 1.0e-12));
}

TEST(RigidTransform, PreventsConstantVelocityScaleCubing)
{
  Eigen::Isometry3d previous = Eigen::Isometry3d::Identity();
  previous.linear() *= 1.0001;
  Eigen::Isometry3d current = previous;
  current.translation().x() = 0.25;

  const Eigen::Isometry3d rigid_previous =
    gole::slam_3d_core::projectToRigidIsometry(previous);
  const Eigen::Isometry3d rigid_current =
    gole::slam_3d_core::projectToRigidIsometry(current);
  const Eigen::Isometry3d delta =
    gole::slam_3d_core::projectToRigidIsometry(
    rigid_previous.inverse() * rigid_current);
  const Eigen::Isometry3d prediction =
    gole::slam_3d_core::projectToRigidIsometry(rigid_current * delta);

  EXPECT_NEAR(prediction.linear().determinant(), 1.0, 1.0e-12);
  EXPECT_NEAR(
    (prediction.linear().transpose() * prediction.linear() -
    Eigen::Matrix3d::Identity()).norm(),
    0.0,
    1.0e-12);
}

}  // namespace
