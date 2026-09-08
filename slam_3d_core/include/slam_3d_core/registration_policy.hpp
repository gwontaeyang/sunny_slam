#ifndef SLAM_3D_CORE__REGISTRATION_POLICY_HPP_
#define SLAM_3D_CORE__REGISTRATION_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Geometry>

#include "slam_3d_core/rigid_transform.hpp"

namespace gole
{
namespace slam_3d_core
{

struct PlaceRecognitionPoseMetrics
{
  bool finite{false};
  double prediction_jump_translation_m{0.0};
  double prediction_jump_rotation_deg{0.0};
  double map_correction_translation_m{0.0};
  double map_correction_rotation_deg{0.0};
};

struct PlaceRecognitionVerticalGateDecision
{
  bool eligible{false};
  bool used_registration_only_recovery{false};
  double active_limit_m{0.0};
};

struct PlaceRecognitionConsistencyTranslationDecision
{
  double active_limit_m{0.0};
  bool used_registration_only_recovery{false};
};

struct SameLayerReferenceCandidate
{
  int keyframe_index{-1};
  double x_m{0.0};
  double y_m{0.0};
  double z_m{0.0};
};

struct SameLayerReferenceSelection
{
  bool found{false};
  bool within_max_distance{false};
  int keyframe_index{-1};
  double horizontal_distance_m{std::numeric_limits<double>::infinity()};
  double z_m{0.0};
};

struct SameLayerGraphTargetCorrection
{
  bool valid{false};
  double requested_z_correction_m{0.0};
  Eigen::Isometry3d graph_target_pose{Eigen::Isometry3d::Identity()};
};

enum class PoseGraphSameLayerConstraintMode
{
  kDisabled,
  kRegistrationOnlyVerticalRecovery,
  kHighConfidenceSingleFloorReturn
};

struct PoseGraphSameLayerConstraintSelection
{
  bool enabled{false};
  double max_horizontal_distance_m{0.0};
  PoseGraphSameLayerConstraintMode mode{PoseGraphSameLayerConstraintMode::kDisabled};
};

inline PoseGraphSameLayerConstraintSelection selectPoseGraphSameLayerConstraint(
  const bool registration_only_vertical_recovery_used,
  const double registration_only_max_horizontal_distance_m,
  const bool minimum_interval_override_applied,
  const bool single_floor_return_enabled,
  const double single_floor_return_max_horizontal_distance_m) noexcept
{
  PoseGraphSameLayerConstraintSelection selection;
  const double return_limit = std::max(
    0.0, single_floor_return_max_horizontal_distance_m);
  if (minimum_interval_override_applied && single_floor_return_enabled &&
    return_limit > 0.0)
  {
    selection.enabled = true;
    selection.max_horizontal_distance_m = return_limit;
    selection.mode = PoseGraphSameLayerConstraintMode::kHighConfidenceSingleFloorReturn;
    return selection;
  }

  const double recovery_limit = std::max(
    0.0, registration_only_max_horizontal_distance_m);
  if (registration_only_vertical_recovery_used && recovery_limit > 0.0) {
    selection.enabled = true;
    selection.max_horizontal_distance_m = recovery_limit;
    selection.mode = PoseGraphSameLayerConstraintMode::kRegistrationOnlyVerticalRecovery;
  }
  return selection;
}

inline const char * poseGraphSameLayerConstraintModeString(
  const PoseGraphSameLayerConstraintMode mode) noexcept
{
  switch (mode) {
    case PoseGraphSameLayerConstraintMode::kDisabled:
      return "disabled";
    case PoseGraphSameLayerConstraintMode::kRegistrationOnlyVerticalRecovery:
      return "registration_only_vertical_recovery";
    case PoseGraphSameLayerConstraintMode::kHighConfidenceSingleFloorReturn:
      return "high_confidence_single_floor_return";
  }
  return "unknown";
}

inline double effectivePoseGraphVerticalTranslationWeight(
  const double base_translation_weight,
  const double single_floor_multiplier,
  const bool high_confidence_single_floor_constraint) noexcept
{
  const double base_weight = std::max(0.0, base_translation_weight);
  if (!high_confidence_single_floor_constraint ||
    !std::isfinite(single_floor_multiplier))
  {
    return base_weight;
  }
  return base_weight * std::max(1.0, single_floor_multiplier);
}

inline double weightedMapTranslationSquaredError(
  const Eigen::Vector3d & expected_map_translation,
  const Eigen::Vector3d & actual_map_translation,
  const double horizontal_weight,
  const double vertical_weight) noexcept
{
  if (!expected_map_translation.allFinite() || !actual_map_translation.allFinite()) {
    return std::numeric_limits<double>::infinity();
  }

  const Eigen::Vector3d error = actual_map_translation - expected_map_translation;
  return std::max(0.0, horizontal_weight) * error.head<2>().squaredNorm() +
         std::max(0.0, vertical_weight) * error.z() * error.z();
}

struct SameLayerPoseGraphZSeedStats
{
  bool valid{false};
  bool applied{false};
  double requested_endpoint_correction_m{0.0};
  double horizontal_path_length_m{0.0};
  double max_z_increment_m{0.0};
};

struct DistributedPoseGraphSeedStats
{
  bool valid{false};
  bool applied{false};
  double path_length_m{0.0};
  double endpoint_correction_translation_m{0.0};
  double endpoint_correction_rotation_deg{0.0};
  double max_pose_update_translation_m{0.0};
  double max_pose_update_rotation_deg{0.0};
  double max_relative_translation_error_m{0.0};
  double max_relative_rotation_error_deg{0.0};
};

inline SameLayerReferenceSelection selectNearestSameLayerReference(
  const double target_x_m,
  const double target_y_m,
  const double max_horizontal_distance_m,
  const std::vector<SameLayerReferenceCandidate> & candidates) noexcept
{
  SameLayerReferenceSelection selection;
  if (!std::isfinite(target_x_m) || !std::isfinite(target_y_m)) {
    return selection;
  }

  for (const auto & candidate : candidates) {
    if (candidate.keyframe_index < 0 || !std::isfinite(candidate.x_m) ||
      !std::isfinite(candidate.y_m) || !std::isfinite(candidate.z_m))
    {
      continue;
    }
    const double distance = std::hypot(
      candidate.x_m - target_x_m,
      candidate.y_m - target_y_m);
    if (!selection.found || distance < selection.horizontal_distance_m) {
      selection.found = true;
      selection.keyframe_index = candidate.keyframe_index;
      selection.horizontal_distance_m = distance;
      selection.z_m = candidate.z_m;
    }
  }

  const double limit = std::max(0.0, max_horizontal_distance_m);
  selection.within_max_distance = selection.found && limit > 0.0 &&
    selection.horizontal_distance_m <= limit;
  return selection;
}

inline SameLayerGraphTargetCorrection transportSameLayerZCorrectionToGraphTarget(
  const Eigen::Isometry3d & matched_current_pose,
  const double historical_reference_z_m,
  const Eigen::Isometry3d & graph_target_pose) noexcept
{
  SameLayerGraphTargetCorrection correction;
  if (!matched_current_pose.matrix().allFinite() ||
    !graph_target_pose.matrix().allFinite() ||
    !std::isfinite(historical_reference_z_m))
  {
    return correction;
  }

  const Eigen::Isometry3d rigid_current = projectToRigidIsometry(matched_current_pose);
  correction.graph_target_pose = projectToRigidIsometry(graph_target_pose);
  correction.requested_z_correction_m =
    historical_reference_z_m - rigid_current.translation().z();
  correction.graph_target_pose.translation().z() += correction.requested_z_correction_m;
  correction.valid = correction.graph_target_pose.matrix().allFinite();
  return correction;
}

inline SameLayerPoseGraphZSeedStats seedSameLayerPoseGraphZCorrection(
  std::vector<Eigen::Isometry3d> & poses,
  const int anchor_index,
  const double graph_target_z_m) noexcept
{
  SameLayerPoseGraphZSeedStats stats;
  const int last_index = static_cast<int>(poses.size()) - 1;
  if (anchor_index < 0 || anchor_index >= last_index ||
    !std::isfinite(graph_target_z_m))
  {
    return stats;
  }

  std::vector<double> cumulative_distance(
    static_cast<std::size_t>(last_index - anchor_index + 1), 0.0);
  for (int index = anchor_index; index <= last_index; ++index) {
    if (!poses[static_cast<std::size_t>(index)].matrix().allFinite()) {
      return stats;
    }
    if (index == anchor_index) {
      continue;
    }
    const Eigen::Vector3d delta =
      poses[static_cast<std::size_t>(index)].translation() -
      poses[static_cast<std::size_t>(index - 1)].translation();
    cumulative_distance[static_cast<std::size_t>(index - anchor_index)] =
      cumulative_distance[static_cast<std::size_t>(index - anchor_index - 1)] +
      std::hypot(delta.x(), delta.y());
  }

  stats.valid = true;
  stats.horizontal_path_length_m = cumulative_distance.back();
  stats.requested_endpoint_correction_m =
    graph_target_z_m - poses.back().translation().z();
  const bool use_horizontal_distance = stats.horizontal_path_length_m > 1.0e-6;
  double previous_ratio = 0.0;
  for (int index = anchor_index + 1; index <= last_index; ++index) {
    const std::size_t offset = static_cast<std::size_t>(index - anchor_index);
    const double ratio = use_horizontal_distance ?
      cumulative_distance[offset] / stats.horizontal_path_length_m :
      static_cast<double>(index - anchor_index) /
      static_cast<double>(last_index - anchor_index);
    poses[static_cast<std::size_t>(index)].translation().z() +=
      ratio * stats.requested_endpoint_correction_m;
    stats.max_z_increment_m = std::max(
      stats.max_z_increment_m,
      std::abs((ratio - previous_ratio) * stats.requested_endpoint_correction_m));
    previous_ratio = ratio;
  }
  stats.applied = std::abs(stats.requested_endpoint_correction_m) > 1.0e-12;
  return stats;
}

inline DistributedPoseGraphSeedStats seedDistributedPoseGraphCorrection(
  std::vector<Eigen::Isometry3d> & poses,
  const int anchor_index,
  const Eigen::Isometry3d & graph_target_pose) noexcept
{
  DistributedPoseGraphSeedStats stats;
  const int last_index = static_cast<int>(poses.size()) - 1;
  if (anchor_index < 0 || anchor_index >= last_index ||
    !graph_target_pose.matrix().allFinite())
  {
    return stats;
  }

  std::vector<double> cumulative_distance(
    static_cast<std::size_t>(last_index - anchor_index + 1), 0.0);
  for (int index = anchor_index; index <= last_index; ++index) {
    if (!poses[static_cast<std::size_t>(index)].matrix().allFinite()) {
      return stats;
    }
    if (index == anchor_index) {
      continue;
    }
    cumulative_distance[static_cast<std::size_t>(index - anchor_index)] =
      cumulative_distance[static_cast<std::size_t>(index - anchor_index - 1)] +
      (poses[static_cast<std::size_t>(index)].translation() -
      poses[static_cast<std::size_t>(index - 1)].translation()).norm();
  }

  const std::vector<Eigen::Isometry3d> original = poses;
  const Eigen::Isometry3d endpoint_correction = projectToRigidIsometry(
    graph_target_pose * original.back().inverse());
  Eigen::Quaterniond endpoint_rotation(endpoint_correction.linear());
  endpoint_rotation.normalize();
  stats.path_length_m = cumulative_distance.back();
  stats.endpoint_correction_translation_m = endpoint_correction.translation().norm();
  stats.endpoint_correction_rotation_deg =
    Eigen::AngleAxisd(endpoint_rotation).angle() * 180.0 / M_PI;

  const bool use_distance = stats.path_length_m > 1.0e-6;
  for (int index = anchor_index + 1; index <= last_index; ++index) {
    const std::size_t offset = static_cast<std::size_t>(index - anchor_index);
    const double ratio = use_distance ?
      cumulative_distance[offset] / stats.path_length_m :
      static_cast<double>(index - anchor_index) /
      static_cast<double>(last_index - anchor_index);
    Eigen::Isometry3d interpolated = Eigen::Isometry3d::Identity();
    interpolated.translation() = endpoint_correction.translation() * ratio;
    interpolated.linear() = Eigen::Quaterniond::Identity().slerp(
      ratio, endpoint_rotation).toRotationMatrix();
    poses[static_cast<std::size_t>(index)] = projectToRigidIsometry(
      interpolated * original[static_cast<std::size_t>(index)]);
    stats.max_pose_update_translation_m = std::max(
      stats.max_pose_update_translation_m,
      (poses[static_cast<std::size_t>(index)].translation() -
      original[static_cast<std::size_t>(index)].translation()).norm());
    const Eigen::Isometry3d pose_update = projectToRigidIsometry(
      poses[static_cast<std::size_t>(index)] *
      original[static_cast<std::size_t>(index)].inverse());
    stats.max_pose_update_rotation_deg = std::max(
      stats.max_pose_update_rotation_deg,
      Eigen::AngleAxisd(Eigen::Quaterniond(pose_update.linear()).normalized()).angle() *
      180.0 / M_PI);
  }

  for (int index = anchor_index + 1; index <= last_index; ++index) {
    const Eigen::Isometry3d original_relative = projectToRigidIsometry(
      original[static_cast<std::size_t>(index - 1)].inverse() *
      original[static_cast<std::size_t>(index)]);
    const Eigen::Isometry3d seeded_relative = projectToRigidIsometry(
      poses[static_cast<std::size_t>(index - 1)].inverse() *
      poses[static_cast<std::size_t>(index)]);
    const Eigen::Isometry3d relative_error = projectToRigidIsometry(
      original_relative.inverse() * seeded_relative);
    stats.max_relative_translation_error_m = std::max(
      stats.max_relative_translation_error_m,
      relative_error.translation().norm());
    stats.max_relative_rotation_error_deg = std::max(
      stats.max_relative_rotation_error_deg,
      Eigen::AngleAxisd(Eigen::Quaterniond(relative_error.linear()).normalized()).angle() *
      180.0 / M_PI);
  }

  stats.valid = poses.back().matrix().allFinite() &&
    poses.back().matrix().isApprox(
    projectToRigidIsometry(graph_target_pose).matrix(), 1.0e-8);
  if (!stats.valid) {
    poses = original;
    return stats;
  }
  stats.applied = stats.endpoint_correction_translation_m > 1.0e-12 ||
    stats.endpoint_correction_rotation_deg > 1.0e-10;
  return stats;
}

inline bool placeRecognitionRecoveryConsistencyContinuous(
  const bool current_used_registration_only_vertical_recovery,
  const bool previous_used_registration_only_vertical_recovery,
  const int current_submap_id,
  const int previous_submap_id,
  const double elapsed_s,
  const double max_gap_s) noexcept
{
  return current_used_registration_only_vertical_recovery &&
         previous_used_registration_only_vertical_recovery &&
         current_submap_id >= 0 && previous_submap_id >= 0 &&
         std::abs(current_submap_id - previous_submap_id) <= 1 &&
         std::isfinite(elapsed_s) && elapsed_s >= 0.0 &&
         std::isfinite(max_gap_s) && max_gap_s > 0.0 && elapsed_s <= max_gap_s;
}

inline double effectivePlaceRecognitionTranslationJumpLimit(
  const double normal_limit_m,
  const double registration_only_recovery_limit_m,
  const bool used_registration_only_vertical_recovery) noexcept
{
  const double normal_limit = std::max(0.0, normal_limit_m);
  if (!used_registration_only_vertical_recovery) {
    return normal_limit;
  }
  return std::max(normal_limit, std::max(0.0, registration_only_recovery_limit_m));
}

inline double effectivePlaceRecognitionAlignmentCorrectionLimit(
  const double normal_limit,
  const double registration_only_recovery_limit,
  const double mapping_refinement_limit,
  const bool used_registration_only_vertical_recovery,
  const bool mapping_refinement_applied) noexcept
{
  double limit = effectivePlaceRecognitionTranslationJumpLimit(
    normal_limit,
    registration_only_recovery_limit,
    used_registration_only_vertical_recovery);
  if (mapping_refinement_applied) {
    limit = std::max(limit, std::max(0.0, mapping_refinement_limit));
  }
  return limit;
}

inline PlaceRecognitionConsistencyTranslationDecision
evaluatePlaceRecognitionConsistencyTranslationLimit(
  const double normal_limit_m,
  const double registration_only_recovery_limit_m,
  const bool current_used_registration_only_vertical_recovery,
  const bool previous_used_registration_only_vertical_recovery) noexcept
{
  PlaceRecognitionConsistencyTranslationDecision decision;
  const double normal_limit = std::max(0.0, normal_limit_m);
  decision.active_limit_m = normal_limit;

  const double recovery_limit = std::max(0.0, registration_only_recovery_limit_m);
  if (normal_limit <= 0.0 || recovery_limit <= normal_limit ||
    !current_used_registration_only_vertical_recovery ||
    !previous_used_registration_only_vertical_recovery)
  {
    return decision;
  }

  decision.active_limit_m = recovery_limit;
  decision.used_registration_only_recovery = true;
  return decision;
}

inline PlaceRecognitionVerticalGateDecision evaluatePlaceRecognitionVerticalGate(
  const double vertical_separation_m,
  const double normal_limit_m,
  const double registration_only_recovery_limit_m,
  const bool current_has_registration_only_support,
  const bool candidate_has_registration_only_support) noexcept
{
  PlaceRecognitionVerticalGateDecision decision;
  if (!std::isfinite(vertical_separation_m) || vertical_separation_m < 0.0) {
    return decision;
  }

  const double normal_limit = std::max(0.0, normal_limit_m);
  decision.active_limit_m = normal_limit;
  if (normal_limit <= 0.0 || vertical_separation_m <= normal_limit) {
    decision.eligible = true;
    return decision;
  }

  const double recovery_limit = std::max(0.0, registration_only_recovery_limit_m);
  if (recovery_limit <= normal_limit ||
    !current_has_registration_only_support ||
    !candidate_has_registration_only_support ||
    vertical_separation_m > recovery_limit)
  {
    return decision;
  }

  decision.eligible = true;
  decision.used_registration_only_recovery = true;
  decision.active_limit_m = recovery_limit;
  return decision;
}

struct GapAwareTranslationGateConfig
{
  bool enabled{false};
  double base_max_translation_m{0.0};
  double min_gap_s{0.0};
  double max_speed_mps{0.0};
  double max_translation_m{0.0};
  double min_inlier_ratio{0.0};
  double max_inlier_rmse_m{0.0};
};

struct TranslationGateDecision
{
  double max_translation_m{0.0};
  bool gap_relaxation_applied{false};
};

struct RegistrationHypothesisQuality
{
  bool accepted{false};
  double fitness{0.0};
  double inlier_ratio{0.0};
};

struct RegistrationHypothesisDecision
{
  bool select_alternative{false};
  double primary_score{std::numeric_limits<double>::infinity()};
  double alternative_score{std::numeric_limits<double>::infinity()};
};

enum class MappingCloudRefinementReason
{
  kDisabled,
  kInsufficientSourcePoints,
  kInsufficientTargetPoints,
  kRegistrationRejected,
  kInvalidMetrics,
  kLowInlierRatio,
  kLowTargetCoverage,
  kHighInlierRmse,
  kLargeTranslation,
  kLargeRotation,
  kLowObservableRank,
  kIllConditioned,
  kInsufficientImprovement,
  kPrimaryRegistrationRegression,
  kAccepted
};

struct MappingCloudRefinementConfig
{
  bool enabled{false};
  int min_source_points{0};
  int min_target_points{0};
  double min_inlier_ratio{0.0};
  double min_target_coverage_ratio{0.0};
  double max_inlier_rmse_m{0.0};
  double max_translation_m{0.0};
  double max_rotation_deg{0.0};
  int min_observable_rank{0};
  double max_hessian_condition{0.0};
  double min_score_improvement_ratio{0.0};
  double max_primary_score_regression_ratio{0.0};
};

struct MappingCloudRefinementMetrics
{
  bool registration_accepted{false};
  int source_points{0};
  int target_points{0};
  double before_inlier_ratio{0.0};
  double before_inlier_rmse_m{std::numeric_limits<double>::infinity()};
  double after_inlier_ratio{0.0};
  double after_inlier_rmse_m{std::numeric_limits<double>::infinity()};
  double before_target_coverage_ratio{0.0};
  double after_target_coverage_ratio{0.0};
  double correction_translation_m{0.0};
  double correction_rotation_deg{0.0};
  int observable_rank{0};
  double hessian_condition{0.0};
  double primary_before_inlier_ratio{0.0};
  double primary_before_inlier_rmse_m{std::numeric_limits<double>::infinity()};
  double primary_after_inlier_ratio{0.0};
  double primary_after_inlier_rmse_m{std::numeric_limits<double>::infinity()};
};

struct MappingCloudRefinementDecision
{
  bool apply{false};
  MappingCloudRefinementReason reason{MappingCloudRefinementReason::kDisabled};
  double before_score{std::numeric_limits<double>::infinity()};
  double after_score{std::numeric_limits<double>::infinity()};
  double primary_before_score{std::numeric_limits<double>::infinity()};
  double primary_after_score{std::numeric_limits<double>::infinity()};
};

enum class MappingCloudVerifiedCoarseReason
{
  kDisabled,
  kRefinementDidNotStopAtInsufficientImprovement,
  kInvalidMetrics,
  kLowInlierRatio,
  kHighInlierRmse,
  kLowTargetCoverage,
  kMappingScoreRegression,
  kPrimaryScoreRegression,
  kAccepted
};

struct MappingCloudVerifiedCoarseConfig
{
  bool enabled{false};
  double min_inlier_ratio{0.0};
  double max_inlier_rmse_m{0.0};
  double min_target_coverage_ratio{0.0};
  double max_score_regression_ratio{0.0};
};

struct MappingCloudVerifiedCoarseDecision
{
  bool accept{false};
  MappingCloudVerifiedCoarseReason reason{MappingCloudVerifiedCoarseReason::kDisabled};
};

enum class PoseGraphMinimumIntervalReason
{
  kInactive,
  kInvalidElapsedTime,
  kOverrideDisabled,
  kMappingRefinementRequired,
  kInvalidQualityMetrics,
  kDescriptorDistance,
  kLowInlierRatio,
  kHighInlierRmse,
  kLowTargetCoverage,
  kOverrideAccepted
};

struct PoseGraphMinimumIntervalOverrideConfig
{
  bool enabled{false};
  double max_descriptor_distance{0.12};
  double min_inlier_ratio{0.90};
  double max_inlier_rmse_m{0.15};
  double min_target_coverage_ratio{0.50};
};

struct PoseGraphMinimumIntervalMetrics
{
  bool have_previous_correction{false};
  double elapsed_since_previous_correction_s{0.0};
  bool mapping_refinement_verified{false};
  double descriptor_distance{std::numeric_limits<double>::infinity()};
  double inlier_ratio{0.0};
  double inlier_rmse_m{std::numeric_limits<double>::infinity()};
  double target_coverage_ratio{0.0};
};

struct PoseGraphMinimumIntervalDecision
{
  bool allow{false};
  bool interval_active{false};
  bool override_applied{false};
  PoseGraphMinimumIntervalReason reason{PoseGraphMinimumIntervalReason::kInactive};
};

inline bool isHighConfidencePoseGraphConstraint(
  const PoseGraphMinimumIntervalOverrideConfig & config,
  const PoseGraphMinimumIntervalMetrics & metrics) noexcept
{
  return config.enabled && metrics.mapping_refinement_verified &&
         std::isfinite(metrics.descriptor_distance) &&
         std::isfinite(metrics.inlier_ratio) &&
         std::isfinite(metrics.inlier_rmse_m) &&
         std::isfinite(metrics.target_coverage_ratio) &&
         metrics.descriptor_distance <= config.max_descriptor_distance &&
         metrics.inlier_ratio >= config.min_inlier_ratio &&
         metrics.inlier_rmse_m <= config.max_inlier_rmse_m &&
         metrics.target_coverage_ratio >= config.min_target_coverage_ratio;
}

inline const char * poseGraphMinimumIntervalReasonString(
  const PoseGraphMinimumIntervalReason reason) noexcept
{
  switch (reason) {
    case PoseGraphMinimumIntervalReason::kInactive:
      return "inactive";
    case PoseGraphMinimumIntervalReason::kInvalidElapsedTime:
      return "invalid_elapsed_time";
    case PoseGraphMinimumIntervalReason::kOverrideDisabled:
      return "override_disabled";
    case PoseGraphMinimumIntervalReason::kMappingRefinementRequired:
      return "mapping_refinement_required";
    case PoseGraphMinimumIntervalReason::kInvalidQualityMetrics:
      return "invalid_quality_metrics";
    case PoseGraphMinimumIntervalReason::kDescriptorDistance:
      return "descriptor_distance";
    case PoseGraphMinimumIntervalReason::kLowInlierRatio:
      return "low_inlier_ratio";
    case PoseGraphMinimumIntervalReason::kHighInlierRmse:
      return "high_inlier_rmse";
    case PoseGraphMinimumIntervalReason::kLowTargetCoverage:
      return "low_target_coverage";
    case PoseGraphMinimumIntervalReason::kOverrideAccepted:
      return "high_confidence_override";
  }
  return "unknown";
}

inline PoseGraphMinimumIntervalDecision evaluatePoseGraphMinimumInterval(
  const double minimum_interval_s,
  const PoseGraphMinimumIntervalOverrideConfig & config,
  const PoseGraphMinimumIntervalMetrics & metrics) noexcept
{
  PoseGraphMinimumIntervalDecision decision;
  const double interval_s = std::max(0.0, minimum_interval_s);
  if (!metrics.have_previous_correction || interval_s <= 0.0) {
    decision.allow = true;
    return decision;
  }

  if (!std::isfinite(metrics.elapsed_since_previous_correction_s) ||
    metrics.elapsed_since_previous_correction_s < 0.0)
  {
    decision.interval_active = true;
    decision.reason = PoseGraphMinimumIntervalReason::kInvalidElapsedTime;
    return decision;
  }
  if (metrics.elapsed_since_previous_correction_s >= interval_s) {
    decision.allow = true;
    return decision;
  }

  decision.interval_active = true;
  if (!config.enabled) {
    decision.reason = PoseGraphMinimumIntervalReason::kOverrideDisabled;
    return decision;
  }
  if (!metrics.mapping_refinement_verified) {
    decision.reason = PoseGraphMinimumIntervalReason::kMappingRefinementRequired;
    return decision;
  }
  if (!std::isfinite(metrics.descriptor_distance) ||
    !std::isfinite(metrics.inlier_ratio) ||
    !std::isfinite(metrics.inlier_rmse_m) ||
    !std::isfinite(metrics.target_coverage_ratio))
  {
    decision.reason = PoseGraphMinimumIntervalReason::kInvalidQualityMetrics;
    return decision;
  }
  if (metrics.descriptor_distance > config.max_descriptor_distance) {
    decision.reason = PoseGraphMinimumIntervalReason::kDescriptorDistance;
    return decision;
  }
  if (metrics.inlier_ratio < config.min_inlier_ratio) {
    decision.reason = PoseGraphMinimumIntervalReason::kLowInlierRatio;
    return decision;
  }
  if (metrics.inlier_rmse_m > config.max_inlier_rmse_m) {
    decision.reason = PoseGraphMinimumIntervalReason::kHighInlierRmse;
    return decision;
  }
  if (metrics.target_coverage_ratio < config.min_target_coverage_ratio) {
    decision.reason = PoseGraphMinimumIntervalReason::kLowTargetCoverage;
    return decision;
  }

  decision.allow = true;
  decision.override_applied = true;
  decision.reason = PoseGraphMinimumIntervalReason::kOverrideAccepted;
  return decision;
}

enum class PoseGraphDuplicateSubmapPairReason
{
  kFirstObservation,
  kOverrideDisabled,
  kHighConfidenceRequired,
  kTargetAnchorAlreadyUsed,
  kEdgeLimit,
  kOverrideAccepted
};

struct PoseGraphDuplicateSubmapPairOverrideConfig
{
  bool enabled{false};
  int max_edges{1};
};

struct PoseGraphDuplicateSubmapPairMetrics
{
  int accepted_edges{0};
  bool high_confidence_interval_override{false};
  bool target_anchor_already_used{false};
};

struct PoseGraphDuplicateSubmapPairDecision
{
  bool allow{false};
  bool override_applied{false};
  PoseGraphDuplicateSubmapPairReason reason{
    PoseGraphDuplicateSubmapPairReason::kFirstObservation};
};

inline const char * poseGraphDuplicateSubmapPairReasonString(
  const PoseGraphDuplicateSubmapPairReason reason) noexcept
{
  switch (reason) {
    case PoseGraphDuplicateSubmapPairReason::kFirstObservation:
      return "first_observation";
    case PoseGraphDuplicateSubmapPairReason::kOverrideDisabled:
      return "override_disabled";
    case PoseGraphDuplicateSubmapPairReason::kHighConfidenceRequired:
      return "high_confidence_required";
    case PoseGraphDuplicateSubmapPairReason::kTargetAnchorAlreadyUsed:
      return "target_anchor_already_used";
    case PoseGraphDuplicateSubmapPairReason::kEdgeLimit:
      return "edge_limit";
    case PoseGraphDuplicateSubmapPairReason::kOverrideAccepted:
      return "confirmed_distinct_anchor_override";
  }
  return "unknown";
}

inline PoseGraphDuplicateSubmapPairDecision evaluatePoseGraphDuplicateSubmapPair(
  const PoseGraphDuplicateSubmapPairOverrideConfig & config,
  const PoseGraphDuplicateSubmapPairMetrics & metrics) noexcept
{
  PoseGraphDuplicateSubmapPairDecision decision;
  if (metrics.accepted_edges <= 0) {
    decision.allow = true;
    return decision;
  }
  if (!config.enabled) {
    decision.reason = PoseGraphDuplicateSubmapPairReason::kOverrideDisabled;
    return decision;
  }
  if (!metrics.high_confidence_interval_override) {
    decision.reason = PoseGraphDuplicateSubmapPairReason::kHighConfidenceRequired;
    return decision;
  }
  if (metrics.target_anchor_already_used) {
    decision.reason = PoseGraphDuplicateSubmapPairReason::kTargetAnchorAlreadyUsed;
    return decision;
  }
  if (metrics.accepted_edges >= std::max(1, config.max_edges)) {
    decision.reason = PoseGraphDuplicateSubmapPairReason::kEdgeLimit;
    return decision;
  }

  decision.allow = true;
  decision.override_applied = true;
  decision.reason = PoseGraphDuplicateSubmapPairReason::kOverrideAccepted;
  return decision;
}

inline const char * mappingCloudRefinementReasonString(
  const MappingCloudRefinementReason reason) noexcept
{
  switch (reason) {
    case MappingCloudRefinementReason::kDisabled:
      return "disabled";
    case MappingCloudRefinementReason::kInsufficientSourcePoints:
      return "insufficient_source_points";
    case MappingCloudRefinementReason::kInsufficientTargetPoints:
      return "insufficient_target_points";
    case MappingCloudRefinementReason::kRegistrationRejected:
      return "registration_rejected";
    case MappingCloudRefinementReason::kInvalidMetrics:
      return "invalid_metrics";
    case MappingCloudRefinementReason::kLowInlierRatio:
      return "low_inlier_ratio";
    case MappingCloudRefinementReason::kLowTargetCoverage:
      return "low_target_coverage";
    case MappingCloudRefinementReason::kHighInlierRmse:
      return "high_inlier_rmse";
    case MappingCloudRefinementReason::kLargeTranslation:
      return "large_translation";
    case MappingCloudRefinementReason::kLargeRotation:
      return "large_rotation";
    case MappingCloudRefinementReason::kLowObservableRank:
      return "low_observable_rank";
    case MappingCloudRefinementReason::kIllConditioned:
      return "ill_conditioned";
    case MappingCloudRefinementReason::kInsufficientImprovement:
      return "insufficient_improvement";
    case MappingCloudRefinementReason::kPrimaryRegistrationRegression:
      return "primary_registration_regression";
    case MappingCloudRefinementReason::kAccepted:
      return "accepted";
  }
  return "unknown";
}

inline double mappingCloudOverlapScore(
  const double inlier_ratio,
  const double inlier_rmse_m) noexcept
{
  if (!std::isfinite(inlier_ratio) || inlier_ratio <= 0.0 ||
    !std::isfinite(inlier_rmse_m) || inlier_rmse_m < 0.0)
  {
    return std::numeric_limits<double>::infinity();
  }
  return inlier_rmse_m * inlier_rmse_m / std::max(0.05, inlier_ratio);
}

inline MappingCloudRefinementDecision evaluateMappingCloudRefinement(
  const MappingCloudRefinementConfig & config,
  const MappingCloudRefinementMetrics & metrics) noexcept
{
  MappingCloudRefinementDecision decision;
  decision.before_score = mappingCloudOverlapScore(
    metrics.before_inlier_ratio, metrics.before_inlier_rmse_m);
  decision.after_score = mappingCloudOverlapScore(
    metrics.after_inlier_ratio, metrics.after_inlier_rmse_m);
  decision.primary_before_score = mappingCloudOverlapScore(
    metrics.primary_before_inlier_ratio, metrics.primary_before_inlier_rmse_m);
  decision.primary_after_score = mappingCloudOverlapScore(
    metrics.primary_after_inlier_ratio, metrics.primary_after_inlier_rmse_m);

  if (!config.enabled) {
    return decision;
  }
  if (metrics.source_points < std::max(0, config.min_source_points)) {
    decision.reason = MappingCloudRefinementReason::kInsufficientSourcePoints;
    return decision;
  }
  if (metrics.target_points < std::max(0, config.min_target_points)) {
    decision.reason = MappingCloudRefinementReason::kInsufficientTargetPoints;
    return decision;
  }
  if (!metrics.registration_accepted) {
    decision.reason = MappingCloudRefinementReason::kRegistrationRejected;
    return decision;
  }
  if (!std::isfinite(decision.before_score) || !std::isfinite(decision.after_score) ||
    !std::isfinite(decision.primary_before_score) ||
    !std::isfinite(decision.primary_after_score) ||
    !std::isfinite(metrics.correction_translation_m) ||
    !std::isfinite(metrics.correction_rotation_deg))
  {
    decision.reason = MappingCloudRefinementReason::kInvalidMetrics;
    return decision;
  }
  if (metrics.after_inlier_ratio < std::max(0.0, config.min_inlier_ratio)) {
    decision.reason = MappingCloudRefinementReason::kLowInlierRatio;
    return decision;
  }
  if (!std::isfinite(metrics.after_target_coverage_ratio) ||
    metrics.after_target_coverage_ratio <
    std::max(0.0, config.min_target_coverage_ratio))
  {
    decision.reason = MappingCloudRefinementReason::kLowTargetCoverage;
    return decision;
  }
  if (config.max_inlier_rmse_m > 0.0 &&
    metrics.after_inlier_rmse_m > config.max_inlier_rmse_m)
  {
    decision.reason = MappingCloudRefinementReason::kHighInlierRmse;
    return decision;
  }
  if (config.max_translation_m > 0.0 &&
    metrics.correction_translation_m > config.max_translation_m)
  {
    decision.reason = MappingCloudRefinementReason::kLargeTranslation;
    return decision;
  }
  if (config.max_rotation_deg > 0.0 &&
    metrics.correction_rotation_deg > config.max_rotation_deg)
  {
    decision.reason = MappingCloudRefinementReason::kLargeRotation;
    return decision;
  }
  if (config.min_observable_rank > 0 &&
    metrics.observable_rank < config.min_observable_rank)
  {
    decision.reason = MappingCloudRefinementReason::kLowObservableRank;
    return decision;
  }
  if (config.max_hessian_condition > 0.0 &&
    (!std::isfinite(metrics.hessian_condition) || metrics.hessian_condition <= 0.0 ||
    metrics.hessian_condition > config.max_hessian_condition))
  {
    decision.reason = MappingCloudRefinementReason::kIllConditioned;
    return decision;
  }

  const double required_improvement_ratio = 1.0 - std::clamp(
    config.min_score_improvement_ratio, 0.0, 1.0);
  if (!(decision.after_score < decision.before_score * required_improvement_ratio)) {
    decision.reason = MappingCloudRefinementReason::kInsufficientImprovement;
    return decision;
  }

  const double allowed_primary_ratio = 1.0 + std::max(
    0.0, config.max_primary_score_regression_ratio);
  if (decision.primary_after_score > decision.primary_before_score * allowed_primary_ratio) {
    decision.reason = MappingCloudRefinementReason::kPrimaryRegistrationRegression;
    return decision;
  }

  decision.apply = true;
  decision.reason = MappingCloudRefinementReason::kAccepted;
  return decision;
}

inline const char * mappingCloudVerifiedCoarseReasonString(
  const MappingCloudVerifiedCoarseReason reason) noexcept
{
  switch (reason) {
    case MappingCloudVerifiedCoarseReason::kDisabled:
      return "disabled";
    case MappingCloudVerifiedCoarseReason::kRefinementDidNotStopAtInsufficientImprovement:
      return "refinement_did_not_stop_at_insufficient_improvement";
    case MappingCloudVerifiedCoarseReason::kInvalidMetrics:
      return "invalid_metrics";
    case MappingCloudVerifiedCoarseReason::kLowInlierRatio:
      return "low_inlier_ratio";
    case MappingCloudVerifiedCoarseReason::kHighInlierRmse:
      return "high_inlier_rmse";
    case MappingCloudVerifiedCoarseReason::kLowTargetCoverage:
      return "low_target_coverage";
    case MappingCloudVerifiedCoarseReason::kMappingScoreRegression:
      return "mapping_score_regression";
    case MappingCloudVerifiedCoarseReason::kPrimaryScoreRegression:
      return "primary_score_regression";
    case MappingCloudVerifiedCoarseReason::kAccepted:
      return "accepted";
  }
  return "unknown";
}

inline MappingCloudVerifiedCoarseDecision evaluateMappingCloudVerifiedCoarse(
  const MappingCloudVerifiedCoarseConfig & config,
  const MappingCloudRefinementConfig & refinement_config,
  const MappingCloudRefinementDecision & refinement_decision,
  const MappingCloudRefinementMetrics & metrics) noexcept
{
  MappingCloudVerifiedCoarseDecision decision;
  if (!config.enabled) {
    return decision;
  }
  if (refinement_decision.reason != MappingCloudRefinementReason::kInsufficientImprovement) {
    decision.reason =
      MappingCloudVerifiedCoarseReason::kRefinementDidNotStopAtInsufficientImprovement;
    return decision;
  }

  if (!std::isfinite(refinement_decision.before_score) ||
    !std::isfinite(refinement_decision.after_score) ||
    !std::isfinite(refinement_decision.primary_before_score) ||
    !std::isfinite(refinement_decision.primary_after_score) ||
    !std::isfinite(metrics.before_inlier_ratio) ||
    !std::isfinite(metrics.after_inlier_ratio) ||
    !std::isfinite(metrics.before_inlier_rmse_m) ||
    !std::isfinite(metrics.after_inlier_rmse_m) ||
    !std::isfinite(metrics.before_target_coverage_ratio) ||
    !std::isfinite(metrics.after_target_coverage_ratio))
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kInvalidMetrics;
    return decision;
  }

  const double min_inlier_ratio = std::max(0.0, config.min_inlier_ratio);
  if (metrics.before_inlier_ratio < min_inlier_ratio ||
    metrics.after_inlier_ratio < min_inlier_ratio)
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kLowInlierRatio;
    return decision;
  }
  if (config.max_inlier_rmse_m > 0.0 &&
    (metrics.before_inlier_rmse_m > config.max_inlier_rmse_m ||
    metrics.after_inlier_rmse_m > config.max_inlier_rmse_m))
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kHighInlierRmse;
    return decision;
  }
  const double min_target_coverage = std::max(0.0, config.min_target_coverage_ratio);
  if (metrics.before_target_coverage_ratio < min_target_coverage ||
    metrics.after_target_coverage_ratio < min_target_coverage)
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kLowTargetCoverage;
    return decision;
  }

  const double allowed_mapping_ratio = 1.0 + std::max(
    0.0, config.max_score_regression_ratio);
  if (refinement_decision.after_score >
    refinement_decision.before_score * allowed_mapping_ratio)
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kMappingScoreRegression;
    return decision;
  }
  const double allowed_primary_ratio = 1.0 + std::max(
    0.0, refinement_config.max_primary_score_regression_ratio);
  if (refinement_decision.primary_after_score >
    refinement_decision.primary_before_score * allowed_primary_ratio)
  {
    decision.reason = MappingCloudVerifiedCoarseReason::kPrimaryScoreRegression;
    return decision;
  }

  decision.accept = true;
  decision.reason = MappingCloudVerifiedCoarseReason::kAccepted;
  return decision;
}

inline double registrationHypothesisQualityScore(
  const RegistrationHypothesisQuality & quality) noexcept
{
  if (!quality.accepted || !std::isfinite(quality.fitness) ||
    !std::isfinite(quality.inlier_ratio) || quality.inlier_ratio <= 0.0)
  {
    return std::numeric_limits<double>::infinity();
  }
  return quality.fitness / std::max(0.05, quality.inlier_ratio);
}

inline RegistrationHypothesisDecision selectRegistrationHypothesis(
  const RegistrationHypothesisQuality & primary,
  const RegistrationHypothesisQuality & alternative,
  const double min_score_improvement_ratio) noexcept
{
  RegistrationHypothesisDecision decision;
  decision.primary_score = registrationHypothesisQualityScore(primary);
  decision.alternative_score = registrationHypothesisQualityScore(alternative);
  if (!std::isfinite(decision.alternative_score)) {
    return decision;
  }
  if (!std::isfinite(decision.primary_score)) {
    decision.select_alternative = true;
    return decision;
  }

  const double required_ratio = 1.0 - std::clamp(
    min_score_improvement_ratio,
    0.0,
    1.0);
  decision.select_alternative =
    decision.alternative_score < decision.primary_score * required_ratio;
  return decision;
}

inline TranslationGateDecision computeGapAwareTranslationGate(
  const GapAwareTranslationGateConfig & config,
  const double scan_gap_s,
  const double inlier_ratio,
  const double inlier_rmse_m) noexcept
{
  TranslationGateDecision decision;
  decision.max_translation_m = std::max(0.0, config.base_max_translation_m);
  if (!config.enabled || decision.max_translation_m <= 0.0 ||
    !std::isfinite(scan_gap_s) || scan_gap_s < config.min_gap_s ||
    !std::isfinite(inlier_ratio) || inlier_ratio < config.min_inlier_ratio ||
    !std::isfinite(inlier_rmse_m) ||
    (config.max_inlier_rmse_m > 0.0 && inlier_rmse_m > config.max_inlier_rmse_m) ||
    config.max_speed_mps <= 0.0 || config.max_translation_m <= 0.0)
  {
    return decision;
  }

  const double gap_limit = std::min(
    config.max_translation_m,
    config.max_speed_mps * scan_gap_s);
  if (std::isfinite(gap_limit) && gap_limit > decision.max_translation_m) {
    decision.max_translation_m = gap_limit;
    decision.gap_relaxation_applied = true;
  }
  return decision;
}

inline double quaternionRotationAngleDeg(const Eigen::Matrix3d & rotation) noexcept
{
  Eigen::Quaterniond quaternion(rotation);
  if (!quaternion.coeffs().allFinite() || quaternion.norm() <= 1.0e-12) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  quaternion.normalize();
  constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;
  return 2.0 * std::acos(std::clamp(std::abs(quaternion.w()), 0.0, 1.0)) *
         radians_to_degrees;
}

inline PlaceRecognitionPoseMetrics computePlaceRecognitionPoseMetrics(
  const Eigen::Isometry3d & prediction,
  const Eigen::Isometry3d & bounded_place_pose) noexcept
{
  PlaceRecognitionPoseMetrics metrics;
  if (!prediction.matrix().allFinite() || !bounded_place_pose.matrix().allFinite()) {
    return metrics;
  }

  const Eigen::Isometry3d rigid_prediction = projectToRigidIsometry(prediction);
  const Eigen::Isometry3d rigid_place_pose = projectToRigidIsometry(bounded_place_pose);
  const Eigen::Isometry3d prediction_T_place =
    rigid_prediction.inverse() * rigid_place_pose;
  const Eigen::Isometry3d map_correction =
    rigid_place_pose * rigid_prediction.inverse();
  metrics.prediction_jump_translation_m = prediction_T_place.translation().norm();
  metrics.prediction_jump_rotation_deg =
    quaternionRotationAngleDeg(prediction_T_place.linear());
  metrics.map_correction_translation_m = map_correction.translation().norm();
  metrics.map_correction_rotation_deg =
    quaternionRotationAngleDeg(map_correction.linear());
  metrics.finite =
    std::isfinite(metrics.prediction_jump_translation_m) &&
    std::isfinite(metrics.prediction_jump_rotation_deg) &&
    std::isfinite(metrics.map_correction_translation_m) &&
    std::isfinite(metrics.map_correction_rotation_deg);
  return metrics;
}

inline Eigen::Isometry3d transportCurrentPlaceMatchToKeyframe(
  const Eigen::Isometry3d & current_place_pose,
  const Eigen::Isometry3d & current_reference_pose,
  const Eigen::Isometry3d & keyframe_pose) noexcept
{
  const Eigen::Isometry3d map_correction = projectToRigidIsometry(
    current_place_pose * current_reference_pose.inverse());
  return projectToRigidIsometry(map_correction * keyframe_pose);
}

inline bool placePredictionJumpWithinLimits(
  const PlaceRecognitionPoseMetrics & metrics,
  const double max_translation_m,
  const double max_rotation_deg) noexcept
{
  if (!metrics.finite) {
    return false;
  }
  if (max_translation_m > 0.0 &&
    metrics.prediction_jump_translation_m > max_translation_m)
  {
    return false;
  }
  if (max_rotation_deg > 0.0 &&
    metrics.prediction_jump_rotation_deg > max_rotation_deg)
  {
    return false;
  }
  return true;
}

inline bool shouldValidateAcceptedPlaceRecognition(
  const bool place_recognition_accepted,
  const bool pre_place_result_available,
  const bool pose_graph_validation_required) noexcept
{
  return
    place_recognition_accepted &&
    pre_place_result_available &&
    pose_graph_validation_required;
}

inline bool shouldRestorePrePlaceResult(
  const bool accepted_place_result_available,
  const bool place_candidate_shock_quarantined,
  const bool validation_attempted,
  const bool pose_graph_correction_applied) noexcept
{
  return accepted_place_result_available &&
         (place_candidate_shock_quarantined ||
         (validation_attempted && !pose_graph_correction_applied));
}

inline bool shouldAttemptPoseGraphCorrection(
  const bool validation_required,
  const bool place_candidate_shock_quarantined) noexcept
{
  return validation_required && !place_candidate_shock_quarantined;
}

inline bool shouldPreserveVerifiedLargeLoopMeasurement(
  const bool mapping_cloud_verified,
  const bool large_pose_jump_accepted) noexcept
{
  return mapping_cloud_verified && large_pose_jump_accepted;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__REGISTRATION_POLICY_HPP_
