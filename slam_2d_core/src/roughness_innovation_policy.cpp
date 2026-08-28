#include "slam_2d_core/roughness_innovation_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace slam_2d_core
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

void updateScore(double value, double threshold, double & score, bool & available)
{
  if (threshold <= 0.0 || !std::isfinite(value) || !std::isfinite(threshold)) {
    return;
  }
  score = std::max(score, value / threshold);
  available = true;
}

double covarianceImprovementRatio(const double rough, const double post)
{
  if (
    !std::isfinite(rough) || !std::isfinite(post) ||
    rough <= 0.0 || post <= 0.0)
  {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return rough / post;
}

std::int64_t holdUntilStamp(
  const std::int64_t scan_stamp_ns,
  const double duration_s)
{
  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  const long double duration_ns =
    static_cast<long double>(duration_s) * kNanosecondsPerSecond;
  const long double available_ns =
    static_cast<long double>(std::numeric_limits<std::int64_t>::max()) -
    static_cast<long double>(scan_stamp_ns);
  if (duration_ns >= available_ns) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return scan_stamp_ns + static_cast<std::int64_t>(duration_ns);
}

void applyLoopClosingSuppression(
  const RoughnessNearChainGateInput & input,
  RoughnessNearChainGateDecision & decision)
{
  if (!input.loop_closing_suppression_enabled) {
    decision.next_loop_closing_suppression_until_stamp_ns = 0;
    return;
  }

  if (decision.active) {
    decision.suppress_loop_closing = true;
    decision.next_loop_closing_suppression_until_stamp_ns = std::max(
      input.loop_closing_suppression_until_stamp_ns,
      holdUntilStamp(input.scan_stamp_ns, input.loop_closing_cooldown_s));
    return;
  }

  if (
    input.loop_closing_suppression_until_stamp_ns != 0 &&
    input.scan_stamp_ns <= input.loop_closing_suppression_until_stamp_ns)
  {
    decision.suppress_loop_closing = true;
    decision.loop_closing_cooldown = true;
    decision.next_loop_closing_suppression_until_stamp_ns =
      input.loop_closing_suppression_until_stamp_ns;
    return;
  }
  decision.next_loop_closing_suppression_until_stamp_ns = 0;
}

double gatedThreshold(
  const double baseline_threshold,
  const double configured_gated_threshold)
{
  if (!std::isfinite(configured_gated_threshold)) {
    return baseline_threshold;
  }
  return std::max(baseline_threshold, configured_gated_threshold);
}

double normalizedAngle(const double angle_rad)
{
  return std::atan2(std::sin(angle_rad), std::cos(angle_rad));
}

}  // namespace

double normalizedRoughnessScore(
  const RoughnessSignal & signal,
  const RoughnessThresholds & thresholds)
{
  double score = 0.0;
  bool available = false;
  updateScore(signal.tilt_deg, thresholds.tilt_deg, score, available);
  updateScore(signal.gyro_xy_rad_s, thresholds.gyro_xy_rad_s, score, available);
  updateScore(
    signal.accel_norm_deviation_m_s2,
    thresholds.accel_norm_deviation_m_s2,
    score,
    available);
  updateScore(
    signal.tilt_rate_deg_s,
    thresholds.tilt_rate_deg_s,
    score,
    available);
  updateScore(
    signal.accel_norm_rate_m_s3,
    thresholds.accel_norm_rate_m_s3,
    score,
    available);
  return available ? score : std::numeric_limits<double>::quiet_NaN();
}

const char * roughnessNearChainGateReasonName(
  const RoughnessNearChainGateReason reason) noexcept
{
  switch (reason) {
    case RoughnessNearChainGateReason::kDisabled:
      return "disabled";
    case RoughnessNearChainGateReason::kIncomplete:
      return "incomplete";
    case RoughnessNearChainGateReason::kInvalidInput:
      return "invalid_input";
    case RoughnessNearChainGateReason::kNormal:
      return "normal";
    case RoughnessNearChainGateReason::kRough:
      return "rough";
    case RoughnessNearChainGateReason::kSpatialRevisit:
      return "spatial_revisit";
    case RoughnessNearChainGateReason::kSpatialPassage:
      return "spatial_passage";
    case RoughnessNearChainGateReason::kRecovery:
      return "recovery";
    case RoughnessNearChainGateReason::kTimestampRollback:
      return "timestamp_rollback";
  }
  return "unknown";
}

RoughnessNearChainGateDecision roughnessNearChainGateDecision(
  const RoughnessNearChainGateInput & input)
{
  RoughnessNearChainGateDecision decision;
  decision.next_last_scan_stamp_ns = input.scan_stamp_ns;
  decision.baseline_threshold = input.baseline_link_match_threshold;
  decision.requested_threshold = input.baseline_link_match_threshold;

  if (!input.enabled) {
    return decision;
  }

  if (
    input.last_scan_stamp_ns != 0 &&
    input.scan_stamp_ns < input.last_scan_stamp_ns)
  {
    decision.reason = RoughnessNearChainGateReason::kTimestampRollback;
    return decision;
  }

  decision.next_recovery_until_stamp_ns = input.recovery_until_stamp_ns;
  decision.next_loop_closing_suppression_until_stamp_ns =
    input.loop_closing_suppression_until_stamp_ns;
  if (
    !std::isfinite(input.recovery_duration_s) ||
    input.recovery_duration_s < 0.0 ||
    !std::isfinite(input.loop_closing_cooldown_s) ||
    input.loop_closing_cooldown_s < 0.0 ||
    !std::isfinite(input.baseline_link_match_threshold))
  {
    decision.reason = RoughnessNearChainGateReason::kInvalidInput;
    return decision;
  }

  const bool spatial_trigger =
    input.spatial_revisit || input.spatial_passage;
  const bool recovery_active =
    input.scan_stamp_ns <= input.recovery_until_stamp_ns;
  bool roughness_evidence_valid = true;
  if (input.imu_window_complete) {
    if (
      !std::isfinite(input.roughness_score) ||
      !std::isfinite(input.roughness_score_gate) ||
      input.roughness_score_gate < 0.0)
    {
      roughness_evidence_valid = false;
      if (
        !input.extend_recovery_on_spatial_trigger ||
        (!spatial_trigger && !recovery_active))
      {
        decision.reason = RoughnessNearChainGateReason::kInvalidInput;
        return decision;
      }
    }
  }
  if (input.imu_window_complete && roughness_evidence_valid) {
    decision.rough_trigger =
      input.rough_sample_count >= input.minimum_rough_samples &&
      input.roughness_score >= input.roughness_score_gate;
  }
  if (decision.rough_trigger) {
    decision.reason = RoughnessNearChainGateReason::kRough;
    decision.active = true;
    decision.next_recovery_until_stamp_ns = std::max(
      input.recovery_until_stamp_ns,
      holdUntilStamp(input.scan_stamp_ns, input.recovery_duration_s));
    decision.requested_threshold = gatedThreshold(
      input.baseline_link_match_threshold,
      input.gated_link_match_threshold);
    applyLoopClosingSuppression(input, decision);
    return decision;
  }

  if (spatial_trigger) {
    decision.reason = input.spatial_revisit ?
      RoughnessNearChainGateReason::kSpatialRevisit :
      RoughnessNearChainGateReason::kSpatialPassage;
    decision.active = true;
    if (input.extend_recovery_on_spatial_trigger) {
      decision.next_recovery_until_stamp_ns = std::max(
        input.recovery_until_stamp_ns,
        holdUntilStamp(input.scan_stamp_ns, input.recovery_duration_s));
    }
    decision.requested_threshold = gatedThreshold(
      input.baseline_link_match_threshold,
      input.gated_link_match_threshold);
    applyLoopClosingSuppression(input, decision);
    return decision;
  }

  if (
    input.extend_recovery_on_spatial_trigger &&
    recovery_active)
  {
    decision.reason = RoughnessNearChainGateReason::kRecovery;
    decision.active = true;
    decision.requested_threshold = gatedThreshold(
      input.baseline_link_match_threshold,
      input.gated_link_match_threshold);
    applyLoopClosingSuppression(input, decision);
    return decision;
  }

  if (!input.imu_window_complete) {
    decision.reason = RoughnessNearChainGateReason::kIncomplete;
    applyLoopClosingSuppression(input, decision);
    return decision;
  }

  if (recovery_active) {
    decision.reason = RoughnessNearChainGateReason::kRecovery;
    decision.active = true;
    decision.requested_threshold = gatedThreshold(
      input.baseline_link_match_threshold,
      input.gated_link_match_threshold);
    applyLoopClosingSuppression(input, decision);
    return decision;
  }

  decision.reason = RoughnessNearChainGateReason::kNormal;
  decision.next_recovery_until_stamp_ns = 0;
  applyLoopClosingSuppression(input, decision);
  return decision;
}

RoughnessSpatialRevisitDecision roughnessSpatialRevisitDecision(
  const bool enabled,
  const double current_x_m,
  const double current_y_m,
  const double current_heading_rad,
  const std::int64_t current_stamp_ns,
  const std::vector<RoughnessSpatialZone> & zones,
  const double radius_m,
  const double minimum_revisit_separation_s,
  const double minimum_heading_delta_rad)
{
  RoughnessSpatialRevisitDecision decision;
  if (!enabled) {
    return decision;
  }
  if (
    !std::isfinite(current_x_m) ||
    !std::isfinite(current_y_m) ||
    !std::isfinite(current_heading_rad) ||
    current_stamp_ns <= 0 ||
    !std::isfinite(radius_m) || radius_m < 0.0 ||
    !std::isfinite(minimum_revisit_separation_s) ||
    minimum_revisit_separation_s < 0.0 ||
    !std::isfinite(minimum_heading_delta_rad) ||
    minimum_heading_delta_rad < 0.0 ||
    minimum_heading_delta_rad > kPi)
  {
    return decision;
  }
  decision.input_valid = true;

  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  const long double minimum_separation_ns =
    static_cast<long double>(minimum_revisit_separation_s) *
    kNanosecondsPerSecond;
  double nearest_distance_m = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < zones.size(); ++index) {
    const auto & zone = zones[index];
    if (
      !std::isfinite(zone.x_m) ||
      !std::isfinite(zone.y_m) ||
      !std::isfinite(zone.heading_rad) ||
      zone.rough_stamp_ns <= 0 ||
      current_stamp_ns <= zone.rough_stamp_ns)
    {
      continue;
    }
    const long double separation_ns =
      static_cast<long double>(current_stamp_ns) -
      static_cast<long double>(zone.rough_stamp_ns);
    if (separation_ns < minimum_separation_ns) {
      continue;
    }

    const double heading_delta_rad = std::fabs(
      normalizedAngle(current_heading_rad - zone.heading_rad));
    if (heading_delta_rad < minimum_heading_delta_rad) {
      continue;
    }
    const double distance_m =
      std::hypot(current_x_m - zone.x_m, current_y_m - zone.y_m);
    if (distance_m >= nearest_distance_m) {
      continue;
    }
    nearest_distance_m = distance_m;
    decision.matched_zone_index = index;
    decision.nearest_distance_m = distance_m;
    decision.heading_delta_rad = heading_delta_rad;
    decision.source_rough_stamp_ns = zone.rough_stamp_ns;
  }

  decision.matched =
    decision.matched_zone_index != std::numeric_limits<std::size_t>::max() &&
    decision.nearest_distance_m <= radius_m;
  return decision;
}

RoughnessSpatialPassageDecision roughnessSpatialPassageDecision(
  const bool enabled,
  const double current_x_m,
  const double current_y_m,
  const std::int64_t current_stamp_ns,
  const std::vector<RoughnessSpatialZone> & zones,
  const double radius_m,
  const double maximum_age_s)
{
  RoughnessSpatialPassageDecision decision;
  if (!enabled) {
    return decision;
  }
  if (
    !std::isfinite(current_x_m) ||
    !std::isfinite(current_y_m) ||
    current_stamp_ns <= 0 ||
    !std::isfinite(radius_m) || radius_m < 0.0 ||
    !std::isfinite(maximum_age_s) || maximum_age_s < 0.0)
  {
    return decision;
  }
  decision.input_valid = true;

  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  const long double maximum_age_ns =
    static_cast<long double>(maximum_age_s) * kNanosecondsPerSecond;
  double nearest_distance_m = std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < zones.size(); ++index) {
    const auto & zone = zones[index];
    const std::int64_t latest_rough_stamp_ns =
      zone.last_rough_stamp_ns > 0 ?
      zone.last_rough_stamp_ns : zone.rough_stamp_ns;
    if (
      !std::isfinite(zone.x_m) ||
      !std::isfinite(zone.y_m) ||
      zone.rough_stamp_ns <= 0 ||
      latest_rough_stamp_ns < zone.rough_stamp_ns ||
      current_stamp_ns <= latest_rough_stamp_ns)
    {
      continue;
    }
    const long double age_ns =
      static_cast<long double>(current_stamp_ns) -
      static_cast<long double>(latest_rough_stamp_ns);
    if (age_ns > maximum_age_ns) {
      continue;
    }
    const double distance_m =
      std::hypot(current_x_m - zone.x_m, current_y_m - zone.y_m);
    if (distance_m >= nearest_distance_m) {
      continue;
    }
    nearest_distance_m = distance_m;
    decision.matched_zone_index = index;
    decision.nearest_distance_m = distance_m;
    decision.source_age_s = static_cast<double>(
      age_ns / kNanosecondsPerSecond);
    decision.source_rough_stamp_ns = latest_rough_stamp_ns;
  }

  decision.matched =
    decision.matched_zone_index != std::numeric_limits<std::size_t>::max() &&
    decision.nearest_distance_m <= radius_m;
  return decision;
}

RoughnessPoseGraphAnchorRefreshResult refreshRoughnessPoseGraphAnchors(
  std::vector<RoughnessPoseGraphAnchor> & anchors,
  const std::vector<RoughnessCorrectedScanPose> & corrected_scan_poses)
{
  RoughnessPoseGraphAnchorRefreshResult result;
  std::vector<RoughnessPoseGraphAnchor> retained;
  retained.reserve(anchors.size());
  result.zones.reserve(anchors.size());

  for (auto anchor : anchors) {
    if (
      anchor.scan_unique_id < 0 ||
      anchor.scan_state_id < 0 ||
      anchor.zone.rough_stamp_ns <= 0 ||
      anchor.zone.last_rough_stamp_ns < anchor.zone.rough_stamp_ns)
    {
      ++result.invalid_anchor_count;
      continue;
    }

    const auto corrected = std::find_if(
      corrected_scan_poses.begin(),
      corrected_scan_poses.end(),
      [&anchor](const RoughnessCorrectedScanPose & candidate) {
        return
        candidate.scan_unique_id == anchor.scan_unique_id &&
        candidate.scan_state_id == anchor.scan_state_id;
      });
    if (corrected == corrected_scan_poses.end()) {
      ++result.missing_scan_count;
      continue;
    }
    if (
      !std::isfinite(corrected->x_m) ||
      !std::isfinite(corrected->y_m) ||
      !std::isfinite(corrected->heading_rad))
    {
      ++result.invalid_corrected_pose_count;
      continue;
    }

    anchor.zone.x_m = corrected->x_m;
    anchor.zone.y_m = corrected->y_m;
    anchor.zone.heading_rad = corrected->heading_rad;
    result.zones.push_back(anchor.zone);
    retained.push_back(std::move(anchor));
  }

  anchors.swap(retained);
  return result;
}

RoughnessPoseGraphAnchorRememberResult rememberRoughnessPoseGraphAnchor(
  std::vector<RoughnessPoseGraphAnchor> & anchors,
  const RoughnessPoseGraphAnchor & anchor,
  const double merge_radius_m,
  const double minimum_revisit_separation_s,
  const std::size_t maximum_anchor_count)
{
  RoughnessPoseGraphAnchorRememberResult result;
  if (
    anchor.scan_unique_id < 0 ||
    anchor.scan_state_id < 0 ||
    !std::isfinite(anchor.zone.x_m) ||
    !std::isfinite(anchor.zone.y_m) ||
    !std::isfinite(anchor.zone.heading_rad) ||
    anchor.zone.rough_stamp_ns <= 0 ||
    anchor.zone.last_rough_stamp_ns < anchor.zone.rough_stamp_ns ||
    !std::isfinite(merge_radius_m) || merge_radius_m < 0.0 ||
    !std::isfinite(minimum_revisit_separation_s) ||
    minimum_revisit_separation_s < 0.0 ||
    maximum_anchor_count == 0)
  {
    return result;
  }

  constexpr long double kNanosecondsPerSecond = 1000000000.0L;
  const long double same_pass_limit_ns =
    static_cast<long double>(minimum_revisit_separation_s) *
    kNanosecondsPerSecond;
  for (auto & existing : anchors) {
    if (
      anchor.zone.rough_stamp_ns < existing.zone.rough_stamp_ns ||
      static_cast<long double>(
        anchor.zone.rough_stamp_ns - existing.zone.rough_stamp_ns) >=
      same_pass_limit_ns)
    {
      continue;
    }
    if (
      std::hypot(
        anchor.zone.x_m - existing.zone.x_m,
        anchor.zone.y_m - existing.zone.y_m) > merge_radius_m)
    {
      continue;
    }
    existing.zone.last_rough_stamp_ns = std::max(
      existing.zone.last_rough_stamp_ns > 0 ?
      existing.zone.last_rough_stamp_ns : existing.zone.rough_stamp_ns,
      anchor.zone.last_rough_stamp_ns > 0 ?
      anchor.zone.last_rough_stamp_ns : anchor.zone.rough_stamp_ns);
    result.accepted = true;
    result.merged = true;
    return result;
  }

  if (anchors.size() >= maximum_anchor_count) {
    const auto oldest = std::min_element(
      anchors.begin(),
      anchors.end(),
      [](const RoughnessPoseGraphAnchor & lhs, const RoughnessPoseGraphAnchor & rhs) {
        const std::int64_t lhs_latest =
        lhs.zone.last_rough_stamp_ns > 0 ?
        lhs.zone.last_rough_stamp_ns : lhs.zone.rough_stamp_ns;
        const std::int64_t rhs_latest =
        rhs.zone.last_rough_stamp_ns > 0 ?
        rhs.zone.last_rough_stamp_ns : rhs.zone.rough_stamp_ns;
        return lhs_latest < rhs_latest;
      });
    anchors.erase(oldest);
    result.evicted = true;
  }
  anchors.push_back(anchor);
  result.accepted = true;
  result.inserted = true;
  return result;
}

RoughnessInnovationDecision roughnessInnovationDecision(
  const double roughness_score,
  const bool imu_window_complete,
  const bool sustained_roughness,
  const double roughness_score_gate,
  const double translation_innovation_m,
  const double yaw_innovation_rad,
  const double translation_gate_m,
  const double yaw_gate_rad)
{
  RoughnessInnovationDecision decision;
  decision.rough =
    imu_window_complete &&
    sustained_roughness &&
    std::isfinite(roughness_score) &&
    std::isfinite(roughness_score_gate) &&
    roughness_score >= roughness_score_gate;
  decision.translation_large =
    std::isfinite(translation_innovation_m) &&
    std::isfinite(translation_gate_m) &&
    translation_gate_m >= 0.0 &&
    translation_innovation_m >= translation_gate_m;
  decision.yaw_large =
    std::isfinite(yaw_innovation_rad) &&
    std::isfinite(yaw_gate_rad) &&
    yaw_gate_rad >= 0.0 &&
    std::fabs(yaw_innovation_rad) >= yaw_gate_rad;
  decision.reject =
    decision.rough && (decision.translation_large || decision.yaw_large);
  return decision;
}

RoughnessPostConfirmationDecision roughnessPostConfirmationDecision(
  const bool rough_translation_large,
  const bool rough_yaw_large,
  const double rough_response,
  const double rough_xy_covariance_max_eigenvalue,
  const double rough_yaw_covariance,
  const double post_response,
  const double post_xy_covariance_max_eigenvalue,
  const double post_yaw_covariance,
  const double transform_disagreement_translation_m,
  const double transform_disagreement_yaw_rad,
  const double min_transform_disagreement_translation_m,
  const double min_transform_disagreement_yaw_rad,
  const double min_response_gain,
  const double min_covariance_improvement_ratio)
{
  RoughnessPostConfirmationDecision decision;
  if (
    std::isfinite(rough_response) &&
    std::isfinite(post_response))
  {
    decision.response_gain = post_response - rough_response;
  }
  decision.translation_covariance_improvement_ratio =
    covarianceImprovementRatio(
    rough_xy_covariance_max_eigenvalue,
    post_xy_covariance_max_eigenvalue);
  decision.yaw_covariance_improvement_ratio =
    covarianceImprovementRatio(
    rough_yaw_covariance,
    post_yaw_covariance);

  const bool valid_response_gate =
    std::isfinite(min_response_gain) && min_response_gain >= 0.0;
  const bool valid_covariance_gate =
    std::isfinite(min_covariance_improvement_ratio) &&
    min_covariance_improvement_ratio >= 1.0;
  decision.response_improved =
    valid_response_gate &&
    std::isfinite(decision.response_gain) &&
    decision.response_gain >= min_response_gain;
  decision.translation_covariance_improved =
    valid_covariance_gate &&
    std::isfinite(decision.translation_covariance_improvement_ratio) &&
    decision.translation_covariance_improvement_ratio >=
    min_covariance_improvement_ratio;
  decision.yaw_covariance_improved =
    valid_covariance_gate &&
    std::isfinite(decision.yaw_covariance_improvement_ratio) &&
    decision.yaw_covariance_improvement_ratio >=
    min_covariance_improvement_ratio;

  decision.translation_inconsistent =
    rough_translation_large &&
    std::isfinite(transform_disagreement_translation_m) &&
    std::isfinite(min_transform_disagreement_translation_m) &&
    min_transform_disagreement_translation_m >= 0.0 &&
    transform_disagreement_translation_m >=
    min_transform_disagreement_translation_m;
  decision.yaw_inconsistent =
    rough_yaw_large &&
    std::isfinite(transform_disagreement_yaw_rad) &&
    std::isfinite(min_transform_disagreement_yaw_rad) &&
    min_transform_disagreement_yaw_rad >= 0.0 &&
    std::fabs(transform_disagreement_yaw_rad) >=
    min_transform_disagreement_yaw_rad;

  decision.translation_confirmed_transient =
    decision.translation_inconsistent &&
    decision.response_improved &&
    decision.translation_covariance_improved;
  decision.yaw_confirmed_transient =
    decision.yaw_inconsistent &&
    decision.response_improved &&
    decision.yaw_covariance_improved;
  decision.confirmed_transient =
    decision.translation_confirmed_transient ||
    decision.yaw_confirmed_transient;
  return decision;
}

}  // namespace slam_2d_core
