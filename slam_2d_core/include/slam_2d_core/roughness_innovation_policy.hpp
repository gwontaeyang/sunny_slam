#ifndef SLAM_2D_CORE__ROUGHNESS_INNOVATION_POLICY_HPP_
#define SLAM_2D_CORE__ROUGHNESS_INNOVATION_POLICY_HPP_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace slam_2d_core
{

struct RoughnessSignal
{
  double tilt_deg{0.0};
  double gyro_xy_rad_s{0.0};
  double accel_norm_deviation_m_s2{0.0};
  double tilt_rate_deg_s{0.0};
  double accel_norm_rate_m_s3{0.0};
};

struct RoughnessThresholds
{
  double tilt_deg{2.0};
  double gyro_xy_rad_s{0.05};
  double accel_norm_deviation_m_s2{0.35};
  double tilt_rate_deg_s{5.0};
  double accel_norm_rate_m_s3{80.0};
};

struct RoughnessInnovationDecision
{
  bool rough{false};
  bool translation_large{false};
  bool yaw_large{false};
  bool reject{false};
};

struct RoughnessPostConfirmationDecision
{
  bool response_improved{false};
  bool translation_covariance_improved{false};
  bool yaw_covariance_improved{false};
  bool translation_inconsistent{false};
  bool yaw_inconsistent{false};
  bool translation_confirmed_transient{false};
  bool yaw_confirmed_transient{false};
  bool confirmed_transient{false};
  double response_gain{std::numeric_limits<double>::quiet_NaN()};
  double translation_covariance_improvement_ratio{
    std::numeric_limits<double>::quiet_NaN()};
  double yaw_covariance_improvement_ratio{
    std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessSpatialZone
{
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  double heading_rad{std::numeric_limits<double>::quiet_NaN()};
  // The first stamp anchors revisit separation; the last stamp tracks passage age.
  std::int64_t rough_stamp_ns{0};
  std::int64_t last_rough_stamp_ns{0};
};

struct RoughnessPoseGraphAnchor
{
  std::int32_t scan_unique_id{-1};
  std::int32_t scan_state_id{-1};
  RoughnessSpatialZone zone;
};

struct RoughnessCorrectedScanPose
{
  std::int32_t scan_unique_id{-1};
  std::int32_t scan_state_id{-1};
  double x_m{std::numeric_limits<double>::quiet_NaN()};
  double y_m{std::numeric_limits<double>::quiet_NaN()};
  double heading_rad{std::numeric_limits<double>::quiet_NaN()};
};

struct RoughnessPoseGraphAnchorRefreshResult
{
  std::vector<RoughnessSpatialZone> zones;
  std::size_t invalid_anchor_count{0};
  std::size_t missing_scan_count{0};
  std::size_t invalid_corrected_pose_count{0};
};

struct RoughnessPoseGraphAnchorRememberResult
{
  bool accepted{false};
  bool inserted{false};
  bool merged{false};
  bool evicted{false};
};

struct RoughnessSpatialRevisitDecision
{
  bool input_valid{false};
  bool matched{false};
  std::size_t matched_zone_index{std::numeric_limits<std::size_t>::max()};
  double nearest_distance_m{std::numeric_limits<double>::quiet_NaN()};
  double heading_delta_rad{std::numeric_limits<double>::quiet_NaN()};
  std::int64_t source_rough_stamp_ns{0};
};

struct RoughnessSpatialPassageDecision
{
  bool input_valid{false};
  bool matched{false};
  std::size_t matched_zone_index{std::numeric_limits<std::size_t>::max()};
  double nearest_distance_m{std::numeric_limits<double>::quiet_NaN()};
  double source_age_s{std::numeric_limits<double>::quiet_NaN()};
  std::int64_t source_rough_stamp_ns{0};
};

enum class RoughnessNearChainGateReason
{
  kDisabled,
  kIncomplete,
  kInvalidInput,
  kNormal,
  kRough,
  kSpatialRevisit,
  kSpatialPassage,
  kRecovery,
  kTimestampRollback,
};

struct RoughnessNearChainGateInput
{
  bool enabled{false};
  std::int64_t scan_stamp_ns{0};
  std::int64_t last_scan_stamp_ns{0};
  std::int64_t recovery_until_stamp_ns{0};
  bool imu_window_complete{false};
  std::size_t rough_sample_count{0};
  double roughness_score{std::numeric_limits<double>::quiet_NaN()};
  double roughness_score_gate{std::numeric_limits<double>::quiet_NaN()};
  std::size_t minimum_rough_samples{1};
  bool spatial_revisit{false};
  bool spatial_passage{false};
  bool extend_recovery_on_spatial_trigger{false};
  double recovery_duration_s{0.0};
  bool loop_closing_suppression_enabled{false};
  std::int64_t loop_closing_suppression_until_stamp_ns{0};
  double loop_closing_cooldown_s{0.0};
  double baseline_link_match_threshold{0.0};
  double gated_link_match_threshold{0.0};
};

struct RoughnessNearChainGateDecision
{
  RoughnessNearChainGateReason reason{
    RoughnessNearChainGateReason::kDisabled};
  bool active{false};
  bool rough_trigger{false};
  bool suppress_loop_closing{false};
  bool loop_closing_cooldown{false};
  std::int64_t next_last_scan_stamp_ns{0};
  std::int64_t next_recovery_until_stamp_ns{0};
  std::int64_t next_loop_closing_suppression_until_stamp_ns{0};
  double baseline_threshold{0.0};
  double requested_threshold{0.0};
};

double normalizedRoughnessScore(
  const RoughnessSignal & signal,
  const RoughnessThresholds & thresholds);

const char * roughnessNearChainGateReasonName(
  RoughnessNearChainGateReason reason) noexcept;

RoughnessNearChainGateDecision roughnessNearChainGateDecision(
  const RoughnessNearChainGateInput & input);

RoughnessSpatialRevisitDecision roughnessSpatialRevisitDecision(
  bool enabled,
  double current_x_m,
  double current_y_m,
  double current_heading_rad,
  std::int64_t current_stamp_ns,
  const std::vector<RoughnessSpatialZone> & zones,
  double radius_m,
  double minimum_revisit_separation_s,
  double minimum_heading_delta_rad);

RoughnessSpatialPassageDecision roughnessSpatialPassageDecision(
  bool enabled,
  double current_x_m,
  double current_y_m,
  std::int64_t current_stamp_ns,
  const std::vector<RoughnessSpatialZone> & zones,
  double radius_m,
  double maximum_age_s);

RoughnessPoseGraphAnchorRefreshResult refreshRoughnessPoseGraphAnchors(
  std::vector<RoughnessPoseGraphAnchor> & anchors,
  const std::vector<RoughnessCorrectedScanPose> & corrected_scan_poses);

RoughnessPoseGraphAnchorRememberResult rememberRoughnessPoseGraphAnchor(
  std::vector<RoughnessPoseGraphAnchor> & anchors,
  const RoughnessPoseGraphAnchor & anchor,
  double merge_radius_m,
  double minimum_revisit_separation_s,
  std::size_t maximum_anchor_count);

RoughnessInnovationDecision roughnessInnovationDecision(
  double roughness_score,
  bool imu_window_complete,
  bool sustained_roughness,
  double roughness_score_gate,
  double translation_innovation_m,
  double yaw_innovation_rad,
  double translation_gate_m,
  double yaw_gate_rad);

RoughnessPostConfirmationDecision roughnessPostConfirmationDecision(
  bool rough_translation_large,
  bool rough_yaw_large,
  double rough_response,
  double rough_xy_covariance_max_eigenvalue,
  double rough_yaw_covariance,
  double post_response,
  double post_xy_covariance_max_eigenvalue,
  double post_yaw_covariance,
  double transform_disagreement_translation_m,
  double transform_disagreement_yaw_rad,
  double min_transform_disagreement_translation_m,
  double min_transform_disagreement_yaw_rad,
  double min_response_gain,
  double min_covariance_improvement_ratio);

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_INNOVATION_POLICY_HPP_
