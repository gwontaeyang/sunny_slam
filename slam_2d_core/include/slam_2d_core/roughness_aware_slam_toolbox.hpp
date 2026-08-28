#ifndef SLAM_2D_CORE__ROUGHNESS_AWARE_SLAM_TOOLBOX_HPP_
#define SLAM_2D_CORE__ROUGHNESS_AWARE_SLAM_TOOLBOX_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "slam_2d_core/adaptive_roughness_memory.hpp"
#include "slam_2d_core/karto_loop_closure_observer.hpp"
#include "slam_2d_core/karto_scan_matcher_vector.hpp"
#include "slam_2d_core/karto_near_chain_observer.hpp"
#include "slam_2d_core/near_chain_rotation_consistency.hpp"
#include "slam_2d_core/ordered_stamp_audit.hpp"
#include "slam_2d_core/roughness_innovation_policy.hpp"
#include "slam_2d_core/roughness_loop_memory.hpp"
#include "slam_2d_core/roughness_place_recognition.hpp"
#include "slam_2d_core/roughness_place_scan_chain.hpp"
#include "slam_2d_core/roughness_scan_conditioner.hpp"
#include "slam_toolbox/slam_toolbox_common.hpp"

namespace slam_2d_core
{

class RoughnessAwareSlamToolbox : public slam_toolbox::SlamToolbox
{
public:
  explicit RoughnessAwareSlamToolbox(rclcpp::NodeOptions options);
  ~RoughnessAwareSlamToolbox() override;

  void configure() override;
  void shutdown() noexcept;

protected:
  void laserCallback(sensor_msgs::msg::LaserScan::ConstSharedPtr scan) override;
  karto::LocalizedRangeScan * addScan(
    karto::LaserRangeFinder * laser,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    karto::Pose2 & odom_pose) override;
  bool deserializePoseGraphCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Request> req,
    std::shared_ptr<slam_toolbox::srv::DeserializePoseGraph::Response> resp) override;
  bool clearQueueCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<slam_toolbox::srv::ClearQueue::Request> req,
    std::shared_ptr<slam_toolbox::srv::ClearQueue::Response> resp);
  bool finalizePendingScansCallback(
    const std::shared_ptr<rmw_request_id_t> request_header,
    const std::shared_ptr<slam_toolbox::srv::ClearQueue::Request> req,
    std::shared_ptr<slam_toolbox::srv::ClearQueue::Response> resp);

private:
  struct ImuSample
  {
    std::int64_t stamp_ns{0};
    RoughnessSignal signal;
    double score{std::numeric_limits<double>::quiet_NaN()};
    bool quality_valid{false};
  };

  struct PendingScan
  {
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
    karto::Pose2 odom_pose;
    std::chrono::steady_clock::time_point enqueued_at;
    std::uint64_t mapper_generation{0};
  };

  struct WindowDecision
  {
    double score{std::numeric_limits<double>::quiet_NaN()};
    std::size_t sample_count{0};
    std::size_t rough_sample_count{0};
    double rough_duration_s{0.0};
    bool complete{false};
    bool sustained{false};
    bool nearest_imu_available{false};
    bool nearest_imu_quality_valid{false};
    double nearest_imu_dt_s{std::numeric_limits<double>::quiet_NaN()};
    double nearest_imu_score{std::numeric_limits<double>::quiet_NaN()};
    RoughnessSignal peak_signal{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t mapper_generation{0};
  };

  enum class ScanConditioningMode
  {
    kObserve,
    kDrop,
    kThrottle,
    kQuarantine,
    kStabilize,
    kResidualFilter,
  };

  struct ScanConditioningResult
  {
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
    std::string action{"allow"};
    bool drop{false};
    bool rough_active{false};
    bool hard_active{false};
    bool imu_available{false};
    bool recovery_active{false};
    bool recovery_fault{false};
    double score{std::numeric_limits<double>::quiet_NaN()};
    double nearest_imu_dt_s{std::numeric_limits<double>::quiet_NaN()};
    std::size_t input_valid_beams{0};
    std::size_t output_valid_beams{0};
    std::size_t overlap_beams{0};
    std::size_t removed_beams{0};
  };

  struct QueuedScan
  {
    sensor_msgs::msg::LaserScan::ConstSharedPtr scan;
    karto::Pose2 odom_pose;
    WindowDecision decision;
    std::uint64_t sequence{0};
  };

  struct ScanMatchInnovation
  {
    bool available{false};
    double response{std::numeric_limits<double>::quiet_NaN()};
    double dx_m{std::numeric_limits<double>::quiet_NaN()};
    double dy_m{std::numeric_limits<double>::quiet_NaN()};
    double translation_m{std::numeric_limits<double>::quiet_NaN()};
    double yaw_rad{std::numeric_limits<double>::quiet_NaN()};
    double covariance_xx{std::numeric_limits<double>::quiet_NaN()};
    double covariance_xy{std::numeric_limits<double>::quiet_NaN()};
    double covariance_yy{std::numeric_limits<double>::quiet_NaN()};
    double covariance_yawyaw{std::numeric_limits<double>::quiet_NaN()};
    double covariance_xy_max_eigenvalue{std::numeric_limits<double>::quiet_NaN()};
    double covariance_xy_condition{std::numeric_limits<double>::quiet_NaN()};
    double scan_match_covariance_weighted_disagreement_sq{
      std::numeric_limits<double>::quiet_NaN()};
    double odom_correction_x_m{std::numeric_limits<double>::quiet_NaN()};
    double odom_correction_y_m{std::numeric_limits<double>::quiet_NaN()};
    double odom_correction_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
    double predicted_x_m{std::numeric_limits<double>::quiet_NaN()};
    double predicted_y_m{std::numeric_limits<double>::quiet_NaN()};
    double predicted_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
    double matched_x_m{std::numeric_limits<double>::quiet_NaN()};
    double matched_y_m{std::numeric_limits<double>::quiet_NaN()};
    double matched_yaw_rad{std::numeric_limits<double>::quiet_NaN()};
  };

  struct FrozenScanContext
  {
    std::string sensor_name;
    karto::Pose2 last_odometric_pose;
    karto::Pose2 last_corrected_pose;
    std::vector<std::unique_ptr<karto::LocalizedRangeScan>> scan_storage;
    karto::LocalizedRangeScanVector scans;
    std::vector<std::int32_t> source_state_ids;
  };

  struct PostConfirmationCandidate
  {
    std::int64_t stamp_ns{0};
    double roughness_score{std::numeric_limits<double>::quiet_NaN()};
    bool translation_large{false};
    bool yaw_large{false};
    ScanMatchInnovation innovation;
    FrozenScanContext context;
  };

  void configureScanSubscription();
  void imuCallback(sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void flushPendingScans(bool allow_wall_timeout);
  void queueAcceptedScan(
    const PendingScan & pending,
    const WindowDecision & decision);
  void runScanWorker();
  karto::LocalizedRangeScan * processQueuedScan(const QueuedScan & queued);
  WindowDecision windowDecisionLocked(std::int64_t scan_stamp_ns) const;
  karto::LocalizedRangeScan * processScanWithWindow(
    karto::LaserRangeFinder * laser,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    karto::Pose2 & odom_pose,
    const std::optional<WindowDecision> & window);
  void updateAdmittedScanHeader(const std_msgs::msg::Header & header);
  karto::LocalizedRangeScan * processAndTrackScan(
    karto::LaserRangeFinder * laser,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    karto::Pose2 & odom_pose,
    const std::optional<WindowDecision> & window);
  ScanMatchInnovation previewScanMatchInnovation(
    karto::LaserRangeFinder * laser,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    karto::Pose2 & odom_pose,
    const FrozenScanContext * frozen_context = nullptr,
    FrozenScanContext * captured_context = nullptr);
  ScanMatchInnovation previewLocalizedScanMatchInnovationLocked(
    karto::LocalizedRangeScan * range_scan,
    karto::Mapper * mapper,
    const FrozenScanContext * frozen_context,
    FrozenScanContext * captured_context);
  void trackProcessedScanLocked(
    karto::LocalizedRangeScan * processed_scan,
    karto::Mapper * mapper);
  void updateRoughnessPlaceRecognitionLocked(
    const karto::LocalizedRangeScan * processed_scan,
    std::int64_t stamp_ns,
    const std::optional<WindowDecision> & window);
  void evaluateRoughnessPlaceScanChainUpdateLocked(
    RoughnessPlaceScanChainUpdate update,
    karto::Mapper * mapper);
  std::optional<karto::Pose2> predictPoseGraphSpatialPoseLocked(
    const karto::LocalizedRangeScan * range_scan) const;
  std::vector<RoughnessSpatialZone> refreshPoseGraphSpatialAnchorsLocked(
    karto::Mapper * mapper);
  void rememberPoseGraphSpatialAnchorLocked(
    const karto::LocalizedRangeScan * processed_scan,
    std::int64_t rough_stamp_ns);
  void resetMapperDerivedState(const char * reason);
  void evaluatePendingPostConfirmation(
    karto::LaserRangeFinder * laser,
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    karto::Pose2 & odom_pose,
    const std::optional<WindowDecision> & window);
  bool armPostConfirmation(
    std::int64_t stamp_ns,
    double roughness_score,
    const RoughnessInnovationDecision & gate,
    const ScanMatchInnovation & innovation,
    FrozenScanContext context);
  void clearPostConfirmation(const char * reason);
  void rememberRoughZoneLocked(const RoughnessSpatialZone & zone);
  ScanConditioningResult conditionScan(
    const sensor_msgs::msg::LaserScan::ConstSharedPtr & scan,
    const std::optional<WindowDecision> & window);
  std::optional<geometry_msgs::msg::TransformStamped> stableScanTransform(
    const sensor_msgs::msg::LaserScan & current,
    const sensor_msgs::msg::LaserScan & stable) const;
  void rememberStableConditioningScan(
    const sensor_msgs::msg::LaserScan & scan,
    std::int64_t stamp_ns,
    const LaserScanQuality & quality);
  void resetScanConditioningState(const char * reason);
  void resetScanConditioningStateLocked();
  static const char * scanConditioningModeName(ScanConditioningMode mode) noexcept;
  void logDiagnostics();

  static std::int64_t messageStampNs(const builtin_interfaces::msg::Time & stamp);
  static double tiltDeg(const sensor_msgs::msg::Imu & msg);
  static double normalizedAngle(double angle_rad);

  bool enabled_{false};
  std::string mode_{"observe"};
  std::string imu_topic_{"/imu/data"};
  double window_before_s_{0.10};
  double window_after_s_{0.10};
  double pending_timeout_s_{0.50};
  double imu_buffer_duration_s_{2.0};
  double imu_epoch_reset_threshold_s_{1.0};
  double roughness_score_gate_{2.75};
  double min_rough_duration_s_{0.02};
  double translation_innovation_gate_m_{0.04};
  double yaw_innovation_gate_rad_{0.05235987755982989};
  double near_chain_recovery_duration_s_{2.0};
  double near_chain_loop_closing_cooldown_s_{0.0};
  double near_chain_gated_min_response_{1.1};
  std::string roughness_constraint_mode_name_;
  RoughnessConstraintMode roughness_constraint_mode_{
    RoughnessConstraintMode::kInterventionOff};
  bool enable_near_chain_candidate_observer_{false};
  bool enable_loop_closure_candidate_observer_{false};
  std::size_t near_chain_observer_min_parity_candidates_{20};
  AdaptiveRoughnessMemoryConfig adaptive_memory_config_;
  NearChainQualityConfig near_chain_quality_config_;
  AdaptiveConstraintPolicyConfig adaptive_constraint_policy_config_;
  double adaptive_dynamic_threshold_max_{1.0};
  NearChainRotationConsistencyConfig rotation_consistency_config_;
  double spatial_memory_radius_m_{0.6};
  double spatial_memory_min_revisit_separation_s_{20.0};
  double spatial_memory_min_heading_delta_rad_{2.2};
  double spatial_memory_merge_radius_m_{0.10};
  double spatial_passage_radius_m_{0.55};
  double spatial_passage_max_age_s_{3.5};
  double post_confirmation_max_age_s_{2.0};
  double post_confirmation_min_response_{0.30};
  double post_confirmation_min_response_gain_{0.05};
  double post_confirmation_min_covariance_improvement_ratio_{1.5};
  double post_confirmation_translation_disagreement_gate_m_{0.03};
  double post_confirmation_yaw_disagreement_gate_rad_{0.017453292519943295};
  double diagnostics_period_s_{10.0};
  std::size_t min_rough_samples_{3};
  std::size_t max_pending_scans_{2000};
  std::size_t near_chain_min_rough_samples_{1};
  std::size_t spatial_memory_max_zones_{200};
  std::size_t post_confirmation_min_base_scans_{8};
  std::size_t post_confirmation_max_base_scans_{10};
  std::size_t roughness_place_descriptor_bins_{16};
  std::size_t roughness_place_max_observations_per_event_{512};
  std::size_t roughness_place_max_events_{200};
  std::size_t roughness_place_candidate_top_k_{5};
  std::size_t roughness_place_min_state_separation_{20};
  double roughness_place_min_revisit_separation_s_{20.0};
  RoughnessPlaceScanChainConfig roughness_place_scan_chain_config_;
  RoughnessPlaceConstraintThresholds roughness_place_constraint_thresholds_;
  std::string roughness_place_constraint_mode_{"observe"};
  std::string roughness_place_constraint_anchor_strategy_{"best_single"};
  double roughness_place_constraint_pair_covariance_scale_{1.0};
  double roughness_place_constraint_group_covariance_scale_{1.0};
  double roughness_place_constraint_xy_variance_floor_m2_{0.01};
  double roughness_place_constraint_yaw_variance_floor_rad2_{
    0.00030461741978670857};
  std::size_t imu_qos_depth_{2000};
  std::size_t scan_qos_depth_{5};
  bool enable_near_chain_gate_{false};
  bool disable_loop_closing_during_near_chain_gate_{false};
  bool enable_spatial_memory_gate_{false};
  bool spatial_memory_use_pose_graph_anchors_{false};
  bool enable_spatial_passage_hysteresis_{false};
  bool enable_post_confirmation_{true};
  bool enable_roughness_place_recognition_observer_{true};
  bool enable_roughness_place_scan_match_observer_{false};
  bool roughness_place_evaluate_both_directions_{false};
  bool roughness_place_enable_reference_chain_observer_{false};
  bool imu_qos_reliable_{false};
  bool scan_qos_reliable_{false};
  bool enable_scan_conditioning_{false};
  bool scan_conditioning_use_nearest_imu_{true};
  bool scan_conditioning_throttle_drop_hard_{true};
  bool scan_conditioning_quarantine_fail_open_on_imu_fault_{true};
  bool scan_conditioning_residual_promote_filtered_reference_{false};
  std::string scan_conditioning_mode_name_{"observe"};
  std::string scan_conditioning_stabilize_fallback_mode_{"drop"};
  std::string scan_conditioning_residual_fallback_mode_{"observe"};
  std::string scan_conditioning_residual_direction_name_{"closer"};
  ScanConditioningMode scan_conditioning_mode_{ScanConditioningMode::kObserve};
  ResidualFilterDirection scan_conditioning_residual_direction_{
    ResidualFilterDirection::kCloser};
  double scan_conditioning_max_scan_imu_dt_s_{0.2};
  double scan_conditioning_hold_duration_s_{0.1};
  double scan_conditioning_hard_hold_duration_s_{0.0};
  double scan_conditioning_hard_score_gate_{3.0};
  double scan_conditioning_throttle_min_publish_period_s_{0.5};
  double scan_conditioning_max_continuous_drop_duration_s_{0.0};
  double scan_conditioning_stabilize_max_source_age_s_{0.75};
  double scan_conditioning_stabilize_max_event_duration_s_{0.5};
  double scan_conditioning_stabilize_min_publish_period_s_{0.1};
  double scan_conditioning_stabilize_max_translation_m_{0.2};
  double scan_conditioning_stabilize_max_yaw_rad_{0.15};
  double scan_conditioning_stabilize_max_roll_pitch_rad_{0.05};
  double scan_conditioning_stabilize_min_cached_valid_ratio_{0.2};
  double scan_conditioning_stabilize_min_output_valid_ratio_{0.15};
  double scan_conditioning_residual_max_source_age_s_{0.75};
  double scan_conditioning_residual_max_translation_m_{0.2};
  double scan_conditioning_residual_max_yaw_rad_{0.15};
  double scan_conditioning_residual_max_roll_pitch_rad_{0.05};
  double scan_conditioning_residual_threshold_m_{0.08};
  double scan_conditioning_residual_min_cached_valid_ratio_{0.2};
  double scan_conditioning_residual_min_overlap_ratio_{0.35};
  double scan_conditioning_residual_min_output_valid_ratio_{0.2};
  double scan_conditioning_residual_max_removed_ratio_{0.03};
  double scan_conditioning_quarantine_min_valid_ratio_{0.2};
  std::size_t scan_conditioning_stabilize_min_output_valid_beams_{20};
  std::size_t scan_conditioning_residual_neighbor_bins_{1};
  std::size_t scan_conditioning_residual_min_overlap_beams_{100};
  std::size_t scan_conditioning_residual_min_output_valid_beams_{100};
  RoughnessRecoveryConfig scan_conditioning_recovery_config_;
  ImuQualityRequirements scan_conditioning_imu_requirements_;
  RoughnessThresholds roughness_thresholds_;
  std::unordered_set<std::int64_t> experimental_reject_scan_stamps_ns_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::TimerBase::SharedPtr pending_timer_;
  std::shared_ptr<rclcpp::Service<slam_toolbox::srv::ClearQueue>> clear_queue_service_;
  std::shared_ptr<rclcpp::Service<slam_toolbox::srv::ClearQueue>>
  finalize_pending_service_;
  std::chrono::steady_clock::time_point last_diagnostics_at_;

  mutable std::mutex roughness_mutex_;
  mutable std::mutex frontend_queue_mutex_;
  mutable std::mutex mapper_epoch_mutex_;
  mutable std::mutex scan_queue_mutex_;
  mutable std::mutex scan_conditioning_mutex_;
  std::condition_variable scan_queue_cv_;
  std::deque<ImuSample> imu_samples_;
  std::deque<PendingScan> pending_scans_;
  std::deque<QueuedScan> scan_queue_;
  std::map<std::string, karto::LocalizedRangeScan *> preview_last_scans_;
  std::map<std::string, karto::LocalizedRangeScanVector> preview_running_scans_;
  std::optional<ImuSample> previous_distinct_imu_;
  std::optional<sensor_msgs::msg::LaserScan> stable_conditioning_scan_;
  std::int64_t stable_conditioning_scan_stamp_ns_{0};
  std::int64_t scan_conditioning_last_scan_stamp_ns_{0};
  std::int64_t scan_conditioning_last_rough_stamp_ns_{0};
  std::int64_t scan_conditioning_last_hard_stamp_ns_{0};
  std::int64_t scan_conditioning_event_start_stamp_ns_{0};
  std::int64_t scan_conditioning_last_publish_stamp_ns_{0};
  std::int64_t scan_conditioning_drop_start_stamp_ns_{0};
  RoughnessRecoveryGate scan_conditioning_recovery_gate_;
  ImuUnavailableFailOpenGate scan_conditioning_imu_fail_open_gate_;
  std::thread scan_worker_;
  std::atomic<bool> stop_scan_worker_{false};
  std::atomic<bool> shutdown_started_{false};
  std::uint64_t next_scan_sequence_{0};

  mutable std::mutex post_confirmation_mutex_;
  std::optional<PostConfirmationCandidate> pending_post_confirmation_;

  mutable std::mutex roughness_place_mutex_;
  RoughnessPlaceEventTracker roughness_place_event_tracker_;
  RoughnessPlaceScanChainTracker roughness_place_scan_chain_tracker_;
  std::vector<RoughnessPlaceEvent> roughness_place_events_;
  std::map<std::uint64_t, RoughnessPlaceScanChainEvent>
  roughness_place_scan_chain_events_;
  std::vector<RoughnessPlaceConstraintPassage>
  roughness_place_injected_passages_;
  std::vector<RoughnessPlaceConstraintPassage>
  roughness_place_reference_chain_observed_passages_;

  mutable std::mutex near_chain_state_mutex_;
  std::int64_t near_chain_last_scan_stamp_ns_{0};
  std::int64_t near_chain_recovery_until_stamp_ns_{0};
  std::int64_t near_chain_loop_closing_suppression_until_stamp_ns_{0};
  std::vector<RoughnessSpatialZone> roughness_spatial_zones_;
  std::vector<RoughnessPoseGraphAnchor> roughness_pose_graph_anchors_;
  AdaptiveRoughnessMemory adaptive_roughness_memory_;
  RoughnessLoopMemory roughness_loop_memory_;
  NearChainRotationConsistency near_chain_rotation_consistency_;
  std::size_t near_chain_observer_parity_candidate_streak_{0};
  bool near_chain_observer_parity_trusted_{false};

  std::atomic<std::uint64_t> mapper_generation_{0};
  OrderedStampAudit received_scan_stamp_audit_;
  OrderedStampAudit received_imu_stamp_audit_;
  std::atomic<std::uint64_t> received_scans_{0};
  std::atomic<std::uint64_t> received_imus_{0};
  std::atomic<std::uint64_t> queued_scans_{0};
  std::atomic<std::uint64_t> scan_worker_in_flight_{0};
  std::atomic<std::uint64_t> scan_worker_completed_{0};
  std::atomic<std::uint64_t> complete_windows_{0};
  std::atomic<std::uint64_t> rough_scans_{0};
  std::atomic<std::uint64_t> innovation_candidates_{0};
  std::atomic<std::uint64_t> rejected_scans_{0};
  std::atomic<std::uint64_t> unavailable_innovations_{0};
  std::atomic<std::uint64_t> duplicate_imu_samples_{0};
  std::atomic<std::uint64_t> out_of_order_imu_samples_{0};
  std::atomic<std::uint64_t> imu_epoch_resets_{0};
  std::atomic<std::uint64_t> pending_overflow_fail_open_{0};
  std::atomic<std::uint64_t> stale_mapper_generation_scan_drop_count_{0};
  std::atomic<std::uint64_t> post_confirmation_armed_{0};
  std::atomic<std::uint64_t> post_confirmation_evaluated_{0};
  std::atomic<std::uint64_t> post_confirmation_confirmed_transient_{0};
  std::atomic<std::uint64_t> post_confirmation_unavailable_{0};
  std::atomic<std::uint64_t> post_confirmation_expired_{0};
  std::atomic<std::uint64_t> post_confirmation_overlap_{0};
  std::atomic<std::uint64_t> roughness_place_event_started_{0};
  std::atomic<std::uint64_t> roughness_place_event_finalized_{0};
  std::atomic<std::uint64_t> roughness_place_descriptor_valid_{0};
  std::atomic<std::uint64_t> roughness_place_candidate_available_{0};
  std::atomic<std::uint64_t> roughness_place_timestamp_reset_{0};
  std::atomic<std::uint64_t> roughness_place_scan_match_attempted_{0};
  std::atomic<std::uint64_t> roughness_place_scan_match_available_{0};
  std::atomic<std::uint64_t> roughness_place_scan_match_fail_open_{0};
  std::atomic<std::uint64_t> roughness_place_direction_hypotheses_{0};
  std::atomic<std::uint64_t> roughness_place_direction_overrides_{0};
  std::atomic<std::uint64_t> roughness_place_direction_fail_open_{0};
  std::atomic<std::uint64_t> roughness_place_reference_chain_attempted_{0};
  std::atomic<std::uint64_t> roughness_place_reference_chain_available_{0};
  std::atomic<std::uint64_t> roughness_place_reference_chain_fail_open_{0};
  std::atomic<std::uint64_t>
  roughness_place_reference_chain_event_accepted_{0};
  std::atomic<std::uint64_t>
  roughness_place_reference_chain_correlated_duplicate_{0};
  std::atomic<std::uint64_t> roughness_place_quality_accepted_{0};
  std::atomic<std::uint64_t> roughness_place_quality_rejected_{0};
  std::atomic<std::uint64_t> roughness_place_quality_fail_open_{0};
  std::atomic<std::uint64_t> roughness_place_constraint_injected_{0};
  std::atomic<std::uint64_t> roughness_place_constraint_duplicate_{0};
  std::atomic<std::uint64_t>
  roughness_place_constraint_correlated_duplicate_{0};
  std::atomic<std::uint64_t> roughness_place_constraint_failure_{0};
  std::atomic<std::uint64_t> near_chain_diag_count_{0};
  std::atomic<std::uint64_t> near_chain_rough_gate_count_{0};
  std::atomic<std::uint64_t> near_chain_spatial_revisit_gate_count_{0};
  std::atomic<std::uint64_t> near_chain_spatial_passage_gate_count_{0};
  std::atomic<std::uint64_t> near_chain_recovery_gate_count_{0};
  std::atomic<std::uint64_t> near_chain_incomplete_fail_open_count_{0};
  std::atomic<std::uint64_t> near_chain_timestamp_reset_count_{0};
  std::atomic<std::uint64_t> near_chain_threshold_override_count_{0};
  std::atomic<std::uint64_t> near_chain_threshold_restore_failure_count_{0};
  std::atomic<std::uint64_t> near_chain_loop_closing_suppression_count_{0};
  std::atomic<std::uint64_t> near_chain_loop_closing_cooldown_count_{0};
  std::atomic<std::uint64_t> near_chain_loop_closing_restore_failure_count_{0};
  std::atomic<std::uint64_t> near_chain_new_edge_count_{0};
  std::atomic<std::uint64_t> near_chain_new_nonsequential_edge_count_{0};
  std::atomic<std::uint64_t> near_chain_exact_candidate_count_{0};
  std::atomic<std::uint64_t> near_chain_exact_observer_unavailable_count_{0};
  std::atomic<std::uint64_t> near_chain_exact_parity_failure_count_{0};
  std::atomic<std::uint64_t> near_chain_quality_available_count_{0};
  std::atomic<std::uint64_t> near_chain_adaptive_accept_count_{0};
  std::atomic<std::uint64_t> near_chain_adaptive_downweight_shadow_count_{0};
  std::atomic<std::uint64_t> near_chain_adaptive_reject_count_{0};
  std::atomic<std::uint64_t> near_chain_adaptive_fail_open_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_batch_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_risky_candidate_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_proposed_reject_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_applied_reject_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_fail_open_count_{0};
  std::atomic<std::uint64_t> near_chain_rotation_timestamp_reset_count_{0};
  std::atomic<std::uint64_t> loop_closure_candidate_count_{0};
  std::atomic<std::uint64_t> loop_closure_accepted_count_{0};
  std::atomic<std::uint64_t> loop_closure_actual_edge_count_{0};
  std::atomic<std::uint64_t> loop_closure_accepted_without_edge_count_{0};
  std::atomic<std::uint64_t> loop_closure_graph_warmup_count_{0};
  std::atomic<std::uint64_t> loop_closure_pairing_failure_count_{0};
  std::atomic<std::uint64_t> spatial_memory_zone_insert_count_{0};
  std::atomic<std::uint64_t> spatial_memory_zone_merge_count_{0};
  std::atomic<std::uint64_t> spatial_memory_zone_evict_count_{0};
  std::atomic<std::uint64_t> spatial_memory_invalid_pose_count_{0};
  std::atomic<std::uint64_t> spatial_memory_pose_graph_refresh_count_{0};
  std::atomic<std::uint64_t> spatial_memory_pose_graph_stale_remove_count_{0};
  std::atomic<std::uint64_t> spatial_memory_pose_graph_invalid_remove_count_{0};
  std::atomic<std::uint64_t> spatial_memory_pose_graph_prediction_failure_count_{0};
  std::atomic<std::uint64_t> spatial_memory_pose_graph_anchor_insert_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_allow_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_drop_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_bridge_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_quarantine_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_stabilized_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_residual_filtered_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_fallback_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_imu_quality_reject_count_{0};
  std::atomic<std::uint64_t> scan_conditioning_tf_failure_count_{0};
};

}  // namespace slam_2d_core

#endif  // SLAM_2D_CORE__ROUGHNESS_AWARE_SLAM_TOOLBOX_HPP_
