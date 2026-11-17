#ifndef SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_
#define SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "slam_3d_core/ground_motion_constraint.hpp"
#include "slam_3d_core/imu_relative_attitude_constraint.hpp"
#include "slam_3d_core/imu_shock_policy.hpp"
#include "slam_3d_core/indoor_terrain_evidence.hpp"
#include "slam_3d_core/pose_graph_optimizer.hpp"
#include "slam_3d_core/registration_information.hpp"
#include "slam_3d_core/registration_policy.hpp"
#include "slam_3d_core/spatial_mapping_policy.hpp"
#include "std_msgs/msg/int64.hpp"
#include "slam_3d_core/submap_policy.hpp"
#include "slam_3d_core/keyframe_archive.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "slam_3d_core/terrain_preview_evidence.hpp"
#include "slam_3d_core/voxel_surfel_registration.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

namespace gole
{
namespace slam_3d_core
{

class ScanToMapIcp : public rclcpp::Node
{
public:
  explicit ScanToMapIcp(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using PointT = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<PointT>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;
  using TerrainPointT = pcl::PointXYZI;
  using TerrainCloudT = pcl::PointCloud<TerrainPointT>;
  using TerrainCloudPtr = TerrainCloudT::Ptr;
  using TerrainCloudConstPtr = TerrainCloudT::ConstPtr;
  using TerrainSupportVector = std::vector<TerrainSupportPoint>;
  using TerrainSupportVectorPtr = std::shared_ptr<TerrainSupportVector>;
  using TerrainSupportVectorConstPtr = std::shared_ptr<const TerrainSupportVector>;
  using Matrix6d = Eigen::Matrix<double, 6, 6>;
  using Vector6d = Eigen::Matrix<double, 6, 1>;

  struct TerrainSensorScan
  {
    CloudPtr cloud_base;
    Eigen::Vector3d sensor_origin_base{Eigen::Vector3d::Zero()};
    std::vector<TerrainLabel> labels;
    std::vector<double> roughness_scores;
  };

  struct Keyframe
  {
    std::uint64_t archive_id{0};
    rclcpp::Time stamp;
    Eigen::Isometry3d map_T_base{Eigen::Isometry3d::Identity()};
    // Registration evidence may include support-surface points. Mapping
    // evidence is kept separate so published/saved maps can remain obstacle-only.
    CloudPtr cloud_base;
    CloudPtr mapping_cloud_base;
    // Preserve mapping views before terrain classification prunes its own copy.
    std::vector<TerrainSensorScan> archive_sensor_scans;
    std::vector<TerrainSensorScan> terrain_sensor_scans;
    CloudPtr cloud_map;
    CloudPtr mapping_cloud_map;
    std::vector<float> descriptor;
    int registration_only_support_points{0};
  };

  struct Submap
  {
    int id{-1};
    std::vector<std::size_t> keyframe_indices;
    CloudPtr cloud_map;
    CloudPtr mapping_cloud_map;
    // Completed submaps retain one immutable local geometry snapshot. Pose
    // graph updates move that snapshot rigidly with its anchor keyframe.
    CloudPtr frozen_cloud_anchor;
    CloudPtr frozen_mapping_cloud_anchor;
    std::size_t frozen_anchor_keyframe_index{0U};
    bool frozen_geometry_valid{false};
    int frozen_mapping_input_points{0};
    int frozen_mapping_confirmed_points{0};
    std::vector<float> descriptor;
    double accumulated_travel_m{0.0};
    bool completed{false};
    SubmapClosureReason closure_reason{SubmapClosureReason::kNone};
  };

  struct AnchoredTerrainTraversalSample
  {
    rclcpp::Time stamp;
    std::size_t anchor_keyframe_index{0U};
    Eigen::Vector3d position_anchor{Eigen::Vector3d::Zero()};
    TerrainTraversalResponse response{TerrainTraversalResponse::kUnknown};
    double roughness_score{0.0};
    bool continuity_break_before{false};
  };

  struct PoseGraphEdge
  {
    int from_index{0};
    int to_index{0};
    Eigen::Isometry3d from_T_to{Eigen::Isometry3d::Identity()};
    double translation_weight{1.0};
    double vertical_translation_weight{1.0};
    double rotation_weight{1.0};
    Matrix6d information_matrix{Matrix6d::Identity()};
    bool registration_information_used{false};
    int registration_observable_rank{0};
    double registration_hessian_condition{0.0};
    double registration_information_min_gain{1.0};
    bool loop_closure{false};
  };

  struct PoseGraphStats
  {
    bool applied{false};
    bool new_loop_rejected{false};
    bool same_layer_z_seed_applied{false};
    bool distributed_loop_seed_applied{false};
    int iterations{0};
    int nodes{0};
    int edges{0};
    int loop_edges{0};
    int robust_loop_edges_downweighted{0};
    int solver_stages{0};
    int solver_successful_steps{0};
    int solver_unsuccessful_steps{0};
    int dcs_downweighted_loop_edges{0};
    int registration_information_edges{0};
    int configured_diagonal_information_edges{0};
    int minimum_registration_observable_rank{0};
    int ground_sequence_applied_edges{0};
    double residual_before{0.0};
    double residual_after{0.0};
    double robust_objective_before{0.0};
    double robust_objective_after{0.0};
    double residual_before_ground_sequence{0.0};
    double robust_objective_before_ground_sequence{0.0};
    double min_loop_robust_weight{1.0};
    double new_loop_robust_weight{1.0};
    double dcs_min_scale{1.0};
    double new_loop_dcs_scale{1.0};
    double solver_first_stage_initial_cost{0.0};
    double solver_last_stage_final_cost{0.0};
    double solver_final_gradient_max_norm{0.0};
    double maximum_registration_hessian_condition{0.0};
    double minimum_registration_information_gain{1.0};
    double max_update_translation_m{0.0};
    double max_update_rotation_deg{0.0};
    double same_layer_z_seed_correction_m{0.0};
    double same_layer_z_seed_path_length_m{0.0};
    double same_layer_z_seed_max_increment_m{0.0};
    double distributed_loop_seed_path_length_m{0.0};
    double distributed_loop_seed_endpoint_translation_m{0.0};
    double distributed_loop_seed_endpoint_rotation_deg{0.0};
    double distributed_loop_seed_max_relative_translation_error_m{0.0};
    double distributed_loop_seed_max_relative_rotation_error_deg{0.0};
    double ground_sequence_max_raw_body_z_step_m{0.0};
    double ground_sequence_max_constrained_body_z_step_m{0.0};
    double optimization_runtime_ms{0.0};
    double iteration_runtime_ms{0.0};
    double mean_iteration_runtime_ms{0.0};
    std::string solver_status{"not_run"};
    std::string convergence_reason{"not_run"};
    std::string rejection_reason;
  };

  struct RegistrationResult
  {
    bool converged{false};
    bool accepted{false};
    double fitness{0.0};
    double inlier_ratio{0.0};
    double inlier_rmse_m{0.0};
    double correction_translation_m{0.0};
    double correction_rotation_deg{0.0};
    double registration_runtime_ms{0.0};
    double registration_scan_gap_s{0.0};
    double registration_translation_gate_m{0.0};
    double transform_determinant{1.0};
    double transform_orthogonality_error{0.0};
    double max_input_cloud_stamp_delta_s{0.0};
    int inlier_count{0};
    int iterations{0};
    int input_clouds_expected{1};
    int input_clouds_merged{1};
    int registration_only_clouds_merged{0};
    int registration_only_points_before_balance{0};
    int registration_only_points_after_balance{0};
    int mapping_scan_points_before_downsample{0};
    int mapping_scan_points_after_downsample{0};
    int scan_points_before_downsample{0};
    int scan_points_after_downsample{0};
    int consecutive_pose_quarantine_count{0};
    double scan_voxel_leaf_m{0.0};
    bool scan_voxel_adapted{false};
    bool shock_quarantined{false};
    bool pose_quarantined{false};
    bool pose_quarantine_tracking_reset{false};
    bool keyframe_added{false};
    bool keyframe_quarantined{false};
    bool deskew_attempted{false};
    bool deskew_applied{false};
    bool scan_to_scan_prediction_used{false};
    bool imu_rotation_prediction_used{false};
    bool place_recognition_attempted{false};
    bool place_recognition_skipped{false};
    bool place_recognition_accepted{false};
    bool place_recognition_registration_only_vertical_recovery_used{false};
    bool place_recognition_registration_only_consistency_relaxation_used{false};
    bool place_recognition_mapping_refinement_attempted{false};
    bool place_recognition_mapping_refinement_applied{false};
    bool place_recognition_mapping_refinement_verified{false};
    bool place_recognition_large_pose_jump_provisional{false};
    bool place_recognition_large_pose_jump_accepted{false};
    bool place_recognition_verified_large_loop_measurement_preserved{false};
    bool place_recognition_pose_metrics_evaluated{false};
    bool place_recognition_pose_metrics_valid{false};
    bool local_map_spatial_transition_bootstrap{false};
    bool pose_graph_correction_applied{false};
    bool pose_graph_new_loop_rejected{false};
    bool pose_graph_minimum_interval_override_applied{false};
    bool pose_graph_duplicate_submap_pair_override_applied{false};
    bool pose_graph_same_layer_z_constraint_evaluated{false};
    bool pose_graph_same_layer_z_constraint_applied{false};
    bool pose_graph_same_layer_z_seed_applied{false};
    bool pose_graph_high_confidence_large_loop{false};
    bool pose_graph_distributed_loop_seed_applied{false};
    bool gravity_accel_used{false};
    bool ground_motion_constraint_evaluated{false};
    bool ground_motion_constraint_applied{false};
    bool ground_motion_constraint_roughness_relaxed{false};
    bool ground_motion_post_constraint_quality_evaluated{false};
    bool ground_motion_post_constraint_quality_accepted{false};
    bool imu_relative_attitude_constraint_evaluated{false};
    bool imu_relative_attitude_constraint_applied{false};
    bool tracking_scan_updated_during_shock{false};
    bool tracking_scan_updated_after_rejection{false};
    bool imu_roughness_evaluated{false};
    bool imu_roughness_event{false};
    bool imu_roughness_stable_evidence{false};
    bool imu_roughness_recent_event{false};
    bool imu_shock_active{false};
    bool imu_shock_time_reset{false};
    bool imu_shock_just_recovered{false};
    bool path_pose_appended{false};
    bool constant_velocity_prediction_suppressed_by_shock{false};
    bool registration_trimmed_correspondences_applied{false};
    bool registration_refinement_attempted{false};
    bool registration_refinement_applied{false};
    bool registration_gap_relaxation_applied{false};
    bool surfel_converged{false};
    bool registration_information_valid{false};
    bool pose_graph_new_loop_registration_information_used{false};
    bool cloud_sync_wait_timed_out{false};
    bool cloud_sync_watermark_resolved{false};
    bool primary_cloud_gap_before{false};
    bool inter_lidar_wheel_wait_timed_out{false};
    bool inter_lidar_gap_recovery_hypothesis_attempted{false};
    bool inter_lidar_gap_recovery_selected_uncompensated{false};
    bool deskew_imu_watermark_waited{false};
    bool deskew_imu_wait_timed_out{false};
    std::int64_t deskew_required_imu_stamp_ns{0};
    std::int64_t deskew_dispatch_imu_stamp_ns{0};
    std::int64_t imu_snapshot_end_stamp_ns{0};
    int deskew_points_with_time{0};
    int deskew_point_count{0};
    int imu_snapshot_sample_count{0};
    int inter_lidar_motion_attempted_count{0};
    int inter_lidar_motion_applied_count{0};
    int inter_lidar_motion_dropped_count{0};
    int imu_roughness_sample_count{0};
    int imu_shock_recovery_stable_scan_count{0};
    int place_recognition_candidates{0};
    int place_recognition_eligible_descriptors{0};
    int place_recognition_registration_only_vertical_recovery_descriptors{0};
    int place_recognition_candidate_index{-1};
    int place_recognition_candidate_submap_id{-1};
    int place_recognition_candidate_consistency_count{0};
    int place_recognition_mapping_refinement_source_points{0};
    int place_recognition_mapping_refinement_target_points{0};
    int place_recognition_mapping_refinement_observable_rank{0};
    int pose_graph_correction_keyframes{0};
    int pose_graph_optimization_iterations{0};
    int pose_graph_edges{0};
    int pose_graph_loop_edges{0};
    int pose_graph_robust_loop_edges_downweighted{0};
    int pose_graph_solver_stages{0};
    int pose_graph_solver_successful_steps{0};
    int pose_graph_solver_unsuccessful_steps{0};
    int pose_graph_dcs_downweighted_loop_edges{0};
    int pose_graph_duplicate_submap_pair_previous_edges{0};
    int pose_graph_same_layer_reference_keyframe_index{-1};
    int pose_graph_ground_sequence_applied_edges{0};
    int registration_coarse_iterations{0};
    int registration_refinement_iterations{0};
    int surfel_target_voxels{0};
    int surfel_planar_surfels{0};
    int surfel_correspondences{0};
    int surfel_observable_rank{0};
    int registration_information_observable_rank{0};
    int pose_graph_registration_information_edges{0};
    int pose_graph_configured_diagonal_information_edges{0};
    int pose_graph_minimum_registration_observable_rank{0};
    double deskew_time_ratio{0.0};
    double deskew_scan_duration_s{0.0};
    double inter_lidar_motion_max_stamp_delta_s{0.0};
    double inter_lidar_motion_max_translation_m{0.0};
    double inter_lidar_motion_max_rotation_deg{0.0};
    double inter_lidar_gap_recovery_compensated_score{-1.0};
    double inter_lidar_gap_recovery_uncompensated_score{-1.0};
    double inter_lidar_gap_recovery_uncompensated_fitness{-1.0};
    double inter_lidar_gap_recovery_uncompensated_inlier_ratio{-1.0};
    double inter_lidar_gap_recovery_uncompensated_inlier_rmse_m{-1.0};
    double inter_lidar_gap_recovery_runtime_ms{0.0};
    double scan_to_scan_reference_age_s{0.0};
    double place_recognition_descriptor_distance{0.0};
    double place_recognition_nearest_descriptor_distance{0.0};
    double place_recognition_yaw_offset_deg{0.0};
    double place_recognition_candidate_consistency_translation_m{0.0};
    double place_recognition_candidate_consistency_rotation_deg{0.0};
    double place_recognition_candidate_consistency_translation_limit_m{0.0};
    double place_recognition_prediction_jump_translation_m{0.0};
    double place_recognition_prediction_jump_rotation_deg{0.0};
    double place_recognition_map_correction_translation_m{0.0};
    double place_recognition_map_correction_rotation_deg{0.0};
    double place_recognition_alignment_correction_translation_m{0.0};
    double place_recognition_alignment_correction_rotation_deg{0.0};
    double place_recognition_mapping_refinement_before_inlier_ratio{0.0};
    double place_recognition_mapping_refinement_before_inlier_rmse_m{0.0};
    double place_recognition_mapping_refinement_after_inlier_ratio{0.0};
    double place_recognition_mapping_refinement_after_inlier_rmse_m{0.0};
    double place_recognition_mapping_refinement_before_target_coverage_ratio{0.0};
    double place_recognition_mapping_refinement_after_target_coverage_ratio{0.0};
    double place_recognition_mapping_refinement_before_score{0.0};
    double place_recognition_mapping_refinement_after_score{0.0};
    double place_recognition_mapping_refinement_primary_before_score{0.0};
    double place_recognition_mapping_refinement_primary_after_score{0.0};
    double place_recognition_mapping_refinement_correction_translation_m{0.0};
    double place_recognition_mapping_refinement_correction_rotation_deg{0.0};
    double place_recognition_mapping_refinement_hessian_condition{0.0};
    double place_recognition_mapping_refinement_runtime_ms{0.0};
    double place_recognition_candidate_registration_inlier_ratio{0.0};
    double place_recognition_candidate_registration_inlier_rmse_m{0.0};
    double place_recognition_candidate_pose_jump_translation_m{0.0};
    double place_recognition_candidate_pose_jump_rotation_deg{0.0};
    double pose_graph_correction_translation_m{0.0};
    double pose_graph_correction_rotation_deg{0.0};
    double pose_graph_constraint_inlier_ratio{0.0};
    double pose_graph_constraint_inlier_rmse_m{0.0};
    double pose_graph_constraint_target_coverage_ratio{0.0};
    double pose_graph_minimum_interval_elapsed_s{-1.0};
    double pose_graph_residual_before{0.0};
    double pose_graph_residual_after{0.0};
    double pose_graph_robust_objective_before{0.0};
    double pose_graph_robust_objective_after{0.0};
    double pose_graph_residual_before_ground_sequence{0.0};
    double pose_graph_robust_objective_before_ground_sequence{0.0};
    double pose_graph_min_loop_robust_weight{1.0};
    double pose_graph_new_loop_robust_weight{1.0};
    double pose_graph_dcs_min_scale{1.0};
    double pose_graph_new_loop_dcs_scale{1.0};
    double pose_graph_solver_first_stage_initial_cost{0.0};
    double pose_graph_solver_last_stage_final_cost{0.0};
    double pose_graph_solver_final_gradient_max_norm{0.0};
    double pose_graph_max_update_translation_m{0.0};
    double pose_graph_max_update_rotation_deg{0.0};
    double pose_graph_same_layer_reference_xy_distance_m{-1.0};
    double pose_graph_same_layer_matched_current_z_before_m{0.0};
    double pose_graph_same_layer_reference_z_m{0.0};
    double pose_graph_same_layer_requested_z_correction_m{0.0};
    double pose_graph_same_layer_graph_target_z_m{0.0};
    double pose_graph_same_layer_z_translation_weight{0.0};
    double pose_graph_same_layer_z_seed_correction_m{0.0};
    double pose_graph_same_layer_z_seed_path_length_m{0.0};
    double pose_graph_same_layer_z_seed_max_increment_m{0.0};
    double pose_graph_distributed_loop_seed_path_length_m{0.0};
    double pose_graph_distributed_loop_seed_endpoint_translation_m{0.0};
    double pose_graph_distributed_loop_seed_endpoint_rotation_deg{0.0};
    double pose_graph_distributed_loop_seed_max_relative_translation_error_m{0.0};
    double pose_graph_distributed_loop_seed_max_relative_rotation_error_deg{0.0};
    double pose_graph_ground_sequence_max_raw_body_z_step_m{0.0};
    double pose_graph_ground_sequence_max_constrained_body_z_step_m{0.0};
    double pose_graph_optimization_runtime_ms{0.0};
    double pose_graph_iteration_runtime_ms{0.0};
    double pose_graph_mean_iteration_runtime_ms{0.0};
    double gravity_accel_age_s{-1.0};
    double gravity_accel_norm_mps2{0.0};
    double gravity_filter_alpha{0.0};
    double gravity_roll_deg{0.0};
    double gravity_pitch_deg{0.0};
    double ground_motion_raw_body_z_step_m{0.0};
    double ground_motion_constrained_body_z_step_m{0.0};
    double ground_motion_max_body_z_step_m{0.0};
    double imu_relative_attitude_target_roll_deg{0.0};
    double imu_relative_attitude_target_pitch_deg{0.0};
    double imu_relative_attitude_raw_roll_error_deg{0.0};
    double imu_relative_attitude_raw_pitch_error_deg{0.0};
    double imu_relative_attitude_applied_roll_correction_deg{0.0};
    double imu_relative_attitude_applied_pitch_correction_deg{0.0};
    double imu_roughness_max_angular_velocity_radps{0.0};
    double imu_roughness_max_angular_accel_radps2{0.0};
    double imu_roughness_max_linear_accel_mps2{0.0};
    double imu_roughness_max_accel_norm_delta_mps2{0.0};
    double imu_roughness_max_linear_jerk_mps3{0.0};
    double imu_roughness_time_since_event_s{0.0};
    double imu_shock_time_since_event_s{0.0};
    double registration_quality_max_correspondence_m{0.0};
    double registration_coarse_fitness{0.0};
    double registration_coarse_inlier_ratio{0.0};
    double registration_coarse_inlier_rmse_m{0.0};
    double registration_refinement_fitness{0.0};
    double registration_refinement_inlier_ratio{0.0};
    double registration_refinement_inlier_rmse_m{0.0};
    double surfel_correspondence_ratio{0.0};
    double surfel_rmse_m{0.0};
    double surfel_median_abs_residual_m{0.0};
    double surfel_p95_abs_residual_m{0.0};
    double surfel_robust_cost{0.0};
    double surfel_robust_downweight_ratio{0.0};
    double surfel_hessian_condition{0.0};
    double surfel_target_build_time_ms{0.0};
    double surfel_optimization_time_ms{0.0};
    double surfel_total_time_ms{0.0};
    double registration_information_hessian_condition{0.0};
    double registration_information_evaluation_runtime_ms{0.0};
    double pose_graph_maximum_registration_hessian_condition{0.0};
    double pose_graph_minimum_registration_information_gain{1.0};
    double pose_graph_new_loop_information_min_gain{1.0};
    std::array<double, 6> surfel_hessian_eigenvalues{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    Eigen::Vector3d surfel_weak_rotation_axes{Eigen::Vector3d::Zero()};
    Eigen::Vector3d surfel_weak_translation_axes{Eigen::Vector3d::Zero()};
    Matrix6d registration_information_hessian_map_rt{Matrix6d::Zero()};
    Eigen::Isometry3d registration_information_pose_map_T_base{
      Eigen::Isometry3d::Identity()};
    Eigen::Isometry3d map_T_base{Eigen::Isometry3d::Identity()};
    std::string rejection_reason;
    std::string shock_quarantine_reason;
    std::string keyframe_quarantine_reason;
    std::string deskew_status;
    std::string deskew_point_time_field;
    std::string inter_lidar_motion_status{"disabled"};
    std::string inter_lidar_gap_recovery_selected_input{"not_attempted"};
    std::string scan_to_scan_prediction_status{"disabled"};
    std::string imu_rotation_prediction_status{"disabled"};
    std::string imu_roughness_reason{"disabled"};
    std::string imu_shock_phase{"disabled"};
    std::string imu_shock_observation{"unknown"};
    std::string place_recognition_trigger{"none"};
    std::string place_recognition_skip_reason;
    std::string place_recognition_candidate_registration_status{"not_evaluated"};
    std::string place_recognition_mapping_refinement_status{"disabled"};
    std::string pose_graph_rejection_reason;
    std::string pose_graph_minimum_interval_status{"not_evaluated"};
    std::string pose_graph_duplicate_submap_pair_status{"not_evaluated"};
    std::string pose_graph_new_loop_information_status{"not_evaluated"};
    std::string pose_graph_solver_status{"not_run"};
    std::string pose_graph_convergence_reason{"not_run"};
    std::string pose_graph_same_layer_reference_source{"not_evaluated"};
    std::string gravity_constraint_status{"disabled"};
    std::string ground_motion_constraint_status{"not_evaluated"};
    std::string imu_relative_attitude_constraint_status{"disabled"};
    std::string surfel_reason{"not_selected"};
    std::string registration_information_status{"disabled"};
  };

  struct DeskewStats
  {
    bool attempted{false};
    bool applied{false};
    int points_with_time{0};
    int point_count{0};
    double time_ratio{0.0};
    double scan_duration_s{0.0};
    double scan_start_time_s{0.0};
    double scan_end_time_s{0.0};
    bool have_scan_time_bounds{false};
    std::string status{"disabled"};
    std::string point_time_field;
  };

  struct InterLidarMotionStats
  {
    int attempted_count{0};
    int applied_count{0};
    int dropped_count{0};
    double max_stamp_delta_s{0.0};
    double max_translation_m{0.0};
    double max_rotation_deg{0.0};
    bool gap_recovery_hypothesis_attempted{false};
    bool gap_recovery_selected_uncompensated{false};
    double gap_recovery_compensated_score{-1.0};
    double gap_recovery_uncompensated_score{-1.0};
    double gap_recovery_uncompensated_fitness{-1.0};
    double gap_recovery_uncompensated_inlier_ratio{-1.0};
    double gap_recovery_uncompensated_inlier_rmse_m{-1.0};
    double gap_recovery_runtime_ms{0.0};
    std::string gap_recovery_selected_input{"not_attempted"};
    std::string status{"disabled"};
  };

  struct ImuSample
  {
    rclcpp::Time stamp;
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
    bool orientation_valid{false};
  };

  struct GravitySample
  {
    rclcpp::Time stamp;
    Eigen::Vector3d accel_base{Eigen::Vector3d::UnitZ()};
    double filter_alpha{0.0};
  };

  struct PendingPrimaryCloud
  {
    sensor_msgs::msg::PointCloud2::SharedPtr message;
    std::chrono::steady_clock::time_point queued_at;
    std::int64_t required_imu_stamp_ns{0};
    std::int64_t scan_start_stamp_ns{0};
    std::int64_t scan_end_stamp_ns{0};
    bool have_required_imu_stamp{false};
    bool have_scan_time_bounds{false};
    bool imu_watermark_waited{false};
    bool terrain_continuity_break_before{false};
  };

  struct PendingAuxiliaryCloud
  {
    sensor_msgs::msg::PointCloud2::SharedPtr message;
    std::int64_t required_imu_stamp_ns{0};
    std::int64_t scan_start_stamp_ns{0};
    std::int64_t scan_end_stamp_ns{0};
    bool have_required_imu_stamp{false};
    bool have_scan_time_bounds{false};
  };

  struct AuxiliaryCloudInput
  {
    std::string name;
    std::string topic;
    bool enabled{false};
    bool registration_only{false};
    std::deque<PendingAuxiliaryCloud> queue;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;
    std::uint64_t consumed_count{0};
    std::uint64_t missing_count{0};
    std::uint64_t stale_drop_count{0};
    std::uint64_t queue_overflow_count{0};
    std::uint64_t duplicate_count{0};
    std::uint64_t out_of_order_count{0};
    std::int64_t latest_stamp_ns{0};
    bool have_latest_stamp{false};
  };

  struct RegistrationOnlyCloudStats
  {
    int clouds_merged{0};
    int points_before_balance{0};
    int points_after_balance{0};
  };

  struct SynchronizedCloudSet
  {
    sensor_msgs::msg::PointCloud2::SharedPtr primary;
    std::array<sensor_msgs::msg::PointCloud2::SharedPtr, 2> auxiliary;
    int expected_cloud_count{1};
    int merged_cloud_count{1};
    double max_stamp_delta_s{0.0};
    bool wait_timed_out{false};
    bool watermark_resolved{false};
    bool wheel_wait_timed_out{false};
    bool imu_watermark_waited{false};
    bool imu_wait_timed_out{false};
    bool terrain_continuity_break_before{false};
    std::int64_t required_imu_stamp_ns{0};
    std::int64_t scan_start_stamp_ns{0};
    std::int64_t scan_end_stamp_ns{0};
    bool have_scan_time_bounds{false};
    std::int64_t dispatch_imu_stamp_ns{0};
    std::int64_t imu_snapshot_end_stamp_ns{0};
    std::vector<ImuSample> imu_samples;
    std::vector<GravitySample> gravity_samples;
  };

  struct WheelOdomSample
  {
    rclcpp::Time stamp;
    Eigen::Isometry3d odom_T_base{Eigen::Isometry3d::Identity()};
  };

  struct ImuRoughnessStats
  {
    bool evaluated{false};
    bool event{false};
    bool stable_evidence{false};
    bool recent_event{false};
    bool shock_active{false};
    bool shock_time_reset{false};
    bool shock_just_recovered{false};
    int sample_count{0};
    int shock_recovery_stable_scan_count{0};
    double max_angular_velocity_radps{0.0};
    double max_angular_accel_radps2{0.0};
    double max_linear_accel_mps2{0.0};
    double max_accel_norm_delta_mps2{0.0};
    double max_linear_jerk_mps3{0.0};
    double max_translational_response_score{0.0};
    std::int64_t max_translational_response_stamp_ns{0};
    double time_since_event_s{0.0};
    double shock_time_since_event_s{0.0};
    std::string reason{"disabled"};
    std::string shock_phase{"disabled"};
    std::string shock_observation{"unknown"};
  };

  void setupParameters();
  void setupRosInterfaces();

  void primaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void auxiliaryCloudCallback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg,
    std::size_t sensor_index);
  void dispatchQueuedClouds();
  std::optional<SynchronizedCloudSet> takeSynchronizedCloudSet();
  void wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void mapPublishTimerCallback();

  void processCloud(const SynchronizedCloudSet & cloud_set);
  CloudPtr buildBaseFrameScan(
    const SynchronizedCloudSet & cloud_set,
    DeskewStats & deskew_stats,
    InterLidarMotionStats & inter_lidar_motion_stats,
    int & merged_cloud_count,
    CloudPtr & mapping_scan_base,
    CloudPtr & uncompensated_scan_base,
    CloudPtr & uncompensated_mapping_scan_base,
    RegistrationOnlyCloudStats & registration_only_stats,
    std::vector<TerrainSensorScan> & terrain_sensor_scans,
    std::vector<TerrainSensorScan> & uncompensated_terrain_sensor_scans,
    bool build_uncompensated_hypothesis);
  CloudPtr convertAndTransformCloud(
    const sensor_msgs::msg::PointCloud2 & msg,
    const std::vector<ImuSample> & imu_samples,
    DeskewStats & deskew_stats,
    Eigen::Vector3d * sensor_origin_base = nullptr);
  CloudPtr filterCloud(const CloudConstPtr & cloud) const;
  CloudPtr downsampleCloud(const CloudConstPtr & cloud, double leaf_size) const;
  CloudPtr downsampleScanAdaptive(
    const CloudConstPtr & cloud,
    double & effective_leaf_size) const;
  CloudPtr balanceRegistrationOnlyCloud(
    const CloudConstPtr & cloud,
    std::size_t mapping_point_count) const;
  CloudPtr transformCloud(const CloudConstPtr & cloud, const Eigen::Isometry3d & transform) const;
  CloudPtr filterMapCloud(const CloudConstPtr & cloud) const;
  bool deskewCloudInBaseFrame(
    const sensor_msgs::msg::PointCloud2 & msg,
    const CloudPtr & cloud_base,
    const std::vector<ImuSample> & imu_samples,
    DeskewStats & stats) const;
  bool compensateAuxiliaryCloudToPrimaryStamp(
    CloudPtr & cloud_base,
    const rclcpp::Time & cloud_stamp,
    const rclcpp::Time & primary_stamp,
    const std::vector<ImuSample> & imu_samples,
    Eigen::Vector3d & sensor_origin_base,
    InterLidarMotionStats & stats);
  void applyDeskewStatsToResult(
    const DeskewStats & stats,
    RegistrationResult & result) const;
  void applyInterLidarMotionStatsToResult(
    const InterLidarMotionStats & stats,
    RegistrationResult & result) const;
  ImuRoughnessStats evaluateImuRoughness(
    const rclcpp::Time & start_stamp,
    const rclcpp::Time & end_stamp,
    bool have_start_stamp,
    const std::vector<ImuSample> & imu_samples) const;
  ImuRoughnessStats evaluateAndUpdateImuShock(
    const rclcpp::Time & cloud_stamp,
    const DeskewStats & deskew_stats,
    const std::vector<ImuSample> & imu_samples);
  void applyImuRoughnessStatsToResult(
    const ImuRoughnessStats & stats,
    RegistrationResult & result) const;

  std::optional<Eigen::Isometry3d> lookupTransformToBase(
    const sensor_msgs::msg::PointCloud2 & msg);
  std::optional<WheelOdomSample> wheelOdomAtLocked(
    const rclcpp::Time & stamp,
    bool update_prediction_diagnostics = true,
    bool require_bracket = false);
  Eigen::Isometry3d predictWithWheelOdom(
    const rclcpp::Time & stamp,
    bool allow_constant_velocity_fallback = true,
    const std::optional<double> & imu_pitch_rad = std::nullopt);
  Eigen::Isometry3d predictWithConstantVelocity(bool allow_extrapolation = true) const;
  Eigen::Isometry3d predictWithMotionPriors(
    const CloudConstPtr & scan_base,
    const std::vector<ImuSample> & imu_samples,
    const rclcpp::Time & previous_scan_stamp,
    const rclcpp::Time & current_scan_stamp,
    bool have_previous_scan_stamp,
    bool suppress_constant_velocity_prediction,
    bool & used_scan_to_scan,
    bool & used_imu_rotation_prediction,
    bool & constant_velocity_prediction_suppressed,
    std::string & imu_rotation_prediction_status,
    double & scan_to_scan_reference_age_s,
    std::string & scan_to_scan_prediction_status);
  std::optional<Eigen::Quaterniond> integrateImuRotationDelta(
    const rclcpp::Time & start_stamp,
    const rclcpp::Time & end_stamp,
    const std::vector<ImuSample> & imu_samples,
    std::string & status) const;
  std::optional<Eigen::Matrix3d> imuWorldBaseOrientationAt(
    const rclcpp::Time & stamp,
    const std::vector<ImuSample> & imu_samples,
    double max_time_delta_s,
    std::string & status) const;
  std::optional<Eigen::Quaterniond> integrateAngularVelocityRotationDelta(
    const rclcpp::Time & start_stamp,
    const rclcpp::Time & end_stamp,
    const std::vector<ImuSample> & imu_samples,
    double max_interval_s,
    double angular_velocity_scale,
    std::string & status) const;
  Eigen::Isometry3d clampRelativeMotion(const Eigen::Isometry3d & delta) const;
  Eigen::Isometry3d constrainMotionPriorFallback(
    const Eigen::Isometry3d & candidate) const;
  Eigen::Isometry3d constrainImuRelativeAttitude(
    const Eigen::Isometry3d & candidate,
    const std::optional<Eigen::Matrix3d> & current_world_R_base,
    const std::string & sample_status,
    RegistrationResult * result) const;
  Eigen::Isometry3d constrainPlanarPose(
    const Eigen::Isometry3d & candidate,
    const Eigen::Isometry3d & reference,
    const std::optional<GravitySample> & gravity_sample) const;
  std::optional<GravitySample> gravitySampleAt(
    const rclcpp::Time & stamp,
    const std::vector<GravitySample> & gravity_samples,
    bool & stale) const;
  void commitWheelPrediction();
  void recordAcceptedPose();
  void recordAcceptedPoseOrRebaseAfterShock();
  void rebaseAcceptedPoseHistory();
  void recordTrackingScan(
    const CloudConstPtr & scan_base,
    const Eigen::Isometry3d & pose,
    const rclcpp::Time & stamp);
  RegistrationResult registerScanToMap(
    const CloudConstPtr & scan_base,
    const CloudConstPtr & local_map,
    const Eigen::Isometry3d & prediction,
    double registration_scan_gap_s = 0.0) const;
  void evaluateRegistrationOverlap(
    const CloudT & aligned_scan,
    const CloudConstPtr & local_map,
    double max_correspondence_m,
    RegistrationResult & result) const;
  void refreshRegistrationQualityAtPose(
    const CloudConstPtr & scan_base,
    const CloudConstPtr & local_map,
    const Eigen::Isometry3d & prediction,
    RegistrationResult & result) const;
  bool registrationCorrectionWithinLimits(RegistrationResult & result) const;

  CloudPtr buildLocalMap(const Eigen::Isometry3d & center_pose) const;
  CloudPtr buildCandidateMapAroundKeyframe(int keyframe_index) const;
  CloudPtr buildCandidateMapAroundSubmap(std::size_t submap_index) const;
  CloudPtr buildCandidateMappingMapAroundKeyframe(
    int keyframe_index,
    int latest_candidate_exclusive) const;
  CloudPtr buildCandidateMappingMapAroundSubmap(std::size_t submap_index) const;
  void refinePlaceRecognitionWithMappingCloud(
    const CloudConstPtr & registration_scan_base,
    const CloudConstPtr & mapping_scan_base,
    const CloudConstPtr & registration_candidate_map,
    const CloudConstPtr & mapping_candidate_map,
    const Eigen::Isometry3d & candidate_seed_pose,
    RegistrationResult & result) const;
  void initializeFromFirstScan(
    const CloudConstPtr & registration_scan_base,
    const CloudConstPtr & mapping_scan_base,
    int registration_only_support_points,
    const std::vector<TerrainSensorScan> & terrain_sensor_scans,
    const rclcpp::Time & stamp,
    const std::optional<GravitySample> & gravity_sample,
    bool gravity_sample_stale);
  void addKeyframe(
    const CloudConstPtr & registration_scan_base,
    const CloudConstPtr & mapping_scan_base,
    int registration_only_support_points,
    const std::vector<TerrainSensorScan> & terrain_sensor_scans,
    const rclcpp::Time & stamp,
    const std::optional<GravitySample> & gravity_sample,
    bool gravity_sample_stale,
    const RegistrationResult * registration_result = nullptr);
  void attachRegistrationInformation(
    const CloudConstPtr & source,
    const CloudConstPtr & target,
    RegistrationResult & result) const;
  bool shouldAddKeyframe(const Eigen::Isometry3d & pose) const;
  std::string shockQuarantineReason(const RegistrationResult & result) const;
  std::vector<float> computeScanDescriptor(const CloudConstPtr & scan_base) const;
  double descriptorDistance(
    const std::vector<float> & lhs,
    const std::vector<float> & rhs,
    int & best_shift) const;
  bool tryPlaceRecognition(
    const CloudConstPtr & scan_base,
    const CloudConstPtr & mapping_scan_base,
    const std::vector<float> & scan_descriptor,
    const Eigen::Isometry3d & reference_pose,
    const rclcpp::Time & stamp,
    int registration_only_support_points,
    RegistrationResult & result);
  std::size_t submapAnchorKeyframeIndex(const Submap & submap) const;
  bool hasCompatibleSubmap(const Eigen::Isometry3d & center_pose) const;
  void closeActiveSubmapForSpatialTransition();
  void appendLatestKeyframeToSubmaps();
  CloudPtr buildConfirmedSubmapMappingCloudAnchor(
    const Submap & submap,
    std::size_t anchor_index) const;
  void refreshSubmap(std::size_t submap_index);
  void rebuildSubmaps();
  void pruneSubmapsAfterOldestKeyframeRemoval();
  int completedSubmapCount() const;
  int frozenCompletedSubmapCount() const;
  void refreshKeyframeCloudMap(std::size_t keyframe_index);
  void prunePoseGraphEdgesAfterOldestKeyframeRemoval();
  double poseGraphEdgeSquaredError(
    const PoseGraphEdge & edge,
    const std::vector<Eigen::Isometry3d> & poses) const;
  static PoseGraphConstraint poseGraphConstraintFromEdge(const PoseGraphEdge & edge);
  double poseGraphLoopRobustWeight(const PoseGraphEdge & edge, double squared_error) const;
  double poseGraphResidual(
    const std::vector<Eigen::Isometry3d> & poses,
    bool apply_robust_loop_kernel) const;
  PoseGraphStats optimizePoseGraphWithLoop(
    int anchor_index,
    const Eigen::Isometry3d & target_loop_pose,
    const std::vector<Eigen::Isometry3d> & previous_keyframe_poses,
    bool same_layer_z_constraint,
    double loop_vertical_translation_weight,
    bool distributed_loop_seed,
    const RegistrationInformationResult * registration_information);
  void updatePathFromKeyframeCorrections(
    const std::vector<Eigen::Isometry3d> & previous_keyframe_poses);
  void applyPoseGraphCorrection(
    const rclcpp::Time & stamp,
    const Eigen::Isometry3d & reference_pose,
    const std::optional<GravitySample> & gravity_sample,
    const std::optional<Eigen::Matrix3d> & current_world_R_base,
    const std::string & imu_attitude_sample_status,
    RegistrationResult & result);
  void rebuildGlobalMap();
  void exportKeyframes(
    const std::shared_ptr<std_srvs::srv::Trigger::Response> & response);
  void classifyKeyframeTerrain(
    Keyframe & keyframe,
    const std::optional<GravitySample> & gravity_sample,
    bool gravity_sample_stale);
  void appendKeyframeTerrainEvidence(const Keyframe & keyframe);
  void rebuildTerrainEvidenceMap();
  void refreshTerrainEvidenceCloud();
  void refreshTerrainPreviewCloud();
  void recordTerrainTraversalSample(
    const rclcpp::Time & stamp,
    const RegistrationResult & result);
  bool appendTerrainTraversalSample(
    const rclcpp::Time & stamp,
    const Eigen::Vector3d & position_map,
    TerrainTraversalResponse response,
    double roughness_score,
    bool continuity_break_before = false);
  std::optional<Eigen::Vector3d> anchoredTerrainTraversalPosition(
    const AnchoredTerrainTraversalSample & sample,
    const std::vector<Eigen::Isometry3d> * keyframe_poses = nullptr) const;
  void clearPendingTerrainShock(bool count_unlocalized);
  void updateTerrainTraversalSamplesFromKeyframeCorrections(
    const std::vector<Eigen::Isometry3d> & previous_keyframe_poses);
  void pruneTerrainTraversalSamplesAfterOldestKeyframeRemoval();
  void refreshMapPublishSnapshots();

  void publishState(const rclcpp::Time & stamp, RegistrationResult & result);
  void publishPoseGraphMarkers(const rclcpp::Time & stamp);
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const CloudConstPtr & cloud,
    const std::string & frame_id,
    const rclcpp::Time & stamp) const;
  void publishTerrainCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const TerrainCloudConstPtr & cloud,
    const std::string & frame_id,
    const rclcpp::Time & stamp) const;
  void publishTerrainColorCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const TerrainCloudConstPtr & cloud,
    const std::string & frame_id,
    const rclcpp::Time & stamp) const;
  void publishTerrainSupportCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const TerrainSupportVectorConstPtr & evidence,
    const std::string & frame_id,
    const rclcpp::Time & stamp) const;
  void publishDiagnostics(const rclcpp::Time & stamp, const RegistrationResult & result);

  static Eigen::Isometry3d odomToIsometry(const nav_msgs::msg::Odometry & msg);
  static geometry_msgs::msg::Pose isometryToPose(const Eigen::Isometry3d & pose);
  static geometry_msgs::msg::TransformStamped isometryToTransform(
    const Eigen::Isometry3d & pose,
    const std::string & parent_frame,
    const std::string & child_frame,
    const rclcpp::Time & stamp);
  static double yawFromIsometry(const Eigen::Isometry3d & pose);

  std::string map_frame_;
  std::string odom_frame_;
  std::string base_frame_;
  std::string primary_cloud_topic_;
  std::string wheel_odom_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string path_topic_;
  std::string local_map_topic_;
  std::string global_map_topic_;
  std::string keyframe_archive_directory_;
  double keyframe_archive_voxel_leaf_m_{0.22};
  std::uint64_t next_keyframe_archive_id_{0};
  std::uint64_t pruned_keyframe_count_{0};
  std::string terrain_evidence_topic_;
  std::string terrain_evidence_colored_topic_;
  std::string terrain_support_topic_;
  std::string terrain_preview_evidence_topic_;
  std::string pose_graph_marker_topic_;
  std::string health_topic_;
  std::string imu_watermark_topic_;
  std::string preprocessed_scan_topic_;
  std::string registration_method_;
  std::string inter_lidar_motion_compensation_source_{"wheel_odom"};
  std::string inter_lidar_motion_failure_policy_;
  std::string wheel_odom_expected_frame_;
  std::string wheel_odom_expected_child_frame_;
  std::string imu_expected_frame_;
  std::string mapping_spatial_mode_{"planar"};

  bool cloud_subscription_reliable_{false};
  bool cloud_sync_wait_for_nearest_candidate_{false};
  bool cloud_sync_allow_wall_timeout_fallback_{true};
  bool imu_subscription_reliable_{false};
  bool use_tf_transforms_{true};
  bool reject_cloud_on_tf_failure_{true};
  bool reject_cloud_on_deskew_failure_{false};
  bool use_wheel_prediction_{true};
  bool use_constant_velocity_prediction_{true};
  bool use_scan_to_scan_prediction_{false};
  bool use_scan_to_scan_fallback_{false};
  bool add_scan_to_scan_fallback_keyframes_{false};
  bool use_motion_prior_fallback_{false};
  bool add_motion_prior_fallback_keyframes_{false};
  bool adaptive_scan_voxel_enabled_{false};
  bool quarantine_keyframes_on_shock_{false};
  bool quarantine_pose_on_shock_{false};
  bool record_tracking_scan_on_pose_quarantine_{false};
  bool record_tracking_scan_during_shock_{false};
  bool place_recognition_enabled_{false};
  bool place_recognition_periodic_enabled_{false};
  bool place_recognition_require_pose_graph_correction_{false};
  bool place_recognition_mapping_cloud_refinement_enabled_{false};
  bool place_recognition_mapping_refinement_required_for_pose_graph_{false};
  bool place_recognition_mapping_refinement_large_pose_jump_enabled_{false};
  bool place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled_{false};
  bool pose_graph_correction_enabled_{false};
  bool pose_graph_planar_correction_enabled_{false};
  bool pose_graph_robust_loop_enabled_{false};
  bool pose_graph_distributed_loop_seed_enabled_{false};
  bool pose_graph_minimum_interval_override_single_floor_z_enabled_{false};
  bool planar_pose_constraint_enabled_{false};
  bool gravity_roll_pitch_constraint_enabled_{false};
  bool imu_relative_roll_pitch_constraint_enabled_{false};
  bool map_z_filter_enabled_{false};
  bool deskew_enabled_{false};
  bool deskew_wait_for_imu_watermark_{false};
  bool inter_lidar_motion_compensation_enabled_{false};
  bool inter_lidar_gap_recovery_hypothesis_enabled_{false};
  bool wheel_odom_require_expected_frames_{true};
  bool imu_require_expected_frame_{false};
  bool imu_rotation_prediction_enabled_{false};
  bool imu_roughness_detection_enabled_{false};
  bool imu_shock_policy_enabled_{false};
  bool suppress_constant_velocity_on_imu_shock_{true};
  bool omit_shock_poses_from_path_{true};
  bool robust_registration_enabled_{false};
  bool registration_gap_relaxation_enabled_{false};
  bool submap_enabled_{false};
  bool place_recognition_use_submaps_{false};
  bool publish_tf_{false};
  bool publish_preprocessed_scan_{false};
  bool crop_box_enabled_{true};
  bool initialized_{false};
  std::atomic_bool processing_{false};
  std::atomic_bool have_dispatched_primary_stamp_{false};
  std::atomic<std::int64_t> latest_dispatched_primary_stamp_ns_{0};

  double max_secondary_stamp_delta_s_{0.05};
  double max_auxiliary_stamp_delta_s_{0.05};
  double cloud_sync_wait_timeout_s_{0.05};
  double cloud_sync_reorder_window_s_{0.02};
  double cloud_processing_period_s_{0.005};
  double min_range_m_{0.5};
  double max_range_m_{80.0};
  double z_min_m_{-5.0};
  double z_max_m_{8.0};
  double scan_voxel_leaf_m_{0.25};
  double adaptive_scan_voxel_leaf_m_{0.15};
  double registration_only_cloud_voxel_leaf_m_{0.30};
  double registration_only_cloud_max_point_ratio_{0.50};
  double map_voxel_leaf_m_{0.35};
  double local_map_radius_m_{35.0};
  double local_map_vertical_radius_m_{0.0};
  double keyframe_translation_m_{0.8};
  double keyframe_rotation_deg_{8.0};
  double keyframe_vertical_translation_m_{0.0};
  double keyframe_rotation_3d_deg_{0.0};
  double icp_max_correspondence_m_{2.0};
  double fitness_score_threshold_{2.0};
  double robust_registration_trim_ratio_{0.85};
  double robust_registration_refinement_max_correspondence_m_{0.8};
  double robust_registration_quality_max_correspondence_m_{0.8};
  double constant_velocity_translation_limit_m_{2.0};
  double constant_velocity_rotation_limit_deg_{30.0};
  double wheel_odom_buffer_duration_s_{30.0};
  double wheel_odom_max_time_delta_s_{0.15};
  double wheel_odom_max_interpolation_gap_s_{0.25};
  bool wheel_imu_slope_z_prediction_enabled_{false};
  double wheel_imu_slope_z_prediction_max_pitch_deg_{25.0};
  double wheel_imu_slope_z_prediction_weight_{0.5};
  double wheel_odom_timestamp_reset_threshold_s_{0.5};
  double motion_prior_fallback_max_step_m_{0.0};
  double motion_prior_fallback_max_z_step_m_{0.0};
  double motion_prior_fallback_max_untrusted_distance_m_{0.0};
  double motion_prior_fallback_max_untrusted_z_m_{0.0};
  double motion_prior_fallback_max_rotation_deg_{0.0};
  double max_registration_correction_translation_m_{4.0};
  double max_registration_correction_rotation_deg_{60.0};
  double registration_gap_relaxation_min_gap_s_{0.5};
  double registration_gap_relaxation_max_speed_mps_{0.0};
  double registration_gap_relaxation_max_translation_m_{0.0};
  double registration_gap_relaxation_min_inlier_ratio_{0.0};
  double registration_gap_relaxation_max_inlier_rmse_m_{0.0};
  double deskew_point_time_scale_{0.0};
  double deskew_point_time_offset_s_{0.0};
  double deskew_max_scan_duration_s_{0.2};
  double deskew_max_imu_gap_s_{0.05};
  double deskew_imu_buffer_duration_s_{5.0};
  double deskew_imu_wait_timeout_s_{0.25};
  double initialization_warmup_duration_s_{0.0};
  double deskew_min_time_ratio_{0.5};
  double deskew_time_bin_s_{0.001};
  double deskew_angular_velocity_scale_{1.0};
  double inter_lidar_motion_compensation_max_dt_s_{0.05};
  double inter_lidar_imu_rotation_scale_{1.0};
  double inter_lidar_gap_recovery_hypothesis_min_gap_s_{0.5};
  double inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_{0.02};
  double deskew_imu_to_base_roll_deg_{0.0};
  double deskew_imu_to_base_pitch_deg_{0.0};
  double deskew_imu_to_base_yaw_deg_{0.0};
  double imu_rotation_prediction_max_dt_s_{0.5};
  double imu_rotation_prediction_scale_{1.0};
  double imu_orientation_prediction_max_time_delta_s_{0.05};
  double imu_relative_attitude_max_time_delta_s_{0.05};
  double imu_roughness_window_s_{0.35};
  double imu_roughness_recent_event_window_s_{1.0};
  double imu_roughness_angular_velocity_threshold_radps_{2.5};
  double imu_roughness_angular_accel_threshold_radps2_{40.0};
  double imu_roughness_accel_norm_delta_threshold_mps2_{3.0};
  double imu_roughness_linear_jerk_threshold_mps3_{60.0};
  double imu_shock_hold_duration_s_{0.5};
  double keyframe_quarantine_correction_translation_m_{0.8};
  double place_recognition_registration_only_keyframe_quarantine_correction_translation_m_{0.0};
  double place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m_{0.0};
  double keyframe_quarantine_correction_rotation_deg_{15.0};
  double place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg_{0.0};
  double place_recognition_periodic_min_interval_s_{3.0};
  double place_recognition_relocalization_min_interval_s_{0.0};
  double place_recognition_shock_relocalization_cooldown_s_{0.0};
  double place_recognition_max_descriptor_distance_{0.20};
  double place_recognition_min_inlier_ratio_{0.35};
  double place_recognition_max_inlier_rmse_m_{0.8};
  double place_recognition_max_pose_jump_m_{3.0};
  double place_recognition_max_pose_jump_rotation_deg_{45.0};
  double place_recognition_mapping_refinement_large_pose_jump_max_translation_m_{0.0};
  double place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_{0.0};
  double place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_{0.0};
  double place_recognition_max_yaw_offset_deg_{180.0};
  double place_recognition_max_vertical_separation_m_{0.0};
  double place_recognition_registration_only_vertical_recovery_m_{0.0};
  double place_recognition_registration_only_same_layer_xy_max_distance_m_{0.0};
  double place_recognition_registration_only_max_pose_jump_m_{0.0};
  double place_recognition_submap_consistency_max_translation_m_{0.0};
  double place_recognition_registration_only_consistency_max_translation_m_{0.0};
  double place_recognition_registration_only_consistency_max_gap_s_{0.0};
  double place_recognition_submap_consistency_max_rotation_deg_{0.0};
  double place_recognition_mapping_refinement_quality_max_correspondence_m_{0.8};
  double place_recognition_mapping_refinement_min_target_coverage_ratio_{0.0};
  double place_recognition_mapping_refinement_target_confirmation_voxel_m_{0.35};
  double submap_max_travel_m_{20.0};
  double pose_graph_correction_max_translation_m_{5.0};
  double pose_graph_correction_max_rotation_deg_{45.0};
  double pose_graph_correction_min_interval_s_{1.0};
  double pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_{1.5};
  double pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_{1.0};
  double pose_graph_optimization_step_{0.15};
  double pose_graph_odom_translation_weight_{1.0};
  double pose_graph_odom_rotation_weight_{1.0};
  double pose_graph_loop_translation_weight_{1.0};
  double pose_graph_loop_rotation_weight_{1.0};
  double pose_graph_loop_cauchy_scale_{1.0};
  double pose_graph_min_new_loop_robust_weight_{0.0};
  double pose_graph_max_iteration_translation_step_m_{0.25};
  double pose_graph_max_iteration_rotation_step_deg_{5.0};
  double pose_graph_max_loop_residual_before_{0.0};
  double pose_graph_max_residual_after_{0.0};
  double pose_graph_max_residual_ratio_{0.0};
  double planar_pose_max_z_correction_m_{0.0};
  double gravity_accel_low_pass_alpha_{0.02};
  double gravity_accel_time_constant_s_{0.0};
  double gravity_accel_scale_{1.0};
  double gravity_accel_min_norm_mps2_{6.0};
  double gravity_accel_max_norm_mps2_{13.0};
  double gravity_accel_max_age_s_{0.1};
  double imu_reorder_tolerance_s_{0.02};
  double imu_timestamp_reset_threshold_s_{0.5};
  double map_z_filter_min_m_{-5.0};
  double map_z_filter_max_m_{8.0};
  double min_registration_inlier_ratio_{0.15};
  double max_registration_inlier_rmse_m_{1.5};
  double crop_min_x_{-0.6};
  double crop_max_x_{0.6};
  double crop_min_y_{-0.45};
  double crop_max_y_{0.45};
  double crop_min_z_{-0.8};
  double crop_max_z_{0.8};
  double map_publish_period_s_{1.0};
  double scan_to_scan_max_reference_age_s_{0.0};

  int min_scan_points_{200};
  int adaptive_scan_voxel_min_points_{900};
  int registration_only_cloud_max_points_{5000};
  int place_recognition_descriptor_bins_{60};
  int place_recognition_descriptor_rings_{1};
  int place_recognition_min_keyframes_{25};
  int place_recognition_exclude_recent_keyframes_{20};
  int place_recognition_max_candidates_{4};
  int place_recognition_candidate_map_neighbors_{6};
  int place_recognition_submap_min_consecutive_matches_{1};
  int place_recognition_mapping_refinement_min_source_points_{200};
  int place_recognition_mapping_refinement_min_target_points_{1000};
  int place_recognition_mapping_refinement_min_target_keyframe_observations_{2};
  int pose_graph_correction_min_keyframes_{5};
  int pose_graph_optimization_iterations_{25};
  int pose_quarantine_tracking_reset_interval_{0};
  int executor_thread_count_{2};
  int cloud_subscription_depth_{5};
  int cloud_sync_queue_size_{10};
  int imu_subscription_depth_{400};
  int imu_shock_recovery_stable_scans_{3};
  int wheel_odom_buffer_max_samples_{5000};
  int max_icp_iterations_{40};
  int robust_registration_refinement_iterations_{15};
  int max_keyframes_{500};
  int max_local_keyframes_{80};
  int submap_max_keyframes_{30};
  int submap_overlap_keyframes_{5};
  int max_local_submaps_{3};
  int max_path_poses_{10000};

  std::string deskew_point_time_field_{"auto"};
  std::string imu_rotation_prediction_source_{"angular_velocity"};
  std::string pose_graph_solver_{"legacy_relaxation"};
  std::string pose_graph_information_mode_{"configured_diagonal"};
  PoseGraphMinimumIntervalOverrideConfig pose_graph_minimum_interval_override_config_;
  PoseGraphDuplicateSubmapPairOverrideConfig
    pose_graph_duplicate_submap_pair_override_config_;
  GroundMotionConstraintConfig ground_motion_constraint_config_;
  ImuRelativeAttitudeConstraintConfig imu_relative_attitude_constraint_config_;
  ImuShockPolicyConfig imu_shock_policy_config_;
  ImuShockPolicyState imu_shock_policy_state_;
  IndoorTerrainConfig terrain_config_;
  TerrainAccumulatorConfig terrain_accumulator_config_;
  TerrainEvidenceAccumulator terrain_accumulator_;
  TerrainPreviewConfig terrain_preview_config_;
  TerrainPreviewTracker terrain_preview_tracker_;
  std::string last_terrain_gravity_status_{"disabled"};
  std::size_t terrain_last_keyframe_input_points_{0U};
  std::size_t terrain_last_keyframe_retained_points_{0U};
  std::uint64_t terrain_gravity_unavailable_keyframe_count_{0};
  std::uint64_t terrain_gravity_stale_keyframe_count_{0};
  SpatialMappingPolicyConfig spatial_mapping_config_;
  SubmapPolicyConfig submap_policy_config_;
  PoseGraphOptimizationConfig pose_graph_optimizer_config_;
  RegistrationInformationConfig pose_graph_information_config_;
  VoxelSurfelRegistration::Config surfel_config_;
  MappingCloudRefinementConfig mapping_cloud_refinement_config_;
  MappingCloudVerifiedCoarseConfig mapping_cloud_verified_coarse_config_;

  mutable std::mutex state_mutex_;
  mutable std::mutex cloud_sync_mutex_;
  mutable std::mutex wheel_mutex_;
  mutable std::mutex imu_mutex_;
  mutable std::mutex map_publish_snapshot_mutex_;

  Eigen::Isometry3d map_T_base_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d latest_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_used_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d pending_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d previous_accepted_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_accepted_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d previous_tracking_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_place_candidate_correction_{Eigen::Isometry3d::Identity()};
  Eigen::Matrix3d initial_imu_world_R_base_{Eigen::Matrix3d::Identity()};

  bool have_latest_wheel_odom_{false};
  bool have_last_used_wheel_odom_{false};
  bool have_pending_wheel_odom_{false};
  bool have_previous_accepted_pose_{false};
  bool have_last_accepted_pose_{false};
  bool have_previous_tracking_scan_{false};
  bool have_initial_imu_world_R_base_{false};
  bool have_first_primary_cloud_stamp_{false};
  bool have_last_primary_cloud_stamp_{false};
  bool have_last_processed_scan_stamp_{false};
  bool have_last_pose_graph_correction_stamp_{false};
  bool have_last_place_recognition_attempt_stamp_{false};
  bool have_last_place_candidate_correction_{false};
  bool last_place_candidate_used_registration_only_vertical_recovery_{false};
  rclcpp::Time last_place_candidate_stamp_{0, 0, RCL_ROS_TIME};
  bool have_gravity_accel_base_{false};
  bool have_last_gravity_accel_stamp_{false};
  bool have_last_imu_roughness_event_stamp_{false};
  bool have_last_imu_roughness_evaluation_stamp_{false};
  bool constant_velocity_rebase_pending_{false};
  rclcpp::Time latest_imu_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time latest_wheel_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_used_wheel_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time pending_wheel_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time wheel_prediction_query_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time wheel_prediction_lower_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time wheel_prediction_upper_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time first_primary_cloud_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_primary_cloud_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_processed_scan_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time previous_tracking_scan_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_pose_graph_correction_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_place_recognition_attempt_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_gravity_accel_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_imu_roughness_event_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_imu_roughness_evaluation_stamp_{0, 0, RCL_ROS_TIME};

  std::vector<Keyframe> keyframes_;
  std::vector<Submap> submaps_;
  std::vector<PoseGraphEdge> pose_graph_edges_;
  std::map<std::pair<int, int>, std::set<std::int64_t>>
  accepted_loop_submap_pair_anchor_stamps_;
  int next_submap_id_{0};
  int last_place_candidate_submap_id_{-1};
  int consecutive_place_candidate_submap_matches_{0};
  std::deque<WheelOdomSample> wheel_odom_buffer_;
  std::deque<ImuSample> imu_buffer_;
  std::deque<GravitySample> gravity_buffer_;
  Eigen::Vector3d gravity_accel_base_{Eigen::Vector3d::UnitZ()};
  double latest_gravity_filter_alpha_{0.0};
  double wheel_prediction_max_bracket_delta_s_{0.0};
  std::string wheel_prediction_status_{"not_evaluated"};
  std::uint64_t wheel_odom_duplicate_count_{0};
  std::uint64_t wheel_odom_out_of_order_count_{0};
  std::uint64_t wheel_odom_frame_rejected_count_{0};
  std::uint64_t wheel_odom_invalid_pose_rejected_count_{0};
  std::uint64_t wheel_odom_timestamp_rewind_rejected_count_{0};
  int consecutive_pose_quarantine_count_{0};
  int imu_duplicate_dropped_count_{0};
  int imu_out_of_order_dropped_count_{0};
  int imu_rewind_dropped_count_{0};
  int imu_timestamp_reset_count_{0};
  int imu_frame_rejected_count_{0};
  int imu_nonfinite_rejected_count_{0};
  std::uint64_t gravity_shock_sample_rejected_count_{0};
  CloudPtr global_map_;
  CloudPtr latest_local_map_;
  TerrainCloudPtr terrain_evidence_map_;
  TerrainSupportVectorPtr terrain_support_evidence_map_;
  TerrainCloudPtr terrain_preview_evidence_map_;
  CloudConstPtr local_map_publish_snapshot_;
  CloudConstPtr global_map_publish_snapshot_;
  TerrainCloudConstPtr terrain_publish_snapshot_;
  TerrainSupportVectorConstPtr terrain_support_publish_snapshot_;
  TerrainCloudConstPtr terrain_preview_publish_snapshot_;
  TerrainClassificationStats last_terrain_classification_;
  CloudPtr previous_tracking_scan_base_;
  nav_msgs::msg::Path path_msg_;
  RegistrationResult last_registration_;
  std::vector<AnchoredTerrainTraversalSample> terrain_traversal_samples_;
  bool terrain_pending_shock_event_{false};
  bool terrain_pending_shock_bridge_invalidated_{false};
  double terrain_pending_shock_score_{0.0};
  rclcpp::Time terrain_pending_shock_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time terrain_pending_shock_pre_sample_stamp_{0, 0, RCL_ROS_TIME};
  std::uint64_t terrain_unlocalized_shock_event_count_{0};
  bool terrain_traversal_continuity_break_pending_{false};
  bool terrain_response_episode_active_{false};

  std::deque<PendingPrimaryCloud> pending_primary_clouds_;
  std::array<AuxiliaryCloudInput, 2> auxiliary_cloud_inputs_;
  std::uint64_t primary_sync_queue_overflow_count_{0};
  std::uint64_t primary_sync_duplicate_count_{0};
  std::uint64_t primary_sync_out_of_order_count_{0};
  std::uint64_t cloud_sync_watermark_resolution_count_{0};
  std::uint64_t cloud_sync_wall_timeout_count_{0};
  std::uint64_t inter_lidar_wheel_wait_timeout_count_{0};
  std::uint64_t deskew_imu_watermark_waited_count_{0};
  std::uint64_t deskew_imu_wait_timeout_count_{0};
  std::int64_t latest_primary_sync_stamp_ns_{0};
  bool have_latest_primary_sync_stamp_{false};
  std::atomic<std::uint64_t> map_publish_lock_skip_count_{0};
  std::atomic<std::uint64_t> map_publish_snapshot_update_count_{0};
  std::atomic<std::uint64_t> map_publish_callback_count_{0};
  std::atomic<double> map_publish_runtime_ms_{0.0};
  std::atomic<double> map_publish_runtime_ms_max_{0.0};
  std::atomic<double> terrain_support_publish_runtime_ms_{0.0};
  std::atomic<double> terrain_support_publish_runtime_ms_max_{0.0};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr primary_cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::CallbackGroup::SharedPtr primary_cloud_callback_group_;
  rclcpp::CallbackGroup::SharedPtr auxiliary_cloud_callback_group_;
  rclcpp::CallbackGroup::SharedPtr cloud_processing_callback_group_;
  rclcpp::CallbackGroup::SharedPtr wheel_callback_group_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr export_keyframes_service_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_evidence_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_evidence_colored_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_support_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_preview_evidence_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    pose_graph_marker_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_pub_;
  rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr imu_watermark_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr preprocessed_scan_pub_;

  rclcpp::TimerBase::SharedPtr map_publish_timer_;
  rclcpp::TimerBase::SharedPtr cloud_processing_timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_
