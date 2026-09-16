#include "parameter_profile.hpp"

#include <stdexcept>

namespace gole
{
namespace slam_3d_core
{
namespace
{

using Defaults = std::map<std::string, rclcpp::ParameterValue>;

Defaults indoorDefaults()
{
  return {
    {"map_frame", rclcpp::ParameterValue("map")},
    {"odom_frame", rclcpp::ParameterValue("odom")},
    {"base_frame", rclcpp::ParameterValue("base_link")},
    {"publish_tf", rclcpp::ParameterValue(false)},
    {"use_tertiary_cloud", rclcpp::ParameterValue(false)},
    {"cloud_subscription_depth", rclcpp::ParameterValue(10)},
    {"wheel_odom_topic", rclcpp::ParameterValue("/unused/wheel_odom_disabled")},
    {"use_wheel_prediction", rclcpp::ParameterValue(false)},
    {"imu_topic", rclcpp::ParameterValue("/imu/data")},
    {"imu_subscription_reliable", rclcpp::ParameterValue(true)},
    {"imu_require_expected_frame", rclcpp::ParameterValue(true)},
    {"imu_expected_frame", rclcpp::ParameterValue("cv7_link")},
    {"deskew_enabled", rclcpp::ParameterValue(true)},
    {"deskew_wait_for_imu_watermark", rclcpp::ParameterValue(true)},
    {"deskew_imu_wait_timeout_s", rclcpp::ParameterValue(0.15)},
    {"reject_cloud_on_deskew_failure", rclcpp::ParameterValue(true)},
    {"deskew_point_time_field", rclcpp::ParameterValue("offset_time")},
    {"deskew_point_time_scale", rclcpp::ParameterValue(0.0)},
    {"deskew_point_time_offset_s", rclcpp::ParameterValue(0.0)},
    {"deskew_max_scan_duration_s", rclcpp::ParameterValue(0.15)},
    {"deskew_max_imu_gap_s", rclcpp::ParameterValue(0.03)},
    {"deskew_min_time_ratio", rclcpp::ParameterValue(0.8)},
    {"deskew_angular_velocity_scale", rclcpp::ParameterValue(1.0)},
    {"deskew_imu_to_base_roll_deg", rclcpp::ParameterValue(0.0)},
    {"deskew_imu_to_base_pitch_deg", rclcpp::ParameterValue(0.0)},
    {"deskew_imu_to_base_yaw_deg", rclcpp::ParameterValue(0.0)},
    {"imu_roughness_detection_enabled", rclcpp::ParameterValue(true)},
    {"imu_shock_policy_enabled", rclcpp::ParameterValue(true)},
    {"imu_roughness_window_s", rclcpp::ParameterValue(0.15)},
    {"imu_roughness_recent_event_window_s", rclcpp::ParameterValue(0.75)},
    {"imu_roughness_angular_velocity_threshold_radps", rclcpp::ParameterValue(1.2)},
    {"imu_roughness_angular_accel_threshold_radps2", rclcpp::ParameterValue(15.0)},
    {"imu_roughness_accel_norm_delta_threshold_mps2", rclcpp::ParameterValue(1.5)},
    {"imu_roughness_linear_jerk_threshold_mps3", rclcpp::ParameterValue(250.0)},
    {"imu_shock_hold_duration_s", rclcpp::ParameterValue(0.35)},
    {"quarantine_keyframes_on_shock", rclcpp::ParameterValue(true)},
    {"gravity_accel_scale", rclcpp::ParameterValue(1.0)},
    {"gravity_accel_time_constant_s", rclcpp::ParameterValue(0.7)},
    {"gravity_accel_min_norm_mps2", rclcpp::ParameterValue(7.5)},
    {"gravity_accel_max_norm_mps2", rclcpp::ParameterValue(12.5)},
    {"gravity_accel_max_age_s", rclcpp::ParameterValue(0.03)},
    {"ground_vehicle_nominal_max_body_z_step_m", rclcpp::ParameterValue(0.08)},
    {"ground_vehicle_roughness_max_body_z_step_m", rclcpp::ParameterValue(0.08)},
    {"use_tf_transforms", rclcpp::ParameterValue(true)},
    {"reject_cloud_on_tf_failure", rclcpp::ParameterValue(true)},
    {"icp_max_correspondence_m", rclcpp::ParameterValue(1.2)},
    {"fitness_score_threshold", rclcpp::ParameterValue(0.8)},
    {"robust_registration_enabled", rclcpp::ParameterValue(true)},
    {"robust_registration_refinement_max_correspondence_m", rclcpp::ParameterValue(0.45)},
    {"robust_registration_refinement_iterations", rclcpp::ParameterValue(12)},
    {"robust_registration_quality_max_correspondence_m", rclcpp::ParameterValue(0.45)},
    {"min_registration_inlier_ratio", rclcpp::ParameterValue(0.25)},
    {"max_registration_inlier_rmse_m", rclcpp::ParameterValue(0.45)},
    {"max_registration_correction_translation_m", rclcpp::ParameterValue(1.5)},
    {"max_registration_correction_rotation_deg", rclcpp::ParameterValue(30.0)},
    {"registration_gap_relaxation_enabled", rclcpp::ParameterValue(true)},
    {"registration_gap_relaxation_max_speed_mps", rclcpp::ParameterValue(1.5)},
    {"registration_gap_relaxation_max_translation_m", rclcpp::ParameterValue(3.0)},
    {"registration_gap_relaxation_min_inlier_ratio", rclcpp::ParameterValue(0.5)},
    {"registration_gap_relaxation_max_inlier_rmse_m", rclcpp::ParameterValue(0.3)},
    {"z_min_m", rclcpp::ParameterValue(-4.0)},
    {"z_max_m", rclcpp::ParameterValue(6.0)},
    {"temporal_static_map_enabled", rclcpp::ParameterValue(false)},
    {"keyframe_archive_directory", rclcpp::ParameterValue("")},
    {"crop_min_x", rclcpp::ParameterValue(-0.7)},
    {"crop_max_x", rclcpp::ParameterValue(0.7)},
    {"crop_min_y", rclcpp::ParameterValue(-0.55)},
    {"crop_max_y", rclcpp::ParameterValue(0.55)},
    {"crop_max_z", rclcpp::ParameterValue(0.9)},
    {"terrain_evidence_enabled", rclcpp::ParameterValue(true)},
    {"terrain_radial_bin_size_m", rclcpp::ParameterValue(0.1)},
    {"terrain_classification_voxel_size_m", rclcpp::ParameterValue(0.04)},
    {"terrain_max_profile_height_above_base_m", rclcpp::ParameterValue(0.35)},
    {"terrain_max_adjacent_gap_m", rclcpp::ParameterValue(0.35)},
    {"terrain_ramp_plane_validation_enabled", rclcpp::ParameterValue(true)},
    {"terrain_transition_support_bins", rclcpp::ParameterValue(2)},
    {"terrain_min_view_label_confidence", rclcpp::ParameterValue(0.67)},
    {"terrain_min_label_confidence", rclcpp::ParameterValue(0.67)},
    {"terrain_traversal_confirmation_enabled", rclcpp::ParameterValue(true)},
    {"terrain_traversal_association_radius_m", rclcpp::ParameterValue(0.75)},
    {"terrain_traversal_exit_radius_m", rclcpp::ParameterValue(0.95)},
    {"terrain_traversal_max_sample_gap_s", rclcpp::ParameterValue(1.0)},
    {"terrain_traversal_min_label_confidence", rclcpp::ParameterValue(0.67)},
    {"mapping_spatial_mode", rclcpp::ParameterValue("se3")},
    {"keyframe_translation_m", rclcpp::ParameterValue(0.45)},
    {"keyframe_rotation_deg", rclcpp::ParameterValue(5.0)},
    {"keyframe_vertical_translation_m", rclcpp::ParameterValue(0.2)},
    {"keyframe_rotation_3d_deg", rclcpp::ParameterValue(6.0)},
    {"local_map_radius_m", rclcpp::ParameterValue(15.0)},
    {"local_map_vertical_radius_m", rclcpp::ParameterValue(3.0)},
    {"max_local_keyframes", rclcpp::ParameterValue(72)},
    {"max_keyframes", rclcpp::ParameterValue(3000)},
    {"max_path_poses", rclcpp::ParameterValue(30000)},
    {"map_publish_period_s", rclcpp::ParameterValue(2.0)},
    {"submap_enabled", rclcpp::ParameterValue(true)},
    {"submap_max_keyframes", rclcpp::ParameterValue(24)},
    {"submap_max_travel_m", rclcpp::ParameterValue(10.0)},
    {"place_recognition_enabled", rclcpp::ParameterValue(true)},
    {"place_recognition_periodic_enabled", rclcpp::ParameterValue(true)},
    {"place_recognition_require_pose_graph_correction", rclcpp::ParameterValue(true)},
    {"place_recognition_periodic_min_interval_s", rclcpp::ParameterValue(2.0)},
    {"place_recognition_relocalization_min_interval_s", rclcpp::ParameterValue(2.0)},
    {"place_recognition_shock_relocalization_cooldown_s", rclcpp::ParameterValue(2.0)},
    {"place_recognition_descriptor_rings", rclcpp::ParameterValue(20)},
    {"place_recognition_min_keyframes", rclcpp::ParameterValue(40)},
    {"place_recognition_exclude_recent_keyframes", rclcpp::ParameterValue(30)},
    {"place_recognition_max_candidates", rclcpp::ParameterValue(6)},
    {"place_recognition_max_descriptor_distance", rclcpp::ParameterValue(0.22)},
    {"place_recognition_max_inlier_rmse_m", rclcpp::ParameterValue(0.45)},
    {"place_recognition_max_pose_jump_m", rclcpp::ParameterValue(4.0)},
    {"place_recognition_max_pose_jump_rotation_deg", rclcpp::ParameterValue(35.0)},
    {"place_recognition_use_submaps", rclcpp::ParameterValue(true)},
    {"place_recognition_submap_min_consecutive_matches", rclcpp::ParameterValue(2)},
    {"place_recognition_max_vertical_separation_m", rclcpp::ParameterValue(2.0)},
    {"place_recognition_submap_consistency_max_translation_m", rclcpp::ParameterValue(0.75)},
    {"place_recognition_submap_consistency_max_rotation_deg", rclcpp::ParameterValue(8.0)},
    {"pose_graph_correction_enabled", rclcpp::ParameterValue(true)},
    {"pose_graph_solver", rclcpp::ParameterValue("ceres_se3")},
    {"pose_graph_planar_correction_enabled", rclcpp::ParameterValue(false)},
    {"pose_graph_robust_loop_enabled", rclcpp::ParameterValue(true)},
    {"pose_graph_correction_min_keyframes", rclcpp::ParameterValue(12)},
    {"pose_graph_correction_max_translation_m", rclcpp::ParameterValue(4.0)},
    {"pose_graph_correction_max_rotation_deg", rclcpp::ParameterValue(35.0)},
    {"pose_graph_correction_min_interval_s", rclcpp::ParameterValue(2.0)},
    {"pose_graph_optimization_step", rclcpp::ParameterValue(0.12)},
    {"pose_graph_loop_translation_weight", rclcpp::ParameterValue(2.0)},
    {"pose_graph_loop_rotation_weight", rclcpp::ParameterValue(1.5)},
    {"pose_graph_ceres_num_threads", rclcpp::ParameterValue(2)},
    {"pose_graph_min_new_loop_robust_weight", rclcpp::ParameterValue(0.08)},
    {"pose_graph_max_iteration_translation_step_m", rclcpp::ParameterValue(0.15)},
    {"pose_graph_max_iteration_rotation_step_deg", rclcpp::ParameterValue(3.0)},
    {"pose_graph_max_loop_residual_before", rclcpp::ParameterValue(12.0)},
    {"pose_graph_max_residual_after", rclcpp::ParameterValue(2.0)},
    {"pose_graph_max_residual_ratio", rclcpp::ParameterValue(1.0)},
  };
}

Defaults profileDefaults(const std::string & name)
{
  if (name == "default") {
    return {};
  }
  if (name == "hesai_triple") {
    return {
      {"map_frame", rclcpp::ParameterValue("map")},
      {"odom_frame", rclcpp::ParameterValue("odom")},
      {"base_frame", rclcpp::ParameterValue("base_link")},
      {"publish_tf", rclcpp::ParameterValue(false)},
      {"primary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_center/points_timed")},
      {"secondary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_front/points_timed")},
      {"tertiary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_rear/points_timed")},
      {"use_secondary_cloud", rclcpp::ParameterValue(true)},
      {"use_tertiary_cloud", rclcpp::ParameterValue(true)},
      {"cloud_subscription_reliable", rclcpp::ParameterValue(true)},
      {"max_secondary_stamp_delta_s", rclcpp::ParameterValue(0.01)},
      {"max_auxiliary_stamp_delta_s", rclcpp::ParameterValue(0.01)},
      {"cloud_sync_wait_timeout_s", rclcpp::ParameterValue(0.5)},
      {"wheel_odom_topic", rclcpp::ParameterValue("/motor_driver_node/odom")},
      {"imu_topic", rclcpp::ParameterValue("/imu/data")},
      {"imu_require_expected_frame", rclcpp::ParameterValue(false)},
      {"imu_expected_frame", rclcpp::ParameterValue("")},
      {"use_wheel_prediction", rclcpp::ParameterValue(true)},
      {"use_tf_transforms", rclcpp::ParameterValue(true)},
      {"reject_cloud_on_tf_failure", rclcpp::ParameterValue(true)},
      {"deskew_enabled", rclcpp::ParameterValue(true)},
      {"deskew_wait_for_imu_watermark", rclcpp::ParameterValue(true)},
      {"inter_lidar_motion_compensation_enabled", rclcpp::ParameterValue(true)},
      {"inter_lidar_imu_rotation_scale", rclcpp::ParameterValue(1.0)},
      {"reject_cloud_on_deskew_failure", rclcpp::ParameterValue(true)},
      {"deskew_point_time_field", rclcpp::ParameterValue("offset_time")},
      {"deskew_point_time_scale", rclcpp::ParameterValue(0.0)},
      {"deskew_point_time_offset_s", rclcpp::ParameterValue(0.0)},
      {"deskew_angular_velocity_scale", rclcpp::ParameterValue(1.0)},
      {"deskew_imu_to_base_roll_deg", rclcpp::ParameterValue(0.0)},
      {"deskew_imu_to_base_pitch_deg", rclcpp::ParameterValue(0.0)},
      {"deskew_imu_to_base_yaw_deg", rclcpp::ParameterValue(0.0)},
      {"registration_method", rclcpp::ParameterValue("gicp")},
      {"max_range_m", rclcpp::ParameterValue(80.0)},
      {"scan_voxel_leaf_m", rclcpp::ParameterValue(0.25)},
      {"map_voxel_leaf_m", rclcpp::ParameterValue(0.35)},
      {"temporal_static_map_enabled", rclcpp::ParameterValue(false)},
      {"keyframe_archive_directory", rclcpp::ParameterValue("")},
      {"terrain_roughness_preview_enabled", rclcpp::ParameterValue(false)},
      {"terrain_traversal_min_label_confidence", rclcpp::ParameterValue(0.67)},
      {"mapping_spatial_mode", rclcpp::ParameterValue("planar")},
      {"pose_graph_correction_enabled", rclcpp::ParameterValue(false)},
      {"pose_graph_solver", rclcpp::ParameterValue("legacy_relaxation")},
      {"pose_graph_planar_correction_enabled", rclcpp::ParameterValue(false)},
      {"gravity_accel_scale", rclcpp::ParameterValue(1.0)},
    };
  }
  if (name == "indoor_xt32") {
    auto result = indoorDefaults();
    const Defaults specific{
      {"primary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_center/points_timed")},
      {"secondary_cloud_topic", rclcpp::ParameterValue(
          "/unused/slam_3d/input/hesai_front/points_timed")},
      {"tertiary_cloud_topic", rclcpp::ParameterValue(
          "/unused/slam_3d/input/hesai_rear/points_timed")},
      {"use_secondary_cloud", rclcpp::ParameterValue(false)},
      {"cloud_sync_queue_size", rclcpp::ParameterValue(20)},
      {"constant_velocity_translation_limit_m", rclcpp::ParameterValue(0.3)},
      {"constant_velocity_rotation_limit_deg", rclcpp::ParameterValue(10.0)},
      {"registration_method", rclcpp::ParameterValue("voxel_surfel")},
      {"max_range_m", rclcpp::ParameterValue(40.0)},
      {"scan_voxel_leaf_m", rclcpp::ParameterValue(0.18)},
      {"map_voxel_leaf_m", rclcpp::ParameterValue(0.22)},
      {"min_scan_points", rclcpp::ParameterValue(300)},
      {"surfel_min_points_per_voxel", rclcpp::ParameterValue(5)},
      {"surfel_max_planarity_ratio", rclcpp::ParameterValue(0.25)},
      {"surfel_min_correspondences", rclcpp::ParameterValue(150)},
      {"surfel_min_correspondence_ratio", rclcpp::ParameterValue(0.08)},
      {"surfel_max_iterations", rclcpp::ParameterValue(12)},
      {"surfel_max_backtracking_steps", rclcpp::ParameterValue(5)},
      {"surfel_rotation_convergence_rad", rclcpp::ParameterValue(0.0002)},
      {"surfel_translation_convergence_m", rclcpp::ParameterValue(0.0002)},
      {"surfel_relative_cost_convergence", rclcpp::ParameterValue(1e-05)},
      {"surfel_max_quality_rmse_m", rclcpp::ParameterValue(0.3)},
      {"surfel_max_quality_p95_m", rclcpp::ParameterValue(0.5)},
      {"terrain_seed_max_range_m", rclcpp::ParameterValue(6.0)},
      {"terrain_min_points_per_bin", rclcpp::ParameterValue(1)},
      {"terrain_floor_quantile", rclcpp::ParameterValue(0.05)},
      {"pose_graph_optimization_iterations", rclcpp::ParameterValue(80)},
    };
    result.insert(specific.begin(), specific.end());
    return result;
  }
  if (name == "indoor_xt32_gicp") {
    auto result = indoorDefaults();
    const Defaults specific{
      {"primary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_center/points_timed")},
      {"secondary_cloud_topic", rclcpp::ParameterValue(
          "/unused/slam_3d/input/hesai_front/points_timed")},
      {"tertiary_cloud_topic", rclcpp::ParameterValue(
          "/unused/slam_3d/input/hesai_rear/points_timed")},
      {"use_secondary_cloud", rclcpp::ParameterValue(false)},
      {"cloud_sync_queue_size", rclcpp::ParameterValue(20)},
      {"use_constant_velocity_prediction", rclcpp::ParameterValue(false)},
      {"registration_method", rclcpp::ParameterValue("gicp")},
      {"max_range_m", rclcpp::ParameterValue(40.0)},
      {"scan_voxel_leaf_m", rclcpp::ParameterValue(0.12)},
      {"map_voxel_leaf_m", rclcpp::ParameterValue(0.18)},
      {"min_scan_points", rclcpp::ParameterValue(300)},
      {"terrain_seed_max_range_m", rclcpp::ParameterValue(6.0)},
      {"terrain_min_points_per_bin", rclcpp::ParameterValue(1)},
      {"terrain_floor_quantile", rclcpp::ParameterValue(0.05)},
      {"pose_graph_optimization_iterations", rclcpp::ParameterValue(40)},
    };
    result.insert(specific.begin(), specific.end());
    return result;
  }
  if (name == "indoor_dual_jt128") {
    auto result = indoorDefaults();
    const Defaults specific{
      {"primary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_front/points_timed")},
      {"secondary_cloud_topic", rclcpp::ParameterValue("/slam_3d/input/hesai_rear/points_timed")},
      {"tertiary_cloud_topic", rclcpp::ParameterValue(
          "/unused/slam_3d/input/hesai_center/points_timed")},
      {"use_secondary_cloud", rclcpp::ParameterValue(true)},
      {"cloud_sync_queue_size", rclcpp::ParameterValue(24)},
      {"max_secondary_stamp_delta_s", rclcpp::ParameterValue(0.015)},
      {"max_auxiliary_stamp_delta_s", rclcpp::ParameterValue(0.015)},
      {"cloud_sync_reorder_window_s", rclcpp::ParameterValue(0.015)},
      {"inter_lidar_motion_compensation_enabled", rclcpp::ParameterValue(true)},
      {"inter_lidar_motion_compensation_source", rclcpp::ParameterValue("imu_rotation")},
      {"inter_lidar_motion_compensation_max_dt_s", rclcpp::ParameterValue(0.02)},
      {"inter_lidar_imu_rotation_scale", rclcpp::ParameterValue(1.0)},
      {"inter_lidar_gap_recovery_hypothesis_enabled", rclcpp::ParameterValue(true)},
      {"use_constant_velocity_prediction", rclcpp::ParameterValue(false)},
      {"registration_method", rclcpp::ParameterValue("gicp")},
      {"max_range_m", rclcpp::ParameterValue(45.0)},
      {"scan_voxel_leaf_m", rclcpp::ParameterValue(0.15)},
      {"map_voxel_leaf_m", rclcpp::ParameterValue(0.2)},
      {"min_scan_points", rclcpp::ParameterValue(500)},
      {"terrain_azimuth_bins", rclcpp::ParameterValue(144)},
      {"terrain_min_points_per_bin", rclcpp::ParameterValue(3)},
      {"terrain_ground_band_m", rclcpp::ParameterValue(0.08)},
      {"terrain_ramp_min_run_bins", rclcpp::ParameterValue(4)},
      {"terrain_evidence_voxel_size_m", rclcpp::ParameterValue(0.18)},
      {"pose_graph_optimization_iterations", rclcpp::ParameterValue(40)},
    };
    result.insert(specific.begin(), specific.end());
    return result;
  }
  throw std::invalid_argument(
          "Unknown parameter_profile='" + name +
          "'; expected default, hesai_triple, indoor_xt32, indoor_xt32_gicp, indoor_dual_jt128");
}

}  // namespace

ParameterProfile::ParameterProfile(rclcpp::Node & node)
: node_(node)
{
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = "Built-in startup defaults; explicit ROS parameters take precedence";
  descriptor.read_only = true;
  const auto name =
    node_.declare_parameter<std::string>("parameter_profile", "default", descriptor);
  defaults_ = profileDefaults(name);
  RCLCPP_INFO(node_.get_logger(), "Parameter profile selected: %s", name.c_str());
}

void ParameterProfile::validateConsumed() const
{
  if (!defaults_.empty()) {
    throw std::logic_error("Undeclared parameter in profile: " + defaults_.begin()->first);
  }
}

}  // namespace slam_3d_core
}  // namespace gole
