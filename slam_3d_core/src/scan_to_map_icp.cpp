#include "slam_3d_core/scan_to_map_icp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <utility>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "slam_3d_core/registration_policy.hpp"

namespace gole
{
namespace slam_3d_core
{

namespace
{

template<typename T>
std::string toString(const T & value)
{
  std::ostringstream stream;
  stream << value;
  return stream.str();
}

double transformDistance2d(const Eigen::Isometry3d & a, const Eigen::Isometry3d & b)
{
  const Eigen::Vector3d delta = a.translation() - b.translation();
  return std::hypot(delta.x(), delta.y());
}

double normalizedAngle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

double rotationAngleDeg(const Eigen::Matrix3d & rotation)
{
  Eigen::Quaterniond q(rotation);
  q.normalize();
  return Eigen::AngleAxisd(q).angle() * 180.0 / M_PI;
}

Eigen::Isometry3d normalizeIsometry(const Eigen::Isometry3d & transform)
{
  Eigen::Isometry3d normalized = transform;
  Eigen::Quaterniond q(normalized.linear());
  q.normalize();
  normalized.linear() = q.toRotationMatrix();
  return normalized;
}

Eigen::Isometry3d interpolateTransform(
  const Eigen::Isometry3d & transform,
  const double translation_ratio,
  const double rotation_ratio)
{
  const double clamped_translation_ratio = std::clamp(translation_ratio, 0.0, 1.0);
  const double clamped_rotation_ratio = std::clamp(rotation_ratio, 0.0, 1.0);

  Eigen::Quaterniond rotation(transform.linear());
  rotation.normalize();

  Eigen::Isometry3d interpolated = Eigen::Isometry3d::Identity();
  interpolated.translation() = transform.translation() * clamped_translation_ratio;
  interpolated.linear() =
    Eigen::Quaterniond::Identity().slerp(clamped_rotation_ratio, rotation).toRotationMatrix();
  return interpolated;
}

Eigen::Isometry3d limitCorrectionStep(
  const Eigen::Isometry3d & correction,
  const double max_translation_m,
  const double max_rotation_deg)
{
  double translation_ratio = 1.0;
  const double translation_norm = correction.translation().norm();
  if (max_translation_m > 0.0 && translation_norm > max_translation_m) {
    translation_ratio = max_translation_m / translation_norm;
  }

  double rotation_ratio = 1.0;
  const double rotation_deg = rotationAngleDeg(correction.linear());
  if (max_rotation_deg > 0.0 && rotation_deg > max_rotation_deg) {
    rotation_ratio = max_rotation_deg / rotation_deg;
  }

  return interpolateTransform(correction, translation_ratio, rotation_ratio);
}

Eigen::Isometry3d planarizeCorrection(const Eigen::Isometry3d & correction)
{
  Eigen::Isometry3d planar = Eigen::Isometry3d::Identity();
  planar.translation().x() = correction.translation().x();
  planar.translation().y() = correction.translation().y();
  planar.translation().z() = 0.0;
  const double yaw = std::atan2(correction.linear()(1, 0), correction.linear()(0, 0));
  planar.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return planar;
}

Eigen::Isometry3d poseMsgToIsometry(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
  const Eigen::Quaterniond q(
    pose.orientation.w,
    pose.orientation.x,
    pose.orientation.y,
    pose.orientation.z);
  transform.linear() = q.normalized().toRotationMatrix();
  transform.translation() =
    Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return transform;
}

Eigen::Quaterniond deltaRotationFromAngularVelocity(
  const Eigen::Vector3d & angular_velocity,
  const double dt_s)
{
  const Eigen::Vector3d rotation_vector = angular_velocity * dt_s;
  const double angle = rotation_vector.norm();
  if (!std::isfinite(angle) || angle <= 1.0e-12) {
    return Eigen::Quaterniond::Identity();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(angle, rotation_vector / angle));
}

Eigen::Matrix3d rotationFromRpyDegrees(
  const double roll_deg,
  const double pitch_deg,
  const double yaw_deg)
{
  const double roll = roll_deg * M_PI / 180.0;
  const double pitch = pitch_deg * M_PI / 180.0;
  const double yaw = yaw_deg * M_PI / 180.0;
  const Eigen::AngleAxisd roll_rotation(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_rotation(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());
  return (yaw_rotation * pitch_rotation * roll_rotation).toRotationMatrix();
}

std::optional<double> readPointFieldAsDouble(
  const std::uint8_t * data,
  const sensor_msgs::msg::PointField & field)
{
  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8: {
        std::int8_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::UINT8: {
        std::uint8_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::INT16: {
        std::int16_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::UINT16: {
        std::uint16_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::INT32: {
        std::int32_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::UINT32: {
        std::uint32_t value = 0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::FLOAT32: {
        float value = 0.0F;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return static_cast<double>(value);
      }
    case sensor_msgs::msg::PointField::FLOAT64: {
        double value = 0.0;
        std::memcpy(&value, data + field.offset, sizeof(value));
        return value;
      }
    default:
      return std::nullopt;
  }
}

bool fieldWithinPointStep(
  const sensor_msgs::msg::PointField & field,
  const std::uint32_t point_step)
{
  std::uint32_t datatype_size = 0;
  switch (field.datatype) {
    case sensor_msgs::msg::PointField::INT8:
    case sensor_msgs::msg::PointField::UINT8:
      datatype_size = 1;
      break;
    case sensor_msgs::msg::PointField::INT16:
    case sensor_msgs::msg::PointField::UINT16:
      datatype_size = 2;
      break;
    case sensor_msgs::msg::PointField::INT32:
    case sensor_msgs::msg::PointField::UINT32:
    case sensor_msgs::msg::PointField::FLOAT32:
      datatype_size = 4;
      break;
    case sensor_msgs::msg::PointField::FLOAT64:
      datatype_size = 8;
      break;
    default:
      return false;
  }
  const std::uint32_t count = std::max<std::uint32_t>(field.count, 1);
  return field.offset + datatype_size * count <= point_step;
}

}  // namespace

ScanToMapIcp::ScanToMapIcp(const rclcpp::NodeOptions & options)
: Node("scan_to_map_icp_node", options),
  global_map_(new CloudT),
  latest_local_map_(new CloudT)
{
  setupParameters();
  setupRosInterfaces();

  path_msg_.header.frame_id = map_frame_;

  RCLCPP_INFO(
    get_logger(),
    "scan_to_map_icp_node ready: primary='%s', secondary_enabled=%s, odom='%s'",
    primary_cloud_topic_.c_str(),
    use_secondary_cloud_ ? "true" : "false",
    odom_topic_.c_str());
}

void ScanToMapIcp::setupParameters()
{
  map_frame_ = declare_parameter<std::string>("map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

  primary_cloud_topic_ = declare_parameter<std::string>(
    "primary_cloud_topic",
    "/slam_3d/input/mid360_front/points_debug");
  secondary_cloud_topic_ = declare_parameter<std::string>(
    "secondary_cloud_topic",
    "/slam_3d/input/mid360_rear/points_debug");
  wheel_odom_topic_ = declare_parameter<std::string>("wheel_odom_topic", "/motor_driver_node/odom");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data");

  odom_topic_ = declare_parameter<std::string>("odom_topic", "/slam_3d/frontend/odom");
  path_topic_ = declare_parameter<std::string>("path_topic", "/slam_3d/frontend/path");
  local_map_topic_ = declare_parameter<std::string>(
    "local_map_topic",
    "/slam_3d/map/active_submap");
  global_map_topic_ = declare_parameter<std::string>(
    "global_map_topic",
    "/slam_3d/map/global_cloud");
  health_topic_ = declare_parameter<std::string>("health_topic", "/slam_3d/health");
  imu_watermark_topic_ = declare_parameter<std::string>(
    "imu_watermark_topic",
    "/slam_3d/imu_watermark_ns");

  registration_method_ = declare_parameter<std::string>("registration_method", "gicp");
  use_secondary_cloud_ = declare_parameter<bool>("use_secondary_cloud", false);
  cloud_subscription_reliable_ = declare_parameter<bool>(
    "cloud_subscription_reliable",
    false);
  use_tf_transforms_ = declare_parameter<bool>("use_tf_transforms", true);
  use_wheel_prediction_ = declare_parameter<bool>("use_wheel_prediction", true);
  use_constant_velocity_prediction_ = declare_parameter<bool>(
    "use_constant_velocity_prediction",
    true);
  use_scan_to_scan_prediction_ = declare_parameter<bool>("use_scan_to_scan_prediction", false);
  use_scan_to_scan_fallback_ = declare_parameter<bool>("use_scan_to_scan_fallback", false);
  add_scan_to_scan_fallback_keyframes_ = declare_parameter<bool>(
    "add_scan_to_scan_fallback_keyframes",
    false);
  use_motion_prior_fallback_ = declare_parameter<bool>("use_motion_prior_fallback", false);
  add_motion_prior_fallback_keyframes_ = declare_parameter<bool>(
    "add_motion_prior_fallback_keyframes",
    false);
  adaptive_scan_voxel_enabled_ = declare_parameter<bool>("adaptive_scan_voxel_enabled", false);
  quarantine_keyframes_on_shock_ = declare_parameter<bool>(
    "quarantine_keyframes_on_shock",
    false);
  quarantine_pose_on_shock_ = declare_parameter<bool>("quarantine_pose_on_shock", false);
  record_tracking_scan_on_pose_quarantine_ = declare_parameter<bool>(
    "record_tracking_scan_on_pose_quarantine",
    false);
  place_recognition_enabled_ = declare_parameter<bool>("place_recognition_enabled", false);
  place_recognition_periodic_enabled_ = declare_parameter<bool>(
    "place_recognition_periodic_enabled",
    false);
  place_recognition_require_pose_graph_correction_ = declare_parameter<bool>(
    "place_recognition_require_pose_graph_correction",
    false);
  pose_graph_correction_enabled_ = declare_parameter<bool>(
    "pose_graph_correction_enabled",
    false);
  pose_graph_planar_correction_enabled_ = declare_parameter<bool>(
    "pose_graph_planar_correction_enabled",
    false);
  pose_graph_robust_loop_enabled_ = declare_parameter<bool>(
    "pose_graph_robust_loop_enabled",
    false);
  planar_pose_constraint_enabled_ = declare_parameter<bool>(
    "planar_pose_constraint_enabled",
    false);
  gravity_roll_pitch_constraint_enabled_ = declare_parameter<bool>(
    "gravity_roll_pitch_constraint_enabled",
    false);
  map_z_filter_enabled_ = declare_parameter<bool>(
    "map_z_filter_enabled",
    false);
  deskew_enabled_ = declare_parameter<bool>("deskew_enabled", false);
  imu_rotation_prediction_enabled_ = declare_parameter<bool>(
    "imu_rotation_prediction_enabled",
    false);
  imu_roughness_detection_enabled_ = declare_parameter<bool>(
    "imu_roughness_detection_enabled",
    false);
  robust_registration_enabled_ = declare_parameter<bool>(
    "robust_registration_enabled",
    false);
  publish_tf_ = declare_parameter<bool>("publish_tf", false);
  crop_box_enabled_ = declare_parameter<bool>("crop_box_enabled", true);

  max_secondary_stamp_delta_s_ = declare_parameter<double>("max_secondary_stamp_delta_s", 0.05);
  min_range_m_ = declare_parameter<double>("min_range_m", 0.5);
  max_range_m_ = declare_parameter<double>("max_range_m", 80.0);
  z_min_m_ = declare_parameter<double>("z_min_m", -5.0);
  z_max_m_ = declare_parameter<double>("z_max_m", 8.0);
  scan_voxel_leaf_m_ = declare_parameter<double>("scan_voxel_leaf_m", 0.25);
  adaptive_scan_voxel_leaf_m_ = declare_parameter<double>("adaptive_scan_voxel_leaf_m", 0.15);
  map_voxel_leaf_m_ = declare_parameter<double>("map_voxel_leaf_m", 0.35);
  local_map_radius_m_ = declare_parameter<double>("local_map_radius_m", 35.0);
  keyframe_translation_m_ = declare_parameter<double>("keyframe_translation_m", 0.8);
  keyframe_rotation_deg_ = declare_parameter<double>("keyframe_rotation_deg", 8.0);
  icp_max_correspondence_m_ = declare_parameter<double>("icp_max_correspondence_m", 2.0);
  fitness_score_threshold_ = declare_parameter<double>("fitness_score_threshold", 2.0);
  robust_registration_trim_ratio_ = declare_parameter<double>(
    "robust_registration_trim_ratio",
    0.85);
  robust_registration_refinement_max_correspondence_m_ = declare_parameter<double>(
    "robust_registration_refinement_max_correspondence_m",
    0.8);
  robust_registration_quality_max_correspondence_m_ = declare_parameter<double>(
    "robust_registration_quality_max_correspondence_m",
    0.8);
  constant_velocity_translation_limit_m_ = declare_parameter<double>(
    "constant_velocity_translation_limit_m",
    2.0);
  constant_velocity_rotation_limit_deg_ = declare_parameter<double>(
    "constant_velocity_rotation_limit_deg",
    30.0);
  motion_prior_fallback_max_step_m_ = declare_parameter<double>(
    "motion_prior_fallback_max_step_m",
    0.0);
  motion_prior_fallback_max_z_step_m_ = declare_parameter<double>(
    "motion_prior_fallback_max_z_step_m",
    0.0);
  motion_prior_fallback_max_untrusted_distance_m_ = declare_parameter<double>(
    "motion_prior_fallback_max_untrusted_distance_m",
    0.0);
  motion_prior_fallback_max_untrusted_z_m_ = declare_parameter<double>(
    "motion_prior_fallback_max_untrusted_z_m",
    0.0);
  motion_prior_fallback_max_rotation_deg_ = declare_parameter<double>(
    "motion_prior_fallback_max_rotation_deg",
    0.0);
  max_registration_correction_translation_m_ = declare_parameter<double>(
    "max_registration_correction_translation_m",
    4.0);
  max_registration_correction_rotation_deg_ = declare_parameter<double>(
    "max_registration_correction_rotation_deg",
    60.0);
  deskew_point_time_field_ = declare_parameter<std::string>("deskew_point_time_field", "auto");
  deskew_point_time_scale_ = declare_parameter<double>("deskew_point_time_scale", 0.0);
  deskew_point_time_offset_s_ = declare_parameter<double>("deskew_point_time_offset_s", 0.0);
  deskew_max_scan_duration_s_ = declare_parameter<double>("deskew_max_scan_duration_s", 0.2);
  deskew_max_imu_gap_s_ = declare_parameter<double>("deskew_max_imu_gap_s", 0.05);
  deskew_imu_buffer_duration_s_ = declare_parameter<double>("deskew_imu_buffer_duration_s", 5.0);
  deskew_min_time_ratio_ = declare_parameter<double>("deskew_min_time_ratio", 0.5);
  deskew_time_bin_s_ = declare_parameter<double>("deskew_time_bin_s", 0.001);
  deskew_angular_velocity_scale_ = declare_parameter<double>(
    "deskew_angular_velocity_scale",
    1.0);
  deskew_imu_to_base_roll_deg_ = declare_parameter<double>(
    "deskew_imu_to_base_roll_deg",
    0.0);
  deskew_imu_to_base_pitch_deg_ = declare_parameter<double>(
    "deskew_imu_to_base_pitch_deg",
    0.0);
  deskew_imu_to_base_yaw_deg_ = declare_parameter<double>(
    "deskew_imu_to_base_yaw_deg",
    0.0);
  imu_rotation_prediction_max_dt_s_ = declare_parameter<double>(
    "imu_rotation_prediction_max_dt_s",
    0.5);
  imu_rotation_prediction_scale_ = declare_parameter<double>(
    "imu_rotation_prediction_scale",
    1.0);
  imu_roughness_window_s_ = declare_parameter<double>(
    "imu_roughness_window_s",
    0.35);
  imu_roughness_recent_event_window_s_ = declare_parameter<double>(
    "imu_roughness_recent_event_window_s",
    1.0);
  imu_roughness_angular_velocity_threshold_radps_ = declare_parameter<double>(
    "imu_roughness_angular_velocity_threshold_radps",
    2.5);
  imu_roughness_angular_accel_threshold_radps2_ = declare_parameter<double>(
    "imu_roughness_angular_accel_threshold_radps2",
    40.0);
  imu_roughness_accel_norm_delta_threshold_mps2_ = declare_parameter<double>(
    "imu_roughness_accel_norm_delta_threshold_mps2",
    3.0);
  imu_roughness_linear_jerk_threshold_mps3_ = declare_parameter<double>(
    "imu_roughness_linear_jerk_threshold_mps3",
    60.0);
  keyframe_quarantine_correction_translation_m_ = declare_parameter<double>(
    "keyframe_quarantine_correction_translation_m",
    0.8);
  keyframe_quarantine_correction_rotation_deg_ = declare_parameter<double>(
    "keyframe_quarantine_correction_rotation_deg",
    15.0);
  place_recognition_periodic_min_interval_s_ = declare_parameter<double>(
    "place_recognition_periodic_min_interval_s",
    3.0);
  place_recognition_relocalization_min_interval_s_ = declare_parameter<double>(
    "place_recognition_relocalization_min_interval_s",
    0.0);
  place_recognition_shock_relocalization_cooldown_s_ = declare_parameter<double>(
    "place_recognition_shock_relocalization_cooldown_s",
    0.0);
  place_recognition_max_descriptor_distance_ = declare_parameter<double>(
    "place_recognition_max_descriptor_distance",
    0.20);
  place_recognition_min_inlier_ratio_ = declare_parameter<double>(
    "place_recognition_min_inlier_ratio",
    0.35);
  place_recognition_max_inlier_rmse_m_ = declare_parameter<double>(
    "place_recognition_max_inlier_rmse_m",
    0.8);
  place_recognition_max_pose_jump_m_ = declare_parameter<double>(
    "place_recognition_max_pose_jump_m",
    3.0);
  place_recognition_max_pose_jump_rotation_deg_ = declare_parameter<double>(
    "place_recognition_max_pose_jump_rotation_deg",
    45.0);
  place_recognition_max_yaw_offset_deg_ = declare_parameter<double>(
    "place_recognition_max_yaw_offset_deg",
    180.0);
  pose_graph_correction_max_translation_m_ = declare_parameter<double>(
    "pose_graph_correction_max_translation_m",
    5.0);
  pose_graph_correction_max_rotation_deg_ = declare_parameter<double>(
    "pose_graph_correction_max_rotation_deg",
    45.0);
  pose_graph_correction_min_interval_s_ = declare_parameter<double>(
    "pose_graph_correction_min_interval_s",
    1.0);
  pose_graph_optimization_step_ = declare_parameter<double>(
    "pose_graph_optimization_step",
    0.15);
  pose_graph_odom_translation_weight_ = declare_parameter<double>(
    "pose_graph_odom_translation_weight",
    1.0);
  pose_graph_odom_rotation_weight_ = declare_parameter<double>(
    "pose_graph_odom_rotation_weight",
    1.0);
  pose_graph_loop_translation_weight_ = declare_parameter<double>(
    "pose_graph_loop_translation_weight",
    1.0);
  pose_graph_loop_rotation_weight_ = declare_parameter<double>(
    "pose_graph_loop_rotation_weight",
    1.0);
  pose_graph_loop_cauchy_scale_ = declare_parameter<double>(
    "pose_graph_loop_cauchy_scale",
    1.0);
  pose_graph_min_new_loop_robust_weight_ = declare_parameter<double>(
    "pose_graph_min_new_loop_robust_weight",
    0.0);
  pose_graph_max_iteration_translation_step_m_ = declare_parameter<double>(
    "pose_graph_max_iteration_translation_step_m",
    0.25);
  pose_graph_max_iteration_rotation_step_deg_ = declare_parameter<double>(
    "pose_graph_max_iteration_rotation_step_deg",
    5.0);
  pose_graph_max_loop_residual_before_ = declare_parameter<double>(
    "pose_graph_max_loop_residual_before",
    0.0);
  pose_graph_max_residual_after_ = declare_parameter<double>(
    "pose_graph_max_residual_after",
    0.0);
  pose_graph_max_residual_ratio_ = declare_parameter<double>(
    "pose_graph_max_residual_ratio",
    0.0);
  planar_pose_max_z_correction_m_ = declare_parameter<double>(
    "planar_pose_max_z_correction_m",
    0.0);
  gravity_accel_low_pass_alpha_ = declare_parameter<double>(
    "gravity_accel_low_pass_alpha",
    0.02);
  gravity_accel_time_constant_s_ = declare_parameter<double>(
    "gravity_accel_time_constant_s",
    0.0);
  gravity_accel_min_norm_mps2_ = declare_parameter<double>(
    "gravity_accel_min_norm_mps2",
    6.0);
  gravity_accel_max_norm_mps2_ = declare_parameter<double>(
    "gravity_accel_max_norm_mps2",
    13.0);
  gravity_accel_max_age_s_ = declare_parameter<double>(
    "gravity_accel_max_age_s",
    0.1);
  imu_reorder_tolerance_s_ = declare_parameter<double>(
    "imu_reorder_tolerance_s",
    0.02);
  imu_timestamp_reset_threshold_s_ = declare_parameter<double>(
    "imu_timestamp_reset_threshold_s",
    0.5);
  map_z_filter_min_m_ = declare_parameter<double>(
    "map_z_filter_min_m",
    -5.0);
  map_z_filter_max_m_ = declare_parameter<double>(
    "map_z_filter_max_m",
    8.0);
  min_registration_inlier_ratio_ = declare_parameter<double>(
    "min_registration_inlier_ratio",
    0.15);
  max_registration_inlier_rmse_m_ = declare_parameter<double>(
    "max_registration_inlier_rmse_m",
    1.5);
  map_publish_period_s_ = declare_parameter<double>("map_publish_period_s", 1.0);
  scan_to_scan_max_reference_age_s_ = declare_parameter<double>(
    "scan_to_scan_max_reference_age_s",
    0.0);

  crop_min_x_ = declare_parameter<double>("crop_min_x", -0.6);
  crop_max_x_ = declare_parameter<double>("crop_max_x", 0.6);
  crop_min_y_ = declare_parameter<double>("crop_min_y", -0.45);
  crop_max_y_ = declare_parameter<double>("crop_max_y", 0.45);
  crop_min_z_ = declare_parameter<double>("crop_min_z", -0.8);
  crop_max_z_ = declare_parameter<double>("crop_max_z", 0.8);

  min_scan_points_ = declare_parameter<int>("min_scan_points", 200);
  adaptive_scan_voxel_min_points_ = declare_parameter<int>("adaptive_scan_voxel_min_points", 900);
  place_recognition_descriptor_bins_ = declare_parameter<int>(
    "place_recognition_descriptor_bins",
    60);
  place_recognition_descriptor_rings_ = declare_parameter<int>(
    "place_recognition_descriptor_rings",
    1);
  place_recognition_min_keyframes_ = declare_parameter<int>(
    "place_recognition_min_keyframes",
    25);
  place_recognition_exclude_recent_keyframes_ = declare_parameter<int>(
    "place_recognition_exclude_recent_keyframes",
    20);
  place_recognition_max_candidates_ = declare_parameter<int>(
    "place_recognition_max_candidates",
    4);
  place_recognition_candidate_map_neighbors_ = declare_parameter<int>(
    "place_recognition_candidate_map_neighbors",
    6);
  pose_graph_correction_min_keyframes_ = declare_parameter<int>(
    "pose_graph_correction_min_keyframes",
    5);
  pose_graph_optimization_iterations_ = declare_parameter<int>(
    "pose_graph_optimization_iterations",
    25);
  pose_quarantine_tracking_reset_interval_ = declare_parameter<int>(
    "pose_quarantine_tracking_reset_interval",
    0);
  cloud_subscription_depth_ = declare_parameter<int>("cloud_subscription_depth", 5);
  imu_subscription_depth_ = declare_parameter<int>("imu_subscription_depth", 400);
  max_icp_iterations_ = declare_parameter<int>("max_icp_iterations", 40);
  robust_registration_refinement_iterations_ = declare_parameter<int>(
    "robust_registration_refinement_iterations",
    15);
  max_keyframes_ = declare_parameter<int>("max_keyframes", 500);
  max_local_keyframes_ = declare_parameter<int>("max_local_keyframes", 80);
  max_path_poses_ = declare_parameter<int>("max_path_poses", 10000);

  if (registration_method_ != "icp" && registration_method_ != "gicp") {
    RCLCPP_WARN(
      get_logger(),
      "Unknown registration_method '%s'. Falling back to gicp.",
      registration_method_.c_str());
    registration_method_ = "gicp";
  }
}

void ScanToMapIcp::setupRosInterfaces()
{
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  auto cloud_qos = rclcpp::QoS(
    rclcpp::KeepLast(std::max(1, cloud_subscription_depth_))).durability_volatile();
  if (cloud_subscription_reliable_) {
    cloud_qos.reliable();
  } else {
    cloud_qos.best_effort();
  }

  primary_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    primary_cloud_topic_,
    cloud_qos,
    std::bind(&ScanToMapIcp::primaryCloudCallback, this, std::placeholders::_1));

  if (use_secondary_cloud_) {
    const auto secondary_cloud_qos = rclcpp::QoS(
      rclcpp::KeepLast(std::max(1, cloud_subscription_depth_)))
      .best_effort()
      .durability_volatile();
    secondary_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      secondary_cloud_topic_,
      secondary_cloud_qos,
      std::bind(&ScanToMapIcp::secondaryCloudCallback, this, std::placeholders::_1));
  }

  if (use_wheel_prediction_) {
    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      wheel_odom_topic_,
      rclcpp::QoS(50),
      std::bind(&ScanToMapIcp::wheelOdomCallback, this, std::placeholders::_1));
  }

  imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions imu_subscription_options;
  imu_subscription_options.callback_group = imu_callback_group_;
  const auto imu_qos = rclcpp::QoS(rclcpp::KeepLast(std::max(5, imu_subscription_depth_)))
    .best_effort()
    .durability_volatile();
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    imu_topic_,
    imu_qos,
    std::bind(&ScanToMapIcp::imuCallback, this, std::placeholders::_1),
    imu_subscription_options);

  odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic_, rclcpp::QoS(20));
  path_pub_ = create_publisher<nav_msgs::msg::Path>(path_topic_, rclcpp::QoS(10));
  local_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    local_map_topic_,
    rclcpp::QoS(2).reliable());
  global_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    global_map_topic_,
    rclcpp::QoS(1).reliable());
  health_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    health_topic_,
    rclcpp::QoS(10));
  imu_watermark_pub_ = create_publisher<std_msgs::msg::Int64>(
    imu_watermark_topic_,
    rclcpp::QoS(1).reliable().transient_local());

  const auto map_publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(std::max(map_publish_period_s_, 0.1)));
  map_publish_timer_ = create_wall_timer(
    map_publish_period,
    std::bind(&ScanToMapIcp::mapPublishTimerCallback, this));
}

void ScanToMapIcp::primaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  bool expected = false;
  if (!processing_.compare_exchange_strong(expected, true)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Dropping scan because previous registration is still running");
    return;
  }

  processCloud(msg);
  processing_.store(false);
}

void ScanToMapIcp::secondaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(secondary_mutex_);
  latest_secondary_msg_ = msg;
}

void ScanToMapIcp::wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(wheel_mutex_);
  latest_wheel_odom_ = odomToIsometry(*msg);
  latest_wheel_stamp_ = msg->header.stamp;
  have_latest_wheel_odom_ = true;
}

void ScanToMapIcp::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  ImuSample sample;
  sample.stamp = msg->header.stamp;
  sample.angular_velocity = Eigen::Vector3d(
    msg->angular_velocity.x,
    msg->angular_velocity.y,
    msg->angular_velocity.z);
  sample.linear_acceleration = Eigen::Vector3d(
    msg->linear_acceleration.x,
    msg->linear_acceleration.y,
    msg->linear_acceleration.z);

  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    const double buffer_duration = std::max(0.1, deskew_imu_buffer_duration_s_);
    if (!imu_buffer_.empty()) {
      const double dt_from_latest = (sample.stamp - imu_buffer_.back().stamp).seconds();
      if (dt_from_latest <= 0.0) {
        const double rewind_s = -dt_from_latest;
        const double reorder_tolerance_s = std::max(0.0, imu_reorder_tolerance_s_);
        const double reset_threshold_s = std::max(
          reorder_tolerance_s,
          std::max(1.0e-6, imu_timestamp_reset_threshold_s_));
        if (rewind_s <= std::numeric_limits<double>::epsilon()) {
          ++imu_duplicate_dropped_count_;
        } else {
          ++imu_rewind_dropped_count_;
        }
        if (rewind_s >= reset_threshold_s) {
          ++imu_timestamp_reset_count_;
        }
        ++imu_out_of_order_dropped_count_;
        return;
      }
    }

    latest_imu_stamp_ = sample.stamp;
    imu_buffer_.push_back(sample);

    if (gravity_roll_pitch_constraint_enabled_) {
      const Eigen::Matrix3d imu_to_base = rotationFromRpyDegrees(
        deskew_imu_to_base_roll_deg_,
        deskew_imu_to_base_pitch_deg_,
        deskew_imu_to_base_yaw_deg_);
      const Eigen::Vector3d accel_base = imu_to_base * Eigen::Vector3d(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);
      const double accel_norm = accel_base.norm();
      const double min_norm = std::max(0.0, gravity_accel_min_norm_mps2_);
      const double max_norm = std::max(min_norm, gravity_accel_max_norm_mps2_);
      if (std::isfinite(accel_norm) && accel_norm >= min_norm && accel_norm <= max_norm) {
        double alpha = std::clamp(gravity_accel_low_pass_alpha_, 0.0, 1.0);
        const double time_constant = std::max(0.0, gravity_accel_time_constant_s_);
        if (time_constant > 0.0 && have_last_gravity_accel_stamp_) {
          const double gravity_dt = (sample.stamp - last_gravity_accel_stamp_).seconds();
          if (std::isfinite(gravity_dt) && gravity_dt > 0.0) {
            alpha = -std::expm1(-gravity_dt / time_constant);
          } else {
            alpha = 0.0;
          }
        }
        if (!have_gravity_accel_base_ || alpha >= 1.0) {
          gravity_accel_base_ = accel_base;
          have_gravity_accel_base_ = true;
        } else if (alpha > 0.0) {
          gravity_accel_base_ = (1.0 - alpha) * gravity_accel_base_ + alpha * accel_base;
        }
        latest_gravity_filter_alpha_ = alpha;
        last_gravity_accel_stamp_ = sample.stamp;
        have_last_gravity_accel_stamp_ = true;
        gravity_buffer_.push_back(
          GravitySample{sample.stamp, gravity_accel_base_, latest_gravity_filter_alpha_});
      }
    }

    while (!imu_buffer_.empty() &&
      (sample.stamp - imu_buffer_.front().stamp).seconds() > buffer_duration)
    {
      imu_buffer_.pop_front();
    }
    while (!gravity_buffer_.empty() &&
      (sample.stamp - gravity_buffer_.front().stamp).seconds() > buffer_duration)
    {
      gravity_buffer_.pop_front();
    }
  }

  std_msgs::msg::Int64 watermark;
  watermark.data = sample.stamp.nanoseconds();
  imu_watermark_pub_->publish(watermark);
}

void ScanToMapIcp::mapPublishTimerCallback()
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  const auto stamp = now();
  publishCloud(local_map_pub_, latest_local_map_, map_frame_, stamp);
  publishCloud(global_map_pub_, global_map_, map_frame_, stamp);
}

void ScanToMapIcp::processCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & primary_msg)
{
  const rclcpp::Time stamp(primary_msg->header.stamp);
  bool gravity_sample_stale = false;
  const auto gravity_sample = gravitySampleAt(stamp, gravity_sample_stale);
  DeskewStats deskew_stats;
  auto scan_base = buildBaseFrameScan(*primary_msg, deskew_stats);
  auto attach_deskew_stats = [this, &deskew_stats](RegistrationResult & result) {
      applyDeskewStatsToResult(deskew_stats, result);
    };
  auto attach_gravity_stats =
    [this, &gravity_sample, &gravity_sample_stale, &stamp](
    RegistrationResult & registration_result)
    {
      if (!gravity_roll_pitch_constraint_enabled_) {
        registration_result.gravity_constraint_status = "disabled";
        return;
      }
      if (!gravity_sample.has_value()) {
        registration_result.gravity_constraint_status = gravity_sample_stale ?
          "stale_at_cloud_stamp" : "unavailable_at_cloud_stamp";
        return;
      }
      registration_result.gravity_accel_used = true;
      registration_result.gravity_accel_age_s =
        std::max(0.0, (stamp - gravity_sample->stamp).seconds());
      registration_result.gravity_accel_norm_mps2 = gravity_sample->accel_base.norm();
      registration_result.gravity_filter_alpha = gravity_sample->filter_alpha;
      const Eigen::Vector3d normalized_gravity = gravity_sample->accel_base.normalized();
      registration_result.gravity_roll_deg =
        std::atan2(normalized_gravity.y(), normalized_gravity.z()) * 180.0 / M_PI;
      registration_result.gravity_pitch_deg =
        std::atan2(
        -normalized_gravity.x(),
        std::hypot(normalized_gravity.y(), normalized_gravity.z())) * 180.0 / M_PI;
      registration_result.gravity_constraint_status = "timestamped_snapshot";
    };

  if (!scan_base || static_cast<int>(scan_base->size()) < min_scan_points_) {
    const int scan_points = scan_base ? static_cast<int>(scan_base->size()) : 0;
    RegistrationResult result;
    result.rejection_reason = "scan_too_sparse_before_downsample";
    result.scan_points_before_downsample = scan_points;
    result.scan_points_after_downsample = scan_points;
    result.scan_voxel_leaf_m = scan_voxel_leaf_m_;
    attach_deskew_stats(result);
    attach_gravity_stats(result);
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      consecutive_pose_quarantine_count_ = 0;
      last_registration_ = result;
      publishDiagnostics(stamp, result);
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Filtered scan has too few points for registration");
    return;
  }

  const int scan_points_before_downsample = static_cast<int>(scan_base->size());
  double effective_scan_voxel_leaf_m = scan_voxel_leaf_m_;
  scan_base = downsampleScanAdaptive(scan_base, effective_scan_voxel_leaf_m);
  const int scan_points_after_downsample = static_cast<int>(scan_base->size());
  const bool scan_voxel_adapted = std::abs(effective_scan_voxel_leaf_m - scan_voxel_leaf_m_) >
    std::numeric_limits<double>::epsilon();

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (static_cast<int>(scan_base->size()) < min_scan_points_) {
    RegistrationResult result;
    result.rejection_reason = "scan_too_sparse_after_downsample";
    result.scan_points_before_downsample = scan_points_before_downsample;
    result.scan_points_after_downsample = scan_points_after_downsample;
    result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    result.scan_voxel_adapted = scan_voxel_adapted;
    attach_deskew_stats(result);
    attach_gravity_stats(result);
    consecutive_pose_quarantine_count_ = 0;
    last_registration_ = result;
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Downsampled scan has too few points for registration");
    publishDiagnostics(stamp, result);
    return;
  }

  const auto scan_descriptor = computeScanDescriptor(scan_base);
  const bool have_previous_scan_stamp = have_last_processed_scan_stamp_;
  const rclcpp::Time previous_scan_stamp = last_processed_scan_stamp_;
  last_processed_scan_stamp_ = stamp;
  have_last_processed_scan_stamp_ = true;
  auto imu_roughness_stats = evaluateImuRoughness(
    previous_scan_stamp,
    stamp,
    have_previous_scan_stamp);
  if (imu_roughness_stats.evaluated) {
    if (imu_roughness_stats.event) {
      last_imu_roughness_event_stamp_ = stamp;
      have_last_imu_roughness_event_stamp_ = true;
    }
    if (have_last_imu_roughness_event_stamp_) {
      imu_roughness_stats.time_since_event_s =
        std::max(0.0, (stamp - last_imu_roughness_event_stamp_).seconds());
      const double recent_window_s = std::max(0.0, imu_roughness_recent_event_window_s_);
      imu_roughness_stats.recent_event =
        recent_window_s > 0.0 && imu_roughness_stats.time_since_event_s <= recent_window_s;
    }
  }
  auto attach_imu_roughness_stats =
    [this, &imu_roughness_stats](RegistrationResult & registration_result) {
      applyImuRoughnessStatsToResult(imu_roughness_stats, registration_result);
    };

  if (!initialized_) {
    initializeFromFirstScan(scan_base, stamp);
    last_registration_.scan_points_before_downsample = scan_points_before_downsample;
    last_registration_.scan_points_after_downsample = scan_points_after_downsample;
    last_registration_.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    last_registration_.scan_voxel_adapted = scan_voxel_adapted;
    last_registration_.keyframe_added = true;
    attach_deskew_stats(last_registration_);
    attach_imu_roughness_stats(last_registration_);
    attach_gravity_stats(last_registration_);
    publishState(stamp);
    publishDiagnostics(stamp, last_registration_);
    return;
  }

  bool used_scan_to_scan_prediction = false;
  bool used_imu_rotation_prediction = false;
  std::string imu_rotation_prediction_status{"disabled"};
  double scan_to_scan_reference_age_s = 0.0;
  std::string scan_to_scan_prediction_status{"disabled"};
  auto attach_prediction_stats = [&](RegistrationResult & registration_result) {
      registration_result.scan_to_scan_prediction_used = used_scan_to_scan_prediction;
      registration_result.scan_to_scan_reference_age_s = scan_to_scan_reference_age_s;
      registration_result.scan_to_scan_prediction_status = scan_to_scan_prediction_status;
      registration_result.imu_rotation_prediction_used = used_imu_rotation_prediction;
      registration_result.imu_rotation_prediction_status = imu_rotation_prediction_status;
    };
  const auto prediction = predictWithMotionPriors(
    scan_base,
    previous_scan_stamp,
    stamp,
    have_previous_scan_stamp,
    used_scan_to_scan_prediction,
    used_imu_rotation_prediction,
    imu_rotation_prediction_status,
    scan_to_scan_reference_age_s,
    scan_to_scan_prediction_status);
  auto local_map = buildLocalMap(prediction);
  if (!local_map || static_cast<int>(local_map->size()) < min_scan_points_) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Local map has too few points; adding scan as a keyframe without registration");
    map_T_base_ = prediction;
    commitWheelPrediction();
    consecutive_pose_quarantine_count_ = 0;
    recordAcceptedPose();
    recordTrackingScan(scan_base, map_T_base_, stamp);
    addKeyframe(scan_base, stamp);
    RegistrationResult result;
    result.converged = true;
    result.accepted = true;
    result.map_T_base = map_T_base_;
    result.rejection_reason = "local_map_too_small";
    result.scan_points_before_downsample = scan_points_before_downsample;
    result.scan_points_after_downsample = scan_points_after_downsample;
    result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    result.scan_voxel_adapted = scan_voxel_adapted;
    result.keyframe_added = true;
    attach_prediction_stats(result);
    attach_deskew_stats(result);
    attach_imu_roughness_stats(result);
    attach_gravity_stats(result);
    last_registration_ = result;
    publishState(stamp);
    publishDiagnostics(stamp, result);
    return;
  }

  auto result = registerScanToMap(scan_base, local_map, prediction);
  result.scan_points_before_downsample = scan_points_before_downsample;
  result.scan_points_after_downsample = scan_points_after_downsample;
  result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
  result.scan_voxel_adapted = scan_voxel_adapted;
  attach_prediction_stats(result);
  std::optional<RegistrationResult> pre_place_recognition_result;

  const std::string local_result_shock_reason = shockQuarantineReason(result);
  const bool local_result_shock_relocalization =
    result.accepted && !local_result_shock_reason.empty();
  const bool local_result_needs_relocalization =
    !result.accepted || local_result_shock_relocalization;
  const Eigen::Isometry3d revisit_probe_pose = result.accepted ?
    constrainPlanarPose(
    constrainMotionPriorFallback(result.map_T_base), prediction, gravity_sample) :
    prediction;
  const bool keyframe_revisit_probe =
    result.accepted && shouldAddKeyframe(revisit_probe_pose);
  bool periodic_place_recognition = false;
  if (place_recognition_enabled_ && place_recognition_periodic_enabled_ &&
    keyframe_revisit_probe && !local_result_needs_relocalization &&
    static_cast<int>(keyframes_.size()) >= place_recognition_min_keyframes_)
  {
    const double min_interval_s = std::max(0.0, place_recognition_periodic_min_interval_s_);
    periodic_place_recognition = !have_last_place_recognition_attempt_stamp_ ||
      min_interval_s <= 0.0 ||
      (stamp - last_place_recognition_attempt_stamp_).seconds() >= min_interval_s;
  }
  if (place_recognition_enabled_ &&
    (local_result_needs_relocalization || periodic_place_recognition))
  {
    const std::string place_trigger =
      local_result_needs_relocalization ? "relocalization" : "periodic";
    double relocalization_min_interval_s =
      std::max(0.0, place_recognition_relocalization_min_interval_s_);
    if (local_result_shock_relocalization) {
      relocalization_min_interval_s = std::max(
        relocalization_min_interval_s,
        std::max(0.0, place_recognition_shock_relocalization_cooldown_s_));
    }
    const bool relocalization_cooldown_active = local_result_needs_relocalization &&
      relocalization_min_interval_s > 0.0 &&
      have_last_place_recognition_attempt_stamp_ &&
      (stamp - last_place_recognition_attempt_stamp_).seconds() < relocalization_min_interval_s;
    if (relocalization_cooldown_active) {
      result.place_recognition_skipped = true;
      result.place_recognition_trigger = place_trigger;
      result.place_recognition_skip_reason = local_result_shock_relocalization ?
        "shock_relocalization_cooldown" : "relocalization_cooldown";
    } else {
      result.place_recognition_attempted = true;
      result.place_recognition_trigger = place_trigger;
      last_place_recognition_attempt_stamp_ = stamp;
      have_last_place_recognition_attempt_stamp_ = true;
      RegistrationResult place_result;
      const bool place_accepted = tryPlaceRecognition(
        scan_base,
        scan_descriptor,
        prediction,
        place_result);
      place_result.place_recognition_trigger = place_trigger;
      result.place_recognition_candidates = place_result.place_recognition_candidates;
      result.place_recognition_candidate_index = place_result.place_recognition_candidate_index;
      result.place_recognition_descriptor_distance =
        place_result.place_recognition_descriptor_distance;
      result.place_recognition_yaw_offset_deg = place_result.place_recognition_yaw_offset_deg;
      if (place_accepted) {
        pre_place_recognition_result = result;
        place_result.scan_points_before_downsample = scan_points_before_downsample;
        place_result.scan_points_after_downsample = scan_points_after_downsample;
        place_result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
        place_result.scan_voxel_adapted = scan_voxel_adapted;
        place_result.place_recognition_attempted = true;
        attach_prediction_stats(place_result);
        attach_deskew_stats(place_result);
        attach_imu_roughness_stats(place_result);
        attach_gravity_stats(place_result);
        result = place_result;
      }
    }
  }
  attach_deskew_stats(result);
  attach_imu_roughness_stats(result);
  attach_gravity_stats(result);

  bool pose_graph_correction_already_applied = false;
  const bool accepted_place_result =
    result.place_recognition_accepted && pre_place_recognition_result.has_value();
  const bool require_pose_graph_validation =
    place_recognition_require_pose_graph_correction_ || pose_graph_robust_loop_enabled_;
  const bool validate_place_result = shouldValidateAcceptedPlaceRecognition(
    result.place_recognition_accepted,
    pre_place_recognition_result.has_value(),
    require_pose_graph_validation);
  if (accepted_place_result) {
    // Evaluate the same bounded place pose that the normal commit path would use.
    result.map_T_base = constrainPlanarPose(
      constrainMotionPriorFallback(result.map_T_base),
      prediction,
      gravity_sample);
    const PlaceRecognitionPoseMetrics place_pose_metrics =
      computePlaceRecognitionPoseMetrics(prediction, result.map_T_base);
    result.place_recognition_pose_metrics_evaluated = true;
    result.place_recognition_pose_metrics_valid = place_pose_metrics.finite;
    result.place_recognition_prediction_jump_translation_m =
      place_pose_metrics.prediction_jump_translation_m;
    result.place_recognition_prediction_jump_rotation_deg =
      place_pose_metrics.prediction_jump_rotation_deg;
    result.place_recognition_map_correction_translation_m =
      place_pose_metrics.map_correction_translation_m;
    result.place_recognition_map_correction_rotation_deg =
      place_pose_metrics.map_correction_rotation_deg;
    const std::string place_candidate_shock_reason = shockQuarantineReason(result);
    const bool attempt_pose_graph_correction = shouldAttemptPoseGraphCorrection(
      validate_place_result,
      !place_candidate_shock_reason.empty());
    if (attempt_pose_graph_correction) {
      applyPoseGraphCorrection(stamp, prediction, gravity_sample, result);
    } else {
      result.pose_graph_correction_applied = false;
      result.pose_graph_rejection_reason =
        "place_candidate_shock_" + place_candidate_shock_reason;
    }
    if (result.pose_graph_correction_applied) {
      local_map = buildLocalMap(map_T_base_);
      pose_graph_correction_already_applied = true;
    } else if (shouldRestorePrePlaceResult(
        accepted_place_result,
        !place_candidate_shock_reason.empty(),
        validate_place_result,
        result.pose_graph_correction_applied))
    {
      const RegistrationResult rejected_place_result = result;
      result = *pre_place_recognition_result;
      result.place_recognition_attempted = rejected_place_result.place_recognition_attempted;
      result.place_recognition_skipped = true;
      result.place_recognition_accepted = false;
      result.place_recognition_trigger = rejected_place_result.place_recognition_trigger;
      result.place_recognition_skip_reason = "pose_graph_correction_rejected";
      result.place_recognition_candidates = rejected_place_result.place_recognition_candidates;
      result.place_recognition_candidate_index =
        rejected_place_result.place_recognition_candidate_index;
      result.place_recognition_descriptor_distance =
        rejected_place_result.place_recognition_descriptor_distance;
      result.place_recognition_yaw_offset_deg =
        rejected_place_result.place_recognition_yaw_offset_deg;
      result.place_recognition_pose_metrics_evaluated =
        rejected_place_result.place_recognition_pose_metrics_evaluated;
      result.place_recognition_pose_metrics_valid =
        rejected_place_result.place_recognition_pose_metrics_valid;
      result.place_recognition_prediction_jump_translation_m =
        rejected_place_result.place_recognition_prediction_jump_translation_m;
      result.place_recognition_prediction_jump_rotation_deg =
        rejected_place_result.place_recognition_prediction_jump_rotation_deg;
      result.place_recognition_map_correction_translation_m =
        rejected_place_result.place_recognition_map_correction_translation_m;
      result.place_recognition_map_correction_rotation_deg =
        rejected_place_result.place_recognition_map_correction_rotation_deg;
      result.pose_graph_correction_keyframes =
        rejected_place_result.pose_graph_correction_keyframes;
      result.pose_graph_optimization_iterations =
        rejected_place_result.pose_graph_optimization_iterations;
      result.pose_graph_edges = rejected_place_result.pose_graph_edges;
      result.pose_graph_loop_edges = rejected_place_result.pose_graph_loop_edges;
      result.pose_graph_robust_loop_edges_downweighted =
        rejected_place_result.pose_graph_robust_loop_edges_downweighted;
      result.pose_graph_residual_before = rejected_place_result.pose_graph_residual_before;
      result.pose_graph_residual_after = rejected_place_result.pose_graph_residual_after;
      result.pose_graph_robust_objective_before =
        rejected_place_result.pose_graph_robust_objective_before;
      result.pose_graph_robust_objective_after =
        rejected_place_result.pose_graph_robust_objective_after;
      result.pose_graph_min_loop_robust_weight =
        rejected_place_result.pose_graph_min_loop_robust_weight;
      result.pose_graph_new_loop_robust_weight =
        rejected_place_result.pose_graph_new_loop_robust_weight;
      result.pose_graph_max_update_translation_m =
        rejected_place_result.pose_graph_max_update_translation_m;
      result.pose_graph_max_update_rotation_deg =
        rejected_place_result.pose_graph_max_update_rotation_deg;
      result.pose_graph_new_loop_rejected =
        rejected_place_result.pose_graph_new_loop_rejected;
      result.pose_graph_rejection_reason =
        rejected_place_result.pose_graph_rejection_reason;
      if (!result.pose_graph_rejection_reason.empty()) {
        result.place_recognition_skip_reason =
          "pose_graph_" + result.pose_graph_rejection_reason;
      }
      attach_deskew_stats(result);
      attach_imu_roughness_stats(result);
      attach_gravity_stats(result);
    }
  }
  last_registration_ = result;

  if (!result.accepted) {
    consecutive_pose_quarantine_count_ = 0;
    if (use_scan_to_scan_fallback_ && used_scan_to_scan_prediction) {
      RCLCPP_WARN(
        get_logger(),
        "Scan-to-map rejected (%s); accepting scan-to-scan fallback pose",
        result.rejection_reason.c_str());
      map_T_base_ = constrainPlanarPose(
        constrainMotionPriorFallback(prediction), map_T_base_, gravity_sample);
      commitWheelPrediction();
      result.accepted = true;
      result.converged = true;
      result.rejection_reason = "scan_to_scan_fallback";
      result.map_T_base = map_T_base_;
      result.shock_quarantine_reason = shockQuarantineReason(result);
      result.shock_quarantined = !result.shock_quarantine_reason.empty();
      if (!result.shock_quarantined) {
        consecutive_pose_quarantine_count_ = 0;
        recordAcceptedPose();
        recordTrackingScan(scan_base, map_T_base_, stamp);
      }
      last_registration_ = result;
      latest_local_map_ = local_map;
      if (add_scan_to_scan_fallback_keyframes_ && shouldAddKeyframe(map_T_base_)) {
        if (!result.shock_quarantined) {
          addKeyframe(scan_base, stamp);
          result.keyframe_added = true;
        } else {
          result.keyframe_quarantined = true;
          result.keyframe_quarantine_reason = result.shock_quarantine_reason;
        }
      }
      last_registration_ = result;
      publishState(stamp);
      publishDiagnostics(stamp, result);
      return;
    }

    if (use_motion_prior_fallback_) {
      RCLCPP_WARN(
        get_logger(),
        "Scan-to-map rejected (%s); accepting motion-prior fallback pose",
        result.rejection_reason.c_str());
      map_T_base_ = constrainPlanarPose(
        constrainMotionPriorFallback(prediction), map_T_base_, gravity_sample);
      commitWheelPrediction();
      result.accepted = true;
      result.converged = true;
      result.rejection_reason = "motion_prior_fallback";
      result.map_T_base = map_T_base_;
      result.shock_quarantine_reason = shockQuarantineReason(result);
      result.shock_quarantined = !result.shock_quarantine_reason.empty();
      if (!result.shock_quarantined) {
        consecutive_pose_quarantine_count_ = 0;
        recordAcceptedPose();
        recordTrackingScan(scan_base, map_T_base_, stamp);
      }
      last_registration_ = result;
      latest_local_map_ = local_map;
      if (add_motion_prior_fallback_keyframes_ && shouldAddKeyframe(map_T_base_)) {
        if (!result.shock_quarantined) {
          addKeyframe(scan_base, stamp);
          result.keyframe_added = true;
        } else {
          result.keyframe_quarantined = true;
          result.keyframe_quarantine_reason = result.shock_quarantine_reason;
        }
      }
      last_registration_ = result;
      publishState(stamp);
      publishDiagnostics(stamp, result);
      return;
    }

    RCLCPP_WARN(
      get_logger(),
      "Registration rejected: reason=%s converged=%s fitness=%.4f threshold=%.4f "
      "inliers=%d ratio=%.3f rmse=%.3fm correction=%.3fm/%.1fdeg",
      result.rejection_reason.c_str(),
      result.converged ? "true" : "false",
      result.fitness,
      fitness_score_threshold_,
      result.inlier_count,
      result.inlier_ratio,
      result.inlier_rmse_m,
      result.correction_translation_m,
      result.correction_rotation_deg);
    if (!quarantine_keyframes_on_shock_) {
      recordTrackingScan(scan_base, prediction, stamp);
    }
    publishDiagnostics(stamp, result);
    return;
  }

  result.shock_quarantine_reason = shockQuarantineReason(result);
  result.shock_quarantined = !result.shock_quarantine_reason.empty();
  if (result.shock_quarantined && quarantine_pose_on_shock_) {
    map_T_base_ = constrainPlanarPose(
      constrainMotionPriorFallback(prediction), map_T_base_, gravity_sample);
    commitWheelPrediction();
    result.pose_quarantined = true;
    result.map_T_base = map_T_base_;
    result.rejection_reason = "shock_pose_quarantine";
    ++consecutive_pose_quarantine_count_;
    result.consecutive_pose_quarantine_count = consecutive_pose_quarantine_count_;
    const int tracking_reset_interval =
      std::max(0, pose_quarantine_tracking_reset_interval_);
    const bool periodic_tracking_reset =
      tracking_reset_interval > 0 &&
      consecutive_pose_quarantine_count_ % tracking_reset_interval == 0;
    if (record_tracking_scan_on_pose_quarantine_ || periodic_tracking_reset) {
      recordTrackingScan(scan_base, map_T_base_, stamp);
      result.pose_quarantine_tracking_reset = true;
    }
    latest_local_map_ = local_map;
    last_registration_ = result;
    publishState(stamp);
    publishDiagnostics(stamp, result);
    return;
  }
  consecutive_pose_quarantine_count_ = 0;
  if (!pose_graph_correction_already_applied) {
    map_T_base_ = constrainPlanarPose(
      constrainMotionPriorFallback(result.map_T_base), prediction, gravity_sample);
    result.map_T_base = map_T_base_;
  }
  if (result.place_recognition_accepted && !pose_graph_correction_already_applied) {
    applyPoseGraphCorrection(stamp, prediction, gravity_sample, result);
    local_map = buildLocalMap(map_T_base_);
  }
  commitWheelPrediction();
  if (!result.shock_quarantined) {
    consecutive_pose_quarantine_count_ = 0;
    recordAcceptedPose();
    recordTrackingScan(scan_base, map_T_base_, stamp);
  }
  latest_local_map_ = local_map;

  if (shouldAddKeyframe(map_T_base_)) {
    if (!result.shock_quarantined) {
      addKeyframe(scan_base, stamp);
      result.keyframe_added = true;
    } else {
      result.keyframe_quarantined = true;
      result.keyframe_quarantine_reason = result.shock_quarantine_reason;
    }
  }

  last_registration_ = result;
  publishState(stamp);
  publishDiagnostics(stamp, result);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildBaseFrameScan(
  const sensor_msgs::msg::PointCloud2 & primary_msg,
  DeskewStats & deskew_stats)
{
  auto merged = convertAndTransformCloud(primary_msg, deskew_stats);
  if (!merged) {
    return nullptr;
  }

  if (use_secondary_cloud_) {
    sensor_msgs::msg::PointCloud2::SharedPtr secondary_msg;
    {
      std::lock_guard<std::mutex> lock(secondary_mutex_);
      secondary_msg = latest_secondary_msg_;
    }

    if (secondary_msg) {
      const rclcpp::Time primary_stamp(primary_msg.header.stamp);
      const rclcpp::Time secondary_stamp(secondary_msg->header.stamp);
      const double delta_s = std::abs((primary_stamp - secondary_stamp).seconds());
      if (delta_s <= max_secondary_stamp_delta_s_) {
        DeskewStats secondary_deskew_stats;
        auto secondary = convertAndTransformCloud(*secondary_msg, secondary_deskew_stats);
        if (secondary) {
          *merged += *secondary;
          deskew_stats.point_count += secondary_deskew_stats.point_count;
          deskew_stats.points_with_time += secondary_deskew_stats.points_with_time;
          if (deskew_stats.point_count > 0) {
            deskew_stats.time_ratio =
              static_cast<double>(deskew_stats.points_with_time) /
              static_cast<double>(deskew_stats.point_count);
          }
          deskew_stats.scan_duration_s = std::max(
            deskew_stats.scan_duration_s,
            secondary_deskew_stats.scan_duration_s);
          deskew_stats.applied = deskew_stats.applied || secondary_deskew_stats.applied;
          if (deskew_stats.point_time_field.empty()) {
            deskew_stats.point_time_field = secondary_deskew_stats.point_time_field;
          }
          if (deskew_stats.status == "disabled" || deskew_stats.status == "applied") {
            deskew_stats.status = secondary_deskew_stats.status;
          }
        }
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          2000,
          "Skipping secondary cloud: stamp delta %.3f s exceeds %.3f s",
          delta_s,
          max_secondary_stamp_delta_s_);
      }
    }
  }

  return filterCloud(merged);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::convertAndTransformCloud(
  const sensor_msgs::msg::PointCloud2 & msg,
  DeskewStats & deskew_stats)
{
  auto cloud = std::make_shared<CloudT>();
  pcl::fromROSMsg(msg, *cloud);
  if (cloud->empty()) {
    return cloud;
  }

  const auto base_T_sensor = lookupTransformToBase(msg);
  auto cloud_base = transformCloud(cloud, base_T_sensor);
  deskewCloudInBaseFrame(msg, cloud_base, deskew_stats);
  return cloud_base;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::filterCloud(const CloudConstPtr & cloud) const
{
  auto filtered = std::make_shared<CloudT>();
  filtered->reserve(cloud->size());

  for (const auto & point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }

    const double range = std::sqrt(
      static_cast<double>(point.x) * point.x +
      static_cast<double>(point.y) * point.y +
      static_cast<double>(point.z) * point.z);
    if (range < min_range_m_ || range > max_range_m_) {
      continue;
    }
    if (point.z < z_min_m_ || point.z > z_max_m_) {
      continue;
    }

    if (crop_box_enabled_ &&
      point.x >= crop_min_x_ && point.x <= crop_max_x_ &&
      point.y >= crop_min_y_ && point.y <= crop_max_y_ &&
      point.z >= crop_min_z_ && point.z <= crop_max_z_)
    {
      continue;
    }

    filtered->push_back(point);
  }

  filtered->width = static_cast<std::uint32_t>(filtered->size());
  filtered->height = 1;
  filtered->is_dense = true;
  return filtered;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::downsampleCloud(
  const CloudConstPtr & cloud,
  const double leaf_size) const
{
  if (!cloud || cloud->empty() || leaf_size <= 0.0) {
    auto copy = std::make_shared<CloudT>();
    if (cloud) {
      *copy = *cloud;
    }
    return copy;
  }

  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize(
    static_cast<float>(leaf_size),
    static_cast<float>(leaf_size),
    static_cast<float>(leaf_size));
  voxel_grid.setInputCloud(cloud);

  auto downsampled = std::make_shared<CloudT>();
  voxel_grid.filter(*downsampled);
  return downsampled;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::downsampleScanAdaptive(
  const CloudConstPtr & cloud,
  double & effective_leaf_size) const
{
  effective_leaf_size = scan_voxel_leaf_m_;
  auto downsampled = downsampleCloud(cloud, scan_voxel_leaf_m_);

  if (!adaptive_scan_voxel_enabled_ || !cloud ||
    adaptive_scan_voxel_leaf_m_ <= 0.0 ||
    adaptive_scan_voxel_leaf_m_ >= scan_voxel_leaf_m_)
  {
    return downsampled;
  }

  if (static_cast<int>(downsampled->size()) >= adaptive_scan_voxel_min_points_) {
    return downsampled;
  }

  effective_leaf_size = adaptive_scan_voxel_leaf_m_;
  return downsampleCloud(cloud, adaptive_scan_voxel_leaf_m_);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::transformCloud(
  const CloudConstPtr & cloud,
  const Eigen::Isometry3d & transform) const
{
  auto transformed = std::make_shared<CloudT>();
  transformed->reserve(cloud->size());

  for (const auto & point : cloud->points) {
    const Eigen::Vector3d p(point.x, point.y, point.z);
    const Eigen::Vector3d q = transform * p;
    transformed->push_back(
      PointT(
        static_cast<float>(q.x()),
        static_cast<float>(q.y()),
        static_cast<float>(q.z())));
  }

  transformed->width = static_cast<std::uint32_t>(transformed->size());
  transformed->height = 1;
  transformed->is_dense = cloud->is_dense;
  return transformed;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::filterMapCloud(const CloudConstPtr & cloud) const
{
  if (!map_z_filter_enabled_ || !cloud) {
    auto copy = std::make_shared<CloudT>();
    if (cloud) {
      *copy = *cloud;
    }
    return copy;
  }

  const double min_z = std::min(map_z_filter_min_m_, map_z_filter_max_m_);
  const double max_z = std::max(map_z_filter_min_m_, map_z_filter_max_m_);
  auto filtered = std::make_shared<CloudT>();
  filtered->reserve(cloud->size());
  for (const auto & point : cloud->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    if (point.z < min_z || point.z > max_z) {
      continue;
    }
    filtered->push_back(point);
  }
  filtered->width = static_cast<std::uint32_t>(filtered->size());
  filtered->height = 1;
  filtered->is_dense = true;
  return filtered;
}

bool ScanToMapIcp::deskewCloudInBaseFrame(
  const sensor_msgs::msg::PointCloud2 & msg,
  const CloudPtr & cloud_base,
  DeskewStats & stats) const
{
  stats = DeskewStats{};
  if (!deskew_enabled_) {
    stats.status = "disabled";
    return false;
  }

  stats.attempted = true;
  if (!cloud_base || cloud_base->empty()) {
    stats.status = "empty_cloud";
    return false;
  }
  if (msg.is_bigendian) {
    stats.status = "big_endian_unsupported";
    return false;
  }
  if (msg.width == 0 || msg.point_step == 0 || msg.row_step == 0) {
    stats.status = "invalid_cloud_layout";
    return false;
  }

  const sensor_msgs::msg::PointField * time_field = nullptr;
  auto find_field = [&msg](const std::string & name) -> const sensor_msgs::msg::PointField * {
      for (const auto & field : msg.fields) {
        if (field.name == name && fieldWithinPointStep(field, msg.point_step)) {
          return &field;
        }
      }
      return nullptr;
    };

  if (deskew_point_time_field_ != "auto") {
    time_field = find_field(deskew_point_time_field_);
  } else {
    const std::array<std::string, 6> candidate_fields = {
      "timestamp",
      "timestamps",
      "time",
      "t",
      "offset_time",
      "stamps"};
    for (const auto & name : candidate_fields) {
      time_field = find_field(name);
      if (time_field) {
        break;
      }
    }
  }

  if (!time_field) {
    stats.status = "missing_point_time";
    return false;
  }
  stats.point_time_field = time_field->name;

  const double header_time_s = rclcpp::Time(msg.header.stamp).seconds();
  auto decode_point_time = [this, &header_time_s, &time_field](const double raw_time) {
      if (!std::isfinite(raw_time)) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      double decoded_time_s = 0.0;
      if (deskew_point_time_scale_ > 0.0) {
        const double scaled = raw_time * deskew_point_time_scale_;
        if (std::abs(scaled) > 1.0e8) {
          decoded_time_s = scaled;
        } else {
          decoded_time_s = header_time_s + scaled;
        }
        return decoded_time_s + deskew_point_time_offset_s_;
      }
      const double abs_raw_time = std::abs(raw_time);
      if (abs_raw_time > 1.0e8) {
        decoded_time_s = raw_time;
        return decoded_time_s + deskew_point_time_offset_s_;
      }
      if (time_field->name == "offset_time") {
        decoded_time_s = header_time_s + raw_time * 1.0e-9;
        return decoded_time_s + deskew_point_time_offset_s_;
      }
      if (abs_raw_time > 1.0e6) {
        decoded_time_s = header_time_s + raw_time * 1.0e-9;
        return decoded_time_s + deskew_point_time_offset_s_;
      }
      if (abs_raw_time > 1.0e3) {
        decoded_time_s = header_time_s + raw_time * 1.0e-6;
        return decoded_time_s + deskew_point_time_offset_s_;
      }
      decoded_time_s = header_time_s + raw_time;
      return decoded_time_s + deskew_point_time_offset_s_;
    };

  const std::size_t cloud_point_count =
    static_cast<std::size_t>(msg.width) * static_cast<std::size_t>(msg.height);
  const std::size_t point_count = std::min(cloud_base->points.size(), cloud_point_count);
  stats.point_count = static_cast<int>(point_count);
  if (point_count == 0) {
    stats.status = "empty_cloud";
    return false;
  }

  std::vector<double> point_times(point_count, std::numeric_limits<double>::quiet_NaN());
  double min_time_s = std::numeric_limits<double>::infinity();
  double max_time_s = -std::numeric_limits<double>::infinity();

  for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
    const std::size_t row = point_index / static_cast<std::size_t>(msg.width);
    const std::size_t col = point_index % static_cast<std::size_t>(msg.width);
    const std::size_t byte_index =
      row * static_cast<std::size_t>(msg.row_step) +
      col * static_cast<std::size_t>(msg.point_step);
    if (byte_index + msg.point_step > msg.data.size()) {
      continue;
    }

    const auto raw_value = readPointFieldAsDouble(msg.data.data() + byte_index, *time_field);
    if (!raw_value) {
      continue;
    }
    const double point_time_s = decode_point_time(*raw_value);
    if (!std::isfinite(point_time_s)) {
      continue;
    }
    point_times[point_index] = point_time_s;
    min_time_s = std::min(min_time_s, point_time_s);
    max_time_s = std::max(max_time_s, point_time_s);
    ++stats.points_with_time;
  }

  stats.time_ratio =
    static_cast<double>(stats.points_with_time) / static_cast<double>(stats.point_count);
  if (stats.points_with_time == 0 ||
    stats.time_ratio < std::clamp(deskew_min_time_ratio_, 0.0, 1.0))
  {
    stats.status = "insufficient_point_time";
    return false;
  }

  stats.scan_duration_s = max_time_s - min_time_s;
  const double max_scan_duration = std::max(0.0, deskew_max_scan_duration_s_);
  if (stats.scan_duration_s <= 1.0e-6) {
    stats.status = "zero_scan_duration";
    return false;
  }
  if (max_scan_duration > 0.0 && stats.scan_duration_s > max_scan_duration) {
    stats.status = "scan_duration_out_of_range";
    return false;
  }

  std::vector<ImuSample> imu_samples;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_samples.assign(imu_buffer_.begin(), imu_buffer_.end());
  }
  if (imu_samples.empty()) {
    stats.status = "missing_imu";
    return false;
  }

  const double base_time_s = min_time_s;
  const double reference_time_s = std::clamp(header_time_s, min_time_s, max_time_s);
  const double end_time_s = max_time_s;
  const double max_imu_gap = std::max(0.001, deskew_max_imu_gap_s_);
  auto sample_time_s = [](const ImuSample & sample) {
      return sample.stamp.seconds();
    };

  if (sample_time_s(imu_samples.front()) > base_time_s + max_imu_gap ||
    sample_time_s(imu_samples.back()) < end_time_s - max_imu_gap)
  {
    stats.status = "insufficient_imu_coverage";
    return false;
  }

  double previous_coverage_time_s = base_time_s;
  bool saw_sample_in_window = false;
  for (const auto & sample : imu_samples) {
    const double stamp_s = sample_time_s(sample);
    if (stamp_s < base_time_s) {
      continue;
    }
    if (stamp_s > end_time_s) {
      break;
    }
    saw_sample_in_window = true;
    if (stamp_s - previous_coverage_time_s > max_imu_gap) {
      stats.status = "imu_gap";
      return false;
    }
    previous_coverage_time_s = stamp_s;
  }
  if (!saw_sample_in_window || end_time_s - previous_coverage_time_s > max_imu_gap) {
    stats.status = "imu_gap";
    return false;
  }

  const Eigen::Matrix3d imu_to_base = rotationFromRpyDegrees(
    deskew_imu_to_base_roll_deg_,
    deskew_imu_to_base_pitch_deg_,
    deskew_imu_to_base_yaw_deg_);
  const double angular_velocity_scale = deskew_angular_velocity_scale_;
  for (auto & sample : imu_samples) {
    sample.angular_velocity = imu_to_base * sample.angular_velocity * angular_velocity_scale;
  }

  std::size_t start_index = 0;
  while (start_index + 1 < imu_samples.size() &&
    sample_time_s(imu_samples[start_index + 1]) <= base_time_s)
  {
    ++start_index;
  }

  struct RotationCache
  {
    double stamp_s{0.0};
    Eigen::Quaterniond reference_R_current{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
  };

  std::vector<RotationCache> rotation_cache;
  rotation_cache.reserve(imu_samples.size() + 1);
  Eigen::Quaterniond current_rotation = Eigen::Quaterniond::Identity();
  double current_time_s = base_time_s;
  Eigen::Vector3d current_angular_velocity = imu_samples[start_index].angular_velocity;
  rotation_cache.push_back({current_time_s, current_rotation, current_angular_velocity});

  std::size_t next_sample_index =
    sample_time_s(imu_samples[start_index]) <= base_time_s ? start_index + 1 : start_index;
  for (; next_sample_index < imu_samples.size(); ++next_sample_index) {
    const auto & sample = imu_samples[next_sample_index];
    const double next_time_s = sample_time_s(sample);
    if (next_time_s > end_time_s) {
      break;
    }
    if (next_time_s > current_time_s) {
      const double dt_s = next_time_s - current_time_s;
      current_rotation =
        (current_rotation * deltaRotationFromAngularVelocity(current_angular_velocity, dt_s))
        .normalized();
      current_time_s = next_time_s;
    }
    current_angular_velocity = sample.angular_velocity;
    rotation_cache.push_back({current_time_s, current_rotation, current_angular_velocity});
  }

  auto rotation_at = [&rotation_cache](const double point_time_s) {
      const auto cache_it = std::upper_bound(
        rotation_cache.begin(),
        rotation_cache.end(),
        point_time_s,
        [](const double time_s, const RotationCache & cache) {
          return time_s < cache.stamp_s;
        });
      const auto & cache = cache_it == rotation_cache.begin() ?
        rotation_cache.front() : *(cache_it - 1);
      const double dt_s = std::max(0.0, point_time_s - cache.stamp_s);
      return (cache.reference_R_current *
             deltaRotationFromAngularVelocity(cache.angular_velocity, dt_s)).normalized();
    };

  const Eigen::Quaterniond base_R_reference = rotation_at(reference_time_s);
  const double time_bin_s = std::max(0.0, deskew_time_bin_s_);
  std::vector<Eigen::Quaterniond> binned_rotations;
  if (time_bin_s > 0.0) {
    const int bin_count = std::max(
      1,
      static_cast<int>(std::ceil((end_time_s - base_time_s) / time_bin_s)) + 1);
    binned_rotations.reserve(static_cast<std::size_t>(bin_count));
    for (int bin = 0; bin < bin_count; ++bin) {
      const double bin_time_s = std::min(
        end_time_s,
        base_time_s + static_cast<double>(bin) * time_bin_s);
      binned_rotations.push_back(rotation_at(bin_time_s));
    }
  }

  for (std::size_t point_index = 0; point_index < point_count; ++point_index) {
    const double point_time_s = point_times[point_index];
    if (!std::isfinite(point_time_s)) {
      continue;
    }

    Eigen::Quaterniond base_R_point = Eigen::Quaterniond::Identity();
    if (!binned_rotations.empty()) {
      const int bin = std::clamp(
        static_cast<int>(std::llround((point_time_s - base_time_s) / time_bin_s)),
        0,
        static_cast<int>(binned_rotations.size()) - 1);
      base_R_point = binned_rotations[static_cast<std::size_t>(bin)];
    } else {
      base_R_point = rotation_at(point_time_s);
    }
    const Eigen::Quaterniond point_rotation =
      (base_R_reference.conjugate() * base_R_point).normalized();

    auto & point = cloud_base->points[point_index];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const Eigen::Vector3d corrected =
      point_rotation * Eigen::Vector3d(point.x, point.y, point.z);
    point.x = static_cast<float>(corrected.x());
    point.y = static_cast<float>(corrected.y());
    point.z = static_cast<float>(corrected.z());
  }

  stats.applied = true;
  stats.status = "applied";
  return true;
}

void ScanToMapIcp::applyDeskewStatsToResult(
  const DeskewStats & stats,
  RegistrationResult & result) const
{
  result.deskew_attempted = stats.attempted;
  result.deskew_applied = stats.applied;
  result.deskew_points_with_time = stats.points_with_time;
  result.deskew_point_count = stats.point_count;
  result.deskew_time_ratio = stats.time_ratio;
  result.deskew_scan_duration_s = stats.scan_duration_s;
  result.deskew_status = stats.status;
  result.deskew_point_time_field = stats.point_time_field;
}

ScanToMapIcp::ImuRoughnessStats ScanToMapIcp::evaluateImuRoughness(
  const rclcpp::Time & start_stamp,
  const rclcpp::Time & end_stamp,
  const bool have_start_stamp) const
{
  ImuRoughnessStats stats;
  if (!imu_roughness_detection_enabled_) {
    return stats;
  }
  stats.evaluated = true;
  stats.reason = "nominal";

  const double window_s = std::max(0.01, imu_roughness_window_s_);
  double start_s = end_stamp.seconds() - window_s;
  if (have_start_stamp) {
    start_s = std::max(start_s, start_stamp.seconds());
  }
  const double end_s = end_stamp.seconds();
  if (!std::isfinite(start_s) || !std::isfinite(end_s) || end_s < start_s) {
    stats.reason = "invalid_window";
    return stats;
  }

  std::vector<ImuSample> samples;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    for (const auto & sample : imu_buffer_) {
      const double sample_s = sample.stamp.seconds();
      if (sample_s >= start_s && sample_s <= end_s) {
        samples.push_back(sample);
      }
    }
  }

  stats.sample_count = static_cast<int>(samples.size());
  if (samples.empty()) {
    stats.reason = "no_imu_samples";
    return stats;
  }

  double previous_accel_norm = samples.front().linear_acceleration.norm();
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto & sample = samples[index];
    const double angular_velocity_norm = sample.angular_velocity.norm();
    const double linear_accel_norm = sample.linear_acceleration.norm();
    if (std::isfinite(angular_velocity_norm)) {
      stats.max_angular_velocity_radps =
        std::max(stats.max_angular_velocity_radps, angular_velocity_norm);
    }
    if (std::isfinite(linear_accel_norm)) {
      stats.max_linear_accel_mps2 =
        std::max(stats.max_linear_accel_mps2, linear_accel_norm);
      const double accel_norm_delta = std::abs(linear_accel_norm - previous_accel_norm);
      stats.max_accel_norm_delta_mps2 =
        std::max(stats.max_accel_norm_delta_mps2, accel_norm_delta);
      previous_accel_norm = linear_accel_norm;
    }

    if (index == 0U) {
      continue;
    }

    const auto & previous = samples[index - 1U];
    const double dt_s = (sample.stamp - previous.stamp).seconds();
    if (!std::isfinite(dt_s) || dt_s <= 1.0e-6) {
      continue;
    }
    const double angular_accel =
      (sample.angular_velocity - previous.angular_velocity).norm() / dt_s;
    const double linear_jerk =
      (sample.linear_acceleration - previous.linear_acceleration).norm() / dt_s;
    if (std::isfinite(angular_accel)) {
      stats.max_angular_accel_radps2 =
        std::max(stats.max_angular_accel_radps2, angular_accel);
    }
    if (std::isfinite(linear_jerk)) {
      stats.max_linear_jerk_mps3 =
        std::max(stats.max_linear_jerk_mps3, linear_jerk);
    }
  }

  if (stats.max_angular_velocity_radps >
    std::max(0.0, imu_roughness_angular_velocity_threshold_radps_))
  {
    stats.event = true;
    stats.reason = "angular_velocity";
  }
  if (stats.max_angular_accel_radps2 >
    std::max(0.0, imu_roughness_angular_accel_threshold_radps2_))
  {
    stats.event = true;
    stats.reason = stats.reason == "nominal" ? "angular_accel" : stats.reason + "+angular_accel";
  }
  if (stats.max_accel_norm_delta_mps2 >
    std::max(0.0, imu_roughness_accel_norm_delta_threshold_mps2_))
  {
    stats.event = true;
    stats.reason = stats.reason == "nominal" ? "accel_norm_delta" :
      stats.reason + "+accel_norm_delta";
  }
  if (stats.max_linear_jerk_mps3 >
    std::max(0.0, imu_roughness_linear_jerk_threshold_mps3_))
  {
    stats.event = true;
    stats.reason = stats.reason == "nominal" ? "linear_jerk" : stats.reason + "+linear_jerk";
  }

  return stats;
}

void ScanToMapIcp::applyImuRoughnessStatsToResult(
  const ImuRoughnessStats & stats,
  RegistrationResult & result) const
{
  result.imu_roughness_evaluated = stats.evaluated;
  result.imu_roughness_event = stats.event;
  result.imu_roughness_recent_event = stats.recent_event;
  result.imu_roughness_sample_count = stats.sample_count;
  result.imu_roughness_max_angular_velocity_radps = stats.max_angular_velocity_radps;
  result.imu_roughness_max_angular_accel_radps2 = stats.max_angular_accel_radps2;
  result.imu_roughness_max_linear_accel_mps2 = stats.max_linear_accel_mps2;
  result.imu_roughness_max_accel_norm_delta_mps2 = stats.max_accel_norm_delta_mps2;
  result.imu_roughness_max_linear_jerk_mps3 = stats.max_linear_jerk_mps3;
  result.imu_roughness_time_since_event_s = stats.time_since_event_s;
  result.imu_roughness_reason = stats.reason;
}

Eigen::Isometry3d ScanToMapIcp::lookupTransformToBase(
  const sensor_msgs::msg::PointCloud2 & msg)
{
  if (!use_tf_transforms_ || msg.header.frame_id.empty() || msg.header.frame_id == base_frame_) {
    return Eigen::Isometry3d::Identity();
  }

  try {
    const auto tf = tf_buffer_->lookupTransform(
      base_frame_,
      msg.header.frame_id,
      msg.header.stamp,
      rclcpp::Duration::from_seconds(0.05));

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    const auto & translation = tf.transform.translation;
    const auto & rotation = tf.transform.rotation;
    const Eigen::Quaterniond q(rotation.w, rotation.x, rotation.y, rotation.z);
    transform.linear() = q.normalized().toRotationMatrix();
    transform.translation() = Eigen::Vector3d(translation.x, translation.y, translation.z);
    return transform;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "TF lookup failed %s -> %s: %s. Using identity for this scan.",
      msg.header.frame_id.c_str(),
      base_frame_.c_str(),
      ex.what());
    return Eigen::Isometry3d::Identity();
  }
}

Eigen::Isometry3d ScanToMapIcp::predictWithWheelOdom()
{
  Eigen::Isometry3d prediction = map_T_base_;
  if (!use_wheel_prediction_) {
    return predictWithConstantVelocity();
  }

  std::lock_guard<std::mutex> lock(wheel_mutex_);
  if (!have_latest_wheel_odom_) {
    return predictWithConstantVelocity();
  }

  if (!have_last_used_wheel_odom_) {
    last_used_wheel_odom_ = latest_wheel_odom_;
    have_last_used_wheel_odom_ = true;
    return predictWithConstantVelocity();
  }

  pending_wheel_odom_ = latest_wheel_odom_;
  have_pending_wheel_odom_ = true;
  const Eigen::Isometry3d delta = last_used_wheel_odom_.inverse() * pending_wheel_odom_;
  prediction = map_T_base_ * delta;
  return prediction;
}

Eigen::Isometry3d ScanToMapIcp::predictWithConstantVelocity() const
{
  if (!use_constant_velocity_prediction_ || !have_previous_accepted_pose_) {
    return map_T_base_;
  }

  const Eigen::Isometry3d delta = previous_accepted_pose_.inverse() * last_accepted_pose_;
  return map_T_base_ * clampRelativeMotion(delta);
}

Eigen::Isometry3d ScanToMapIcp::clampRelativeMotion(const Eigen::Isometry3d & delta) const
{
  Eigen::Isometry3d clamped = delta;
  const double max_translation = std::max(0.0, constant_velocity_translation_limit_m_);
  const double translation_norm = clamped.translation().norm();
  if (max_translation > 0.0 && translation_norm > max_translation) {
    clamped.translation() *= max_translation / translation_norm;
  }

  const double max_rotation = std::max(0.0, constant_velocity_rotation_limit_deg_) * M_PI / 180.0;
  if (max_rotation > 0.0) {
    Eigen::AngleAxisd rotation(clamped.linear());
    double angle = normalizedAngle(rotation.angle());
    if (std::abs(angle) > max_rotation) {
      angle = std::copysign(max_rotation, angle);
      clamped.linear() = Eigen::AngleAxisd(angle, rotation.axis()).toRotationMatrix();
    }
  }

  return clamped;
}

Eigen::Isometry3d ScanToMapIcp::constrainMotionPriorFallback(
  const Eigen::Isometry3d & candidate) const
{
  Eigen::Isometry3d constrained = candidate;

  auto clamp_translation_around =
    [](Eigen::Vector3d translation,
      const Eigen::Vector3d & reference,
      double max_distance,
      double max_z) {
      Eigen::Vector3d delta = translation - reference;
      if (max_distance > 0.0) {
        const double distance = delta.norm();
        if (distance > max_distance) {
          translation = reference + delta * (max_distance / distance);
          delta = translation - reference;
        }
      }
      if (max_z > 0.0) {
        delta.z() = std::clamp(delta.z(), -max_z, max_z);
        translation.z() = reference.z() + delta.z();
      }
      return translation;
    };

  Eigen::Vector3d translation = constrained.translation();
  if (have_last_accepted_pose_) {
    translation = clamp_translation_around(
      translation,
      last_accepted_pose_.translation(),
      motion_prior_fallback_max_untrusted_distance_m_,
      motion_prior_fallback_max_untrusted_z_m_);
  }

  translation = clamp_translation_around(
    translation,
    map_T_base_.translation(),
    motion_prior_fallback_max_step_m_,
    motion_prior_fallback_max_z_step_m_);
  constrained.translation() = translation;

  const double max_rotation =
    std::max(0.0, motion_prior_fallback_max_rotation_deg_) * M_PI / 180.0;
  if (max_rotation > 0.0) {
    Eigen::Quaterniond reference(map_T_base_.linear());
    Eigen::Quaterniond target(constrained.linear());
    reference.normalize();
    target.normalize();

    const double angular_distance = reference.angularDistance(target);
    if (angular_distance > max_rotation) {
      const double ratio = max_rotation / angular_distance;
      constrained.linear() = reference.slerp(ratio, target).normalized().toRotationMatrix();
    }
  }

  return constrained;
}

std::optional<ScanToMapIcp::GravitySample> ScanToMapIcp::gravitySampleAt(
  const rclcpp::Time & stamp,
  bool & stale) const
{
  stale = false;
  if (!gravity_roll_pitch_constraint_enabled_) {
    return std::nullopt;
  }

  std::lock_guard<std::mutex> lock(imu_mutex_);
  for (auto iterator = gravity_buffer_.rbegin(); iterator != gravity_buffer_.rend(); ++iterator) {
    if (iterator->stamp <= stamp && iterator->accel_base.allFinite() &&
      iterator->accel_base.norm() > 1.0e-6)
    {
      const double age_s = (stamp - iterator->stamp).seconds();
      const double max_age_s = std::max(0.0, gravity_accel_max_age_s_);
      if (max_age_s > 0.0 && age_s > max_age_s) {
        stale = true;
        return std::nullopt;
      }
      return *iterator;
    }
  }
  return std::nullopt;
}

Eigen::Isometry3d ScanToMapIcp::constrainPlanarPose(
  const Eigen::Isometry3d & candidate,
  const Eigen::Isometry3d & reference,
  const std::optional<GravitySample> & gravity_sample) const
{
  if (!planar_pose_constraint_enabled_ && !gravity_roll_pitch_constraint_enabled_) {
    return candidate;
  }
  if (!planar_pose_constraint_enabled_ && !gravity_sample.has_value()) {
    return candidate;
  }

  Eigen::Isometry3d constrained = Eigen::Isometry3d::Identity();
  constrained.translation() = candidate.translation();
  const double max_z_correction = std::max(0.0, planar_pose_max_z_correction_m_);
  if (max_z_correction > 0.0) {
    const double z_delta = std::clamp(
      candidate.translation().z() - reference.translation().z(),
      -max_z_correction,
      max_z_correction);
    constrained.translation().z() = reference.translation().z() + z_delta;
  }
  const double yaw = yawFromIsometry(candidate);

  const bool have_gravity = gravity_sample.has_value();
  const Eigen::Vector3d gravity_accel = have_gravity ?
    gravity_sample->accel_base.normalized() :
    Eigen::Vector3d::UnitZ();

  double roll = 0.0;
  double pitch = 0.0;
  if (have_gravity) {
    roll = std::atan2(gravity_accel.y(), gravity_accel.z());
    pitch = std::atan2(
      -gravity_accel.x(),
      std::hypot(gravity_accel.y(), gravity_accel.z()));
  }

  const Eigen::AngleAxisd roll_rotation(roll, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_rotation(pitch, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());
  constrained.linear() = (yaw_rotation * pitch_rotation * roll_rotation).toRotationMatrix();
  return constrained;
}

Eigen::Isometry3d ScanToMapIcp::predictWithMotionPriors(
  const CloudConstPtr & scan_base,
  const rclcpp::Time & previous_scan_stamp,
  const rclcpp::Time & current_scan_stamp,
  const bool have_previous_scan_stamp,
  bool & used_scan_to_scan,
  bool & used_imu_rotation_prediction,
  std::string & imu_rotation_prediction_status,
  double & scan_to_scan_reference_age_s,
  std::string & scan_to_scan_prediction_status)
{
  used_scan_to_scan = false;
  used_imu_rotation_prediction = false;
  imu_rotation_prediction_status = imu_rotation_prediction_enabled_ ? "unavailable" : "disabled";
  scan_to_scan_reference_age_s = 0.0;
  scan_to_scan_prediction_status = use_scan_to_scan_prediction_ ? "unavailable" : "disabled";

  Eigen::Isometry3d fallback = predictWithWheelOdom();
  Eigen::Isometry3d scan_to_scan_guess = Eigen::Isometry3d::Identity();
  std::optional<Eigen::Quaterniond> previous_scan_imu_delta;
  if (imu_rotation_prediction_enabled_) {
    if (have_previous_scan_stamp) {
      previous_scan_imu_delta = integrateImuRotationDelta(
        previous_scan_stamp,
        current_scan_stamp,
        imu_rotation_prediction_status);
    } else {
      imu_rotation_prediction_status = "missing_previous_scan_stamp";
    }
    if (previous_scan_imu_delta) {
      const Eigen::Matrix3d imu_rotation =
        previous_scan_imu_delta->normalized().toRotationMatrix();
      fallback.linear() = map_T_base_.linear() * imu_rotation;
      used_imu_rotation_prediction = true;
      imu_rotation_prediction_status = "applied";
    }
  }

  if (!use_scan_to_scan_prediction_ || !have_previous_tracking_scan_ ||
    !previous_tracking_scan_base_)
  {
    if (use_scan_to_scan_prediction_) {
      scan_to_scan_prediction_status = "missing_tracking_reference";
    }
    return fallback;
  }

  scan_to_scan_reference_age_s =
    (current_scan_stamp - previous_tracking_scan_stamp_).seconds();
  if (!std::isfinite(scan_to_scan_reference_age_s) ||
    scan_to_scan_reference_age_s <= 0.0)
  {
    scan_to_scan_prediction_status = "invalid_tracking_reference_age";
    return fallback;
  }
  const double max_reference_age_s = std::max(0.0, scan_to_scan_max_reference_age_s_);
  if (max_reference_age_s > 0.0 &&
    scan_to_scan_reference_age_s > max_reference_age_s)
  {
    scan_to_scan_prediction_status = "stale_tracking_reference";
    return fallback;
  }

  if (imu_rotation_prediction_enabled_) {
    const bool tracking_reference_is_previous_scan =
      have_previous_scan_stamp &&
      std::abs((previous_tracking_scan_stamp_ - previous_scan_stamp).seconds()) <= 1.0e-6;
    std::optional<Eigen::Quaterniond> tracking_imu_delta;
    std::string tracking_imu_status{"unavailable"};
    if (tracking_reference_is_previous_scan) {
      tracking_imu_delta = previous_scan_imu_delta;
      tracking_imu_status = imu_rotation_prediction_status;
    } else {
      tracking_imu_delta = integrateImuRotationDelta(
        previous_tracking_scan_stamp_,
        current_scan_stamp,
        tracking_imu_status);
    }
    if (tracking_imu_delta) {
      scan_to_scan_guess.linear() =
        tracking_imu_delta->normalized().toRotationMatrix();
      used_imu_rotation_prediction = true;
      imu_rotation_prediction_status = "applied";
    } else if (!previous_scan_imu_delta) {
      imu_rotation_prediction_status = tracking_imu_status;
    }
  }

  auto relative = registerScanToMap(
    scan_base,
    previous_tracking_scan_base_,
    scan_to_scan_guess);
  if (!relative.accepted) {
    scan_to_scan_prediction_status = "registration_rejected";
    return fallback;
  }

  used_scan_to_scan = true;
  scan_to_scan_prediction_status = "applied";
  return previous_tracking_pose_ * relative.map_T_base;
}

std::optional<Eigen::Quaterniond> ScanToMapIcp::integrateImuRotationDelta(
  const rclcpp::Time & start_stamp,
  const rclcpp::Time & end_stamp,
  std::string & status) const
{
  status = imu_rotation_prediction_enabled_ ? "unavailable" : "disabled";
  if (!imu_rotation_prediction_enabled_) {
    return std::nullopt;
  }

  const double start_s = start_stamp.seconds();
  const double end_s = end_stamp.seconds();
  const double dt_s = end_s - start_s;
  if (!std::isfinite(dt_s) || dt_s <= 1.0e-6) {
    status = "invalid_interval";
    return std::nullopt;
  }

  const double max_dt_s = std::max(0.0, imu_rotation_prediction_max_dt_s_);
  if (max_dt_s > 0.0 && dt_s > max_dt_s) {
    status = "interval_too_long";
    return std::nullopt;
  }

  std::vector<ImuSample> imu_samples;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_samples.assign(imu_buffer_.begin(), imu_buffer_.end());
  }
  if (imu_samples.empty()) {
    status = "missing_imu";
    return std::nullopt;
  }

  auto sample_time_s = [](const ImuSample & sample) {
      return sample.stamp.seconds();
    };
  const double max_imu_gap = std::max(0.001, deskew_max_imu_gap_s_);
  if (sample_time_s(imu_samples.front()) > start_s + max_imu_gap ||
    sample_time_s(imu_samples.back()) < end_s - max_imu_gap)
  {
    status = "insufficient_imu_coverage";
    return std::nullopt;
  }

  const Eigen::Matrix3d imu_to_base = rotationFromRpyDegrees(
    deskew_imu_to_base_roll_deg_,
    deskew_imu_to_base_pitch_deg_,
    deskew_imu_to_base_yaw_deg_);
  const double angular_velocity_scale =
    deskew_angular_velocity_scale_ * imu_rotation_prediction_scale_;
  for (auto & sample : imu_samples) {
    sample.angular_velocity = imu_to_base * sample.angular_velocity * angular_velocity_scale;
  }

  std::size_t start_index = 0;
  while (start_index + 1 < imu_samples.size() &&
    sample_time_s(imu_samples[start_index + 1]) <= start_s)
  {
    ++start_index;
  }

  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  double current_time_s = start_s;
  Eigen::Vector3d current_angular_velocity = imu_samples[start_index].angular_velocity;
  std::size_t next_sample_index =
    sample_time_s(imu_samples[start_index]) <= start_s ? start_index + 1 : start_index;
  for (; next_sample_index < imu_samples.size(); ++next_sample_index) {
    const auto & sample = imu_samples[next_sample_index];
    const double next_time_s = sample_time_s(sample);
    if (next_time_s > end_s) {
      break;
    }
    if (next_time_s - current_time_s > max_imu_gap) {
      status = "imu_gap";
      return std::nullopt;
    }
    if (next_time_s > current_time_s) {
      rotation =
        (rotation * deltaRotationFromAngularVelocity(
          current_angular_velocity,
          next_time_s - current_time_s)).normalized();
      current_time_s = next_time_s;
    }
    current_angular_velocity = sample.angular_velocity;
  }

  if (end_s - current_time_s > max_imu_gap) {
    status = "imu_gap";
    return std::nullopt;
  }
  if (end_s > current_time_s) {
    rotation =
      (rotation * deltaRotationFromAngularVelocity(
        current_angular_velocity,
        end_s - current_time_s)).normalized();
  }

  status = "applied";
  return rotation.normalized();
}

void ScanToMapIcp::commitWheelPrediction()
{
  std::lock_guard<std::mutex> lock(wheel_mutex_);
  if (!have_pending_wheel_odom_) {
    return;
  }

  last_used_wheel_odom_ = pending_wheel_odom_;
  have_pending_wheel_odom_ = false;
}

void ScanToMapIcp::recordAcceptedPose()
{
  if (!have_last_accepted_pose_) {
    last_accepted_pose_ = map_T_base_;
    have_last_accepted_pose_ = true;
    return;
  }

  previous_accepted_pose_ = last_accepted_pose_;
  last_accepted_pose_ = map_T_base_;
  have_previous_accepted_pose_ = true;
}

void ScanToMapIcp::recordTrackingScan(
  const CloudConstPtr & scan_base,
  const Eigen::Isometry3d & pose,
  const rclcpp::Time & stamp)
{
  previous_tracking_scan_base_ = std::make_shared<CloudT>(*scan_base);
  previous_tracking_pose_ = pose;
  previous_tracking_scan_stamp_ = stamp;
  have_previous_tracking_scan_ = true;
}

ScanToMapIcp::RegistrationResult ScanToMapIcp::registerScanToMap(
  const CloudConstPtr & scan_base,
  const CloudConstPtr & local_map,
  const Eigen::Isometry3d & prediction) const
{
  const double coarse_max_correspondence_m = std::max(1.0e-3, icp_max_correspondence_m_);
  const double requested_quality_distance = robust_registration_enabled_ ?
    robust_registration_quality_max_correspondence_m_ : coarse_max_correspondence_m;
  const double quality_max_correspondence_m = std::clamp(
    requested_quality_distance,
    1.0e-3,
    coarse_max_correspondence_m);
  const double trim_ratio = std::clamp(robust_registration_trim_ratio_, 0.05, 1.0);
  const bool use_trimmed_correspondences =
    robust_registration_enabled_ && registration_method_ == "icp" && trim_ratio < 0.999;

  auto run_stage = [this, &scan_base, &local_map, &prediction,
      quality_max_correspondence_m, trim_ratio](
    const Eigen::Matrix4f & guess,
    const double max_correspondence_m,
    const int max_iterations,
    const bool use_trimmed) {
      RegistrationResult stage_result;
      stage_result.map_T_base = prediction;
      stage_result.fitness = std::numeric_limits<double>::infinity();
      stage_result.registration_quality_max_correspondence_m =
        quality_max_correspondence_m;
      CloudT aligned;

      if (registration_method_ == "icp") {
        pcl::IterativeClosestPoint<PointT, PointT> icp;
        icp.setInputSource(scan_base);
        icp.setInputTarget(local_map);
        icp.setMaximumIterations(std::max(1, max_iterations));
        icp.setMaxCorrespondenceDistance(std::max(1.0e-3, max_correspondence_m));
        if (use_trimmed) {
          pcl::registration::CorrespondenceRejectorTrimmed::Ptr trimmed_rejector(
            new pcl::registration::CorrespondenceRejectorTrimmed());
          trimmed_rejector->setOverlapRatio(static_cast<float>(trim_ratio));
          trimmed_rejector->setMinCorrespondences(
            static_cast<unsigned int>(std::max(3, min_scan_points_)));
          icp.addCorrespondenceRejector(trimmed_rejector);
        }
        icp.align(aligned, guess);
        stage_result.converged = icp.hasConverged();
        stage_result.fitness = icp.getFitnessScore(quality_max_correspondence_m);
        stage_result.iterations = icp.nr_iterations_;
        stage_result.map_T_base.matrix() = icp.getFinalTransformation().cast<double>();
      } else {
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> gicp;
        gicp.setInputSource(scan_base);
        gicp.setInputTarget(local_map);
        gicp.setMaximumIterations(std::max(1, max_iterations));
        gicp.setMaxCorrespondenceDistance(std::max(1.0e-3, max_correspondence_m));
        gicp.align(aligned, guess);
        stage_result.converged = gicp.hasConverged();
        stage_result.fitness = gicp.getFitnessScore(quality_max_correspondence_m);
        stage_result.iterations = gicp.nr_iterations_;
        stage_result.map_T_base.matrix() = gicp.getFinalTransformation().cast<double>();
      }

      const Eigen::Matrix3d linear = stage_result.map_T_base.linear();
      stage_result.transform_determinant = linear.determinant();
      stage_result.transform_orthogonality_error =
        (linear.transpose() * linear - Eigen::Matrix3d::Identity()).norm();
      const Eigen::Isometry3d correction = prediction.inverse() * stage_result.map_T_base;
      stage_result.correction_translation_m = correction.translation().norm();
      stage_result.correction_rotation_deg =
        Eigen::AngleAxisd(correction.linear()).angle() * 180.0 / M_PI;
      evaluateRegistrationOverlap(
        aligned,
        local_map,
        quality_max_correspondence_m,
        stage_result);
      stage_result.accepted = registrationCorrectionWithinLimits(stage_result);
      return stage_result;
    };

  auto result = run_stage(
    prediction.matrix().cast<float>(),
    coarse_max_correspondence_m,
    max_icp_iterations_,
    use_trimmed_correspondences);
  result.registration_trimmed_correspondences_applied = use_trimmed_correspondences;
  result.registration_coarse_iterations = result.iterations;
  result.registration_coarse_fitness = result.fitness;
  result.registration_coarse_inlier_ratio = result.inlier_ratio;
  result.registration_coarse_inlier_rmse_m = result.inlier_rmse_m;

  const double refinement_max_correspondence_m = std::clamp(
    robust_registration_refinement_max_correspondence_m_,
    1.0e-3,
    coarse_max_correspondence_m);
  const bool refinement_configured = robust_registration_enabled_ &&
    robust_registration_refinement_iterations_ > 0 &&
    refinement_max_correspondence_m + 1.0e-6 < coarse_max_correspondence_m;
  const bool coarse_transform_valid = result.converged && result.map_T_base.matrix().allFinite();
  if (!refinement_configured || !coarse_transform_valid) {
    return result;
  }

  result.registration_refinement_attempted = true;
  auto refined_result = run_stage(
    result.map_T_base.matrix().cast<float>(),
    refinement_max_correspondence_m,
    robust_registration_refinement_iterations_,
    use_trimmed_correspondences);
  result.registration_refinement_iterations = refined_result.iterations;
  result.registration_refinement_fitness = refined_result.fitness;
  result.registration_refinement_inlier_ratio = refined_result.inlier_ratio;
  result.registration_refinement_inlier_rmse_m = refined_result.inlier_rmse_m;

  auto registration_quality_score = [](const RegistrationResult & candidate) {
      if (!std::isfinite(candidate.fitness)) {
        return std::numeric_limits<double>::infinity();
      }
      return candidate.fitness / std::max(0.05, candidate.inlier_ratio);
    };
  const bool refinement_improved_quality = !result.accepted ||
    registration_quality_score(refined_result) + 1.0e-9 < registration_quality_score(result);
  if (!refined_result.accepted || !refinement_improved_quality) {
    return result;
  }

  refined_result.registration_trimmed_correspondences_applied = use_trimmed_correspondences;
  refined_result.registration_refinement_attempted = true;
  refined_result.registration_refinement_applied = true;
  refined_result.registration_coarse_iterations = result.registration_coarse_iterations;
  refined_result.registration_refinement_iterations = result.registration_refinement_iterations;
  refined_result.registration_coarse_fitness = result.registration_coarse_fitness;
  refined_result.registration_coarse_inlier_ratio = result.registration_coarse_inlier_ratio;
  refined_result.registration_coarse_inlier_rmse_m = result.registration_coarse_inlier_rmse_m;
  refined_result.registration_refinement_fitness = result.registration_refinement_fitness;
  refined_result.registration_refinement_inlier_ratio =
    result.registration_refinement_inlier_ratio;
  refined_result.registration_refinement_inlier_rmse_m =
    result.registration_refinement_inlier_rmse_m;
  refined_result.iterations =
    result.registration_coarse_iterations + result.registration_refinement_iterations;
  return refined_result;
}

void ScanToMapIcp::evaluateRegistrationOverlap(
  const CloudT & aligned_scan,
  const CloudConstPtr & local_map,
  const double max_correspondence_m,
  RegistrationResult & result) const
{
  if (aligned_scan.empty() || !local_map || local_map->empty() ||
    max_correspondence_m <= 0.0)
  {
    return;
  }

  pcl::KdTreeFLANN<PointT> tree;
  tree.setInputCloud(local_map);

  const double max_distance_sq = max_correspondence_m * max_correspondence_m;
  double squared_error_sum = 0.0;
  int valid_points = 0;
  int inliers = 0;
  std::vector<int> indices(1);
  std::vector<float> squared_distances(1);

  for (const auto & point : aligned_scan.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    ++valid_points;
    if (tree.nearestKSearch(point, 1, indices, squared_distances) <= 0) {
      continue;
    }
    const double squared_distance = squared_distances.front();
    if (squared_distance > max_distance_sq) {
      continue;
    }
    ++inliers;
    squared_error_sum += squared_distance;
  }

  result.inlier_count = inliers;
  if (valid_points > 0) {
    result.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(valid_points);
  }
  if (inliers > 0) {
    result.inlier_rmse_m = std::sqrt(squared_error_sum / static_cast<double>(inliers));
  }
}

bool ScanToMapIcp::registrationCorrectionWithinLimits(
  RegistrationResult & result) const
{
  if (!result.converged) {
    result.rejection_reason = "not_converged";
    return false;
  }
  if (!result.map_T_base.matrix().allFinite() ||
    std::abs(result.transform_determinant - 1.0) > 0.05 ||
    result.transform_orthogonality_error > 0.05)
  {
    result.rejection_reason = "invalid_transform";
    return false;
  }
  if (!std::isfinite(result.fitness) || result.fitness > fitness_score_threshold_) {
    result.rejection_reason = "fitness";
    return false;
  }
  if (result.inlier_count <= 0 || result.inlier_ratio < min_registration_inlier_ratio_) {
    result.rejection_reason = "inlier_ratio";
    return false;
  }
  const double max_rmse = std::max(0.0, max_registration_inlier_rmse_m_);
  if (max_rmse > 0.0 && result.inlier_rmse_m > max_rmse) {
    result.rejection_reason = "inlier_rmse";
    return false;
  }
  if (!std::isfinite(result.correction_translation_m) ||
    !std::isfinite(result.correction_rotation_deg))
  {
    result.rejection_reason = "invalid_correction";
    return false;
  }

  const double max_translation = std::max(0.0, max_registration_correction_translation_m_);
  if (max_translation > 0.0 && result.correction_translation_m > max_translation) {
    result.rejection_reason = "translation_correction";
    return false;
  }

  const double max_rotation = std::max(0.0, max_registration_correction_rotation_deg_);
  if (max_rotation > 0.0 && result.correction_rotation_deg > max_rotation) {
    result.rejection_reason = "rotation_correction";
    return false;
  }

  result.rejection_reason = "accepted";
  return true;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildLocalMap(
  const Eigen::Isometry3d & center_pose) const
{
  auto local_map = std::make_shared<CloudT>();
  if (keyframes_.empty()) {
    return local_map;
  }

  int selected = 0;
  for (auto it = keyframes_.rbegin(); it != keyframes_.rend(); ++it) {
    if (selected >= max_local_keyframes_) {
      break;
    }
    if (transformDistance2d(it->map_T_base, center_pose) > local_map_radius_m_) {
      continue;
    }
    *local_map += *(it->cloud_map);
    ++selected;
  }

  if (local_map->empty()) {
    const auto & latest = keyframes_.back();
    *local_map += *(latest.cloud_map);
  }

  return downsampleCloud(local_map, map_voxel_leaf_m_);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildCandidateMapAroundKeyframe(
  const int keyframe_index) const
{
  auto candidate_map = std::make_shared<CloudT>();
  if (keyframes_.empty() || keyframe_index < 0 ||
    keyframe_index >= static_cast<int>(keyframes_.size()))
  {
    return candidate_map;
  }

  const int neighbor_count = std::max(0, place_recognition_candidate_map_neighbors_);
  const int first = std::max(0, keyframe_index - neighbor_count);
  const int last = std::min(
    static_cast<int>(keyframes_.size()) - 1,
    keyframe_index + neighbor_count);
  for (int index = first; index <= last; ++index) {
    *candidate_map += *(keyframes_[index].cloud_map);
  }

  return downsampleCloud(candidate_map, map_voxel_leaf_m_);
}

void ScanToMapIcp::initializeFromFirstScan(
  const CloudConstPtr & scan_base,
  const rclcpp::Time & stamp)
{
  initialized_ = true;
  map_T_base_ = Eigen::Isometry3d::Identity();
  last_keyframe_pose_ = map_T_base_;
  last_registration_.converged = true;
  last_registration_.accepted = true;
  last_registration_.fitness = 0.0;
  last_registration_.iterations = 0;
  last_registration_.map_T_base = map_T_base_;
  last_registration_.rejection_reason = "initialized";
  recordAcceptedPose();
  recordTrackingScan(scan_base, map_T_base_, stamp);
  if (use_wheel_prediction_) {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    if (have_latest_wheel_odom_) {
      last_used_wheel_odom_ = latest_wheel_odom_;
      have_last_used_wheel_odom_ = true;
      have_pending_wheel_odom_ = false;
    }
  }
  addKeyframe(scan_base, stamp);
  RCLCPP_INFO(get_logger(), "Initialized scan-to-map SLAM from first scan");
}

void ScanToMapIcp::addKeyframe(const CloudConstPtr & scan_base, const rclcpp::Time & stamp)
{
  const int new_keyframe_index = static_cast<int>(keyframes_.size());
  if (new_keyframe_index > 0) {
    const auto & previous = keyframes_.back();
    PoseGraphEdge odom_edge;
    odom_edge.from_index = new_keyframe_index - 1;
    odom_edge.to_index = new_keyframe_index;
    odom_edge.from_T_to = previous.map_T_base.inverse() * map_T_base_;
    odom_edge.translation_weight = std::max(0.0, pose_graph_odom_translation_weight_);
    odom_edge.rotation_weight = std::max(0.0, pose_graph_odom_rotation_weight_);
    odom_edge.loop_closure = false;
    pose_graph_edges_.push_back(odom_edge);
  }

  Keyframe keyframe;
  keyframe.stamp = stamp;
  keyframe.map_T_base = map_T_base_;
  keyframe.cloud_base = downsampleCloud(scan_base, map_voxel_leaf_m_);
  keyframe.cloud_map = filterMapCloud(transformCloud(keyframe.cloud_base, map_T_base_));
  keyframe.descriptor = computeScanDescriptor(scan_base);
  keyframes_.push_back(std::move(keyframe));
  last_keyframe_pose_ = map_T_base_;

  bool pruned_oldest_keyframe = false;
  if (static_cast<int>(keyframes_.size()) > max_keyframes_) {
    keyframes_.erase(keyframes_.begin());
    prunePoseGraphEdgesAfterOldestKeyframeRemoval();
    pruned_oldest_keyframe = true;
  }

  if (pruned_oldest_keyframe) {
    rebuildGlobalMap();
    return;
  }

  const auto & latest = keyframes_.back();
  if (latest.cloud_map && !latest.cloud_map->empty()) {
    if (!global_map_) {
      global_map_ = std::make_shared<CloudT>();
    }
    *global_map_ += *(latest.cloud_map);
    global_map_ = downsampleCloud(global_map_, map_voxel_leaf_m_);
  }
  latest_local_map_ = buildLocalMap(map_T_base_);
}

bool ScanToMapIcp::shouldAddKeyframe(const Eigen::Isometry3d & pose) const
{
  const double translation = transformDistance2d(pose, last_keyframe_pose_);
  const double yaw_delta = std::abs(
    normalizedAngle(
      yawFromIsometry(pose) -
      yawFromIsometry(last_keyframe_pose_)));
  const double rotation_threshold = keyframe_rotation_deg_ * M_PI / 180.0;
  return translation >= keyframe_translation_m_ || yaw_delta >= rotation_threshold;
}

std::string ScanToMapIcp::shockQuarantineReason(const RegistrationResult & result) const
{
  if (!quarantine_keyframes_on_shock_) {
    return "";
  }

  if (result.place_recognition_accepted &&
    result.place_recognition_pose_metrics_evaluated)
  {
    const PlaceRecognitionPoseMetrics place_pose_metrics{
      result.place_recognition_pose_metrics_valid,
      result.place_recognition_prediction_jump_translation_m,
      result.place_recognition_prediction_jump_rotation_deg,
      result.place_recognition_map_correction_translation_m,
      result.place_recognition_map_correction_rotation_deg};
    if (!place_pose_metrics.finite) {
      return "place_pose_metrics_invalid";
    }

    const double max_alignment_translation = std::max(
      0.0, keyframe_quarantine_correction_translation_m_);
    if (max_alignment_translation > 0.0 &&
      result.correction_translation_m > max_alignment_translation)
    {
      return "place_alignment_correction_translation";
    }
    const double max_alignment_rotation = std::max(
      0.0, keyframe_quarantine_correction_rotation_deg_);
    if (max_alignment_rotation > 0.0 &&
      result.correction_rotation_deg > max_alignment_rotation)
    {
      return "place_alignment_correction_rotation";
    }

    const double max_prediction_translation = std::max(
      0.0, place_recognition_max_pose_jump_m_);
    const double max_prediction_rotation = std::max(
      0.0, place_recognition_max_pose_jump_rotation_deg_);
    if (!placePredictionJumpWithinLimits(
        place_pose_metrics,
        max_prediction_translation,
        max_prediction_rotation))
    {
      if (max_prediction_translation > 0.0 &&
        place_pose_metrics.prediction_jump_translation_m > max_prediction_translation)
      {
        return "place_prediction_jump_translation";
      }
      return "place_prediction_jump_rotation";
    }
    return "";
  }

  const double max_translation = std::max(0.0, keyframe_quarantine_correction_translation_m_);
  if (max_translation > 0.0 && result.correction_translation_m > max_translation) {
    return "correction_translation";
  }

  const double max_rotation = std::max(0.0, keyframe_quarantine_correction_rotation_deg_);
  if (max_rotation > 0.0 && result.correction_rotation_deg > max_rotation) {
    return "correction_rotation";
  }

  return "";
}

std::vector<float> ScanToMapIcp::computeScanDescriptor(const CloudConstPtr & scan_base) const
{
  const int sectors = std::max(8, place_recognition_descriptor_bins_);
  const int rings = std::max(1, place_recognition_descriptor_rings_);
  std::vector<float> descriptor(static_cast<std::size_t>(rings * sectors), 0.0F);
  if (!scan_base || scan_base->empty()) {
    return descriptor;
  }

  const double normalization_range = std::max(1.0, max_range_m_);
  const double z_min = std::min(z_min_m_, z_max_m_);
  const double z_max = std::max(z_min_m_, z_max_m_);
  const double z_span = std::max(1.0e-3, z_max - z_min);
  for (const auto & point : scan_base->points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const double range_xy = std::hypot(point.x, point.y);
    if (range_xy <= 1.0e-3) {
      continue;
    }
    double angle = std::atan2(point.y, point.x);
    if (angle < 0.0) {
      angle += 2.0 * M_PI;
    }
    const int sector = std::min(
      sectors - 1,
      static_cast<int>(std::floor(angle / (2.0 * M_PI) * static_cast<double>(sectors))));
    int ring = 0;
    float value = static_cast<float>(std::min(range_xy, normalization_range) / normalization_range);
    if (rings > 1) {
      ring = std::min(
        rings - 1,
        static_cast<int>(
          std::floor(
            std::min(range_xy, normalization_range) / normalization_range *
            static_cast<double>(rings))));
      const double normalized_z = std::clamp(
        (static_cast<double>(point.z) - z_min) / z_span,
        0.0,
        1.0);
      value = static_cast<float>(0.05 + 0.95 * normalized_z);
    }
    const auto index = static_cast<std::size_t>(ring * sectors + sector);
    descriptor[index] = std::max(descriptor[index], value);
  }

  return descriptor;
}

double ScanToMapIcp::descriptorDistance(
  const std::vector<float> & lhs,
  const std::vector<float> & rhs,
  int & best_shift) const
{
  best_shift = 0;
  if (lhs.empty() || rhs.empty() || lhs.size() != rhs.size()) {
    return std::numeric_limits<double>::infinity();
  }

  const int sectors = std::max(8, place_recognition_descriptor_bins_);
  if (lhs.size() % static_cast<std::size_t>(sectors) != 0U) {
    return std::numeric_limits<double>::infinity();
  }

  const int rings = static_cast<int>(lhs.size() / static_cast<std::size_t>(sectors));
  const bool sparse_ring_sector_descriptor = rings > 1;
  double best_distance = std::numeric_limits<double>::infinity();
  for (int shift = 0; shift < sectors; ++shift) {
    double squared_error_sum = 0.0;
    int compared_bins = 0;
    for (int ring = 0; ring < rings; ++ring) {
      for (int sector = 0; sector < sectors; ++sector) {
        const int shifted_sector = (sector + shift) % sectors;
        const auto lhs_index = static_cast<std::size_t>(ring * sectors + shifted_sector);
        const auto rhs_index = static_cast<std::size_t>(ring * sectors + sector);
        if (sparse_ring_sector_descriptor &&
          lhs[lhs_index] <= 0.0F && rhs[rhs_index] <= 0.0F)
        {
          continue;
        }
        const double error = static_cast<double>(lhs[lhs_index]) -
          static_cast<double>(rhs[rhs_index]);
        squared_error_sum += error * error;
        ++compared_bins;
      }
    }
    if (compared_bins == 0) {
      continue;
    }
    const double distance = std::sqrt(squared_error_sum / static_cast<double>(compared_bins));
    if (distance < best_distance) {
      best_distance = distance;
      best_shift = shift;
    }
  }

  return best_distance;
}

bool ScanToMapIcp::tryPlaceRecognition(
  const CloudConstPtr & scan_base,
  const std::vector<float> & scan_descriptor,
  const Eigen::Isometry3d & reference_pose,
  RegistrationResult & result) const
{
  result.place_recognition_attempted = true;
  result.place_recognition_candidate_index = -1;
  result.place_recognition_descriptor_distance = std::numeric_limits<double>::infinity();
  result.place_recognition_yaw_offset_deg = 0.0;

  if (!place_recognition_enabled_ || scan_descriptor.empty() ||
    static_cast<int>(keyframes_.size()) < place_recognition_min_keyframes_)
  {
    return false;
  }

  struct Candidate
  {
    int index{-1};
    int shift{0};
    double distance{std::numeric_limits<double>::infinity()};
  };

  std::vector<Candidate> candidates;
  const int latest_candidate_exclusive = std::max(
    0,
    static_cast<int>(keyframes_.size()) -
    std::max(0, place_recognition_exclude_recent_keyframes_));
  for (int index = 0; index < latest_candidate_exclusive; ++index) {
    int shift = 0;
    const double distance = descriptorDistance(
      scan_descriptor,
      keyframes_[static_cast<std::size_t>(index)].descriptor,
      shift);
    if (!std::isfinite(distance) || distance > place_recognition_max_descriptor_distance_) {
      continue;
    }
    candidates.push_back(Candidate{index, shift, distance});
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.distance < rhs.distance;
    });

  const int max_candidates = std::max(1, place_recognition_max_candidates_);
  if (static_cast<int>(candidates.size()) > max_candidates) {
    candidates.resize(static_cast<std::size_t>(max_candidates));
  }
  result.place_recognition_candidates = static_cast<int>(candidates.size());
  if (candidates.empty()) {
    return false;
  }

  RegistrationResult best_result;
  double best_score = std::numeric_limits<double>::infinity();
  bool have_match = false;

  const double yaw_bin_width = 2.0 * M_PI /
    static_cast<double>(std::max(8, place_recognition_descriptor_bins_));
  for (const auto & candidate : candidates) {
    const auto candidate_map = buildCandidateMapAroundKeyframe(candidate.index);
    if (!candidate_map || static_cast<int>(candidate_map->size()) < min_scan_points_) {
      continue;
    }

    Eigen::Isometry3d candidate_pose =
      keyframes_[static_cast<std::size_t>(candidate.index)].map_T_base;
    const double yaw_offset =
      normalizedAngle(-static_cast<double>(candidate.shift) * yaw_bin_width);
    const double max_yaw_offset = std::max(0.0, place_recognition_max_yaw_offset_deg_);
    if (max_yaw_offset < 180.0 && std::abs(yaw_offset) * 180.0 / M_PI > max_yaw_offset) {
      continue;
    }
    candidate_pose.linear() =
      candidate_pose.linear() *
      Eigen::AngleAxisd(yaw_offset, Eigen::Vector3d::UnitZ()).toRotationMatrix();

    auto candidate_result = registerScanToMap(scan_base, candidate_map, candidate_pose);
    if (!candidate_result.accepted) {
      continue;
    }
    if (candidate_result.inlier_ratio < place_recognition_min_inlier_ratio_) {
      continue;
    }
    const double max_rmse = std::max(0.0, place_recognition_max_inlier_rmse_m_);
    if (max_rmse > 0.0 && candidate_result.inlier_rmse_m > max_rmse) {
      continue;
    }
    const Eigen::Isometry3d pose_jump = reference_pose.inverse() * candidate_result.map_T_base;
    const double max_jump = std::max(0.0, place_recognition_max_pose_jump_m_);
    if (max_jump > 0.0 && pose_jump.translation().norm() > max_jump) {
      continue;
    }
    const double max_jump_rotation = std::max(
      0.0,
      place_recognition_max_pose_jump_rotation_deg_);
    if (max_jump_rotation > 0.0 &&
      Eigen::AngleAxisd(pose_jump.linear()).angle() * 180.0 / M_PI > max_jump_rotation)
    {
      continue;
    }

    const double score = candidate_result.fitness +
      (1.0 - candidate_result.inlier_ratio) +
      0.1 * candidate.distance;
    if (score < best_score) {
      best_score = score;
      best_result = candidate_result;
      best_result.place_recognition_attempted = true;
      best_result.place_recognition_accepted = true;
      best_result.place_recognition_candidates = result.place_recognition_candidates;
      best_result.place_recognition_candidate_index = candidate.index;
      best_result.place_recognition_descriptor_distance = candidate.distance;
      best_result.place_recognition_yaw_offset_deg = yaw_offset * 180.0 / M_PI;
      best_result.rejection_reason = "place_recognition";
      have_match = true;
    }
  }

  if (!have_match) {
    const auto & best_candidate = candidates.front();
    result.place_recognition_candidate_index = best_candidate.index;
    result.place_recognition_descriptor_distance = best_candidate.distance;
    result.place_recognition_yaw_offset_deg =
      normalizedAngle(-static_cast<double>(best_candidate.shift) * yaw_bin_width) * 180.0 / M_PI;
    return false;
  }

  result = best_result;
  return true;
}

void ScanToMapIcp::refreshKeyframeCloudMap(const std::size_t keyframe_index)
{
  if (keyframe_index >= keyframes_.size()) {
    return;
  }
  auto & keyframe = keyframes_[keyframe_index];
  if (!keyframe.cloud_base || keyframe.cloud_base->empty()) {
    return;
  }
  keyframe.cloud_map = filterMapCloud(transformCloud(keyframe.cloud_base, keyframe.map_T_base));
}

void ScanToMapIcp::prunePoseGraphEdgesAfterOldestKeyframeRemoval()
{
  std::vector<PoseGraphEdge> retained_edges;
  retained_edges.reserve(pose_graph_edges_.size());
  for (auto edge : pose_graph_edges_) {
    if (edge.from_index <= 0 || edge.to_index <= 0) {
      continue;
    }
    --edge.from_index;
    --edge.to_index;
    retained_edges.push_back(edge);
  }
  pose_graph_edges_ = std::move(retained_edges);
}

double ScanToMapIcp::poseGraphEdgeSquaredError(
  const PoseGraphEdge & edge,
  const std::vector<Eigen::Isometry3d> & poses) const
{
  if (edge.from_index < 0 || edge.to_index < 0 ||
    edge.from_index >= static_cast<int>(poses.size()) ||
    edge.to_index >= static_cast<int>(poses.size()))
  {
    return std::numeric_limits<double>::infinity();
  }

  const auto & from = poses[static_cast<std::size_t>(edge.from_index)];
  const auto & to = poses[static_cast<std::size_t>(edge.to_index)];
  const Eigen::Isometry3d expected_to = from * edge.from_T_to;
  const Eigen::Isometry3d error = expected_to.inverse() * to;
  const double translation_weight = std::max(0.0, edge.translation_weight);
  const double rotation_weight = std::max(0.0, edge.rotation_weight);
  const double rotation_rad = rotationAngleDeg(error.linear()) * M_PI / 180.0;
  return translation_weight * error.translation().squaredNorm() +
         rotation_weight * rotation_rad * rotation_rad;
}

double ScanToMapIcp::poseGraphLoopRobustWeight(
  const PoseGraphEdge & edge,
  const double squared_error) const
{
  if (!pose_graph_robust_loop_enabled_ || !edge.loop_closure) {
    return 1.0;
  }
  if (!std::isfinite(squared_error)) {
    return 0.0;
  }

  const double scale = std::max(1.0e-6, pose_graph_loop_cauchy_scale_);
  const double scale_squared = scale * scale;
  return scale_squared / (scale_squared + std::max(0.0, squared_error));
}

double ScanToMapIcp::poseGraphResidual(
  const std::vector<Eigen::Isometry3d> & poses,
  const bool apply_robust_loop_kernel) const
{
  double residual = 0.0;
  const double scale = std::max(1.0e-6, pose_graph_loop_cauchy_scale_);
  const double scale_squared = scale * scale;
  for (const auto & edge : pose_graph_edges_) {
    const double squared_error = poseGraphEdgeSquaredError(edge, poses);
    if (!std::isfinite(squared_error)) {
      continue;
    }

    if (apply_robust_loop_kernel && pose_graph_robust_loop_enabled_ && edge.loop_closure) {
      residual += scale_squared * std::log1p(squared_error / scale_squared);
    } else {
      residual += squared_error;
    }
  }
  return residual;
}

ScanToMapIcp::PoseGraphStats ScanToMapIcp::optimizePoseGraphWithLoop(
  const int anchor_index,
  const Eigen::Isometry3d & target_loop_pose,
  const std::vector<Eigen::Isometry3d> & previous_keyframe_poses)
{
  PoseGraphStats stats;
  stats.nodes = static_cast<int>(keyframes_.size());
  if (stats.nodes < 2 || previous_keyframe_poses.size() != keyframes_.size()) {
    return stats;
  }

  const int last_index = stats.nodes - 1;
  if (anchor_index < 0 || anchor_index >= last_index) {
    return stats;
  }

  PoseGraphEdge loop_edge;
  loop_edge.from_index = anchor_index;
  loop_edge.to_index = last_index;
  loop_edge.from_T_to =
    previous_keyframe_poses[static_cast<std::size_t>(anchor_index)].inverse() *
    target_loop_pose;
  loop_edge.translation_weight = std::max(0.0, pose_graph_loop_translation_weight_);
  loop_edge.rotation_weight = std::max(0.0, pose_graph_loop_rotation_weight_);
  loop_edge.loop_closure = true;
  pose_graph_edges_.push_back(loop_edge);

  std::vector<Eigen::Isometry3d> poses = previous_keyframe_poses;
  std::vector<Eigen::Isometry3d> best_poses = poses;
  auto refresh_retained_loop_diagnostics = [&]() {
      stats.edges = static_cast<int>(pose_graph_edges_.size());
      stats.loop_edges = 0;
      stats.robust_loop_edges_downweighted = 0;
      stats.min_loop_robust_weight = 1.0;
      for (const auto & edge : pose_graph_edges_) {
        if (!edge.loop_closure) {
          continue;
        }
        ++stats.loop_edges;
        const double robust_weight = poseGraphLoopRobustWeight(
          edge,
          poseGraphEdgeSquaredError(edge, previous_keyframe_poses));
        stats.min_loop_robust_weight = std::min(
          stats.min_loop_robust_weight,
          robust_weight);
        if (robust_weight < 0.999) {
          ++stats.robust_loop_edges_downweighted;
        }
      }
    };
  auto reject_new_loop = [&](const std::string & reason) {
      stats.new_loop_rejected = true;
      stats.rejection_reason = reason;
      pose_graph_edges_.pop_back();
      refresh_retained_loop_diagnostics();
    };
  refresh_retained_loop_diagnostics();

  stats.new_loop_robust_weight = poseGraphLoopRobustWeight(
    loop_edge,
    poseGraphEdgeSquaredError(loop_edge, poses));
  const double min_new_loop_robust_weight =
    std::clamp(pose_graph_min_new_loop_robust_weight_, 0.0, 1.0);
  if (pose_graph_robust_loop_enabled_ &&
    stats.new_loop_robust_weight < min_new_loop_robust_weight)
  {
    stats.residual_before = poseGraphResidual(poses, false);
    stats.robust_objective_before = poseGraphResidual(poses, true);
    reject_new_loop("new_loop_robust_weight");
    stats.residual_after = poseGraphResidual(poses, false);
    stats.robust_objective_after = poseGraphResidual(poses, true);
    return stats;
  }
  stats.residual_before = poseGraphResidual(poses, false);
  stats.robust_objective_before = poseGraphResidual(poses, true);
  double best_objective = stats.robust_objective_before;

  const int max_iterations = std::max(1, pose_graph_optimization_iterations_);
  const double base_step = std::clamp(pose_graph_optimization_step_, 0.0, 1.0);
  const double max_translation_step =
    std::max(0.0, pose_graph_max_iteration_translation_step_m_);
  const double max_rotation_step =
    std::max(0.0, pose_graph_max_iteration_rotation_step_deg_);

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    double iteration_max_translation = 0.0;
    double iteration_max_rotation = 0.0;

    for (const auto & edge : pose_graph_edges_) {
      if (edge.from_index < 0 || edge.to_index < 0 ||
        edge.from_index >= static_cast<int>(poses.size()) ||
        edge.to_index >= static_cast<int>(poses.size()))
      {
        continue;
      }

      const int from_index = edge.from_index;
      const int to_index = edge.to_index;
      const Eigen::Isometry3d expected_to =
        poses[static_cast<std::size_t>(from_index)] * edge.from_T_to;
      Eigen::Isometry3d correction =
        normalizeIsometry(expected_to * poses[static_cast<std::size_t>(to_index)].inverse());
      if (pose_graph_planar_correction_enabled_) {
        correction = planarizeCorrection(correction);
      }

      const double to_share = from_index == 0 ? 1.0 : 0.5;
      const double from_share = to_index == 0 ? 1.0 : 0.5;
      const double robust_weight = poseGraphLoopRobustWeight(
        edge,
        poseGraphEdgeSquaredError(edge, poses));
      const double translation_step =
        std::clamp(
        base_step * std::max(0.0, edge.translation_weight) * robust_weight,
        0.0,
        1.0);
      const double rotation_step =
        std::clamp(
        base_step * std::max(0.0, edge.rotation_weight) * robust_weight,
        0.0,
        1.0);

      if (to_index > 0) {
        Eigen::Isometry3d to_correction = interpolateTransform(
          correction,
          translation_step * to_share,
          rotation_step * to_share);
        to_correction = limitCorrectionStep(
          to_correction,
          max_translation_step,
          max_rotation_step);
        iteration_max_translation = std::max(
          iteration_max_translation,
          to_correction.translation().norm());
        iteration_max_rotation = std::max(
          iteration_max_rotation,
          rotationAngleDeg(to_correction.linear()));
        poses[static_cast<std::size_t>(to_index)] =
          normalizeIsometry(to_correction * poses[static_cast<std::size_t>(to_index)]);
      }

      if (from_index > 0) {
        Eigen::Isometry3d from_correction = interpolateTransform(
          correction.inverse(),
          translation_step * from_share,
          rotation_step * from_share);
        from_correction = limitCorrectionStep(
          from_correction,
          max_translation_step,
          max_rotation_step);
        iteration_max_translation = std::max(
          iteration_max_translation,
          from_correction.translation().norm());
        iteration_max_rotation = std::max(
          iteration_max_rotation,
          rotationAngleDeg(from_correction.linear()));
        poses[static_cast<std::size_t>(from_index)] =
          normalizeIsometry(from_correction * poses[static_cast<std::size_t>(from_index)]);
      }
    }

    const double objective = poseGraphResidual(poses, true);
    if (objective < best_objective) {
      best_objective = objective;
      best_poses = poses;
    }
    stats.max_update_translation_m = std::max(
      stats.max_update_translation_m,
      iteration_max_translation);
    stats.max_update_rotation_deg = std::max(
      stats.max_update_rotation_deg,
      iteration_max_rotation);
    stats.iterations = iteration + 1;

    if (iteration_max_translation < 1.0e-4 && iteration_max_rotation < 1.0e-3) {
      break;
    }
  }

  stats.residual_after = poseGraphResidual(best_poses, false);
  stats.robust_objective_after = best_objective;
  if (!std::isfinite(stats.residual_before) || !std::isfinite(stats.residual_after) ||
    !std::isfinite(stats.robust_objective_before) ||
    !std::isfinite(stats.robust_objective_after) ||
    stats.robust_objective_after >= stats.robust_objective_before)
  {
    reject_new_loop("objective_not_improved");
    return stats;
  }
  const double max_residual_after = std::max(0.0, pose_graph_max_residual_after_);
  if (max_residual_after > 0.0 && stats.residual_after > max_residual_after) {
    reject_new_loop("residual_after");
    return stats;
  }
  const double max_residual_ratio = std::max(0.0, pose_graph_max_residual_ratio_);
  if (max_residual_ratio > 0.0 && stats.residual_before > 1.0e-6 &&
    stats.residual_after / stats.residual_before > max_residual_ratio)
  {
    reject_new_loop("residual_ratio");
    return stats;
  }

  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    keyframes_[index].map_T_base = normalizeIsometry(best_poses[index]);
    refreshKeyframeCloudMap(index);
  }

  stats.applied = true;
  return stats;
}

void ScanToMapIcp::updatePathFromKeyframeCorrections(
  const std::vector<Eigen::Isometry3d> & previous_keyframe_poses)
{
  if (previous_keyframe_poses.size() != keyframes_.size() || keyframes_.empty()) {
    return;
  }

  std::vector<Eigen::Isometry3d> corrections;
  corrections.reserve(keyframes_.size());
  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    corrections.push_back(
      normalizeIsometry(keyframes_[index].map_T_base * previous_keyframe_poses[index].inverse()));
  }

  for (auto & pose_stamped : path_msg_.poses) {
    const double pose_stamp_s = rclcpp::Time(pose_stamped.header.stamp).seconds();
    std::size_t right_index = 0;
    while (right_index < keyframes_.size() &&
      keyframes_[right_index].stamp.seconds() < pose_stamp_s)
    {
      ++right_index;
    }

    Eigen::Isometry3d correction = corrections.back();
    if (right_index == 0) {
      correction = corrections.front();
    } else if (right_index < keyframes_.size()) {
      const std::size_t left_index = right_index - 1;
      const double left_stamp_s = keyframes_[left_index].stamp.seconds();
      const double right_stamp_s = keyframes_[right_index].stamp.seconds();
      double ratio = 0.0;
      if (right_stamp_s - left_stamp_s > 1.0e-6) {
        ratio = std::clamp(
          (pose_stamp_s - left_stamp_s) / (right_stamp_s - left_stamp_s),
          0.0,
          1.0);
      }
      const Eigen::Isometry3d relative_correction =
        normalizeIsometry(corrections[right_index] * corrections[left_index].inverse());
      correction = normalizeIsometry(
        interpolateTransform(relative_correction, ratio, ratio) * corrections[left_index]);
    }

    pose_stamped.pose = isometryToPose(correction * poseMsgToIsometry(pose_stamped.pose));
  }
}

void ScanToMapIcp::applyPoseGraphCorrection(
  const rclcpp::Time & stamp,
  const Eigen::Isometry3d & reference_pose,
  const std::optional<GravitySample> & gravity_sample,
  RegistrationResult & result)
{
  result.pose_graph_correction_applied = false;
  result.pose_graph_new_loop_rejected = false;
  result.pose_graph_rejection_reason.clear();
  result.pose_graph_correction_keyframes = 0;
  result.pose_graph_correction_translation_m = 0.0;
  result.pose_graph_correction_rotation_deg = 0.0;
  result.pose_graph_optimization_iterations = 0;
  result.pose_graph_edges = static_cast<int>(pose_graph_edges_.size());
  result.pose_graph_loop_edges = 0;
  result.pose_graph_robust_loop_edges_downweighted = 0;
  for (const auto & edge : pose_graph_edges_) {
    if (edge.loop_closure) {
      ++result.pose_graph_loop_edges;
    }
  }
  result.pose_graph_residual_before = 0.0;
  result.pose_graph_residual_after = 0.0;
  result.pose_graph_robust_objective_before = 0.0;
  result.pose_graph_robust_objective_after = 0.0;
  result.pose_graph_min_loop_robust_weight = 1.0;
  result.pose_graph_new_loop_robust_weight = 1.0;
  result.pose_graph_max_update_translation_m = 0.0;
  result.pose_graph_max_update_rotation_deg = 0.0;

  if (!pose_graph_correction_enabled_ || !result.place_recognition_accepted) {
    result.pose_graph_rejection_reason = "disabled_or_no_place_match";
    return;
  }
  const double min_interval_s = std::max(0.0, pose_graph_correction_min_interval_s_);
  if (have_last_pose_graph_correction_stamp_ &&
    min_interval_s > 0.0 &&
    (stamp - last_pose_graph_correction_stamp_).seconds() < min_interval_s)
  {
    result.pose_graph_rejection_reason = "minimum_interval";
    return;
  }
  const int anchor_index = result.place_recognition_candidate_index;
  const int keyframe_count = static_cast<int>(keyframes_.size());
  if (anchor_index < 0 || anchor_index >= keyframe_count ||
    keyframe_count < std::max(2, pose_graph_correction_min_keyframes_))
  {
    result.pose_graph_rejection_reason = "invalid_anchor_or_keyframe_count";
    return;
  }

  const Eigen::Isometry3d raw_correction =
    normalizeIsometry(result.map_T_base * reference_pose.inverse());
  const Eigen::Isometry3d correction = pose_graph_planar_correction_enabled_ ?
    planarizeCorrection(raw_correction) : raw_correction;
  const Eigen::Vector3d correction_translation = correction.translation();
  Eigen::Quaterniond correction_rotation(correction.linear());
  correction_rotation.normalize();

  const double correction_translation_m = correction_translation.norm();
  const double correction_rotation_deg =
    Eigen::AngleAxisd(correction_rotation).angle() * 180.0 / M_PI;
  result.pose_graph_correction_translation_m = correction_translation_m;
  result.pose_graph_correction_rotation_deg = correction_rotation_deg;

  const double max_translation = std::max(0.0, pose_graph_correction_max_translation_m_);
  if (max_translation > 0.0 && correction_translation_m > max_translation) {
    result.pose_graph_rejection_reason = "correction_translation";
    return;
  }
  const double max_rotation = std::max(0.0, pose_graph_correction_max_rotation_deg_);
  if (max_rotation > 0.0 && correction_rotation_deg > max_rotation) {
    result.pose_graph_rejection_reason = "correction_rotation";
    return;
  }

  const int last_index = keyframe_count - 1;
  if (anchor_index >= last_index) {
    result.pose_graph_rejection_reason = "anchor_not_historical";
    return;
  }

  const Eigen::Isometry3d target_loop_pose =
    normalizeIsometry(correction * keyframes_.back().map_T_base);
  const Eigen::Isometry3d prospective_loop_error =
    normalizeIsometry(target_loop_pose.inverse() * keyframes_.back().map_T_base);
  const double prospective_loop_rotation_rad =
    rotationAngleDeg(prospective_loop_error.linear()) * M_PI / 180.0;
  const double prospective_loop_residual =
    std::max(0.0, pose_graph_loop_translation_weight_) *
    prospective_loop_error.translation().squaredNorm() +
    std::max(0.0, pose_graph_loop_rotation_weight_) *
    prospective_loop_rotation_rad * prospective_loop_rotation_rad;
  result.pose_graph_residual_before = prospective_loop_residual;
  const double max_loop_residual_before = std::max(0.0, pose_graph_max_loop_residual_before_);
  if (max_loop_residual_before > 0.0 &&
    prospective_loop_residual > max_loop_residual_before)
  {
    result.pose_graph_rejection_reason = "loop_residual_before";
    return;
  }

  std::vector<Eigen::Isometry3d> previous_keyframe_poses;
  previous_keyframe_poses.reserve(keyframes_.size());
  for (const auto & keyframe : keyframes_) {
    previous_keyframe_poses.push_back(keyframe.map_T_base);
  }

  const PoseGraphStats stats = optimizePoseGraphWithLoop(
    anchor_index,
    target_loop_pose,
    previous_keyframe_poses);
  result.pose_graph_optimization_iterations = stats.iterations;
  result.pose_graph_edges = stats.edges;
  result.pose_graph_loop_edges = stats.loop_edges;
  result.pose_graph_robust_loop_edges_downweighted = stats.robust_loop_edges_downweighted;
  result.pose_graph_residual_before = stats.residual_before;
  result.pose_graph_residual_after = stats.residual_after;
  result.pose_graph_robust_objective_before = stats.robust_objective_before;
  result.pose_graph_robust_objective_after = stats.robust_objective_after;
  result.pose_graph_min_loop_robust_weight = stats.min_loop_robust_weight;
  result.pose_graph_new_loop_robust_weight = stats.new_loop_robust_weight;
  result.pose_graph_max_update_translation_m = stats.max_update_translation_m;
  result.pose_graph_max_update_rotation_deg = stats.max_update_rotation_deg;
  result.pose_graph_new_loop_rejected = stats.new_loop_rejected;
  result.pose_graph_rejection_reason = stats.rejection_reason;

  if (!stats.applied) {
    return;
  }

  updatePathFromKeyframeCorrections(previous_keyframe_poses);
  const Eigen::Isometry3d latest_keyframe_correction =
    normalizeIsometry(keyframes_.back().map_T_base * previous_keyframe_poses.back().inverse());
  map_T_base_ = constrainPlanarPose(
    normalizeIsometry(latest_keyframe_correction * reference_pose),
    reference_pose,
    gravity_sample);
  result.map_T_base = map_T_base_;
  last_keyframe_pose_ = keyframes_.back().map_T_base;
  rebuildGlobalMap();
  last_pose_graph_correction_stamp_ = stamp;
  have_last_pose_graph_correction_stamp_ = true;

  if (have_last_accepted_pose_) {
    previous_accepted_pose_ = result.map_T_base;
    last_accepted_pose_ = result.map_T_base;
    have_previous_accepted_pose_ = true;
  }

  result.pose_graph_correction_applied = true;
  result.pose_graph_correction_keyframes = stats.nodes - anchor_index - 1;
}

void ScanToMapIcp::rebuildGlobalMap()
{
  auto merged = std::make_shared<CloudT>();
  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    if (!keyframes_[index].cloud_map || keyframes_[index].cloud_map->empty()) {
      refreshKeyframeCloudMap(index);
    }
    const auto & keyframe = keyframes_[index];
    if (!keyframe.cloud_map || keyframe.cloud_map->empty()) {
      continue;
    }
    *merged += *(keyframe.cloud_map);
  }

  global_map_ = downsampleCloud(merged, map_voxel_leaf_m_);
  latest_local_map_ = buildLocalMap(map_T_base_);
}

void ScanToMapIcp::publishState(const rclcpp::Time & stamp)
{
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose = isometryToPose(map_T_base_);

  const double covariance = last_registration_.converged ?
    std::max(last_registration_.fitness, 0.01) : 100.0;
  odom.pose.covariance[0] = covariance;
  odom.pose.covariance[7] = covariance;
  odom.pose.covariance[14] = covariance * 2.0;
  odom.pose.covariance[21] = covariance;
  odom.pose.covariance[28] = covariance;
  odom.pose.covariance[35] = covariance;
  odom_pub_->publish(odom);

  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header = odom.header;
  pose_stamped.pose = odom.pose.pose;
  path_msg_.header.stamp = stamp;
  path_msg_.poses.push_back(pose_stamped);
  if (max_path_poses_ > 0 &&
    static_cast<int>(path_msg_.poses.size()) > max_path_poses_)
  {
    path_msg_.poses.erase(path_msg_.poses.begin());
  }
  path_pub_->publish(path_msg_);

  if (publish_tf_) {
    tf_broadcaster_->sendTransform(
      isometryToTransform(map_T_base_, map_frame_, base_frame_, stamp));
  }
}

void ScanToMapIcp::publishCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
  const CloudConstPtr & cloud,
  const std::string & frame_id,
  const rclcpp::Time & stamp) const
{
  if (!publisher || !cloud || cloud->empty()) {
    return;
  }

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  publisher->publish(msg);
}

void ScanToMapIcp::publishDiagnostics(
  const rclcpp::Time & stamp,
  const RegistrationResult & result)
{
  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = stamp;

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "scan_to_map_icp";
  status.hardware_id = "slam_3d_core";
  status.level = result.accepted ? diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  status.message = result.accepted ? "registration accepted" : "registration rejected";

  rclcpp::Time latest_wheel_stamp;
  {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    latest_wheel_stamp = latest_wheel_stamp_;
  }

  bool have_gravity_accel = false;
  Eigen::Vector3d gravity_accel = Eigen::Vector3d::UnitZ();
  double gravity_filter_alpha = 0.0;
  int imu_duplicate_dropped_count = 0;
  int imu_out_of_order_dropped_count = 0;
  int imu_rewind_dropped_count = 0;
  int imu_timestamp_reset_count = 0;
  rclcpp::Time latest_imu_stamp;
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    latest_imu_stamp = latest_imu_stamp_;
    have_gravity_accel = have_gravity_accel_base_;
    gravity_accel = gravity_accel_base_;
    gravity_filter_alpha = latest_gravity_filter_alpha_;
    imu_duplicate_dropped_count = imu_duplicate_dropped_count_;
    imu_out_of_order_dropped_count = imu_out_of_order_dropped_count_;
    imu_rewind_dropped_count = imu_rewind_dropped_count_;
    imu_timestamp_reset_count = imu_timestamp_reset_count_;
  }
  const double gravity_accel_norm = gravity_accel.norm();
  double gravity_roll_deg = 0.0;
  double gravity_pitch_deg = 0.0;
  if (have_gravity_accel && gravity_accel_norm > 1.0e-6 && gravity_accel.allFinite()) {
    const Eigen::Vector3d normalized_gravity = gravity_accel.normalized();
    gravity_roll_deg =
      std::atan2(normalized_gravity.y(), normalized_gravity.z()) * 180.0 / M_PI;
    gravity_pitch_deg =
      std::atan2(
      -normalized_gravity.x(),
      std::hypot(normalized_gravity.y(), normalized_gravity.z())) * 180.0 / M_PI;
  }

  auto add_value = [&status](const std::string & key, const std::string & value) {
      diagnostic_msgs::msg::KeyValue kv;
      kv.key = key;
      kv.value = value;
      status.values.push_back(kv);
    };

  add_value("converged", result.converged ? "true" : "false");
  add_value("accepted", result.accepted ? "true" : "false");
  add_value("rejection_reason", result.rejection_reason);
  add_value("fitness", toString(result.fitness));
  add_value("inlier_count", toString(result.inlier_count));
  add_value("inlier_ratio", toString(result.inlier_ratio));
  add_value("inlier_rmse_m", toString(result.inlier_rmse_m));
  add_value(
    "robust_registration",
    robust_registration_enabled_ ? "enabled" : "disabled");
  add_value(
    "registration_trimmed_correspondences_applied",
    result.registration_trimmed_correspondences_applied ? "true" : "false");
  add_value(
    "registration_refinement_attempted",
    result.registration_refinement_attempted ? "true" : "false");
  add_value(
    "registration_refinement_applied",
    result.registration_refinement_applied ? "true" : "false");
  add_value(
    "registration_quality_max_correspondence_m",
    toString(result.registration_quality_max_correspondence_m));
  add_value("registration_coarse_iterations", toString(result.registration_coarse_iterations));
  add_value("registration_coarse_fitness", toString(result.registration_coarse_fitness));
  add_value(
    "registration_coarse_inlier_ratio",
    toString(result.registration_coarse_inlier_ratio));
  add_value(
    "registration_coarse_inlier_rmse_m",
    toString(result.registration_coarse_inlier_rmse_m));
  add_value(
    "registration_refinement_iterations",
    toString(result.registration_refinement_iterations));
  add_value(
    "registration_refinement_fitness",
    toString(result.registration_refinement_fitness));
  add_value(
    "registration_refinement_inlier_ratio",
    toString(result.registration_refinement_inlier_ratio));
  add_value(
    "registration_refinement_inlier_rmse_m",
    toString(result.registration_refinement_inlier_rmse_m));
  add_value("correction_translation_m", toString(result.correction_translation_m));
  add_value("correction_rotation_deg", toString(result.correction_rotation_deg));
  add_value("transform_determinant", toString(result.transform_determinant));
  add_value("transform_orthogonality_error", toString(result.transform_orthogonality_error));
  add_value("iterations", toString(result.iterations));
  add_value("scan_points_before_downsample", toString(result.scan_points_before_downsample));
  add_value("scan_points_after_downsample", toString(result.scan_points_after_downsample));
  add_value("scan_voxel_leaf_m", toString(result.scan_voxel_leaf_m));
  add_value("scan_voxel_adapted", result.scan_voxel_adapted ? "true" : "false");
  add_value("adaptive_scan_voxel", adaptive_scan_voxel_enabled_ ? "enabled" : "disabled");
  add_value("deskew", deskew_enabled_ ? "enabled" : "disabled");
  add_value("deskew_attempted", result.deskew_attempted ? "true" : "false");
  add_value("deskew_applied", result.deskew_applied ? "true" : "false");
  add_value("deskew_status", result.deskew_status);
  add_value("deskew_point_time_field", result.deskew_point_time_field);
  add_value("deskew_points_with_time", toString(result.deskew_points_with_time));
  add_value("deskew_point_count", toString(result.deskew_point_count));
  add_value("deskew_time_ratio", toString(result.deskew_time_ratio));
  add_value("deskew_scan_duration_s", toString(result.deskew_scan_duration_s));
  add_value(
    "imu_roughness_detection",
    imu_roughness_detection_enabled_ ? "enabled" : "disabled");
  add_value(
    "imu_roughness_evaluated",
    result.imu_roughness_evaluated ? "true" : "false");
  add_value("imu_roughness_event", result.imu_roughness_event ? "true" : "false");
  add_value(
    "imu_roughness_recent_event",
    result.imu_roughness_recent_event ? "true" : "false");
  add_value("imu_roughness_reason", result.imu_roughness_reason);
  add_value("imu_roughness_sample_count", toString(result.imu_roughness_sample_count));
  add_value(
    "imu_roughness_max_angular_velocity_radps",
    toString(result.imu_roughness_max_angular_velocity_radps));
  add_value(
    "imu_roughness_max_angular_accel_radps2",
    toString(result.imu_roughness_max_angular_accel_radps2));
  add_value(
    "imu_roughness_max_linear_accel_mps2",
    toString(result.imu_roughness_max_linear_accel_mps2));
  add_value(
    "imu_roughness_max_accel_norm_delta_mps2",
    toString(result.imu_roughness_max_accel_norm_delta_mps2));
  add_value(
    "imu_roughness_max_linear_jerk_mps3",
    toString(result.imu_roughness_max_linear_jerk_mps3));
  add_value(
    "imu_roughness_time_since_event_s",
    toString(result.imu_roughness_time_since_event_s));
  add_value("shock_quarantined", result.shock_quarantined ? "true" : "false");
  add_value("shock_quarantine_reason", result.shock_quarantine_reason);
  add_value(
    "quarantine_keyframes_on_shock",
    quarantine_keyframes_on_shock_ ? "enabled" : "disabled");
  add_value(
    "quarantine_pose_on_shock",
    quarantine_pose_on_shock_ ? "enabled" : "disabled");
  add_value("pose_quarantined", result.pose_quarantined ? "true" : "false");
  add_value(
    "consecutive_pose_quarantine_count",
    toString(result.consecutive_pose_quarantine_count));
  add_value(
    "pose_quarantine_tracking_reset",
    result.pose_quarantine_tracking_reset ? "true" : "false");
  add_value(
    "record_tracking_scan_on_pose_quarantine",
    record_tracking_scan_on_pose_quarantine_ ? "enabled" : "disabled");
  add_value(
    "pose_quarantine_tracking_reset_interval",
    toString(pose_quarantine_tracking_reset_interval_));
  add_value("keyframe_added", result.keyframe_added ? "true" : "false");
  add_value("keyframe_quarantined", result.keyframe_quarantined ? "true" : "false");
  add_value("keyframe_quarantine_reason", result.keyframe_quarantine_reason);
  add_value(
    "place_recognition",
    place_recognition_enabled_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_periodic",
    place_recognition_periodic_enabled_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_require_pose_graph_correction",
    place_recognition_require_pose_graph_correction_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_pose_graph_validation_effective",
    (place_recognition_require_pose_graph_correction_ || pose_graph_robust_loop_enabled_) ?
    "enabled" : "disabled");
  add_value(
    "place_recognition_attempted",
    result.place_recognition_attempted ? "true" : "false");
  add_value(
    "place_recognition_skipped",
    result.place_recognition_skipped ? "true" : "false");
  add_value(
    "place_recognition_accepted",
    result.place_recognition_accepted ? "true" : "false");
  add_value("place_recognition_trigger", result.place_recognition_trigger);
  add_value("place_recognition_skip_reason", result.place_recognition_skip_reason);
  add_value(
    "place_recognition_candidates",
    toString(result.place_recognition_candidates));
  add_value(
    "place_recognition_candidate_index",
    toString(result.place_recognition_candidate_index));
  add_value(
    "place_recognition_descriptor_distance",
    toString(result.place_recognition_descriptor_distance));
  add_value(
    "place_recognition_yaw_offset_deg",
    toString(result.place_recognition_yaw_offset_deg));
  add_value(
    "place_recognition_pose_metrics_evaluated",
    result.place_recognition_pose_metrics_evaluated ? "true" : "false");
  add_value(
    "place_recognition_pose_metrics_valid",
    result.place_recognition_pose_metrics_valid ? "true" : "false");
  add_value(
    "place_recognition_prediction_jump_translation_m",
    toString(result.place_recognition_prediction_jump_translation_m));
  add_value(
    "place_recognition_prediction_jump_rotation_deg",
    toString(result.place_recognition_prediction_jump_rotation_deg));
  add_value(
    "place_recognition_map_correction_translation_m",
    toString(result.place_recognition_map_correction_translation_m));
  add_value(
    "place_recognition_map_correction_rotation_deg",
    toString(result.place_recognition_map_correction_rotation_deg));
  add_value(
    "pose_graph_correction",
    pose_graph_correction_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_planar_correction",
    pose_graph_planar_correction_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_robust_loop",
    pose_graph_robust_loop_enabled_ ? "enabled" : "disabled");
  add_value("pose_graph_loop_cauchy_scale", toString(pose_graph_loop_cauchy_scale_));
  add_value(
    "pose_graph_min_new_loop_robust_weight",
    toString(pose_graph_min_new_loop_robust_weight_));
  add_value(
    "planar_pose_constraint",
    planar_pose_constraint_enabled_ ? "enabled" : "disabled");
  add_value("planar_pose_max_z_correction_m", toString(planar_pose_max_z_correction_m_));
  add_value(
    "gravity_roll_pitch_constraint",
    gravity_roll_pitch_constraint_enabled_ ? "enabled" : "disabled");
  add_value("gravity_constraint_status", result.gravity_constraint_status);
  add_value("gravity_accel_available", result.gravity_accel_used ? "true" : "false");
  add_value("gravity_accel_age_s", toString(result.gravity_accel_age_s));
  add_value("gravity_accel_norm_mps2", toString(result.gravity_accel_norm_mps2));
  add_value(
    "gravity_accel_filter_mode",
    gravity_accel_time_constant_s_ > 0.0 ? "time_constant" : "legacy_alpha");
  add_value("gravity_accel_time_constant_s", toString(gravity_accel_time_constant_s_));
  add_value("gravity_accel_max_age_s", toString(gravity_accel_max_age_s_));
  add_value("gravity_accel_effective_alpha", toString(result.gravity_filter_alpha));
  add_value("gravity_roll_deg", toString(result.gravity_roll_deg));
  add_value("gravity_pitch_deg", toString(result.gravity_pitch_deg));
  add_value("latest_gravity_accel_available", have_gravity_accel ? "true" : "false");
  add_value("latest_gravity_accel_norm_mps2", toString(gravity_accel_norm));
  add_value("latest_gravity_accel_effective_alpha", toString(gravity_filter_alpha));
  add_value("latest_gravity_roll_deg", toString(gravity_roll_deg));
  add_value("latest_gravity_pitch_deg", toString(gravity_pitch_deg));
  add_value("map_z_filter", map_z_filter_enabled_ ? "enabled" : "disabled");
  add_value("map_z_filter_min_m", toString(map_z_filter_min_m_));
  add_value("map_z_filter_max_m", toString(map_z_filter_max_m_));
  add_value(
    "pose_graph_correction_applied",
    result.pose_graph_correction_applied ? "true" : "false");
  add_value(
    "pose_graph_new_loop_rejected",
    result.pose_graph_new_loop_rejected ? "true" : "false");
  add_value("pose_graph_rejection_reason", result.pose_graph_rejection_reason);
  add_value(
    "pose_graph_correction_keyframes",
    toString(result.pose_graph_correction_keyframes));
  add_value(
    "pose_graph_correction_translation_m",
    toString(result.pose_graph_correction_translation_m));
  add_value(
    "pose_graph_correction_rotation_deg",
    toString(result.pose_graph_correction_rotation_deg));
  add_value(
    "pose_graph_optimization_iterations",
    toString(result.pose_graph_optimization_iterations));
  add_value("pose_graph_edges", toString(result.pose_graph_edges));
  add_value("pose_graph_loop_edges", toString(result.pose_graph_loop_edges));
  add_value(
    "pose_graph_robust_loop_edges_downweighted",
    toString(result.pose_graph_robust_loop_edges_downweighted));
  add_value("pose_graph_max_loop_residual_before", toString(pose_graph_max_loop_residual_before_));
  add_value("pose_graph_residual_before", toString(result.pose_graph_residual_before));
  add_value("pose_graph_residual_after", toString(result.pose_graph_residual_after));
  add_value(
    "pose_graph_robust_objective_before",
    toString(result.pose_graph_robust_objective_before));
  add_value(
    "pose_graph_robust_objective_after",
    toString(result.pose_graph_robust_objective_after));
  add_value(
    "pose_graph_min_loop_robust_weight",
    toString(result.pose_graph_min_loop_robust_weight));
  add_value(
    "pose_graph_new_loop_robust_weight",
    toString(result.pose_graph_new_loop_robust_weight));
  add_value(
    "pose_graph_max_update_translation_m",
    toString(result.pose_graph_max_update_translation_m));
  add_value(
    "pose_graph_max_update_rotation_deg",
    toString(result.pose_graph_max_update_rotation_deg));
  add_value("keyframes", toString(keyframes_.size()));
  add_value("global_map_points", toString(global_map_ ? global_map_->size() : 0));
  add_value(
    "constant_velocity_prediction",
    use_constant_velocity_prediction_ ? "enabled" : "disabled");
  add_value("constant_velocity_ready", have_previous_accepted_pose_ ? "true" : "false");
  add_value(
    "imu_rotation_prediction",
    imu_rotation_prediction_enabled_ ? "enabled" : "disabled");
  add_value(
    "imu_rotation_prediction_used",
    result.imu_rotation_prediction_used ? "true" : "false");
  add_value("imu_rotation_prediction_status", result.imu_rotation_prediction_status);
  add_value("scan_to_scan_prediction", use_scan_to_scan_prediction_ ? "enabled" : "disabled");
  add_value(
    "scan_to_scan_prediction_used",
    result.scan_to_scan_prediction_used ? "true" : "false");
  add_value("scan_to_scan_prediction_status", result.scan_to_scan_prediction_status);
  add_value(
    "scan_to_scan_reference_age_s",
    toString(result.scan_to_scan_reference_age_s));
  add_value(
    "scan_to_scan_max_reference_age_s",
    toString(scan_to_scan_max_reference_age_s_));
  add_value("scan_to_scan_fallback", use_scan_to_scan_fallback_ ? "enabled" : "disabled");
  add_value(
    "scan_to_scan_fallback_keyframes",
    add_scan_to_scan_fallback_keyframes_ ? "enabled" : "disabled");
  add_value("motion_prior_fallback", use_motion_prior_fallback_ ? "enabled" : "disabled");
  add_value(
    "motion_prior_fallback_keyframes",
    add_motion_prior_fallback_keyframes_ ? "enabled" : "disabled");
  add_value("latest_imu_stamp_ns", toString(latest_imu_stamp.nanoseconds()));
  add_value("imu_watermark_topic", imu_watermark_topic_);
  add_value("imu_subscription_depth", toString(imu_subscription_depth_));
  add_value("imu_reorder_tolerance_s", toString(imu_reorder_tolerance_s_));
  add_value(
    "imu_timestamp_reset_threshold_s",
    toString(imu_timestamp_reset_threshold_s_));
  add_value("imu_duplicate_dropped_count", toString(imu_duplicate_dropped_count));
  add_value("imu_out_of_order_dropped_count", toString(imu_out_of_order_dropped_count));
  add_value("imu_rewind_dropped_count", toString(imu_rewind_dropped_count));
  add_value("imu_timestamp_reset_count", toString(imu_timestamp_reset_count));
  add_value(
    "imu_timestamp_reset_policy",
    "detect_and_drop_without_partial_state_reset");
  add_value(
    "cloud_subscription_reliability",
    cloud_subscription_reliable_ ? "reliable" : "best_effort");
  add_value("secondary_cloud_subscription_reliability", "best_effort");
  add_value("cloud_subscription_depth", toString(cloud_subscription_depth_));
  add_value("latest_wheel_stamp_ns", toString(latest_wheel_stamp.nanoseconds()));

  array.status.push_back(status);
  health_pub_->publish(array);
}

Eigen::Isometry3d ScanToMapIcp::odomToIsometry(const nav_msgs::msg::Odometry & msg)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  const auto & position = msg.pose.pose.position;
  const auto & orientation = msg.pose.pose.orientation;
  const Eigen::Quaterniond q(
    orientation.w,
    orientation.x,
    orientation.y,
    orientation.z);
  pose.linear() = q.normalized().toRotationMatrix();
  pose.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return pose;
}

geometry_msgs::msg::Pose ScanToMapIcp::isometryToPose(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Pose msg;
  const Eigen::Quaterniond q(pose.linear());
  msg.position.x = pose.translation().x();
  msg.position.y = pose.translation().y();
  msg.position.z = pose.translation().z();
  msg.orientation.x = q.x();
  msg.orientation.y = q.y();
  msg.orientation.z = q.z();
  msg.orientation.w = q.w();
  return msg;
}

geometry_msgs::msg::TransformStamped ScanToMapIcp::isometryToTransform(
  const Eigen::Isometry3d & pose,
  const std::string & parent_frame,
  const std::string & child_frame,
  const rclcpp::Time & stamp)
{
  geometry_msgs::msg::TransformStamped msg;
  const Eigen::Quaterniond q(pose.linear());
  msg.header.stamp = stamp;
  msg.header.frame_id = parent_frame;
  msg.child_frame_id = child_frame;
  msg.transform.translation.x = pose.translation().x();
  msg.transform.translation.y = pose.translation().y();
  msg.transform.translation.z = pose.translation().z();
  msg.transform.rotation.x = q.x();
  msg.transform.rotation.y = q.y();
  msg.transform.rotation.z = q.z();
  msg.transform.rotation.w = q.w();
  return msg;
}

double ScanToMapIcp::yawFromIsometry(const Eigen::Isometry3d & pose)
{
  return std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
}

}  // namespace slam_3d_core
}  // namespace gole
