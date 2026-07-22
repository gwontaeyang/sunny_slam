#ifndef SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_
#define SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
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
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

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

  struct Keyframe
  {
    rclcpp::Time stamp;
    Eigen::Isometry3d map_T_base{Eigen::Isometry3d::Identity()};
    CloudPtr cloud_base;
    CloudPtr cloud_map;
    std::vector<float> descriptor;
  };

  struct PoseGraphEdge
  {
    int from_index{0};
    int to_index{0};
    Eigen::Isometry3d from_T_to{Eigen::Isometry3d::Identity()};
    double translation_weight{1.0};
    double rotation_weight{1.0};
    bool loop_closure{false};
  };

  struct PoseGraphStats
  {
    bool applied{false};
    int iterations{0};
    int nodes{0};
    int edges{0};
    int loop_edges{0};
    int robust_loop_edges_downweighted{0};
    double residual_before{0.0};
    double residual_after{0.0};
    double robust_objective_before{0.0};
    double robust_objective_after{0.0};
    double min_loop_robust_weight{1.0};
    double new_loop_robust_weight{1.0};
    double max_update_translation_m{0.0};
    double max_update_rotation_deg{0.0};
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
    double transform_determinant{1.0};
    double transform_orthogonality_error{0.0};
    int inlier_count{0};
    int iterations{0};
    int scan_points_before_downsample{0};
    int scan_points_after_downsample{0};
    double scan_voxel_leaf_m{0.0};
    bool scan_voxel_adapted{false};
    bool shock_quarantined{false};
    bool pose_quarantined{false};
    bool keyframe_added{false};
    bool keyframe_quarantined{false};
    bool deskew_attempted{false};
    bool deskew_applied{false};
    bool imu_rotation_prediction_used{false};
    bool place_recognition_attempted{false};
    bool place_recognition_skipped{false};
    bool place_recognition_accepted{false};
    bool pose_graph_correction_applied{false};
    bool imu_roughness_evaluated{false};
    bool imu_roughness_event{false};
    bool imu_roughness_recent_event{false};
    bool registration_trimmed_correspondences_applied{false};
    bool registration_refinement_attempted{false};
    bool registration_refinement_applied{false};
    int deskew_points_with_time{0};
    int deskew_point_count{0};
    int imu_roughness_sample_count{0};
    int place_recognition_candidates{0};
    int place_recognition_candidate_index{-1};
    int pose_graph_correction_keyframes{0};
    int pose_graph_optimization_iterations{0};
    int pose_graph_edges{0};
    int pose_graph_loop_edges{0};
    int pose_graph_robust_loop_edges_downweighted{0};
    int registration_coarse_iterations{0};
    int registration_refinement_iterations{0};
    double deskew_time_ratio{0.0};
    double deskew_scan_duration_s{0.0};
    double place_recognition_descriptor_distance{0.0};
    double place_recognition_yaw_offset_deg{0.0};
    double pose_graph_correction_translation_m{0.0};
    double pose_graph_correction_rotation_deg{0.0};
    double pose_graph_residual_before{0.0};
    double pose_graph_residual_after{0.0};
    double pose_graph_robust_objective_before{0.0};
    double pose_graph_robust_objective_after{0.0};
    double pose_graph_min_loop_robust_weight{1.0};
    double pose_graph_new_loop_robust_weight{1.0};
    double pose_graph_max_update_translation_m{0.0};
    double pose_graph_max_update_rotation_deg{0.0};
    double imu_roughness_max_angular_velocity_radps{0.0};
    double imu_roughness_max_angular_accel_radps2{0.0};
    double imu_roughness_max_linear_accel_mps2{0.0};
    double imu_roughness_max_accel_norm_delta_mps2{0.0};
    double imu_roughness_max_linear_jerk_mps3{0.0};
    double imu_roughness_time_since_event_s{0.0};
    double registration_quality_max_correspondence_m{0.0};
    double registration_coarse_fitness{0.0};
    double registration_coarse_inlier_ratio{0.0};
    double registration_coarse_inlier_rmse_m{0.0};
    double registration_refinement_fitness{0.0};
    double registration_refinement_inlier_ratio{0.0};
    double registration_refinement_inlier_rmse_m{0.0};
    Eigen::Isometry3d map_T_base{Eigen::Isometry3d::Identity()};
    std::string rejection_reason;
    std::string shock_quarantine_reason;
    std::string keyframe_quarantine_reason;
    std::string deskew_status;
    std::string deskew_point_time_field;
    std::string imu_rotation_prediction_status{"disabled"};
    std::string imu_roughness_reason{"disabled"};
    std::string place_recognition_trigger{"none"};
    std::string place_recognition_skip_reason;
  };

  struct DeskewStats
  {
    bool attempted{false};
    bool applied{false};
    int points_with_time{0};
    int point_count{0};
    double time_ratio{0.0};
    double scan_duration_s{0.0};
    std::string status{"disabled"};
    std::string point_time_field;
  };

  struct ImuSample
  {
    rclcpp::Time stamp;
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()};
    Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
  };

  struct ImuRoughnessStats
  {
    bool evaluated{false};
    bool event{false};
    bool recent_event{false};
    int sample_count{0};
    double max_angular_velocity_radps{0.0};
    double max_angular_accel_radps2{0.0};
    double max_linear_accel_mps2{0.0};
    double max_accel_norm_delta_mps2{0.0};
    double max_linear_jerk_mps3{0.0};
    double time_since_event_s{0.0};
    std::string reason{"disabled"};
  };

  void setupParameters();
  void setupRosInterfaces();

  void primaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void secondaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
  void mapPublishTimerCallback();

  void processCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & primary_msg);
  CloudPtr buildBaseFrameScan(
    const sensor_msgs::msg::PointCloud2 & primary_msg,
    DeskewStats & deskew_stats);
  CloudPtr convertAndTransformCloud(
    const sensor_msgs::msg::PointCloud2 & msg,
    DeskewStats & deskew_stats);
  CloudPtr filterCloud(const CloudConstPtr & cloud) const;
  CloudPtr downsampleCloud(const CloudConstPtr & cloud, double leaf_size) const;
  CloudPtr downsampleScanAdaptive(
    const CloudConstPtr & cloud,
    double & effective_leaf_size) const;
  CloudPtr transformCloud(const CloudConstPtr & cloud, const Eigen::Isometry3d & transform) const;
  CloudPtr filterMapCloud(const CloudConstPtr & cloud) const;
  bool deskewCloudInBaseFrame(
    const sensor_msgs::msg::PointCloud2 & msg,
    const CloudPtr & cloud_base,
    DeskewStats & stats) const;
  void applyDeskewStatsToResult(
    const DeskewStats & stats,
    RegistrationResult & result) const;
  ImuRoughnessStats evaluateImuRoughness(
    const rclcpp::Time & start_stamp,
    const rclcpp::Time & end_stamp,
    bool have_start_stamp) const;
  void applyImuRoughnessStatsToResult(
    const ImuRoughnessStats & stats,
    RegistrationResult & result) const;

  Eigen::Isometry3d lookupTransformToBase(const sensor_msgs::msg::PointCloud2 & msg);
  Eigen::Isometry3d predictWithWheelOdom();
  Eigen::Isometry3d predictWithConstantVelocity() const;
  Eigen::Isometry3d predictWithMotionPriors(
    const CloudConstPtr & scan_base,
    const rclcpp::Time & previous_scan_stamp,
    const rclcpp::Time & current_scan_stamp,
    bool have_previous_scan_stamp,
    bool & used_scan_to_scan,
    bool & used_imu_rotation_prediction,
    std::string & imu_rotation_prediction_status);
  std::optional<Eigen::Quaterniond> integrateImuRotationDelta(
    const rclcpp::Time & start_stamp,
    const rclcpp::Time & end_stamp,
    std::string & status) const;
  Eigen::Isometry3d clampRelativeMotion(const Eigen::Isometry3d & delta) const;
  Eigen::Isometry3d constrainMotionPriorFallback(
    const Eigen::Isometry3d & candidate) const;
  Eigen::Isometry3d constrainPlanarPose(
    const Eigen::Isometry3d & candidate,
    const Eigen::Isometry3d & reference) const;
  void commitWheelPrediction();
  void recordAcceptedPose();
  void recordTrackingScan(const CloudConstPtr & scan_base, const Eigen::Isometry3d & pose);
  RegistrationResult registerScanToMap(
    const CloudConstPtr & scan_base,
    const CloudConstPtr & local_map,
    const Eigen::Isometry3d & prediction) const;
  void evaluateRegistrationOverlap(
    const CloudT & aligned_scan,
    const CloudConstPtr & local_map,
    double max_correspondence_m,
    RegistrationResult & result) const;
  bool registrationCorrectionWithinLimits(RegistrationResult & result) const;

  CloudPtr buildLocalMap(const Eigen::Isometry3d & center_pose) const;
  CloudPtr buildCandidateMapAroundKeyframe(int keyframe_index) const;
  void initializeFromFirstScan(const CloudConstPtr & scan_base, const rclcpp::Time & stamp);
  void addKeyframe(const CloudConstPtr & scan_base, const rclcpp::Time & stamp);
  bool shouldAddKeyframe(const Eigen::Isometry3d & pose) const;
  std::string shockQuarantineReason(const RegistrationResult & result) const;
  std::vector<float> computeScanDescriptor(const CloudConstPtr & scan_base) const;
  double descriptorDistance(
    const std::vector<float> & lhs,
    const std::vector<float> & rhs,
    int & best_shift) const;
  bool tryPlaceRecognition(
    const CloudConstPtr & scan_base,
    const std::vector<float> & scan_descriptor,
    const Eigen::Isometry3d & reference_pose,
    RegistrationResult & result) const;
  void refreshKeyframeCloudMap(std::size_t keyframe_index);
  void prunePoseGraphEdgesAfterOldestKeyframeRemoval();
  double poseGraphEdgeSquaredError(
    const PoseGraphEdge & edge,
    const std::vector<Eigen::Isometry3d> & poses) const;
  double poseGraphLoopRobustWeight(const PoseGraphEdge & edge, double squared_error) const;
  double poseGraphResidual(
    const std::vector<Eigen::Isometry3d> & poses,
    bool apply_robust_loop_kernel) const;
  PoseGraphStats optimizePoseGraphWithLoop(
    int anchor_index,
    const Eigen::Isometry3d & target_loop_pose,
    const std::vector<Eigen::Isometry3d> & previous_keyframe_poses);
  void updatePathFromKeyframeCorrections(
    const std::vector<Eigen::Isometry3d> & previous_keyframe_poses);
  void applyPoseGraphCorrection(
    const rclcpp::Time & stamp,
    const Eigen::Isometry3d & reference_pose,
    RegistrationResult & result);
  void rebuildGlobalMap();

  void publishState(const rclcpp::Time & stamp);
  void publishCloud(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
    const CloudConstPtr & cloud,
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
  std::string secondary_cloud_topic_;
  std::string wheel_odom_topic_;
  std::string imu_topic_;
  std::string odom_topic_;
  std::string path_topic_;
  std::string local_map_topic_;
  std::string global_map_topic_;
  std::string health_topic_;
  std::string registration_method_;

  bool use_secondary_cloud_{false};
  bool use_tf_transforms_{true};
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
  bool place_recognition_enabled_{false};
  bool place_recognition_periodic_enabled_{false};
  bool place_recognition_require_pose_graph_correction_{false};
  bool pose_graph_correction_enabled_{false};
  bool pose_graph_planar_correction_enabled_{false};
  bool pose_graph_robust_loop_enabled_{false};
  bool planar_pose_constraint_enabled_{false};
  bool gravity_roll_pitch_constraint_enabled_{false};
  bool map_z_filter_enabled_{false};
  bool deskew_enabled_{false};
  bool imu_rotation_prediction_enabled_{false};
  bool imu_roughness_detection_enabled_{false};
  bool robust_registration_enabled_{false};
  bool publish_tf_{false};
  bool crop_box_enabled_{true};
  bool initialized_{false};
  std::atomic_bool processing_{false};

  double max_secondary_stamp_delta_s_{0.05};
  double min_range_m_{0.5};
  double max_range_m_{80.0};
  double z_min_m_{-5.0};
  double z_max_m_{8.0};
  double scan_voxel_leaf_m_{0.25};
  double adaptive_scan_voxel_leaf_m_{0.15};
  double map_voxel_leaf_m_{0.35};
  double local_map_radius_m_{35.0};
  double keyframe_translation_m_{0.8};
  double keyframe_rotation_deg_{8.0};
  double icp_max_correspondence_m_{2.0};
  double fitness_score_threshold_{2.0};
  double robust_registration_trim_ratio_{0.85};
  double robust_registration_refinement_max_correspondence_m_{0.8};
  double robust_registration_quality_max_correspondence_m_{0.8};
  double constant_velocity_translation_limit_m_{2.0};
  double constant_velocity_rotation_limit_deg_{30.0};
  double motion_prior_fallback_max_step_m_{0.0};
  double motion_prior_fallback_max_z_step_m_{0.0};
  double motion_prior_fallback_max_untrusted_distance_m_{0.0};
  double motion_prior_fallback_max_untrusted_z_m_{0.0};
  double motion_prior_fallback_max_rotation_deg_{0.0};
  double max_registration_correction_translation_m_{4.0};
  double max_registration_correction_rotation_deg_{60.0};
  double deskew_point_time_scale_{0.0};
  double deskew_point_time_offset_s_{0.0};
  double deskew_max_scan_duration_s_{0.2};
  double deskew_max_imu_gap_s_{0.05};
  double deskew_imu_buffer_duration_s_{5.0};
  double deskew_min_time_ratio_{0.5};
  double deskew_time_bin_s_{0.001};
  double deskew_angular_velocity_scale_{1.0};
  double deskew_imu_to_base_roll_deg_{0.0};
  double deskew_imu_to_base_pitch_deg_{0.0};
  double deskew_imu_to_base_yaw_deg_{0.0};
  double imu_rotation_prediction_max_dt_s_{0.5};
  double imu_rotation_prediction_scale_{1.0};
  double imu_roughness_window_s_{0.35};
  double imu_roughness_recent_event_window_s_{1.0};
  double imu_roughness_angular_velocity_threshold_radps_{2.5};
  double imu_roughness_angular_accel_threshold_radps2_{40.0};
  double imu_roughness_accel_norm_delta_threshold_mps2_{3.0};
  double imu_roughness_linear_jerk_threshold_mps3_{60.0};
  double keyframe_quarantine_correction_translation_m_{0.8};
  double keyframe_quarantine_correction_rotation_deg_{15.0};
  double place_recognition_periodic_min_interval_s_{3.0};
  double place_recognition_relocalization_min_interval_s_{0.0};
  double place_recognition_shock_relocalization_cooldown_s_{0.0};
  double place_recognition_max_descriptor_distance_{0.20};
  double place_recognition_min_inlier_ratio_{0.35};
  double place_recognition_max_inlier_rmse_m_{0.8};
  double place_recognition_max_pose_jump_m_{3.0};
  double place_recognition_max_pose_jump_rotation_deg_{45.0};
  double place_recognition_max_yaw_offset_deg_{180.0};
  double pose_graph_correction_max_translation_m_{5.0};
  double pose_graph_correction_max_rotation_deg_{45.0};
  double pose_graph_correction_min_interval_s_{1.0};
  double pose_graph_optimization_step_{0.15};
  double pose_graph_odom_translation_weight_{1.0};
  double pose_graph_odom_rotation_weight_{1.0};
  double pose_graph_loop_translation_weight_{1.0};
  double pose_graph_loop_rotation_weight_{1.0};
  double pose_graph_loop_cauchy_scale_{1.0};
  double pose_graph_max_iteration_translation_step_m_{0.25};
  double pose_graph_max_iteration_rotation_step_deg_{5.0};
  double pose_graph_max_loop_residual_before_{0.0};
  double pose_graph_max_residual_after_{0.0};
  double pose_graph_max_residual_ratio_{0.0};
  double planar_pose_max_z_correction_m_{0.0};
  double gravity_accel_low_pass_alpha_{0.02};
  double gravity_accel_min_norm_mps2_{6.0};
  double gravity_accel_max_norm_mps2_{13.0};
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

  int min_scan_points_{200};
  int adaptive_scan_voxel_min_points_{900};
  int place_recognition_descriptor_bins_{60};
  int place_recognition_descriptor_rings_{1};
  int place_recognition_min_keyframes_{25};
  int place_recognition_exclude_recent_keyframes_{20};
  int place_recognition_max_candidates_{4};
  int place_recognition_candidate_map_neighbors_{6};
  int pose_graph_correction_min_keyframes_{5};
  int pose_graph_optimization_iterations_{25};
  int imu_subscription_depth_{400};
  int max_icp_iterations_{40};
  int robust_registration_refinement_iterations_{15};
  int max_keyframes_{500};
  int max_local_keyframes_{80};
  int max_path_poses_{10000};

  std::string deskew_point_time_field_{"auto"};

  mutable std::mutex state_mutex_;
  mutable std::mutex secondary_mutex_;
  mutable std::mutex wheel_mutex_;
  mutable std::mutex imu_mutex_;

  Eigen::Isometry3d map_T_base_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_keyframe_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d latest_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_used_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d pending_wheel_odom_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d previous_accepted_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d last_accepted_pose_{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d previous_tracking_pose_{Eigen::Isometry3d::Identity()};

  bool have_latest_wheel_odom_{false};
  bool have_last_used_wheel_odom_{false};
  bool have_pending_wheel_odom_{false};
  bool have_previous_accepted_pose_{false};
  bool have_last_accepted_pose_{false};
  bool have_previous_tracking_scan_{false};
  bool have_last_processed_scan_stamp_{false};
  bool have_last_pose_graph_correction_stamp_{false};
  bool have_last_place_recognition_attempt_stamp_{false};
  bool have_gravity_accel_base_{false};
  bool have_last_imu_roughness_event_stamp_{false};
  rclcpp::Time latest_imu_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time latest_wheel_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_processed_scan_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_pose_graph_correction_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_place_recognition_attempt_stamp_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_imu_roughness_event_stamp_{0, 0, RCL_ROS_TIME};

  std::vector<Keyframe> keyframes_;
  std::vector<PoseGraphEdge> pose_graph_edges_;
  std::deque<ImuSample> imu_buffer_;
  Eigen::Vector3d gravity_accel_base_{Eigen::Vector3d::UnitZ()};
  CloudPtr global_map_;
  CloudPtr latest_local_map_;
  CloudPtr previous_tracking_scan_base_;
  nav_msgs::msg::Path path_msg_;
  RegistrationResult last_registration_;

  sensor_msgs::msg::PointCloud2::SharedPtr latest_secondary_msg_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr primary_cloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr secondary_cloud_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::CallbackGroup::SharedPtr imu_callback_group_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr local_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_map_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_pub_;

  rclcpp::TimerBase::SharedPtr map_publish_timer_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__SCAN_TO_MAP_ICP_HPP_
