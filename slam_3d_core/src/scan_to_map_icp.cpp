#include "slam_3d_core/scan_to_map_icp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/icp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>

#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "slam_3d_core/registration_policy.hpp"
#include "slam_3d_core/rigid_transform.hpp"
#include "slam_3d_core/scan_context_descriptor.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "visualization_msgs/msg/marker.hpp"

namespace gole
{
namespace slam_3d_core
{

namespace
{

struct ConfirmedVoxelKey
{
  std::int64_t x{0};
  std::int64_t y{0};
  std::int64_t z{0};

  bool operator==(const ConfirmedVoxelKey & other) const noexcept
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct ConfirmedVoxelKeyHash
{
  std::size_t operator()(const ConfirmedVoxelKey & key) const noexcept
  {
    std::size_t seed = std::hash<std::int64_t>{}(key.x);
    seed ^= std::hash<std::int64_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    seed ^= std::hash<std::int64_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

struct ConfirmedVoxelAccumulator
{
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  int point_count{0};
  int keyframe_observations{0};
};

ConfirmedVoxelKey confirmedVoxelKey(
  const Eigen::Vector3d & point,
  const double voxel_size_m)
{
  return ConfirmedVoxelKey{
    static_cast<std::int64_t>(std::floor(point.x() / voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.y() / voxel_size_m)),
    static_cast<std::int64_t>(std::floor(point.z() / voxel_size_m))};
}

template<typename T>
std::string toString(const T & value)
{
  std::ostringstream stream;
  stream << value;
  return stream.str();
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

const sensor_msgs::msg::PointField * findPointTimeField(
  const sensor_msgs::msg::PointCloud2 & msg,
  const std::string & configured_field)
{
  auto find_field = [&msg](const std::string & name) {
      for (const auto & field : msg.fields) {
        if (field.name == name && fieldWithinPointStep(field, msg.point_step)) {
          return &field;
        }
      }
      return static_cast<const sensor_msgs::msg::PointField *>(nullptr);
    };

  if (configured_field != "auto") {
    return find_field(configured_field);
  }

  const std::array<std::string, 6> candidate_fields = {
    "timestamp",
    "timestamps",
    "time",
    "t",
    "offset_time",
    "stamps"};
  for (const auto & name : candidate_fields) {
    if (const auto * field = find_field(name)) {
      return field;
    }
  }
  return nullptr;
}

double decodePointTimeSeconds(
  const double raw_time,
  const std::string & field_name,
  const double header_time_s,
  const double configured_scale,
  const double configured_offset_s)
{
  if (!std::isfinite(raw_time)) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double decoded_time_s = 0.0;
  if (configured_scale > 0.0) {
    const double scaled = raw_time * configured_scale;
    decoded_time_s = std::abs(scaled) > 1.0e8 ? scaled : header_time_s + scaled;
  } else if (field_name == "offset_time") {
    decoded_time_s = header_time_s + raw_time * 1.0e-9;
  } else {
    const double abs_raw_time = std::abs(raw_time);
    if (abs_raw_time > 1.0e8) {
      decoded_time_s = raw_time;
    } else if (abs_raw_time > 1.0e6) {
      decoded_time_s = header_time_s + raw_time * 1.0e-9;
    } else if (abs_raw_time > 1.0e3) {
      decoded_time_s = header_time_s + raw_time * 1.0e-6;
    } else {
      decoded_time_s = header_time_s + raw_time;
    }
  }
  return decoded_time_s + configured_offset_s;
}

struct PointCloudTimeBoundsNs
{
  std::int64_t start_stamp_ns{0};
  std::int64_t end_stamp_ns{0};
};

double pointTimeComparisonToleranceS(
  const double header_time_s,
  const double start_time_s,
  const double end_time_s)
{
  const double magnitude_s = std::max(
    {1.0, std::abs(header_time_s), std::abs(start_time_s), std::abs(end_time_s)});
  return std::max(
    1.0e-9,
    4.0 * std::numeric_limits<double>::epsilon() * magnitude_s);
}

bool pointTimeBoundsPlausible(
  const double header_time_s,
  const double start_time_s,
  const double end_time_s,
  const double max_scan_duration_s)
{
  if (!std::isfinite(header_time_s) || !std::isfinite(start_time_s) ||
    !std::isfinite(end_time_s) || end_time_s < start_time_s)
  {
    return false;
  }

  const double maximum_duration_s = std::max(0.0, max_scan_duration_s);
  if (maximum_duration_s <= 0.0) {
    return false;
  }
  const double timestamp_tolerance_s = pointTimeComparisonToleranceS(
    header_time_s, start_time_s, end_time_s);
  if (end_time_s - start_time_s > maximum_duration_s + timestamp_tolerance_s) {
    return false;
  }

  // A valid scan header may refer to its start, end, or an instant between
  // them. Neither bound may be farther than one configured scan duration from
  // the header, so a uniformly shifted field cannot gain a second duration of
  // causal look-ahead.
  return start_time_s >= header_time_s - maximum_duration_s - timestamp_tolerance_s &&
         end_time_s <= header_time_s + maximum_duration_s + timestamp_tolerance_s;
}

std::optional<PointCloudTimeBoundsNs> pointCloudTimeBoundsNs(
  const sensor_msgs::msg::PointCloud2 & msg,
  const std::string & configured_field,
  const double configured_scale,
  const double configured_offset_s,
  const double max_scan_duration_s)
{
  if (msg.is_bigendian || msg.width == 0 || msg.height == 0 ||
    msg.point_step == 0 || msg.row_step == 0)
  {
    return std::nullopt;
  }

  const auto * time_field = findPointTimeField(msg, configured_field);
  if (!time_field) {
    return std::nullopt;
  }

  const double header_time_s = rclcpp::Time(msg.header.stamp).seconds();
  double min_time_s = std::numeric_limits<double>::infinity();
  double max_time_s = -std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < msg.height; ++row) {
    for (std::size_t col = 0; col < msg.width; ++col) {
      const std::size_t byte_index =
        row * static_cast<std::size_t>(msg.row_step) +
        col * static_cast<std::size_t>(msg.point_step);
      if (byte_index + msg.point_step > msg.data.size()) {
        continue;
      }
      const auto raw_value =
        readPointFieldAsDouble(msg.data.data() + byte_index, *time_field);
      if (!raw_value) {
        continue;
      }
      const double point_time_s = decodePointTimeSeconds(
        *raw_value,
        time_field->name,
        header_time_s,
        configured_scale,
        configured_offset_s);
      if (std::isfinite(point_time_s)) {
        min_time_s = std::min(min_time_s, point_time_s);
        max_time_s = std::max(max_time_s, point_time_s);
      }
    }
  }

  if (!pointTimeBoundsPlausible(
      header_time_s, min_time_s, max_time_s, max_scan_duration_s))
  {
    return std::nullopt;
  }
  return PointCloudTimeBoundsNs{
    static_cast<std::int64_t>(std::llround(min_time_s * 1.0e9)),
    static_cast<std::int64_t>(std::llround(max_time_s * 1.0e9))};
}

}  // namespace

ScanToMapIcp::ScanToMapIcp(const rclcpp::NodeOptions & options)
: Node("scan_to_map_icp_node", options),
  global_map_(new CloudT),
  latest_local_map_(new CloudT),
  terrain_evidence_map_(new TerrainCloudT),
  terrain_support_evidence_map_(std::make_shared<TerrainSupportVector>()),
  terrain_preview_evidence_map_(new TerrainCloudT)
{
  setupParameters();
  setupRosInterfaces();
  refreshMapPublishSnapshots();

  path_msg_.header.frame_id = map_frame_;

  RCLCPP_INFO(
    get_logger(),
    "scan_to_map_icp_node ready: primary='%s', auxiliary_clouds=%d, odom='%s'",
    primary_cloud_topic_.c_str(),
    static_cast<int>(
      std::count_if(
        auxiliary_cloud_inputs_.begin(),
        auxiliary_cloud_inputs_.end(),
        [](const AuxiliaryCloudInput & input) {return input.enabled;})),
    odom_topic_.c_str());
}

void ScanToMapIcp::setupParameters()
{
  map_frame_ = declare_parameter<std::string>("map_frame", "map");
  odom_frame_ = declare_parameter<std::string>("odom_frame", "odom");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");

  primary_cloud_topic_ = declare_parameter<std::string>(
    "primary_cloud_topic",
    "/slam_3d/input/hesai_center/points_timed");
  auxiliary_cloud_inputs_[0].name = "secondary";
  auxiliary_cloud_inputs_[0].topic = declare_parameter<std::string>(
    "secondary_cloud_topic",
    "/slam_3d/input/hesai_front/points_timed");
  auxiliary_cloud_inputs_[1].name = "tertiary";
  auxiliary_cloud_inputs_[1].topic = declare_parameter<std::string>(
    "tertiary_cloud_topic",
    "/slam_3d/input/hesai_rear/points_timed");
  wheel_odom_topic_ = declare_parameter<std::string>("wheel_odom_topic", "/motor_driver_node/odom");
  imu_topic_ = declare_parameter<std::string>("imu_topic", "/imu/data");
  wheel_odom_expected_frame_ = declare_parameter<std::string>(
    "wheel_odom_expected_frame",
    odom_frame_);
  wheel_odom_expected_child_frame_ = declare_parameter<std::string>(
    "wheel_odom_expected_child_frame",
    base_frame_);
  imu_expected_frame_ = declare_parameter<std::string>(
    "imu_expected_frame",
    "");

  odom_topic_ = declare_parameter<std::string>("odom_topic", "/slam_3d/frontend/odom");
  path_topic_ = declare_parameter<std::string>("path_topic", "/slam_3d/frontend/path");
  local_map_topic_ = declare_parameter<std::string>(
    "local_map_topic",
    "/slam_3d/map/active_submap");
  global_map_topic_ = declare_parameter<std::string>(
    "global_map_topic",
    "/slam_3d/map/global_cloud");
  keyframe_archive_directory_ = declare_parameter<std::string>(
    "keyframe_archive_directory", "");
  keyframe_archive_voxel_leaf_m_ = declare_parameter<double>(
    "keyframe_archive_voxel_leaf_m", 0.22);
  if (!std::isfinite(keyframe_archive_voxel_leaf_m_) || keyframe_archive_voxel_leaf_m_ <= 0.0) {
    throw std::invalid_argument("keyframe_archive_voxel_leaf_m must be finite and positive");
  }
  terrain_evidence_topic_ = declare_parameter<std::string>(
    "terrain_evidence_topic",
    "/slam_3d/terrain/evidence");
  terrain_evidence_colored_topic_ = declare_parameter<std::string>(
    "terrain_evidence_colored_topic",
    "/slam_3d/terrain/evidence_colored");
  terrain_support_topic_ = declare_parameter<std::string>(
    "terrain_support_evidence_topic",
    "/slam_3d/terrain/support_evidence");
  terrain_preview_evidence_topic_ = declare_parameter<std::string>(
    "terrain_preview_evidence_topic",
    "/slam_3d/terrain/preview_evidence");
  pose_graph_marker_topic_ = declare_parameter<std::string>(
    "pose_graph_marker_topic",
    "/slam_3d/backend/pose_graph_markers");
  health_topic_ = declare_parameter<std::string>("health_topic", "/slam_3d/health");
  imu_watermark_topic_ = declare_parameter<std::string>(
    "imu_watermark_topic",
    "/slam_3d/imu_watermark_ns");
  preprocessed_scan_topic_ = declare_parameter<std::string>(
    "preprocessed_scan_topic",
    "/slam_3d/frontend/deskewed_points");

  registration_method_ = declare_parameter<std::string>("registration_method", "gicp");
  mapping_spatial_mode_ = declare_parameter<std::string>("mapping_spatial_mode", "planar");
  auxiliary_cloud_inputs_[0].enabled =
    declare_parameter<bool>("use_secondary_cloud", false);
  auxiliary_cloud_inputs_[1].enabled =
    declare_parameter<bool>("use_tertiary_cloud", false);
  auxiliary_cloud_inputs_[0].registration_only =
    declare_parameter<bool>("secondary_cloud_registration_only", false);
  auxiliary_cloud_inputs_[1].registration_only =
    declare_parameter<bool>("tertiary_cloud_registration_only", false);
  cloud_subscription_reliable_ = declare_parameter<bool>(
    "cloud_subscription_reliable",
    false);
  imu_subscription_reliable_ = declare_parameter<bool>(
    "imu_subscription_reliable",
    false);
  use_tf_transforms_ = declare_parameter<bool>("use_tf_transforms", true);
  reject_cloud_on_tf_failure_ = declare_parameter<bool>(
    "reject_cloud_on_tf_failure",
    true);
  reject_cloud_on_deskew_failure_ = declare_parameter<bool>(
    "reject_cloud_on_deskew_failure",
    false);
  use_wheel_prediction_ = declare_parameter<bool>("use_wheel_prediction", true);
  wheel_odom_require_expected_frames_ = declare_parameter<bool>(
    "wheel_odom_require_expected_frames",
    true);
  imu_require_expected_frame_ = declare_parameter<bool>(
    "imu_require_expected_frame",
    false);
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
  record_tracking_scan_during_shock_ = declare_parameter<bool>(
    "record_tracking_scan_during_shock",
    false);
  place_recognition_enabled_ = declare_parameter<bool>("place_recognition_enabled", false);
  place_recognition_periodic_enabled_ = declare_parameter<bool>(
    "place_recognition_periodic_enabled",
    false);
  place_recognition_require_pose_graph_correction_ = declare_parameter<bool>(
    "place_recognition_require_pose_graph_correction",
    false);
  place_recognition_mapping_cloud_refinement_enabled_ = declare_parameter<bool>(
    "place_recognition_mapping_cloud_refinement_enabled",
    false);
  place_recognition_mapping_refinement_required_for_pose_graph_ =
    declare_parameter<bool>(
    "place_recognition_mapping_refinement_required_for_pose_graph",
    false);
  place_recognition_mapping_refinement_large_pose_jump_enabled_ = declare_parameter<bool>(
    "place_recognition_mapping_refinement_large_pose_jump_enabled",
    false);
  place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled_ =
    declare_parameter<bool>(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled",
    false);
  mapping_cloud_refinement_config_.enabled =
    place_recognition_mapping_cloud_refinement_enabled_;
  mapping_cloud_verified_coarse_config_.enabled =
    place_recognition_mapping_refinement_large_pose_jump_verified_coarse_enabled_;
  pose_graph_correction_enabled_ = declare_parameter<bool>(
    "pose_graph_correction_enabled",
    false);
  pose_graph_solver_ = declare_parameter<std::string>(
    "pose_graph_solver",
    "legacy_relaxation");
  pose_graph_information_mode_ = declare_parameter<std::string>(
    "pose_graph_information_mode",
    "configured_diagonal");
  pose_graph_planar_correction_enabled_ = declare_parameter<bool>(
    "pose_graph_planar_correction_enabled",
    false);
  pose_graph_robust_loop_enabled_ = declare_parameter<bool>(
    "pose_graph_robust_loop_enabled",
    false);
  pose_graph_distributed_loop_seed_enabled_ = declare_parameter<bool>(
    "pose_graph_distributed_loop_seed_enabled",
    false);
  pose_graph_minimum_interval_override_config_.enabled = declare_parameter<bool>(
    "pose_graph_minimum_interval_override_enabled",
    false);
  pose_graph_duplicate_submap_pair_override_config_.enabled = declare_parameter<bool>(
    "pose_graph_duplicate_submap_pair_override_enabled",
    false);
  pose_graph_duplicate_submap_pair_override_config_.max_edges = declare_parameter<int>(
    "pose_graph_duplicate_submap_pair_override_max_edges",
    1);
  pose_graph_minimum_interval_override_single_floor_z_enabled_ = declare_parameter<bool>(
    "pose_graph_minimum_interval_override_single_floor_z_enabled",
    false);
  planar_pose_constraint_enabled_ = declare_parameter<bool>(
    "planar_pose_constraint_enabled",
    false);
  ground_motion_constraint_config_.enabled = declare_parameter<bool>(
    "ground_vehicle_motion_constraint_enabled",
    false);
  gravity_roll_pitch_constraint_enabled_ = declare_parameter<bool>(
    "gravity_roll_pitch_constraint_enabled",
    false);
  imu_relative_roll_pitch_constraint_enabled_ = declare_parameter<bool>(
    "imu_relative_roll_pitch_constraint_enabled",
    false);
  imu_relative_attitude_constraint_config_.enabled =
    imu_relative_roll_pitch_constraint_enabled_;
  map_z_filter_enabled_ = declare_parameter<bool>(
    "map_z_filter_enabled",
    false);
  deskew_enabled_ = declare_parameter<bool>("deskew_enabled", false);
  deskew_wait_for_imu_watermark_ = declare_parameter<bool>(
    "deskew_wait_for_imu_watermark",
    false);
  inter_lidar_motion_compensation_enabled_ = declare_parameter<bool>(
    "inter_lidar_motion_compensation_enabled",
    false);
  inter_lidar_gap_recovery_hypothesis_enabled_ = declare_parameter<bool>(
    "inter_lidar_gap_recovery_hypothesis_enabled",
    false);
  inter_lidar_motion_compensation_source_ = declare_parameter<std::string>(
    "inter_lidar_motion_compensation_source",
    "wheel_odom");
  inter_lidar_motion_failure_policy_ = declare_parameter<std::string>(
    "inter_lidar_motion_failure_policy",
    "drop_auxiliary");
  imu_rotation_prediction_enabled_ = declare_parameter<bool>(
    "imu_rotation_prediction_enabled",
    false);
  imu_rotation_prediction_source_ = declare_parameter<std::string>(
    "imu_rotation_prediction_source",
    "angular_velocity");
  imu_roughness_detection_enabled_ = declare_parameter<bool>(
    "imu_roughness_detection_enabled",
    false);
  imu_shock_policy_enabled_ = declare_parameter<bool>(
    "imu_shock_policy_enabled",
    false);
  robust_registration_enabled_ = declare_parameter<bool>(
    "robust_registration_enabled",
    false);
  registration_gap_relaxation_enabled_ = declare_parameter<bool>(
    "registration_gap_relaxation_enabled",
    false);
  submap_enabled_ = declare_parameter<bool>("submap_enabled", false);
  place_recognition_use_submaps_ = declare_parameter<bool>(
    "place_recognition_use_submaps",
    false);
  publish_tf_ = declare_parameter<bool>("publish_tf", false);
  publish_preprocessed_scan_ = declare_parameter<bool>(
    "publish_preprocessed_scan",
    false);
  crop_box_enabled_ = declare_parameter<bool>("crop_box_enabled", true);

  cloud_sync_wait_for_nearest_candidate_ = declare_parameter<bool>(
    "cloud_sync_wait_for_nearest_candidate",
    false);
  cloud_sync_allow_wall_timeout_fallback_ = declare_parameter<bool>(
    "cloud_sync_allow_wall_timeout_fallback",
    true);
  max_secondary_stamp_delta_s_ = declare_parameter<double>("max_secondary_stamp_delta_s", 0.05);
  max_auxiliary_stamp_delta_s_ = declare_parameter<double>(
    "max_auxiliary_stamp_delta_s",
    max_secondary_stamp_delta_s_);
  cloud_sync_wait_timeout_s_ = declare_parameter<double>(
    "cloud_sync_wait_timeout_s",
    0.05);
  cloud_sync_reorder_window_s_ = declare_parameter<double>(
    "cloud_sync_reorder_window_s",
    0.02);
  cloud_processing_period_s_ = declare_parameter<double>(
    "cloud_processing_period_s",
    0.005);
  executor_thread_count_ = declare_parameter<int>("executor_thread_count", 2);
  if (executor_thread_count_ < 1) {
    throw std::invalid_argument("executor_thread_count must be at least 1");
  }
  min_range_m_ = declare_parameter<double>("min_range_m", 0.5);
  max_range_m_ = declare_parameter<double>("max_range_m", 80.0);
  z_min_m_ = declare_parameter<double>("z_min_m", -5.0);
  z_max_m_ = declare_parameter<double>("z_max_m", 8.0);
  scan_voxel_leaf_m_ = declare_parameter<double>("scan_voxel_leaf_m", 0.25);
  adaptive_scan_voxel_leaf_m_ = declare_parameter<double>("adaptive_scan_voxel_leaf_m", 0.15);
  registration_only_cloud_voxel_leaf_m_ = declare_parameter<double>(
    "registration_only_cloud_voxel_leaf_m",
    0.30);
  registration_only_cloud_max_point_ratio_ = declare_parameter<double>(
    "registration_only_cloud_max_point_ratio",
    0.50);
  map_voxel_leaf_m_ = declare_parameter<double>("map_voxel_leaf_m", 0.35);
  if (declare_parameter<bool>("temporal_static_map_enabled", false)) {
    throw std::invalid_argument(
            "temporal_static_map_enabled moved to slam_3d_map_cleanup; disable the old flag, "
            "set keyframe_archive_directory, export keyframes and run temporal_map_cleanup");
  }
  local_map_radius_m_ = declare_parameter<double>("local_map_radius_m", 35.0);
  local_map_vertical_radius_m_ = declare_parameter<double>(
    "local_map_vertical_radius_m",
    0.0);
  keyframe_translation_m_ = declare_parameter<double>("keyframe_translation_m", 0.8);
  keyframe_rotation_deg_ = declare_parameter<double>("keyframe_rotation_deg", 8.0);
  keyframe_vertical_translation_m_ = declare_parameter<double>(
    "keyframe_vertical_translation_m",
    0.0);
  keyframe_rotation_3d_deg_ = declare_parameter<double>(
    "keyframe_rotation_3d_deg",
    0.0);
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
  wheel_odom_buffer_duration_s_ = declare_parameter<double>(
    "wheel_odom_buffer_duration_s",
    30.0);
  wheel_odom_max_time_delta_s_ = declare_parameter<double>(
    "wheel_odom_max_time_delta_s",
    0.15);
  wheel_odom_max_interpolation_gap_s_ = declare_parameter<double>(
    "wheel_odom_max_interpolation_gap_s",
    0.25);
  wheel_imu_slope_z_prediction_enabled_ = declare_parameter<bool>(
    "wheel_imu_slope_z_prediction_enabled", false);
  wheel_imu_slope_z_prediction_max_pitch_deg_ = declare_parameter<double>(
    "wheel_imu_slope_z_prediction_max_pitch_deg", 25.0);
  wheel_imu_slope_z_prediction_weight_ = declare_parameter<double>(
    "wheel_imu_slope_z_prediction_weight", 0.5);
  wheel_odom_timestamp_reset_threshold_s_ = declare_parameter<double>(
    "wheel_odom_timestamp_reset_threshold_s",
    0.5);
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
  registration_gap_relaxation_min_gap_s_ = declare_parameter<double>(
    "registration_gap_relaxation_min_gap_s",
    0.5);
  registration_gap_relaxation_max_speed_mps_ = declare_parameter<double>(
    "registration_gap_relaxation_max_speed_mps",
    0.0);
  registration_gap_relaxation_max_translation_m_ = declare_parameter<double>(
    "registration_gap_relaxation_max_translation_m",
    0.0);
  registration_gap_relaxation_min_inlier_ratio_ = declare_parameter<double>(
    "registration_gap_relaxation_min_inlier_ratio",
    0.0);
  registration_gap_relaxation_max_inlier_rmse_m_ = declare_parameter<double>(
    "registration_gap_relaxation_max_inlier_rmse_m",
    0.0);
  deskew_point_time_field_ = declare_parameter<std::string>("deskew_point_time_field", "auto");
  deskew_point_time_scale_ = declare_parameter<double>("deskew_point_time_scale", 0.0);
  deskew_point_time_offset_s_ = declare_parameter<double>("deskew_point_time_offset_s", 0.0);
  deskew_max_scan_duration_s_ = declare_parameter<double>("deskew_max_scan_duration_s", 0.2);
  deskew_max_imu_gap_s_ = declare_parameter<double>("deskew_max_imu_gap_s", 0.05);
  deskew_imu_buffer_duration_s_ = declare_parameter<double>("deskew_imu_buffer_duration_s", 5.0);
  deskew_imu_wait_timeout_s_ = declare_parameter<double>(
    "deskew_imu_wait_timeout_s",
    0.25);
  initialization_warmup_duration_s_ = declare_parameter<double>(
    "initialization_warmup_duration_s",
    0.0);
  deskew_min_time_ratio_ = declare_parameter<double>("deskew_min_time_ratio", 0.5);
  deskew_time_bin_s_ = declare_parameter<double>("deskew_time_bin_s", 0.001);
  deskew_angular_velocity_scale_ = declare_parameter<double>(
    "deskew_angular_velocity_scale",
    1.0);
  inter_lidar_motion_compensation_max_dt_s_ = declare_parameter<double>(
    "inter_lidar_motion_compensation_max_dt_s",
    0.05);
  inter_lidar_imu_rotation_scale_ = declare_parameter<double>(
    "inter_lidar_imu_rotation_scale",
    1.0);
  inter_lidar_gap_recovery_hypothesis_min_gap_s_ = declare_parameter<double>(
    "inter_lidar_gap_recovery_hypothesis_min_gap_s",
    0.5);
  inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_ =
    declare_parameter<double>(
    "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio",
    0.02);
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
  imu_orientation_prediction_max_time_delta_s_ = declare_parameter<double>(
    "imu_orientation_prediction_max_time_delta_s",
    0.05);
  imu_relative_attitude_max_time_delta_s_ = declare_parameter<double>(
    "imu_relative_attitude_max_time_delta_s",
    0.05);
  imu_relative_attitude_constraint_config_.max_roll_pitch_correction_deg =
    declare_parameter<double>(
    "imu_relative_roll_pitch_max_correction_deg",
    0.0);
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
  imu_shock_hold_duration_s_ = declare_parameter<double>(
    "imu_shock_hold_duration_s",
    0.5);
  imu_shock_recovery_stable_scans_ = declare_parameter<int>(
    "imu_shock_recovery_stable_scans",
    3);
  suppress_constant_velocity_on_imu_shock_ = declare_parameter<bool>(
    "suppress_constant_velocity_on_imu_shock",
    true);
  omit_shock_poses_from_path_ = declare_parameter<bool>(
    "omit_shock_poses_from_path",
    true);
  imu_shock_policy_config_.enabled =
    imu_roughness_detection_enabled_ && imu_shock_policy_enabled_;
  imu_shock_policy_config_.hold_duration_s = imu_shock_hold_duration_s_;
  imu_shock_policy_config_.recovery_stable_scans = imu_shock_recovery_stable_scans_;
  keyframe_quarantine_correction_translation_m_ = declare_parameter<double>(
    "keyframe_quarantine_correction_translation_m",
    0.8);
  place_recognition_registration_only_keyframe_quarantine_correction_translation_m_ =
    declare_parameter<double>(
    "place_recognition_registration_only_keyframe_quarantine_correction_translation_m",
    0.0);
  place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m",
    0.0);
  keyframe_quarantine_correction_rotation_deg_ = declare_parameter<double>(
    "keyframe_quarantine_correction_rotation_deg",
    15.0);
  place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg",
    0.0);
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
  place_recognition_mapping_refinement_large_pose_jump_max_translation_m_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_max_translation_m",
    0.0);
  place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg",
    0.0);
  place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance",
    0.0);
  mapping_cloud_verified_coarse_config_.min_inlier_ratio = declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_inlier_ratio",
    0.0);
  mapping_cloud_verified_coarse_config_.max_inlier_rmse_m = declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_inlier_rmse_m",
    0.0);
  mapping_cloud_verified_coarse_config_.min_target_coverage_ratio =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_target_coverage_ratio",
    0.0);
  mapping_cloud_verified_coarse_config_.max_score_regression_ratio =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_score_regression_ratio",
    0.0);
  place_recognition_max_yaw_offset_deg_ = declare_parameter<double>(
    "place_recognition_max_yaw_offset_deg",
    180.0);
  place_recognition_max_vertical_separation_m_ = declare_parameter<double>(
    "place_recognition_max_vertical_separation_m",
    0.0);
  place_recognition_registration_only_vertical_recovery_m_ = declare_parameter<double>(
    "place_recognition_registration_only_vertical_recovery_m",
    0.0);
  place_recognition_registration_only_same_layer_xy_max_distance_m_ =
    declare_parameter<double>(
    "place_recognition_registration_only_same_layer_xy_max_distance_m",
    0.0);
  place_recognition_registration_only_max_pose_jump_m_ = declare_parameter<double>(
    "place_recognition_registration_only_max_pose_jump_m",
    0.0);
  place_recognition_submap_consistency_max_translation_m_ = declare_parameter<double>(
    "place_recognition_submap_consistency_max_translation_m",
    0.0);
  place_recognition_registration_only_consistency_max_translation_m_ =
    declare_parameter<double>(
    "place_recognition_registration_only_consistency_max_translation_m",
    0.0);
  place_recognition_registration_only_consistency_max_gap_s_ = declare_parameter<double>(
    "place_recognition_registration_only_consistency_max_gap_s",
    0.0);
  place_recognition_submap_consistency_max_rotation_deg_ = declare_parameter<double>(
    "place_recognition_submap_consistency_max_rotation_deg",
    0.0);
  place_recognition_mapping_refinement_quality_max_correspondence_m_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_quality_max_correspondence_m",
    0.8);
  place_recognition_mapping_refinement_target_confirmation_voxel_m_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_target_confirmation_voxel_m",
    0.35);
  mapping_cloud_refinement_config_.min_inlier_ratio = declare_parameter<double>(
    "place_recognition_mapping_refinement_min_inlier_ratio",
    0.20);
  place_recognition_mapping_refinement_min_target_coverage_ratio_ =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_min_target_coverage_ratio",
    0.0);
  mapping_cloud_refinement_config_.min_target_coverage_ratio =
    place_recognition_mapping_refinement_min_target_coverage_ratio_;
  mapping_cloud_refinement_config_.max_inlier_rmse_m = declare_parameter<double>(
    "place_recognition_mapping_refinement_max_inlier_rmse_m",
    0.30);
  mapping_cloud_refinement_config_.max_translation_m = declare_parameter<double>(
    "place_recognition_mapping_refinement_max_translation_m",
    0.75);
  mapping_cloud_refinement_config_.max_rotation_deg = declare_parameter<double>(
    "place_recognition_mapping_refinement_max_rotation_deg",
    5.0);
  mapping_cloud_refinement_config_.max_hessian_condition = declare_parameter<double>(
    "place_recognition_mapping_refinement_max_hessian_condition",
    0.0);
  mapping_cloud_refinement_config_.min_score_improvement_ratio = declare_parameter<double>(
    "place_recognition_mapping_refinement_min_score_improvement_ratio",
    0.05);
  mapping_cloud_refinement_config_.max_primary_score_regression_ratio =
    declare_parameter<double>(
    "place_recognition_mapping_refinement_max_primary_score_regression_ratio",
    0.05);
  submap_max_travel_m_ = declare_parameter<double>(
    "submap_max_travel_m",
    20.0);
  pose_graph_correction_max_translation_m_ = declare_parameter<double>(
    "pose_graph_correction_max_translation_m",
    5.0);
  pose_graph_correction_max_rotation_deg_ = declare_parameter<double>(
    "pose_graph_correction_max_rotation_deg",
    45.0);
  pose_graph_correction_min_interval_s_ = declare_parameter<double>(
    "pose_graph_correction_min_interval_s",
    1.0);
  pose_graph_minimum_interval_override_config_.max_descriptor_distance =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_max_descriptor_distance",
    0.12);
  pose_graph_minimum_interval_override_config_.min_inlier_ratio =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_min_inlier_ratio",
    0.90);
  pose_graph_minimum_interval_override_config_.max_inlier_rmse_m =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_max_inlier_rmse_m",
    0.15);
  pose_graph_minimum_interval_override_config_.min_target_coverage_ratio =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_min_target_coverage_ratio",
    0.50);
  pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_ =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_single_floor_xy_max_distance_m",
    1.5);
  pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_ =
    declare_parameter<double>(
    "pose_graph_minimum_interval_override_single_floor_z_weight_multiplier",
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
  pose_graph_information_config_.relative_eigen_threshold = declare_parameter<double>(
    "pose_graph_information_relative_eigen_threshold",
    pose_graph_information_config_.relative_eigen_threshold);
  pose_graph_information_config_.strong_eigenvalue_ratio = declare_parameter<double>(
    "pose_graph_information_strong_eigenvalue_ratio",
    pose_graph_information_config_.strong_eigenvalue_ratio);
  pose_graph_information_config_.weak_information_ratio = declare_parameter<double>(
    "pose_graph_information_weak_ratio",
    pose_graph_information_config_.weak_information_ratio);
  pose_graph_loop_cauchy_scale_ = declare_parameter<double>(
    "pose_graph_loop_cauchy_scale",
    1.0);
  pose_graph_optimizer_config_.max_iterations_per_stage = declare_parameter<int>(
    "pose_graph_ceres_max_iterations_per_stage",
    20);
  pose_graph_optimizer_config_.robust_scale_factors =
    declare_parameter<std::vector<double>>(
    "pose_graph_ceres_robust_scale_factors",
    std::vector<double>{4.0, 2.0, 1.0});
  pose_graph_optimizer_config_.dcs_enabled = declare_parameter<bool>(
    "pose_graph_ceres_dcs_enabled",
    true);
  pose_graph_optimizer_config_.dcs_start_stage = declare_parameter<int>(
    "pose_graph_ceres_dcs_start_stage",
    0);
  pose_graph_optimizer_config_.dcs_phi = declare_parameter<double>(
    "pose_graph_ceres_dcs_phi",
    5.0);
  pose_graph_optimizer_config_.dcs_min_scale = declare_parameter<double>(
    "pose_graph_ceres_dcs_min_scale",
    0.05);
  pose_graph_optimizer_config_.function_tolerance = declare_parameter<double>(
    "pose_graph_ceres_function_tolerance",
    1.0e-6);
  pose_graph_optimizer_config_.gradient_tolerance = declare_parameter<double>(
    "pose_graph_ceres_gradient_tolerance",
    1.0e-10);
  pose_graph_optimizer_config_.parameter_tolerance = declare_parameter<double>(
    "pose_graph_ceres_parameter_tolerance",
    1.0e-8);
  pose_graph_optimizer_config_.num_threads = declare_parameter<int>(
    "pose_graph_ceres_num_threads",
    1);
  pose_graph_optimizer_config_.use_nonmonotonic_steps = declare_parameter<bool>(
    "pose_graph_ceres_use_nonmonotonic_steps",
    false);
  pose_graph_optimizer_config_.robust_loop_enabled = pose_graph_robust_loop_enabled_;
  pose_graph_optimizer_config_.loop_cauchy_scale = pose_graph_loop_cauchy_scale_;
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
  ground_motion_constraint_config_.nominal_max_body_z_step_m =
    declare_parameter<double>(
    "ground_vehicle_nominal_max_body_z_step_m",
    0.0);
  ground_motion_constraint_config_.roughness_max_body_z_step_m =
    declare_parameter<double>(
    "ground_vehicle_roughness_max_body_z_step_m",
    0.0);
  gravity_accel_low_pass_alpha_ = declare_parameter<double>(
    "gravity_accel_low_pass_alpha",
    0.02);
  gravity_accel_time_constant_s_ = declare_parameter<double>(
    "gravity_accel_time_constant_s",
    0.0);
  gravity_accel_scale_ = declare_parameter<double>(
    "gravity_accel_scale",
    1.0);
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

  terrain_config_.enabled = declare_parameter<bool>(
    "terrain_evidence_enabled",
    false);
  terrain_config_.azimuth_bins = declare_parameter<int>(
    "terrain_azimuth_bins",
    120);
  terrain_config_.radial_bin_size_m = declare_parameter<double>(
    "terrain_radial_bin_size_m",
    0.25);
  terrain_config_.min_range_m = declare_parameter<double>(
    "terrain_min_range_m",
    0.8);
  terrain_config_.max_range_m = declare_parameter<double>(
    "terrain_max_range_m",
    12.0);
  terrain_config_.seed_max_range_m = declare_parameter<double>(
    "terrain_seed_max_range_m",
    3.0);
  terrain_config_.classification_voxel_size_m = declare_parameter<double>(
    "terrain_classification_voxel_size_m",
    0.10);
  terrain_config_.min_points_per_bin = declare_parameter<int>(
    "terrain_min_points_per_bin",
    2);
  terrain_config_.floor_quantile = declare_parameter<double>(
    "terrain_floor_quantile",
    0.20);
  terrain_config_.seed_reference_height_m = declare_parameter<double>(
    "terrain_seed_reference_height_m",
    0.0);
  terrain_config_.seed_height_cluster_tolerance_m = declare_parameter<double>(
    "terrain_seed_height_cluster_tolerance_m",
    0.10);
  terrain_config_.seed_min_cluster_bins = declare_parameter<int>(
    "terrain_seed_min_cluster_bins",
    3);
  terrain_config_.max_profile_height_above_base_m = declare_parameter<double>(
    "terrain_max_profile_height_above_base_m",
    0.50);
  terrain_config_.max_seed_height_delta_m = declare_parameter<double>(
    "terrain_max_seed_height_delta_m",
    0.45);
  terrain_config_.ground_band_m = declare_parameter<double>(
    "terrain_ground_band_m",
    0.10);
  terrain_config_.max_adjacent_gap_m = declare_parameter<double>(
    "terrain_max_adjacent_gap_m",
    0.55);
  terrain_config_.floor_max_slope_deg = declare_parameter<double>(
    "terrain_floor_max_slope_deg",
    4.0);
  terrain_config_.ramp_min_slope_deg = declare_parameter<double>(
    "terrain_ramp_min_slope_deg",
    5.0);
  terrain_config_.ramp_max_slope_deg = declare_parameter<double>(
    "terrain_ramp_max_slope_deg",
    22.0);
  terrain_config_.ramp_min_run_bins = declare_parameter<int>(
    "terrain_ramp_min_run_bins",
    3);
  terrain_config_.ramp_min_run_length_m = declare_parameter<double>(
    "terrain_ramp_min_run_length_m",
    0.50);
  terrain_config_.ramp_plane_validation_enabled = declare_parameter<bool>(
    "terrain_ramp_plane_validation_enabled",
    false);
  terrain_config_.ramp_plane_neighborhood_radius_m = declare_parameter<double>(
    "terrain_ramp_plane_neighborhood_radius_m",
    0.75);
  terrain_config_.ramp_plane_min_points = declare_parameter<int>(
    "terrain_ramp_plane_min_points",
    12);
  terrain_config_.ramp_plane_min_length_m = declare_parameter<double>(
    "terrain_ramp_plane_min_length_m",
    0.50);
  terrain_config_.ramp_plane_min_width_m = declare_parameter<double>(
    "terrain_ramp_plane_min_width_m",
    0.20);
  terrain_config_.ramp_plane_max_rmse_m = declare_parameter<double>(
    "terrain_ramp_plane_max_rmse_m",
    0.05);
  terrain_config_.ramp_plane_max_outlier_distance_m = declare_parameter<double>(
    "terrain_ramp_plane_max_outlier_distance_m",
    0.08);
  terrain_config_.ramp_plane_min_inlier_ratio = declare_parameter<double>(
    "terrain_ramp_plane_min_inlier_ratio",
    0.70);
  terrain_config_.roughness_preview_enabled = declare_parameter<bool>(
    "terrain_roughness_preview_enabled",
    true);
  terrain_config_.roughness_min_height_m = declare_parameter<double>(
    "terrain_roughness_min_height_m",
    0.025);
  terrain_config_.roughness_max_score = declare_parameter<double>(
    "terrain_roughness_max_score",
    4.0);
  terrain_config_.step_min_height_m = declare_parameter<double>(
    "terrain_step_min_height_m",
    0.08);
  terrain_config_.step_max_height_m = declare_parameter<double>(
    "terrain_step_max_height_m",
    0.30);
  terrain_config_.drop_min_height_m = declare_parameter<double>(
    "terrain_drop_min_height_m",
    0.12);
  terrain_config_.drop_max_height_m = declare_parameter<double>(
    "terrain_drop_max_height_m",
    0.60);
  terrain_config_.transition_support_bins = declare_parameter<int>(
    "terrain_transition_support_bins",
    1);
  terrain_config_.staircase_enabled = declare_parameter<bool>(
    "terrain_staircase_enabled",
    true);
  terrain_config_.staircase_min_risers = declare_parameter<int>(
    "terrain_staircase_min_risers",
    3);
  terrain_config_.staircase_min_riser_height_m = declare_parameter<double>(
    "terrain_staircase_min_riser_height_m",
    0.11);
  terrain_config_.staircase_max_riser_height_m = declare_parameter<double>(
    "terrain_staircase_max_riser_height_m",
    0.30);
  terrain_config_.staircase_min_tread_depth_m = declare_parameter<double>(
    "terrain_staircase_min_tread_depth_m",
    0.15);
  terrain_config_.staircase_max_tread_depth_m = declare_parameter<double>(
    "terrain_staircase_max_tread_depth_m",
    0.45);
  terrain_config_.staircase_max_riser_height_deviation_m = declare_parameter<double>(
    "terrain_staircase_max_riser_height_deviation_m",
    0.06);
  terrain_config_.staircase_max_tread_depth_deviation_m = declare_parameter<double>(
    "terrain_staircase_max_tread_depth_deviation_m",
    0.15);
  terrain_config_.staircase_min_slope_deg = declare_parameter<double>(
    "terrain_staircase_min_slope_deg",
    20.0);
  terrain_config_.staircase_max_slope_deg = declare_parameter<double>(
    "terrain_staircase_max_slope_deg",
    55.0);
  terrain_config_.staircase_max_profile_height_above_base_m = declare_parameter<double>(
    "terrain_staircase_max_profile_height_above_base_m",
    2.0);
  terrain_config_.require_gravity_alignment = declare_parameter<bool>(
    "terrain_require_gravity_alignment",
    true);
  terrain_accumulator_config_.voxel_size_m = declare_parameter<double>(
    "terrain_evidence_voxel_size_m",
    0.20);
  terrain_accumulator_config_.min_confirmations = declare_parameter<int>(
    "terrain_min_confirmations",
    2);
  terrain_accumulator_config_.min_viewpoint_distance_m = declare_parameter<double>(
    "terrain_confirmation_min_viewpoint_distance_m",
    0.30);
  terrain_accumulator_config_.min_view_label_confidence = declare_parameter<double>(
    "terrain_min_view_label_confidence",
    0.60);
  terrain_accumulator_config_.min_label_confidence = declare_parameter<double>(
    "terrain_min_label_confidence",
    0.60);
  terrain_accumulator_config_.probable_support_min_confidence = declare_parameter<double>(
    "terrain_probable_support_min_confidence",
    0.60);
  terrain_accumulator_config_.probable_support_max_roughness_score =
    declare_parameter<double>(
    "terrain_probable_support_max_roughness_score",
    1.50);
  terrain_accumulator_config_.probable_support_require_confirmed_connectivity =
    declare_parameter<bool>(
    "terrain_probable_support_require_confirmed_connectivity",
    true);
  terrain_accumulator_config_.probable_support_max_anchor_distance_m =
    declare_parameter<double>(
    "terrain_probable_support_max_anchor_distance_m",
    0.45);
  terrain_accumulator_config_.probable_support_max_neighbor_height_delta_m =
    declare_parameter<double>(
    "terrain_probable_support_max_neighbor_height_delta_m",
    0.16);
  terrain_accumulator_config_.ramp_traversability_scale = declare_parameter<double>(
    "terrain_ramp_traversability_scale",
    0.75);
  terrain_accumulator_.configure(terrain_accumulator_config_);
  terrain_preview_config_.enabled = declare_parameter<bool>(
    "terrain_traversal_confirmation_enabled",
    false);
  terrain_preview_config_.voxel_size_m = terrain_accumulator_config_.voxel_size_m;
  terrain_preview_config_.association_radius_m = declare_parameter<double>(
    "terrain_traversal_association_radius_m",
    0.55);
  terrain_preview_config_.exit_radius_m = declare_parameter<double>(
    "terrain_traversal_exit_radius_m",
    0.75);
  terrain_preview_config_.max_vertical_delta_m = declare_parameter<double>(
    "terrain_traversal_max_vertical_delta_m",
    0.40);
  terrain_preview_config_.max_sample_gap_s = declare_parameter<double>(
    "terrain_traversal_max_sample_gap_s",
    0.50);
  terrain_preview_config_.max_swept_segment_length_m = declare_parameter<double>(
    "terrain_traversal_max_swept_segment_length_m",
    2.0);
  terrain_preview_config_.min_ahead_distance_m = declare_parameter<double>(
    "terrain_preview_min_ahead_distance_m",
    0.75);
  terrain_preview_config_.max_lateral_distance_m = declare_parameter<double>(
    "terrain_preview_max_lateral_distance_m",
    1.0);
  terrain_preview_config_.min_label_confidence = declare_parameter<double>(
    "terrain_traversal_min_label_confidence",
    terrain_accumulator_config_.min_label_confidence);
  terrain_preview_config_.roughness_score_gate = declare_parameter<double>(
    "terrain_preview_roughness_score_gate",
    1.0);
  terrain_preview_config_.ramp_shock_prior = declare_parameter<double>(
    "terrain_preview_ramp_shock_prior",
    0.25);
  terrain_preview_config_.positive_step_shock_prior = declare_parameter<double>(
    "terrain_preview_positive_step_shock_prior",
    1.0);
  terrain_preview_config_.min_passage_samples = declare_parameter<int>(
    "terrain_traversal_min_passage_samples",
    2);
  terrain_preview_config_.min_passage_duration_s = declare_parameter<double>(
    "terrain_traversal_min_passage_duration_s",
    0.05);
  terrain_preview_tracker_.configure(terrain_preview_config_);

  min_scan_points_ = declare_parameter<int>("min_scan_points", 200);
  adaptive_scan_voxel_min_points_ = declare_parameter<int>("adaptive_scan_voxel_min_points", 900);
  registration_only_cloud_max_points_ = declare_parameter<int>(
    "registration_only_cloud_max_points",
    5000);
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
  place_recognition_submap_min_consecutive_matches_ = declare_parameter<int>(
    "place_recognition_submap_min_consecutive_matches",
    1);
  place_recognition_mapping_refinement_min_source_points_ = declare_parameter<int>(
    "place_recognition_mapping_refinement_min_source_points",
    200);
  place_recognition_mapping_refinement_min_target_points_ = declare_parameter<int>(
    "place_recognition_mapping_refinement_min_target_points",
    1000);
  place_recognition_mapping_refinement_min_target_keyframe_observations_ =
    declare_parameter<int>(
    "place_recognition_mapping_refinement_min_target_keyframe_observations",
    2);
  mapping_cloud_refinement_config_.min_source_points =
    place_recognition_mapping_refinement_min_source_points_;
  mapping_cloud_refinement_config_.min_target_points =
    place_recognition_mapping_refinement_min_target_points_;
  mapping_cloud_refinement_config_.min_observable_rank = declare_parameter<int>(
    "place_recognition_mapping_refinement_min_observable_rank",
    0);
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
  cloud_sync_queue_size_ = declare_parameter<int>("cloud_sync_queue_size", 10);
  imu_subscription_depth_ = declare_parameter<int>("imu_subscription_depth", 400);
  wheel_odom_buffer_max_samples_ = declare_parameter<int>(
    "wheel_odom_buffer_max_samples",
    5000);
  max_icp_iterations_ = declare_parameter<int>("max_icp_iterations", 40);
  robust_registration_refinement_iterations_ = declare_parameter<int>(
    "robust_registration_refinement_iterations",
    15);
  max_keyframes_ = declare_parameter<int>("max_keyframes", 500);
  max_local_keyframes_ = declare_parameter<int>("max_local_keyframes", 80);
  submap_max_keyframes_ = declare_parameter<int>("submap_max_keyframes", 30);
  submap_overlap_keyframes_ = declare_parameter<int>("submap_overlap_keyframes", 5);
  max_local_submaps_ = declare_parameter<int>("max_local_submaps", 3);
  max_path_poses_ = declare_parameter<int>("max_path_poses", 10000);

  surfel_config_.voxel_size = declare_parameter<double>(
    "surfel_voxel_size_m",
    surfel_config_.voxel_size);
  surfel_config_.min_points_per_voxel = static_cast<std::size_t>(std::max(
      0,
      static_cast<int>(declare_parameter<int>(
        "surfel_min_points_per_voxel",
        static_cast<int>(surfel_config_.min_points_per_voxel)))));
  surfel_config_.max_planarity_ratio = declare_parameter<double>(
    "surfel_max_planarity_ratio",
    surfel_config_.max_planarity_ratio);
  surfel_config_.min_tangent_variance = declare_parameter<double>(
    "surfel_min_tangent_variance",
    surfel_config_.min_tangent_variance);
  surfel_config_.max_plane_distance = declare_parameter<double>(
    "surfel_max_plane_distance_m",
    surfel_config_.max_plane_distance);
  surfel_config_.max_center_distance = declare_parameter<double>(
    "surfel_max_center_distance_m",
    surfel_config_.max_center_distance);
  surfel_config_.min_correspondences = static_cast<std::size_t>(std::max(
      0,
      static_cast<int>(declare_parameter<int>(
        "surfel_min_correspondences",
        static_cast<int>(surfel_config_.min_correspondences)))));
  surfel_config_.min_correspondence_ratio = declare_parameter<double>(
    "surfel_min_correspondence_ratio",
    surfel_config_.min_correspondence_ratio);
  surfel_config_.huber_delta = declare_parameter<double>(
    "surfel_huber_delta_m",
    surfel_config_.huber_delta);
  surfel_config_.max_iterations = declare_parameter<int>(
    "surfel_max_iterations",
    surfel_config_.max_iterations);
  surfel_config_.max_backtracking_steps = declare_parameter<int>(
    "surfel_max_backtracking_steps",
    surfel_config_.max_backtracking_steps);
  surfel_config_.backtracking_factor = declare_parameter<double>(
    "surfel_backtracking_factor",
    surfel_config_.backtracking_factor);
  surfel_config_.rotation_convergence_rad = declare_parameter<double>(
    "surfel_rotation_convergence_rad",
    surfel_config_.rotation_convergence_rad);
  surfel_config_.translation_convergence_m = declare_parameter<double>(
    "surfel_translation_convergence_m",
    surfel_config_.translation_convergence_m);
  surfel_config_.relative_cost_convergence = declare_parameter<double>(
    "surfel_relative_cost_convergence",
    surfel_config_.relative_cost_convergence);
  surfel_config_.max_rotation_step_rad = declare_parameter<double>(
    "surfel_max_rotation_step_rad",
    surfel_config_.max_rotation_step_rad);
  surfel_config_.max_translation_step_m = declare_parameter<double>(
    "surfel_max_translation_step_m",
    surfel_config_.max_translation_step_m);
  surfel_config_.rotation_scale = declare_parameter<double>(
    "surfel_rotation_scale_m_per_rad",
    surfel_config_.rotation_scale);
  surfel_config_.relative_eigen_threshold = declare_parameter<double>(
    "surfel_relative_eigen_threshold",
    surfel_config_.relative_eigen_threshold);
  surfel_config_.min_observable_rank = declare_parameter<int>(
    "surfel_min_observable_rank",
    surfel_config_.min_observable_rank);
  surfel_config_.max_quality_rmse = declare_parameter<double>(
    "surfel_max_quality_rmse_m",
    surfel_config_.max_quality_rmse);
  surfel_config_.max_quality_p95 = declare_parameter<double>(
    "surfel_max_quality_p95_m",
    surfel_config_.max_quality_p95);

  if (
    registration_method_ != "icp" &&
    registration_method_ != "gicp" &&
    registration_method_ != "voxel_surfel")
  {
    RCLCPP_WARN(
      get_logger(),
      "Unknown registration_method '%s'. Falling back to gicp.",
      registration_method_.c_str());
    registration_method_ = "gicp";
  }
  const auto parsed_spatial_mode = parseMappingSpatialMode(mapping_spatial_mode_);
  if (!parsed_spatial_mode.has_value()) {
    RCLCPP_WARN(
      get_logger(),
      "Unknown mapping_spatial_mode '%s'. Falling back to planar.",
      mapping_spatial_mode_.c_str());
    mapping_spatial_mode_ = "planar";
    spatial_mapping_config_.mode = MappingSpatialMode::kPlanar;
  } else {
    spatial_mapping_config_.mode = *parsed_spatial_mode;
  }
  spatial_mapping_config_.keyframe_translation_m = keyframe_translation_m_;
  spatial_mapping_config_.keyframe_yaw_deg = keyframe_rotation_deg_;
  spatial_mapping_config_.keyframe_vertical_translation_m =
    keyframe_vertical_translation_m_;
  spatial_mapping_config_.keyframe_rotation_3d_deg = keyframe_rotation_3d_deg_;
  spatial_mapping_config_.local_map_radius_m = local_map_radius_m_;
  spatial_mapping_config_.local_map_vertical_radius_m = local_map_vertical_radius_m_;
  submap_policy_config_.max_keyframes = submap_max_keyframes_;
  submap_policy_config_.max_travel_m = submap_max_travel_m_;
  submap_policy_config_.overlap_keyframes = submap_overlap_keyframes_;
  if (
    imu_rotation_prediction_source_ != "angular_velocity" &&
    imu_rotation_prediction_source_ != "orientation")
  {
    RCLCPP_WARN(
      get_logger(),
      "Unknown imu_rotation_prediction_source '%s'. Falling back to angular_velocity.",
      imu_rotation_prediction_source_.c_str());
    imu_rotation_prediction_source_ = "angular_velocity";
  }
  if (!std::isfinite(gravity_accel_scale_) ||
    std::abs(gravity_accel_scale_) <= std::numeric_limits<double>::epsilon())
  {
    RCLCPP_WARN(
      get_logger(),
      "gravity_accel_scale must be finite and non-zero. Falling back to 1.0.");
    gravity_accel_scale_ = 1.0;
  }
  auto sanitize_ground_motion_limit =
    [this](double & limit, const char * parameter_name) {
      if (std::isfinite(limit) && limit >= 0.0) {
        return;
      }
      RCLCPP_WARN(
        get_logger(),
        "%s must be finite and non-negative. Falling back to 0.0.",
        parameter_name);
      limit = 0.0;
    };
  sanitize_ground_motion_limit(
    ground_motion_constraint_config_.nominal_max_body_z_step_m,
    "ground_vehicle_nominal_max_body_z_step_m");
  sanitize_ground_motion_limit(
    ground_motion_constraint_config_.roughness_max_body_z_step_m,
    "ground_vehicle_roughness_max_body_z_step_m");
  if (
    ground_motion_constraint_config_.roughness_max_body_z_step_m <
    ground_motion_constraint_config_.nominal_max_body_z_step_m)
  {
    RCLCPP_WARN(
      get_logger(),
      "ground_vehicle_roughness_max_body_z_step_m must be greater than or "
      "equal to the nominal limit. Using the nominal limit.");
    ground_motion_constraint_config_.roughness_max_body_z_step_m =
      ground_motion_constraint_config_.nominal_max_body_z_step_m;
  }
  auto require_finite_nonnegative =
    [](const double value, const char * parameter_name) {
      if (!std::isfinite(value) || value < 0.0) {
        throw std::invalid_argument(
                std::string(parameter_name) +
                " must be finite and non-negative");
      }
    };
  require_finite_nonnegative(
    registration_gap_relaxation_min_gap_s_,
    "registration_gap_relaxation_min_gap_s");
  require_finite_nonnegative(
    registration_gap_relaxation_max_speed_mps_,
    "registration_gap_relaxation_max_speed_mps");
  require_finite_nonnegative(
    registration_gap_relaxation_max_translation_m_,
    "registration_gap_relaxation_max_translation_m");
  require_finite_nonnegative(
    registration_gap_relaxation_min_inlier_ratio_,
    "registration_gap_relaxation_min_inlier_ratio");
  require_finite_nonnegative(
    registration_gap_relaxation_max_inlier_rmse_m_,
    "registration_gap_relaxation_max_inlier_rmse_m");
  require_finite_nonnegative(
    inter_lidar_motion_compensation_max_dt_s_,
    "inter_lidar_motion_compensation_max_dt_s");
  require_finite_nonnegative(
    imu_relative_attitude_max_time_delta_s_,
    "imu_relative_attitude_max_time_delta_s");
  require_finite_nonnegative(
    initialization_warmup_duration_s_,
    "initialization_warmup_duration_s");
  require_finite_nonnegative(
    imu_relative_attitude_constraint_config_.max_roll_pitch_correction_deg,
    "imu_relative_roll_pitch_max_correction_deg");
  if (!std::isfinite(registration_only_cloud_voxel_leaf_m_) ||
    registration_only_cloud_voxel_leaf_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "registration_only_cloud_voxel_leaf_m must be finite and greater than zero");
  }
  if (!std::isfinite(registration_only_cloud_max_point_ratio_) ||
    registration_only_cloud_max_point_ratio_ <= 0.0)
  {
    throw std::invalid_argument(
            "registration_only_cloud_max_point_ratio must be finite and greater than zero");
  }
  if (registration_only_cloud_max_points_ < 0) {
    throw std::invalid_argument(
            "registration_only_cloud_max_points must be non-negative");
  }
  if (!std::isfinite(deskew_max_scan_duration_s_) ||
    deskew_max_scan_duration_s_ <= 0.0)
  {
    throw std::invalid_argument(
            "deskew_max_scan_duration_s must be finite and greater than zero");
  }
  if (
    !std::isfinite(inter_lidar_imu_rotation_scale_) ||
    std::abs(inter_lidar_imu_rotation_scale_) <= std::numeric_limits<double>::epsilon())
  {
    throw std::invalid_argument(
            "inter_lidar_imu_rotation_scale must be finite and non-zero");
  }
  require_finite_nonnegative(
    inter_lidar_gap_recovery_hypothesis_min_gap_s_,
    "inter_lidar_gap_recovery_hypothesis_min_gap_s");
  require_finite_nonnegative(
    inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_,
    "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio");
  if (inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_ > 1.0) {
    throw std::invalid_argument(
            "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio "
            "must be <= 1.0");
  }
  if (registration_gap_relaxation_min_inlier_ratio_ > 1.0) {
    throw std::invalid_argument(
            "registration_gap_relaxation_min_inlier_ratio must be <= 1.0");
  }
  require_finite_nonnegative(
    pose_graph_correction_max_translation_m_,
    "pose_graph_correction_max_translation_m");
  require_finite_nonnegative(
    pose_graph_correction_max_rotation_deg_,
    "pose_graph_correction_max_rotation_deg");
  require_finite_nonnegative(
    pose_graph_correction_min_interval_s_,
    "pose_graph_correction_min_interval_s");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_config_.max_descriptor_distance,
    "pose_graph_minimum_interval_override_max_descriptor_distance");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_config_.min_inlier_ratio,
    "pose_graph_minimum_interval_override_min_inlier_ratio");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_config_.max_inlier_rmse_m,
    "pose_graph_minimum_interval_override_max_inlier_rmse_m");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_config_.min_target_coverage_ratio,
    "pose_graph_minimum_interval_override_min_target_coverage_ratio");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_,
    "pose_graph_minimum_interval_override_single_floor_xy_max_distance_m");
  require_finite_nonnegative(
    pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_,
    "pose_graph_minimum_interval_override_single_floor_z_weight_multiplier");
  if (pose_graph_minimum_interval_override_config_.min_inlier_ratio > 1.0) {
    throw std::invalid_argument(
            "pose_graph_minimum_interval_override_min_inlier_ratio must be <= 1.0");
  }
  if (pose_graph_minimum_interval_override_config_.min_target_coverage_ratio > 1.0) {
    throw std::invalid_argument(
            "pose_graph_minimum_interval_override_min_target_coverage_ratio must be <= 1.0");
  }
  if (pose_graph_minimum_interval_override_config_.enabled &&
    (pose_graph_minimum_interval_override_config_.max_descriptor_distance <= 0.0 ||
    pose_graph_minimum_interval_override_config_.max_inlier_rmse_m <= 0.0 ||
    pose_graph_minimum_interval_override_config_.min_inlier_ratio <= 0.0 ||
    pose_graph_minimum_interval_override_config_.min_target_coverage_ratio <= 0.0))
  {
    throw std::invalid_argument(
            "pose_graph minimum-interval override quality thresholds must be greater than zero");
  }
  if (pose_graph_duplicate_submap_pair_override_config_.max_edges < 1 ||
    (pose_graph_duplicate_submap_pair_override_config_.enabled &&
    pose_graph_duplicate_submap_pair_override_config_.max_edges < 2))
  {
    throw std::invalid_argument(
            "pose_graph duplicate-submap-pair override requires max_edges >= 2 when enabled");
  }
  if (pose_graph_minimum_interval_override_single_floor_z_enabled_ &&
    (!pose_graph_minimum_interval_override_config_.enabled ||
    pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_ <= 0.0 ||
    pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_ < 1.0))
  {
    throw std::invalid_argument(
            "pose_graph single-floor return constraint requires an enabled minimum-interval "
            "override, a positive XY distance, and a Z weight multiplier >= 1.0");
  }
  require_finite_nonnegative(
    pose_graph_optimization_step_,
    "pose_graph_optimization_step");
  require_finite_nonnegative(
    pose_graph_odom_translation_weight_,
    "pose_graph_odom_translation_weight");
  require_finite_nonnegative(
    pose_graph_odom_rotation_weight_,
    "pose_graph_odom_rotation_weight");
  require_finite_nonnegative(
    pose_graph_loop_translation_weight_,
    "pose_graph_loop_translation_weight");
  require_finite_nonnegative(
    pose_graph_loop_rotation_weight_,
    "pose_graph_loop_rotation_weight");
  require_finite_nonnegative(
    pose_graph_loop_cauchy_scale_,
    "pose_graph_loop_cauchy_scale");
  require_finite_nonnegative(
    pose_graph_min_new_loop_robust_weight_,
    "pose_graph_min_new_loop_robust_weight");
  require_finite_nonnegative(
    pose_graph_max_iteration_translation_step_m_,
    "pose_graph_max_iteration_translation_step_m");
  require_finite_nonnegative(
    pose_graph_max_iteration_rotation_step_deg_,
    "pose_graph_max_iteration_rotation_step_deg");
  require_finite_nonnegative(
    pose_graph_max_loop_residual_before_,
    "pose_graph_max_loop_residual_before");
  require_finite_nonnegative(
    pose_graph_max_residual_after_,
    "pose_graph_max_residual_after");
  require_finite_nonnegative(
    pose_graph_max_residual_ratio_,
    "pose_graph_max_residual_ratio");
  require_finite_nonnegative(
    local_map_vertical_radius_m_,
    "local_map_vertical_radius_m");
  require_finite_nonnegative(
    keyframe_vertical_translation_m_,
    "keyframe_vertical_translation_m");
  require_finite_nonnegative(
    keyframe_rotation_3d_deg_,
    "keyframe_rotation_3d_deg");
  require_finite_nonnegative(
    place_recognition_max_vertical_separation_m_,
    "place_recognition_max_vertical_separation_m");
  require_finite_nonnegative(
    place_recognition_registration_only_vertical_recovery_m_,
    "place_recognition_registration_only_vertical_recovery_m");
  require_finite_nonnegative(
    place_recognition_registration_only_same_layer_xy_max_distance_m_,
    "place_recognition_registration_only_same_layer_xy_max_distance_m");
  require_finite_nonnegative(
    place_recognition_registration_only_max_pose_jump_m_,
    "place_recognition_registration_only_max_pose_jump_m");
  require_finite_nonnegative(
    place_recognition_mapping_refinement_large_pose_jump_max_translation_m_,
    "place_recognition_mapping_refinement_large_pose_jump_max_translation_m");
  require_finite_nonnegative(
    place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_,
    "place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg");
  require_finite_nonnegative(
    place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_,
    "place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance");
  require_finite_nonnegative(
    place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m_,
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m");
  require_finite_nonnegative(
    place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg_,
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg");
  if (place_recognition_registration_only_vertical_recovery_m_ > 0.0 &&
    place_recognition_max_vertical_separation_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "place_recognition_registration_only_vertical_recovery_m requires a positive "
            "place_recognition_max_vertical_separation_m");
  }
  if (place_recognition_registration_only_vertical_recovery_m_ > 0.0 &&
    place_recognition_registration_only_vertical_recovery_m_ <=
    place_recognition_max_vertical_separation_m_)
  {
    throw std::invalid_argument(
            "place_recognition_registration_only_vertical_recovery_m must exceed "
            "place_recognition_max_vertical_separation_m");
  }
  if (place_recognition_registration_only_same_layer_xy_max_distance_m_ > 0.0 &&
    place_recognition_registration_only_vertical_recovery_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "place_recognition_registration_only_same_layer_xy_max_distance_m requires "
            "place_recognition_registration_only_vertical_recovery_m");
  }
  require_finite_nonnegative(
    place_recognition_submap_consistency_max_translation_m_,
    "place_recognition_submap_consistency_max_translation_m");
  require_finite_nonnegative(
    place_recognition_registration_only_consistency_max_translation_m_,
    "place_recognition_registration_only_consistency_max_translation_m");
  require_finite_nonnegative(
    place_recognition_registration_only_consistency_max_gap_s_,
    "place_recognition_registration_only_consistency_max_gap_s");
  if (place_recognition_registration_only_consistency_max_translation_m_ > 0.0 &&
    place_recognition_registration_only_vertical_recovery_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "place_recognition_registration_only_consistency_max_translation_m requires "
            "place_recognition_registration_only_vertical_recovery_m");
  }
  if (place_recognition_registration_only_consistency_max_translation_m_ > 0.0 &&
    (place_recognition_submap_consistency_max_translation_m_ <= 0.0 ||
    place_recognition_registration_only_consistency_max_translation_m_ <=
    place_recognition_submap_consistency_max_translation_m_))
  {
    throw std::invalid_argument(
            "place_recognition_registration_only_consistency_max_translation_m must exceed "
            "a positive place_recognition_submap_consistency_max_translation_m");
  }
  require_finite_nonnegative(
    place_recognition_submap_consistency_max_rotation_deg_,
    "place_recognition_submap_consistency_max_rotation_deg");
  if (!std::isfinite(place_recognition_mapping_refinement_quality_max_correspondence_m_) ||
    place_recognition_mapping_refinement_quality_max_correspondence_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "place_recognition_mapping_refinement_quality_max_correspondence_m "
            "must be finite and greater than zero");
  }
  if (!std::isfinite(
      place_recognition_mapping_refinement_target_confirmation_voxel_m_) ||
    place_recognition_mapping_refinement_target_confirmation_voxel_m_ <= 0.0)
  {
    throw std::invalid_argument(
            "place_recognition_mapping_refinement_target_confirmation_voxel_m "
            "must be finite and greater than zero");
  }
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.min_inlier_ratio,
    "place_recognition_mapping_refinement_min_inlier_ratio");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.min_target_coverage_ratio,
    "place_recognition_mapping_refinement_min_target_coverage_ratio");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.max_inlier_rmse_m,
    "place_recognition_mapping_refinement_max_inlier_rmse_m");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.max_translation_m,
    "place_recognition_mapping_refinement_max_translation_m");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.max_rotation_deg,
    "place_recognition_mapping_refinement_max_rotation_deg");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.max_hessian_condition,
    "place_recognition_mapping_refinement_max_hessian_condition");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.min_score_improvement_ratio,
    "place_recognition_mapping_refinement_min_score_improvement_ratio");
  require_finite_nonnegative(
    mapping_cloud_refinement_config_.max_primary_score_regression_ratio,
    "place_recognition_mapping_refinement_max_primary_score_regression_ratio");
  require_finite_nonnegative(
    mapping_cloud_verified_coarse_config_.min_inlier_ratio,
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_inlier_ratio");
  require_finite_nonnegative(
    mapping_cloud_verified_coarse_config_.max_inlier_rmse_m,
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_inlier_rmse_m");
  require_finite_nonnegative(
    mapping_cloud_verified_coarse_config_.min_target_coverage_ratio,
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_target_coverage_ratio");
  require_finite_nonnegative(
    mapping_cloud_verified_coarse_config_.max_score_regression_ratio,
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_score_regression_ratio");
  if (mapping_cloud_refinement_config_.min_inlier_ratio > 1.0 ||
    mapping_cloud_refinement_config_.min_target_coverage_ratio > 1.0 ||
    mapping_cloud_refinement_config_.min_score_improvement_ratio > 1.0 ||
    mapping_cloud_verified_coarse_config_.min_inlier_ratio > 1.0 ||
    mapping_cloud_verified_coarse_config_.min_target_coverage_ratio > 1.0)
  {
    throw std::invalid_argument(
            "place recognition mapping refinement ratios must be <= 1.0");
  }
  if (place_recognition_mapping_refinement_min_source_points_ < 1 ||
    place_recognition_mapping_refinement_min_target_points_ < 1)
  {
    throw std::invalid_argument(
            "place recognition mapping refinement point limits must be >= 1");
  }
  if (place_recognition_mapping_refinement_min_target_keyframe_observations_ < 1) {
    throw std::invalid_argument(
            "place_recognition_mapping_refinement_min_target_keyframe_observations "
            "must be >= 1");
  }
  if (mapping_cloud_refinement_config_.min_observable_rank < 0 ||
    mapping_cloud_refinement_config_.min_observable_rank > 6)
  {
    throw std::invalid_argument(
            "place_recognition_mapping_refinement_min_observable_rank must be in [0, 6]");
  }
  require_finite_nonnegative(submap_max_travel_m_, "submap_max_travel_m");
  if (pose_graph_optimization_step_ > 1.0) {
    throw std::invalid_argument("pose_graph_optimization_step must be <= 1.0");
  }
  if (pose_graph_min_new_loop_robust_weight_ > 1.0) {
    throw std::invalid_argument(
            "pose_graph_min_new_loop_robust_weight must be <= 1.0");
  }
  if (pose_graph_robust_loop_enabled_ &&
    pose_graph_loop_cauchy_scale_ <= 0.0)
  {
    throw std::invalid_argument(
            "pose_graph_loop_cauchy_scale must be > 0 when robust loops are enabled");
  }
  if (pose_graph_solver_ != "legacy_relaxation" &&
    pose_graph_solver_ != "ceres_se3")
  {
    throw std::invalid_argument(
            "pose_graph_solver must be legacy_relaxation or ceres_se3");
  }
  if (pose_graph_information_mode_ != "configured_diagonal" &&
    pose_graph_information_mode_ != "registration_observability")
  {
    throw std::invalid_argument(
            "pose_graph_information_mode must be configured_diagonal or "
            "registration_observability");
  }
  if (pose_graph_information_mode_ == "registration_observability" &&
    pose_graph_solver_ != "ceres_se3")
  {
    throw std::invalid_argument(
            "pose_graph_information_mode=registration_observability requires "
            "pose_graph_solver=ceres_se3");
  }
  if (!validRegistrationInformationConfig(pose_graph_information_config_)) {
    throw std::invalid_argument("pose graph registration information configuration is invalid");
  }
  if (pose_graph_solver_ == "ceres_se3" && pose_graph_planar_correction_enabled_) {
    throw std::invalid_argument(
            "pose_graph_solver=ceres_se3 requires full 3D pose graph correction");
  }
  if (pose_graph_optimizer_config_.dcs_enabled &&
    pose_graph_optimizer_config_.dcs_start_stage >=
    static_cast<int>(pose_graph_optimizer_config_.robust_scale_factors.size()))
  {
    throw std::invalid_argument(
            "pose_graph_ceres_dcs_start_stage must select an existing robust stage");
  }
  if (!validPoseGraphOptimizationConfig(pose_graph_optimizer_config_)) {
    throw std::invalid_argument("pose graph Ceres optimizer configuration is invalid");
  }
  if (place_recognition_submap_min_consecutive_matches_ < 1) {
    throw std::invalid_argument(
            "place_recognition_submap_min_consecutive_matches must be >= 1");
  }
  if (submap_enabled_) {
    if (submap_max_keyframes_ > 0 && submap_max_keyframes_ < 2) {
      throw std::invalid_argument(
              "submap_max_keyframes must be 0 or >= 2");
    }
    if (submap_max_keyframes_ == 0 && submap_max_travel_m_ <= 0.0) {
      throw std::invalid_argument(
              "submap_enabled requires submap_max_keyframes >= 2 or "
              "submap_max_travel_m > 0");
    }
    if (submap_overlap_keyframes_ < 0) {
      throw std::invalid_argument("submap_overlap_keyframes must be >= 0");
    }
    if (submap_max_keyframes_ >= 2 &&
      submap_overlap_keyframes_ >= submap_max_keyframes_)
    {
      throw std::invalid_argument(
              "submap_overlap_keyframes must be smaller than submap_max_keyframes");
    }
    if (max_local_submaps_ < 1) {
      throw std::invalid_argument("max_local_submaps must be >= 1");
    }
  }
  if (place_recognition_use_submaps_ && !submap_enabled_) {
    throw std::invalid_argument(
            "place_recognition_use_submaps requires submap_enabled");
  }
  if (place_recognition_mapping_cloud_refinement_enabled_ &&
    (!place_recognition_enabled_ || !submap_enabled_ || !place_recognition_use_submaps_))
  {
    throw std::invalid_argument(
            "mapping-cloud refinement requires place recognition with completed submaps");
  }
  if (place_recognition_mapping_cloud_refinement_enabled_ &&
    (registration_method_ != "voxel_surfel" ||
    mapping_cloud_refinement_config_.min_observable_rank < 1 ||
    mapping_cloud_refinement_config_.max_hessian_condition <= 0.0))
  {
    throw std::invalid_argument(
            "mapping-cloud refinement requires voxel_surfel and active observability gates");
  }
  if (place_recognition_mapping_refinement_required_for_pose_graph_ &&
    (!place_recognition_mapping_cloud_refinement_enabled_ ||
    !pose_graph_correction_enabled_))
  {
    throw std::invalid_argument(
            "required mapping-cloud pose-graph verification needs mapping-cloud refinement "
            "and pose-graph correction");
  }
  if (place_recognition_mapping_refinement_large_pose_jump_enabled_ &&
    (!place_recognition_mapping_cloud_refinement_enabled_ ||
    !pose_graph_correction_enabled_ ||
    place_recognition_mapping_refinement_large_pose_jump_max_translation_m_ <=
    place_recognition_max_pose_jump_m_ ||
    place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_ <= 0.0 ||
    place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_ <= 0.0))
  {
    throw std::invalid_argument(
            "large pose-jump recovery requires mapping-cloud refinement, pose-graph "
            "correction, a translation limit above the normal pose-jump limit, and "
            "positive rotation/descriptor limits");
  }
  if (mapping_cloud_verified_coarse_config_.enabled &&
    (!place_recognition_mapping_refinement_large_pose_jump_enabled_ ||
    mapping_cloud_verified_coarse_config_.min_inlier_ratio <= 0.0 ||
    mapping_cloud_verified_coarse_config_.max_inlier_rmse_m <= 0.0 ||
    mapping_cloud_verified_coarse_config_.min_target_coverage_ratio <= 0.0))
  {
    throw std::invalid_argument(
            "verified coarse mapping-cloud acceptance requires large pose-jump recovery "
            "and positive inlier, RMSE, and target-coverage gates");
  }
  if (pose_graph_distributed_loop_seed_enabled_ &&
    (!place_recognition_mapping_refinement_large_pose_jump_enabled_ ||
    !pose_graph_robust_loop_enabled_ ||
    !pose_graph_minimum_interval_override_config_.enabled))
  {
    throw std::invalid_argument(
            "distributed loop seeding requires large pose-jump recovery, robust loops, "
            "and the independent high-confidence quality gates");
  }
  if (quarantine_pose_on_shock_ && !quarantine_keyframes_on_shock_) {
    throw std::invalid_argument(
            "quarantine_pose_on_shock requires quarantine_keyframes_on_shock");
  }
  if (imu_shock_policy_enabled_ && !imu_roughness_detection_enabled_) {
    throw std::invalid_argument(
            "imu_shock_policy_enabled requires imu_roughness_detection_enabled");
  }
  require_finite_nonnegative(imu_shock_hold_duration_s_, "imu_shock_hold_duration_s");
  if (imu_shock_recovery_stable_scans_ < 1) {
    throw std::invalid_argument("imu_shock_recovery_stable_scans must be >= 1");
  }
  if (terrain_config_.enabled) {
    if (terrain_evidence_topic_.empty()) {
      throw std::invalid_argument("terrain_evidence_topic must not be empty");
    }
    if (terrain_evidence_colored_topic_.empty()) {
      throw std::invalid_argument("terrain_evidence_colored_topic must not be empty");
    }
    if (terrain_support_topic_.empty()) {
      throw std::invalid_argument("terrain_support_evidence_topic must not be empty");
    }
    if (terrain_preview_config_.enabled && terrain_preview_evidence_topic_.empty()) {
      throw std::invalid_argument("terrain_preview_evidence_topic must not be empty");
    }
    if (terrain_config_.azimuth_bins < 8 || terrain_config_.min_points_per_bin < 1 ||
      terrain_config_.seed_min_cluster_bins < 1 ||
      terrain_config_.ramp_min_run_bins < 2 || terrain_config_.transition_support_bins < 1 ||
      terrain_config_.staircase_min_risers < 2)
    {
      throw std::invalid_argument(
              "terrain bin and support counts are below their minimum values");
    }
    if (!std::isfinite(terrain_config_.floor_quantile) ||
      terrain_config_.floor_quantile < 0.0 || terrain_config_.floor_quantile > 1.0)
    {
      throw std::invalid_argument("terrain_floor_quantile must be in [0, 1]");
    }
    require_finite_nonnegative(terrain_config_.min_range_m, "terrain_min_range_m");
    require_finite_nonnegative(terrain_config_.max_range_m, "terrain_max_range_m");
    require_finite_nonnegative(
      terrain_config_.radial_bin_size_m, "terrain_radial_bin_size_m");
    require_finite_nonnegative(
      terrain_config_.seed_max_range_m, "terrain_seed_max_range_m");
    require_finite_nonnegative(
      terrain_config_.classification_voxel_size_m,
      "terrain_classification_voxel_size_m");
    if (!std::isfinite(terrain_config_.seed_reference_height_m)) {
      throw std::invalid_argument("terrain_seed_reference_height_m must be finite");
    }
    require_finite_nonnegative(
      terrain_config_.seed_height_cluster_tolerance_m,
      "terrain_seed_height_cluster_tolerance_m");
    require_finite_nonnegative(
      terrain_config_.max_profile_height_above_base_m,
      "terrain_max_profile_height_above_base_m");
    require_finite_nonnegative(
      terrain_config_.max_seed_height_delta_m, "terrain_max_seed_height_delta_m");
    require_finite_nonnegative(terrain_config_.ground_band_m, "terrain_ground_band_m");
    require_finite_nonnegative(
      terrain_config_.max_adjacent_gap_m, "terrain_max_adjacent_gap_m");
    require_finite_nonnegative(
      terrain_config_.floor_max_slope_deg, "terrain_floor_max_slope_deg");
    require_finite_nonnegative(
      terrain_config_.ramp_min_slope_deg, "terrain_ramp_min_slope_deg");
    require_finite_nonnegative(
      terrain_config_.ramp_max_slope_deg, "terrain_ramp_max_slope_deg");
    require_finite_nonnegative(
      terrain_config_.ramp_min_run_length_m, "terrain_ramp_min_run_length_m");
    require_finite_nonnegative(
      terrain_config_.ramp_plane_neighborhood_radius_m,
      "terrain_ramp_plane_neighborhood_radius_m");
    require_finite_nonnegative(
      terrain_config_.ramp_plane_min_length_m,
      "terrain_ramp_plane_min_length_m");
    require_finite_nonnegative(
      terrain_config_.ramp_plane_min_width_m,
      "terrain_ramp_plane_min_width_m");
    require_finite_nonnegative(
      terrain_config_.ramp_plane_max_rmse_m,
      "terrain_ramp_plane_max_rmse_m");
    require_finite_nonnegative(
      terrain_config_.ramp_plane_max_outlier_distance_m,
      "terrain_ramp_plane_max_outlier_distance_m");
    if (!std::isfinite(terrain_config_.ramp_plane_min_inlier_ratio) ||
      terrain_config_.ramp_plane_min_inlier_ratio < 0.0 ||
      terrain_config_.ramp_plane_min_inlier_ratio > 1.0)
    {
      throw std::invalid_argument("terrain_ramp_plane_min_inlier_ratio must be in [0, 1]");
    }
    if (terrain_config_.ramp_plane_validation_enabled &&
      (terrain_config_.ramp_plane_neighborhood_radius_m <= 0.0 ||
      terrain_config_.ramp_plane_min_points < 3 ||
      terrain_config_.ramp_plane_min_length_m <= 0.0 ||
      terrain_config_.ramp_plane_min_width_m <= 0.0 ||
      terrain_config_.ramp_plane_max_rmse_m <= 0.0 ||
      terrain_config_.ramp_plane_max_outlier_distance_m <= 0.0 ||
      terrain_config_.ramp_plane_min_inlier_ratio <= 0.0))
    {
      throw std::invalid_argument(
              "enabled terrain ramp plane validation requires positive bounds, at least "
              "three points, and a nonzero inlier ratio");
    }
    require_finite_nonnegative(
      terrain_config_.step_min_height_m, "terrain_step_min_height_m");
    require_finite_nonnegative(
      terrain_config_.step_max_height_m, "terrain_step_max_height_m");
    require_finite_nonnegative(
      terrain_config_.drop_min_height_m, "terrain_drop_min_height_m");
    require_finite_nonnegative(
      terrain_config_.drop_max_height_m, "terrain_drop_max_height_m");
    require_finite_nonnegative(
      terrain_config_.staircase_min_riser_height_m,
      "terrain_staircase_min_riser_height_m");
    require_finite_nonnegative(
      terrain_config_.staircase_max_riser_height_m,
      "terrain_staircase_max_riser_height_m");
    require_finite_nonnegative(
      terrain_config_.staircase_min_tread_depth_m,
      "terrain_staircase_min_tread_depth_m");
    require_finite_nonnegative(
      terrain_config_.staircase_max_tread_depth_m,
      "terrain_staircase_max_tread_depth_m");
    require_finite_nonnegative(
      terrain_config_.staircase_max_riser_height_deviation_m,
      "terrain_staircase_max_riser_height_deviation_m");
    require_finite_nonnegative(
      terrain_config_.staircase_max_tread_depth_deviation_m,
      "terrain_staircase_max_tread_depth_deviation_m");
    require_finite_nonnegative(
      terrain_config_.staircase_min_slope_deg,
      "terrain_staircase_min_slope_deg");
    require_finite_nonnegative(
      terrain_config_.staircase_max_slope_deg,
      "terrain_staircase_max_slope_deg");
    require_finite_nonnegative(
      terrain_config_.staircase_max_profile_height_above_base_m,
      "terrain_staircase_max_profile_height_above_base_m");
    require_finite_nonnegative(
      terrain_config_.roughness_min_height_m,
      "terrain_roughness_min_height_m");
    require_finite_nonnegative(
      terrain_config_.roughness_max_score,
      "terrain_roughness_max_score");
    require_finite_nonnegative(
      terrain_accumulator_config_.voxel_size_m, "terrain_evidence_voxel_size_m");
    require_finite_nonnegative(
      terrain_accumulator_config_.min_viewpoint_distance_m,
      "terrain_confirmation_min_viewpoint_distance_m");
    require_finite_nonnegative(
      terrain_accumulator_config_.probable_support_max_roughness_score,
      "terrain_probable_support_max_roughness_score");
    require_finite_nonnegative(
      terrain_accumulator_config_.probable_support_max_anchor_distance_m,
      "terrain_probable_support_max_anchor_distance_m");
    require_finite_nonnegative(
      terrain_accumulator_config_.probable_support_max_neighbor_height_delta_m,
      "terrain_probable_support_max_neighbor_height_delta_m");
    require_finite_nonnegative(
      terrain_preview_config_.association_radius_m,
      "terrain_traversal_association_radius_m");
    require_finite_nonnegative(
      terrain_preview_config_.exit_radius_m,
      "terrain_traversal_exit_radius_m");
    require_finite_nonnegative(
      terrain_preview_config_.max_vertical_delta_m,
      "terrain_traversal_max_vertical_delta_m");
    require_finite_nonnegative(
      terrain_preview_config_.max_sample_gap_s,
      "terrain_traversal_max_sample_gap_s");
    require_finite_nonnegative(
      terrain_preview_config_.max_swept_segment_length_m,
      "terrain_traversal_max_swept_segment_length_m");
    require_finite_nonnegative(
      terrain_preview_config_.min_ahead_distance_m,
      "terrain_preview_min_ahead_distance_m");
    require_finite_nonnegative(
      terrain_preview_config_.max_lateral_distance_m,
      "terrain_preview_max_lateral_distance_m");
    require_finite_nonnegative(
      terrain_preview_config_.roughness_score_gate,
      "terrain_preview_roughness_score_gate");
    require_finite_nonnegative(
      terrain_preview_config_.ramp_shock_prior,
      "terrain_preview_ramp_shock_prior");
    require_finite_nonnegative(
      terrain_preview_config_.positive_step_shock_prior,
      "terrain_preview_positive_step_shock_prior");
    require_finite_nonnegative(
      terrain_preview_config_.min_passage_duration_s,
      "terrain_traversal_min_passage_duration_s");
    if (!std::isfinite(terrain_accumulator_config_.min_label_confidence) ||
      terrain_accumulator_config_.min_label_confidence <= 0.5 ||
      terrain_accumulator_config_.min_label_confidence > 1.0)
    {
      throw std::invalid_argument("terrain_min_label_confidence must be in (0.5, 1]");
    }
    if (!std::isfinite(terrain_accumulator_config_.min_view_label_confidence) ||
      terrain_accumulator_config_.min_view_label_confidence <= 0.5 ||
      terrain_accumulator_config_.min_view_label_confidence > 1.0)
    {
      throw std::invalid_argument(
              "terrain_min_view_label_confidence must be in (0.5, 1]");
    }
    if (!std::isfinite(terrain_accumulator_config_.probable_support_min_confidence) ||
      terrain_accumulator_config_.probable_support_min_confidence <= 0.5 ||
      terrain_accumulator_config_.probable_support_min_confidence >
      terrain_accumulator_config_.min_label_confidence)
    {
      throw std::invalid_argument(
              "terrain_probable_support_min_confidence must be in "
              "(0.5, terrain_min_label_confidence]");
    }
    if (!std::isfinite(terrain_accumulator_config_.ramp_traversability_scale) ||
      terrain_accumulator_config_.ramp_traversability_scale < 0.0 ||
      terrain_accumulator_config_.ramp_traversability_scale > 1.0)
    {
      throw std::invalid_argument("terrain_ramp_traversability_scale must be in [0, 1]");
    }
    if (terrain_config_.radial_bin_size_m <= 0.0 ||
      terrain_config_.classification_voxel_size_m <= 0.0 ||
      terrain_config_.seed_height_cluster_tolerance_m <= 0.0 ||
      terrain_config_.max_range_m <= terrain_config_.min_range_m ||
      terrain_config_.seed_max_range_m < terrain_config_.min_range_m ||
      terrain_config_.seed_max_range_m > terrain_config_.max_range_m ||
      terrain_config_.max_adjacent_gap_m < terrain_config_.radial_bin_size_m ||
      terrain_config_.ramp_min_slope_deg < terrain_config_.floor_max_slope_deg ||
      terrain_config_.ramp_max_slope_deg < terrain_config_.ramp_min_slope_deg ||
      terrain_config_.ramp_max_slope_deg >= 90.0 ||
      terrain_config_.ramp_min_run_length_m <= 0.0 ||
      terrain_config_.roughness_min_height_m <= 0.0 ||
      terrain_config_.roughness_max_score < 1.0 ||
      terrain_config_.step_min_height_m <= 0.0 ||
      terrain_config_.step_max_height_m < terrain_config_.step_min_height_m ||
      terrain_config_.drop_min_height_m <= 0.0 ||
      terrain_config_.drop_max_height_m < terrain_config_.drop_min_height_m ||
      terrain_config_.staircase_min_riser_height_m <= 0.0 ||
      terrain_config_.staircase_max_riser_height_m <
      terrain_config_.staircase_min_riser_height_m ||
      terrain_config_.staircase_min_tread_depth_m <= 0.0 ||
      terrain_config_.staircase_max_tread_depth_m <
      terrain_config_.staircase_min_tread_depth_m ||
      terrain_config_.staircase_min_slope_deg >= 90.0 ||
      terrain_config_.staircase_max_slope_deg < terrain_config_.staircase_min_slope_deg ||
      terrain_config_.staircase_max_slope_deg >= 90.0 ||
      terrain_config_.staircase_max_profile_height_above_base_m <
      terrain_config_.max_profile_height_above_base_m ||
      terrain_accumulator_config_.voxel_size_m <= 0.0 ||
      terrain_accumulator_config_.min_confirmations < 1 ||
      terrain_accumulator_config_.min_viewpoint_distance_m < 0.0 ||
      terrain_accumulator_config_.probable_support_max_roughness_score <= 0.0 ||
      terrain_accumulator_config_.probable_support_max_anchor_distance_m <= 0.0 ||
      terrain_accumulator_config_.probable_support_max_neighbor_height_delta_m <= 0.0 ||
      terrain_preview_config_.association_radius_m <= 0.0 ||
      terrain_preview_config_.exit_radius_m < terrain_preview_config_.association_radius_m ||
      terrain_preview_config_.max_vertical_delta_m <= 0.0 ||
      terrain_preview_config_.max_swept_segment_length_m <= 0.0 ||
      terrain_preview_config_.roughness_score_gate <= 0.0 ||
      terrain_preview_config_.min_passage_samples < 1 ||
      terrain_preview_config_.min_label_confidence <= 0.5 ||
      terrain_preview_config_.min_label_confidence > 1.0)
    {
      throw std::invalid_argument("indoor terrain evidence parameter ordering is invalid");
    }
  }
  if (
    inter_lidar_motion_compensation_source_ != "wheel_odom" &&
    inter_lidar_motion_compensation_source_ != "imu_rotation")
  {
    RCLCPP_WARN(
      get_logger(),
      "Unknown inter_lidar_motion_compensation_source '%s'. Falling back to wheel_odom.",
      inter_lidar_motion_compensation_source_.c_str());
    inter_lidar_motion_compensation_source_ = "wheel_odom";
  }
  if (
    inter_lidar_motion_failure_policy_ != "drop_auxiliary" &&
    inter_lidar_motion_failure_policy_ != "merge_uncompensated")
  {
    RCLCPP_WARN(
      get_logger(),
      "Unknown inter_lidar_motion_failure_policy '%s'. Falling back to drop_auxiliary.",
      inter_lidar_motion_failure_policy_.c_str());
    inter_lidar_motion_failure_policy_ = "drop_auxiliary";
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

  primary_cloud_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions cloud_subscription_options;
  cloud_subscription_options.callback_group = primary_cloud_callback_group_;
  primary_cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    primary_cloud_topic_,
    cloud_qos,
    std::bind(&ScanToMapIcp::primaryCloudCallback, this, std::placeholders::_1),
    cloud_subscription_options);

  auxiliary_cloud_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);
  for (std::size_t index = 0; index < auxiliary_cloud_inputs_.size(); ++index) {
    auto & input = auxiliary_cloud_inputs_[index];
    if (!input.enabled) {
      continue;
    }
    rclcpp::SubscriptionOptions auxiliary_subscription_options;
    auxiliary_subscription_options.callback_group =
      auxiliary_cloud_callback_group_;
    input.subscription = create_subscription<sensor_msgs::msg::PointCloud2>(
      input.topic,
      cloud_qos,
      [this, index](const sensor_msgs::msg::PointCloud2::SharedPtr message) {
        auxiliaryCloudCallback(message, index);
      },
      auxiliary_subscription_options);
  }

  const bool use_wheel_inter_lidar_compensation =
    inter_lidar_motion_compensation_enabled_ &&
    inter_lidar_motion_compensation_source_ == "wheel_odom";
  if (use_wheel_prediction_ || use_wheel_inter_lidar_compensation) {
    wheel_callback_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions wheel_subscription_options;
    wheel_subscription_options.callback_group = wheel_callback_group_;
    wheel_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      wheel_odom_topic_,
      rclcpp::QoS(50),
      std::bind(&ScanToMapIcp::wheelOdomCallback, this, std::placeholders::_1),
      wheel_subscription_options);
  }

  imu_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  rclcpp::SubscriptionOptions imu_subscription_options;
  imu_subscription_options.callback_group = imu_callback_group_;
  auto imu_qos = rclcpp::QoS(
    rclcpp::KeepLast(std::max(5, imu_subscription_depth_))).durability_volatile();
  if (imu_subscription_reliable_) {
    imu_qos.reliable();
  } else {
    imu_qos.best_effort();
  }
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
  export_keyframes_service_ = create_service<std_srvs::srv::Trigger>(
    "~/export_keyframes",
    [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
      exportKeyframes(response);
    });
  terrain_evidence_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    terrain_evidence_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  terrain_evidence_colored_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    terrain_evidence_colored_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  terrain_support_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    terrain_support_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  terrain_preview_evidence_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
    terrain_preview_evidence_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  pose_graph_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    pose_graph_marker_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  health_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
    health_topic_,
    rclcpp::QoS(10));
  imu_watermark_pub_ = create_publisher<std_msgs::msg::Int64>(
    imu_watermark_topic_,
    rclcpp::QoS(1).reliable().transient_local());
  if (publish_preprocessed_scan_) {
    preprocessed_scan_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      preprocessed_scan_topic_,
      rclcpp::QoS(10).reliable());
  }

  const auto map_publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(std::max(map_publish_period_s_, 0.1)));
  map_publish_timer_ = create_wall_timer(
    map_publish_period,
    std::bind(&ScanToMapIcp::mapPublishTimerCallback, this));
  cloud_processing_callback_group_ = create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
  const auto cloud_processing_period =
    std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(std::max(cloud_processing_period_s_, 0.001)));
  cloud_processing_timer_ = create_wall_timer(
    cloud_processing_period,
    std::bind(&ScanToMapIcp::dispatchQueuedClouds, this),
    cloud_processing_callback_group_);
}

void ScanToMapIcp::primaryCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  std::int64_t required_imu_stamp_ns = 0;
  std::int64_t scan_start_stamp_ns = 0;
  std::int64_t scan_end_stamp_ns = 0;
  bool have_required_imu_stamp = false;
  bool have_scan_time_bounds = false;
  const bool imu_time_horizon_required =
    deskew_enabled_ ||
    imu_rotation_prediction_enabled_ ||
    imu_roughness_detection_enabled_ ||
    (inter_lidar_motion_compensation_enabled_ &&
    inter_lidar_motion_compensation_source_ == "imu_rotation");
  if (imu_time_horizon_required) {
    const auto cloud_time_bounds = pointCloudTimeBoundsNs(
      *msg,
      deskew_point_time_field_,
      deskew_point_time_scale_,
      deskew_point_time_offset_s_,
      deskew_max_scan_duration_s_);
    if (cloud_time_bounds.has_value()) {
      scan_start_stamp_ns = cloud_time_bounds->start_stamp_ns;
      scan_end_stamp_ns = cloud_time_bounds->end_stamp_ns;
      required_imu_stamp_ns = scan_end_stamp_ns;
      have_required_imu_stamp = true;
      have_scan_time_bounds = true;
    }
  }

  std::lock_guard<std::mutex> lock(cloud_sync_mutex_);
  const auto stamp = rclcpp::Time(msg->header.stamp);
  const auto insertion = std::lower_bound(
    pending_primary_clouds_.begin(),
    pending_primary_clouds_.end(),
    stamp,
    [](const PendingPrimaryCloud & candidate, const rclcpp::Time & target) {
      return rclcpp::Time(candidate.message->header.stamp) < target;
    });
  if (
    insertion != pending_primary_clouds_.end() &&
    rclcpp::Time(insertion->message->header.stamp) == stamp)
  {
    insertion->message = msg;
    insertion->required_imu_stamp_ns = required_imu_stamp_ns;
    insertion->scan_start_stamp_ns = scan_start_stamp_ns;
    insertion->scan_end_stamp_ns = scan_end_stamp_ns;
    insertion->have_required_imu_stamp = have_required_imu_stamp;
    insertion->have_scan_time_bounds = have_scan_time_bounds;
    ++primary_sync_duplicate_count_;
  } else {
    if (have_latest_primary_sync_stamp_ && stamp.nanoseconds() < latest_primary_sync_stamp_ns_) {
      ++primary_sync_out_of_order_count_;
    }
    pending_primary_clouds_.insert(
      insertion,
      PendingPrimaryCloud{
          msg,
          std::chrono::steady_clock::now(),
          required_imu_stamp_ns,
          scan_start_stamp_ns,
          scan_end_stamp_ns,
          have_required_imu_stamp,
          have_scan_time_bounds,
          false,
          false});
  }
  if (!have_latest_primary_sync_stamp_ || stamp.nanoseconds() > latest_primary_sync_stamp_ns_) {
    latest_primary_sync_stamp_ns_ = stamp.nanoseconds();
    have_latest_primary_sync_stamp_ = true;
  }

  const auto queue_limit = static_cast<std::size_t>(
    std::max(1, cloud_sync_queue_size_));
  bool omitted_pending_cloud = false;
  while (pending_primary_clouds_.size() > queue_limit) {
    const auto victim_stamp_ns = rclcpp::Time(
      pending_primary_clouds_.front().message->header.stamp).nanoseconds();
    const bool victim_already_dispatched =
      have_dispatched_primary_stamp_.load() &&
      victim_stamp_ns <= latest_dispatched_primary_stamp_ns_.load();
    if (!victim_already_dispatched) {
      omitted_pending_cloud = true;
    }
    pending_primary_clouds_.pop_front();
    ++primary_sync_queue_overflow_count_;
  }
  if (omitted_pending_cloud && !pending_primary_clouds_.empty()) {
    pending_primary_clouds_.front().terrain_continuity_break_before = true;
  }
}

void ScanToMapIcp::auxiliaryCloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg,
  const std::size_t sensor_index)
{
  std::int64_t required_imu_stamp_ns = 0;
  std::int64_t scan_start_stamp_ns = 0;
  std::int64_t scan_end_stamp_ns = 0;
  bool have_required_imu_stamp = false;
  bool have_scan_time_bounds = false;
  const bool imu_time_horizon_required =
    deskew_enabled_ ||
    imu_rotation_prediction_enabled_ ||
    imu_roughness_detection_enabled_ ||
    (inter_lidar_motion_compensation_enabled_ &&
    inter_lidar_motion_compensation_source_ == "imu_rotation");
  if (imu_time_horizon_required) {
    const auto cloud_time_bounds = pointCloudTimeBoundsNs(
      *msg,
      deskew_point_time_field_,
      deskew_point_time_scale_,
      deskew_point_time_offset_s_,
      deskew_max_scan_duration_s_);
    if (cloud_time_bounds.has_value()) {
      scan_start_stamp_ns = cloud_time_bounds->start_stamp_ns;
      scan_end_stamp_ns = cloud_time_bounds->end_stamp_ns;
      required_imu_stamp_ns = scan_end_stamp_ns;
      have_required_imu_stamp = true;
      have_scan_time_bounds = true;
    }
  }

  std::lock_guard<std::mutex> lock(cloud_sync_mutex_);
  auto & input = auxiliary_cloud_inputs_.at(sensor_index);
  const auto stamp = rclcpp::Time(msg->header.stamp);
  const auto insertion = std::lower_bound(
    input.queue.begin(),
    input.queue.end(),
    stamp,
    [](const PendingAuxiliaryCloud & candidate,
    const rclcpp::Time & target)
    {
      return rclcpp::Time(candidate.message->header.stamp) < target;
    });
  if (
    insertion != input.queue.end() &&
    rclcpp::Time(insertion->message->header.stamp) == stamp)
  {
    insertion->message = msg;
    insertion->required_imu_stamp_ns = required_imu_stamp_ns;
    insertion->scan_start_stamp_ns = scan_start_stamp_ns;
    insertion->scan_end_stamp_ns = scan_end_stamp_ns;
    insertion->have_required_imu_stamp = have_required_imu_stamp;
    insertion->have_scan_time_bounds = have_scan_time_bounds;
    ++input.duplicate_count;
  } else {
    if (input.have_latest_stamp && stamp.nanoseconds() < input.latest_stamp_ns) {
      ++input.out_of_order_count;
    }
    input.queue.insert(
      insertion,
      PendingAuxiliaryCloud{
          msg,
          required_imu_stamp_ns,
          scan_start_stamp_ns,
          scan_end_stamp_ns,
          have_required_imu_stamp,
          have_scan_time_bounds});
  }
  if (!input.have_latest_stamp || stamp.nanoseconds() > input.latest_stamp_ns) {
    input.latest_stamp_ns = stamp.nanoseconds();
    input.have_latest_stamp = true;
  }

  const auto queue_limit = static_cast<std::size_t>(
    std::max(1, cloud_sync_queue_size_));
  while (input.queue.size() > queue_limit) {
    input.queue.pop_front();
    ++input.queue_overflow_count;
  }
}

void ScanToMapIcp::dispatchQueuedClouds()
{
  bool expected = false;
  if (!processing_.compare_exchange_strong(expected, true)) {
    return;
  }

  // Process one scan per timer invocation so sensor callbacks can advance
  // their timestamp watermarks between queued scans after a slow graph update.
  auto cloud_set = takeSynchronizedCloudSet();
  if (cloud_set.has_value()) {
    processCloud(*cloud_set);
  }
  processing_.store(false);
}

std::optional<ScanToMapIcp::SynchronizedCloudSet>
ScanToMapIcp::takeSynchronizedCloudSet()
{
  const bool use_wheel_inter_lidar_compensation =
    inter_lidar_motion_compensation_enabled_ &&
    inter_lidar_motion_compensation_source_ == "wheel_odom";
  const bool use_imu_inter_lidar_compensation =
    inter_lidar_motion_compensation_enabled_ &&
    inter_lidar_motion_compensation_source_ == "imu_rotation";
  {
    std::lock_guard<std::mutex> cloud_lock(cloud_sync_mutex_);
    if (pending_primary_clouds_.empty()) {
      return std::nullopt;
    }
  }
  bool have_wheel_watermark = false;
  std::int64_t wheel_watermark_ns = 0;
  if (use_wheel_inter_lidar_compensation) {
    std::lock_guard<std::mutex> wheel_lock(wheel_mutex_);
    have_wheel_watermark = have_latest_wheel_odom_;
    wheel_watermark_ns = latest_wheel_stamp_.nanoseconds();
  }

  bool have_imu_watermark = false;
  std::int64_t imu_watermark_ns = 0;
  std::vector<ImuSample> imu_samples;
  std::vector<GravitySample> gravity_samples;
  const bool snapshot_imu_for_cloud =
    deskew_enabled_ ||
    use_imu_inter_lidar_compensation ||
    imu_rotation_prediction_enabled_ ||
    imu_roughness_detection_enabled_;
  const bool snapshot_gravity_for_cloud =
    gravity_roll_pitch_constraint_enabled_ || terrain_config_.enabled;
  if (snapshot_imu_for_cloud || snapshot_gravity_for_cloud) {
    std::lock_guard<std::mutex> imu_lock(imu_mutex_);
    have_imu_watermark = !imu_buffer_.empty();
    if (have_imu_watermark) {
      imu_watermark_ns = latest_imu_stamp_.nanoseconds();
      if (snapshot_imu_for_cloud) {
        imu_samples.assign(imu_buffer_.begin(), imu_buffer_.end());
      }
    }
    if (snapshot_gravity_for_cloud) {
      gravity_samples.assign(gravity_buffer_.begin(), gravity_buffer_.end());
    }
  }

  std::lock_guard<std::mutex> lock(cloud_sync_mutex_);
  // Only this mutually-exclusive dispatcher removes primary clouds, but keep
  // the guard in case shutdown or a future queue policy changes that contract.
  if (pending_primary_clouds_.empty()) {
    return std::nullopt;
  }

  auto & pending = pending_primary_clouds_.front();
  const auto primary_stamp_ns =
    rclcpp::Time(pending.message->header.stamp).nanoseconds();
  const auto tolerance_ns = static_cast<std::int64_t>(
    std::llround(std::max(0.0, max_auxiliary_stamp_delta_s_) * 1.0e9));
  const auto reorder_window_ns = static_cast<std::int64_t>(
    std::llround(std::max(0.0, cloud_sync_reorder_window_s_) * 1.0e9));
  const double queued_wall_age_s = std::chrono::duration<double>(
    std::chrono::steady_clock::now() - pending.queued_at).count();
  const bool timed_out = cloud_sync_allow_wall_timeout_fallback_ &&
    queued_wall_age_s >=
    std::max(0.0, cloud_sync_wait_timeout_s_);

  std::array<std::optional<std::size_t>, 2> selected_indices;
  std::array<double, 2> selected_deltas_s{0.0, 0.0};
  bool all_enabled_inputs_ready = true;
  bool all_enabled_inputs_decided = true;
  std::int64_t required_wheel_stamp_ns = primary_stamp_ns;
  std::int64_t required_imu_stamp_ns = pending.required_imu_stamp_ns;
  bool have_required_imu_stamp = pending.have_required_imu_stamp;
  std::int64_t scan_start_stamp_ns = pending.scan_start_stamp_ns;
  std::int64_t scan_end_stamp_ns = pending.scan_end_stamp_ns;
  bool have_scan_time_bounds = pending.have_scan_time_bounds;
  if (snapshot_imu_for_cloud || snapshot_gravity_for_cloud) {
    required_imu_stamp_ns = std::max(required_imu_stamp_ns, primary_stamp_ns);
    have_required_imu_stamp = true;
  }

  for (std::size_t index = 0; index < auxiliary_cloud_inputs_.size(); ++index) {
    auto & input = auxiliary_cloud_inputs_[index];
    if (!input.enabled) {
      continue;
    }

    while (!input.queue.empty()) {
      const auto stamp_ns =
        rclcpp::Time(input.queue.front().message->header.stamp).nanoseconds();
      if (stamp_ns >= primary_stamp_ns - tolerance_ns) {
        break;
      }
      input.queue.pop_front();
      ++input.stale_drop_count;
    }

    std::int64_t minimum_delta_ns = std::numeric_limits<std::int64_t>::max();
    for (std::size_t queue_index = 0; queue_index < input.queue.size(); ++queue_index) {
      const auto stamp_ns =
        rclcpp::Time(input.queue[queue_index].message->header.stamp).nanoseconds();
      const auto delta_ns = std::abs(stamp_ns - primary_stamp_ns);
      if (delta_ns < minimum_delta_ns) {
        minimum_delta_ns = delta_ns;
        selected_indices[index] = queue_index;
      }
      if (stamp_ns > primary_stamp_ns + tolerance_ns) {
        break;
      }
    }
    const bool auxiliary_watermark_passed =
      input.have_latest_stamp &&
      input.latest_stamp_ns > primary_stamp_ns + tolerance_ns + reorder_window_ns;
    if (!selected_indices[index].has_value() ||
      minimum_delta_ns > tolerance_ns)
    {
      selected_indices[index].reset();
      all_enabled_inputs_ready = false;
      if (!auxiliary_watermark_passed) {
        all_enabled_inputs_decided = false;
      }
    } else {
      // A merely valid candidate is not necessarily the nearest candidate:
      // a closer future auxiliary callback may still be in flight. Exact
      // timestamp matches are already optimal and can dispatch immediately.
      if (cloud_sync_wait_for_nearest_candidate_ && minimum_delta_ns > 0 &&
        !auxiliary_watermark_passed)
      {
        all_enabled_inputs_decided = false;
      }
      selected_deltas_s[index] =
        static_cast<double>(minimum_delta_ns) * 1.0e-9;
      const auto selected_stamp_ns = rclcpp::Time(
        input.queue[*selected_indices[index]].message->header.stamp).nanoseconds();
      required_wheel_stamp_ns = std::max(required_wheel_stamp_ns, selected_stamp_ns);
      if (use_imu_inter_lidar_compensation) {
        required_imu_stamp_ns = std::max(required_imu_stamp_ns, selected_stamp_ns);
        have_required_imu_stamp = true;
      }
      if (snapshot_imu_for_cloud) {
        const auto & selected = input.queue[*selected_indices[index]];
        if (selected.have_required_imu_stamp) {
          required_imu_stamp_ns = std::max(
            required_imu_stamp_ns,
            selected.required_imu_stamp_ns);
          have_required_imu_stamp = true;
        }
        if (selected.have_scan_time_bounds) {
          if (!have_scan_time_bounds) {
            scan_start_stamp_ns = selected.scan_start_stamp_ns;
            scan_end_stamp_ns = selected.scan_end_stamp_ns;
            have_scan_time_bounds = true;
          } else {
            scan_start_stamp_ns = std::min(
              scan_start_stamp_ns, selected.scan_start_stamp_ns);
            scan_end_stamp_ns = std::max(
              scan_end_stamp_ns, selected.scan_end_stamp_ns);
          }
        }
      }
    }
  }

  if (!all_enabled_inputs_decided && !timed_out) {
    return std::nullopt;
  }

  const bool wheel_watermark_ready =
    !use_wheel_inter_lidar_compensation ||
    (have_wheel_watermark && wheel_watermark_ns >= required_wheel_stamp_ns);
  if (!wheel_watermark_ready && !timed_out) {
    return std::nullopt;
  }

  const bool imu_watermark_required =
    have_required_imu_stamp &&
    ((snapshot_imu_for_cloud && deskew_wait_for_imu_watermark_) ||
    snapshot_gravity_for_cloud ||
    use_imu_inter_lidar_compensation);
  const bool imu_watermark_ready =
    !imu_watermark_required ||
    (have_imu_watermark && imu_watermark_ns >= required_imu_stamp_ns);
  const bool imu_wait_timed_out =
    cloud_sync_allow_wall_timeout_fallback_ &&
    imu_watermark_required &&
    !imu_watermark_ready &&
    queued_wall_age_s >= std::max(0.0, deskew_imu_wait_timeout_s_);
  if (!imu_watermark_ready && !imu_wait_timed_out) {
    pending.imu_watermark_waited = true;
    return std::nullopt;
  }

  // Every computation for this cloud must see the same causal IMU prefix.
  // Dispatch timing may expose additional future samples to one callback, but
  // those samples must not change deskew, roughness, or motion prediction.
  const std::int64_t imu_snapshot_horizon_ns = have_required_imu_stamp ?
    required_imu_stamp_ns : primary_stamp_ns;
  const auto first_future_imu = std::upper_bound(
    imu_samples.begin(),
    imu_samples.end(),
    imu_snapshot_horizon_ns,
    [](const std::int64_t horizon_ns, const ImuSample & sample) {
      return horizon_ns < sample.stamp.nanoseconds();
    });
  imu_samples.erase(first_future_imu, imu_samples.end());
  const auto first_future_gravity = std::upper_bound(
    gravity_samples.begin(),
    gravity_samples.end(),
    imu_snapshot_horizon_ns,
    [](const std::int64_t horizon_ns, const GravitySample & sample) {
      return horizon_ns < sample.stamp.nanoseconds();
    });
  gravity_samples.erase(first_future_gravity, gravity_samples.end());

  SynchronizedCloudSet cloud_set;
  cloud_set.primary = pending.message;
  cloud_set.wait_timed_out = !all_enabled_inputs_decided && timed_out;
  cloud_set.watermark_resolved =
    !all_enabled_inputs_ready && all_enabled_inputs_decided;
  cloud_set.wheel_wait_timed_out =
    use_wheel_inter_lidar_compensation && !wheel_watermark_ready && timed_out;
  cloud_set.imu_watermark_waited = pending.imu_watermark_waited;
  cloud_set.imu_wait_timed_out = imu_wait_timed_out;
  cloud_set.terrain_continuity_break_before =
    pending.terrain_continuity_break_before;
  cloud_set.required_imu_stamp_ns = required_imu_stamp_ns;
  cloud_set.scan_start_stamp_ns = scan_start_stamp_ns;
  cloud_set.scan_end_stamp_ns = scan_end_stamp_ns;
  cloud_set.have_scan_time_bounds = have_scan_time_bounds;
  cloud_set.dispatch_imu_stamp_ns = have_imu_watermark ? imu_watermark_ns : 0;
  cloud_set.imu_snapshot_end_stamp_ns = imu_samples.empty() ?
    0 : imu_samples.back().stamp.nanoseconds();
  cloud_set.imu_samples = std::move(imu_samples);
  cloud_set.gravity_samples = std::move(gravity_samples);
  if (cloud_set.watermark_resolved) {
    ++cloud_sync_watermark_resolution_count_;
  }
  if (cloud_set.wait_timed_out || cloud_set.wheel_wait_timed_out ||
    cloud_set.imu_wait_timed_out)
  {
    ++cloud_sync_wall_timeout_count_;
  }
  if (cloud_set.wheel_wait_timed_out) {
    ++inter_lidar_wheel_wait_timeout_count_;
  }
  if (cloud_set.imu_watermark_waited) {
    ++deskew_imu_watermark_waited_count_;
  }
  if (cloud_set.imu_wait_timed_out) {
    ++deskew_imu_wait_timeout_count_;
  }
  const auto previous_dispatched_stamp_ns =
    latest_dispatched_primary_stamp_ns_.load();
  if (!have_dispatched_primary_stamp_.load() ||
    primary_stamp_ns > previous_dispatched_stamp_ns)
  {
    latest_dispatched_primary_stamp_ns_.store(primary_stamp_ns);
    have_dispatched_primary_stamp_.store(true);
  }
  pending_primary_clouds_.pop_front();

  for (std::size_t index = 0; index < auxiliary_cloud_inputs_.size(); ++index) {
    auto & input = auxiliary_cloud_inputs_[index];
    if (!input.enabled) {
      continue;
    }
    ++cloud_set.expected_cloud_count;
    if (!selected_indices[index].has_value()) {
      ++input.missing_count;
      continue;
    }

    const auto selected_index = *selected_indices[index];
    for (std::size_t dropped = 0; dropped < selected_index; ++dropped) {
      input.queue.pop_front();
      ++input.stale_drop_count;
    }
    cloud_set.auxiliary[index] = input.queue.front().message;
    input.queue.pop_front();
    ++input.consumed_count;
    ++cloud_set.merged_cloud_count;
    cloud_set.max_stamp_delta_s =
      std::max(cloud_set.max_stamp_delta_s, selected_deltas_s[index]);
  }
  return cloud_set;
}

void ScanToMapIcp::wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(wheel_mutex_);
  if (
    wheel_odom_require_expected_frames_ &&
    (msg->header.frame_id != wheel_odom_expected_frame_ ||
    msg->child_frame_id != wheel_odom_expected_child_frame_))
  {
    ++wheel_odom_frame_rejected_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected wheel odom frame '%s' -> '%s'; expected '%s' -> '%s'",
      msg->header.frame_id.c_str(),
      msg->child_frame_id.c_str(),
      wheel_odom_expected_frame_.c_str(),
      wheel_odom_expected_child_frame_.c_str());
    return;
  }

  const auto & position = msg->pose.pose.position;
  const auto & orientation = msg->pose.pose.orientation;
  const Eigen::Quaterniond quaternion(
    orientation.w,
    orientation.x,
    orientation.y,
    orientation.z);
  if (
    !std::isfinite(position.x) ||
    !std::isfinite(position.y) ||
    !std::isfinite(position.z) ||
    !quaternion.coeffs().allFinite() ||
    !std::isfinite(quaternion.norm()) ||
    quaternion.norm() <= 1.0e-6)
  {
    ++wheel_odom_invalid_pose_rejected_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected wheel odom with a non-finite position or invalid quaternion");
    return;
  }

  const WheelOdomSample sample{rclcpp::Time(msg->header.stamp), odomToIsometry(*msg)};
  if (
    have_latest_wheel_odom_ &&
    sample.stamp < latest_wheel_stamp_ &&
    (latest_wheel_stamp_ - sample.stamp).seconds() >=
    std::max(1.0e-6, wheel_odom_timestamp_reset_threshold_s_))
  {
    ++wheel_odom_timestamp_rewind_rejected_count_;
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected wheel odom timestamp rewind of %.3fs",
      (latest_wheel_stamp_ - sample.stamp).seconds());
    return;
  }
  const auto insertion = std::lower_bound(
    wheel_odom_buffer_.begin(),
    wheel_odom_buffer_.end(),
    sample.stamp,
    [](const WheelOdomSample & candidate, const rclcpp::Time & stamp) {
      return candidate.stamp < stamp;
    });
  if (insertion != wheel_odom_buffer_.end() && insertion->stamp == sample.stamp) {
    *insertion = sample;
    ++wheel_odom_duplicate_count_;
  } else {
    if (insertion != wheel_odom_buffer_.end()) {
      ++wheel_odom_out_of_order_count_;
    }
    wheel_odom_buffer_.insert(insertion, sample);
  }

  if (!have_latest_wheel_odom_ || sample.stamp >= latest_wheel_stamp_) {
    latest_wheel_odom_ = sample.odom_T_base;
    latest_wheel_stamp_ = sample.stamp;
    have_latest_wheel_odom_ = true;
  }

  const double buffer_duration_s = std::max(0.0, wheel_odom_buffer_duration_s_);
  while (!wheel_odom_buffer_.empty() && buffer_duration_s > 0.0 &&
    (latest_wheel_stamp_ - wheel_odom_buffer_.front().stamp).seconds() >
    buffer_duration_s)
  {
    wheel_odom_buffer_.pop_front();
  }
  const std::size_t max_samples = static_cast<std::size_t>(
    std::max(2, wheel_odom_buffer_max_samples_));
  while (wheel_odom_buffer_.size() > max_samples) {
    wheel_odom_buffer_.pop_front();
  }
}

void ScanToMapIcp::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
  if (
    imu_require_expected_frame_ &&
    msg->header.frame_id != imu_expected_frame_)
  {
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      ++imu_frame_rejected_count_;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected IMU frame '%s'; expected '%s'",
      msg->header.frame_id.c_str(),
      imu_expected_frame_.c_str());
    return;
  }

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
  if (!sample.angular_velocity.allFinite() ||
    !sample.linear_acceleration.allFinite())
  {
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      ++imu_nonfinite_rejected_count_;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "Rejected IMU sample with non-finite angular velocity or acceleration");
    return;
  }
  sample.orientation = Eigen::Quaterniond(
    msg->orientation.w,
    msg->orientation.x,
    msg->orientation.y,
    msg->orientation.z);
  const double orientation_norm = sample.orientation.norm();
  sample.orientation_valid =
    msg->orientation_covariance[0] >= 0.0 &&
    sample.orientation.coeffs().allFinite() &&
    std::isfinite(orientation_norm) &&
    orientation_norm > 1.0e-6;
  if (sample.orientation_valid) {
    sample.orientation.normalize();
  } else {
    sample.orientation = Eigen::Quaterniond::Identity();
  }

  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    const double buffer_duration = std::max(0.1, deskew_imu_buffer_duration_s_);
    bool instantaneous_roughness_event = false;
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
      if (imu_shock_policy_enabled_ && imu_roughness_detection_enabled_ &&
        dt_from_latest > 1.0e-6)
      {
        const ImuSample & previous = imu_buffer_.back();
        const double angular_accel =
          (sample.angular_velocity - previous.angular_velocity).norm() / dt_from_latest;
        const double accel_norm_delta = std::abs(
          sample.linear_acceleration.norm() - previous.linear_acceleration.norm());
        const double linear_jerk =
          (sample.linear_acceleration - previous.linear_acceleration).norm() / dt_from_latest;
        instantaneous_roughness_event =
          sample.angular_velocity.norm() >
          std::max(0.0, imu_roughness_angular_velocity_threshold_radps_) ||
          angular_accel >
          std::max(0.0, imu_roughness_angular_accel_threshold_radps2_) ||
          accel_norm_delta >
          std::max(0.0, imu_roughness_accel_norm_delta_threshold_mps2_) ||
          linear_jerk >
          std::max(0.0, imu_roughness_linear_jerk_threshold_mps3_);
      }
    } else if (imu_shock_policy_enabled_ && imu_roughness_detection_enabled_) {
      instantaneous_roughness_event =
        sample.angular_velocity.norm() >
        std::max(0.0, imu_roughness_angular_velocity_threshold_radps_);
    }

    latest_imu_stamp_ = sample.stamp;
    imu_buffer_.push_back(sample);

    const bool gravity_observation_requested =
      gravity_roll_pitch_constraint_enabled_ || terrain_config_.enabled;
    if (gravity_observation_requested && !instantaneous_roughness_event) {
      const Eigen::Matrix3d imu_to_base = rotationFromRpyDegrees(
        deskew_imu_to_base_roll_deg_,
        deskew_imu_to_base_pitch_deg_,
        deskew_imu_to_base_yaw_deg_);
      const Eigen::Vector3d accel_base =
        gravity_accel_scale_ * imu_to_base * Eigen::Vector3d(
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
    } else if (gravity_observation_requested && instantaneous_roughness_event) {
      ++gravity_shock_sample_rejected_count_;
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
  const auto callback_start = std::chrono::steady_clock::now();
  CloudConstPtr local_map;
  CloudConstPtr global_map;
  TerrainCloudConstPtr terrain_evidence;
  TerrainSupportVectorConstPtr terrain_support;
  TerrainCloudConstPtr terrain_preview_evidence;
  {
    std::lock_guard<std::mutex> lock(map_publish_snapshot_mutex_);
    local_map = local_map_publish_snapshot_;
    global_map = global_map_publish_snapshot_;
    terrain_evidence = terrain_publish_snapshot_;
    terrain_support = terrain_support_publish_snapshot_;
    terrain_preview_evidence = terrain_preview_publish_snapshot_;
  }

  const auto stamp = now();
  publishCloud(local_map_pub_, local_map, map_frame_, stamp);
  publishCloud(global_map_pub_, global_map, map_frame_, stamp);
  publishTerrainCloud(terrain_evidence_pub_, terrain_evidence, map_frame_, stamp);
  publishTerrainColorCloud(
    terrain_evidence_colored_pub_, terrain_evidence, map_frame_, stamp);
  const auto support_publish_start = std::chrono::steady_clock::now();
  publishTerrainSupportCloud(
    terrain_support_pub_, terrain_support, map_frame_, stamp);
  const double support_publish_runtime_ms =
    std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - support_publish_start).count();
  publishTerrainCloud(
    terrain_preview_evidence_pub_,
    terrain_preview_evidence,
    map_frame_,
    stamp);

  const double callback_runtime_ms =
    std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - callback_start).count();
  const auto update_max = [](std::atomic<double> & maximum, const double value) {
      double previous = maximum.load();
      while (value > previous && !maximum.compare_exchange_weak(previous, value)) {
      }
    };
  terrain_support_publish_runtime_ms_.store(support_publish_runtime_ms);
  update_max(terrain_support_publish_runtime_ms_max_, support_publish_runtime_ms);
  map_publish_runtime_ms_.store(callback_runtime_ms);
  update_max(map_publish_runtime_ms_max_, callback_runtime_ms);
  ++map_publish_callback_count_;
}

void ScanToMapIcp::processCloud(const SynchronizedCloudSet & cloud_set)
{
  const rclcpp::Time stamp(cloud_set.primary->header.stamp);
  bool build_uncompensated_hypothesis = false;
  bool initialization_warmup_active = false;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (have_last_primary_cloud_stamp_ && stamp <= last_primary_cloud_stamp_) {
      RegistrationResult result;
      result.rejection_reason = "primary_cloud_timestamp_not_increasing";
      result.input_clouds_expected = cloud_set.expected_cloud_count;
      result.input_clouds_merged = cloud_set.merged_cloud_count;
      result.max_input_cloud_stamp_delta_s = cloud_set.max_stamp_delta_s;
      result.cloud_sync_wait_timed_out = cloud_set.wait_timed_out;
      result.cloud_sync_watermark_resolved = cloud_set.watermark_resolved;
      result.primary_cloud_gap_before = cloud_set.terrain_continuity_break_before;
      result.inter_lidar_wheel_wait_timed_out = cloud_set.wheel_wait_timed_out;
      result.deskew_imu_watermark_waited = cloud_set.imu_watermark_waited;
      result.deskew_imu_wait_timed_out = cloud_set.imu_wait_timed_out;
      result.deskew_required_imu_stamp_ns = cloud_set.required_imu_stamp_ns;
      result.deskew_dispatch_imu_stamp_ns = cloud_set.dispatch_imu_stamp_ns;
      result.imu_snapshot_end_stamp_ns = cloud_set.imu_snapshot_end_stamp_ns;
      result.imu_snapshot_sample_count = static_cast<int>(cloud_set.imu_samples.size());
      last_registration_ = result;
      publishDiagnostics(stamp, result);
      return;
    }
    last_primary_cloud_stamp_ = stamp;
    have_last_primary_cloud_stamp_ = true;
    if (!have_first_primary_cloud_stamp_) {
      first_primary_cloud_stamp_ = stamp;
      have_first_primary_cloud_stamp_ = true;
    }
    initialization_warmup_active =
      !initialized_ &&
      (stamp - first_primary_cloud_stamp_).seconds() <
      initialization_warmup_duration_s_;
    if (cloud_set.terrain_continuity_break_before) {
      terrain_traversal_continuity_break_pending_ = true;
      if (terrain_pending_shock_event_) {
        terrain_pending_shock_bridge_invalidated_ = true;
      }
    }
    const double pending_scan_gap_s = have_last_processed_scan_stamp_ ?
      std::max(0.0, (stamp - last_processed_scan_stamp_).seconds()) : 0.0;
    build_uncompensated_hypothesis =
      inter_lidar_gap_recovery_hypothesis_enabled_ &&
      inter_lidar_motion_compensation_enabled_ &&
      initialized_ &&
      have_last_processed_scan_stamp_ &&
      pending_scan_gap_s >=
      std::max(0.0, inter_lidar_gap_recovery_hypothesis_min_gap_s_);
  }
  if (initialization_warmup_active) {
    RegistrationResult result;
    result.rejection_reason = "initialization_warmup";
    result.input_clouds_expected = cloud_set.expected_cloud_count;
    result.input_clouds_merged = cloud_set.merged_cloud_count;
    result.max_input_cloud_stamp_delta_s = cloud_set.max_stamp_delta_s;
    result.cloud_sync_wait_timed_out = cloud_set.wait_timed_out;
    result.cloud_sync_watermark_resolved = cloud_set.watermark_resolved;
    result.primary_cloud_gap_before = cloud_set.terrain_continuity_break_before;
    result.inter_lidar_wheel_wait_timed_out = cloud_set.wheel_wait_timed_out;
    result.deskew_imu_watermark_waited = cloud_set.imu_watermark_waited;
    result.deskew_imu_wait_timed_out = cloud_set.imu_wait_timed_out;
    result.deskew_required_imu_stamp_ns = cloud_set.required_imu_stamp_ns;
    result.deskew_dispatch_imu_stamp_ns = cloud_set.dispatch_imu_stamp_ns;
    result.imu_snapshot_end_stamp_ns = cloud_set.imu_snapshot_end_stamp_ns;
    result.imu_snapshot_sample_count = static_cast<int>(cloud_set.imu_samples.size());
    std::lock_guard<std::mutex> lock(state_mutex_);
    last_registration_ = result;
    publishDiagnostics(stamp, result);
    return;
  }
  bool gravity_sample_stale = false;
  const auto gravity_sample = gravitySampleAt(
    stamp, cloud_set.gravity_samples, gravity_sample_stale);
  std::string imu_attitude_sample_status =
    imu_relative_roll_pitch_constraint_enabled_ ? "unavailable" : "disabled";
  const auto current_imu_world_R_base = imu_relative_roll_pitch_constraint_enabled_ ?
    imuWorldBaseOrientationAt(
    stamp,
    cloud_set.imu_samples,
    imu_relative_attitude_max_time_delta_s_,
    imu_attitude_sample_status) :
    std::nullopt;
  DeskewStats deskew_stats;
  InterLidarMotionStats inter_lidar_motion_stats;
  int merged_cloud_count = 0;
  CloudPtr mapping_scan_base;
  CloudPtr uncompensated_scan_base;
  CloudPtr uncompensated_mapping_scan_base;
  RegistrationOnlyCloudStats registration_only_stats;
  std::vector<TerrainSensorScan> terrain_sensor_scans;
  std::vector<TerrainSensorScan> uncompensated_terrain_sensor_scans;
  auto scan_base = buildBaseFrameScan(
    cloud_set,
    deskew_stats,
    inter_lidar_motion_stats,
    merged_cloud_count,
    mapping_scan_base,
    uncompensated_scan_base,
    uncompensated_mapping_scan_base,
    registration_only_stats,
    terrain_sensor_scans,
    uncompensated_terrain_sensor_scans,
    build_uncompensated_hypothesis);
  if (cloud_set.have_scan_time_bounds) {
    const double scan_start_time_s =
      static_cast<double>(cloud_set.scan_start_stamp_ns) * 1.0e-9;
    const double scan_end_time_s =
      static_cast<double>(cloud_set.scan_end_stamp_ns) * 1.0e-9;
    if (!deskew_stats.have_scan_time_bounds) {
      deskew_stats.scan_start_time_s = scan_start_time_s;
      deskew_stats.scan_end_time_s = scan_end_time_s;
      deskew_stats.have_scan_time_bounds = true;
    } else {
      deskew_stats.scan_start_time_s = std::min(
        deskew_stats.scan_start_time_s, scan_start_time_s);
      deskew_stats.scan_end_time_s = std::max(
        deskew_stats.scan_end_time_s, scan_end_time_s);
    }
    deskew_stats.scan_duration_s = std::max(
      deskew_stats.scan_duration_s,
      deskew_stats.scan_end_time_s - deskew_stats.scan_start_time_s);
  }
  const auto imu_roughness_stats = evaluateAndUpdateImuShock(
    stamp,
    deskew_stats,
    cloud_set.imu_samples);
  int mapping_scan_points_before_downsample = mapping_scan_base ?
    static_cast<int>(mapping_scan_base->size()) : 0;
  int mapping_scan_points_after_downsample = mapping_scan_points_before_downsample;
  auto attach_input_stats =
    [this, &deskew_stats, &inter_lidar_motion_stats, &cloud_set, &merged_cloud_count,
      &registration_only_stats, &mapping_scan_points_before_downsample,
      &mapping_scan_points_after_downsample](
    RegistrationResult & result)
    {
      applyDeskewStatsToResult(deskew_stats, result);
      applyInterLidarMotionStatsToResult(inter_lidar_motion_stats, result);
      result.input_clouds_expected = cloud_set.expected_cloud_count;
      result.input_clouds_merged = merged_cloud_count;
      result.registration_only_clouds_merged = registration_only_stats.clouds_merged;
      result.registration_only_points_before_balance =
        registration_only_stats.points_before_balance;
      result.registration_only_points_after_balance =
        registration_only_stats.points_after_balance;
      result.mapping_scan_points_before_downsample =
        mapping_scan_points_before_downsample;
      result.mapping_scan_points_after_downsample =
        mapping_scan_points_after_downsample;
      result.max_input_cloud_stamp_delta_s = cloud_set.max_stamp_delta_s;
      result.cloud_sync_wait_timed_out = cloud_set.wait_timed_out;
      result.cloud_sync_watermark_resolved = cloud_set.watermark_resolved;
      result.primary_cloud_gap_before = cloud_set.terrain_continuity_break_before;
      result.inter_lidar_wheel_wait_timed_out = cloud_set.wheel_wait_timed_out;
      result.deskew_imu_watermark_waited = cloud_set.imu_watermark_waited;
      result.deskew_imu_wait_timed_out = cloud_set.imu_wait_timed_out;
      result.deskew_required_imu_stamp_ns = cloud_set.required_imu_stamp_ns;
      result.deskew_dispatch_imu_stamp_ns = cloud_set.dispatch_imu_stamp_ns;
      result.imu_snapshot_end_stamp_ns = cloud_set.imu_snapshot_end_stamp_ns;
      result.imu_snapshot_sample_count = static_cast<int>(cloud_set.imu_samples.size());
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
  auto attach_imu_roughness_stats =
    [this, &imu_roughness_stats](RegistrationResult & registration_result) {
      applyImuRoughnessStatsToResult(imu_roughness_stats, registration_result);
    };
  auto attach_imu_attitude_availability =
    [this, &current_imu_world_R_base, &imu_attitude_sample_status](
    RegistrationResult & registration_result) {
      if (!imu_relative_roll_pitch_constraint_enabled_) {
        registration_result.imu_relative_attitude_constraint_status = "disabled";
      } else if (!current_imu_world_R_base) {
        registration_result.imu_relative_attitude_constraint_status =
          "sample_" + imu_attitude_sample_status;
      } else if (!have_initial_imu_world_R_base_) {
        registration_result.imu_relative_attitude_constraint_status =
          "reference_unavailable";
      } else {
        registration_result.imu_relative_attitude_constraint_status =
          "available_not_evaluated";
      }
    };

  if (!scan_base || static_cast<int>(scan_base->size()) < min_scan_points_) {
    const int scan_points = scan_base ? static_cast<int>(scan_base->size()) : 0;
    RegistrationResult result;
    const bool rejected_for_deskew =
      !scan_base &&
      deskew_enabled_ &&
      reject_cloud_on_deskew_failure_ &&
      deskew_stats.attempted &&
      !deskew_stats.applied;
    result.rejection_reason = rejected_for_deskew ?
      "deskew_" + (deskew_stats.status.empty() ? std::string("failed") : deskew_stats.status) :
      "scan_too_sparse_before_downsample";
    result.scan_points_before_downsample = scan_points;
    result.scan_points_after_downsample = scan_points;
    result.scan_voxel_leaf_m = scan_voxel_leaf_m_;
    attach_input_stats(result);
    attach_imu_roughness_stats(result);
    attach_gravity_stats(result);
    attach_imu_attitude_availability(result);
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
      "Rejected scan before registration: reason=%s points=%d min_points=%d",
      result.rejection_reason.c_str(),
      scan_points,
      min_scan_points_);
    return;
  }

  int scan_points_before_downsample = static_cast<int>(scan_base->size());
  const int uncompensated_scan_points_before_downsample =
    uncompensated_scan_base ? static_cast<int>(uncompensated_scan_base->size()) : 0;
  double effective_scan_voxel_leaf_m = scan_voxel_leaf_m_;
  scan_base = downsampleScanAdaptive(scan_base, effective_scan_voxel_leaf_m);
  mapping_scan_base = downsampleCloud(mapping_scan_base, scan_voxel_leaf_m_);
  mapping_scan_points_after_downsample = mapping_scan_base ?
    static_cast<int>(mapping_scan_base->size()) : 0;
  if (uncompensated_scan_base) {
    uncompensated_scan_base = downsampleCloud(
      uncompensated_scan_base,
      effective_scan_voxel_leaf_m);
    if (static_cast<int>(uncompensated_scan_base->size()) < min_scan_points_) {
      uncompensated_scan_base.reset();
    }
  }
  if (uncompensated_mapping_scan_base) {
    uncompensated_mapping_scan_base = downsampleCloud(
      uncompensated_mapping_scan_base,
      scan_voxel_leaf_m_);
  }
  int scan_points_after_downsample = static_cast<int>(scan_base->size());
  const bool scan_voxel_adapted = std::abs(effective_scan_voxel_leaf_m - scan_voxel_leaf_m_) >
    std::numeric_limits<double>::epsilon();

  if (publish_preprocessed_scan_ &&
    scan_base && static_cast<int>(scan_base->size()) >= min_scan_points_)
  {
    publishCloud(preprocessed_scan_pub_, scan_base, base_frame_, stamp);
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  if (static_cast<int>(scan_base->size()) < min_scan_points_) {
    RegistrationResult result;
    result.rejection_reason = "scan_too_sparse_after_downsample";
    result.scan_points_before_downsample = scan_points_before_downsample;
    result.scan_points_after_downsample = scan_points_after_downsample;
    result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    result.scan_voxel_adapted = scan_voxel_adapted;
    attach_input_stats(result);
    attach_imu_roughness_stats(result);
    attach_gravity_stats(result);
    attach_imu_attitude_availability(result);
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

  // Registration-only support surfaces improve geometric alignment, but they
  // are intentionally excluded from place retrieval so a large common floor
  // plane cannot dominate the descriptor.
  auto scan_descriptor = computeScanDescriptor(mapping_scan_base);
  const bool have_previous_scan_stamp = have_last_processed_scan_stamp_;
  const rclcpp::Time previous_scan_stamp = last_processed_scan_stamp_;
  const double registration_scan_gap_s = have_previous_scan_stamp ?
    std::max(0.0, (stamp - previous_scan_stamp).seconds()) : 0.0;
  last_processed_scan_stamp_ = stamp;
  have_last_processed_scan_stamp_ = true;
  auto constrain_candidate_pose =
    [this, &gravity_sample, &imu_roughness_stats, &current_imu_world_R_base,
      &imu_attitude_sample_status](
    const Eigen::Isometry3d & candidate,
    const Eigen::Isometry3d & planar_reference,
    RegistrationResult * registration_result) {
      const Eigen::Isometry3d prior_constrained =
        constrainMotionPriorFallback(candidate);
      const Eigen::Isometry3d imu_constrained = constrainImuRelativeAttitude(
        prior_constrained,
        current_imu_world_R_base,
        imu_attitude_sample_status,
        registration_result);
      const GroundMotionConstraintResult ground_constraint =
        constrainGroundVehicleMotion(
        imu_constrained,
        map_T_base_,
        ground_motion_constraint_config_,
        imu_roughness_stats.recent_event);
      if (registration_result != nullptr) {
        registration_result->ground_motion_constraint_evaluated =
          ground_constraint.evaluated;
        registration_result->ground_motion_constraint_applied =
          ground_constraint.applied;
        registration_result->ground_motion_constraint_roughness_relaxed =
          ground_constraint.roughness_relaxed;
        registration_result->ground_motion_raw_body_z_step_m =
          ground_constraint.raw_body_z_step_m;
        registration_result->ground_motion_constrained_body_z_step_m =
          ground_constraint.constrained_body_z_step_m;
        registration_result->ground_motion_max_body_z_step_m =
          ground_constraint.max_body_z_step_m;
        registration_result->ground_motion_constraint_status =
          ground_constraint.status;
      }
      return constrainPlanarPose(
        ground_constraint.pose, planar_reference, gravity_sample);
    };

  if (!initialized_ && quarantine_keyframes_on_shock_ && imu_roughness_stats.shock_active) {
    RegistrationResult result;
    result.rejection_reason = "initialization_waiting_for_imu_stability";
    result.shock_quarantined = true;
    result.shock_quarantine_reason =
      "imu_" + imu_roughness_stats.shock_phase;
    result.keyframe_quarantined = true;
    result.keyframe_quarantine_reason = result.shock_quarantine_reason;
    result.scan_points_before_downsample = scan_points_before_downsample;
    result.scan_points_after_downsample = scan_points_after_downsample;
    result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    result.scan_voxel_adapted = scan_voxel_adapted;
    attach_input_stats(result);
    attach_imu_roughness_stats(result);
    attach_gravity_stats(result);
    attach_imu_attitude_availability(result);
    last_registration_ = result;
    publishDiagnostics(stamp, result);
    return;
  }

  if (!initialized_) {
    if (imu_relative_roll_pitch_constraint_enabled_ && !current_imu_world_R_base) {
      RegistrationResult result;
      result.rejection_reason = "initialization_waiting_for_imu_orientation";
      result.scan_points_before_downsample = scan_points_before_downsample;
      result.scan_points_after_downsample = scan_points_after_downsample;
      result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
      result.scan_voxel_adapted = scan_voxel_adapted;
      attach_input_stats(result);
      attach_imu_roughness_stats(result);
      attach_gravity_stats(result);
      attach_imu_attitude_availability(result);
      last_registration_ = result;
      publishDiagnostics(stamp, result);
      return;
    }
    if (current_imu_world_R_base) {
      initial_imu_world_R_base_ = *current_imu_world_R_base;
      have_initial_imu_world_R_base_ = true;
    }
    initializeFromFirstScan(
      scan_base,
      mapping_scan_base,
      registration_only_stats.points_after_balance,
      terrain_sensor_scans,
      stamp,
      gravity_sample,
      gravity_sample_stale);
    if (constant_velocity_rebase_pending_) {
      rebaseAcceptedPoseHistory();
      constant_velocity_rebase_pending_ = false;
    }
    last_registration_.scan_points_before_downsample = scan_points_before_downsample;
    last_registration_.scan_points_after_downsample = scan_points_after_downsample;
    last_registration_.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    last_registration_.scan_voxel_adapted = scan_voxel_adapted;
    last_registration_.keyframe_added = true;
    attach_input_stats(last_registration_);
    attach_imu_roughness_stats(last_registration_);
    attach_gravity_stats(last_registration_);
    last_registration_.imu_relative_attitude_constraint_status =
      imu_relative_roll_pitch_constraint_enabled_ ? "reference_initialized" : "disabled";
    publishState(stamp, last_registration_);
    publishDiagnostics(stamp, last_registration_);
    return;
  }

  bool used_scan_to_scan_prediction = false;
  bool used_imu_rotation_prediction = false;
  bool constant_velocity_prediction_suppressed = false;
  std::string imu_rotation_prediction_status{"disabled"};
  double scan_to_scan_reference_age_s = 0.0;
  std::string scan_to_scan_prediction_status{"disabled"};
  auto attach_prediction_stats = [&](RegistrationResult & registration_result) {
      registration_result.scan_to_scan_prediction_used = used_scan_to_scan_prediction;
      registration_result.scan_to_scan_reference_age_s = scan_to_scan_reference_age_s;
      registration_result.scan_to_scan_prediction_status = scan_to_scan_prediction_status;
      registration_result.imu_rotation_prediction_used = used_imu_rotation_prediction;
      registration_result.imu_rotation_prediction_status = imu_rotation_prediction_status;
      registration_result.constant_velocity_prediction_suppressed_by_shock =
        constant_velocity_prediction_suppressed;
    };
  const auto prediction = predictWithMotionPriors(
    scan_base,
    cloud_set.imu_samples,
    previous_scan_stamp,
    stamp,
    have_previous_scan_stamp,
    suppress_constant_velocity_on_imu_shock_ &&
    (imu_roughness_stats.shock_active || constant_velocity_rebase_pending_),
    used_scan_to_scan_prediction,
    used_imu_rotation_prediction,
    constant_velocity_prediction_suppressed,
    imu_rotation_prediction_status,
    scan_to_scan_reference_age_s,
    scan_to_scan_prediction_status);
  const bool compatible_submap_available = hasCompatibleSubmap(prediction);
  auto local_map = buildLocalMap(prediction);
  if (!local_map || static_cast<int>(local_map->size()) < min_scan_points_) {
    const bool spatial_transition_bootstrap =
      submap_enabled_ &&
      spatial_mapping_config_.mode != MappingSpatialMode::kPlanar &&
      !compatible_submap_available;
    if (quarantine_keyframes_on_shock_ && imu_roughness_stats.shock_active) {
      RegistrationResult result;
      result.rejection_reason = "local_map_bootstrap_quarantined_on_imu_shock";
      result.shock_quarantined = true;
      result.shock_quarantine_reason =
        "imu_" + imu_roughness_stats.shock_phase;
      result.keyframe_quarantined = true;
      result.keyframe_quarantine_reason = result.shock_quarantine_reason;
      result.scan_points_before_downsample = scan_points_before_downsample;
      result.scan_points_after_downsample = scan_points_after_downsample;
      result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
      result.scan_voxel_adapted = scan_voxel_adapted;
      attach_prediction_stats(result);
      attach_input_stats(result);
      attach_imu_roughness_stats(result);
      attach_gravity_stats(result);
      latest_local_map_ = local_map;
      last_registration_ = result;
      publishDiagnostics(stamp, result);
      return;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      spatial_transition_bootstrap ?
      "No spatially compatible submap; starting a new submap from the motion prior" :
      "Local map has too few points; adding scan as a keyframe without registration");
    RegistrationResult result;
    map_T_base_ = constrain_candidate_pose(
      prediction, map_T_base_, &result);
    commitWheelPrediction();
    consecutive_pose_quarantine_count_ = 0;
    recordAcceptedPoseOrRebaseAfterShock();
    recordTrackingScan(scan_base, map_T_base_, stamp);
    if (spatial_transition_bootstrap) {
      closeActiveSubmapForSpatialTransition();
    }
    addKeyframe(
      scan_base,
      mapping_scan_base,
      registration_only_stats.points_after_balance,
      terrain_sensor_scans,
      stamp,
      gravity_sample,
      gravity_sample_stale);
    result.converged = true;
    result.accepted = true;
    result.local_map_spatial_transition_bootstrap = spatial_transition_bootstrap;
    result.map_T_base = map_T_base_;
    result.rejection_reason = spatial_transition_bootstrap ?
      "local_map_spatial_transition_bootstrap" : "local_map_too_small";
    result.scan_points_before_downsample = scan_points_before_downsample;
    result.scan_points_after_downsample = scan_points_after_downsample;
    result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
    result.scan_voxel_adapted = scan_voxel_adapted;
    result.keyframe_added = true;
    attach_prediction_stats(result);
    attach_input_stats(result);
    attach_imu_roughness_stats(result);
    attach_gravity_stats(result);
    last_registration_ = result;
    publishState(stamp, result);
    publishDiagnostics(stamp, result);
    return;
  }

  const auto registration_start = std::chrono::steady_clock::now();
  auto result = registerScanToMap(
    scan_base,
    local_map,
    prediction,
    registration_scan_gap_s);
  result.registration_runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - registration_start).count();
  if (
    uncompensated_scan_base &&
    inter_lidar_motion_stats.applied_count > 0 &&
    registration_scan_gap_s >=
    std::max(0.0, inter_lidar_gap_recovery_hypothesis_min_gap_s_))
  {
    const auto hypothesis_start = std::chrono::steady_clock::now();
    const auto uncompensated_registration_start = std::chrono::steady_clock::now();
    auto uncompensated_result = registerScanToMap(
      uncompensated_scan_base,
      local_map,
      prediction,
      registration_scan_gap_s);
    uncompensated_result.registration_runtime_ms =
      std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - uncompensated_registration_start).count();
    const auto hypothesis_end = std::chrono::steady_clock::now();
    const RegistrationHypothesisQuality compensated_quality{
      result.accepted, result.fitness, result.inlier_ratio};
    const RegistrationHypothesisQuality uncompensated_quality{
      uncompensated_result.accepted,
      uncompensated_result.fitness,
      uncompensated_result.inlier_ratio};
    const auto hypothesis_decision = selectRegistrationHypothesis(
      compensated_quality,
      uncompensated_quality,
      inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_);
    const auto finite_score_or_negative = [](const double score) {
        return std::isfinite(score) ? score : -1.0;
      };
    inter_lidar_motion_stats.gap_recovery_hypothesis_attempted = true;
    inter_lidar_motion_stats.gap_recovery_compensated_score =
      finite_score_or_negative(hypothesis_decision.primary_score);
    inter_lidar_motion_stats.gap_recovery_uncompensated_score =
      finite_score_or_negative(hypothesis_decision.alternative_score);
    inter_lidar_motion_stats.gap_recovery_uncompensated_fitness =
      uncompensated_result.fitness;
    inter_lidar_motion_stats.gap_recovery_uncompensated_inlier_ratio =
      uncompensated_result.inlier_ratio;
    inter_lidar_motion_stats.gap_recovery_uncompensated_inlier_rmse_m =
      uncompensated_result.inlier_rmse_m;
    inter_lidar_motion_stats.gap_recovery_runtime_ms =
      std::chrono::duration<double, std::milli>(
      hypothesis_end - hypothesis_start).count();
    inter_lidar_motion_stats.gap_recovery_selected_input = "compensated";
    if (hypothesis_decision.select_alternative) {
      result = std::move(uncompensated_result);
      scan_base = uncompensated_scan_base;
      mapping_scan_base = uncompensated_mapping_scan_base;
      mapping_scan_points_after_downsample = mapping_scan_base ?
        static_cast<int>(mapping_scan_base->size()) : 0;
      terrain_sensor_scans = uncompensated_terrain_sensor_scans;
      scan_points_before_downsample = uncompensated_scan_points_before_downsample;
      scan_points_after_downsample = static_cast<int>(scan_base->size());
      scan_descriptor = computeScanDescriptor(mapping_scan_base);
      inter_lidar_motion_stats.gap_recovery_selected_uncompensated = true;
      inter_lidar_motion_stats.gap_recovery_selected_input = "uncompensated";
    }
  }
  result.scan_points_before_downsample = scan_points_before_downsample;
  result.scan_points_after_downsample = scan_points_after_downsample;
  result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
  result.scan_voxel_adapted = scan_voxel_adapted;
  attach_prediction_stats(result);
  if (result.accepted) {
    const Eigen::Isometry3d unconstrained_pose = result.map_T_base;
    result.map_T_base = constrain_candidate_pose(
      result.map_T_base, prediction, &result);
    if (!result.map_T_base.matrix().isApprox(
        unconstrained_pose.matrix(), 1.0e-9))
    {
      result.ground_motion_post_constraint_quality_evaluated = true;
      refreshRegistrationQualityAtPose(
        scan_base, local_map, prediction, result);
      result.accepted = registrationCorrectionWithinLimits(result);
      result.ground_motion_post_constraint_quality_accepted = result.accepted;
      if (!result.accepted) {
        result.rejection_reason =
          "post_motion_constraint_" + result.rejection_reason;
      }
    }
  }
  std::optional<RegistrationResult> pre_place_recognition_result;

  // Shock gating must be available before deciding whether place recognition,
  // pose commit, or persistent-map mutation is allowed for this scan.
  attach_imu_roughness_stats(result);
  const std::string local_result_shock_reason = shockQuarantineReason(result);
  const bool local_result_shock_relocalization =
    result.accepted && !local_result_shock_reason.empty();
  const bool local_result_needs_relocalization =
    !result.accepted || local_result_shock_relocalization;
  const Eigen::Isometry3d revisit_probe_pose = result.accepted ?
    constrain_candidate_pose(result.map_T_base, prediction, nullptr) :
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
  const bool place_recognition_requested =
    local_result_needs_relocalization || periodic_place_recognition;
  if (place_recognition_enabled_ && place_recognition_requested &&
    imu_roughness_stats.shock_active)
  {
    result.place_recognition_skipped = true;
    result.place_recognition_trigger =
      local_result_needs_relocalization ? "relocalization" : "periodic";
    result.place_recognition_skip_reason = "imu_shock_persistent_state_quarantine";
  } else if (place_recognition_enabled_ && place_recognition_requested) {
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
        mapping_scan_base,
        scan_descriptor,
        prediction,
        stamp,
        registration_only_stats.points_after_balance,
        place_result);
      place_result.place_recognition_trigger = place_trigger;
      result.place_recognition_candidates = place_result.place_recognition_candidates;
      result.place_recognition_eligible_descriptors =
        place_result.place_recognition_eligible_descriptors;
      result.place_recognition_registration_only_vertical_recovery_descriptors =
        place_result.place_recognition_registration_only_vertical_recovery_descriptors;
      result.place_recognition_registration_only_vertical_recovery_used =
        place_result.place_recognition_registration_only_vertical_recovery_used;
      result.place_recognition_registration_only_consistency_relaxation_used =
        place_result.place_recognition_registration_only_consistency_relaxation_used;
      result.place_recognition_mapping_refinement_attempted =
        place_result.place_recognition_mapping_refinement_attempted;
      result.place_recognition_mapping_refinement_applied =
        place_result.place_recognition_mapping_refinement_applied;
      result.place_recognition_mapping_refinement_verified =
        place_result.place_recognition_mapping_refinement_verified;
      result.place_recognition_large_pose_jump_provisional =
        place_result.place_recognition_large_pose_jump_provisional;
      result.place_recognition_large_pose_jump_accepted =
        place_result.place_recognition_large_pose_jump_accepted;
      result.place_recognition_mapping_refinement_status =
        place_result.place_recognition_mapping_refinement_status;
      result.place_recognition_mapping_refinement_source_points =
        place_result.place_recognition_mapping_refinement_source_points;
      result.place_recognition_mapping_refinement_target_points =
        place_result.place_recognition_mapping_refinement_target_points;
      result.place_recognition_mapping_refinement_observable_rank =
        place_result.place_recognition_mapping_refinement_observable_rank;
      result.place_recognition_mapping_refinement_before_inlier_ratio =
        place_result.place_recognition_mapping_refinement_before_inlier_ratio;
      result.place_recognition_mapping_refinement_before_inlier_rmse_m =
        place_result.place_recognition_mapping_refinement_before_inlier_rmse_m;
      result.place_recognition_mapping_refinement_after_inlier_ratio =
        place_result.place_recognition_mapping_refinement_after_inlier_ratio;
      result.place_recognition_mapping_refinement_after_inlier_rmse_m =
        place_result.place_recognition_mapping_refinement_after_inlier_rmse_m;
      result.place_recognition_mapping_refinement_before_target_coverage_ratio =
        place_result.place_recognition_mapping_refinement_before_target_coverage_ratio;
      result.place_recognition_mapping_refinement_after_target_coverage_ratio =
        place_result.place_recognition_mapping_refinement_after_target_coverage_ratio;
      result.place_recognition_mapping_refinement_before_score =
        place_result.place_recognition_mapping_refinement_before_score;
      result.place_recognition_mapping_refinement_after_score =
        place_result.place_recognition_mapping_refinement_after_score;
      result.place_recognition_mapping_refinement_primary_before_score =
        place_result.place_recognition_mapping_refinement_primary_before_score;
      result.place_recognition_mapping_refinement_primary_after_score =
        place_result.place_recognition_mapping_refinement_primary_after_score;
      result.place_recognition_mapping_refinement_correction_translation_m =
        place_result.place_recognition_mapping_refinement_correction_translation_m;
      result.place_recognition_mapping_refinement_correction_rotation_deg =
        place_result.place_recognition_mapping_refinement_correction_rotation_deg;
      result.place_recognition_mapping_refinement_hessian_condition =
        place_result.place_recognition_mapping_refinement_hessian_condition;
      result.place_recognition_mapping_refinement_runtime_ms =
        place_result.place_recognition_mapping_refinement_runtime_ms;
      result.place_recognition_candidate_registration_status =
        place_result.place_recognition_candidate_registration_status;
      result.place_recognition_candidate_registration_inlier_ratio =
        place_result.place_recognition_candidate_registration_inlier_ratio;
      result.place_recognition_candidate_registration_inlier_rmse_m =
        place_result.place_recognition_candidate_registration_inlier_rmse_m;
      result.place_recognition_candidate_pose_jump_translation_m =
        place_result.place_recognition_candidate_pose_jump_translation_m;
      result.place_recognition_candidate_pose_jump_rotation_deg =
        place_result.place_recognition_candidate_pose_jump_rotation_deg;
      result.place_recognition_alignment_correction_translation_m =
        place_result.place_recognition_alignment_correction_translation_m;
      result.place_recognition_alignment_correction_rotation_deg =
        place_result.place_recognition_alignment_correction_rotation_deg;
      result.place_recognition_candidate_index = place_result.place_recognition_candidate_index;
      result.place_recognition_candidate_submap_id =
        place_result.place_recognition_candidate_submap_id;
      result.place_recognition_candidate_consistency_count =
        place_result.place_recognition_candidate_consistency_count;
      result.place_recognition_candidate_consistency_translation_m =
        place_result.place_recognition_candidate_consistency_translation_m;
      result.place_recognition_candidate_consistency_rotation_deg =
        place_result.place_recognition_candidate_consistency_rotation_deg;
      result.place_recognition_candidate_consistency_translation_limit_m =
        place_result.place_recognition_candidate_consistency_translation_limit_m;
      result.place_recognition_descriptor_distance =
        place_result.place_recognition_descriptor_distance;
      result.place_recognition_nearest_descriptor_distance =
        place_result.place_recognition_nearest_descriptor_distance;
      result.place_recognition_yaw_offset_deg = place_result.place_recognition_yaw_offset_deg;
      if (place_result.place_recognition_skipped) {
        result.place_recognition_skipped = true;
        result.place_recognition_skip_reason = place_result.place_recognition_skip_reason;
      }
      if (place_accepted) {
        pre_place_recognition_result = result;
        place_result.scan_points_before_downsample = scan_points_before_downsample;
        place_result.scan_points_after_downsample = scan_points_after_downsample;
        place_result.scan_voxel_leaf_m = effective_scan_voxel_leaf_m;
        place_result.scan_voxel_adapted = scan_voxel_adapted;
        place_result.place_recognition_attempted = true;
        attach_prediction_stats(place_result);
        attach_input_stats(place_result);
        attach_imu_roughness_stats(place_result);
        attach_gravity_stats(place_result);
        result = place_result;
      }
    }
  }
  attach_input_stats(result);
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
    // A strictly verified historical large-loop measurement must reach the
    // graph unchanged. Accepted sequential keyframes retain the frontend
    // IMU/ground constraints, and the graph result has a post-solve ground
    // sequence gate. Re-clamping here would mix the historical loop target
    // with the still-drifted active pose.
    const Eigen::Isometry3d unconstrained_place_pose = result.map_T_base;
    result.place_recognition_verified_large_loop_measurement_preserved =
      shouldPreserveVerifiedLargeLoopMeasurement(
      result.place_recognition_mapping_refinement_verified,
      result.place_recognition_large_pose_jump_accepted);
    if (!result.place_recognition_verified_large_loop_measurement_preserved) {
      result.map_T_base = constrain_candidate_pose(
        result.map_T_base, prediction, &result);
    }
    bool post_constraint_place_quality_accepted = true;
    if (!result.map_T_base.matrix().isApprox(
        unconstrained_place_pose.matrix(), 1.0e-9))
    {
      result.ground_motion_post_constraint_quality_evaluated = true;
      const auto place_validation_map = buildLocalMap(result.map_T_base);
      refreshRegistrationQualityAtPose(
        scan_base, place_validation_map, prediction, result);
      const double max_place_rmse =
        std::max(0.0, place_recognition_max_inlier_rmse_m_);
      post_constraint_place_quality_accepted =
        result.inlier_count > 0 &&
        std::isfinite(result.inlier_ratio) &&
        result.inlier_ratio >= place_recognition_min_inlier_ratio_ &&
        std::isfinite(result.inlier_rmse_m) &&
        (max_place_rmse <= 0.0 || result.inlier_rmse_m <= max_place_rmse);
      result.ground_motion_post_constraint_quality_accepted =
        post_constraint_place_quality_accepted;
    }
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
    if (!post_constraint_place_quality_accepted) {
      result.pose_graph_correction_applied = false;
      result.pose_graph_rejection_reason = "post_motion_constraint_quality";
    } else if (attempt_pose_graph_correction) {
      applyPoseGraphCorrection(
        stamp,
        prediction,
        gravity_sample,
        current_imu_world_R_base,
        imu_attitude_sample_status,
        result);
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
      result.place_recognition_eligible_descriptors =
        rejected_place_result.place_recognition_eligible_descriptors;
      result.place_recognition_registration_only_vertical_recovery_descriptors =
        rejected_place_result.place_recognition_registration_only_vertical_recovery_descriptors;
      result.place_recognition_registration_only_vertical_recovery_used =
        rejected_place_result.place_recognition_registration_only_vertical_recovery_used;
      result.place_recognition_registration_only_consistency_relaxation_used =
        rejected_place_result.place_recognition_registration_only_consistency_relaxation_used;
      result.place_recognition_mapping_refinement_attempted =
        rejected_place_result.place_recognition_mapping_refinement_attempted;
      result.place_recognition_mapping_refinement_applied =
        rejected_place_result.place_recognition_mapping_refinement_applied;
      result.place_recognition_mapping_refinement_verified =
        rejected_place_result.place_recognition_mapping_refinement_verified;
      result.place_recognition_large_pose_jump_provisional =
        rejected_place_result.place_recognition_large_pose_jump_provisional;
      result.place_recognition_large_pose_jump_accepted =
        rejected_place_result.place_recognition_large_pose_jump_accepted;
      result.place_recognition_verified_large_loop_measurement_preserved =
        rejected_place_result.place_recognition_verified_large_loop_measurement_preserved;
      result.place_recognition_mapping_refinement_status =
        rejected_place_result.place_recognition_mapping_refinement_status;
      result.place_recognition_mapping_refinement_source_points =
        rejected_place_result.place_recognition_mapping_refinement_source_points;
      result.place_recognition_mapping_refinement_target_points =
        rejected_place_result.place_recognition_mapping_refinement_target_points;
      result.place_recognition_mapping_refinement_observable_rank =
        rejected_place_result.place_recognition_mapping_refinement_observable_rank;
      result.place_recognition_mapping_refinement_before_inlier_ratio =
        rejected_place_result.place_recognition_mapping_refinement_before_inlier_ratio;
      result.place_recognition_mapping_refinement_before_inlier_rmse_m =
        rejected_place_result.place_recognition_mapping_refinement_before_inlier_rmse_m;
      result.place_recognition_mapping_refinement_after_inlier_ratio =
        rejected_place_result.place_recognition_mapping_refinement_after_inlier_ratio;
      result.place_recognition_mapping_refinement_after_inlier_rmse_m =
        rejected_place_result.place_recognition_mapping_refinement_after_inlier_rmse_m;
      result.place_recognition_mapping_refinement_before_target_coverage_ratio =
        rejected_place_result.
        place_recognition_mapping_refinement_before_target_coverage_ratio;
      result.place_recognition_mapping_refinement_after_target_coverage_ratio =
        rejected_place_result.
        place_recognition_mapping_refinement_after_target_coverage_ratio;
      result.place_recognition_mapping_refinement_before_score =
        rejected_place_result.place_recognition_mapping_refinement_before_score;
      result.place_recognition_mapping_refinement_after_score =
        rejected_place_result.place_recognition_mapping_refinement_after_score;
      result.place_recognition_mapping_refinement_primary_before_score =
        rejected_place_result.place_recognition_mapping_refinement_primary_before_score;
      result.place_recognition_mapping_refinement_primary_after_score =
        rejected_place_result.place_recognition_mapping_refinement_primary_after_score;
      result.place_recognition_mapping_refinement_correction_translation_m =
        rejected_place_result.place_recognition_mapping_refinement_correction_translation_m;
      result.place_recognition_mapping_refinement_correction_rotation_deg =
        rejected_place_result.place_recognition_mapping_refinement_correction_rotation_deg;
      result.place_recognition_mapping_refinement_hessian_condition =
        rejected_place_result.place_recognition_mapping_refinement_hessian_condition;
      result.place_recognition_mapping_refinement_runtime_ms =
        rejected_place_result.place_recognition_mapping_refinement_runtime_ms;
      result.place_recognition_candidate_registration_status =
        rejected_place_result.place_recognition_candidate_registration_status;
      result.place_recognition_candidate_registration_inlier_ratio =
        rejected_place_result.place_recognition_candidate_registration_inlier_ratio;
      result.place_recognition_candidate_registration_inlier_rmse_m =
        rejected_place_result.place_recognition_candidate_registration_inlier_rmse_m;
      result.place_recognition_candidate_pose_jump_translation_m =
        rejected_place_result.place_recognition_candidate_pose_jump_translation_m;
      result.place_recognition_candidate_pose_jump_rotation_deg =
        rejected_place_result.place_recognition_candidate_pose_jump_rotation_deg;
      result.place_recognition_alignment_correction_translation_m =
        rejected_place_result.place_recognition_alignment_correction_translation_m;
      result.place_recognition_alignment_correction_rotation_deg =
        rejected_place_result.place_recognition_alignment_correction_rotation_deg;
      result.place_recognition_candidate_index =
        rejected_place_result.place_recognition_candidate_index;
      result.place_recognition_candidate_submap_id =
        rejected_place_result.place_recognition_candidate_submap_id;
      result.place_recognition_candidate_consistency_count =
        rejected_place_result.place_recognition_candidate_consistency_count;
      result.place_recognition_candidate_consistency_translation_m =
        rejected_place_result.place_recognition_candidate_consistency_translation_m;
      result.place_recognition_candidate_consistency_rotation_deg =
        rejected_place_result.place_recognition_candidate_consistency_rotation_deg;
      result.place_recognition_candidate_consistency_translation_limit_m =
        rejected_place_result.place_recognition_candidate_consistency_translation_limit_m;
      result.place_recognition_descriptor_distance =
        rejected_place_result.place_recognition_descriptor_distance;
      result.place_recognition_nearest_descriptor_distance =
        rejected_place_result.place_recognition_nearest_descriptor_distance;
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
      result.pose_graph_correction_translation_m =
        rejected_place_result.pose_graph_correction_translation_m;
      result.pose_graph_correction_rotation_deg =
        rejected_place_result.pose_graph_correction_rotation_deg;
      result.pose_graph_constraint_inlier_ratio =
        rejected_place_result.pose_graph_constraint_inlier_ratio;
      result.pose_graph_constraint_inlier_rmse_m =
        rejected_place_result.pose_graph_constraint_inlier_rmse_m;
      result.pose_graph_constraint_target_coverage_ratio =
        rejected_place_result.pose_graph_constraint_target_coverage_ratio;
      result.pose_graph_minimum_interval_override_applied =
        rejected_place_result.pose_graph_minimum_interval_override_applied;
      result.pose_graph_minimum_interval_elapsed_s =
        rejected_place_result.pose_graph_minimum_interval_elapsed_s;
      result.pose_graph_minimum_interval_status =
        rejected_place_result.pose_graph_minimum_interval_status;
      result.pose_graph_duplicate_submap_pair_override_applied =
        rejected_place_result.pose_graph_duplicate_submap_pair_override_applied;
      result.pose_graph_duplicate_submap_pair_previous_edges =
        rejected_place_result.pose_graph_duplicate_submap_pair_previous_edges;
      result.pose_graph_duplicate_submap_pair_status =
        rejected_place_result.pose_graph_duplicate_submap_pair_status;
      result.pose_graph_same_layer_z_constraint_evaluated =
        rejected_place_result.pose_graph_same_layer_z_constraint_evaluated;
      result.pose_graph_same_layer_z_constraint_applied =
        rejected_place_result.pose_graph_same_layer_z_constraint_applied;
      result.pose_graph_same_layer_z_seed_applied =
        rejected_place_result.pose_graph_same_layer_z_seed_applied;
      result.pose_graph_high_confidence_large_loop =
        rejected_place_result.pose_graph_high_confidence_large_loop;
      result.pose_graph_distributed_loop_seed_applied =
        rejected_place_result.pose_graph_distributed_loop_seed_applied;
      result.pose_graph_same_layer_reference_keyframe_index =
        rejected_place_result.pose_graph_same_layer_reference_keyframe_index;
      result.pose_graph_same_layer_reference_xy_distance_m =
        rejected_place_result.pose_graph_same_layer_reference_xy_distance_m;
      result.pose_graph_same_layer_matched_current_z_before_m =
        rejected_place_result.pose_graph_same_layer_matched_current_z_before_m;
      result.pose_graph_same_layer_reference_z_m =
        rejected_place_result.pose_graph_same_layer_reference_z_m;
      result.pose_graph_same_layer_requested_z_correction_m =
        rejected_place_result.pose_graph_same_layer_requested_z_correction_m;
      result.pose_graph_same_layer_graph_target_z_m =
        rejected_place_result.pose_graph_same_layer_graph_target_z_m;
      result.pose_graph_same_layer_z_translation_weight =
        rejected_place_result.pose_graph_same_layer_z_translation_weight;
      result.pose_graph_same_layer_reference_source =
        rejected_place_result.pose_graph_same_layer_reference_source;
      result.pose_graph_correction_keyframes =
        rejected_place_result.pose_graph_correction_keyframes;
      result.pose_graph_optimization_iterations =
        rejected_place_result.pose_graph_optimization_iterations;
      result.pose_graph_solver_stages =
        rejected_place_result.pose_graph_solver_stages;
      result.pose_graph_solver_successful_steps =
        rejected_place_result.pose_graph_solver_successful_steps;
      result.pose_graph_solver_unsuccessful_steps =
        rejected_place_result.pose_graph_solver_unsuccessful_steps;
      result.pose_graph_dcs_downweighted_loop_edges =
        rejected_place_result.pose_graph_dcs_downweighted_loop_edges;
      result.pose_graph_registration_information_edges =
        rejected_place_result.pose_graph_registration_information_edges;
      result.pose_graph_configured_diagonal_information_edges =
        rejected_place_result.pose_graph_configured_diagonal_information_edges;
      result.pose_graph_minimum_registration_observable_rank =
        rejected_place_result.pose_graph_minimum_registration_observable_rank;
      result.pose_graph_maximum_registration_hessian_condition =
        rejected_place_result.pose_graph_maximum_registration_hessian_condition;
      result.pose_graph_minimum_registration_information_gain =
        rejected_place_result.pose_graph_minimum_registration_information_gain;
      result.pose_graph_new_loop_registration_information_used =
        rejected_place_result.pose_graph_new_loop_registration_information_used;
      result.pose_graph_new_loop_information_status =
        rejected_place_result.pose_graph_new_loop_information_status;
      result.pose_graph_new_loop_information_min_gain =
        rejected_place_result.pose_graph_new_loop_information_min_gain;
      result.pose_graph_optimization_runtime_ms =
        rejected_place_result.pose_graph_optimization_runtime_ms;
      result.pose_graph_iteration_runtime_ms =
        rejected_place_result.pose_graph_iteration_runtime_ms;
      result.pose_graph_mean_iteration_runtime_ms =
        rejected_place_result.pose_graph_mean_iteration_runtime_ms;
      result.pose_graph_solver_status = rejected_place_result.pose_graph_solver_status;
      result.pose_graph_convergence_reason =
        rejected_place_result.pose_graph_convergence_reason;
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
      result.pose_graph_residual_before_ground_sequence =
        rejected_place_result.pose_graph_residual_before_ground_sequence;
      result.pose_graph_robust_objective_before_ground_sequence =
        rejected_place_result.pose_graph_robust_objective_before_ground_sequence;
      result.pose_graph_min_loop_robust_weight =
        rejected_place_result.pose_graph_min_loop_robust_weight;
      result.pose_graph_new_loop_robust_weight =
        rejected_place_result.pose_graph_new_loop_robust_weight;
      result.pose_graph_dcs_min_scale =
        rejected_place_result.pose_graph_dcs_min_scale;
      result.pose_graph_new_loop_dcs_scale =
        rejected_place_result.pose_graph_new_loop_dcs_scale;
      result.pose_graph_solver_first_stage_initial_cost =
        rejected_place_result.pose_graph_solver_first_stage_initial_cost;
      result.pose_graph_solver_last_stage_final_cost =
        rejected_place_result.pose_graph_solver_last_stage_final_cost;
      result.pose_graph_solver_final_gradient_max_norm =
        rejected_place_result.pose_graph_solver_final_gradient_max_norm;
      result.pose_graph_max_update_translation_m =
        rejected_place_result.pose_graph_max_update_translation_m;
      result.pose_graph_max_update_rotation_deg =
        rejected_place_result.pose_graph_max_update_rotation_deg;
      result.pose_graph_same_layer_z_seed_correction_m =
        rejected_place_result.pose_graph_same_layer_z_seed_correction_m;
      result.pose_graph_same_layer_z_seed_path_length_m =
        rejected_place_result.pose_graph_same_layer_z_seed_path_length_m;
      result.pose_graph_same_layer_z_seed_max_increment_m =
        rejected_place_result.pose_graph_same_layer_z_seed_max_increment_m;
      result.pose_graph_distributed_loop_seed_path_length_m =
        rejected_place_result.pose_graph_distributed_loop_seed_path_length_m;
      result.pose_graph_distributed_loop_seed_endpoint_translation_m =
        rejected_place_result.pose_graph_distributed_loop_seed_endpoint_translation_m;
      result.pose_graph_distributed_loop_seed_endpoint_rotation_deg =
        rejected_place_result.pose_graph_distributed_loop_seed_endpoint_rotation_deg;
      result.pose_graph_distributed_loop_seed_max_relative_translation_error_m =
        rejected_place_result.
        pose_graph_distributed_loop_seed_max_relative_translation_error_m;
      result.pose_graph_distributed_loop_seed_max_relative_rotation_error_deg =
        rejected_place_result.
        pose_graph_distributed_loop_seed_max_relative_rotation_error_deg;
      result.pose_graph_ground_sequence_applied_edges =
        rejected_place_result.pose_graph_ground_sequence_applied_edges;
      result.pose_graph_ground_sequence_max_raw_body_z_step_m =
        rejected_place_result.pose_graph_ground_sequence_max_raw_body_z_step_m;
      result.pose_graph_ground_sequence_max_constrained_body_z_step_m =
        rejected_place_result.pose_graph_ground_sequence_max_constrained_body_z_step_m;
      result.pose_graph_new_loop_rejected =
        rejected_place_result.pose_graph_new_loop_rejected;
      result.pose_graph_rejection_reason =
        rejected_place_result.pose_graph_rejection_reason;
      if (!result.pose_graph_rejection_reason.empty()) {
        result.place_recognition_skip_reason =
          "pose_graph_" + result.pose_graph_rejection_reason;
      }
      attach_input_stats(result);
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
      map_T_base_ = constrain_candidate_pose(
        prediction, map_T_base_, &result);
      commitWheelPrediction();
      result.accepted = true;
      result.converged = true;
      result.rejection_reason = "scan_to_scan_fallback";
      result.map_T_base = map_T_base_;
      result.shock_quarantine_reason = shockQuarantineReason(result);
      result.shock_quarantined = !result.shock_quarantine_reason.empty();
      if (result.shock_quarantined) {
        constant_velocity_rebase_pending_ = true;
      }
      if (result.shock_quarantined && quarantine_pose_on_shock_) {
        result.pose_quarantined = true;
        result.rejection_reason = "shock_pose_quarantine_scan_to_scan_fallback";
        ++consecutive_pose_quarantine_count_;
        result.consecutive_pose_quarantine_count = consecutive_pose_quarantine_count_;
      } else if (!result.shock_quarantined) {
        consecutive_pose_quarantine_count_ = 0;
        recordAcceptedPoseOrRebaseAfterShock();
        recordTrackingScan(scan_base, map_T_base_, stamp);
      } else if (record_tracking_scan_during_shock_) {
        recordTrackingScan(scan_base, map_T_base_, stamp);
        result.tracking_scan_updated_during_shock = true;
      }
      last_registration_ = result;
      latest_local_map_ = local_map;
      if (add_scan_to_scan_fallback_keyframes_ && shouldAddKeyframe(map_T_base_)) {
        if (!result.shock_quarantined) {
          addKeyframe(
            scan_base,
            mapping_scan_base,
            registration_only_stats.points_after_balance,
            terrain_sensor_scans,
            stamp,
            gravity_sample,
            gravity_sample_stale);
          result.keyframe_added = true;
        } else {
          result.keyframe_quarantined = true;
          result.keyframe_quarantine_reason = result.shock_quarantine_reason;
        }
      }
      last_registration_ = result;
      publishState(stamp, result);
      publishDiagnostics(stamp, result);
      return;
    }

    if (use_motion_prior_fallback_) {
      RCLCPP_WARN(
        get_logger(),
        "Scan-to-map rejected (%s); accepting motion-prior fallback pose",
        result.rejection_reason.c_str());
      map_T_base_ = constrain_candidate_pose(
        prediction, map_T_base_, &result);
      commitWheelPrediction();
      result.accepted = true;
      result.converged = true;
      result.rejection_reason = "motion_prior_fallback";
      result.map_T_base = map_T_base_;
      result.shock_quarantine_reason = shockQuarantineReason(result);
      result.shock_quarantined = !result.shock_quarantine_reason.empty();
      if (result.shock_quarantined) {
        constant_velocity_rebase_pending_ = true;
      }
      if (result.shock_quarantined && quarantine_pose_on_shock_) {
        result.pose_quarantined = true;
        result.rejection_reason = "shock_pose_quarantine_motion_prior_fallback";
        ++consecutive_pose_quarantine_count_;
        result.consecutive_pose_quarantine_count = consecutive_pose_quarantine_count_;
      } else if (!result.shock_quarantined) {
        consecutive_pose_quarantine_count_ = 0;
        recordAcceptedPoseOrRebaseAfterShock();
        recordTrackingScan(scan_base, map_T_base_, stamp);
      } else if (record_tracking_scan_during_shock_) {
        recordTrackingScan(scan_base, map_T_base_, stamp);
        result.tracking_scan_updated_during_shock = true;
      }
      last_registration_ = result;
      latest_local_map_ = local_map;
      if (add_motion_prior_fallback_keyframes_ && shouldAddKeyframe(map_T_base_)) {
        if (!result.shock_quarantined) {
          addKeyframe(
            scan_base,
            mapping_scan_base,
            registration_only_stats.points_after_balance,
            terrain_sensor_scans,
            stamp,
            gravity_sample,
            gravity_sample_stale);
          result.keyframe_added = true;
        } else {
          result.keyframe_quarantined = true;
          result.keyframe_quarantine_reason = result.shock_quarantine_reason;
        }
      }
      last_registration_ = result;
      publishState(stamp, result);
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
    result.shock_quarantine_reason = shockQuarantineReason(result);
    result.shock_quarantined = !result.shock_quarantine_reason.empty();
    if (result.shock_quarantined) {
      constant_velocity_rebase_pending_ = true;
    }
    const bool preserve_legacy_tracking_behavior = !quarantine_keyframes_on_shock_;
    const bool have_trusted_ephemeral_prediction = used_scan_to_scan_prediction;
    const bool allow_ephemeral_tracking_update =
      preserve_legacy_tracking_behavior ||
      (have_trusted_ephemeral_prediction &&
      (!result.shock_quarantined || record_tracking_scan_during_shock_));
    if (allow_ephemeral_tracking_update) {
      recordTrackingScan(
        scan_base,
        constrain_candidate_pose(prediction, map_T_base_, &result),
        stamp);
      result.tracking_scan_updated_after_rejection = true;
      result.tracking_scan_updated_during_shock = result.shock_quarantined;
    }
    last_registration_ = result;
    publishDiagnostics(stamp, result);
    return;
  }

  result.shock_quarantine_reason = shockQuarantineReason(result);
  result.shock_quarantined = !result.shock_quarantine_reason.empty();
  if (result.shock_quarantined) {
    constant_velocity_rebase_pending_ = true;
  }
  if (result.shock_quarantined && quarantine_pose_on_shock_) {
    map_T_base_ = constrain_candidate_pose(
      prediction, map_T_base_, &result);
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
    if (record_tracking_scan_during_shock_) {
      recordTrackingScan(scan_base, map_T_base_, stamp);
      result.tracking_scan_updated_during_shock = true;
    } else if (record_tracking_scan_on_pose_quarantine_ || periodic_tracking_reset) {
      recordTrackingScan(scan_base, map_T_base_, stamp);
      result.pose_quarantine_tracking_reset = true;
    }
    latest_local_map_ = local_map;
    last_registration_ = result;
    publishState(stamp, result);
    publishDiagnostics(stamp, result);
    return;
  }
  consecutive_pose_quarantine_count_ = 0;
  if (!pose_graph_correction_already_applied) {
    map_T_base_ = result.ground_motion_post_constraint_quality_evaluated ?
      result.map_T_base :
      constrain_candidate_pose(result.map_T_base, prediction, &result);
    result.map_T_base = map_T_base_;
  }
  if (result.place_recognition_accepted && !pose_graph_correction_already_applied) {
    applyPoseGraphCorrection(
      stamp,
      prediction,
      gravity_sample,
      current_imu_world_R_base,
      imu_attitude_sample_status,
      result);
    local_map = buildLocalMap(map_T_base_);
  }
  commitWheelPrediction();
  if (!result.shock_quarantined) {
    consecutive_pose_quarantine_count_ = 0;
    recordAcceptedPoseOrRebaseAfterShock();
    recordTrackingScan(scan_base, map_T_base_, stamp);
  } else if (record_tracking_scan_during_shock_) {
    recordTrackingScan(scan_base, map_T_base_, stamp);
    result.tracking_scan_updated_during_shock = true;
  }
  latest_local_map_ = local_map;

  if (shouldAddKeyframe(map_T_base_)) {
    if (!result.shock_quarantined) {
      attachRegistrationInformation(scan_base, local_map, result);
      addKeyframe(
        scan_base,
        mapping_scan_base,
        registration_only_stats.points_after_balance,
        terrain_sensor_scans,
        stamp,
        gravity_sample,
        gravity_sample_stale,
        &result);
      result.keyframe_added = true;
    } else {
      result.keyframe_quarantined = true;
      result.keyframe_quarantine_reason = result.shock_quarantine_reason;
    }
  }

  last_registration_ = result;
  publishState(stamp, result);
  publishDiagnostics(stamp, result);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildBaseFrameScan(
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
  const bool build_uncompensated_hypothesis)
{
  merged_cloud_count = 0;
  mapping_scan_base.reset();
  uncompensated_scan_base.reset();
  uncompensated_mapping_scan_base.reset();
  registration_only_stats = RegistrationOnlyCloudStats{};
  terrain_sensor_scans.clear();
  uncompensated_terrain_sensor_scans.clear();
  const bool preserve_sensor_views =
    terrain_config_.enabled || !keyframe_archive_directory_.empty();
  const double sensor_view_voxel_leaf_m = terrain_config_.enabled ?
    terrain_config_.classification_voxel_size_m :
    keyframe_archive_voxel_leaf_m_;
  Eigen::Vector3d primary_origin_base = Eigen::Vector3d::Zero();
  auto primary = convertAndTransformCloud(
    *cloud_set.primary,
    cloud_set.imu_samples,
    deskew_stats,
    &primary_origin_base);
  if (!primary) {
    return nullptr;
  }
  ++merged_cloud_count;
  auto mapping_merged = std::make_shared<CloudT>(*primary);
  auto registration_only_merged = std::make_shared<CloudT>();
  if (preserve_sensor_views) {
    terrain_sensor_scans.push_back(
      TerrainSensorScan{
          downsampleCloud(
            filterCloud(primary),
            sensor_view_voxel_leaf_m),
          primary_origin_base,
          {},
          {}});
  }
  CloudPtr uncompensated_mapping_merged;
  CloudPtr uncompensated_registration_only_merged;
  if (build_uncompensated_hypothesis) {
    uncompensated_mapping_merged = std::make_shared<CloudT>(*primary);
    uncompensated_registration_only_merged = std::make_shared<CloudT>();
    if (preserve_sensor_views) {
      uncompensated_terrain_sensor_scans.push_back(
        TerrainSensorScan{
            downsampleCloud(
              filterCloud(primary),
              sensor_view_voxel_leaf_m),
            primary_origin_base,
            {},
            {}});
    }
  }

  for (std::size_t index = 0; index < cloud_set.auxiliary.size(); ++index) {
    const auto & message = cloud_set.auxiliary[index];
    if (!message) {
      continue;
    }

    DeskewStats auxiliary_deskew_stats;
    Eigen::Vector3d auxiliary_origin_base = Eigen::Vector3d::Zero();
    auto auxiliary = convertAndTransformCloud(
      *message,
      cloud_set.imu_samples,
      auxiliary_deskew_stats,
      &auxiliary_origin_base);
    if (!auxiliary) {
      continue;
    }
    CloudPtr uncompensated_auxiliary;
    Eigen::Vector3d uncompensated_auxiliary_origin_base = auxiliary_origin_base;
    if (uncompensated_mapping_merged) {
      uncompensated_auxiliary = std::make_shared<CloudT>(*auxiliary);
    }
    const bool motion_compensation_ready = compensateAuxiliaryCloudToPrimaryStamp(
      auxiliary,
      rclcpp::Time(message->header.stamp),
      rclcpp::Time(cloud_set.primary->header.stamp),
      cloud_set.imu_samples,
      auxiliary_origin_base,
      inter_lidar_motion_stats);
    if (
      !motion_compensation_ready &&
      inter_lidar_motion_failure_policy_ == "drop_auxiliary")
    {
      ++inter_lidar_motion_stats.dropped_count;
      continue;
    }
    const bool registration_only =
      auxiliary_cloud_inputs_[index].registration_only;
    if (registration_only) {
      *registration_only_merged += *auxiliary;
      ++registration_only_stats.clouds_merged;
    } else {
      *mapping_merged += *auxiliary;
    }
    if (!registration_only && preserve_sensor_views) {
      terrain_sensor_scans.push_back(
        TerrainSensorScan{
            downsampleCloud(
              filterCloud(auxiliary),
              sensor_view_voxel_leaf_m),
            auxiliary_origin_base,
            {},
            {}});
    }
    if (uncompensated_mapping_merged && uncompensated_auxiliary) {
      if (registration_only) {
        *uncompensated_registration_only_merged += *uncompensated_auxiliary;
      } else {
        *uncompensated_mapping_merged += *uncompensated_auxiliary;
      }
      if (!registration_only && preserve_sensor_views) {
        uncompensated_terrain_sensor_scans.push_back(
          TerrainSensorScan{
              downsampleCloud(
                filterCloud(uncompensated_auxiliary),
                sensor_view_voxel_leaf_m),
              uncompensated_auxiliary_origin_base,
              {},
              {}});
      }
    }
    ++merged_cloud_count;

    const bool primary_or_previous_applied = deskew_stats.applied;
    deskew_stats.attempted =
      deskew_stats.attempted || auxiliary_deskew_stats.attempted;
    deskew_stats.applied =
      deskew_stats.applied || auxiliary_deskew_stats.applied;
    deskew_stats.point_count += auxiliary_deskew_stats.point_count;
    deskew_stats.points_with_time += auxiliary_deskew_stats.points_with_time;
    if (deskew_stats.point_count > 0) {
      deskew_stats.time_ratio =
        static_cast<double>(deskew_stats.points_with_time) /
        static_cast<double>(deskew_stats.point_count);
    }
    deskew_stats.scan_duration_s = std::max(
      deskew_stats.scan_duration_s,
      auxiliary_deskew_stats.scan_duration_s);
    if (auxiliary_deskew_stats.have_scan_time_bounds) {
      if (!deskew_stats.have_scan_time_bounds) {
        deskew_stats.scan_start_time_s = auxiliary_deskew_stats.scan_start_time_s;
        deskew_stats.scan_end_time_s = auxiliary_deskew_stats.scan_end_time_s;
        deskew_stats.have_scan_time_bounds = true;
      } else {
        deskew_stats.scan_start_time_s = std::min(
          deskew_stats.scan_start_time_s,
          auxiliary_deskew_stats.scan_start_time_s);
        deskew_stats.scan_end_time_s = std::max(
          deskew_stats.scan_end_time_s,
          auxiliary_deskew_stats.scan_end_time_s);
      }
    }
    if (deskew_stats.point_time_field.empty()) {
      deskew_stats.point_time_field = auxiliary_deskew_stats.point_time_field;
    }
    if (deskew_enabled_) {
      if (primary_or_previous_applied != auxiliary_deskew_stats.applied) {
        deskew_stats.status = "partially_applied";
      } else if (deskew_stats.applied) {
        deskew_stats.status = "applied";
      } else if (deskew_stats.status != auxiliary_deskew_stats.status) {
        deskew_stats.status = "multiple_failures";
      }
    }
  }

  mapping_scan_base = filterCloud(mapping_merged);
  auto registration_scan_base = std::make_shared<CloudT>(*mapping_scan_base);
  const auto filtered_registration_only = filterCloud(registration_only_merged);
  registration_only_stats.points_before_balance =
    static_cast<int>(filtered_registration_only->size());
  const auto balanced_registration_only = balanceRegistrationOnlyCloud(
    filtered_registration_only,
    mapping_scan_base->size());
  registration_only_stats.points_after_balance =
    static_cast<int>(balanced_registration_only->size());
  *registration_scan_base += *balanced_registration_only;

  if (uncompensated_mapping_merged) {
    uncompensated_mapping_scan_base = filterCloud(uncompensated_mapping_merged);
    uncompensated_scan_base =
      std::make_shared<CloudT>(*uncompensated_mapping_scan_base);
    const auto uncompensated_registration_only = balanceRegistrationOnlyCloud(
      filterCloud(uncompensated_registration_only_merged),
      uncompensated_mapping_scan_base->size());
    *uncompensated_scan_base += *uncompensated_registration_only;
  }
  return registration_scan_base;
}

ScanToMapIcp::CloudPtr ScanToMapIcp::convertAndTransformCloud(
  const sensor_msgs::msg::PointCloud2 & msg,
  const std::vector<ImuSample> & imu_samples,
  DeskewStats & deskew_stats,
  Eigen::Vector3d * sensor_origin_base)
{
  auto cloud = std::make_shared<CloudT>();
  pcl::fromROSMsg(msg, *cloud);
  if (cloud->empty()) {
    return cloud;
  }

  const auto base_T_sensor = lookupTransformToBase(msg);
  if (!base_T_sensor.has_value()) {
    return nullptr;
  }
  if (sensor_origin_base != nullptr) {
    *sensor_origin_base = base_T_sensor->translation();
  }
  auto cloud_base = transformCloud(cloud, *base_T_sensor);
  const bool deskew_applied =
    deskewCloudInBaseFrame(msg, cloud_base, imu_samples, deskew_stats);
  if (deskew_enabled_ && reject_cloud_on_deskew_failure_ && !deskew_applied) {
    return nullptr;
  }
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

ScanToMapIcp::CloudPtr ScanToMapIcp::balanceRegistrationOnlyCloud(
  const CloudConstPtr & cloud,
  const std::size_t mapping_point_count) const
{
  auto balanced = downsampleCloud(cloud, registration_only_cloud_voxel_leaf_m_);
  if (!balanced || balanced->empty()) {
    return balanced;
  }
  if (mapping_point_count == 0U) {
    return std::make_shared<CloudT>();
  }

  const auto ratio_limit = static_cast<std::size_t>(std::floor(
      static_cast<double>(mapping_point_count) *
      registration_only_cloud_max_point_ratio_));
  std::size_t point_limit = std::max<std::size_t>(1U, ratio_limit);
  if (registration_only_cloud_max_points_ > 0) {
    point_limit = std::min(
      point_limit,
      static_cast<std::size_t>(registration_only_cloud_max_points_));
  }
  if (balanced->size() <= point_limit) {
    return balanced;
  }

  auto limited = std::make_shared<CloudT>();
  limited->reserve(point_limit);
  for (std::size_t index = 0U; index < point_limit; ++index) {
    const std::size_t source_index = std::min(
      balanced->size() - 1U,
      index * balanced->size() / point_limit);
    limited->push_back(balanced->points[source_index]);
  }
  limited->width = static_cast<std::uint32_t>(limited->size());
  limited->height = 1U;
  limited->is_dense = balanced->is_dense;
  return limited;
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
  const std::vector<ImuSample> & imu_samples,
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

  const sensor_msgs::msg::PointField * time_field =
    findPointTimeField(msg, deskew_point_time_field_);
  if (!time_field) {
    stats.status = "missing_point_time";
    return false;
  }
  stats.point_time_field = time_field->name;

  const double header_time_s = rclcpp::Time(msg.header.stamp).seconds();
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
    const double point_time_s = decodePointTimeSeconds(
      *raw_value,
      time_field->name,
      header_time_s,
      deskew_point_time_scale_,
      deskew_point_time_offset_s_);
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
  stats.scan_start_time_s = min_time_s;
  stats.scan_end_time_s = max_time_s;
  stats.have_scan_time_bounds = true;
  const double max_scan_duration = std::max(0.0, deskew_max_scan_duration_s_);
  if (stats.scan_duration_s <= 1.0e-6) {
    stats.status = "zero_scan_duration";
    return false;
  }
  const double timestamp_tolerance_s = pointTimeComparisonToleranceS(
    header_time_s, min_time_s, max_time_s);
  if (max_scan_duration > 0.0 &&
    stats.scan_duration_s > max_scan_duration + timestamp_tolerance_s)
  {
    stats.status = "scan_duration_out_of_range";
    return false;
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
  const auto angular_velocity_base =
    [&imu_to_base, angular_velocity_scale](const ImuSample & sample) {
      return imu_to_base * sample.angular_velocity * angular_velocity_scale;
    };

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
  Eigen::Vector3d current_angular_velocity =
    angular_velocity_base(imu_samples[start_index]);
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
    current_angular_velocity = angular_velocity_base(sample);
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

bool ScanToMapIcp::compensateAuxiliaryCloudToPrimaryStamp(
  CloudPtr & cloud_base,
  const rclcpp::Time & cloud_stamp,
  const rclcpp::Time & primary_stamp,
  const std::vector<ImuSample> & imu_samples,
  Eigen::Vector3d & sensor_origin_base,
  InterLidarMotionStats & stats)
{
  if (!inter_lidar_motion_compensation_enabled_) {
    return true;
  }
  if (!cloud_base) {
    stats.status = "cloud_unavailable";
    return false;
  }

  ++stats.attempted_count;
  const double stamp_delta_s = std::abs((primary_stamp - cloud_stamp).seconds());
  stats.max_stamp_delta_s = std::max(stats.max_stamp_delta_s, stamp_delta_s);
  const double max_dt_s = std::max(0.0, inter_lidar_motion_compensation_max_dt_s_);
  if (max_dt_s > 0.0 && stamp_delta_s > max_dt_s) {
    stats.status = stats.applied_count > 0 ? "partially_applied" : "stamp_delta_out_of_range";
    return false;
  }

  Eigen::Isometry3d primary_T_cloud = Eigen::Isometry3d::Identity();
  if (inter_lidar_motion_compensation_source_ == "imu_rotation") {
    if (stamp_delta_s > 1.0e-9) {
      const bool cloud_is_earlier = cloud_stamp < primary_stamp;
      const rclcpp::Time start_stamp = cloud_is_earlier ? cloud_stamp : primary_stamp;
      const rclcpp::Time end_stamp = cloud_is_earlier ? primary_stamp : cloud_stamp;
      std::string imu_status;
      const auto earlier_R_later = integrateAngularVelocityRotationDelta(
        start_stamp,
        end_stamp,
        imu_samples,
        inter_lidar_motion_compensation_max_dt_s_,
        deskew_angular_velocity_scale_ * inter_lidar_imu_rotation_scale_,
        imu_status);
      if (!earlier_R_later.has_value()) {
        stats.status = stats.applied_count > 0 ?
          "partially_applied" : "imu_rotation_" + imu_status;
        return false;
      }
      const Eigen::Quaterniond primary_R_cloud = cloud_is_earlier ?
        earlier_R_later->conjugate() : *earlier_R_later;
      primary_T_cloud.linear() = primary_R_cloud.normalized().toRotationMatrix();
    }
  } else {
    std::optional<WheelOdomSample> cloud_sample;
    std::optional<WheelOdomSample> primary_sample;
    {
      std::lock_guard<std::mutex> lock(wheel_mutex_);
      cloud_sample = wheelOdomAtLocked(cloud_stamp, false, true);
      primary_sample = wheelOdomAtLocked(primary_stamp, false, true);
    }
    if (!cloud_sample.has_value() || !primary_sample.has_value()) {
      stats.status = stats.applied_count > 0 ?
        "partially_applied" : "wheel_odom_unbracketed_or_unavailable";
      return false;
    }
    primary_T_cloud =
      primary_sample->odom_T_base.inverse() * cloud_sample->odom_T_base;
  }

  cloud_base = transformCloud(cloud_base, primary_T_cloud);
  sensor_origin_base = primary_T_cloud * sensor_origin_base;
  ++stats.applied_count;
  stats.max_translation_m = std::max(
    stats.max_translation_m,
    primary_T_cloud.translation().norm());
  stats.max_rotation_deg = std::max(
    stats.max_rotation_deg,
    rotationAngleDeg(primary_T_cloud.linear()));
  stats.status =
    stats.applied_count == stats.attempted_count ? "applied" : "partially_applied";
  return true;
}

void ScanToMapIcp::applyInterLidarMotionStatsToResult(
  const InterLidarMotionStats & stats,
  RegistrationResult & result) const
{
  result.inter_lidar_motion_attempted_count = stats.attempted_count;
  result.inter_lidar_motion_applied_count = stats.applied_count;
  result.inter_lidar_motion_dropped_count = stats.dropped_count;
  result.inter_lidar_motion_max_stamp_delta_s = stats.max_stamp_delta_s;
  result.inter_lidar_motion_max_translation_m = stats.max_translation_m;
  result.inter_lidar_motion_max_rotation_deg = stats.max_rotation_deg;
  result.inter_lidar_gap_recovery_hypothesis_attempted =
    stats.gap_recovery_hypothesis_attempted;
  result.inter_lidar_gap_recovery_selected_uncompensated =
    stats.gap_recovery_selected_uncompensated;
  result.inter_lidar_gap_recovery_compensated_score =
    stats.gap_recovery_compensated_score;
  result.inter_lidar_gap_recovery_uncompensated_score =
    stats.gap_recovery_uncompensated_score;
  result.inter_lidar_gap_recovery_uncompensated_fitness =
    stats.gap_recovery_uncompensated_fitness;
  result.inter_lidar_gap_recovery_uncompensated_inlier_ratio =
    stats.gap_recovery_uncompensated_inlier_ratio;
  result.inter_lidar_gap_recovery_uncompensated_inlier_rmse_m =
    stats.gap_recovery_uncompensated_inlier_rmse_m;
  result.inter_lidar_gap_recovery_runtime_ms =
    stats.gap_recovery_runtime_ms;
  result.inter_lidar_gap_recovery_selected_input =
    stats.gap_recovery_selected_input;
  result.inter_lidar_motion_status = stats.status;
}

ScanToMapIcp::ImuRoughnessStats ScanToMapIcp::evaluateImuRoughness(
  const rclcpp::Time & start_stamp,
  const rclcpp::Time & end_stamp,
  const bool have_start_stamp,
  const std::vector<ImuSample> & imu_samples) const
{
  ImuRoughnessStats stats;
  if (!imu_roughness_detection_enabled_) {
    return stats;
  }
  stats.evaluated = true;
  stats.reason = "nominal";

  const double window_s = std::max(0.01, imu_roughness_window_s_);
  const double start_s = have_start_stamp ?
    start_stamp.seconds() : end_stamp.seconds() - window_s;
  const double end_s = end_stamp.seconds();
  if (!std::isfinite(start_s) || !std::isfinite(end_s) || end_s < start_s) {
    stats.reason = "invalid_window";
    return stats;
  }

  std::vector<ImuSample> samples;
  for (const auto & sample : imu_samples) {
    const double sample_s = sample.stamp.seconds();
    if (sample_s >= start_s && sample_s <= end_s) {
      samples.push_back(sample);
    }
  }

  stats.sample_count = static_cast<int>(samples.size());
  if (samples.empty()) {
    stats.reason = "no_imu_samples";
    return stats;
  }

  bool have_positive_time_delta = false;
  double previous_accel_norm = samples.front().linear_acceleration.norm();
  const auto consider_translational_response = [&stats](
    const double value,
    const double threshold,
    const rclcpp::Time & stamp) {
      if (!std::isfinite(value) || !std::isfinite(threshold) || threshold <= 0.0) {
        return;
      }
      const double score = std::max(0.0, value / threshold);
      if (score > stats.max_translational_response_score) {
        stats.max_translational_response_score = score;
        stats.max_translational_response_stamp_ns = stamp.nanoseconds();
      }
    };
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
      consider_translational_response(
        accel_norm_delta,
        imu_roughness_accel_norm_delta_threshold_mps2_,
        sample.stamp);
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
    have_positive_time_delta = true;
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
      consider_translational_response(
        linear_jerk,
        imu_roughness_linear_jerk_threshold_mps3_,
        sample.stamp);
    }
  }

  bool continuous_imu_coverage = samples.size() >= 2U && have_positive_time_delta;
  if (continuous_imu_coverage) {
    const double max_imu_gap_s = std::max(1.0e-3, deskew_max_imu_gap_s_);
    if (samples.front().stamp.seconds() > start_s + max_imu_gap_s ||
      samples.back().stamp.seconds() < end_s - max_imu_gap_s)
    {
      continuous_imu_coverage = false;
      stats.reason = "insufficient_imu_coverage";
    }
    for (std::size_t index = 1U; continuous_imu_coverage && index < samples.size(); ++index) {
      const double dt_s = (samples[index].stamp - samples[index - 1U].stamp).seconds();
      if (!std::isfinite(dt_s) || dt_s <= 1.0e-6 || dt_s > max_imu_gap_s) {
        continuous_imu_coverage = false;
        stats.reason = "imu_gap";
      }
    }
  }
  stats.stable_evidence = continuous_imu_coverage;
  if (!stats.stable_evidence && stats.reason == "nominal") {
    stats.reason = samples.size() < 2U ?
      "insufficient_imu_samples" : "nonincreasing_imu_samples";
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

ScanToMapIcp::ImuRoughnessStats ScanToMapIcp::evaluateAndUpdateImuShock(
  const rclcpp::Time & cloud_stamp,
  const DeskewStats & deskew_stats,
  const std::vector<ImuSample> & imu_samples)
{
  const bool have_scan_bounds = deskew_stats.have_scan_time_bounds &&
    pointTimeBoundsPlausible(
    cloud_stamp.seconds(),
    deskew_stats.scan_start_time_s,
    deskew_stats.scan_end_time_s,
    deskew_max_scan_duration_s_);
  const rclcpp::Time scan_start_stamp = have_scan_bounds ?
    rclcpp::Time(
    static_cast<std::int64_t>(std::llround(deskew_stats.scan_start_time_s * 1.0e9)),
    cloud_stamp.get_clock_type()) : cloud_stamp;
  const rclcpp::Time point_time_end_stamp = have_scan_bounds ?
    rclcpp::Time(
    static_cast<std::int64_t>(std::llround(deskew_stats.scan_end_time_s * 1.0e9)),
    cloud_stamp.get_clock_type()) : cloud_stamp;
  const rclcpp::Time raw_evaluation_end_stamp =
    point_time_end_stamp < cloud_stamp ? cloud_stamp : point_time_end_stamp;
  if (have_last_imu_roughness_evaluation_stamp_ &&
    raw_evaluation_end_stamp <= last_imu_roughness_evaluation_stamp_)
  {
    ImuRoughnessStats stats;
    stats.reason = "nonadvancing_scan_time_bounds";
    stats.shock_phase = toString(imu_shock_policy_state_.phase);
    stats.shock_observation = toString(ImuShockObservation::kUnknown);
    stats.shock_active =
      imu_shock_policy_state_.phase == ImuShockPhase::kShock ||
      imu_shock_policy_state_.phase == ImuShockPhase::kRecovering;
    stats.recent_event = stats.shock_active;
    stats.shock_recovery_stable_scan_count =
      imu_shock_policy_state_.recovery_stable_scan_count;
    if (imu_shock_policy_state_.have_last_event_stamp) {
      stats.shock_time_since_event_s = std::max(
        0.0,
        last_imu_roughness_evaluation_stamp_.seconds() -
        imu_shock_policy_state_.last_event_stamp_s);
    }
    return stats;
  }
  const rclcpp::Time evaluation_end_stamp = raw_evaluation_end_stamp;

  rclcpp::Time evaluation_start_stamp = scan_start_stamp;
  bool have_evaluation_start = have_scan_bounds;
  if (have_last_imu_roughness_evaluation_stamp_ &&
    last_imu_roughness_evaluation_stamp_ <= evaluation_end_stamp)
  {
    evaluation_start_stamp = last_imu_roughness_evaluation_stamp_;
    have_evaluation_start = true;
  }

  auto stats = evaluateImuRoughness(
    evaluation_start_stamp,
    evaluation_end_stamp,
    have_evaluation_start,
    imu_samples);
  if (have_scan_bounds) {
    const auto scan_coverage = evaluateImuRoughness(
      scan_start_stamp,
      evaluation_end_stamp,
      true,
      imu_samples);
    stats.stable_evidence = scan_coverage.stable_evidence;
    if (!stats.event) {
      stats.reason = scan_coverage.reason;
    }
  }
  if (stats.evaluated) {
    if (stats.event) {
      last_imu_roughness_event_stamp_ = evaluation_end_stamp;
      have_last_imu_roughness_event_stamp_ = true;
    }
    if (have_last_imu_roughness_event_stamp_ &&
      last_imu_roughness_event_stamp_ <= evaluation_end_stamp)
    {
      stats.time_since_event_s = std::max(
        0.0,
        (evaluation_end_stamp - last_imu_roughness_event_stamp_).seconds());
      const double recent_window_s = std::max(0.0, imu_roughness_recent_event_window_s_);
      stats.recent_event =
        recent_window_s > 0.0 && stats.time_since_event_s <= recent_window_s;
    }
  }

  const ImuShockObservation observation = stats.event ?
    ImuShockObservation::kShock :
    (stats.stable_evidence ? ImuShockObservation::kStable : ImuShockObservation::kUnknown);
  const ImuShockPolicyDecision decision = updateImuShockPolicy(
    imu_shock_policy_config_,
    observation,
    evaluation_end_stamp.seconds(),
    imu_shock_policy_state_);
  if (decision.active) {
    constant_velocity_rebase_pending_ = true;
  }
  stats.shock_active = decision.active;
  stats.shock_time_reset = decision.time_reset;
  stats.shock_just_recovered = decision.just_recovered;
  stats.shock_recovery_stable_scan_count = decision.recovery_stable_scan_count;
  stats.shock_time_since_event_s = decision.time_since_event_s;
  stats.shock_phase = toString(decision.phase);
  stats.shock_observation = toString(decision.observation);
  stats.recent_event = stats.recent_event || decision.active;

  if (decision.time_reset) {
    clearPendingTerrainShock(false);
    terrain_response_episode_active_ = false;
    terrain_traversal_continuity_break_pending_ = true;
  }
  if (terrain_config_.enabled && terrain_preview_config_.enabled) {
    // The common shock policy remains conservative for frontend quarantine.
    // Terrain attribution is stricter: normal yaw and yaw acceleration are not
    // evidence that the wheels contacted rough ground, so require a measured
    // translational impulse before creating a spatial response event.
    const double response_score = stats.max_translational_response_score;
    if (response_score >= 1.0 && terrain_pending_shock_event_) {
      // A second pulse may refine the pending event, but it must never create
      // a fresh bridge or clear an invalidation caused by an omitted pose.
      if (response_score > terrain_pending_shock_score_ &&
        stats.max_translational_response_stamp_ns > 0)
      {
        const rclcpp::Time stronger_response_stamp(
          stats.max_translational_response_stamp_ns,
          evaluation_end_stamp.get_clock_type());
        if (terrain_pending_shock_pre_sample_stamp_ < stronger_response_stamp &&
          stronger_response_stamp <= evaluation_end_stamp)
        {
          terrain_pending_shock_score_ = response_score;
          terrain_pending_shock_stamp_ = stronger_response_stamp;
        }
      }
    } else if (response_score >= 1.0 && !terrain_response_episode_active_) {
      terrain_response_episode_active_ = true;
      const bool have_stable_pre_sample = !keyframes_.empty() &&
        !terrain_traversal_samples_.empty() &&
        !terrain_traversal_continuity_break_pending_ &&
        terrain_traversal_samples_.back().response == TerrainTraversalResponse::kStable;
      const rclcpp::Time response_stamp(
        stats.max_translational_response_stamp_ns,
        evaluation_end_stamp.get_clock_type());
      if (have_stable_pre_sample && stats.max_translational_response_stamp_ns > 0 &&
        terrain_traversal_samples_.back().stamp < response_stamp &&
        response_stamp <= evaluation_end_stamp)
      {
        terrain_pending_shock_stamp_ = response_stamp;
        terrain_pending_shock_pre_sample_stamp_ =
          terrain_traversal_samples_.back().stamp;
        terrain_pending_shock_event_ = true;
        terrain_pending_shock_score_ = response_score;
        terrain_pending_shock_bridge_invalidated_ = false;
      } else {
        ++terrain_unlocalized_shock_event_count_;
      }
    } else if (response_score < 1.0) {
      // A yaw-only event must not hold the terrain-specific episode latch.
      // This allows a later, distinct translational impact to be localized.
      terrain_response_episode_active_ = false;
    }
  }

  last_imu_roughness_evaluation_stamp_ = evaluation_end_stamp;
  have_last_imu_roughness_evaluation_stamp_ = true;
  return stats;
}

void ScanToMapIcp::applyImuRoughnessStatsToResult(
  const ImuRoughnessStats & stats,
  RegistrationResult & result) const
{
  result.imu_roughness_evaluated = stats.evaluated;
  result.imu_roughness_event = stats.event;
  result.imu_roughness_stable_evidence = stats.stable_evidence;
  result.imu_roughness_recent_event = stats.recent_event;
  result.imu_shock_active = stats.shock_active;
  result.imu_shock_time_reset = stats.shock_time_reset;
  result.imu_shock_just_recovered = stats.shock_just_recovered;
  result.imu_roughness_sample_count = stats.sample_count;
  result.imu_shock_recovery_stable_scan_count = stats.shock_recovery_stable_scan_count;
  result.imu_roughness_max_angular_velocity_radps = stats.max_angular_velocity_radps;
  result.imu_roughness_max_angular_accel_radps2 = stats.max_angular_accel_radps2;
  result.imu_roughness_max_linear_accel_mps2 = stats.max_linear_accel_mps2;
  result.imu_roughness_max_accel_norm_delta_mps2 = stats.max_accel_norm_delta_mps2;
  result.imu_roughness_max_linear_jerk_mps3 = stats.max_linear_jerk_mps3;
  result.imu_roughness_time_since_event_s = stats.time_since_event_s;
  result.imu_shock_time_since_event_s = stats.shock_time_since_event_s;
  result.imu_roughness_reason = stats.reason;
  result.imu_shock_phase = stats.shock_phase;
  result.imu_shock_observation = stats.shock_observation;
}

std::optional<Eigen::Isometry3d> ScanToMapIcp::lookupTransformToBase(
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
    if (reject_cloud_on_tf_failure_) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "TF lookup failed %s -> %s: %s. Rejecting this sensor cloud.",
        msg.header.frame_id.c_str(),
        base_frame_.c_str(),
        ex.what());
      return std::nullopt;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(),
      *get_clock(),
      2000,
      "TF lookup failed %s -> %s: %s. Identity fallback is enabled.",
      msg.header.frame_id.c_str(),
      base_frame_.c_str(),
      ex.what());
    return Eigen::Isometry3d::Identity();
  }
}

std::optional<ScanToMapIcp::WheelOdomSample> ScanToMapIcp::wheelOdomAtLocked(
  const rclcpp::Time & stamp,
  const bool update_prediction_diagnostics,
  const bool require_bracket)
{
  auto set_status = [this, update_prediction_diagnostics](const char * status) {
      if (update_prediction_diagnostics) {
        wheel_prediction_status_ = status;
      }
    };
  if (update_prediction_diagnostics) {
    wheel_prediction_query_stamp_ = stamp;
    wheel_prediction_lower_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    wheel_prediction_upper_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    wheel_prediction_max_bracket_delta_s_ = 0.0;
  }
  if (wheel_odom_buffer_.empty()) {
    set_status("no_samples");
    return std::nullopt;
  }

  const auto upper = std::lower_bound(
    wheel_odom_buffer_.begin(),
    wheel_odom_buffer_.end(),
    stamp,
    [](const WheelOdomSample & sample, const rclcpp::Time & target) {
      return sample.stamp < target;
    });
  const double max_time_delta_s = std::max(0.0, wheel_odom_max_time_delta_s_);

  if (upper != wheel_odom_buffer_.end() && upper->stamp == stamp) {
    if (update_prediction_diagnostics) {
      wheel_prediction_lower_stamp_ = upper->stamp;
      wheel_prediction_upper_stamp_ = upper->stamp;
    }
    set_status("exact");
    return *upper;
  }
  if (upper == wheel_odom_buffer_.begin()) {
    if (require_bracket) {
      set_status("future_sample_without_lower_bracket");
      return std::nullopt;
    }
    const double delta_s = (upper->stamp - stamp).seconds();
    if (update_prediction_diagnostics) {
      wheel_prediction_upper_stamp_ = upper->stamp;
      wheel_prediction_max_bracket_delta_s_ = std::abs(delta_s);
    }
    if (max_time_delta_s > 0.0 && std::abs(delta_s) > max_time_delta_s) {
      set_status("future_sample_too_far");
      return std::nullopt;
    }
    set_status("nearest_future");
    return *upper;
  }
  if (upper == wheel_odom_buffer_.end()) {
    if (require_bracket) {
      set_status("past_sample_without_upper_bracket");
      return std::nullopt;
    }
    const WheelOdomSample & latest = wheel_odom_buffer_.back();
    const double delta_s = (stamp - latest.stamp).seconds();
    if (update_prediction_diagnostics) {
      wheel_prediction_lower_stamp_ = latest.stamp;
      wheel_prediction_max_bracket_delta_s_ = std::abs(delta_s);
    }
    if (max_time_delta_s > 0.0 && std::abs(delta_s) > max_time_delta_s) {
      set_status("past_sample_too_far");
      return std::nullopt;
    }
    set_status("nearest_past");
    return latest;
  }

  const WheelOdomSample & before = *(upper - 1);
  const WheelOdomSample & after = *upper;
  const double interval_s = (after.stamp - before.stamp).seconds();
  const double before_delta_s = (stamp - before.stamp).seconds();
  const double after_delta_s = (after.stamp - stamp).seconds();
  const double max_bracket_delta_s =
    std::max(std::abs(before_delta_s), std::abs(after_delta_s));
  if (update_prediction_diagnostics) {
    wheel_prediction_lower_stamp_ = before.stamp;
    wheel_prediction_upper_stamp_ = after.stamp;
    wheel_prediction_max_bracket_delta_s_ = max_bracket_delta_s;
  }
  if (!std::isfinite(interval_s) || interval_s <= 0.0) {
    set_status("invalid_bracket");
    return std::nullopt;
  }
  const double max_interpolation_gap_s =
    std::max(0.0, wheel_odom_max_interpolation_gap_s_);
  if (max_interpolation_gap_s > 0.0 && interval_s > max_interpolation_gap_s) {
    set_status("interpolation_gap_too_large");
    return std::nullopt;
  }
  if (max_time_delta_s > 0.0 &&
    max_bracket_delta_s > max_time_delta_s)
  {
    set_status("bracket_too_far");
    return std::nullopt;
  }

  const double ratio = std::clamp(before_delta_s / interval_s, 0.0, 1.0);
  Eigen::Isometry3d interpolated = Eigen::Isometry3d::Identity();
  interpolated.translation() =
    (1.0 - ratio) * before.odom_T_base.translation() +
    ratio * after.odom_T_base.translation();
  const Eigen::Quaterniond before_rotation(before.odom_T_base.linear());
  const Eigen::Quaterniond after_rotation(after.odom_T_base.linear());
  interpolated.linear() =
    before_rotation.normalized().slerp(
    ratio, after_rotation.normalized()).normalized().toRotationMatrix();
  set_status("interpolated");
  return WheelOdomSample{stamp, interpolated};
}

Eigen::Isometry3d ScanToMapIcp::predictWithWheelOdom(
  const rclcpp::Time & stamp,
  const bool allow_constant_velocity_fallback,
  const std::optional<double> & imu_pitch_rad)
{
  const Eigen::Isometry3d rigid_map_T_base = projectToRigidIsometry(map_T_base_);
  Eigen::Isometry3d prediction = rigid_map_T_base;
  if (!use_wheel_prediction_) {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    wheel_prediction_status_ = "disabled";
    have_pending_wheel_odom_ = false;
    return predictWithConstantVelocity(allow_constant_velocity_fallback);
  }

  std::lock_guard<std::mutex> lock(wheel_mutex_);
  have_pending_wheel_odom_ = false;
  const auto sample = wheelOdomAtLocked(stamp);
  if (!sample.has_value()) {
    return predictWithConstantVelocity(allow_constant_velocity_fallback);
  }

  if (!have_last_used_wheel_odom_) {
    last_used_wheel_odom_ = sample->odom_T_base;
    last_used_wheel_stamp_ = sample->stamp;
    have_last_used_wheel_odom_ = true;
    return predictWithConstantVelocity(allow_constant_velocity_fallback);
  }
  if (sample->stamp < last_used_wheel_stamp_) {
    wheel_prediction_status_ = "nonmonotonic_scan_stamp";
    return predictWithConstantVelocity(allow_constant_velocity_fallback);
  }

  pending_wheel_odom_ = sample->odom_T_base;
  pending_wheel_stamp_ = sample->stamp;
  have_pending_wheel_odom_ = true;
  const Eigen::Isometry3d rigid_last_wheel =
    projectToRigidIsometry(last_used_wheel_odom_);
  const Eigen::Isometry3d rigid_pending_wheel =
    projectToRigidIsometry(pending_wheel_odom_);
  Eigen::Isometry3d delta = projectToRigidIsometry(
    rigid_last_wheel.inverse() * rigid_pending_wheel);
  if (wheel_imu_slope_z_prediction_enabled_ && imu_pitch_rad.has_value()) {
    const double max_pitch =
      std::max(0.0, wheel_imu_slope_z_prediction_max_pitch_deg_) * M_PI / 180.0;
    const double bounded_pitch = std::clamp(*imu_pitch_rad, -max_pitch, max_pitch);
    const double predicted_z = delta.translation().x() * std::tan(bounded_pitch);
    const double weight = std::clamp(wheel_imu_slope_z_prediction_weight_, 0.0, 1.0);
    delta.translation().z() = (1.0 - weight) * delta.translation().z() + weight * predicted_z;
  }
  prediction = rigid_map_T_base * delta;
  return projectToRigidIsometry(prediction);
}

Eigen::Isometry3d ScanToMapIcp::predictWithConstantVelocity(
  const bool allow_extrapolation) const
{
  const Eigen::Isometry3d rigid_map_T_base = projectToRigidIsometry(map_T_base_);
  if (!allow_extrapolation || !use_constant_velocity_prediction_ ||
    !have_previous_accepted_pose_)
  {
    return rigid_map_T_base;
  }

  const Eigen::Isometry3d rigid_previous =
    projectToRigidIsometry(previous_accepted_pose_);
  const Eigen::Isometry3d rigid_last =
    projectToRigidIsometry(last_accepted_pose_);
  const Eigen::Isometry3d delta = projectToRigidIsometry(
    rigid_previous.inverse() * rigid_last);
  return projectToRigidIsometry(
    rigid_map_T_base * clampRelativeMotion(delta));
}

Eigen::Isometry3d ScanToMapIcp::clampRelativeMotion(const Eigen::Isometry3d & delta) const
{
  Eigen::Isometry3d clamped = projectToRigidIsometry(delta);
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

  return projectToRigidIsometry(clamped);
}

Eigen::Isometry3d ScanToMapIcp::constrainMotionPriorFallback(
  const Eigen::Isometry3d & candidate) const
{
  Eigen::Isometry3d constrained = projectToRigidIsometry(candidate);

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

  return projectToRigidIsometry(constrained);
}

Eigen::Isometry3d ScanToMapIcp::constrainImuRelativeAttitude(
  const Eigen::Isometry3d & candidate,
  const std::optional<Eigen::Matrix3d> & current_world_R_base,
  const std::string & sample_status,
  RegistrationResult * result) const
{
  const Eigen::Isometry3d rigid_candidate = projectToRigidIsometry(candidate);
  if (!imu_relative_roll_pitch_constraint_enabled_) {
    if (result != nullptr) {
      result->imu_relative_attitude_constraint_status = "disabled";
    }
    return rigid_candidate;
  }
  if (!have_initial_imu_world_R_base_) {
    if (result != nullptr) {
      result->imu_relative_attitude_constraint_status = "reference_unavailable";
    }
    return rigid_candidate;
  }
  if (!current_world_R_base) {
    if (result != nullptr) {
      result->imu_relative_attitude_constraint_status = "sample_" + sample_status;
    }
    return rigid_candidate;
  }

  const ImuRelativeAttitudeConstraintResult constrained =
    constrainImuRelativeRollPitch(
    rigid_candidate,
    initial_imu_world_R_base_,
    *current_world_R_base,
    imu_relative_attitude_constraint_config_);
  if (result != nullptr) {
    result->imu_relative_attitude_constraint_evaluated = constrained.evaluated;
    result->imu_relative_attitude_constraint_applied = constrained.applied;
    result->imu_relative_attitude_target_roll_deg = constrained.target_roll_deg;
    result->imu_relative_attitude_target_pitch_deg = constrained.target_pitch_deg;
    result->imu_relative_attitude_raw_roll_error_deg = constrained.raw_roll_error_deg;
    result->imu_relative_attitude_raw_pitch_error_deg = constrained.raw_pitch_error_deg;
    result->imu_relative_attitude_applied_roll_correction_deg =
      constrained.applied_roll_correction_deg;
    result->imu_relative_attitude_applied_pitch_correction_deg =
      constrained.applied_pitch_correction_deg;
    result->imu_relative_attitude_constraint_status = constrained.status;
  }
  return constrained.pose;
}

std::optional<ScanToMapIcp::GravitySample> ScanToMapIcp::gravitySampleAt(
  const rclcpp::Time & stamp,
  const std::vector<GravitySample> & gravity_samples,
  bool & stale) const
{
  stale = false;
  if (!gravity_roll_pitch_constraint_enabled_ && !terrain_config_.enabled) {
    return std::nullopt;
  }

  for (auto iterator = gravity_samples.rbegin(); iterator != gravity_samples.rend(); ++iterator) {
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
    return projectToRigidIsometry(candidate);
  }
  if (!planar_pose_constraint_enabled_ && !gravity_sample.has_value()) {
    return projectToRigidIsometry(candidate);
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
  return projectToRigidIsometry(constrained);
}

Eigen::Isometry3d ScanToMapIcp::predictWithMotionPriors(
  const CloudConstPtr & scan_base,
  const std::vector<ImuSample> & imu_samples,
  const rclcpp::Time & previous_scan_stamp,
  const rclcpp::Time & current_scan_stamp,
  const bool have_previous_scan_stamp,
  const bool suppress_constant_velocity_prediction,
  bool & used_scan_to_scan,
  bool & used_imu_rotation_prediction,
  bool & constant_velocity_prediction_suppressed,
  std::string & imu_rotation_prediction_status,
  double & scan_to_scan_reference_age_s,
  std::string & scan_to_scan_prediction_status)
{
  used_scan_to_scan = false;
  used_imu_rotation_prediction = false;
  constant_velocity_prediction_suppressed =
    suppress_constant_velocity_prediction && use_constant_velocity_prediction_;
  imu_rotation_prediction_status = imu_rotation_prediction_enabled_ ? "unavailable" : "disabled";
  scan_to_scan_reference_age_s = 0.0;
  scan_to_scan_prediction_status = use_scan_to_scan_prediction_ ? "unavailable" : "disabled";

  std::optional<double> wheel_slope_pitch_rad;
  if (wheel_imu_slope_z_prediction_enabled_) {
    std::string wheel_slope_imu_status;
    const auto wheel_slope_world_R_base = imuWorldBaseOrientationAt(
      current_scan_stamp,
      imu_samples,
      imu_orientation_prediction_max_time_delta_s_,
      wheel_slope_imu_status);
    if (wheel_slope_world_R_base.has_value()) {
      const auto & rotation = *wheel_slope_world_R_base;
      wheel_slope_pitch_rad = std::atan2(
        -rotation(2, 0),
        std::hypot(rotation(0, 0), rotation(1, 0)));
    }
  }
  Eigen::Isometry3d fallback = predictWithWheelOdom(
    current_scan_stamp,
    !suppress_constant_velocity_prediction,
    wheel_slope_pitch_rad);
  Eigen::Isometry3d scan_to_scan_guess = Eigen::Isometry3d::Identity();
  std::optional<Eigen::Quaterniond> previous_scan_imu_delta;
  if (imu_rotation_prediction_enabled_) {
    if (have_previous_scan_stamp) {
      previous_scan_imu_delta = integrateImuRotationDelta(
        previous_scan_stamp,
        current_scan_stamp,
        imu_samples,
        imu_rotation_prediction_status);
    } else {
      imu_rotation_prediction_status = "missing_previous_scan_stamp";
    }
    if (previous_scan_imu_delta) {
      const Eigen::Matrix3d imu_rotation =
        previous_scan_imu_delta->normalized().toRotationMatrix();
      fallback.linear() =
        projectToRigidIsometry(map_T_base_).linear() * imu_rotation;
      fallback = projectToRigidIsometry(fallback);
      used_imu_rotation_prediction = true;
    }
  }

  if (!use_scan_to_scan_prediction_ || !have_previous_tracking_scan_ ||
    !previous_tracking_scan_base_)
  {
    if (use_scan_to_scan_prediction_) {
      scan_to_scan_prediction_status = "missing_tracking_reference";
    }
    return projectToRigidIsometry(fallback);
  }

  scan_to_scan_reference_age_s =
    (current_scan_stamp - previous_tracking_scan_stamp_).seconds();
  if (!std::isfinite(scan_to_scan_reference_age_s) ||
    scan_to_scan_reference_age_s <= 0.0)
  {
    scan_to_scan_prediction_status = "invalid_tracking_reference_age";
    return projectToRigidIsometry(fallback);
  }
  const double max_reference_age_s = std::max(0.0, scan_to_scan_max_reference_age_s_);
  if (max_reference_age_s > 0.0 &&
    scan_to_scan_reference_age_s > max_reference_age_s)
  {
    scan_to_scan_prediction_status = "stale_tracking_reference";
    return projectToRigidIsometry(fallback);
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
        imu_samples,
        tracking_imu_status);
    }
    if (tracking_imu_delta) {
      scan_to_scan_guess.linear() =
        tracking_imu_delta->normalized().toRotationMatrix();
      used_imu_rotation_prediction = true;
      imu_rotation_prediction_status = tracking_imu_status;
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
    return projectToRigidIsometry(fallback);
  }

  used_scan_to_scan = true;
  scan_to_scan_prediction_status = "applied";
  return projectToRigidIsometry(
    projectToRigidIsometry(previous_tracking_pose_) *
    projectToRigidIsometry(relative.map_T_base));
}

std::optional<Eigen::Matrix3d> ScanToMapIcp::imuWorldBaseOrientationAt(
  const rclcpp::Time & stamp,
  const std::vector<ImuSample> & imu_samples,
  const double max_time_delta_s,
  std::string & status) const
{
  status = "unavailable";
  if (imu_samples.empty()) {
    status = "missing_imu";
    return std::nullopt;
  }

  const double target_time_s = stamp.seconds();
  if (!std::isfinite(target_time_s) || !std::isfinite(max_time_delta_s) ||
    max_time_delta_s < 0.0)
  {
    status = "invalid_query";
    return std::nullopt;
  }

  const auto sample_time_s = [](const ImuSample & sample) {
      return sample.stamp.seconds();
    };
  const auto upper = std::lower_bound(
    imu_samples.begin(),
    imu_samples.end(),
    target_time_s,
    [&sample_time_s](const ImuSample & sample, const double time_s) {
      return sample_time_s(sample) < time_s;
    });

  std::optional<Eigen::Quaterniond> world_q_imu;
  if (upper != imu_samples.end()) {
    const double upper_delta_s = std::abs(sample_time_s(*upper) - target_time_s);
    if (upper_delta_s <= 1.0e-9 && upper->orientation_valid) {
      world_q_imu = upper->orientation.normalized();
      status = "exact";
    }
  }

  if (!world_q_imu && upper != imu_samples.begin() && upper != imu_samples.end()) {
    const auto lower = std::prev(upper);
    const double lower_time_s = sample_time_s(*lower);
    const double upper_time_s = sample_time_s(*upper);
    const double lower_delta_s = target_time_s - lower_time_s;
    const double upper_delta_s = upper_time_s - target_time_s;
    const double bracket_s = upper_time_s - lower_time_s;
    if (lower->orientation_valid && upper->orientation_valid &&
      lower_delta_s >= 0.0 && upper_delta_s >= 0.0 &&
      lower_delta_s <= max_time_delta_s && upper_delta_s <= max_time_delta_s &&
      bracket_s > 1.0e-9)
    {
      const double ratio = lower_delta_s / bracket_s;
      world_q_imu = lower->orientation.slerp(ratio, upper->orientation).normalized();
      status = "interpolated";
    }
  }

  if (!world_q_imu) {
    const ImuSample * nearest = nullptr;
    double nearest_delta_s = std::numeric_limits<double>::infinity();
    if (upper != imu_samples.end() && upper->orientation_valid) {
      nearest = &(*upper);
      nearest_delta_s = std::abs(sample_time_s(*upper) - target_time_s);
    }
    if (upper != imu_samples.begin()) {
      const auto lower = std::prev(upper);
      const double lower_delta_s = std::abs(sample_time_s(*lower) - target_time_s);
      if (lower->orientation_valid && lower_delta_s < nearest_delta_s) {
        nearest = &(*lower);
        nearest_delta_s = lower_delta_s;
      }
    }
    if (nearest != nullptr && nearest_delta_s <= max_time_delta_s) {
      world_q_imu = nearest->orientation.normalized();
      status = "nearest";
    }
  }

  if (!world_q_imu || !world_q_imu->coeffs().allFinite()) {
    status = "orientation_unavailable";
    return std::nullopt;
  }

  const Eigen::Matrix3d base_R_imu = rotationFromRpyDegrees(
    deskew_imu_to_base_roll_deg_,
    deskew_imu_to_base_pitch_deg_,
    deskew_imu_to_base_yaw_deg_);
  const Eigen::Matrix3d world_R_base =
    world_q_imu->toRotationMatrix() * base_R_imu.transpose();
  if (!world_R_base.allFinite()) {
    status = "orientation_invalid";
    return std::nullopt;
  }
  return world_R_base;
}

std::optional<Eigen::Quaterniond> ScanToMapIcp::integrateImuRotationDelta(
  const rclcpp::Time & start_stamp,
  const rclcpp::Time & end_stamp,
  const std::vector<ImuSample> & imu_samples,
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

  if (imu_samples.empty()) {
    status = "missing_imu";
    return std::nullopt;
  }

  if (imu_rotation_prediction_source_ == "orientation") {
    std::string start_status;
    const auto start_world_R_base = imuWorldBaseOrientationAt(
      start_stamp,
      imu_samples,
      imu_orientation_prediction_max_time_delta_s_,
      start_status);
    if (!start_world_R_base) {
      status = "orientation_start_unavailable";
      return std::nullopt;
    }
    std::string end_status;
    const auto end_world_R_base = imuWorldBaseOrientationAt(
      end_stamp,
      imu_samples,
      imu_orientation_prediction_max_time_delta_s_,
      end_status);
    if (!end_world_R_base) {
      status = "orientation_end_unavailable";
      return std::nullopt;
    }

    const Eigen::Matrix3d base_start_R_base_end =
      start_world_R_base->transpose() * *end_world_R_base;
    const Eigen::AngleAxisd rotation_delta(base_start_R_base_end);
    const double scaled_angle =
      rotation_delta.angle() * imu_rotation_prediction_scale_;
    if (!std::isfinite(scaled_angle) || !rotation_delta.axis().allFinite()) {
      status = "orientation_delta_invalid";
      return std::nullopt;
    }

    status = "orientation_applied";
    if (std::abs(scaled_angle) <= 1.0e-12) {
      return Eigen::Quaterniond::Identity();
    }
    return Eigen::Quaterniond(
      Eigen::AngleAxisd(scaled_angle, rotation_delta.axis())).normalized();
  }

  const double angular_velocity_scale =
    deskew_angular_velocity_scale_ * imu_rotation_prediction_scale_;
  return integrateAngularVelocityRotationDelta(
    start_stamp,
    end_stamp,
    imu_samples,
    imu_rotation_prediction_max_dt_s_,
    angular_velocity_scale,
    status);
}

std::optional<Eigen::Quaterniond> ScanToMapIcp::integrateAngularVelocityRotationDelta(
  const rclcpp::Time & start_stamp,
  const rclcpp::Time & end_stamp,
  const std::vector<ImuSample> & imu_samples,
  const double max_interval_s,
  const double angular_velocity_scale,
  std::string & status) const
{
  status = "unavailable";
  const double start_s = start_stamp.seconds();
  const double end_s = end_stamp.seconds();
  const double dt_s = end_s - start_s;
  if (!std::isfinite(dt_s) || dt_s <= 1.0e-6) {
    status = "invalid_interval";
    return std::nullopt;
  }
  if (max_interval_s > 0.0 && dt_s > max_interval_s) {
    status = "interval_too_long";
    return std::nullopt;
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
  const auto angular_velocity_base =
    [&imu_to_base, angular_velocity_scale](const ImuSample & sample) {
      return imu_to_base * sample.angular_velocity * angular_velocity_scale;
    };

  std::size_t start_index = 0;
  while (start_index + 1 < imu_samples.size() &&
    sample_time_s(imu_samples[start_index + 1]) <= start_s)
  {
    ++start_index;
  }

  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  double current_time_s = start_s;
  Eigen::Vector3d current_angular_velocity = angular_velocity_base(
    imu_samples[start_index]);
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
    current_angular_velocity = angular_velocity_base(sample);
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

  status = "angular_velocity_applied";
  return rotation.normalized();
}

void ScanToMapIcp::commitWheelPrediction()
{
  std::lock_guard<std::mutex> lock(wheel_mutex_);
  if (!have_pending_wheel_odom_) {
    return;
  }

  last_used_wheel_odom_ = pending_wheel_odom_;
  last_used_wheel_stamp_ = pending_wheel_stamp_;
  have_pending_wheel_odom_ = false;
}

void ScanToMapIcp::recordAcceptedPose()
{
  map_T_base_ = projectToRigidIsometry(map_T_base_);
  if (!have_last_accepted_pose_) {
    last_accepted_pose_ = map_T_base_;
    have_last_accepted_pose_ = true;
    return;
  }

  previous_accepted_pose_ = last_accepted_pose_;
  last_accepted_pose_ = map_T_base_;
  have_previous_accepted_pose_ = true;
}

void ScanToMapIcp::rebaseAcceptedPoseHistory()
{
  map_T_base_ = projectToRigidIsometry(map_T_base_);
  previous_accepted_pose_ = map_T_base_;
  last_accepted_pose_ = map_T_base_;
  have_previous_accepted_pose_ = true;
  have_last_accepted_pose_ = true;
}

void ScanToMapIcp::recordAcceptedPoseOrRebaseAfterShock()
{
  if (constant_velocity_rebase_pending_) {
    rebaseAcceptedPoseHistory();
    constant_velocity_rebase_pending_ = false;
    return;
  }
  recordAcceptedPose();
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
  const Eigen::Isometry3d & prediction,
  const double registration_scan_gap_s) const
{
  if (registration_method_ == "voxel_surfel") {
    const VoxelSurfelRegistration registration(surfel_config_);
    const auto surfel_result = registration.align(*scan_base, *local_map, prediction);

    RegistrationResult result;
    result.surfel_converged = surfel_result.converged;
    result.surfel_reason = surfel_result.reason;
    result.surfel_target_voxels = static_cast<int>(surfel_result.target_voxels);
    result.surfel_planar_surfels = static_cast<int>(surfel_result.planar_surfels);
    result.surfel_correspondences = static_cast<int>(surfel_result.correspondences);
    result.surfel_observable_rank = surfel_result.observable_rank;
    result.surfel_correspondence_ratio = surfel_result.correspondence_ratio;
    result.surfel_rmse_m = surfel_result.rmse;
    result.surfel_median_abs_residual_m = surfel_result.median_abs_residual;
    result.surfel_p95_abs_residual_m = surfel_result.p95_abs_residual;
    result.surfel_robust_cost = surfel_result.robust_cost;
    result.surfel_robust_downweight_ratio = surfel_result.robust_downweight_ratio;
    result.surfel_hessian_condition = surfel_result.hessian_condition;
    result.surfel_target_build_time_ms = surfel_result.target_build_time_ms;
    result.surfel_optimization_time_ms = surfel_result.optimization_time_ms;
    result.surfel_total_time_ms = surfel_result.total_time_ms;
    result.surfel_hessian_eigenvalues = surfel_result.hessian_eigenvalues;
    result.surfel_weak_rotation_axes = surfel_result.weak_rotation_axes;
    result.surfel_weak_translation_axes = surfel_result.weak_translation_axes;
    result.registration_information_valid = surfel_result.hessian_valid;
    result.registration_information_observable_rank = surfel_result.observable_rank;
    result.registration_information_hessian_condition = surfel_result.hessian_condition;
    result.registration_information_evaluation_runtime_ms = surfel_result.total_time_ms;
    result.registration_information_hessian_map_rt = surfel_result.physical_hessian;
    result.registration_information_pose_map_T_base = surfel_result.target_T_source;
    result.registration_information_status = surfel_result.hessian_valid ?
      "voxel_surfel_alignment" : "voxel_surfel_" + surfel_result.reason;
    result.map_T_base = surfel_result.target_T_source;
    result.iterations = surfel_result.iterations;
    result.registration_coarse_iterations = surfel_result.iterations;
    result.fitness = std::isfinite(surfel_result.rmse) ?
      surfel_result.rmse * surfel_result.rmse :
      std::numeric_limits<double>::infinity();
    result.registration_coarse_fitness = result.fitness;
    const double surfel_max_correspondence_m =
      std::max(1.0e-3, surfel_config_.max_center_distance);
    const double requested_quality_distance_m = robust_registration_enabled_ ?
      robust_registration_quality_max_correspondence_m_ :
      surfel_max_correspondence_m;
    result.registration_quality_max_correspondence_m = std::clamp(
      requested_quality_distance_m,
      1.0e-3,
      surfel_max_correspondence_m);

    const Eigen::Matrix3d linear = result.map_T_base.linear();
    result.transform_determinant = linear.determinant();
    result.transform_orthogonality_error =
      (linear.transpose() * linear - Eigen::Matrix3d::Identity()).norm();
    const Eigen::Isometry3d correction = prediction.inverse() * result.map_T_base;
    result.correction_translation_m = correction.translation().norm();
    result.correction_rotation_deg =
      Eigen::AngleAxisd(correction.linear()).angle() * 180.0 / M_PI;
    result.registration_scan_gap_s = std::max(0.0, registration_scan_gap_s);

    CloudT aligned;
    pcl::transformPointCloud(
      *scan_base,
      aligned,
      result.map_T_base.matrix().cast<float>());
    evaluateRegistrationOverlap(
      aligned,
      local_map,
      result.registration_quality_max_correspondence_m,
      result);
    result.registration_coarse_inlier_ratio = result.inlier_ratio;
    result.registration_coarse_inlier_rmse_m = result.inlier_rmse_m;

    if (!surfel_result.accepted) {
      result.rejection_reason = "surfel_" + surfel_result.reason;
      return result;
    }

    // The surfel backend accepts a bounded max-iteration solution after its
    // own quality and observability gates, even if it did not hit a small-step
    // convergence threshold.
    result.converged = true;
    result.accepted = registrationCorrectionWithinLimits(result);
    return result;
  }

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
      registration_scan_gap_s,
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
      stage_result.registration_scan_gap_s = std::max(
        0.0,
        registration_scan_gap_s);
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

void ScanToMapIcp::attachRegistrationInformation(
  const CloudConstPtr & source,
  const CloudConstPtr & target,
  RegistrationResult & result) const
{
  if (pose_graph_information_mode_ != "registration_observability") {
    result.registration_information_status = "configured_diagonal_mode";
    return;
  }
  if (result.registration_information_valid &&
    result.registration_information_hessian_map_rt.allFinite() &&
    result.registration_information_pose_map_T_base.matrix().isApprox(
      result.map_T_base.matrix(), 1.0e-9))
  {
    // Voxel-surfel already evaluates its final robust Hessian after the last
    // accepted step. Rebuilding the same target model at every keyframe only
    // adds callback latency. A pose changed by a later motion constraint still
    // falls through to an exact fixed-pose re-evaluation below.
    result.registration_information_evaluation_runtime_ms = 0.0;
    result.registration_information_status = "reused_registration_linearization";
    return;
  }
  result.registration_information_valid = false;
  result.registration_information_observable_rank = 0;
  result.registration_information_hessian_condition = 0.0;
  result.registration_information_evaluation_runtime_ms = 0.0;
  result.registration_information_hessian_map_rt.setZero();
  result.registration_information_pose_map_T_base = result.map_T_base;
  if (!source || source->empty() || !target || target->empty() ||
    !result.map_T_base.matrix().allFinite())
  {
    result.registration_information_status = "invalid_fixed_pose_inputs";
    return;
  }

  const VoxelSurfelRegistration evaluator(surfel_config_);
  const auto estimate = evaluator.evaluateAtPose(*source, *target, result.map_T_base);
  result.registration_information_valid = estimate.hessian_valid;
  result.registration_information_observable_rank = estimate.observable_rank;
  result.registration_information_hessian_condition = estimate.hessian_condition;
  result.registration_information_evaluation_runtime_ms = estimate.total_time_ms;
  result.registration_information_hessian_map_rt = estimate.physical_hessian;
  result.registration_information_pose_map_T_base = result.map_T_base;
  result.registration_information_status =
    "fixed_pose_point_to_plane_" + estimate.reason;
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

void ScanToMapIcp::refreshRegistrationQualityAtPose(
  const CloudConstPtr & scan_base,
  const CloudConstPtr & local_map,
  const Eigen::Isometry3d & prediction,
  RegistrationResult & result) const
{
  result.inlier_count = 0;
  result.inlier_ratio = 0.0;
  result.inlier_rmse_m = std::numeric_limits<double>::infinity();
  if (!scan_base || scan_base->empty() || !local_map || local_map->empty() ||
    !result.map_T_base.matrix().allFinite())
  {
    return;
  }

  CloudT aligned;
  pcl::transformPointCloud(
    *scan_base,
    aligned,
    result.map_T_base.matrix().cast<float>());
  evaluateRegistrationOverlap(
    aligned,
    local_map,
    result.registration_quality_max_correspondence_m,
    result);

  const Eigen::Matrix3d linear = result.map_T_base.linear();
  result.transform_determinant = linear.determinant();
  result.transform_orthogonality_error =
    (linear.transpose() * linear - Eigen::Matrix3d::Identity()).norm();
  const Eigen::Isometry3d correction =
    projectToRigidIsometry(prediction).inverse() *
    projectToRigidIsometry(result.map_T_base);
  result.correction_translation_m = correction.translation().norm();
  result.correction_rotation_deg = rotationAngleDeg(correction.linear());
}

bool ScanToMapIcp::registrationCorrectionWithinLimits(
  RegistrationResult & result) const
{
  GapAwareTranslationGateConfig translation_gate_config;
  translation_gate_config.enabled = registration_gap_relaxation_enabled_;
  translation_gate_config.base_max_translation_m =
    max_registration_correction_translation_m_;
  translation_gate_config.min_gap_s = registration_gap_relaxation_min_gap_s_;
  translation_gate_config.max_speed_mps =
    registration_gap_relaxation_max_speed_mps_;
  translation_gate_config.max_translation_m =
    registration_gap_relaxation_max_translation_m_;
  translation_gate_config.min_inlier_ratio =
    registration_gap_relaxation_min_inlier_ratio_;
  translation_gate_config.max_inlier_rmse_m =
    registration_gap_relaxation_max_inlier_rmse_m_;
  const TranslationGateDecision translation_gate =
    computeGapAwareTranslationGate(
    translation_gate_config,
    result.registration_scan_gap_s,
    result.inlier_ratio,
    result.inlier_rmse_m);
  result.registration_translation_gate_m = translation_gate.max_translation_m;
  result.registration_gap_relaxation_applied =
    translation_gate.gap_relaxation_applied;

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

  const double max_translation = result.registration_translation_gate_m;
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

  if (submap_enabled_ && !submaps_.empty()) {
    int selected = 0;
    for (auto it = submaps_.rbegin(); it != submaps_.rend(); ++it) {
      if (selected >= max_local_submaps_) {
        break;
      }
      if (!it->cloud_map || it->cloud_map->empty() || it->keyframe_indices.empty()) {
        continue;
      }
      const std::size_t anchor_index = submapAnchorKeyframeIndex(*it);
      if (anchor_index >= keyframes_.size() ||
        !poseWithinLocalMapWindow(
          keyframes_[anchor_index].map_T_base,
          center_pose,
          spatial_mapping_config_))
      {
        continue;
      }
      *local_map += *(it->cloud_map);
      ++selected;
    }

    if (local_map->empty() &&
      spatial_mapping_config_.mode == MappingSpatialMode::kPlanar)
    {
      for (auto it = submaps_.rbegin(); it != submaps_.rend(); ++it) {
        if (it->cloud_map && !it->cloud_map->empty()) {
          *local_map += *(it->cloud_map);
          break;
        }
      }
    }
    return downsampleCloud(local_map, map_voxel_leaf_m_);
  }

  int selected = 0;
  for (auto it = keyframes_.rbegin(); it != keyframes_.rend(); ++it) {
    if (selected >= max_local_keyframes_) {
      break;
    }
    if (!poseWithinLocalMapWindow(
        it->map_T_base,
        center_pose,
        spatial_mapping_config_))
    {
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

ScanToMapIcp::CloudPtr ScanToMapIcp::buildCandidateMapAroundSubmap(
  const std::size_t submap_index) const
{
  auto candidate_map = std::make_shared<CloudT>();
  if (submap_index >= submaps_.size()) {
    return candidate_map;
  }
  const auto & submap = submaps_[submap_index];
  if (submap.cloud_map) {
    *candidate_map = *(submap.cloud_map);
  }
  return downsampleCloud(candidate_map, map_voxel_leaf_m_);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildCandidateMappingMapAroundKeyframe(
  const int keyframe_index,
  const int latest_candidate_exclusive) const
{
  auto candidate_map = std::make_shared<CloudT>();
  const int historical_end = std::min(
    static_cast<int>(keyframes_.size()),
    std::max(0, latest_candidate_exclusive));
  if (keyframe_index < 0 || keyframe_index >= historical_end) {
    return candidate_map;
  }

  const int neighbor_count = std::max(0, place_recognition_candidate_map_neighbors_);
  const int first = std::max(0, keyframe_index - neighbor_count);
  const int last = std::min(historical_end - 1, keyframe_index + neighbor_count);
  for (int index = first; index <= last; ++index) {
    const auto & cloud = keyframes_[static_cast<std::size_t>(index)].mapping_cloud_map;
    if (cloud && !cloud->empty()) {
      *candidate_map += *cloud;
    }
  }
  return downsampleCloud(candidate_map, map_voxel_leaf_m_);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildCandidateMappingMapAroundSubmap(
  const std::size_t submap_index) const
{
  auto candidate_map = std::make_shared<CloudT>();
  if (submap_index >= submaps_.size()) {
    return candidate_map;
  }
  const auto & submap = submaps_[submap_index];
  if (!submap.completed || !submap.mapping_cloud_map) {
    return candidate_map;
  }
  *candidate_map = *submap.mapping_cloud_map;
  return downsampleCloud(candidate_map, map_voxel_leaf_m_);
}

void ScanToMapIcp::refinePlaceRecognitionWithMappingCloud(
  const CloudConstPtr & registration_scan_base,
  const CloudConstPtr & mapping_scan_base,
  const CloudConstPtr & registration_candidate_map,
  const CloudConstPtr & mapping_candidate_map,
  const Eigen::Isometry3d & candidate_seed_pose,
  RegistrationResult & result) const
{
  result.place_recognition_mapping_refinement_status =
    place_recognition_mapping_cloud_refinement_enabled_ ? "not_attempted" : "disabled";
  result.place_recognition_mapping_refinement_applied = false;
  result.place_recognition_mapping_refinement_verified = false;
  if (!place_recognition_mapping_cloud_refinement_enabled_) {
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  result.place_recognition_mapping_refinement_attempted = true;
  MappingCloudRefinementMetrics metrics;
  metrics.source_points = mapping_scan_base ?
    static_cast<int>(mapping_scan_base->size()) : 0;
  metrics.target_points = mapping_candidate_map ?
    static_cast<int>(mapping_candidate_map->size()) : 0;

  auto finish = [&result, &metrics, &start](
    const MappingCloudRefinementDecision & decision) {
      result.place_recognition_mapping_refinement_applied = decision.apply;
      result.place_recognition_mapping_refinement_verified = decision.apply;
      result.place_recognition_mapping_refinement_status =
        mappingCloudRefinementReasonString(decision.reason);
      result.place_recognition_mapping_refinement_source_points = metrics.source_points;
      result.place_recognition_mapping_refinement_target_points = metrics.target_points;
      result.place_recognition_mapping_refinement_observable_rank = metrics.observable_rank;
      result.place_recognition_mapping_refinement_before_inlier_ratio =
        metrics.before_inlier_ratio;
      result.place_recognition_mapping_refinement_before_inlier_rmse_m =
        metrics.before_inlier_rmse_m;
      result.place_recognition_mapping_refinement_after_inlier_ratio =
        metrics.after_inlier_ratio;
      result.place_recognition_mapping_refinement_after_inlier_rmse_m =
        metrics.after_inlier_rmse_m;
      result.place_recognition_mapping_refinement_before_target_coverage_ratio =
        metrics.before_target_coverage_ratio;
      result.place_recognition_mapping_refinement_after_target_coverage_ratio =
        metrics.after_target_coverage_ratio;
      result.place_recognition_mapping_refinement_before_score = decision.before_score;
      result.place_recognition_mapping_refinement_after_score = decision.after_score;
      result.place_recognition_mapping_refinement_primary_before_score =
        decision.primary_before_score;
      result.place_recognition_mapping_refinement_primary_after_score =
        decision.primary_after_score;
      result.place_recognition_mapping_refinement_correction_translation_m =
        metrics.correction_translation_m;
      result.place_recognition_mapping_refinement_correction_rotation_deg =
        metrics.correction_rotation_deg;
      result.place_recognition_mapping_refinement_hessian_condition =
        metrics.hessian_condition;
      result.place_recognition_mapping_refinement_runtime_ms =
        std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    };

  if (metrics.source_points < mapping_cloud_refinement_config_.min_source_points ||
    metrics.target_points < mapping_cloud_refinement_config_.min_target_points)
  {
    finish(evaluateMappingCloudRefinement(mapping_cloud_refinement_config_, metrics));
    return;
  }

  const double quality_distance =
    place_recognition_mapping_refinement_quality_max_correspondence_m_;
  auto quality_at_pose = [this, quality_distance](
    const CloudConstPtr & source,
    const CloudConstPtr & target,
    const Eigen::Isometry3d & pose) {
      RegistrationResult quality;
      quality.map_T_base = pose;
      quality.registration_quality_max_correspondence_m = quality_distance;
      refreshRegistrationQualityAtPose(source, target, pose, quality);
      return quality;
    };

  const Eigen::Isometry3d coarse_pose = projectToRigidIsometry(result.map_T_base);
  const RegistrationResult mapping_before = quality_at_pose(
    mapping_scan_base, mapping_candidate_map, coarse_pose);
  const RegistrationResult mapping_reverse_before = quality_at_pose(
    mapping_candidate_map, mapping_scan_base, coarse_pose.inverse());
  const RegistrationResult primary_before = quality_at_pose(
    registration_scan_base, registration_candidate_map, coarse_pose);
  metrics.before_inlier_ratio = mapping_before.inlier_ratio;
  metrics.before_inlier_rmse_m = mapping_before.inlier_rmse_m;
  metrics.before_target_coverage_ratio = mapping_reverse_before.inlier_ratio;
  metrics.primary_before_inlier_ratio = primary_before.inlier_ratio;
  metrics.primary_before_inlier_rmse_m = primary_before.inlier_rmse_m;

  RegistrationResult refined = registerScanToMap(
    mapping_scan_base, mapping_candidate_map, coarse_pose);
  attachRegistrationInformation(mapping_scan_base, mapping_candidate_map, refined);
  const RegistrationResult mapping_after = quality_at_pose(
    mapping_scan_base, mapping_candidate_map, refined.map_T_base);
  const RegistrationResult mapping_reverse_after = quality_at_pose(
    mapping_candidate_map, mapping_scan_base, refined.map_T_base.inverse());
  const RegistrationResult primary_after = quality_at_pose(
    registration_scan_base, registration_candidate_map, refined.map_T_base);
  metrics.registration_accepted = refined.accepted;
  metrics.after_inlier_ratio = mapping_after.inlier_ratio;
  metrics.after_inlier_rmse_m = mapping_after.inlier_rmse_m;
  metrics.after_target_coverage_ratio = mapping_reverse_after.inlier_ratio;
  metrics.primary_after_inlier_ratio = primary_after.inlier_ratio;
  metrics.primary_after_inlier_rmse_m = primary_after.inlier_rmse_m;
  const Eigen::Isometry3d refinement_delta = projectToRigidIsometry(
    coarse_pose.inverse() * refined.map_T_base);
  metrics.correction_translation_m = refinement_delta.translation().norm();
  metrics.correction_rotation_deg = rotationAngleDeg(refinement_delta.linear());
  metrics.observable_rank = refined.surfel_observable_rank;
  metrics.hessian_condition = refined.surfel_hessian_condition;

  const MappingCloudRefinementDecision decision = evaluateMappingCloudRefinement(
    mapping_cloud_refinement_config_, metrics);
  finish(decision);
  if (!decision.apply && result.place_recognition_large_pose_jump_provisional) {
    const MappingCloudVerifiedCoarseDecision coarse_decision =
      evaluateMappingCloudVerifiedCoarse(
      mapping_cloud_verified_coarse_config_,
      mapping_cloud_refinement_config_,
      decision,
      metrics);
    if (coarse_decision.accept) {
      result.place_recognition_mapping_refinement_verified = true;
      result.place_recognition_mapping_refinement_status = "verified_coarse";
    } else if (mapping_cloud_verified_coarse_config_.enabled) {
      result.place_recognition_mapping_refinement_status +=
        std::string("_verified_coarse_") +
        mappingCloudVerifiedCoarseReasonString(coarse_decision.reason);
    }
  }
  if (!decision.apply) {
    return;
  }

  result.map_T_base = projectToRigidIsometry(refined.map_T_base);
  result.inlier_count = primary_after.inlier_count;
  result.inlier_ratio = primary_after.inlier_ratio;
  result.inlier_rmse_m = primary_after.inlier_rmse_m;
  result.place_recognition_candidate_registration_inlier_ratio = primary_after.inlier_ratio;
  result.place_recognition_candidate_registration_inlier_rmse_m =
    primary_after.inlier_rmse_m;
  result.fitness = primary_after.inlier_rmse_m * primary_after.inlier_rmse_m;
  result.iterations += refined.iterations;
  result.transform_determinant = primary_after.transform_determinant;
  result.transform_orthogonality_error = primary_after.transform_orthogonality_error;
  result.registration_information_valid = refined.registration_information_valid;
  result.registration_information_observable_rank =
    refined.registration_information_observable_rank;
  result.registration_information_hessian_condition =
    refined.registration_information_hessian_condition;
  result.registration_information_evaluation_runtime_ms =
    refined.registration_information_evaluation_runtime_ms;
  result.registration_information_hessian_map_rt =
    refined.registration_information_hessian_map_rt;
  result.registration_information_pose_map_T_base =
    refined.registration_information_pose_map_T_base;
  result.registration_information_status = refined.registration_information_status;
  const Eigen::Isometry3d seed_delta = projectToRigidIsometry(
    candidate_seed_pose.inverse() * result.map_T_base);
  result.correction_translation_m = seed_delta.translation().norm();
  result.correction_rotation_deg = rotationAngleDeg(seed_delta.linear());
}

void ScanToMapIcp::initializeFromFirstScan(
  const CloudConstPtr & registration_scan_base,
  const CloudConstPtr & mapping_scan_base,
  const int registration_only_support_points,
  const std::vector<TerrainSensorScan> & terrain_sensor_scans,
  const rclcpp::Time & stamp,
  const std::optional<GravitySample> & gravity_sample,
  const bool gravity_sample_stale)
{
  initialized_ = true;
  map_T_base_ = Eigen::Isometry3d::Identity();
  last_keyframe_pose_ = map_T_base_;
  last_registration_ = RegistrationResult{};
  last_registration_.converged = true;
  last_registration_.accepted = true;
  last_registration_.fitness = 0.0;
  last_registration_.iterations = 0;
  last_registration_.map_T_base = map_T_base_;
  last_registration_.rejection_reason = "initialized";
  recordAcceptedPose();
  recordTrackingScan(registration_scan_base, map_T_base_, stamp);
  (void)predictWithWheelOdom(stamp);
  addKeyframe(
    registration_scan_base,
    mapping_scan_base,
    registration_only_support_points,
    terrain_sensor_scans,
    stamp,
    gravity_sample,
    gravity_sample_stale);
  RCLCPP_INFO(get_logger(), "Initialized scan-to-map SLAM from first scan");
}

void ScanToMapIcp::addKeyframe(
  const CloudConstPtr & registration_scan_base,
  const CloudConstPtr & mapping_scan_base,
  const int registration_only_support_points,
  const std::vector<TerrainSensorScan> & terrain_sensor_scans,
  const rclcpp::Time & stamp,
  const std::optional<GravitySample> & gravity_sample,
  const bool gravity_sample_stale,
  const RegistrationResult * registration_result)
{
  const int new_keyframe_index = static_cast<int>(keyframes_.size());
  if (new_keyframe_index > 0) {
    const auto & previous = keyframes_.back();
    PoseGraphEdge odom_edge;
    odom_edge.from_index = new_keyframe_index - 1;
    odom_edge.to_index = new_keyframe_index;
    odom_edge.from_T_to = previous.map_T_base.inverse() * map_T_base_;
    odom_edge.translation_weight = std::max(0.0, pose_graph_odom_translation_weight_);
    odom_edge.vertical_translation_weight = odom_edge.translation_weight;
    odom_edge.rotation_weight = std::max(0.0, pose_graph_odom_rotation_weight_);
    Vector6d configured_information;
    configured_information <<
      odom_edge.translation_weight,
      odom_edge.translation_weight,
      odom_edge.vertical_translation_weight,
      odom_edge.rotation_weight,
      odom_edge.rotation_weight,
      odom_edge.rotation_weight;
    odom_edge.information_matrix = configuredDiagonalInformation(configured_information);
    if (pose_graph_information_mode_ == "registration_observability" &&
      registration_result != nullptr && registration_result->registration_information_valid)
    {
      const RegistrationInformationResult information =
        registrationHessianToPoseGraphInformation(
        registration_result->registration_information_hessian_map_rt,
        previous.map_T_base,
        configured_information,
        pose_graph_information_config_);
      if (information.valid) {
        odom_edge.information_matrix = information.information_matrix;
        odom_edge.registration_information_used = true;
        odom_edge.registration_observable_rank = information.observable_rank;
        odom_edge.registration_hessian_condition = information.observable_condition;
        odom_edge.registration_information_min_gain =
          information.minimum_information_gain;
      }
    }
    odom_edge.loop_closure = false;
    pose_graph_edges_.push_back(odom_edge);
  }

  Keyframe keyframe;
  keyframe.archive_id = next_keyframe_archive_id_++;
  keyframe.stamp = stamp;
  keyframe.map_T_base = map_T_base_;
  keyframe.cloud_base = downsampleCloud(registration_scan_base, map_voxel_leaf_m_);
  keyframe.mapping_cloud_base = downsampleCloud(mapping_scan_base, map_voxel_leaf_m_);
  if (terrain_config_.enabled || !keyframe_archive_directory_.empty()) {
    keyframe.terrain_sensor_scans = terrain_sensor_scans;
    const double sensor_view_voxel_leaf_m = terrain_config_.enabled ?
      terrain_config_.classification_voxel_size_m :
      keyframe_archive_voxel_leaf_m_;
    for (auto & sensor_scan : keyframe.terrain_sensor_scans) {
      sensor_scan.cloud_base = downsampleCloud(
        sensor_scan.cloud_base,
        sensor_view_voxel_leaf_m);
      sensor_scan.labels.clear();
      sensor_scan.roughness_scores.clear();
    }
    if (!keyframe_archive_directory_.empty()) {
      keyframe.archive_sensor_scans = keyframe.terrain_sensor_scans;
    }
    if (!terrain_config_.enabled) {
      keyframe.terrain_sensor_scans.clear();
    }
  }
  keyframe.cloud_map = filterMapCloud(transformCloud(keyframe.cloud_base, map_T_base_));
  keyframe.mapping_cloud_map = filterMapCloud(
    transformCloud(keyframe.mapping_cloud_base, map_T_base_));
  keyframe.descriptor = computeScanDescriptor(mapping_scan_base);
  keyframe.registration_only_support_points = std::max(
    0,
    registration_only_support_points);
  classifyKeyframeTerrain(keyframe, gravity_sample, gravity_sample_stale);
  keyframes_.push_back(std::move(keyframe));
  appendKeyframeTerrainEvidence(keyframes_.back());
  appendLatestKeyframeToSubmaps();
  last_keyframe_pose_ = map_T_base_;

  bool pruned_oldest_keyframe = false;
  if (static_cast<int>(keyframes_.size()) > max_keyframes_) {
    keyframes_.erase(keyframes_.begin());
    ++pruned_keyframe_count_;
    prunePoseGraphEdgesAfterOldestKeyframeRemoval();
    pruneTerrainTraversalSamplesAfterOldestKeyframeRemoval();
    pruneSubmapsAfterOldestKeyframeRemoval();
    pruned_oldest_keyframe = true;
  }

  if (pruned_oldest_keyframe) {
    rebuildGlobalMap();
    publishPoseGraphMarkers(stamp);
    return;
  }

  const auto & latest = keyframes_.back();
  if (latest.mapping_cloud_map && !latest.mapping_cloud_map->empty()) {
    auto merged = std::make_shared<CloudT>();
    if (global_map_) {
      *merged = *global_map_;
    }
    *merged += *(latest.mapping_cloud_map);
    global_map_ = downsampleCloud(merged, map_voxel_leaf_m_);
  }
  latest_local_map_ = buildLocalMap(map_T_base_);
  publishPoseGraphMarkers(stamp);
}

void ScanToMapIcp::classifyKeyframeTerrain(
  Keyframe & keyframe,
  const std::optional<GravitySample> & gravity_sample,
  const bool gravity_sample_stale)
{
  for (auto & sensor_scan : keyframe.terrain_sensor_scans) {
    sensor_scan.labels.clear();
    sensor_scan.roughness_scores.clear();
  }
  last_terrain_classification_ = TerrainClassificationStats{};
  terrain_last_keyframe_input_points_ = 0U;
  for (const auto & sensor_scan : keyframe.terrain_sensor_scans) {
    if (sensor_scan.cloud_base) {
      terrain_last_keyframe_input_points_ += sensor_scan.cloud_base->size();
    }
  }
  terrain_last_keyframe_retained_points_ = 0U;
  if (!terrain_config_.enabled || terrain_last_keyframe_input_points_ == 0U) {
    last_terrain_gravity_status_ = terrain_config_.enabled ?
      "terrain_cloud_unavailable" : "disabled";
    return;
  }

  std::optional<Eigen::Matrix3d> gravity_R_base;
  if (gravity_sample.has_value()) {
    gravity_R_base = gravityAlignmentRotation(gravity_sample->accel_base);
  }
  if (terrain_config_.require_gravity_alignment && !gravity_R_base.has_value()) {
    last_terrain_gravity_status_ = gravity_sample_stale ?
      "stale_at_keyframe_stamp" : "unavailable_at_keyframe_stamp";
    if (gravity_sample_stale) {
      ++terrain_gravity_stale_keyframe_count_;
    } else {
      ++terrain_gravity_unavailable_keyframe_count_;
    }
    return;
  }
  if (!gravity_R_base.has_value()) {
    gravity_R_base = Eigen::Matrix3d::Identity();
  }
  last_terrain_gravity_status_ = "timestamped_gravity_aligned";

  double seed_floor_sum = 0.0;
  std::size_t valid_seed_count = 0U;
  for (auto & sensor_scan : keyframe.terrain_sensor_scans) {
    if (!sensor_scan.cloud_base || sensor_scan.cloud_base->empty()) {
      continue;
    }
    std::vector<Eigen::Vector3d> points_gravity;
    points_gravity.reserve(sensor_scan.cloud_base->size());
    for (const auto & point : sensor_scan.cloud_base->points) {
      points_gravity.push_back(
        *gravity_R_base * Eigen::Vector3d(point.x, point.y, point.z));
    }
    const auto classification = classifyIndoorTerrain(
      points_gravity,
      *gravity_R_base * sensor_scan.sensor_origin_base,
      terrain_config_);
    last_terrain_classification_.evaluated =
      last_terrain_classification_.evaluated || classification.stats.evaluated;
    last_terrain_classification_.seed_valid =
      last_terrain_classification_.seed_valid || classification.stats.seed_valid;
    last_terrain_classification_.input_points += classification.stats.input_points;
    last_terrain_classification_.profile_bins += classification.stats.profile_bins;
    last_terrain_classification_.floor_points += classification.stats.floor_points;
    last_terrain_classification_.ramp_points += classification.stats.ramp_points;
    last_terrain_classification_.positive_step_points +=
      classification.stats.positive_step_points;
    last_terrain_classification_.observed_drop_points +=
      classification.stats.observed_drop_points;
    last_terrain_classification_.staircase_points +=
      classification.stats.staircase_points;
    last_terrain_classification_.roughness_preview_points +=
      classification.stats.roughness_preview_points;
    last_terrain_classification_.ramp_plane_candidate_points +=
      classification.stats.ramp_plane_candidate_points;
    last_terrain_classification_.ramp_plane_validated_points +=
      classification.stats.ramp_plane_validated_points;
    last_terrain_classification_.ramp_plane_rejected_points +=
      classification.stats.ramp_plane_rejected_points;
    last_terrain_classification_.ramp_plane_fits_attempted +=
      classification.stats.ramp_plane_fits_attempted;
    last_terrain_classification_.ramp_plane_fits_accepted +=
      classification.stats.ramp_plane_fits_accepted;
    last_terrain_classification_.ramp_plane_rejected_insufficient_points +=
      classification.stats.ramp_plane_rejected_insufficient_points;
    last_terrain_classification_.ramp_plane_rejected_plane_estimation +=
      classification.stats.ramp_plane_rejected_plane_estimation;
    last_terrain_classification_.ramp_plane_rejected_inlier_support +=
      classification.stats.ramp_plane_rejected_inlier_support;
    last_terrain_classification_.ramp_plane_rejected_length +=
      classification.stats.ramp_plane_rejected_length;
    last_terrain_classification_.ramp_plane_rejected_width +=
      classification.stats.ramp_plane_rejected_width;
    last_terrain_classification_.ramp_plane_rejected_slope +=
      classification.stats.ramp_plane_rejected_slope;
    last_terrain_classification_.ramp_plane_rejected_rmse +=
      classification.stats.ramp_plane_rejected_rmse;
    last_terrain_classification_.max_accepted_ramp_plane_rmse_m = std::max(
      last_terrain_classification_.max_accepted_ramp_plane_rmse_m,
      classification.stats.max_accepted_ramp_plane_rmse_m);
    last_terrain_classification_.max_roughness_score = std::max(
      last_terrain_classification_.max_roughness_score,
      classification.stats.max_roughness_score);
    if (classification.stats.seed_valid) {
      seed_floor_sum += classification.stats.seed_floor_z_m;
      ++valid_seed_count;
    }

    auto classified_cloud = std::make_shared<CloudT>();
    classified_cloud->reserve(sensor_scan.cloud_base->size());
    sensor_scan.labels.reserve(classification.labels.size());
    sensor_scan.roughness_scores.reserve(classification.roughness_scores.size());
    for (std::size_t index = 0U; index < classification.labels.size(); ++index) {
      if (classification.labels[index] == TerrainLabel::kUnknown) {
        continue;
      }
      classified_cloud->push_back(sensor_scan.cloud_base->points[index]);
      sensor_scan.labels.push_back(classification.labels[index]);
      sensor_scan.roughness_scores.push_back(classification.roughness_scores[index]);
    }
    classified_cloud->width = static_cast<std::uint32_t>(classified_cloud->size());
    classified_cloud->height = 1U;
    classified_cloud->is_dense = true;
    sensor_scan.cloud_base = classified_cloud;
    terrain_last_keyframe_retained_points_ += classified_cloud->size();
  }
  if (valid_seed_count > 0U) {
    last_terrain_classification_.seed_floor_z_m =
      seed_floor_sum / static_cast<double>(valid_seed_count);
  }
}

void ScanToMapIcp::appendKeyframeTerrainEvidence(const Keyframe & keyframe)
{
  if (!terrain_config_.enabled) {
    return;
  }

  std::vector<Eigen::Vector3d> points_map;
  std::vector<TerrainLabel> labels;
  std::vector<double> roughness_scores;
  for (const auto & sensor_scan : keyframe.terrain_sensor_scans) {
    if (!sensor_scan.cloud_base || sensor_scan.cloud_base->size() != sensor_scan.labels.size() ||
      sensor_scan.cloud_base->size() != sensor_scan.roughness_scores.size())
    {
      continue;
    }
    points_map.reserve(points_map.size() + sensor_scan.cloud_base->size());
    labels.reserve(labels.size() + sensor_scan.labels.size());
    roughness_scores.reserve(roughness_scores.size() + sensor_scan.roughness_scores.size());
    for (const auto & point : sensor_scan.cloud_base->points) {
      points_map.push_back(
        keyframe.map_T_base * Eigen::Vector3d(point.x, point.y, point.z));
    }
    labels.insert(labels.end(), sensor_scan.labels.begin(), sensor_scan.labels.end());
    roughness_scores.insert(
      roughness_scores.end(),
      sensor_scan.roughness_scores.begin(),
      sensor_scan.roughness_scores.end());
  }
  if (points_map.empty()) {
    return;
  }
  terrain_accumulator_.addObservation(
    points_map,
    labels,
    roughness_scores,
    keyframe.map_T_base.translation());
  terrain_preview_tracker_.updateGeometry(
    terrain_accumulator_.evidence(),
    keyframe.stamp.nanoseconds(),
    keyframe.map_T_base.translation(),
    keyframe.map_T_base.linear() * Eigen::Vector3d::UnitX());
  refreshTerrainEvidenceCloud();
  refreshTerrainPreviewCloud();
}

void ScanToMapIcp::rebuildTerrainEvidenceMap()
{
  terrain_accumulator_.clear();
  terrain_preview_tracker_.clear();
  if (!terrain_config_.enabled) {
    terrain_evidence_map_.reset(new TerrainCloudT);
    terrain_support_evidence_map_ = std::make_shared<TerrainSupportVector>();
    terrain_preview_evidence_map_.reset(new TerrainCloudT);
    return;
  }
  const auto append_geometry = [this](const Keyframe & keyframe) {
      std::vector<Eigen::Vector3d> points_map;
      std::vector<TerrainLabel> labels;
      std::vector<double> roughness_scores;
      for (const auto & sensor_scan : keyframe.terrain_sensor_scans) {
        if (!sensor_scan.cloud_base ||
          sensor_scan.cloud_base->size() != sensor_scan.labels.size() ||
          sensor_scan.cloud_base->size() != sensor_scan.roughness_scores.size())
        {
          continue;
        }
        points_map.reserve(points_map.size() + sensor_scan.cloud_base->size());
        labels.reserve(labels.size() + sensor_scan.labels.size());
        roughness_scores.reserve(roughness_scores.size() + sensor_scan.roughness_scores.size());
        for (const auto & point : sensor_scan.cloud_base->points) {
          points_map.push_back(
            keyframe.map_T_base * Eigen::Vector3d(point.x, point.y, point.z));
        }
        labels.insert(labels.end(), sensor_scan.labels.begin(), sensor_scan.labels.end());
        roughness_scores.insert(
          roughness_scores.end(),
          sensor_scan.roughness_scores.begin(),
          sensor_scan.roughness_scores.end());
      }
      if (points_map.empty()) {
        return;
      }
      terrain_accumulator_.addObservation(
        points_map,
        labels,
        roughness_scores,
        keyframe.map_T_base.translation());
      terrain_preview_tracker_.updateGeometry(
        terrain_accumulator_.evidence(),
        keyframe.stamp.nanoseconds(),
        keyframe.map_T_base.translation(),
        keyframe.map_T_base.linear() * Eigen::Vector3d::UnitX());
    };

  // Replay geometry and robot response in timestamp order. Replaying all
  // geometry first would let a feature be "confirmed" by a traversal that
  // happened before LiDAR had ever observed that feature.
  std::size_t keyframe_index = 0U;
  std::size_t sample_index = 0U;
  while (keyframe_index < keyframes_.size() ||
    sample_index < terrain_traversal_samples_.size())
  {
    const bool append_keyframe = keyframe_index < keyframes_.size() &&
      (sample_index >= terrain_traversal_samples_.size() ||
      keyframes_[keyframe_index].stamp <= terrain_traversal_samples_[sample_index].stamp);
    if (append_keyframe) {
      append_geometry(keyframes_[keyframe_index]);
      ++keyframe_index;
      continue;
    }
    const auto & anchored_sample = terrain_traversal_samples_[sample_index];
    ++sample_index;
    if (anchored_sample.anchor_keyframe_index >= keyframes_.size()) {
      continue;
    }
    const auto & anchor = keyframes_[anchored_sample.anchor_keyframe_index];
    (void)terrain_preview_tracker_.addTraversalSample(
      TerrainTraversalSample{
          anchored_sample.stamp.nanoseconds(),
          anchor.map_T_base * anchored_sample.position_anchor,
          anchored_sample.response,
          anchored_sample.roughness_score,
          anchored_sample.continuity_break_before});
  }
  refreshTerrainEvidenceCloud();
  refreshTerrainPreviewCloud();
}

void ScanToMapIcp::refreshTerrainEvidenceCloud()
{
  auto cloud = std::make_shared<TerrainCloudT>();
  auto support = std::make_shared<TerrainSupportVector>();
  if (terrain_config_.enabled) {
    const auto evidence = terrain_accumulator_.confirmedEvidence();
    cloud->reserve(evidence.size());
    for (const auto & item : evidence) {
      TerrainPointT point;
      point.x = static_cast<float>(item.position.x());
      point.y = static_cast<float>(item.position.y());
      point.z = static_cast<float>(item.position.z());
      point.intensity = terrainLabelValue(item.label);
      cloud->push_back(point);
    }
    *support = terrain_accumulator_.supportEvidence();
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  terrain_evidence_map_ = cloud;
  terrain_support_evidence_map_ = support;
}

void ScanToMapIcp::refreshTerrainPreviewCloud()
{
  auto cloud = std::make_shared<TerrainCloudT>();
  if (terrain_config_.enabled && terrain_preview_config_.enabled) {
    const auto evidence = terrain_preview_tracker_.previewEvidence();
    cloud->reserve(evidence.size());
    for (const auto & item : evidence) {
      TerrainPointT point;
      point.x = static_cast<float>(item.position.x());
      point.y = static_cast<float>(item.position.y());
      point.z = static_cast<float>(item.position.z());
      point.intensity = terrainPreviewValue(item.label, item.state);
      cloud->push_back(point);
    }
  }
  cloud->width = static_cast<std::uint32_t>(cloud->size());
  cloud->height = 1;
  cloud->is_dense = true;
  terrain_preview_evidence_map_ = cloud;
}

void ScanToMapIcp::recordTerrainTraversalSample(
  const rclcpp::Time & stamp,
  const RegistrationResult & result)
{
  if (!terrain_config_.enabled || !terrain_preview_config_.enabled ||
    keyframes_.empty() ||
    stamp.nanoseconds() <= 0)
  {
    return;
  }
  if (result.rejection_reason == "primary_cloud_timestamp_not_increasing") {
    return;
  }
  // A duplicate or late diagnostic must not mutate the current causal bridge.
  // In particular, an old rejected scan cannot break a newer valid passage.
  if (!terrain_traversal_samples_.empty() &&
    stamp <= terrain_traversal_samples_.back().stamp)
  {
    return;
  }
  // Never bind an impact to the pose estimated during the vibration itself.
  // The pending event is attached only after a stable accepted pose closes the
  // swept segment from the last pre-impact sample.
  if (result.imu_roughness_event || result.imu_shock_active ||
    result.shock_quarantined || result.pose_quarantined || !result.accepted)
  {
    if (terrain_pending_shock_event_ && (result.pose_quarantined || !result.accepted)) {
      terrain_pending_shock_bridge_invalidated_ = true;
    }
    terrain_traversal_continuity_break_pending_ = true;
    return;
  }
  if (terrain_pending_shock_event_ && !result.imu_roughness_stable_evidence) {
    return;
  }
  TerrainTraversalResponse response = result.imu_roughness_stable_evidence ?
    TerrainTraversalResponse::kStable : TerrainTraversalResponse::kUnknown;
  const auto normalized_ratio = [](const double value, const double threshold) {
      if (!std::isfinite(value) || !std::isfinite(threshold) || threshold <= 0.0) {
        return 0.0;
      }
      return std::max(0.0, value / threshold);
    };
  const double roughness_score = std::max(
      {
        normalized_ratio(
          result.imu_roughness_max_accel_norm_delta_mps2,
          imu_roughness_accel_norm_delta_threshold_mps2_),
        normalized_ratio(
          result.imu_roughness_max_linear_jerk_mps3,
          imu_roughness_linear_jerk_threshold_mps3_)});

  if (terrain_pending_shock_event_) {
    bool localized =
      !terrain_pending_shock_bridge_invalidated_ &&
      !terrain_traversal_samples_.empty();
    const AnchoredTerrainTraversalSample * previous_sample = localized ?
      &terrain_traversal_samples_.back() : nullptr;
    localized = localized && previous_sample->response == TerrainTraversalResponse::kStable &&
      previous_sample->stamp == terrain_pending_shock_pre_sample_stamp_ &&
      previous_sample->stamp < terrain_pending_shock_stamp_ &&
      terrain_pending_shock_stamp_ <= stamp;

    std::optional<Eigen::Vector3d> previous_position;
    if (localized) {
      previous_position = anchoredTerrainTraversalPosition(*previous_sample);
      localized = previous_position.has_value();
    }
    double segment_duration_s = 0.0;
    if (localized) {
      segment_duration_s = (stamp - previous_sample->stamp).seconds();
      const double segment_length_m =
        (map_T_base_.translation() - *previous_position).norm();
      localized = segment_duration_s > 0.0 &&
        (terrain_preview_config_.max_sample_gap_s <= 0.0 ||
        segment_duration_s <= terrain_preview_config_.max_sample_gap_s) &&
        (terrain_preview_config_.max_swept_segment_length_m <= 0.0 ||
        segment_length_m <= terrain_preview_config_.max_swept_segment_length_m);
    }

    if (localized) {
      const double event_ratio = std::clamp(
        (terrain_pending_shock_stamp_ - previous_sample->stamp).seconds() /
        segment_duration_s,
        0.0,
        1.0);
      const Eigen::Vector3d event_position =
        *previous_position +
        event_ratio * (map_T_base_.translation() - *previous_position);
      const bool event_recorded = appendTerrainTraversalSample(
        terrain_pending_shock_stamp_,
        event_position,
        TerrainTraversalResponse::kShock,
        std::max(roughness_score, terrain_pending_shock_score_));
      if (event_recorded) {
        terrain_traversal_continuity_break_pending_ = false;
      }
      clearPendingTerrainShock(!event_recorded);
    } else {
      clearPendingTerrainShock(true);
    }
  }

  const bool sample_recorded = appendTerrainTraversalSample(
    stamp,
    map_T_base_.translation(),
    response,
    roughness_score,
    terrain_traversal_continuity_break_pending_);
  if (sample_recorded) {
    terrain_traversal_continuity_break_pending_ = false;
  }
}

bool ScanToMapIcp::appendTerrainTraversalSample(
  const rclcpp::Time & stamp,
  const Eigen::Vector3d & position_map,
  const TerrainTraversalResponse response,
  const double roughness_score,
  const bool continuity_break_before)
{
  if (keyframes_.empty() || stamp.nanoseconds() <= 0 || !position_map.allFinite() ||
    !std::isfinite(roughness_score) || roughness_score < 0.0 ||
    (!terrain_traversal_samples_.empty() && stamp <= terrain_traversal_samples_.back().stamp))
  {
    return false;
  }

  const std::size_t anchor_index = keyframes_.size() - 1U;
  const Eigen::Vector3d position_anchor =
    keyframes_[anchor_index].map_T_base.inverse() * position_map;
  terrain_traversal_samples_.push_back(
    AnchoredTerrainTraversalSample{
        stamp,
        anchor_index,
        position_anchor,
        response,
        roughness_score,
        continuity_break_before});
  const TerrainTraversalUpdate update = terrain_preview_tracker_.addTraversalSample(
    TerrainTraversalSample{
        stamp.nanoseconds(),
        position_map,
        response,
        roughness_score,
        continuity_break_before});
  if (!update.accepted) {
    terrain_traversal_samples_.pop_back();
    return false;
  }
  if (update.evidence_changed) {
    refreshTerrainPreviewCloud();
  }
  return true;
}

std::optional<Eigen::Vector3d> ScanToMapIcp::anchoredTerrainTraversalPosition(
  const AnchoredTerrainTraversalSample & sample,
  const std::vector<Eigen::Isometry3d> * keyframe_poses) const
{
  const std::size_t pose_count = keyframe_poses == nullptr ?
    keyframes_.size() : keyframe_poses->size();
  if (sample.anchor_keyframe_index >= pose_count || !sample.position_anchor.allFinite()) {
    return std::nullopt;
  }
  const Eigen::Isometry3d & map_T_anchor = keyframe_poses == nullptr ?
    keyframes_[sample.anchor_keyframe_index].map_T_base :
    (*keyframe_poses)[sample.anchor_keyframe_index];
  const Eigen::Vector3d position_map = map_T_anchor * sample.position_anchor;
  return position_map.allFinite() ? std::optional<Eigen::Vector3d>(position_map) : std::nullopt;
}

void ScanToMapIcp::clearPendingTerrainShock(const bool count_unlocalized)
{
  if (terrain_pending_shock_event_ && count_unlocalized) {
    ++terrain_unlocalized_shock_event_count_;
  }
  terrain_pending_shock_event_ = false;
  terrain_pending_shock_bridge_invalidated_ = false;
  terrain_pending_shock_score_ = 0.0;
  terrain_pending_shock_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  terrain_pending_shock_pre_sample_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void ScanToMapIcp::updateTerrainTraversalSamplesFromKeyframeCorrections(
  const std::vector<Eigen::Isometry3d> & previous_keyframe_poses)
{
  if (previous_keyframe_poses.size() != keyframes_.size() || keyframes_.empty()) {
    return;
  }

  std::vector<Eigen::Isometry3d> corrections;
  corrections.reserve(keyframes_.size());
  for (std::size_t index = 0U; index < keyframes_.size(); ++index) {
    corrections.push_back(
      projectToRigidIsometry(
        keyframes_[index].map_T_base * previous_keyframe_poses[index].inverse()));
  }

  for (auto & sample : terrain_traversal_samples_) {
    const auto previous_position = anchoredTerrainTraversalPosition(
      sample,
      &previous_keyframe_poses);
    if (!previous_position.has_value()) {
      continue;
    }

    std::size_t right_index = 0U;
    while (right_index < keyframes_.size() &&
      keyframes_[right_index].stamp < sample.stamp)
    {
      ++right_index;
    }

    Eigen::Isometry3d correction = corrections.back();
    std::size_t new_anchor_index = keyframes_.size() - 1U;
    if (right_index == 0U) {
      correction = corrections.front();
      new_anchor_index = 0U;
    } else if (right_index < keyframes_.size()) {
      const std::size_t left_index = right_index - 1U;
      const double interval_s =
        (keyframes_[right_index].stamp - keyframes_[left_index].stamp).seconds();
      const double ratio = interval_s > 1.0e-6 ?
        std::clamp(
        (sample.stamp - keyframes_[left_index].stamp).seconds() / interval_s,
        0.0,
        1.0) : 0.0;
      const Eigen::Isometry3d relative_correction = projectToRigidIsometry(
        corrections[right_index] * corrections[left_index].inverse());
      correction = projectToRigidIsometry(
        interpolateTransform(relative_correction, ratio, ratio) * corrections[left_index]);
      new_anchor_index = ratio < 0.5 ? left_index : right_index;
    }

    const Eigen::Vector3d corrected_position = correction * *previous_position;
    if (!corrected_position.allFinite()) {
      continue;
    }
    sample.anchor_keyframe_index = new_anchor_index;
    sample.position_anchor =
      keyframes_[new_anchor_index].map_T_base.inverse() * corrected_position;
  }
}

void ScanToMapIcp::pruneTerrainTraversalSamplesAfterOldestKeyframeRemoval()
{
  std::vector<AnchoredTerrainTraversalSample> retained;
  retained.reserve(terrain_traversal_samples_.size());
  for (auto sample : terrain_traversal_samples_) {
    if (sample.anchor_keyframe_index == 0U) {
      continue;
    }
    --sample.anchor_keyframe_index;
    retained.push_back(std::move(sample));
  }
  terrain_traversal_samples_ = std::move(retained);
}

void ScanToMapIcp::refreshMapPublishSnapshots()
{
  std::lock_guard<std::mutex> lock(map_publish_snapshot_mutex_);
  local_map_publish_snapshot_ = latest_local_map_;
  global_map_publish_snapshot_ = global_map_;
  terrain_publish_snapshot_ = terrain_evidence_map_;
  terrain_support_publish_snapshot_ = terrain_support_evidence_map_;
  terrain_preview_publish_snapshot_ = terrain_preview_evidence_map_;
  ++map_publish_snapshot_update_count_;
}

bool ScanToMapIcp::shouldAddKeyframe(const Eigen::Isometry3d & pose) const
{
  return shouldAddSpatialKeyframe(pose, last_keyframe_pose_, spatial_mapping_config_);
}

std::size_t ScanToMapIcp::submapAnchorKeyframeIndex(const Submap & submap) const
{
  if (submap.frozen_geometry_valid) {
    return submap.frozen_anchor_keyframe_index;
  }
  if (submap.keyframe_indices.empty()) {
    return keyframes_.size();
  }
  return submap.keyframe_indices[submap.keyframe_indices.size() / 2U];
}

bool ScanToMapIcp::hasCompatibleSubmap(const Eigen::Isometry3d & center_pose) const
{
  if (!submap_enabled_) {
    return true;
  }
  return std::any_of(
    submaps_.begin(),
    submaps_.end(),
    [this, &center_pose](const Submap & submap) {
      if (!submap.cloud_map || submap.cloud_map->empty() ||
      submap.keyframe_indices.empty())
      {
        return false;
      }
      const std::size_t anchor_index = submapAnchorKeyframeIndex(submap);
      return anchor_index < keyframes_.size() &&
      poseWithinLocalMapWindow(
        keyframes_[anchor_index].map_T_base,
        center_pose,
        spatial_mapping_config_);
    });
}

void ScanToMapIcp::closeActiveSubmapForSpatialTransition()
{
  if (!submap_enabled_ || submaps_.empty()) {
    return;
  }
  auto & active_submap = submaps_.back();
  if (active_submap.completed || active_submap.keyframe_indices.empty()) {
    return;
  }
  if (submaps_.size() >= 2U) {
    const auto & previous_submap = submaps_[submaps_.size() - 2U];
    const bool overlap_only = previous_submap.completed &&
      std::all_of(
      active_submap.keyframe_indices.begin(),
      active_submap.keyframe_indices.end(),
      [&previous_submap](const std::size_t index) {
        return std::find(
          previous_submap.keyframe_indices.begin(),
          previous_submap.keyframe_indices.end(),
          index) != previous_submap.keyframe_indices.end();
      });
    if (overlap_only) {
      active_submap.keyframe_indices.clear();
      active_submap.cloud_map = std::make_shared<CloudT>();
      active_submap.mapping_cloud_map = std::make_shared<CloudT>();
      active_submap.descriptor.clear();
      active_submap.accumulated_travel_m = 0.0;
      return;
    }
  }
  active_submap.completed = true;
  active_submap.closure_reason = SubmapClosureReason::kSpatialTransition;
  refreshSubmap(submaps_.size() - 1U);

  Submap next_submap;
  next_submap.id = next_submap_id_++;
  next_submap.cloud_map = std::make_shared<CloudT>();
  next_submap.mapping_cloud_map = std::make_shared<CloudT>();
  submaps_.push_back(std::move(next_submap));
}

void ScanToMapIcp::appendLatestKeyframeToSubmaps()
{
  if (!submap_enabled_ || keyframes_.empty()) {
    return;
  }

  if (submaps_.empty() || submaps_.back().completed) {
    Submap submap;
    submap.id = next_submap_id_++;
    submap.cloud_map = std::make_shared<CloudT>();
    submap.mapping_cloud_map = std::make_shared<CloudT>();
    submaps_.push_back(std::move(submap));
  }

  const std::size_t latest_index = keyframes_.size() - 1U;
  auto & active_submap = submaps_.back();
  if (active_submap.keyframe_indices.empty() ||
    active_submap.keyframe_indices.back() != latest_index)
  {
    active_submap.keyframe_indices.push_back(latest_index);
  }
  refreshSubmap(submaps_.size() - 1U);

  if (active_submap.keyframe_indices.size() < 2U) {
    return;
  }
  const SubmapClosureReason closure_reason = submapClosureReason(
    active_submap.keyframe_indices.size(),
    active_submap.accumulated_travel_m,
    submap_policy_config_);
  if (closure_reason == SubmapClosureReason::kNone) {
    return;
  }

  active_submap.completed = true;
  active_submap.closure_reason = closure_reason;
  refreshSubmap(submaps_.size() - 1U);

  Submap next_submap;
  next_submap.id = next_submap_id_++;
  next_submap.cloud_map = std::make_shared<CloudT>();
  next_submap.mapping_cloud_map = std::make_shared<CloudT>();
  const std::size_t overlap_start = submapOverlapStartIndex(
    active_submap.keyframe_indices.size(),
    submap_overlap_keyframes_);
  next_submap.keyframe_indices.insert(
    next_submap.keyframe_indices.end(),
    active_submap.keyframe_indices.begin() + static_cast<std::ptrdiff_t>(overlap_start),
    active_submap.keyframe_indices.end());
  submaps_.push_back(std::move(next_submap));
  refreshSubmap(submaps_.size() - 1U);
}

ScanToMapIcp::CloudPtr ScanToMapIcp::buildConfirmedSubmapMappingCloudAnchor(
  const Submap & submap,
  const std::size_t anchor_index) const
{
  auto confirmed = std::make_shared<CloudT>();
  if (anchor_index >= keyframes_.size()) {
    return confirmed;
  }

  const double voxel_size = std::max(
    1.0e-3,
    place_recognition_mapping_refinement_target_confirmation_voxel_m_);
  const int minimum_observations = std::max(
    1,
    place_recognition_mapping_refinement_min_target_keyframe_observations_);
  using VoxelMap = std::unordered_map<
    ConfirmedVoxelKey,
    ConfirmedVoxelAccumulator,
    ConfirmedVoxelKeyHash>;
  VoxelMap aggregate;
  const Eigen::Isometry3d anchor_T_map =
    keyframes_[anchor_index].map_T_base.inverse();

  for (const std::size_t keyframe_index : submap.keyframe_indices) {
    if (keyframe_index >= keyframes_.size()) {
      continue;
    }
    const auto & keyframe = keyframes_[keyframe_index];
    if (!keyframe.mapping_cloud_base || keyframe.mapping_cloud_base->empty()) {
      continue;
    }

    VoxelMap keyframe_voxels;
    const Eigen::Isometry3d anchor_T_base =
      anchor_T_map * keyframe.map_T_base;
    for (const auto & point : keyframe.mapping_cloud_base->points) {
      const Eigen::Vector3d anchor_point =
        anchor_T_base * Eigen::Vector3d(point.x, point.y, point.z);
      if (!anchor_point.allFinite()) {
        continue;
      }
      auto & accumulator = keyframe_voxels[
        confirmedVoxelKey(anchor_point, voxel_size)];
      accumulator.sum += anchor_point;
      ++accumulator.point_count;
    }

    for (const auto & item : keyframe_voxels) {
      if (item.second.point_count <= 0) {
        continue;
      }
      auto & accumulator = aggregate[item.first];
      accumulator.sum += item.second.sum /
        static_cast<double>(item.second.point_count);
      ++accumulator.keyframe_observations;
    }
  }

  confirmed->reserve(aggregate.size());
  for (const auto & item : aggregate) {
    const auto & accumulator = item.second;
    if (accumulator.keyframe_observations < minimum_observations) {
      continue;
    }
    const Eigen::Vector3d centroid = accumulator.sum /
      static_cast<double>(accumulator.keyframe_observations);
    confirmed->push_back(
      PointT(
        static_cast<float>(centroid.x()),
        static_cast<float>(centroid.y()),
        static_cast<float>(centroid.z())));
  }
  confirmed->width = static_cast<std::uint32_t>(confirmed->size());
  confirmed->height = 1U;
  confirmed->is_dense = true;
  return confirmed;
}

void ScanToMapIcp::refreshSubmap(const std::size_t submap_index)
{
  if (submap_index >= submaps_.size()) {
    return;
  }
  auto & submap = submaps_[submap_index];
  if (submap.completed && submap.frozen_geometry_valid) {
    const std::size_t anchor_index = submap.frozen_anchor_keyframe_index;
    if (anchor_index >= keyframes_.size()) {
      submap.cloud_map = std::make_shared<CloudT>();
      submap.mapping_cloud_map = std::make_shared<CloudT>();
      submap.descriptor.clear();
      return;
    }
    const Eigen::Isometry3d & map_T_anchor = keyframes_[anchor_index].map_T_base;
    submap.cloud_map = filterMapCloud(
      transformCloud(submap.frozen_cloud_anchor, map_T_anchor));
    submap.mapping_cloud_map = filterMapCloud(
      transformCloud(submap.frozen_mapping_cloud_anchor, map_T_anchor));
    submap.descriptor = keyframes_[anchor_index].descriptor;
    return;
  }

  auto merged = std::make_shared<CloudT>();
  auto mapping_merged = std::make_shared<CloudT>();
  std::vector<Eigen::Isometry3d> submap_poses;
  submap_poses.reserve(submap.keyframe_indices.size());
  for (const std::size_t keyframe_index : submap.keyframe_indices) {
    if (keyframe_index >= keyframes_.size()) {
      continue;
    }
    submap_poses.push_back(keyframes_[keyframe_index].map_T_base);
    if (!keyframes_[keyframe_index].cloud_map ||
      keyframes_[keyframe_index].cloud_map->empty() ||
      !keyframes_[keyframe_index].mapping_cloud_map)
    {
      refreshKeyframeCloudMap(keyframe_index);
    }
    const auto & cloud_map = keyframes_[keyframe_index].cloud_map;
    if (cloud_map && !cloud_map->empty()) {
      *merged += *cloud_map;
    }
    const auto & mapping_cloud_map = keyframes_[keyframe_index].mapping_cloud_map;
    if (mapping_cloud_map && !mapping_cloud_map->empty()) {
      *mapping_merged += *mapping_cloud_map;
    }
  }
  submap.accumulated_travel_m = accumulatedSpatialTravel(
    submap_poses,
    spatial_mapping_config_.mode);
  submap.cloud_map = downsampleCloud(merged, map_voxel_leaf_m_);
  submap.mapping_cloud_map = downsampleCloud(mapping_merged, map_voxel_leaf_m_);
  submap.descriptor.clear();
  const std::size_t anchor_index = submapAnchorKeyframeIndex(submap);
  if (anchor_index >= keyframes_.size() || !submap.cloud_map || submap.cloud_map->empty()) {
    return;
  }
  // Retrieval compares single-scan descriptors. The full submap cloud remains
  // the geometric verification target after a historical keyframe is found.
  submap.descriptor = keyframes_[anchor_index].descriptor;
  if (submap.completed) {
    const Eigen::Isometry3d anchor_T_map =
      keyframes_[anchor_index].map_T_base.inverse();
    submap.frozen_cloud_anchor = transformCloud(submap.cloud_map, anchor_T_map);
    submap.frozen_mapping_input_points = submap.mapping_cloud_map ?
      static_cast<int>(submap.mapping_cloud_map->size()) : 0;
    if (place_recognition_mapping_cloud_refinement_enabled_) {
      submap.frozen_mapping_cloud_anchor =
        buildConfirmedSubmapMappingCloudAnchor(submap, anchor_index);
    } else {
      submap.frozen_mapping_cloud_anchor = transformCloud(
        submap.mapping_cloud_map, anchor_T_map);
    }
    submap.frozen_mapping_confirmed_points =
      submap.frozen_mapping_cloud_anchor ?
      static_cast<int>(submap.frozen_mapping_cloud_anchor->size()) : 0;
    submap.mapping_cloud_map = filterMapCloud(
      transformCloud(
        submap.frozen_mapping_cloud_anchor,
        keyframes_[anchor_index].map_T_base));
    submap.frozen_anchor_keyframe_index = anchor_index;
    submap.frozen_geometry_valid = true;
  }
}

void ScanToMapIcp::rebuildSubmaps()
{
  if (!submap_enabled_) {
    return;
  }
  for (std::size_t index = 0; index < submaps_.size(); ++index) {
    refreshSubmap(index);
  }
}

void ScanToMapIcp::pruneSubmapsAfterOldestKeyframeRemoval()
{
  if (!submap_enabled_) {
    return;
  }
  for (auto & submap : submaps_) {
    if (submap.frozen_geometry_valid) {
      if (submap.frozen_anchor_keyframe_index == 0U) {
        // The frozen cloud is expressed in the removed anchor frame. Drop the
        // completed target instead of silently re-anchoring uncertain geometry.
        submap.keyframe_indices.clear();
        continue;
      }
      --submap.frozen_anchor_keyframe_index;
    }
    std::vector<std::size_t> retained_indices;
    retained_indices.reserve(submap.keyframe_indices.size());
    for (const std::size_t index : submap.keyframe_indices) {
      if (index > 0U) {
        retained_indices.push_back(index - 1U);
      }
    }
    submap.keyframe_indices = std::move(retained_indices);
  }
  submaps_.erase(
    std::remove_if(
      submaps_.begin(),
      submaps_.end(),
      [](const Submap & submap) {
        return submap.completed && submap.keyframe_indices.empty();
      }),
    submaps_.end());
  rebuildSubmaps();
}

int ScanToMapIcp::completedSubmapCount() const
{
  return static_cast<int>(std::count_if(
           submaps_.begin(),
           submaps_.end(),
           [](const Submap & submap) {return submap.completed;}));
}

int ScanToMapIcp::frozenCompletedSubmapCount() const
{
  return static_cast<int>(std::count_if(
           submaps_.begin(),
           submaps_.end(),
           [](const Submap & submap) {
             return submap.completed && submap.frozen_geometry_valid;
           }));
}

std::string ScanToMapIcp::shockQuarantineReason(const RegistrationResult & result) const
{
  if (!quarantine_keyframes_on_shock_) {
    return "";
  }

  if (result.imu_shock_active) {
    return "imu_" + result.imu_shock_phase;
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

    const double max_alignment_translation =
      effectivePlaceRecognitionAlignmentCorrectionLimit(
      keyframe_quarantine_correction_translation_m_,
      place_recognition_registration_only_keyframe_quarantine_correction_translation_m_,
      place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m_,
      result.place_recognition_registration_only_vertical_recovery_used,
      result.place_recognition_mapping_refinement_verified);
    if (max_alignment_translation > 0.0 &&
      result.place_recognition_alignment_correction_translation_m >
      max_alignment_translation)
    {
      return "place_alignment_correction_translation";
    }
    const double max_alignment_rotation =
      effectivePlaceRecognitionAlignmentCorrectionLimit(
      keyframe_quarantine_correction_rotation_deg_,
      0.0,
      place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg_,
      false,
      result.place_recognition_mapping_refinement_verified);
    if (max_alignment_rotation > 0.0 &&
      result.place_recognition_alignment_correction_rotation_deg >
      max_alignment_rotation)
    {
      return "place_alignment_correction_rotation";
    }

    double max_prediction_translation =
      effectivePlaceRecognitionTranslationJumpLimit(
      place_recognition_max_pose_jump_m_,
      place_recognition_registration_only_max_pose_jump_m_,
      result.place_recognition_registration_only_vertical_recovery_used);
    double max_prediction_rotation = std::max(
      0.0, place_recognition_max_pose_jump_rotation_deg_);
    if (result.place_recognition_large_pose_jump_accepted) {
      max_prediction_translation =
        place_recognition_mapping_refinement_large_pose_jump_max_translation_m_;
      max_prediction_rotation =
        place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_;
    }
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
  const int sectors = std::max(8, place_recognition_descriptor_bins_);
  const ScanContextMatch match = matchScanContextDescriptors(lhs, rhs, sectors);
  best_shift = match.best_shift;
  return match.distance;
}

bool ScanToMapIcp::tryPlaceRecognition(
  const CloudConstPtr & scan_base,
  const CloudConstPtr & mapping_scan_base,
  const std::vector<float> & scan_descriptor,
  const Eigen::Isometry3d & reference_pose,
  const rclcpp::Time & stamp,
  const int registration_only_support_points,
  RegistrationResult & result)
{
  result.place_recognition_attempted = true;
  result.place_recognition_candidate_index = -1;
  result.place_recognition_candidate_submap_id = -1;
  result.place_recognition_candidate_consistency_count = 0;
  result.place_recognition_descriptor_distance = std::numeric_limits<double>::infinity();
  result.place_recognition_nearest_descriptor_distance =
    std::numeric_limits<double>::infinity();
  result.place_recognition_yaw_offset_deg = 0.0;
  result.place_recognition_candidate_consistency_translation_m = 0.0;
  result.place_recognition_candidate_consistency_rotation_deg = 0.0;
  result.place_recognition_registration_only_vertical_recovery_used = false;
  result.place_recognition_registration_only_consistency_relaxation_used = false;
  result.place_recognition_registration_only_vertical_recovery_descriptors = 0;
  result.place_recognition_mapping_refinement_status =
    place_recognition_mapping_cloud_refinement_enabled_ ? "not_reached" : "disabled";
  result.place_recognition_candidate_registration_status = "not_evaluated";
  auto reset_submap_consistency = [this]() {
      last_place_candidate_submap_id_ = -1;
      consecutive_place_candidate_submap_matches_ = 0;
      have_last_place_candidate_correction_ = false;
      last_place_candidate_used_registration_only_vertical_recovery_ = false;
      last_place_candidate_stamp_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      last_place_candidate_correction_ = Eigen::Isometry3d::Identity();
    };
  const auto preserve_recovery_consistency = [this, &stamp]() {
      if (!last_place_candidate_used_registration_only_vertical_recovery_ ||
        !have_last_place_candidate_correction_ ||
        place_recognition_registration_only_consistency_max_gap_s_ <= 0.0)
      {
        return false;
      }
      const double elapsed_s = (stamp - last_place_candidate_stamp_).seconds();
      return std::isfinite(elapsed_s) && elapsed_s >= 0.0 &&
             elapsed_s <= place_recognition_registration_only_consistency_max_gap_s_;
    };

  if (!place_recognition_enabled_ || scan_descriptor.empty() ||
    static_cast<int>(keyframes_.size()) < place_recognition_min_keyframes_)
  {
    return false;
  }

  struct Candidate
  {
    int keyframe_index{-1};
    int submap_index{-1};
    int submap_id{-1};
    int shift{0};
    double distance{std::numeric_limits<double>::infinity()};
    bool registration_only_vertical_recovery{false};
  };

  struct RejectedCandidateDiagnostic
  {
    bool available{false};
    Candidate candidate;
    double score{std::numeric_limits<double>::infinity()};
    double yaw_offset_deg{0.0};
    double inlier_ratio{0.0};
    double inlier_rmse_m{0.0};
    double pose_jump_translation_m{0.0};
    double pose_jump_rotation_deg{0.0};
    std::string status{"not_evaluated"};
  };

  std::vector<Candidate> candidates;
  Candidate nearest_candidate;
  bool have_nearest_candidate = false;
  auto record_nearest = [&nearest_candidate, &have_nearest_candidate](
    const Candidate & candidate) {
      if (!have_nearest_candidate || candidate.distance < nearest_candidate.distance) {
        nearest_candidate = candidate;
        have_nearest_candidate = true;
      }
    };
  const int latest_candidate_exclusive = std::max(
    0,
    static_cast<int>(keyframes_.size()) -
    std::max(0, place_recognition_exclude_recent_keyframes_));
  const double max_vertical_separation =
    std::max(0.0, place_recognition_max_vertical_separation_m_);
  if (place_recognition_use_submaps_) {
    for (std::size_t submap_index = 0; submap_index < submaps_.size(); ++submap_index) {
      const auto & submap = submaps_[submap_index];
      if (!submap.completed || submap.keyframe_indices.empty()) {
        continue;
      }
      if (static_cast<int>(submap.keyframe_indices.back()) >= latest_candidate_exclusive) {
        continue;
      }
      const std::size_t anchor_index = submapAnchorKeyframeIndex(submap);
      if (anchor_index >= keyframes_.size()) {
        continue;
      }
      const bool submap_has_registration_only_support = std::any_of(
        submap.keyframe_indices.begin(),
        submap.keyframe_indices.end(),
        [this, latest_candidate_exclusive](const std::size_t keyframe_index) {
          return keyframe_index < keyframes_.size() &&
          static_cast<int>(keyframe_index) < latest_candidate_exclusive &&
          keyframes_[keyframe_index].registration_only_support_points > 0;
        });
      const double vertical_separation = std::abs(
        keyframes_[anchor_index].map_T_base.translation().z() -
        reference_pose.translation().z());
      const PlaceRecognitionVerticalGateDecision vertical_gate =
        evaluatePlaceRecognitionVerticalGate(
        vertical_separation,
        max_vertical_separation,
        place_recognition_registration_only_vertical_recovery_m_,
        registration_only_support_points > 0,
        submap_has_registration_only_support);
      if (!vertical_gate.eligible) {
        continue;
      }
      Candidate best_submap_candidate;
      bool have_submap_candidate = false;
      for (const std::size_t keyframe_index : submap.keyframe_indices) {
        if (keyframe_index >= keyframes_.size() ||
          static_cast<int>(keyframe_index) >= latest_candidate_exclusive ||
          keyframes_[keyframe_index].descriptor.empty())
        {
          continue;
        }
        ++result.place_recognition_eligible_descriptors;
        if (vertical_gate.used_registration_only_recovery) {
          ++result.place_recognition_registration_only_vertical_recovery_descriptors;
        }
        int shift = 0;
        const double distance = descriptorDistance(
          scan_descriptor,
          keyframes_[keyframe_index].descriptor,
          shift);
        if (!std::isfinite(distance)) {
          continue;
        }
        const Candidate candidate{
          static_cast<int>(keyframe_index),
          static_cast<int>(submap_index),
          submap.id,
          shift,
          distance,
          vertical_gate.used_registration_only_recovery};
        record_nearest(candidate);
        if (!have_submap_candidate || distance < best_submap_candidate.distance) {
          best_submap_candidate = candidate;
          have_submap_candidate = true;
        }
      }
      if (!have_submap_candidate ||
        best_submap_candidate.distance > place_recognition_max_descriptor_distance_)
      {
        continue;
      }
      candidates.push_back(best_submap_candidate);
    }
  } else {
    for (int index = 0; index < latest_candidate_exclusive; ++index) {
      const auto & keyframe = keyframes_[static_cast<std::size_t>(index)];
      const PlaceRecognitionVerticalGateDecision vertical_gate =
        evaluatePlaceRecognitionVerticalGate(
        std::abs(
          keyframe.map_T_base.translation().z() -
          reference_pose.translation().z()),
        max_vertical_separation,
        place_recognition_registration_only_vertical_recovery_m_,
        registration_only_support_points > 0,
        keyframe.registration_only_support_points > 0);
      if (!vertical_gate.eligible) {
        continue;
      }
      ++result.place_recognition_eligible_descriptors;
      if (vertical_gate.used_registration_only_recovery) {
        ++result.place_recognition_registration_only_vertical_recovery_descriptors;
      }
      int shift = 0;
      const double distance = descriptorDistance(
        scan_descriptor,
        keyframes_[static_cast<std::size_t>(index)].descriptor,
        shift);
      if (!std::isfinite(distance)) {
        continue;
      }
      const Candidate candidate{
        index,
        -1,
        -1,
        shift,
        distance,
        vertical_gate.used_registration_only_recovery};
      record_nearest(candidate);
      if (distance > place_recognition_max_descriptor_distance_) {
        continue;
      }
      candidates.push_back(candidate);
    }
  }

  if (have_nearest_candidate) {
    result.place_recognition_nearest_descriptor_distance = nearest_candidate.distance;
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
    if (have_nearest_candidate) {
      result.place_recognition_candidate_index = nearest_candidate.keyframe_index;
      result.place_recognition_candidate_submap_id = nearest_candidate.submap_id;
      result.place_recognition_descriptor_distance = nearest_candidate.distance;
      const double yaw_bin_width = 2.0 * M_PI /
        static_cast<double>(std::max(8, place_recognition_descriptor_bins_));
      result.place_recognition_yaw_offset_deg = normalizedAngle(
        -static_cast<double>(nearest_candidate.shift) * yaw_bin_width) * 180.0 / M_PI;
    }
    if (place_recognition_use_submaps_ && !preserve_recovery_consistency()) {
      reset_submap_consistency();
    }
    return false;
  }

  RegistrationResult best_result;
  RejectedCandidateDiagnostic best_rejected;
  double best_score = std::numeric_limits<double>::infinity();
  bool have_match = false;
  CloudPtr best_candidate_map;
  Eigen::Isometry3d best_candidate_seed_pose{Eigen::Isometry3d::Identity()};
  int best_candidate_submap_index = -1;

  const double yaw_bin_width = 2.0 * M_PI /
    static_cast<double>(std::max(8, place_recognition_descriptor_bins_));
  for (const auto & candidate : candidates) {
    const auto candidate_map = candidate.submap_index >= 0 ?
      buildCandidateMapAroundSubmap(static_cast<std::size_t>(candidate.submap_index)) :
      buildCandidateMapAroundKeyframe(candidate.keyframe_index);
    if (!candidate_map || static_cast<int>(candidate_map->size()) < min_scan_points_) {
      continue;
    }

    Eigen::Isometry3d candidate_pose =
      keyframes_[static_cast<std::size_t>(candidate.keyframe_index)].map_T_base;
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
    const Eigen::Isometry3d pose_jump =
      projectToRigidIsometry(reference_pose.inverse() * candidate_result.map_T_base);
    const double pose_jump_translation_m = pose_jump.translation().norm();
    const double pose_jump_rotation_deg = rotationAngleDeg(pose_jump.linear());
    const double score = candidate_result.fitness +
      (1.0 - candidate_result.inlier_ratio) +
      0.1 * candidate.distance;
    auto record_rejection = [&](const std::string & status) {
        if (best_rejected.available && score >= best_rejected.score) {
          return;
        }
        best_rejected.available = true;
        best_rejected.candidate = candidate;
        best_rejected.score = score;
        best_rejected.yaw_offset_deg = yaw_offset * 180.0 / M_PI;
        best_rejected.inlier_ratio = candidate_result.inlier_ratio;
        best_rejected.inlier_rmse_m = candidate_result.inlier_rmse_m;
        best_rejected.pose_jump_translation_m = pose_jump_translation_m;
        best_rejected.pose_jump_rotation_deg = pose_jump_rotation_deg;
        best_rejected.status = status;
      };
    if (candidate_result.inlier_ratio < place_recognition_min_inlier_ratio_) {
      record_rejection("candidate_low_inlier_ratio");
      continue;
    }
    const double max_rmse = std::max(0.0, place_recognition_max_inlier_rmse_m_);
    if (max_rmse > 0.0 && candidate_result.inlier_rmse_m > max_rmse) {
      record_rejection("candidate_high_inlier_rmse");
      continue;
    }

    const double max_jump = effectivePlaceRecognitionTranslationJumpLimit(
      place_recognition_max_pose_jump_m_,
      place_recognition_registration_only_max_pose_jump_m_,
      candidate.registration_only_vertical_recovery);
    const double max_jump_rotation = std::max(
      0.0,
      place_recognition_max_pose_jump_rotation_deg_);
    const bool normal_translation_accepted =
      max_jump <= 0.0 || pose_jump_translation_m <= max_jump;
    const bool normal_rotation_accepted =
      max_jump_rotation <= 0.0 || pose_jump_rotation_deg <= max_jump_rotation;
    const bool provisional_large_pose_jump =
      place_recognition_mapping_refinement_large_pose_jump_enabled_ &&
      place_recognition_mapping_cloud_refinement_enabled_ &&
      candidate.distance <=
      place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_ &&
      pose_jump_translation_m <=
      place_recognition_mapping_refinement_large_pose_jump_max_translation_m_ &&
      pose_jump_rotation_deg <=
      place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_;
    if ((!normal_translation_accepted || !normal_rotation_accepted) &&
      !provisional_large_pose_jump)
    {
      record_rejection(
        !normal_translation_accepted ?
        "candidate_pose_jump_translation" : "candidate_pose_jump_rotation");
      continue;
    }

    if (score < best_score) {
      best_score = score;
      best_result = candidate_result;
      best_result.place_recognition_attempted = true;
      best_result.place_recognition_accepted = true;
      best_result.place_recognition_candidates = result.place_recognition_candidates;
      best_result.place_recognition_eligible_descriptors =
        result.place_recognition_eligible_descriptors;
      best_result.place_recognition_registration_only_vertical_recovery_descriptors =
        result.place_recognition_registration_only_vertical_recovery_descriptors;
      best_result.place_recognition_registration_only_vertical_recovery_used =
        candidate.registration_only_vertical_recovery;
      best_result.place_recognition_candidate_index = candidate.keyframe_index;
      best_result.place_recognition_candidate_submap_id = candidate.submap_id;
      best_result.place_recognition_descriptor_distance = candidate.distance;
      best_result.place_recognition_nearest_descriptor_distance =
        result.place_recognition_nearest_descriptor_distance;
      best_result.place_recognition_yaw_offset_deg = yaw_offset * 180.0 / M_PI;
      best_result.place_recognition_large_pose_jump_provisional =
        provisional_large_pose_jump &&
        (!normal_translation_accepted || !normal_rotation_accepted);
      best_result.place_recognition_candidate_registration_status =
        best_result.place_recognition_large_pose_jump_provisional ?
        "provisional_large_pose_jump" : "accepted_normal_gate";
      best_result.place_recognition_candidate_registration_inlier_ratio =
        candidate_result.inlier_ratio;
      best_result.place_recognition_candidate_registration_inlier_rmse_m =
        candidate_result.inlier_rmse_m;
      best_result.place_recognition_candidate_pose_jump_translation_m =
        pose_jump_translation_m;
      best_result.place_recognition_candidate_pose_jump_rotation_deg =
        pose_jump_rotation_deg;
      best_result.rejection_reason = "place_recognition";
      best_candidate_map = candidate_map;
      best_candidate_seed_pose = candidate_pose;
      best_candidate_submap_index = candidate.submap_index;
      have_match = true;
    }
  }

  if (!have_match) {
    const auto & best_candidate = best_rejected.available ?
      best_rejected.candidate : candidates.front();
    result.place_recognition_candidate_index = best_candidate.keyframe_index;
    result.place_recognition_candidate_submap_id = best_candidate.submap_id;
    result.place_recognition_descriptor_distance = best_candidate.distance;
    result.place_recognition_yaw_offset_deg = best_rejected.available ?
      best_rejected.yaw_offset_deg :
      normalizedAngle(-static_cast<double>(best_candidate.shift) * yaw_bin_width) * 180.0 / M_PI;
    if (best_rejected.available) {
      result.place_recognition_candidate_registration_status = best_rejected.status;
      result.place_recognition_candidate_registration_inlier_ratio =
        best_rejected.inlier_ratio;
      result.place_recognition_candidate_registration_inlier_rmse_m =
        best_rejected.inlier_rmse_m;
      result.place_recognition_candidate_pose_jump_translation_m =
        best_rejected.pose_jump_translation_m;
      result.place_recognition_candidate_pose_jump_rotation_deg =
        best_rejected.pose_jump_rotation_deg;
      result.place_recognition_skipped = true;
      result.place_recognition_skip_reason = best_rejected.status;
      if (place_recognition_mapping_cloud_refinement_enabled_) {
        result.place_recognition_mapping_refinement_status =
          "not_reached_" + best_rejected.status;
      }
    }
    if (place_recognition_use_submaps_ && !preserve_recovery_consistency()) {
      reset_submap_consistency();
    }
    return false;
  }

  attachRegistrationInformation(scan_base, best_candidate_map, best_result);

  // Rank candidates using one comparable coarse-registration score, then
  // refine only the selected candidate. Refining inside the candidate loop
  // adds hidden callback latency and mixes backend fitness definitions.
  if (place_recognition_mapping_cloud_refinement_enabled_) {
    const auto mapping_candidate_map = best_candidate_submap_index >= 0 ?
      buildCandidateMappingMapAroundSubmap(
      static_cast<std::size_t>(best_candidate_submap_index)) :
      buildCandidateMappingMapAroundKeyframe(
      best_result.place_recognition_candidate_index,
      latest_candidate_exclusive);
    refinePlaceRecognitionWithMappingCloud(
      scan_base,
      mapping_scan_base,
      best_candidate_map,
      mapping_candidate_map,
      best_candidate_seed_pose,
      best_result);

    const bool mapping_verification_required =
      best_result.place_recognition_large_pose_jump_provisional ||
      place_recognition_mapping_refinement_required_for_pose_graph_;
    if (mapping_verification_required &&
      !best_result.place_recognition_mapping_refinement_verified)
    {
      best_result.place_recognition_accepted = false;
      best_result.place_recognition_skipped = true;
      best_result.place_recognition_skip_reason =
        best_result.place_recognition_large_pose_jump_provisional ?
        "large_pose_jump_mapping_refinement_required" :
        "mapping_refinement_required_for_pose_graph";
      best_result.rejection_reason = best_result.place_recognition_skip_reason;
      result = best_result;
      if (place_recognition_use_submaps_ && !preserve_recovery_consistency()) {
        reset_submap_consistency();
      }
      return false;
    }

    if (best_result.place_recognition_mapping_refinement_verified) {
      const double max_rmse = std::max(0.0, place_recognition_max_inlier_rmse_m_);
      const Eigen::Isometry3d refined_pose_jump =
        reference_pose.inverse() * best_result.map_T_base;
      double max_jump = effectivePlaceRecognitionTranslationJumpLimit(
        place_recognition_max_pose_jump_m_,
        place_recognition_registration_only_max_pose_jump_m_,
        best_result.place_recognition_registration_only_vertical_recovery_used);
      double max_jump_rotation = std::max(
        0.0,
        place_recognition_max_pose_jump_rotation_deg_);
      if (best_result.place_recognition_large_pose_jump_provisional) {
        max_jump = place_recognition_mapping_refinement_large_pose_jump_max_translation_m_;
        max_jump_rotation =
          place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_;
      }
      const bool quality_accepted =
        best_result.inlier_ratio >= place_recognition_min_inlier_ratio_ &&
        (max_rmse <= 0.0 || best_result.inlier_rmse_m <= max_rmse);
      const bool translation_accepted =
        max_jump <= 0.0 || refined_pose_jump.translation().norm() <= max_jump;
      const bool rotation_accepted =
        max_jump_rotation <= 0.0 ||
        Eigen::AngleAxisd(refined_pose_jump.linear()).angle() * 180.0 / M_PI <=
        max_jump_rotation;
      if (!quality_accepted || !translation_accepted || !rotation_accepted) {
        best_result.place_recognition_accepted = false;
        best_result.place_recognition_skipped = true;
        best_result.place_recognition_skip_reason =
          !quality_accepted ? "mapping_refinement_quality" :
          (!translation_accepted ? "mapping_refinement_translation" :
          "mapping_refinement_rotation");
        best_result.rejection_reason = best_result.place_recognition_skip_reason;
        result = best_result;
        if (place_recognition_use_submaps_ && !preserve_recovery_consistency()) {
          reset_submap_consistency();
        }
        return false;
      }
      if (best_result.place_recognition_large_pose_jump_provisional) {
        best_result.place_recognition_large_pose_jump_accepted = true;
        best_result.place_recognition_candidate_registration_status =
          "large_pose_jump_mapping_refinement_accepted";
      }
    }
  }

  best_result.place_recognition_alignment_correction_translation_m =
    best_result.correction_translation_m;
  best_result.place_recognition_alignment_correction_rotation_deg =
    best_result.correction_rotation_deg;

  if (place_recognition_use_submaps_) {
    const int candidate_submap_id = best_result.place_recognition_candidate_submap_id;
    const Eigen::Isometry3d candidate_correction = projectToRigidIsometry(
      best_result.map_T_base * reference_pose.inverse());
    bool transform_consistent = false;
    const bool same_normal_submap =
      candidate_submap_id == last_place_candidate_submap_id_ &&
      !best_result.place_recognition_registration_only_vertical_recovery_used &&
      !last_place_candidate_used_registration_only_vertical_recovery_;
    const bool recovery_continuous =
      placeRecognitionRecoveryConsistencyContinuous(
      best_result.place_recognition_registration_only_vertical_recovery_used,
      last_place_candidate_used_registration_only_vertical_recovery_,
      candidate_submap_id,
      last_place_candidate_submap_id_,
      (stamp - last_place_candidate_stamp_).seconds(),
      place_recognition_registration_only_consistency_max_gap_s_);
    if ((same_normal_submap || recovery_continuous) &&
      have_last_place_candidate_correction_)
    {
      const Eigen::Isometry3d consistency_delta = projectToRigidIsometry(
        last_place_candidate_correction_.inverse() * candidate_correction);
      best_result.place_recognition_candidate_consistency_translation_m =
        consistency_delta.translation().norm();
      best_result.place_recognition_candidate_consistency_rotation_deg =
        rotationAngleDeg(consistency_delta.linear());
      const auto consistency_translation =
        evaluatePlaceRecognitionConsistencyTranslationLimit(
        place_recognition_submap_consistency_max_translation_m_,
        place_recognition_registration_only_consistency_max_translation_m_,
        best_result.place_recognition_registration_only_vertical_recovery_used,
        last_place_candidate_used_registration_only_vertical_recovery_);
      const double max_consistency_translation =
        consistency_translation.active_limit_m;
      best_result.place_recognition_candidate_consistency_translation_limit_m =
        max_consistency_translation;
      best_result.place_recognition_registration_only_consistency_relaxation_used =
        consistency_translation.used_registration_only_recovery;
      const double max_consistency_rotation = std::max(
        0.0,
        place_recognition_submap_consistency_max_rotation_deg_);
      transform_consistent =
        (max_consistency_translation <= 0.0 ||
        best_result.place_recognition_candidate_consistency_translation_m <=
        max_consistency_translation) &&
        (max_consistency_rotation <= 0.0 ||
        best_result.place_recognition_candidate_consistency_rotation_deg <=
        max_consistency_rotation);
    }
    if (transform_consistent) {
      ++consecutive_place_candidate_submap_matches_;
    } else {
      last_place_candidate_submap_id_ = candidate_submap_id;
      consecutive_place_candidate_submap_matches_ = 1;
    }
    last_place_candidate_correction_ = candidate_correction;
    have_last_place_candidate_correction_ = true;
    last_place_candidate_used_registration_only_vertical_recovery_ =
      best_result.place_recognition_registration_only_vertical_recovery_used;
    last_place_candidate_stamp_ = stamp;
    best_result.place_recognition_candidate_consistency_count =
      consecutive_place_candidate_submap_matches_;
    if (consecutive_place_candidate_submap_matches_ <
      place_recognition_submap_min_consecutive_matches_)
    {
      result = best_result;
      result.place_recognition_accepted = false;
      result.place_recognition_skipped = true;
      result.place_recognition_skip_reason = "submap_consistency_pending";
      result.rejection_reason = "submap_consistency_pending";
      return false;
    }
  } else {
    reset_submap_consistency();
  }

  result = best_result;
  if (place_recognition_use_submaps_) {
    reset_submap_consistency();
  }
  return true;
}

void ScanToMapIcp::refreshKeyframeCloudMap(const std::size_t keyframe_index)
{
  if (keyframe_index >= keyframes_.size()) {
    return;
  }
  auto & keyframe = keyframes_[keyframe_index];
  if (keyframe.cloud_base && !keyframe.cloud_base->empty()) {
    keyframe.cloud_map = filterMapCloud(
      transformCloud(keyframe.cloud_base, keyframe.map_T_base));
  }
  if (keyframe.mapping_cloud_base && !keyframe.mapping_cloud_base->empty()) {
    keyframe.mapping_cloud_map = filterMapCloud(
      transformCloud(keyframe.mapping_cloud_base, keyframe.map_T_base));
  }
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
  return poseGraphConstraintChiSquared(poseGraphConstraintFromEdge(edge), poses);
}

PoseGraphConstraint ScanToMapIcp::poseGraphConstraintFromEdge(const PoseGraphEdge & edge)
{
  PoseGraphConstraint constraint;
  constraint.from_index = edge.from_index;
  constraint.to_index = edge.to_index;
  constraint.from_T_to = edge.from_T_to;
  constraint.information_matrix = edge.information_matrix;
  constraint.loop_closure = edge.loop_closure;
  return constraint;
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
      return std::numeric_limits<double>::infinity();
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
  const std::vector<Eigen::Isometry3d> & previous_keyframe_poses,
  const bool same_layer_z_constraint,
  const double loop_vertical_translation_weight,
  const bool distributed_loop_seed,
  const RegistrationInformationResult * registration_information)
{
  const auto optimization_started = std::chrono::steady_clock::now();
  PoseGraphStats stats;
  auto finalize = [&](
    const std::string & solver_status,
    const std::string & convergence_reason) -> PoseGraphStats
    {
      stats.solver_status = solver_status;
      stats.convergence_reason = convergence_reason;
      stats.optimization_runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - optimization_started).count();
      stats.mean_iteration_runtime_ms = stats.iterations > 0 ?
        stats.iteration_runtime_ms / static_cast<double>(stats.iterations) : 0.0;
      return stats;
    };
  stats.nodes = static_cast<int>(keyframes_.size());
  if (stats.nodes < 2 || previous_keyframe_poses.size() != keyframes_.size()) {
    return finalize("not_started", "invalid_keyframe_sequence");
  }

  const int last_index = stats.nodes - 1;
  if (anchor_index < 0 || anchor_index >= last_index) {
    return finalize("not_started", "invalid_anchor");
  }

  PoseGraphEdge loop_edge;
  loop_edge.from_index = anchor_index;
  loop_edge.to_index = last_index;
  loop_edge.from_T_to =
    previous_keyframe_poses[static_cast<std::size_t>(anchor_index)].inverse() *
    target_loop_pose;
  loop_edge.translation_weight = std::max(0.0, pose_graph_loop_translation_weight_);
  loop_edge.vertical_translation_weight = std::max(
    0.0, loop_vertical_translation_weight);
  loop_edge.rotation_weight = std::max(0.0, pose_graph_loop_rotation_weight_);
  Vector6d configured_loop_information;
  configured_loop_information <<
    loop_edge.translation_weight,
    loop_edge.translation_weight,
    loop_edge.vertical_translation_weight,
    loop_edge.rotation_weight,
    loop_edge.rotation_weight,
    loop_edge.rotation_weight;
  loop_edge.information_matrix = configuredDiagonalInformation(configured_loop_information);
  if (registration_information != nullptr && registration_information->valid) {
    loop_edge.information_matrix = registration_information->information_matrix;
    loop_edge.registration_information_used = true;
    loop_edge.registration_observable_rank = registration_information->observable_rank;
    loop_edge.registration_hessian_condition = registration_information->observable_condition;
    loop_edge.registration_information_min_gain =
      registration_information->minimum_information_gain;
  }
  loop_edge.loop_closure = true;
  pose_graph_edges_.push_back(loop_edge);

  std::vector<Eigen::Isometry3d> poses = previous_keyframe_poses;
  std::vector<Eigen::Isometry3d> best_poses = poses;
  auto refresh_retained_loop_diagnostics = [&]() {
      stats.edges = static_cast<int>(pose_graph_edges_.size());
      stats.loop_edges = 0;
      stats.robust_loop_edges_downweighted = 0;
      stats.registration_information_edges = 0;
      stats.configured_diagonal_information_edges = 0;
      stats.minimum_registration_observable_rank = 0;
      stats.maximum_registration_hessian_condition = 0.0;
      stats.minimum_registration_information_gain = 1.0;
      stats.min_loop_robust_weight = 1.0;
      bool have_registration_information = false;
      for (const auto & edge : pose_graph_edges_) {
        if (edge.registration_information_used) {
          ++stats.registration_information_edges;
          stats.maximum_registration_hessian_condition = std::max(
            stats.maximum_registration_hessian_condition,
            edge.registration_hessian_condition);
          stats.minimum_registration_information_gain = std::min(
            stats.minimum_registration_information_gain,
            edge.registration_information_min_gain);
          stats.minimum_registration_observable_rank = have_registration_information ?
            std::min(
            stats.minimum_registration_observable_rank,
            edge.registration_observable_rank) : edge.registration_observable_rank;
          have_registration_information = true;
        } else {
          ++stats.configured_diagonal_information_edges;
        }
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

  stats.residual_before = poseGraphResidual(poses, false);
  stats.robust_objective_before = poseGraphResidual(poses, true);
  double best_objective = stats.robust_objective_before;

  if (distributed_loop_seed) {
    const DistributedPoseGraphSeedStats seed_stats =
      seedDistributedPoseGraphCorrection(poses, anchor_index, target_loop_pose);
    stats.distributed_loop_seed_path_length_m = seed_stats.path_length_m;
    stats.distributed_loop_seed_endpoint_translation_m =
      seed_stats.endpoint_correction_translation_m;
    stats.distributed_loop_seed_endpoint_rotation_deg =
      seed_stats.endpoint_correction_rotation_deg;
    stats.distributed_loop_seed_max_relative_translation_error_m =
      seed_stats.max_relative_translation_error_m;
    stats.distributed_loop_seed_max_relative_rotation_error_deg =
      seed_stats.max_relative_rotation_error_deg;
    const double seeded_objective = poseGraphResidual(poses, true);
    if (!seed_stats.valid || !seed_stats.applied ||
      !std::isfinite(seeded_objective) || seeded_objective >= best_objective)
    {
      reject_new_loop(
        !seed_stats.valid ? "distributed_loop_seed_invalid" :
        (!seed_stats.applied ? "distributed_loop_seed_not_applied" :
        "distributed_loop_seed_objective_not_improved"));
      stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
      stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
      return finalize("rejected", stats.rejection_reason);
    }
    stats.distributed_loop_seed_applied = true;
    best_objective = seeded_objective;
    best_poses = poses;
  }

  stats.new_loop_robust_weight = poseGraphLoopRobustWeight(
    loop_edge,
    poseGraphEdgeSquaredError(loop_edge, poses));
  const double min_new_loop_robust_weight =
    std::clamp(pose_graph_min_new_loop_robust_weight_, 0.0, 1.0);
  if (pose_graph_robust_loop_enabled_ &&
    stats.new_loop_robust_weight < min_new_loop_robust_weight)
  {
    reject_new_loop("new_loop_robust_weight");
    stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
    stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
    return finalize("rejected", "new_loop_robust_weight");
  }

  if (same_layer_z_constraint) {
    const std::vector<Eigen::Isometry3d> poses_before_z_seed = poses;
    const SameLayerPoseGraphZSeedStats seed_stats =
      seedSameLayerPoseGraphZCorrection(
      poses, anchor_index, target_loop_pose.translation().z());
    if (!seed_stats.valid) {
      reject_new_loop("same_layer_z_seed_invalid");
      stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
      stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
      return finalize("rejected", "same_layer_z_seed_invalid");
    }
    stats.same_layer_z_seed_correction_m =
      seed_stats.requested_endpoint_correction_m;
    stats.same_layer_z_seed_path_length_m = seed_stats.horizontal_path_length_m;
    stats.same_layer_z_seed_max_increment_m = seed_stats.max_z_increment_m;
    const double seeded_objective = poseGraphResidual(poses, true);
    if (seed_stats.applied && std::isfinite(seeded_objective) &&
      seeded_objective < best_objective)
    {
      stats.same_layer_z_seed_applied = true;
      best_objective = seeded_objective;
      best_poses = poses;
    } else {
      poses = poses_before_z_seed;
    }
  }

  const int max_iterations = std::max(1, pose_graph_optimization_iterations_);
  const double base_step = std::clamp(pose_graph_optimization_step_, 0.0, 1.0);
  const double max_translation_step =
    std::max(0.0, pose_graph_max_iteration_translation_step_m_);
  const double max_rotation_step =
    std::max(0.0, pose_graph_max_iteration_rotation_step_deg_);

  std::string convergence_reason = "max_iterations";
  const auto iteration_started = std::chrono::steady_clock::now();
  if (pose_graph_solver_ == "ceres_se3") {
    std::vector<PoseGraphConstraint> constraints;
    constraints.reserve(pose_graph_edges_.size());
    for (const auto & edge : pose_graph_edges_) {
      constraints.push_back(poseGraphConstraintFromEdge(edge));
    }

    const std::vector<Eigen::Isometry3d> poses_before_solver = poses;
    const PoseGraphOptimizationResult optimization =
      PoseGraphOptimizer(pose_graph_optimizer_config_).optimize(poses, constraints);
    stats.solver_stages = static_cast<int>(optimization.stages.size());
    stats.solver_successful_steps = optimization.successful_steps;
    stats.solver_unsuccessful_steps = optimization.unsuccessful_steps;
    stats.dcs_downweighted_loop_edges = optimization.dcs_downweighted_loops;
    stats.dcs_min_scale = optimization.min_dcs_scale;
    stats.solver_first_stage_initial_cost = optimization.initial_cost;
    stats.solver_last_stage_final_cost = optimization.final_cost;
    stats.solver_final_gradient_max_norm = optimization.final_gradient_max_norm;
    stats.iterations = optimization.total_iterations;
    convergence_reason = "ceres_" + optimization.termination_type;
    if (!optimization.success) {
      reject_new_loop("ceres_" + optimization.failure_reason);
      stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
      stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
      return finalize("failed_ceres_se3", convergence_reason);
    }
    if (optimization.final_dcs_scales.size() == constraints.size()) {
      stats.new_loop_dcs_scale = optimization.final_dcs_scales.back();
    }
    if (pose_graph_optimizer_config_.dcs_enabled &&
      stats.new_loop_dcs_scale < min_new_loop_robust_weight)
    {
      reject_new_loop("new_loop_dcs_scale");
      stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
      stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
      return finalize("rejected_ceres_se3", "dcs_outlier_gate");
    }

    poses = optimization.poses;
    for (std::size_t index = 0; index < poses.size(); ++index) {
      stats.max_update_translation_m = std::max(
        stats.max_update_translation_m,
        (poses[index].translation() - poses_before_solver[index].translation()).norm());
      const Eigen::Matrix3d rotation_delta =
        poses_before_solver[index].linear().transpose() * poses[index].linear();
      stats.max_update_rotation_deg = std::max(
        stats.max_update_rotation_deg,
        rotationAngleDeg(rotation_delta));
    }
    const double objective = poseGraphResidual(poses, true);
    if (std::isfinite(objective) && objective < best_objective) {
      best_objective = objective;
      best_poses = poses;
    }
  } else {
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
          projectToRigidIsometry(
          expected_to * poses[static_cast<std::size_t>(to_index)].inverse());
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
        const double vertical_translation_step =
          std::clamp(
          base_step * std::max(0.0, edge.vertical_translation_weight) * robust_weight,
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
          to_correction.translation().z() =
            correction.translation().z() * vertical_translation_step * to_share;
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
            projectToRigidIsometry(
            to_correction * poses[static_cast<std::size_t>(to_index)]);
        }

        if (from_index > 0) {
          const Eigen::Isometry3d inverse_correction = correction.inverse();
          Eigen::Isometry3d from_correction = interpolateTransform(
            inverse_correction,
            translation_step * from_share,
            rotation_step * from_share);
          from_correction.translation().z() =
            inverse_correction.translation().z() *
            vertical_translation_step * from_share;
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
            projectToRigidIsometry(
            from_correction * poses[static_cast<std::size_t>(from_index)]);
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
        convergence_reason = "step_tolerance";
        break;
      }
    }
  }
  stats.iteration_runtime_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - iteration_started).count();

  stats.residual_before_ground_sequence = poseGraphResidual(best_poses, false);
  stats.robust_objective_before_ground_sequence = poseGraphResidual(best_poses, true);
  // Re-project the optimized sequence so graph corrections cannot introduce
  // body-vertical motion that the frontend explicitly bounded.
  const auto ground_sequence_stats = constrainGroundVehiclePoseSequence(
    best_poses,
    ground_motion_constraint_config_,
    true);
  if (!ground_sequence_stats.valid) {
    reject_new_loop("ground_motion_sequence_invalid");
    stats.residual_after = poseGraphResidual(previous_keyframe_poses, false);
    stats.robust_objective_after = poseGraphResidual(previous_keyframe_poses, true);
    return finalize("rejected", convergence_reason);
  }
  stats.ground_sequence_applied_edges = ground_sequence_stats.applied_edges;
  stats.ground_sequence_max_raw_body_z_step_m =
    ground_sequence_stats.max_raw_body_z_step_m;
  stats.ground_sequence_max_constrained_body_z_step_m =
    ground_sequence_stats.max_constrained_body_z_step_m;

  stats.residual_after = poseGraphResidual(best_poses, false);
  stats.robust_objective_after = poseGraphResidual(best_poses, true);
  if (!std::isfinite(stats.residual_before) || !std::isfinite(stats.residual_after) ||
    !std::isfinite(stats.robust_objective_before) ||
    !std::isfinite(stats.robust_objective_after) ||
    stats.robust_objective_after >= stats.robust_objective_before)
  {
    reject_new_loop("objective_not_improved");
    return finalize("rejected", convergence_reason);
  }
  const double max_residual_after = std::max(0.0, pose_graph_max_residual_after_);
  if (max_residual_after > 0.0 && stats.residual_after > max_residual_after) {
    reject_new_loop("residual_after");
    return finalize("rejected", convergence_reason);
  }
  const double max_residual_ratio = std::max(0.0, pose_graph_max_residual_ratio_);
  if (max_residual_ratio > 0.0 && stats.residual_before > 1.0e-6 &&
    stats.residual_after / stats.residual_before > max_residual_ratio)
  {
    reject_new_loop("residual_ratio");
    return finalize("rejected", convergence_reason);
  }

  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    keyframes_[index].map_T_base = projectToRigidIsometry(best_poses[index]);
    refreshKeyframeCloudMap(index);
  }

  stats.applied = true;
  return finalize(
    pose_graph_solver_ == "ceres_se3" ?
    "accepted_ceres_se3" : "accepted_legacy_relaxation",
    convergence_reason);
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
      projectToRigidIsometry(
        keyframes_[index].map_T_base * previous_keyframe_poses[index].inverse()));
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
        projectToRigidIsometry(corrections[right_index] * corrections[left_index].inverse());
      correction = projectToRigidIsometry(
        interpolateTransform(relative_correction, ratio, ratio) * corrections[left_index]);
    }

    pose_stamped.pose = isometryToPose(correction * poseMsgToIsometry(pose_stamped.pose));
  }
}

void ScanToMapIcp::applyPoseGraphCorrection(
  const rclcpp::Time & stamp,
  const Eigen::Isometry3d & reference_pose,
  const std::optional<GravitySample> & gravity_sample,
  const std::optional<Eigen::Matrix3d> & current_world_R_base,
  const std::string & imu_attitude_sample_status,
  RegistrationResult & result)
{
  result.pose_graph_correction_applied = false;
  result.pose_graph_new_loop_rejected = false;
  result.pose_graph_new_loop_registration_information_used = false;
  result.pose_graph_minimum_interval_override_applied = false;
  result.pose_graph_duplicate_submap_pair_override_applied = false;
  result.pose_graph_minimum_interval_elapsed_s = -1.0;
  result.pose_graph_minimum_interval_status = "not_evaluated";
  result.pose_graph_duplicate_submap_pair_previous_edges = 0;
  result.pose_graph_duplicate_submap_pair_status = "not_evaluated";
  result.pose_graph_new_loop_information_status = "not_evaluated";
  result.pose_graph_same_layer_z_constraint_evaluated = false;
  result.pose_graph_same_layer_z_constraint_applied = false;
  result.pose_graph_same_layer_z_seed_applied = false;
  result.pose_graph_high_confidence_large_loop = false;
  result.pose_graph_distributed_loop_seed_applied = false;
  result.pose_graph_same_layer_reference_keyframe_index = -1;
  result.pose_graph_same_layer_reference_xy_distance_m = -1.0;
  result.pose_graph_same_layer_matched_current_z_before_m = 0.0;
  result.pose_graph_same_layer_reference_z_m = 0.0;
  result.pose_graph_same_layer_requested_z_correction_m = 0.0;
  result.pose_graph_same_layer_graph_target_z_m = 0.0;
  result.pose_graph_same_layer_z_translation_weight = 0.0;
  result.pose_graph_same_layer_z_seed_correction_m = 0.0;
  result.pose_graph_same_layer_z_seed_path_length_m = 0.0;
  result.pose_graph_same_layer_z_seed_max_increment_m = 0.0;
  result.pose_graph_distributed_loop_seed_path_length_m = 0.0;
  result.pose_graph_distributed_loop_seed_endpoint_translation_m = 0.0;
  result.pose_graph_distributed_loop_seed_endpoint_rotation_deg = 0.0;
  result.pose_graph_distributed_loop_seed_max_relative_translation_error_m = 0.0;
  result.pose_graph_distributed_loop_seed_max_relative_rotation_error_deg = 0.0;
  result.pose_graph_ground_sequence_applied_edges = 0;
  result.pose_graph_ground_sequence_max_raw_body_z_step_m = 0.0;
  result.pose_graph_ground_sequence_max_constrained_body_z_step_m = 0.0;
  result.pose_graph_same_layer_reference_source = "not_evaluated";
  result.pose_graph_rejection_reason.clear();
  result.pose_graph_correction_keyframes = 0;
  result.pose_graph_correction_translation_m = 0.0;
  result.pose_graph_correction_rotation_deg = 0.0;
  result.pose_graph_constraint_inlier_ratio = 0.0;
  result.pose_graph_constraint_inlier_rmse_m = 0.0;
  result.pose_graph_constraint_target_coverage_ratio = 0.0;
  result.pose_graph_optimization_iterations = 0;
  result.pose_graph_solver_stages = 0;
  result.pose_graph_solver_successful_steps = 0;
  result.pose_graph_solver_unsuccessful_steps = 0;
  result.pose_graph_dcs_downweighted_loop_edges = 0;
  result.pose_graph_registration_information_edges = 0;
  result.pose_graph_configured_diagonal_information_edges = 0;
  result.pose_graph_minimum_registration_observable_rank = 0;
  result.pose_graph_maximum_registration_hessian_condition = 0.0;
  result.pose_graph_minimum_registration_information_gain = 1.0;
  result.pose_graph_new_loop_information_min_gain = 1.0;
  result.pose_graph_optimization_runtime_ms = 0.0;
  result.pose_graph_iteration_runtime_ms = 0.0;
  result.pose_graph_mean_iteration_runtime_ms = 0.0;
  result.pose_graph_solver_status = "not_run";
  result.pose_graph_convergence_reason = "not_run";
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
  result.pose_graph_residual_before_ground_sequence = 0.0;
  result.pose_graph_robust_objective_before_ground_sequence = 0.0;
  result.pose_graph_min_loop_robust_weight = 1.0;
  result.pose_graph_new_loop_robust_weight = 1.0;
  result.pose_graph_dcs_min_scale = 1.0;
  result.pose_graph_new_loop_dcs_scale = 1.0;
  result.pose_graph_solver_first_stage_initial_cost = 0.0;
  result.pose_graph_solver_last_stage_final_cost = 0.0;
  result.pose_graph_solver_final_gradient_max_norm = 0.0;
  result.pose_graph_max_update_translation_m = 0.0;
  result.pose_graph_max_update_rotation_deg = 0.0;

  if (!pose_graph_correction_enabled_ || !result.place_recognition_accepted) {
    result.pose_graph_rejection_reason = "disabled_or_no_place_match";
    return;
  }
  const int source_submap_id = submaps_.empty() ? -1 : submaps_.back().id;
  const int target_submap_id = result.place_recognition_candidate_submap_id;
  const std::pair<int, int> submap_pair{source_submap_id, target_submap_id};
  PoseGraphMinimumIntervalMetrics interval_metrics;
  interval_metrics.have_previous_correction = have_last_pose_graph_correction_stamp_;
  if (have_last_pose_graph_correction_stamp_) {
    interval_metrics.elapsed_since_previous_correction_s =
      (stamp - last_pose_graph_correction_stamp_).seconds();
    result.pose_graph_minimum_interval_elapsed_s =
      interval_metrics.elapsed_since_previous_correction_s;
  }
  interval_metrics.mapping_refinement_verified =
    result.place_recognition_mapping_refinement_verified;
  interval_metrics.descriptor_distance = result.place_recognition_descriptor_distance;
  // Use the historical candidate/submap registration quality that measured
  // the loop edge. result.inlier_* may have been refreshed against the active,
  // drifted local map after applying motion constraints and is not the same
  // geometric hypothesis.
  interval_metrics.inlier_ratio =
    result.place_recognition_candidate_registration_inlier_ratio;
  interval_metrics.inlier_rmse_m =
    result.place_recognition_candidate_registration_inlier_rmse_m;
  interval_metrics.target_coverage_ratio =
    result.place_recognition_mapping_refinement_applied ?
    result.place_recognition_mapping_refinement_after_target_coverage_ratio :
    result.place_recognition_mapping_refinement_before_target_coverage_ratio;
  result.pose_graph_constraint_inlier_ratio = interval_metrics.inlier_ratio;
  result.pose_graph_constraint_inlier_rmse_m = interval_metrics.inlier_rmse_m;
  result.pose_graph_constraint_target_coverage_ratio =
    interval_metrics.target_coverage_ratio;
  result.pose_graph_high_confidence_large_loop =
    result.place_recognition_large_pose_jump_accepted &&
    pose_graph_distributed_loop_seed_enabled_ &&
    isHighConfidencePoseGraphConstraint(
    pose_graph_minimum_interval_override_config_, interval_metrics);
  const PoseGraphMinimumIntervalDecision interval_decision =
    evaluatePoseGraphMinimumInterval(
    pose_graph_correction_min_interval_s_,
    pose_graph_minimum_interval_override_config_,
    interval_metrics);
  result.pose_graph_minimum_interval_override_applied =
    interval_decision.override_applied;
  result.pose_graph_minimum_interval_status =
    poseGraphMinimumIntervalReasonString(interval_decision.reason);
  if (!interval_decision.allow) {
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
    projectToRigidIsometry(result.map_T_base * reference_pose.inverse());
  const Eigen::Isometry3d correction = pose_graph_planar_correction_enabled_ ?
    planarizeCorrection(raw_correction) : raw_correction;
  // A left correction rotates about the map origin, so its translation
  // coefficient grows with distance from that origin. Gate the physical
  // change at the current pose instead.
  const PlaceRecognitionPoseMetrics correction_metrics =
    computePlaceRecognitionPoseMetrics(
    reference_pose,
    projectToRigidIsometry(correction * reference_pose));
  if (!correction_metrics.finite) {
    result.pose_graph_rejection_reason = "invalid_correction";
    return;
  }
  const double correction_translation_m =
    correction_metrics.prediction_jump_translation_m;
  const double correction_rotation_deg =
    correction_metrics.prediction_jump_rotation_deg;
  result.pose_graph_correction_translation_m = correction_translation_m;
  result.pose_graph_correction_rotation_deg = correction_rotation_deg;

  const double max_translation = result.pose_graph_high_confidence_large_loop ?
    place_recognition_mapping_refinement_large_pose_jump_max_translation_m_ :
    std::max(0.0, pose_graph_correction_max_translation_m_);
  if (max_translation > 0.0 && correction_translation_m > max_translation) {
    result.pose_graph_rejection_reason = "correction_translation";
    return;
  }
  const double max_rotation = result.pose_graph_high_confidence_large_loop ?
    place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_ :
    std::max(0.0, pose_graph_correction_max_rotation_deg_);
  if (max_rotation > 0.0 && correction_rotation_deg > max_rotation) {
    result.pose_graph_rejection_reason = "correction_rotation";
    return;
  }

  const int last_index = keyframe_count - 1;
  if (anchor_index >= last_index) {
    result.pose_graph_rejection_reason = "anchor_not_historical";
    return;
  }

  // The geometric match is measured at the current scan, while the newest
  // graph vertex is the preceding keyframe. Transport the same left-map
  // correction to that keyframe; its keyframe-to-current motion stays intact.
  const Eigen::Isometry3d matched_current_pose =
    projectToRigidIsometry(correction * reference_pose);
  const Eigen::Isometry3d target_loop_pose = transportCurrentPlaceMatchToKeyframe(
    matched_current_pose,
    reference_pose,
    keyframes_.back().map_T_base);
  Eigen::Isometry3d same_layer_target_loop_pose = target_loop_pose;
  int graph_anchor_index = anchor_index;
  const PoseGraphSameLayerConstraintSelection same_layer_selection =
    selectPoseGraphSameLayerConstraint(
    result.place_recognition_registration_only_vertical_recovery_used,
    place_recognition_registration_only_same_layer_xy_max_distance_m_,
    result.pose_graph_minimum_interval_override_applied ||
    result.pose_graph_high_confidence_large_loop,
    pose_graph_minimum_interval_override_single_floor_z_enabled_,
    pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_);
  if (same_layer_selection.enabled) {
    result.pose_graph_same_layer_z_constraint_evaluated = true;
    result.pose_graph_same_layer_matched_current_z_before_m =
      matched_current_pose.translation().z();
    const std::string same_layer_mode =
      poseGraphSameLayerConstraintModeString(same_layer_selection.mode);

    std::vector<SameLayerReferenceCandidate> reference_candidates;
    if (place_recognition_use_submaps_ && result.place_recognition_candidate_submap_id >= 0) {
      const auto submap_it = std::find_if(
        submaps_.begin(), submaps_.end(),
        [&result](const Submap & submap) {
          return submap.id == result.place_recognition_candidate_submap_id;
        });
      if (submap_it != submaps_.end()) {
        reference_candidates.reserve(submap_it->keyframe_indices.size());
        for (const std::size_t keyframe_index : submap_it->keyframe_indices) {
          if (keyframe_index >= keyframes_.size() ||
            static_cast<int>(keyframe_index) >= last_index)
          {
            continue;
          }
          const Eigen::Vector3d & position = keyframes_[keyframe_index].map_T_base.translation();
          reference_candidates.push_back(
            SameLayerReferenceCandidate{
                static_cast<int>(keyframe_index), position.x(), position.y(), position.z()});
        }
        result.pose_graph_same_layer_reference_source =
          "candidate_submap_nearest_" + same_layer_mode;
      }
    }
    if (reference_candidates.empty()) {
      const Eigen::Vector3d & position =
        keyframes_[static_cast<std::size_t>(anchor_index)].map_T_base.translation();
      reference_candidates.push_back(
        SameLayerReferenceCandidate{anchor_index, position.x(), position.y(), position.z()});
      result.pose_graph_same_layer_reference_source =
        "candidate_anchor_fallback_" + same_layer_mode;
    }

    const SameLayerReferenceSelection selection = selectNearestSameLayerReference(
      matched_current_pose.translation().x(),
      matched_current_pose.translation().y(),
      same_layer_selection.max_horizontal_distance_m,
      reference_candidates);
    if (selection.found) {
      result.pose_graph_same_layer_reference_keyframe_index = selection.keyframe_index;
      result.pose_graph_same_layer_reference_xy_distance_m = selection.horizontal_distance_m;
      result.pose_graph_same_layer_reference_z_m = selection.z_m;
    }
    if (selection.within_max_distance) {
      const SameLayerGraphTargetCorrection same_layer_correction =
        transportSameLayerZCorrectionToGraphTarget(
        matched_current_pose, selection.z_m, target_loop_pose);
      if (same_layer_correction.valid) {
        same_layer_target_loop_pose = same_layer_correction.graph_target_pose;
        result.pose_graph_same_layer_requested_z_correction_m =
          same_layer_correction.requested_z_correction_m;
        result.pose_graph_same_layer_graph_target_z_m =
          same_layer_target_loop_pose.translation().z();
        graph_anchor_index = selection.keyframe_index;
        result.pose_graph_same_layer_z_constraint_applied = true;
      }
    }
  }
  const std::int64_t graph_anchor_stamp_ns =
    keyframes_[static_cast<std::size_t>(graph_anchor_index)].stamp.nanoseconds();
  if (source_submap_id >= 0 && target_submap_id >= 0) {
    const auto accepted_pair_it =
      accepted_loop_submap_pair_anchor_stamps_.find(submap_pair);
    PoseGraphDuplicateSubmapPairMetrics duplicate_pair_metrics;
    if (accepted_pair_it != accepted_loop_submap_pair_anchor_stamps_.end()) {
      duplicate_pair_metrics.accepted_edges =
        static_cast<int>(accepted_pair_it->second.size());
      duplicate_pair_metrics.target_anchor_already_used =
        accepted_pair_it->second.count(graph_anchor_stamp_ns) > 0U;
    }
    duplicate_pair_metrics.high_confidence_interval_override =
      interval_decision.override_applied;
    const PoseGraphDuplicateSubmapPairDecision duplicate_pair_decision =
      evaluatePoseGraphDuplicateSubmapPair(
      pose_graph_duplicate_submap_pair_override_config_, duplicate_pair_metrics);
    result.pose_graph_duplicate_submap_pair_previous_edges =
      duplicate_pair_metrics.accepted_edges;
    result.pose_graph_duplicate_submap_pair_override_applied =
      duplicate_pair_decision.override_applied;
    result.pose_graph_duplicate_submap_pair_status =
      poseGraphDuplicateSubmapPairReasonString(duplicate_pair_decision.reason);
    if (!duplicate_pair_decision.allow) {
      result.pose_graph_rejection_reason = "duplicate_submap_pair";
      return;
    }
  }
  const Eigen::Isometry3d & graph_target_loop_pose =
    result.pose_graph_same_layer_z_constraint_applied ?
    same_layer_target_loop_pose : target_loop_pose;
  const bool high_confidence_single_floor_constraint =
    result.pose_graph_same_layer_z_constraint_applied &&
    same_layer_selection.mode ==
    PoseGraphSameLayerConstraintMode::kHighConfidenceSingleFloorReturn;
  const double loop_vertical_translation_weight =
    effectivePoseGraphVerticalTranslationWeight(
    pose_graph_loop_translation_weight_,
    pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_,
    high_confidence_single_floor_constraint);
  result.pose_graph_same_layer_z_translation_weight =
    result.pose_graph_same_layer_z_constraint_applied ?
    loop_vertical_translation_weight : 0.0;
  Vector6d configured_loop_information;
  configured_loop_information <<
    std::max(0.0, pose_graph_loop_translation_weight_),
    std::max(0.0, pose_graph_loop_translation_weight_),
    std::max(0.0, loop_vertical_translation_weight),
    std::max(0.0, pose_graph_loop_rotation_weight_),
    std::max(0.0, pose_graph_loop_rotation_weight_),
    std::max(0.0, pose_graph_loop_rotation_weight_);
  RegistrationInformationResult loop_registration_information;
  const Eigen::Isometry3d latest_keyframe_T_current =
    keyframes_.back().map_T_base.inverse() * reference_pose;
  const Eigen::Isometry3d information_registration_pose = projectToRigidIsometry(
    graph_target_loop_pose * latest_keyframe_T_current);
  const bool information_pose_matches_loop_measurement =
    result.registration_information_pose_map_T_base.matrix().isApprox(
    information_registration_pose.matrix(), 1.0e-6);
  if (pose_graph_information_mode_ == "registration_observability" &&
    result.registration_information_valid && information_pose_matches_loop_measurement)
  {
    const Eigen::Vector3d endpoint_to_registration_origin_map =
      graph_target_loop_pose.linear() * latest_keyframe_T_current.translation();
    loop_registration_information = registrationHessianToPoseGraphInformation(
      result.registration_information_hessian_map_rt,
      keyframes_[static_cast<std::size_t>(graph_anchor_index)].map_T_base,
      configured_loop_information,
      pose_graph_information_config_,
      endpoint_to_registration_origin_map);
  }
  if (pose_graph_information_mode_ != "registration_observability") {
    result.pose_graph_new_loop_information_status = "configured_diagonal_mode";
  } else if (!result.registration_information_valid) {
    result.pose_graph_new_loop_information_status = "registration_information_invalid";
  } else if (!information_pose_matches_loop_measurement) {
    result.pose_graph_new_loop_information_status = "linearization_pose_mismatch";
  } else if (!loop_registration_information.valid) {
    result.pose_graph_new_loop_information_status =
      "conversion_" + loop_registration_information.failure_reason;
  } else {
    result.pose_graph_new_loop_information_status = "registration_observability";
  }
  result.pose_graph_new_loop_registration_information_used =
    loop_registration_information.valid;
  result.pose_graph_new_loop_information_min_gain = loop_registration_information.valid ?
    loop_registration_information.minimum_information_gain : 1.0;
  PoseGraphConstraint prospective_loop_constraint;
  prospective_loop_constraint.from_index = 0;
  prospective_loop_constraint.to_index = 1;
  prospective_loop_constraint.from_T_to =
    keyframes_[static_cast<std::size_t>(graph_anchor_index)].map_T_base.inverse() *
    graph_target_loop_pose;
  prospective_loop_constraint.information_matrix = loop_registration_information.valid ?
    loop_registration_information.information_matrix :
    configuredDiagonalInformation(configured_loop_information);
  prospective_loop_constraint.loop_closure = true;
  const std::vector<Eigen::Isometry3d> prospective_loop_poses{
    keyframes_[static_cast<std::size_t>(graph_anchor_index)].map_T_base,
    keyframes_.back().map_T_base};
  const double prospective_loop_residual = poseGraphConstraintChiSquared(
    prospective_loop_constraint,
    prospective_loop_poses);
  result.pose_graph_residual_before = prospective_loop_residual;
  const double max_loop_residual_before = std::max(0.0, pose_graph_max_loop_residual_before_);
  if (max_loop_residual_before > 0.0 &&
    prospective_loop_residual > max_loop_residual_before &&
    !result.pose_graph_high_confidence_large_loop)
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
    graph_anchor_index,
    graph_target_loop_pose,
    previous_keyframe_poses,
    result.pose_graph_same_layer_z_constraint_applied,
    loop_vertical_translation_weight,
    result.pose_graph_high_confidence_large_loop,
    loop_registration_information.valid ? &loop_registration_information : nullptr);
  result.pose_graph_optimization_iterations = stats.iterations;
  result.pose_graph_solver_stages = stats.solver_stages;
  result.pose_graph_solver_successful_steps = stats.solver_successful_steps;
  result.pose_graph_solver_unsuccessful_steps = stats.solver_unsuccessful_steps;
  result.pose_graph_dcs_downweighted_loop_edges = stats.dcs_downweighted_loop_edges;
  result.pose_graph_registration_information_edges = stats.registration_information_edges;
  result.pose_graph_configured_diagonal_information_edges =
    stats.configured_diagonal_information_edges;
  result.pose_graph_minimum_registration_observable_rank =
    stats.minimum_registration_observable_rank;
  result.pose_graph_maximum_registration_hessian_condition =
    stats.maximum_registration_hessian_condition;
  result.pose_graph_minimum_registration_information_gain =
    stats.minimum_registration_information_gain;
  result.pose_graph_edges = stats.edges;
  result.pose_graph_loop_edges = stats.loop_edges;
  result.pose_graph_robust_loop_edges_downweighted = stats.robust_loop_edges_downweighted;
  result.pose_graph_residual_before = stats.residual_before;
  result.pose_graph_residual_after = stats.residual_after;
  result.pose_graph_robust_objective_before = stats.robust_objective_before;
  result.pose_graph_robust_objective_after = stats.robust_objective_after;
  result.pose_graph_residual_before_ground_sequence =
    stats.residual_before_ground_sequence;
  result.pose_graph_robust_objective_before_ground_sequence =
    stats.robust_objective_before_ground_sequence;
  result.pose_graph_min_loop_robust_weight = stats.min_loop_robust_weight;
  result.pose_graph_new_loop_robust_weight = stats.new_loop_robust_weight;
  result.pose_graph_dcs_min_scale = stats.dcs_min_scale;
  result.pose_graph_new_loop_dcs_scale = stats.new_loop_dcs_scale;
  result.pose_graph_solver_first_stage_initial_cost =
    stats.solver_first_stage_initial_cost;
  result.pose_graph_solver_last_stage_final_cost =
    stats.solver_last_stage_final_cost;
  result.pose_graph_solver_final_gradient_max_norm =
    stats.solver_final_gradient_max_norm;
  result.pose_graph_max_update_translation_m = stats.max_update_translation_m;
  result.pose_graph_max_update_rotation_deg = stats.max_update_rotation_deg;
  result.pose_graph_same_layer_z_seed_applied = stats.same_layer_z_seed_applied;
  result.pose_graph_distributed_loop_seed_applied =
    stats.distributed_loop_seed_applied;
  result.pose_graph_same_layer_z_seed_correction_m =
    stats.same_layer_z_seed_correction_m;
  result.pose_graph_same_layer_z_seed_path_length_m =
    stats.same_layer_z_seed_path_length_m;
  result.pose_graph_same_layer_z_seed_max_increment_m =
    stats.same_layer_z_seed_max_increment_m;
  result.pose_graph_distributed_loop_seed_path_length_m =
    stats.distributed_loop_seed_path_length_m;
  result.pose_graph_distributed_loop_seed_endpoint_translation_m =
    stats.distributed_loop_seed_endpoint_translation_m;
  result.pose_graph_distributed_loop_seed_endpoint_rotation_deg =
    stats.distributed_loop_seed_endpoint_rotation_deg;
  result.pose_graph_distributed_loop_seed_max_relative_translation_error_m =
    stats.distributed_loop_seed_max_relative_translation_error_m;
  result.pose_graph_distributed_loop_seed_max_relative_rotation_error_deg =
    stats.distributed_loop_seed_max_relative_rotation_error_deg;
  result.pose_graph_ground_sequence_applied_edges =
    stats.ground_sequence_applied_edges;
  result.pose_graph_ground_sequence_max_raw_body_z_step_m =
    stats.ground_sequence_max_raw_body_z_step_m;
  result.pose_graph_ground_sequence_max_constrained_body_z_step_m =
    stats.ground_sequence_max_constrained_body_z_step_m;
  result.pose_graph_optimization_runtime_ms = stats.optimization_runtime_ms;
  result.pose_graph_iteration_runtime_ms = stats.iteration_runtime_ms;
  result.pose_graph_mean_iteration_runtime_ms = stats.mean_iteration_runtime_ms;
  result.pose_graph_solver_status = stats.solver_status;
  result.pose_graph_convergence_reason = stats.convergence_reason;
  result.pose_graph_new_loop_rejected = stats.new_loop_rejected;
  result.pose_graph_rejection_reason = stats.rejection_reason;

  if (!stats.applied) {
    return;
  }

  updatePathFromKeyframeCorrections(previous_keyframe_poses);
  updateTerrainTraversalSamplesFromKeyframeCorrections(previous_keyframe_poses);
  const Eigen::Isometry3d latest_keyframe_correction =
    projectToRigidIsometry(
    keyframes_.back().map_T_base * previous_keyframe_poses.back().inverse());
  map_T_base_ = constrainPlanarPose(
    constrainImuRelativeAttitude(
      projectToRigidIsometry(latest_keyframe_correction * reference_pose),
      current_world_R_base,
      imu_attitude_sample_status,
      &result),
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
  if (source_submap_id >= 0 && target_submap_id >= 0) {
    accepted_loop_submap_pair_anchor_stamps_[submap_pair].insert(graph_anchor_stamp_ns);
  }
  result.pose_graph_correction_keyframes = stats.nodes - graph_anchor_index - 1;
  publishPoseGraphMarkers(stamp);
}

void ScanToMapIcp::rebuildGlobalMap()
{
  auto merged = std::make_shared<CloudT>();
  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    if (!keyframes_[index].mapping_cloud_map ||
      keyframes_[index].mapping_cloud_map->empty())
    {
      refreshKeyframeCloudMap(index);
    }
    const auto & keyframe = keyframes_[index];
    if (keyframe.mapping_cloud_map) {
      *merged += *keyframe.mapping_cloud_map;
    }
  }
  global_map_ = downsampleCloud(merged, map_voxel_leaf_m_);
  rebuildSubmaps();
  rebuildTerrainEvidenceMap();
  latest_local_map_ = buildLocalMap(map_T_base_);
}

void ScanToMapIcp::exportKeyframes(
  const std::shared_ptr<std_srvs::srv::Trigger::Response> & response)
{
  if (keyframe_archive_directory_.empty()) {
    response->success = false;
    response->message = "Set keyframe_archive_directory before mapping to retain sensor views.";
    return;
  }
  KeyframeArchive archive;
  {
    // Capture poses and immutable base clouds from one graph revision; perform I/O unlocked.
    std::lock_guard<std::mutex> lock(state_mutex_);
    archive.map_frame = map_frame_;
    archive.base_frame = base_frame_;
    archive.map_z_filter_enabled = map_z_filter_enabled_;
    archive.map_z_min_m = std::min(map_z_filter_min_m_, map_z_filter_max_m_);
    archive.map_z_max_m = std::max(map_z_filter_min_m_, map_z_filter_max_m_);
    archive.pruned_keyframes = pruned_keyframe_count_;
    archive.keyframes.reserve(keyframes_.size());
    for (const auto & keyframe : keyframes_) {
      ArchivedKeyframe entry;
      entry.id = keyframe.archive_id;
      entry.stamp_ns = keyframe.stamp.nanoseconds();
      entry.map_T_base = keyframe.map_T_base;
      for (const auto & scan : keyframe.archive_sensor_scans) {
        if (scan.cloud_base && !scan.cloud_base->empty()) {
          entry.scans.push_back(ArchivedSensorScan{scan.cloud_base, scan.sensor_origin_base});
        }
      }
      archive.keyframes.push_back(std::move(entry));
    }
  }
  try {
    writeKeyframeArchive(keyframe_archive_directory_, archive);
    response->success = true;
    response->message = "Exported " + std::to_string(archive.keyframes.size()) +
      " retained keyframes to " + keyframe_archive_directory_ +
      " (pruned=" + std::to_string(archive.pruned_keyframes) + ")";
    RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
  } catch (const std::exception & error) {
    response->success = false;
    response->message = error.what();
    RCLCPP_ERROR(get_logger(), "Keyframe export failed: %s", error.what());
  }
}

void ScanToMapIcp::publishState(
  const rclcpp::Time & stamp,
  RegistrationResult & result)
{
  recordTerrainTraversalSample(stamp, result);
  refreshMapPublishSnapshots();
  nav_msgs::msg::Odometry odom;
  odom.header.stamp = stamp;
  odom.header.frame_id = map_frame_;
  odom.child_frame_id = base_frame_;
  odom.pose.pose = isometryToPose(map_T_base_);

  const double covariance =
    (result.pose_quarantined || result.shock_quarantined || result.imu_shock_active) ?
    100.0 : (result.converged ? std::max(result.fitness, 0.01) : 100.0);
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
  const bool omit_path_pose = omit_shock_poses_from_path_ &&
    (result.imu_shock_active || result.shock_quarantined || result.pose_quarantined);
  result.path_pose_appended = !omit_path_pose;
  last_registration_.path_pose_appended = result.path_pose_appended;
  if (!omit_path_pose) {
    path_msg_.poses.push_back(pose_stamped);
    if (max_path_poses_ > 0 &&
      static_cast<int>(path_msg_.poses.size()) > max_path_poses_)
    {
      path_msg_.poses.erase(path_msg_.poses.begin());
    }
  }
  path_pub_->publish(path_msg_);

  if (publish_tf_) {
    tf_broadcaster_->sendTransform(
      isometryToTransform(map_T_base_, map_frame_, base_frame_, stamp));
  }
}

void ScanToMapIcp::publishPoseGraphMarkers(const rclcpp::Time & stamp)
{
  if (!pose_graph_marker_pub_ || keyframes_.empty()) {
    return;
  }

  using Marker = visualization_msgs::msg::Marker;
  visualization_msgs::msg::MarkerArray marker_array;

  auto initialize_marker =
    [this, &stamp](Marker & marker, const std::string & marker_namespace, const int id) {
      marker.header.frame_id = map_frame_;
      marker.header.stamp = stamp;
      marker.ns = marker_namespace;
      marker.id = id;
      marker.action = Marker::ADD;
      marker.pose.orientation.w = 1.0;
    };
  auto point_from_pose = [](const Eigen::Isometry3d & pose) {
      geometry_msgs::msg::Point point;
      point.x = pose.translation().x();
      point.y = pose.translation().y();
      point.z = pose.translation().z();
      return point;
    };

  Marker nodes;
  initialize_marker(nodes, "pose_graph_nodes", 0);
  nodes.type = Marker::SPHERE_LIST;
  nodes.scale.x = 0.20;
  nodes.scale.y = 0.20;
  nodes.scale.z = 0.20;
  nodes.points.reserve(keyframes_.size());
  nodes.colors.reserve(keyframes_.size());
  const double color_denominator =
    static_cast<double>(std::max<std::size_t>(1, keyframes_.size() - 1));
  for (std::size_t index = 0; index < keyframes_.size(); ++index) {
    nodes.points.push_back(point_from_pose(keyframes_[index].map_T_base));
    const double ratio = static_cast<double>(index) / color_denominator;
    std_msgs::msg::ColorRGBA color;
    color.r = static_cast<float>(0.15 + 0.80 * ratio);
    color.g = static_cast<float>(0.75 - 0.25 * ratio);
    color.b = static_cast<float>(0.95 - 0.75 * ratio);
    color.a = 1.0F;
    nodes.colors.push_back(color);
  }
  marker_array.markers.push_back(std::move(nodes));

  Marker odometry_edges;
  initialize_marker(odometry_edges, "pose_graph_odometry_edges", 1);
  odometry_edges.type = Marker::LINE_LIST;
  odometry_edges.scale.x = 0.045;
  odometry_edges.color.r = 0.10F;
  odometry_edges.color.g = 0.80F;
  odometry_edges.color.b = 0.45F;
  odometry_edges.color.a = 0.95F;

  Marker loop_edges;
  initialize_marker(loop_edges, "pose_graph_loop_edges", 2);
  loop_edges.type = Marker::LINE_LIST;
  loop_edges.scale.x = 0.12;
  loop_edges.color.r = 0.95F;
  loop_edges.color.g = 0.15F;
  loop_edges.color.b = 0.55F;
  loop_edges.color.a = 1.0F;

  for (const auto & edge : pose_graph_edges_) {
    if (edge.from_index < 0 || edge.to_index < 0 ||
      edge.from_index >= static_cast<int>(keyframes_.size()) ||
      edge.to_index >= static_cast<int>(keyframes_.size()))
    {
      continue;
    }
    auto & marker = edge.loop_closure ? loop_edges : odometry_edges;
    marker.points.push_back(
      point_from_pose(keyframes_[static_cast<std::size_t>(edge.from_index)].map_T_base));
    marker.points.push_back(
      point_from_pose(keyframes_[static_cast<std::size_t>(edge.to_index)].map_T_base));
  }
  marker_array.markers.push_back(std::move(odometry_edges));
  marker_array.markers.push_back(std::move(loop_edges));

  Marker latest_pose;
  initialize_marker(latest_pose, "pose_graph_latest_pose", 3);
  latest_pose.type = Marker::ARROW;
  latest_pose.pose = isometryToPose(keyframes_.back().map_T_base);
  latest_pose.scale.x = 0.80;
  latest_pose.scale.y = 0.14;
  latest_pose.scale.z = 0.20;
  latest_pose.color.r = 1.0F;
  latest_pose.color.g = 0.85F;
  latest_pose.color.b = 0.10F;
  latest_pose.color.a = 1.0F;
  marker_array.markers.push_back(std::move(latest_pose));

  if (submap_enabled_ && !submaps_.empty()) {
    Marker submap_anchors;
    initialize_marker(submap_anchors, "submap_anchors", 4);
    submap_anchors.type = Marker::CUBE_LIST;
    submap_anchors.scale.x = 0.42;
    submap_anchors.scale.y = 0.42;
    submap_anchors.scale.z = 0.42;
    submap_anchors.points.reserve(submaps_.size());
    submap_anchors.colors.reserve(submaps_.size());
    for (const auto & submap : submaps_) {
      const std::size_t anchor_index = submapAnchorKeyframeIndex(submap);
      if (anchor_index >= keyframes_.size()) {
        continue;
      }
      submap_anchors.points.push_back(point_from_pose(keyframes_[anchor_index].map_T_base));
      std_msgs::msg::ColorRGBA color;
      if (submap.completed) {
        color.r = 0.10F;
        color.g = 0.65F;
        color.b = 1.00F;
      } else {
        color.r = 1.00F;
        color.g = 0.75F;
        color.b = 0.10F;
      }
      color.a = 1.0F;
      submap_anchors.colors.push_back(color);
    }
    marker_array.markers.push_back(std::move(submap_anchors));
  }

  pose_graph_marker_pub_->publish(marker_array);
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

void ScanToMapIcp::publishTerrainCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
  const TerrainCloudConstPtr & cloud,
  const std::string & frame_id,
  const rclcpp::Time & stamp) const
{
  if (!publisher || !cloud) {
    return;
  }

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  publisher->publish(msg);
}

void ScanToMapIcp::publishTerrainColorCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
  const TerrainCloudConstPtr & cloud,
  const std::string & frame_id,
  const rclcpp::Time & stamp) const
{
  if (!publisher || !cloud) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZRGB> colored;
  colored.reserve(cloud->size());
  for (const auto & source : cloud->points) {
    const int label_value = static_cast<int>(std::lround(source.intensity));
    TerrainLabel label = TerrainLabel::kUnknown;
    if (label_value >= static_cast<int>(TerrainLabel::kFloor) &&
      label_value <= static_cast<int>(TerrainLabel::kStaircase))
    {
      label = static_cast<TerrainLabel>(label_value);
    }
    const auto color = terrainLabelColor(label);
    pcl::PointXYZRGB point;
    point.x = source.x;
    point.y = source.y;
    point.z = source.z;
    point.r = color[0];
    point.g = color[1];
    point.b = color[2];
    colored.push_back(point);
  }
  colored.width = static_cast<std::uint32_t>(colored.size());
  colored.height = 1U;
  colored.is_dense = cloud->is_dense;

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(colored, msg);
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  publisher->publish(msg);
}

void ScanToMapIcp::publishTerrainSupportCloud(
  const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & publisher,
  const TerrainSupportVectorConstPtr & evidence,
  const std::string & frame_id,
  const rclcpp::Time & stamp) const
{
  if (!publisher || !evidence) {
    return;
  }

  sensor_msgs::msg::PointCloud2 msg;
  msg.header.frame_id = frame_id;
  msg.header.stamp = stamp;
  msg.height = 1U;
  msg.is_bigendian = false;
  msg.is_dense = true;
  sensor_msgs::PointCloud2Modifier modifier(msg);
  modifier.setPointCloud2Fields(
    13,
    "x", 1, sensor_msgs::msg::PointField::FLOAT32,
    "y", 1, sensor_msgs::msg::PointField::FLOAT32,
    "z", 1, sensor_msgs::msg::PointField::FLOAT32,
    "rgb", 1, sensor_msgs::msg::PointField::FLOAT32,
    "support_probability", 1, sensor_msgs::msg::PointField::FLOAT32,
    "geometric_traversability_score", 1, sensor_msgs::msg::PointField::FLOAT32,
    "label_confidence", 1, sensor_msgs::msg::PointField::FLOAT32,
    "roughness_score", 1, sensor_msgs::msg::PointField::FLOAT32,
    "label", 1, sensor_msgs::msg::PointField::UINT8,
    "state", 1, sensor_msgs::msg::PointField::UINT8,
    "connectivity_rejected", 1, sensor_msgs::msg::PointField::UINT8,
    "confirmations", 1, sensor_msgs::msg::PointField::UINT32,
    "observations", 1, sensor_msgs::msg::PointField::UINT32);
  modifier.resize(evidence->size());

  sensor_msgs::PointCloud2Iterator<float> x(msg, "x");
  sensor_msgs::PointCloud2Iterator<float> y(msg, "y");
  sensor_msgs::PointCloud2Iterator<float> z(msg, "z");
  sensor_msgs::PointCloud2Iterator<float> rgb(msg, "rgb");
  sensor_msgs::PointCloud2Iterator<float> support_probability(msg, "support_probability");
  sensor_msgs::PointCloud2Iterator<float> geometric_traversability_score(
    msg, "geometric_traversability_score");
  sensor_msgs::PointCloud2Iterator<float> label_confidence(msg, "label_confidence");
  sensor_msgs::PointCloud2Iterator<float> roughness_score(msg, "roughness_score");
  sensor_msgs::PointCloud2Iterator<std::uint8_t> label(msg, "label");
  sensor_msgs::PointCloud2Iterator<std::uint8_t> state(msg, "state");
  sensor_msgs::PointCloud2Iterator<std::uint8_t> connectivity_rejected(
    msg, "connectivity_rejected");
  sensor_msgs::PointCloud2Iterator<std::uint32_t> confirmations(msg, "confirmations");
  sensor_msgs::PointCloud2Iterator<std::uint32_t> observations(msg, "observations");
  for (const auto & item : *evidence) {
    *x = static_cast<float>(item.position.x());
    *y = static_cast<float>(item.position.y());
    *z = static_cast<float>(item.position.z());
    const auto color = terrainSupportColor(item.label, item.state);
    const std::uint32_t packed_rgb =
      (static_cast<std::uint32_t>(color[0]) << 16U) |
      (static_cast<std::uint32_t>(color[1]) << 8U) |
      static_cast<std::uint32_t>(color[2]);
    float packed_rgb_float = 0.0F;
    std::memcpy(&packed_rgb_float, &packed_rgb, sizeof(packed_rgb_float));
    *rgb = packed_rgb_float;
    *support_probability = static_cast<float>(item.support_probability);
    *geometric_traversability_score =
      static_cast<float>(item.geometric_traversability_score);
    *label_confidence = static_cast<float>(item.label_confidence);
    *roughness_score = static_cast<float>(item.geometric_roughness_score);
    *label = static_cast<std::uint8_t>(item.label);
    *state = static_cast<std::uint8_t>(item.state);
    *connectivity_rejected = item.connectivity_rejected ? 1U : 0U;
    *confirmations = static_cast<std::uint32_t>(std::max(0, item.confirmations));
    *observations = static_cast<std::uint32_t>(std::max(0, item.observations));
    ++x;
    ++y;
    ++z;
    ++rgb;
    ++support_probability;
    ++geometric_traversability_score;
    ++label_confidence;
    ++roughness_score;
    ++label;
    ++state;
    ++connectivity_rejected;
    ++confirmations;
    ++observations;
  }
  publisher->publish(msg);
}

void ScanToMapIcp::publishDiagnostics(
  const rclcpp::Time & stamp,
  const RegistrationResult & result)
{
  // Rejected scans do not pass through publishState(). Mark their omitted
  // motion here as well; accepted samples are idempotently ignored by stamp.
  recordTerrainTraversalSample(stamp, result);

  diagnostic_msgs::msg::DiagnosticArray array;
  array.header.stamp = stamp;

  diagnostic_msgs::msg::DiagnosticStatus status;
  status.name = "scan_to_map_icp";
  status.hardware_id = "slam_3d_core";
  const bool degraded_sensor_input =
    result.input_clouds_merged < result.input_clouds_expected ||
    result.inter_lidar_motion_dropped_count > 0 ||
    result.inter_lidar_motion_applied_count < result.inter_lidar_motion_attempted_count ||
    result.cloud_sync_wait_timed_out ||
    result.inter_lidar_wheel_wait_timed_out ||
    result.deskew_imu_wait_timed_out;
  const bool shock_degraded =
    result.imu_shock_active || result.shock_quarantined || result.pose_quarantined;
  status.level =
    result.accepted && !degraded_sensor_input && !shock_degraded ?
    diagnostic_msgs::msg::DiagnosticStatus::OK :
    diagnostic_msgs::msg::DiagnosticStatus::WARN;
  if (!result.accepted) {
    status.message = "registration rejected";
  } else if (degraded_sensor_input) {
    status.message = "registration accepted with degraded sensor input";
  } else if (shock_degraded) {
    status.message = "registration accepted during IMU shock quarantine";
  } else {
    status.message = "registration accepted";
  }

  rclcpp::Time latest_wheel_stamp;
  rclcpp::Time last_used_wheel_stamp;
  rclcpp::Time wheel_prediction_query_stamp;
  rclcpp::Time wheel_prediction_lower_stamp;
  rclcpp::Time wheel_prediction_upper_stamp;
  std::string wheel_prediction_status;
  double wheel_prediction_max_bracket_delta_s = 0.0;
  std::size_t wheel_odom_buffer_size = 0;
  std::uint64_t wheel_odom_duplicate_count = 0;
  std::uint64_t wheel_odom_out_of_order_count = 0;
  std::uint64_t wheel_odom_frame_rejected_count = 0;
  std::uint64_t wheel_odom_invalid_pose_rejected_count = 0;
  std::uint64_t wheel_odom_timestamp_rewind_rejected_count = 0;
  {
    std::lock_guard<std::mutex> lock(wheel_mutex_);
    latest_wheel_stamp = latest_wheel_stamp_;
    last_used_wheel_stamp = last_used_wheel_stamp_;
    wheel_prediction_query_stamp = wheel_prediction_query_stamp_;
    wheel_prediction_lower_stamp = wheel_prediction_lower_stamp_;
    wheel_prediction_upper_stamp = wheel_prediction_upper_stamp_;
    wheel_prediction_status = wheel_prediction_status_;
    wheel_prediction_max_bracket_delta_s =
      wheel_prediction_max_bracket_delta_s_;
    wheel_odom_buffer_size = wheel_odom_buffer_.size();
    wheel_odom_duplicate_count = wheel_odom_duplicate_count_;
    wheel_odom_out_of_order_count = wheel_odom_out_of_order_count_;
    wheel_odom_frame_rejected_count = wheel_odom_frame_rejected_count_;
    wheel_odom_invalid_pose_rejected_count = wheel_odom_invalid_pose_rejected_count_;
    wheel_odom_timestamp_rewind_rejected_count =
      wheel_odom_timestamp_rewind_rejected_count_;
  }

  bool have_gravity_accel = false;
  Eigen::Vector3d gravity_accel = Eigen::Vector3d::UnitZ();
  double gravity_filter_alpha = 0.0;
  int imu_duplicate_dropped_count = 0;
  int imu_out_of_order_dropped_count = 0;
  int imu_rewind_dropped_count = 0;
  int imu_timestamp_reset_count = 0;
  int imu_frame_rejected_count = 0;
  int imu_nonfinite_rejected_count = 0;
  std::uint64_t gravity_shock_sample_rejected_count = 0;
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
    imu_frame_rejected_count = imu_frame_rejected_count_;
    imu_nonfinite_rejected_count = imu_nonfinite_rejected_count_;
    gravity_shock_sample_rejected_count = gravity_shock_sample_rejected_count_;
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
  add_value("registration_method", registration_method_);
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
  std::ostringstream surfel_eigenvalues;
  for (std::size_t index = 0; index < result.surfel_hessian_eigenvalues.size(); ++index) {
    if (index > 0) {
      surfel_eigenvalues << ",";
    }
    surfel_eigenvalues << result.surfel_hessian_eigenvalues[index];
  }
  add_value("surfel_converged", result.surfel_converged ? "true" : "false");
  add_value("surfel_reason", result.surfel_reason);
  add_value("surfel_target_voxels", toString(result.surfel_target_voxels));
  add_value("surfel_planar_surfels", toString(result.surfel_planar_surfels));
  add_value("surfel_correspondences", toString(result.surfel_correspondences));
  add_value(
    "surfel_correspondence_ratio",
    toString(result.surfel_correspondence_ratio));
  add_value("surfel_rmse_m", toString(result.surfel_rmse_m));
  add_value(
    "surfel_median_abs_residual_m",
    toString(result.surfel_median_abs_residual_m));
  add_value(
    "surfel_p95_abs_residual_m",
    toString(result.surfel_p95_abs_residual_m));
  add_value("surfel_robust_cost", toString(result.surfel_robust_cost));
  add_value(
    "surfel_robust_downweight_ratio",
    toString(result.surfel_robust_downweight_ratio));
  add_value("surfel_observable_rank", toString(result.surfel_observable_rank));
  add_value(
    "surfel_hessian_eigenvalues",
    surfel_eigenvalues.str());
  add_value("surfel_hessian_condition", toString(result.surfel_hessian_condition));
  add_value(
    "registration_information_valid",
    result.registration_information_valid ? "true" : "false");
  add_value("registration_information_status", result.registration_information_status);
  add_value(
    "registration_information_observable_rank",
    toString(result.registration_information_observable_rank));
  add_value(
    "registration_information_hessian_condition",
    toString(result.registration_information_hessian_condition));
  add_value(
    "registration_information_evaluation_runtime_ms",
    toString(result.registration_information_evaluation_runtime_ms));
  add_value(
    "surfel_weak_rotation_x",
    toString(result.surfel_weak_rotation_axes.x()));
  add_value(
    "surfel_weak_rotation_y",
    toString(result.surfel_weak_rotation_axes.y()));
  add_value(
    "surfel_weak_rotation_z",
    toString(result.surfel_weak_rotation_axes.z()));
  add_value(
    "surfel_weak_translation_x",
    toString(result.surfel_weak_translation_axes.x()));
  add_value(
    "surfel_weak_translation_y",
    toString(result.surfel_weak_translation_axes.y()));
  add_value(
    "surfel_weak_translation_z",
    toString(result.surfel_weak_translation_axes.z()));
  add_value(
    "surfel_target_build_time_ms",
    toString(result.surfel_target_build_time_ms));
  add_value(
    "surfel_optimization_time_ms",
    toString(result.surfel_optimization_time_ms));
  add_value("surfel_total_time_ms", toString(result.surfel_total_time_ms));
  add_value("correction_translation_m", toString(result.correction_translation_m));
  add_value("correction_rotation_deg", toString(result.correction_rotation_deg));
  add_value("registration_runtime_ms", toString(result.registration_runtime_ms));
  add_value("registration_scan_gap_s", toString(result.registration_scan_gap_s));
  add_value(
    "registration_translation_gate_m",
    toString(result.registration_translation_gate_m));
  add_value(
    "registration_gap_relaxation_applied",
    result.registration_gap_relaxation_applied ? "true" : "false");
  add_value(
    "registration_gap_relaxation_enabled",
    registration_gap_relaxation_enabled_ ? "true" : "false");
  add_value(
    "registration_gap_relaxation_min_gap_s",
    toString(registration_gap_relaxation_min_gap_s_));
  add_value(
    "registration_gap_relaxation_max_speed_mps",
    toString(registration_gap_relaxation_max_speed_mps_));
  add_value(
    "registration_gap_relaxation_max_translation_m",
    toString(registration_gap_relaxation_max_translation_m_));
  add_value(
    "registration_gap_relaxation_min_inlier_ratio",
    toString(registration_gap_relaxation_min_inlier_ratio_));
  add_value(
    "registration_gap_relaxation_max_inlier_rmse_m",
    toString(registration_gap_relaxation_max_inlier_rmse_m_));
  add_value("transform_determinant", toString(result.transform_determinant));
  add_value("transform_orthogonality_error", toString(result.transform_orthogonality_error));
  add_value("iterations", toString(result.iterations));
  add_value("input_clouds_expected", toString(result.input_clouds_expected));
  add_value("input_clouds_merged", toString(result.input_clouds_merged));
  add_value(
    "secondary_cloud_role",
    auxiliary_cloud_inputs_[0].registration_only ? "registration_only" : "mapping");
  add_value(
    "tertiary_cloud_role",
    auxiliary_cloud_inputs_[1].registration_only ? "registration_only" : "mapping");
  add_value(
    "registration_only_clouds_merged",
    toString(result.registration_only_clouds_merged));
  add_value(
    "registration_only_points_before_balance",
    toString(result.registration_only_points_before_balance));
  add_value(
    "registration_only_points_after_balance",
    toString(result.registration_only_points_after_balance));
  add_value(
    "registration_only_cloud_voxel_leaf_m",
    toString(registration_only_cloud_voxel_leaf_m_));
  add_value(
    "registration_only_cloud_max_point_ratio",
    toString(registration_only_cloud_max_point_ratio_));
  add_value(
    "registration_only_cloud_max_points",
    toString(registration_only_cloud_max_points_));
  add_value(
    "cloud_sync_resolution",
    result.cloud_sync_wait_timed_out ? "wall_timeout" :
    (result.cloud_sync_watermark_resolved ? "timestamp_watermark" : "all_inputs_matched"));
  add_value(
    "cloud_sync_wait_timed_out",
    result.cloud_sync_wait_timed_out ? "true" : "false");
  add_value(
    "primary_cloud_gap_before",
    result.primary_cloud_gap_before ? "true" : "false");
  add_value(
    "cloud_sync_watermark_resolved",
    result.cloud_sync_watermark_resolved ? "true" : "false");
  add_value(
    "inter_lidar_wheel_wait_timed_out",
    result.inter_lidar_wheel_wait_timed_out ? "true" : "false");
  add_value(
    "max_input_cloud_stamp_delta_s",
    toString(result.max_input_cloud_stamp_delta_s));
  add_value("scan_points_before_downsample", toString(result.scan_points_before_downsample));
  add_value("scan_points_after_downsample", toString(result.scan_points_after_downsample));
  add_value(
    "mapping_scan_points_before_downsample",
    toString(result.mapping_scan_points_before_downsample));
  add_value(
    "mapping_scan_points_after_downsample",
    toString(result.mapping_scan_points_after_downsample));
  add_value("scan_voxel_leaf_m", toString(result.scan_voxel_leaf_m));
  add_value("scan_voxel_adapted", result.scan_voxel_adapted ? "true" : "false");
  add_value("adaptive_scan_voxel", adaptive_scan_voxel_enabled_ ? "enabled" : "disabled");
  add_value("deskew", deskew_enabled_ ? "enabled" : "disabled");
  add_value(
    "deskew_failure_policy",
    reject_cloud_on_deskew_failure_ ? "reject_scan" : "keep_undeskewed");
  add_value("deskew_attempted", result.deskew_attempted ? "true" : "false");
  add_value("deskew_applied", result.deskew_applied ? "true" : "false");
  add_value("deskew_status", result.deskew_status);
  add_value("deskew_point_time_field", result.deskew_point_time_field);
  add_value("deskew_points_with_time", toString(result.deskew_points_with_time));
  add_value("deskew_point_count", toString(result.deskew_point_count));
  add_value("deskew_time_ratio", toString(result.deskew_time_ratio));
  add_value("deskew_scan_duration_s", toString(result.deskew_scan_duration_s));
  add_value(
    "deskew_wait_for_imu_watermark",
    deskew_wait_for_imu_watermark_ ? "enabled" : "disabled");
  add_value(
    "deskew_imu_watermark_waited",
    result.deskew_imu_watermark_waited ? "true" : "false");
  add_value(
    "deskew_imu_wait_timed_out",
    result.deskew_imu_wait_timed_out ? "true" : "false");
  add_value(
    "deskew_required_imu_stamp_ns",
    toString(result.deskew_required_imu_stamp_ns));
  add_value(
    "deskew_dispatch_imu_stamp_ns",
    toString(result.deskew_dispatch_imu_stamp_ns));
  add_value(
    "imu_snapshot_end_stamp_ns",
    toString(result.imu_snapshot_end_stamp_ns));
  add_value(
    "imu_snapshot_sample_count",
    toString(result.imu_snapshot_sample_count));
  add_value(
    "deskew_imu_wait_timeout_s",
    toString(deskew_imu_wait_timeout_s_));
  add_value(
    "inter_lidar_motion_compensation",
    inter_lidar_motion_compensation_enabled_ ? "enabled" : "disabled");
  add_value(
    "inter_lidar_motion_compensation_source",
    inter_lidar_motion_compensation_source_);
  add_value(
    "inter_lidar_gap_recovery_hypothesis",
    inter_lidar_gap_recovery_hypothesis_enabled_ ? "enabled" : "disabled");
  add_value("inter_lidar_motion_status", result.inter_lidar_motion_status);
  add_value(
    "inter_lidar_motion_attempted_count",
    toString(result.inter_lidar_motion_attempted_count));
  add_value(
    "inter_lidar_motion_applied_count",
    toString(result.inter_lidar_motion_applied_count));
  add_value(
    "inter_lidar_motion_dropped_count",
    toString(result.inter_lidar_motion_dropped_count));
  add_value(
    "inter_lidar_motion_failure_policy",
    inter_lidar_motion_failure_policy_);
  add_value(
    "inter_lidar_motion_max_stamp_delta_s",
    toString(result.inter_lidar_motion_max_stamp_delta_s));
  add_value(
    "inter_lidar_motion_max_translation_m",
    toString(result.inter_lidar_motion_max_translation_m));
  add_value(
    "inter_lidar_motion_max_rotation_deg",
    toString(result.inter_lidar_motion_max_rotation_deg));
  add_value(
    "inter_lidar_gap_recovery_hypothesis_attempted",
    result.inter_lidar_gap_recovery_hypothesis_attempted ? "true" : "false");
  add_value(
    "inter_lidar_gap_recovery_selected_uncompensated",
    result.inter_lidar_gap_recovery_selected_uncompensated ? "true" : "false");
  add_value(
    "inter_lidar_gap_recovery_selected_input",
    result.inter_lidar_gap_recovery_selected_input);
  add_value(
    "inter_lidar_gap_recovery_compensated_score",
    toString(result.inter_lidar_gap_recovery_compensated_score));
  add_value(
    "inter_lidar_gap_recovery_uncompensated_score",
    toString(result.inter_lidar_gap_recovery_uncompensated_score));
  add_value(
    "inter_lidar_gap_recovery_uncompensated_fitness",
    toString(result.inter_lidar_gap_recovery_uncompensated_fitness));
  add_value(
    "inter_lidar_gap_recovery_uncompensated_inlier_ratio",
    toString(result.inter_lidar_gap_recovery_uncompensated_inlier_ratio));
  add_value(
    "inter_lidar_gap_recovery_uncompensated_inlier_rmse_m",
    toString(result.inter_lidar_gap_recovery_uncompensated_inlier_rmse_m));
  add_value(
    "inter_lidar_gap_recovery_runtime_ms",
    toString(result.inter_lidar_gap_recovery_runtime_ms));
  add_value(
    "inter_lidar_motion_compensation_max_dt_s",
    toString(inter_lidar_motion_compensation_max_dt_s_));
  add_value(
    "inter_lidar_imu_rotation_scale",
    toString(inter_lidar_imu_rotation_scale_));
  add_value(
    "inter_lidar_gap_recovery_hypothesis_min_gap_s",
    toString(inter_lidar_gap_recovery_hypothesis_min_gap_s_));
  add_value(
    "inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio",
    toString(inter_lidar_gap_recovery_hypothesis_min_score_improvement_ratio_));
  add_value(
    "imu_roughness_detection",
    imu_roughness_detection_enabled_ ? "enabled" : "disabled");
  add_value(
    "imu_shock_policy",
    imu_shock_policy_enabled_ ? "enabled" : "disabled");
  add_value(
    "imu_roughness_evaluated",
    result.imu_roughness_evaluated ? "true" : "false");
  add_value("imu_roughness_event", result.imu_roughness_event ? "true" : "false");
  add_value(
    "imu_roughness_stable_evidence",
    result.imu_roughness_stable_evidence ? "true" : "false");
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
  add_value("imu_shock_phase", result.imu_shock_phase);
  add_value("imu_shock_observation", result.imu_shock_observation);
  add_value("imu_shock_active", result.imu_shock_active ? "true" : "false");
  add_value(
    "omit_shock_poses_from_path",
    omit_shock_poses_from_path_ ? "enabled" : "disabled");
  add_value("path_pose_appended", result.path_pose_appended ? "true" : "false");
  add_value(
    "imu_shock_time_reset",
    result.imu_shock_time_reset ? "true" : "false");
  add_value(
    "imu_shock_just_recovered",
    result.imu_shock_just_recovered ? "true" : "false");
  add_value(
    "imu_shock_time_since_event_s",
    toString(result.imu_shock_time_since_event_s));
  add_value(
    "imu_shock_recovery_stable_scan_count",
    toString(result.imu_shock_recovery_stable_scan_count));
  add_value("imu_shock_hold_duration_s", toString(imu_shock_hold_duration_s_));
  add_value(
    "imu_shock_recovery_stable_scans",
    toString(imu_shock_recovery_stable_scans_));
  add_value(
    "constant_velocity_prediction_suppressed_by_shock",
    result.constant_velocity_prediction_suppressed_by_shock ? "true" : "false");
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
    "record_tracking_scan_during_shock",
    record_tracking_scan_during_shock_ ? "enabled" : "disabled");
  add_value(
    "tracking_scan_updated_during_shock",
    result.tracking_scan_updated_during_shock ? "true" : "false");
  add_value(
    "tracking_scan_updated_after_rejection",
    result.tracking_scan_updated_after_rejection ? "true" : "false");
  add_value(
    "pose_quarantine_tracking_reset_interval",
    toString(pose_quarantine_tracking_reset_interval_));
  add_value("keyframe_added", result.keyframe_added ? "true" : "false");
  add_value(
    "local_map_spatial_transition_bootstrap",
    result.local_map_spatial_transition_bootstrap ? "true" : "false");
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
    "place_recognition_eligible_descriptors",
    toString(result.place_recognition_eligible_descriptors));
  add_value(
    "place_recognition_registration_only_vertical_recovery_descriptors",
    toString(result.place_recognition_registration_only_vertical_recovery_descriptors));
  add_value(
    "place_recognition_registration_only_vertical_recovery_used",
    result.place_recognition_registration_only_vertical_recovery_used ? "true" : "false");
  add_value(
    "place_recognition_registration_only_consistency_relaxation_used",
    result.place_recognition_registration_only_consistency_relaxation_used ? "true" : "false");
  add_value(
    "place_recognition_mapping_cloud_refinement",
    place_recognition_mapping_cloud_refinement_enabled_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_mapping_refinement_required_for_pose_graph",
    place_recognition_mapping_refinement_required_for_pose_graph_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_mapping_refinement_attempted",
    result.place_recognition_mapping_refinement_attempted ? "true" : "false");
  add_value(
    "place_recognition_mapping_refinement_applied",
    result.place_recognition_mapping_refinement_applied ? "true" : "false");
  add_value(
    "place_recognition_mapping_refinement_verified",
    result.place_recognition_mapping_refinement_verified ? "true" : "false");
  add_value(
    "place_recognition_large_pose_jump_provisional",
    result.place_recognition_large_pose_jump_provisional ? "true" : "false");
  add_value(
    "place_recognition_large_pose_jump_accepted",
    result.place_recognition_large_pose_jump_accepted ? "true" : "false");
  add_value(
    "place_recognition_verified_large_loop_measurement_preserved",
    result.place_recognition_verified_large_loop_measurement_preserved ? "true" : "false");
  add_value(
    "place_recognition_mapping_refinement_status",
    result.place_recognition_mapping_refinement_status);
  add_value(
    "place_recognition_mapping_refinement_source_points",
    toString(result.place_recognition_mapping_refinement_source_points));
  add_value(
    "place_recognition_mapping_refinement_target_points",
    toString(result.place_recognition_mapping_refinement_target_points));
  add_value(
    "place_recognition_mapping_refinement_before_inlier_ratio",
    toString(result.place_recognition_mapping_refinement_before_inlier_ratio));
  add_value(
    "place_recognition_mapping_refinement_before_inlier_rmse_m",
    toString(result.place_recognition_mapping_refinement_before_inlier_rmse_m));
  add_value(
    "place_recognition_mapping_refinement_after_inlier_ratio",
    toString(result.place_recognition_mapping_refinement_after_inlier_ratio));
  add_value(
    "place_recognition_mapping_refinement_after_inlier_rmse_m",
    toString(result.place_recognition_mapping_refinement_after_inlier_rmse_m));
  add_value(
    "place_recognition_mapping_refinement_before_target_coverage_ratio",
    toString(result.place_recognition_mapping_refinement_before_target_coverage_ratio));
  add_value(
    "place_recognition_mapping_refinement_after_target_coverage_ratio",
    toString(result.place_recognition_mapping_refinement_after_target_coverage_ratio));
  add_value(
    "place_recognition_mapping_refinement_before_score",
    toString(result.place_recognition_mapping_refinement_before_score));
  add_value(
    "place_recognition_mapping_refinement_after_score",
    toString(result.place_recognition_mapping_refinement_after_score));
  add_value(
    "place_recognition_mapping_refinement_primary_before_score",
    toString(result.place_recognition_mapping_refinement_primary_before_score));
  add_value(
    "place_recognition_mapping_refinement_primary_after_score",
    toString(result.place_recognition_mapping_refinement_primary_after_score));
  add_value(
    "place_recognition_mapping_refinement_correction_translation_m",
    toString(result.place_recognition_mapping_refinement_correction_translation_m));
  add_value(
    "place_recognition_mapping_refinement_correction_rotation_deg",
    toString(result.place_recognition_mapping_refinement_correction_rotation_deg));
  add_value(
    "place_recognition_mapping_refinement_observable_rank",
    toString(result.place_recognition_mapping_refinement_observable_rank));
  add_value(
    "place_recognition_mapping_refinement_hessian_condition",
    toString(result.place_recognition_mapping_refinement_hessian_condition));
  add_value(
    "place_recognition_mapping_refinement_runtime_ms",
    toString(result.place_recognition_mapping_refinement_runtime_ms));
  add_value(
    "place_recognition_candidate_index",
    toString(result.place_recognition_candidate_index));
  add_value(
    "place_recognition_candidate_submap_id",
    toString(result.place_recognition_candidate_submap_id));
  add_value(
    "place_recognition_candidate_registration_status",
    result.place_recognition_candidate_registration_status);
  add_value(
    "place_recognition_candidate_registration_inlier_ratio",
    toString(result.place_recognition_candidate_registration_inlier_ratio));
  add_value(
    "place_recognition_candidate_registration_inlier_rmse_m",
    toString(result.place_recognition_candidate_registration_inlier_rmse_m));
  add_value(
    "place_recognition_candidate_pose_jump_translation_m",
    toString(result.place_recognition_candidate_pose_jump_translation_m));
  add_value(
    "place_recognition_candidate_pose_jump_rotation_deg",
    toString(result.place_recognition_candidate_pose_jump_rotation_deg));
  add_value(
    "pose_graph_same_layer_z_constraint_evaluated",
    result.pose_graph_same_layer_z_constraint_evaluated ? "true" : "false");
  add_value(
    "pose_graph_same_layer_z_constraint_applied",
    result.pose_graph_same_layer_z_constraint_applied ? "true" : "false");
  add_value(
    "pose_graph_same_layer_reference_source",
    result.pose_graph_same_layer_reference_source);
  add_value(
    "pose_graph_same_layer_reference_keyframe_index",
    toString(result.pose_graph_same_layer_reference_keyframe_index));
  add_value(
    "pose_graph_same_layer_reference_xy_distance_m",
    toString(result.pose_graph_same_layer_reference_xy_distance_m));
  add_value(
    "pose_graph_same_layer_matched_current_z_before_m",
    toString(result.pose_graph_same_layer_matched_current_z_before_m));
  add_value(
    "pose_graph_same_layer_reference_z_m",
    toString(result.pose_graph_same_layer_reference_z_m));
  add_value(
    "pose_graph_same_layer_requested_z_correction_m",
    toString(result.pose_graph_same_layer_requested_z_correction_m));
  add_value(
    "pose_graph_same_layer_graph_target_z_m",
    toString(result.pose_graph_same_layer_graph_target_z_m));
  add_value(
    "pose_graph_same_layer_z_translation_weight",
    toString(result.pose_graph_same_layer_z_translation_weight));
  add_value(
    "place_recognition_candidate_consistency_count",
    toString(result.place_recognition_candidate_consistency_count));
  add_value(
    "place_recognition_candidate_consistency_translation_m",
    toString(result.place_recognition_candidate_consistency_translation_m));
  add_value(
    "place_recognition_candidate_consistency_rotation_deg",
    toString(result.place_recognition_candidate_consistency_rotation_deg));
  add_value(
    "place_recognition_candidate_consistency_translation_limit_m",
    toString(result.place_recognition_candidate_consistency_translation_limit_m));
  add_value(
    "place_recognition_use_submaps",
    place_recognition_use_submaps_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_submap_min_consecutive_matches",
    toString(place_recognition_submap_min_consecutive_matches_));
  add_value(
    "place_recognition_mapping_refinement_quality_max_correspondence_m",
    toString(place_recognition_mapping_refinement_quality_max_correspondence_m_));
  add_value(
    "place_recognition_mapping_refinement_min_source_points",
    toString(mapping_cloud_refinement_config_.min_source_points));
  add_value(
    "place_recognition_mapping_refinement_min_target_points",
    toString(mapping_cloud_refinement_config_.min_target_points));
  add_value(
    "place_recognition_mapping_refinement_min_target_keyframe_observations",
    toString(place_recognition_mapping_refinement_min_target_keyframe_observations_));
  add_value(
    "place_recognition_mapping_refinement_target_confirmation_voxel_m",
    toString(place_recognition_mapping_refinement_target_confirmation_voxel_m_));
  add_value(
    "place_recognition_mapping_refinement_min_inlier_ratio",
    toString(mapping_cloud_refinement_config_.min_inlier_ratio));
  add_value(
    "place_recognition_mapping_refinement_min_target_coverage_ratio",
    toString(mapping_cloud_refinement_config_.min_target_coverage_ratio));
  add_value(
    "place_recognition_mapping_refinement_max_inlier_rmse_m",
    toString(mapping_cloud_refinement_config_.max_inlier_rmse_m));
  add_value(
    "place_recognition_mapping_refinement_max_translation_m",
    toString(mapping_cloud_refinement_config_.max_translation_m));
  add_value(
    "place_recognition_mapping_refinement_max_rotation_deg",
    toString(mapping_cloud_refinement_config_.max_rotation_deg));
  add_value(
    "place_recognition_mapping_refinement_min_observable_rank",
    toString(mapping_cloud_refinement_config_.min_observable_rank));
  add_value(
    "place_recognition_mapping_refinement_max_hessian_condition",
    toString(mapping_cloud_refinement_config_.max_hessian_condition));
  add_value(
    "place_recognition_mapping_refinement_min_score_improvement_ratio",
    toString(mapping_cloud_refinement_config_.min_score_improvement_ratio));
  add_value(
    "place_recognition_mapping_refinement_max_primary_score_regression_ratio",
    toString(mapping_cloud_refinement_config_.max_primary_score_regression_ratio));
  add_value(
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m",
    toString(
      place_recognition_mapping_refinement_keyframe_quarantine_correction_translation_m_));
  add_value(
    "place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg",
    toString(
      place_recognition_mapping_refinement_keyframe_quarantine_correction_rotation_deg_));
  add_value(
    "place_recognition_max_vertical_separation_m",
    toString(place_recognition_max_vertical_separation_m_));
  add_value(
    "place_recognition_registration_only_vertical_recovery_m",
    toString(place_recognition_registration_only_vertical_recovery_m_));
  add_value(
    "place_recognition_registration_only_max_pose_jump_m",
    toString(place_recognition_registration_only_max_pose_jump_m_));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump",
    place_recognition_mapping_refinement_large_pose_jump_enabled_ ? "enabled" : "disabled");
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_max_translation_m",
    toString(place_recognition_mapping_refinement_large_pose_jump_max_translation_m_));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg",
    toString(place_recognition_mapping_refinement_large_pose_jump_max_rotation_deg_));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance",
    toString(place_recognition_mapping_refinement_large_pose_jump_max_descriptor_distance_));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse",
    mapping_cloud_verified_coarse_config_.enabled ? "enabled" : "disabled");
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_inlier_ratio",
    toString(mapping_cloud_verified_coarse_config_.min_inlier_ratio));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_inlier_rmse_m",
    toString(mapping_cloud_verified_coarse_config_.max_inlier_rmse_m));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_min_target_coverage_ratio",
    toString(mapping_cloud_verified_coarse_config_.min_target_coverage_ratio));
  add_value(
    "place_recognition_mapping_refinement_large_pose_jump_verified_coarse_max_score_regression_ratio",
    toString(mapping_cloud_verified_coarse_config_.max_score_regression_ratio));
  add_value(
    "place_recognition_submap_consistency_max_translation_m",
    toString(place_recognition_submap_consistency_max_translation_m_));
  add_value(
    "place_recognition_registration_only_consistency_max_translation_m",
    toString(place_recognition_registration_only_consistency_max_translation_m_));
  add_value(
    "place_recognition_registration_only_consistency_max_gap_s",
    toString(place_recognition_registration_only_consistency_max_gap_s_));
  add_value(
    "place_recognition_submap_consistency_max_rotation_deg",
    toString(place_recognition_submap_consistency_max_rotation_deg_));
  add_value(
    "place_recognition_descriptor_distance",
    toString(result.place_recognition_descriptor_distance));
  add_value(
    "place_recognition_nearest_descriptor_distance",
    toString(result.place_recognition_nearest_descriptor_distance));
  add_value(
    "place_recognition_max_descriptor_distance",
    toString(place_recognition_max_descriptor_distance_));
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
    "place_recognition_alignment_correction_translation_m",
    toString(result.place_recognition_alignment_correction_translation_m));
  add_value(
    "place_recognition_alignment_correction_rotation_deg",
    toString(result.place_recognition_alignment_correction_rotation_deg));
  add_value(
    "pose_graph_correction",
    pose_graph_correction_enabled_ ? "enabled" : "disabled");
  add_value("pose_graph_solver", pose_graph_solver_);
  add_value("pose_graph_information_mode", pose_graph_information_mode_);
  add_value(
    "pose_graph_information_relative_eigen_threshold",
    toString(pose_graph_information_config_.relative_eigen_threshold));
  add_value(
    "pose_graph_information_strong_eigenvalue_ratio",
    toString(pose_graph_information_config_.strong_eigenvalue_ratio));
  add_value(
    "pose_graph_information_weak_ratio",
    toString(pose_graph_information_config_.weak_information_ratio));
  add_value(
    "pose_graph_planar_correction",
    pose_graph_planar_correction_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_robust_loop",
    pose_graph_robust_loop_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_distributed_loop_seed",
    pose_graph_distributed_loop_seed_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_minimum_interval_override",
    pose_graph_minimum_interval_override_config_.enabled ? "enabled" : "disabled");
  add_value(
    "pose_graph_duplicate_submap_pair_override",
    pose_graph_duplicate_submap_pair_override_config_.enabled ? "enabled" : "disabled");
  add_value(
    "pose_graph_duplicate_submap_pair_override_max_edges",
    toString(pose_graph_duplicate_submap_pair_override_config_.max_edges));
  add_value(
    "pose_graph_minimum_interval_override_max_descriptor_distance",
    toString(pose_graph_minimum_interval_override_config_.max_descriptor_distance));
  add_value(
    "pose_graph_minimum_interval_override_min_inlier_ratio",
    toString(pose_graph_minimum_interval_override_config_.min_inlier_ratio));
  add_value(
    "pose_graph_minimum_interval_override_max_inlier_rmse_m",
    toString(pose_graph_minimum_interval_override_config_.max_inlier_rmse_m));
  add_value(
    "pose_graph_minimum_interval_override_min_target_coverage_ratio",
    toString(pose_graph_minimum_interval_override_config_.min_target_coverage_ratio));
  add_value(
    "pose_graph_minimum_interval_override_single_floor_z",
    pose_graph_minimum_interval_override_single_floor_z_enabled_ ? "enabled" : "disabled");
  add_value(
    "pose_graph_minimum_interval_override_single_floor_xy_max_distance_m",
    toString(pose_graph_minimum_interval_override_single_floor_xy_max_distance_m_));
  add_value(
    "pose_graph_minimum_interval_override_single_floor_z_weight_multiplier",
    toString(pose_graph_minimum_interval_override_single_floor_z_weight_multiplier_));
  add_value("pose_graph_loop_cauchy_scale", toString(pose_graph_loop_cauchy_scale_));
  add_value(
    "pose_graph_ceres_robust_stage_count",
    toString(pose_graph_optimizer_config_.robust_scale_factors.size()));
  add_value(
    "pose_graph_ceres_max_iterations_per_stage",
    toString(pose_graph_optimizer_config_.max_iterations_per_stage));
  add_value(
    "pose_graph_ceres_dcs",
    pose_graph_optimizer_config_.dcs_enabled ? "enabled" : "disabled");
  add_value(
    "pose_graph_ceres_dcs_start_stage",
    toString(pose_graph_optimizer_config_.dcs_start_stage));
  add_value("pose_graph_ceres_dcs_phi", toString(pose_graph_optimizer_config_.dcs_phi));
  add_value(
    "pose_graph_ceres_dcs_min_scale",
    toString(pose_graph_optimizer_config_.dcs_min_scale));
  add_value(
    "pose_graph_min_new_loop_robust_weight",
    toString(pose_graph_min_new_loop_robust_weight_));
  add_value(
    "planar_pose_constraint",
    planar_pose_constraint_enabled_ ? "enabled" : "disabled");
  add_value("planar_pose_max_z_correction_m", toString(planar_pose_max_z_correction_m_));
  add_value(
    "ground_vehicle_motion_constraint",
    ground_motion_constraint_config_.enabled ? "enabled" : "disabled");
  add_value(
    "ground_vehicle_nominal_max_body_z_step_m",
    toString(ground_motion_constraint_config_.nominal_max_body_z_step_m));
  add_value(
    "ground_vehicle_roughness_max_body_z_step_m",
    toString(ground_motion_constraint_config_.roughness_max_body_z_step_m));
  add_value(
    "ground_motion_constraint_evaluated",
    result.ground_motion_constraint_evaluated ? "true" : "false");
  add_value(
    "ground_motion_constraint_applied",
    result.ground_motion_constraint_applied ? "true" : "false");
  add_value(
    "ground_motion_constraint_roughness_relaxed",
    result.ground_motion_constraint_roughness_relaxed ? "true" : "false");
  add_value(
    "ground_motion_post_constraint_quality_evaluated",
    result.ground_motion_post_constraint_quality_evaluated ? "true" : "false");
  add_value(
    "ground_motion_post_constraint_quality_accepted",
    result.ground_motion_post_constraint_quality_accepted ? "true" : "false");
  add_value(
    "ground_motion_constraint_status",
    result.ground_motion_constraint_status);
  add_value(
    "ground_motion_raw_body_z_step_m",
    toString(result.ground_motion_raw_body_z_step_m));
  add_value(
    "ground_motion_constrained_body_z_step_m",
    toString(result.ground_motion_constrained_body_z_step_m));
  add_value(
    "ground_motion_max_body_z_step_m",
    toString(result.ground_motion_max_body_z_step_m));
  add_value(
    "imu_relative_roll_pitch_constraint",
    imu_relative_roll_pitch_constraint_enabled_ ? "enabled" : "disabled");
  add_value(
    "imu_relative_attitude_max_time_delta_s",
    toString(imu_relative_attitude_max_time_delta_s_));
  add_value(
    "imu_relative_roll_pitch_max_correction_deg",
    toString(imu_relative_attitude_constraint_config_.max_roll_pitch_correction_deg));
  add_value(
    "imu_relative_attitude_reference_available",
    have_initial_imu_world_R_base_ ? "true" : "false");
  add_value(
    "imu_relative_attitude_constraint_evaluated",
    result.imu_relative_attitude_constraint_evaluated ? "true" : "false");
  add_value(
    "imu_relative_attitude_constraint_applied",
    result.imu_relative_attitude_constraint_applied ? "true" : "false");
  add_value(
    "imu_relative_attitude_constraint_status",
    result.imu_relative_attitude_constraint_status);
  add_value(
    "imu_relative_attitude_target_roll_deg",
    toString(result.imu_relative_attitude_target_roll_deg));
  add_value(
    "imu_relative_attitude_target_pitch_deg",
    toString(result.imu_relative_attitude_target_pitch_deg));
  add_value(
    "imu_relative_attitude_raw_roll_error_deg",
    toString(result.imu_relative_attitude_raw_roll_error_deg));
  add_value(
    "imu_relative_attitude_raw_pitch_error_deg",
    toString(result.imu_relative_attitude_raw_pitch_error_deg));
  add_value(
    "imu_relative_attitude_applied_roll_correction_deg",
    toString(result.imu_relative_attitude_applied_roll_correction_deg));
  add_value(
    "imu_relative_attitude_applied_pitch_correction_deg",
    toString(result.imu_relative_attitude_applied_pitch_correction_deg));
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
  add_value("gravity_accel_scale", toString(gravity_accel_scale_));
  add_value("gravity_accel_max_age_s", toString(gravity_accel_max_age_s_));
  add_value("gravity_accel_effective_alpha", toString(result.gravity_filter_alpha));
  add_value("gravity_roll_deg", toString(result.gravity_roll_deg));
  add_value("gravity_pitch_deg", toString(result.gravity_pitch_deg));
  add_value("latest_gravity_accel_available", have_gravity_accel ? "true" : "false");
  add_value("latest_gravity_accel_norm_mps2", toString(gravity_accel_norm));
  add_value("latest_gravity_accel_effective_alpha", toString(gravity_filter_alpha));
  add_value("latest_gravity_roll_deg", toString(gravity_roll_deg));
  add_value("latest_gravity_pitch_deg", toString(gravity_pitch_deg));
  add_value(
    "gravity_shock_sample_rejected_count",
    toString(gravity_shock_sample_rejected_count));
  add_value("map_z_filter", map_z_filter_enabled_ ? "enabled" : "disabled");
  add_value("map_z_filter_min_m", toString(map_z_filter_min_m_));
  add_value("map_z_filter_max_m", toString(map_z_filter_max_m_));
  add_value(
    "pose_graph_correction_applied",
    result.pose_graph_correction_applied ? "true" : "false");
  add_value(
    "pose_graph_new_loop_rejected",
    result.pose_graph_new_loop_rejected ? "true" : "false");
  add_value(
    "pose_graph_high_confidence_large_loop",
    result.pose_graph_high_confidence_large_loop ? "true" : "false");
  add_value(
    "pose_graph_minimum_interval_override_applied",
    result.pose_graph_minimum_interval_override_applied ? "true" : "false");
  add_value(
    "pose_graph_duplicate_submap_pair_override_applied",
    result.pose_graph_duplicate_submap_pair_override_applied ? "true" : "false");
  add_value(
    "pose_graph_duplicate_submap_pair_previous_edges",
    toString(result.pose_graph_duplicate_submap_pair_previous_edges));
  add_value(
    "pose_graph_duplicate_submap_pair_status",
    result.pose_graph_duplicate_submap_pair_status);
  add_value(
    "pose_graph_minimum_interval_elapsed_s",
    toString(result.pose_graph_minimum_interval_elapsed_s));
  add_value(
    "pose_graph_minimum_interval_status",
    result.pose_graph_minimum_interval_status);
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
    "pose_graph_constraint_inlier_ratio",
    toString(result.pose_graph_constraint_inlier_ratio));
  add_value(
    "pose_graph_constraint_inlier_rmse_m",
    toString(result.pose_graph_constraint_inlier_rmse_m));
  add_value(
    "pose_graph_constraint_target_coverage_ratio",
    toString(result.pose_graph_constraint_target_coverage_ratio));
  add_value(
    "pose_graph_optimization_iterations",
    toString(result.pose_graph_optimization_iterations));
  add_value("pose_graph_solver_stages", toString(result.pose_graph_solver_stages));
  add_value(
    "pose_graph_solver_successful_steps",
    toString(result.pose_graph_solver_successful_steps));
  add_value(
    "pose_graph_solver_unsuccessful_steps",
    toString(result.pose_graph_solver_unsuccessful_steps));
  add_value(
    "pose_graph_dcs_downweighted_loop_edges",
    toString(result.pose_graph_dcs_downweighted_loop_edges));
  add_value(
    "pose_graph_registration_information_edges",
    toString(result.pose_graph_registration_information_edges));
  add_value(
    "pose_graph_configured_diagonal_information_edges",
    toString(result.pose_graph_configured_diagonal_information_edges));
  add_value(
    "pose_graph_minimum_registration_observable_rank",
    toString(result.pose_graph_minimum_registration_observable_rank));
  add_value(
    "pose_graph_maximum_registration_hessian_condition",
    toString(result.pose_graph_maximum_registration_hessian_condition));
  add_value(
    "pose_graph_minimum_registration_information_gain",
    toString(result.pose_graph_minimum_registration_information_gain));
  add_value(
    "pose_graph_new_loop_registration_information_used",
    result.pose_graph_new_loop_registration_information_used ? "true" : "false");
  add_value(
    "pose_graph_new_loop_information_status",
    result.pose_graph_new_loop_information_status);
  add_value(
    "pose_graph_new_loop_information_min_gain",
    toString(result.pose_graph_new_loop_information_min_gain));
  add_value("pose_graph_dcs_min_scale", toString(result.pose_graph_dcs_min_scale));
  add_value(
    "pose_graph_new_loop_dcs_scale",
    toString(result.pose_graph_new_loop_dcs_scale));
  add_value(
    "pose_graph_solver_first_stage_initial_cost",
    toString(result.pose_graph_solver_first_stage_initial_cost));
  add_value(
    "pose_graph_solver_last_stage_final_cost",
    toString(result.pose_graph_solver_last_stage_final_cost));
  add_value(
    "pose_graph_solver_final_gradient_max_norm",
    toString(result.pose_graph_solver_final_gradient_max_norm));
  add_value(
    "pose_graph_optimization_runtime_ms",
    toString(result.pose_graph_optimization_runtime_ms));
  add_value(
    "pose_graph_iteration_runtime_ms",
    toString(result.pose_graph_iteration_runtime_ms));
  add_value(
    "pose_graph_mean_iteration_runtime_ms",
    toString(result.pose_graph_mean_iteration_runtime_ms));
  add_value("pose_graph_solver_status", result.pose_graph_solver_status);
  add_value(
    "pose_graph_convergence_reason",
    result.pose_graph_convergence_reason);
  add_value("pose_graph_edges", toString(result.pose_graph_edges));
  add_value("pose_graph_loop_edges", toString(result.pose_graph_loop_edges));
  add_value(
    "pose_graph_robust_loop_edges_downweighted",
    toString(result.pose_graph_robust_loop_edges_downweighted));
  add_value("pose_graph_max_loop_residual_before", toString(pose_graph_max_loop_residual_before_));
  add_value("pose_graph_residual_before", toString(result.pose_graph_residual_before));
  add_value("pose_graph_residual_after", toString(result.pose_graph_residual_after));
  add_value(
    "pose_graph_residual_before_ground_sequence",
    toString(result.pose_graph_residual_before_ground_sequence));
  add_value(
    "pose_graph_robust_objective_before",
    toString(result.pose_graph_robust_objective_before));
  add_value(
    "pose_graph_robust_objective_after",
    toString(result.pose_graph_robust_objective_after));
  add_value(
    "pose_graph_robust_objective_before_ground_sequence",
    toString(result.pose_graph_robust_objective_before_ground_sequence));
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
  add_value(
    "pose_graph_same_layer_z_seed_applied",
    result.pose_graph_same_layer_z_seed_applied ? "true" : "false");
  add_value(
    "pose_graph_same_layer_z_seed_correction_m",
    toString(result.pose_graph_same_layer_z_seed_correction_m));
  add_value(
    "pose_graph_same_layer_z_seed_path_length_m",
    toString(result.pose_graph_same_layer_z_seed_path_length_m));
  add_value(
    "pose_graph_same_layer_z_seed_max_increment_m",
    toString(result.pose_graph_same_layer_z_seed_max_increment_m));
  add_value(
    "pose_graph_distributed_loop_seed_applied",
    result.pose_graph_distributed_loop_seed_applied ? "true" : "false");
  add_value(
    "pose_graph_distributed_loop_seed_path_length_m",
    toString(result.pose_graph_distributed_loop_seed_path_length_m));
  add_value(
    "pose_graph_distributed_loop_seed_endpoint_translation_m",
    toString(result.pose_graph_distributed_loop_seed_endpoint_translation_m));
  add_value(
    "pose_graph_distributed_loop_seed_endpoint_rotation_deg",
    toString(result.pose_graph_distributed_loop_seed_endpoint_rotation_deg));
  add_value(
    "pose_graph_distributed_loop_seed_max_relative_translation_error_m",
    toString(result.pose_graph_distributed_loop_seed_max_relative_translation_error_m));
  add_value(
    "pose_graph_distributed_loop_seed_max_relative_rotation_error_deg",
    toString(result.pose_graph_distributed_loop_seed_max_relative_rotation_error_deg));
  add_value(
    "pose_graph_ground_sequence_applied_edges",
    toString(result.pose_graph_ground_sequence_applied_edges));
  add_value(
    "pose_graph_ground_sequence_max_raw_body_z_step_m",
    toString(result.pose_graph_ground_sequence_max_raw_body_z_step_m));
  add_value(
    "pose_graph_ground_sequence_max_constrained_body_z_step_m",
    toString(result.pose_graph_ground_sequence_max_constrained_body_z_step_m));
  add_value("mapping_spatial_mode", mappingSpatialModeName(spatial_mapping_config_.mode));
  add_value("local_map_radius_m", toString(local_map_radius_m_));
  add_value(
    "local_map_vertical_radius_m",
    toString(local_map_vertical_radius_m_));
  add_value("keyframe_translation_m", toString(keyframe_translation_m_));
  add_value("keyframe_rotation_deg", toString(keyframe_rotation_deg_));
  add_value(
    "keyframe_vertical_translation_m",
    toString(keyframe_vertical_translation_m_));
  add_value("keyframe_rotation_3d_deg", toString(keyframe_rotation_3d_deg_));
  add_value("keyframes", toString(keyframes_.size()));
  add_value("submap_enabled", submap_enabled_ ? "true" : "false");
  add_value("submaps", toString(submaps_.size()));
  add_value("completed_submaps", toString(completedSubmapCount()));
  add_value("frozen_completed_submaps", toString(frozenCompletedSubmapCount()));
  std::size_t accepted_loop_submap_pair_edges = 0U;
  for (const auto & pair_anchors : accepted_loop_submap_pair_anchor_stamps_) {
    accepted_loop_submap_pair_edges += pair_anchors.second.size();
  }
  add_value(
    "accepted_loop_submap_pairs",
    toString(accepted_loop_submap_pair_anchor_stamps_.size()));
  add_value(
    "accepted_loop_submap_pair_edges",
    toString(accepted_loop_submap_pair_edges));
  int frozen_mapping_input_points = 0;
  int frozen_mapping_confirmed_points = 0;
  for (const auto & submap : submaps_) {
    if (!submap.completed || !submap.frozen_geometry_valid) {
      continue;
    }
    frozen_mapping_input_points += submap.frozen_mapping_input_points;
    frozen_mapping_confirmed_points += submap.frozen_mapping_confirmed_points;
  }
  add_value(
    "frozen_submap_mapping_input_points",
    toString(frozen_mapping_input_points));
  add_value(
    "frozen_submap_mapping_confirmed_points",
    toString(frozen_mapping_confirmed_points));
  add_value("submap_max_keyframes", toString(submap_max_keyframes_));
  add_value("submap_max_travel_m", toString(submap_max_travel_m_));
  add_value("submap_overlap_keyframes", toString(submap_overlap_keyframes_));
  add_value("max_local_submaps", toString(max_local_submaps_));
  const bool have_active_submap = !submaps_.empty() && !submaps_.back().completed;
  add_value(
    "active_submap_id",
    have_active_submap ? toString(submaps_.back().id) : "unavailable");
  add_value(
    "active_submap_keyframes",
    have_active_submap ? toString(submaps_.back().keyframe_indices.size()) : "0");
  add_value(
    "active_submap_points",
    have_active_submap && submaps_.back().cloud_map ?
    toString(submaps_.back().cloud_map->size()) : "0");
  add_value(
    "active_submap_accumulated_travel_m",
    have_active_submap ? toString(submaps_.back().accumulated_travel_m) : "0");
  add_value("global_map_points", toString(global_map_ ? global_map_->size() : 0));
  add_value("map_cleanup", "external_postprocess");
  add_value("global_map_semantics", "unfiltered_mapping_cloud");
  add_value("keyframe_archive_directory", keyframe_archive_directory_);
  add_value("keyframe_archive_retention",
    keyframe_archive_directory_.empty() ? "disabled" : "enabled");
  add_value("keyframe_archive_pruned_keyframes", toString(pruned_keyframe_count_));
  const TerrainAccumulatorStats terrain_stats = terrain_accumulator_.stats();
  const TerrainPreviewStats terrain_preview_stats = terrain_preview_tracker_.stats();
  add_value("terrain_evidence", terrain_config_.enabled ? "enabled" : "disabled");
  add_value("terrain_evidence_topic", terrain_evidence_topic_);
  add_value("terrain_evidence_colored_topic", terrain_evidence_colored_topic_);
  add_value("terrain_support_evidence_topic", terrain_support_topic_);
  add_value(
    "terrain_probable_support_min_confidence",
    toString(terrain_accumulator_config_.probable_support_min_confidence));
  add_value(
    "terrain_probable_support_max_roughness_score",
    toString(terrain_accumulator_config_.probable_support_max_roughness_score));
  add_value(
    "terrain_probable_support_require_confirmed_connectivity",
    terrain_accumulator_config_.probable_support_require_confirmed_connectivity ?
    "enabled" : "disabled");
  add_value(
    "terrain_probable_support_max_anchor_distance_m",
    toString(terrain_accumulator_config_.probable_support_max_anchor_distance_m));
  add_value(
    "terrain_probable_support_max_neighbor_height_delta_m",
    toString(terrain_accumulator_config_.probable_support_max_neighbor_height_delta_m));
  add_value(
    "terrain_ramp_traversability_scale",
    toString(terrain_accumulator_config_.ramp_traversability_scale));
  add_value(
    "terrain_ramp_plane_validation",
    terrain_config_.ramp_plane_validation_enabled ? "enabled" : "disabled");
  add_value(
    "terrain_ramp_plane_neighborhood_radius_m",
    toString(terrain_config_.ramp_plane_neighborhood_radius_m));
  add_value(
    "terrain_ramp_plane_min_points",
    toString(terrain_config_.ramp_plane_min_points));
  add_value(
    "terrain_ramp_plane_min_length_m",
    toString(terrain_config_.ramp_plane_min_length_m));
  add_value(
    "terrain_ramp_plane_min_width_m",
    toString(terrain_config_.ramp_plane_min_width_m));
  add_value(
    "terrain_ramp_plane_max_rmse_m",
    toString(terrain_config_.ramp_plane_max_rmse_m));
  add_value(
    "terrain_ramp_plane_max_outlier_distance_m",
    toString(terrain_config_.ramp_plane_max_outlier_distance_m));
  add_value(
    "terrain_ramp_plane_min_inlier_ratio",
    toString(terrain_config_.ramp_plane_min_inlier_ratio));
  add_value(
    "terrain_traversal_confirmation",
    terrain_preview_config_.enabled ? "observer_only" : "disabled");
  add_value("terrain_preview_evidence_topic", terrain_preview_evidence_topic_);
  add_value(
    "terrain_preview_pending_shock",
    terrain_pending_shock_event_ ? "true" : "false");
  add_value(
    "terrain_preview_pending_shock_score",
    toString(terrain_pending_shock_score_));
  add_value(
    "terrain_preview_pending_shock_stamp_ns",
    toString(terrain_pending_shock_stamp_.nanoseconds()));
  add_value(
    "terrain_preview_pending_shock_bridge_invalidated",
    terrain_pending_shock_bridge_invalidated_ ? "true" : "false");
  add_value(
    "terrain_preview_continuity_break_pending",
    terrain_traversal_continuity_break_pending_ ? "true" : "false");
  add_value(
    "terrain_preview_unlocalized_shock_events",
    toString(terrain_unlocalized_shock_event_count_));
  add_value("terrain_candidate_voxels", toString(terrain_stats.candidate_voxels));
  add_value("terrain_confirmed_voxels", toString(terrain_stats.confirmed_voxels));
  add_value("terrain_conflicted_voxels", toString(terrain_stats.conflicted_voxels));
  add_value(
    "terrain_probable_support_voxels",
    toString(terrain_stats.probable_support_voxels));
  add_value(
    "terrain_confirmed_support_voxels",
    toString(terrain_stats.confirmed_support_voxels));
  add_value(
    "terrain_probable_hazard_voxels",
    toString(terrain_stats.probable_hazard_voxels));
  add_value(
    "terrain_confirmed_hazard_voxels",
    toString(terrain_stats.confirmed_hazard_voxels));
  add_value("terrain_unresolved_voxels", toString(terrain_stats.unresolved_voxels));
  add_value(
    "terrain_isolated_probable_support_voxels",
    toString(terrain_stats.isolated_probable_support_voxels));
  add_value(
    "terrain_support_rebuild_runtime_ms",
    toString(terrain_stats.support_rebuild_runtime_ms));
  add_value(
    "terrain_ambiguous_view_observations",
    toString(terrain_stats.ambiguous_view_observations));
  add_value("terrain_preview_landmarks", toString(terrain_preview_stats.landmarks));
  add_value(
    "terrain_preview_tentative_landmarks",
    toString(terrain_preview_stats.tentative_landmarks));
  add_value(
    "terrain_preview_geometry_confirmed_landmarks",
    toString(terrain_preview_stats.geometry_confirmed_landmarks));
  add_value(
    "terrain_preview_traversal_confirmed_landmarks",
    toString(terrain_preview_stats.traversal_confirmed_landmarks));
  add_value(
    "terrain_preview_unsafe_traversal_conflict_landmarks",
    toString(terrain_preview_stats.unsafe_traversal_conflict_landmarks));
  add_value(
    "terrain_preview_roughness_landmarks",
    toString(terrain_preview_stats.roughness_preview_landmarks));
  add_value(
    "terrain_preview_ahead_landmarks",
    toString(terrain_preview_stats.ahead_preview_landmarks));
  add_value(
    "terrain_preview_min_ahead_distance_m",
    toString(terrain_preview_config_.min_ahead_distance_m));
  add_value(
    "terrain_preview_max_lateral_distance_m",
    toString(terrain_preview_config_.max_lateral_distance_m));
  add_value(
    "terrain_preview_active_passages",
    toString(terrain_preview_stats.active_passages));
  add_value(
    "terrain_preview_traversal_samples",
    toString(terrain_preview_stats.traversal_samples));
  add_value(
    "terrain_preview_finalized_passages",
    toString(terrain_preview_stats.finalized_passages));
  add_value(
    "terrain_preview_shock_passages",
    toString(terrain_preview_stats.shock_passages));
  add_value(
    "terrain_preview_stable_passages",
    toString(terrain_preview_stats.stable_passages));
  add_value(
    "terrain_preview_unsafe_traversal_conflicts",
    toString(terrain_preview_stats.unsafe_traversal_conflicts));
  add_value(
    "terrain_preview_invalid_samples",
    toString(terrain_preview_stats.invalid_samples));
  add_value(
    "terrain_preview_continuity_breaks",
    toString(terrain_preview_stats.continuity_breaks));
  add_value(
    "terrain_previewed_before_traversal_landmarks",
    toString(terrain_preview_stats.previewed_before_traversal_landmarks));
  add_value(
    "terrain_preview_mean_lead_time_s",
    toString(terrain_preview_stats.mean_preview_lead_time_s));
  add_value(
    "terrain_preview_max_lead_time_s",
    toString(terrain_preview_stats.max_preview_lead_time_s));
  add_value("terrain_gravity_status", last_terrain_gravity_status_);
  add_value(
    "terrain_gravity_unavailable_keyframes",
    toString(terrain_gravity_unavailable_keyframe_count_));
  add_value(
    "terrain_gravity_stale_keyframes",
    toString(terrain_gravity_stale_keyframe_count_));
  add_value(
    "terrain_confirmed_floor_voxels",
    toString(
      terrain_stats.confirmed_by_label[static_cast<std::size_t>(
        TerrainLabel::kFloor)]));
  add_value(
    "terrain_confirmed_ramp_voxels",
    toString(
      terrain_stats.confirmed_by_label[static_cast<std::size_t>(
        TerrainLabel::kRamp)]));
  add_value(
    "terrain_confirmed_positive_step_voxels",
    toString(
      terrain_stats.confirmed_by_label[static_cast<std::size_t>(
        TerrainLabel::kPositiveStep)]));
  add_value(
    "terrain_confirmed_observed_drop_voxels",
    toString(
      terrain_stats.confirmed_by_label[static_cast<std::size_t>(
        TerrainLabel::kObservedDrop)]));
  add_value(
    "terrain_confirmed_staircase_voxels",
    toString(
      terrain_stats.confirmed_by_label[static_cast<std::size_t>(
        TerrainLabel::kStaircase)]));
  add_value(
    "terrain_last_keyframe_seed_valid",
    last_terrain_classification_.seed_valid ? "true" : "false");
  add_value(
    "terrain_last_keyframe_input_points",
    toString(terrain_last_keyframe_input_points_));
  add_value(
    "terrain_last_keyframe_retained_points",
    toString(terrain_last_keyframe_retained_points_));
  add_value(
    "terrain_last_keyframe_seed_floor_z_m",
    toString(last_terrain_classification_.seed_floor_z_m));
  add_value(
    "terrain_last_keyframe_floor_points",
    toString(last_terrain_classification_.floor_points));
  add_value(
    "terrain_last_keyframe_ramp_points",
    toString(last_terrain_classification_.ramp_points));
  add_value(
    "terrain_last_keyframe_positive_step_points",
    toString(last_terrain_classification_.positive_step_points));
  add_value(
    "terrain_last_keyframe_observed_drop_points",
    toString(last_terrain_classification_.observed_drop_points));
  add_value(
    "terrain_last_keyframe_staircase_points",
    toString(last_terrain_classification_.staircase_points));
  add_value(
    "terrain_last_keyframe_roughness_preview_points",
    toString(last_terrain_classification_.roughness_preview_points));
  add_value(
    "terrain_last_keyframe_ramp_plane_candidate_points",
    toString(last_terrain_classification_.ramp_plane_candidate_points));
  add_value(
    "terrain_last_keyframe_ramp_plane_validated_points",
    toString(last_terrain_classification_.ramp_plane_validated_points));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_points",
    toString(last_terrain_classification_.ramp_plane_rejected_points));
  add_value(
    "terrain_last_keyframe_ramp_plane_fits_attempted",
    toString(last_terrain_classification_.ramp_plane_fits_attempted));
  add_value(
    "terrain_last_keyframe_ramp_plane_fits_accepted",
    toString(last_terrain_classification_.ramp_plane_fits_accepted));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_insufficient_points",
    toString(last_terrain_classification_.ramp_plane_rejected_insufficient_points));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_plane_estimation",
    toString(last_terrain_classification_.ramp_plane_rejected_plane_estimation));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_inlier_support",
    toString(last_terrain_classification_.ramp_plane_rejected_inlier_support));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_length",
    toString(last_terrain_classification_.ramp_plane_rejected_length));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_width",
    toString(last_terrain_classification_.ramp_plane_rejected_width));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_slope",
    toString(last_terrain_classification_.ramp_plane_rejected_slope));
  add_value(
    "terrain_last_keyframe_ramp_plane_rejected_rmse",
    toString(last_terrain_classification_.ramp_plane_rejected_rmse));
  add_value(
    "terrain_last_keyframe_ramp_plane_max_accepted_rmse_m",
    toString(last_terrain_classification_.max_accepted_ramp_plane_rmse_m));
  add_value(
    "terrain_last_keyframe_max_roughness_score",
    toString(last_terrain_classification_.max_roughness_score));
  add_value(
    "constant_velocity_prediction",
    use_constant_velocity_prediction_ ? "enabled" : "disabled");
  add_value("constant_velocity_ready", have_previous_accepted_pose_ ? "true" : "false");
  add_value(
    "constant_velocity_rebase_pending",
    constant_velocity_rebase_pending_ ? "true" : "false");
  add_value(
    "imu_rotation_prediction",
    imu_rotation_prediction_enabled_ ? "enabled" : "disabled");
  add_value("imu_rotation_prediction_source", imu_rotation_prediction_source_);
  add_value(
    "imu_orientation_prediction_max_time_delta_s",
    toString(imu_orientation_prediction_max_time_delta_s_));
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
  add_value(
    "imu_subscription_reliability",
    imu_subscription_reliable_ ? "reliable" : "best_effort");
  add_value("imu_reorder_tolerance_s", toString(imu_reorder_tolerance_s_));
  add_value(
    "imu_timestamp_reset_threshold_s",
    toString(imu_timestamp_reset_threshold_s_));
  add_value("imu_duplicate_dropped_count", toString(imu_duplicate_dropped_count));
  add_value("imu_out_of_order_dropped_count", toString(imu_out_of_order_dropped_count));
  add_value("imu_rewind_dropped_count", toString(imu_rewind_dropped_count));
  add_value("imu_timestamp_reset_count", toString(imu_timestamp_reset_count));
  add_value(
    "imu_require_expected_frame",
    imu_require_expected_frame_ ? "enabled" : "disabled");
  add_value("imu_expected_frame", imu_expected_frame_);
  add_value("imu_frame_rejected_count", toString(imu_frame_rejected_count));
  add_value(
    "imu_nonfinite_rejected_count",
    toString(imu_nonfinite_rejected_count));
  add_value(
    "imu_timestamp_reset_policy",
    "detect_and_drop_without_partial_state_reset");
  add_value(
    "cloud_subscription_reliability",
    cloud_subscription_reliable_ ? "reliable" : "best_effort");
  add_value("cloud_subscription_depth", toString(cloud_subscription_depth_));
  add_value("cloud_sync_queue_size", toString(cloud_sync_queue_size_));
  add_value(
    "cloud_sync_wait_for_nearest_candidate",
    cloud_sync_wait_for_nearest_candidate_ ? "enabled" : "disabled");
  add_value(
    "cloud_sync_allow_wall_timeout_fallback",
    cloud_sync_allow_wall_timeout_fallback_ ? "enabled" : "disabled");
  add_value(
    "max_auxiliary_stamp_delta_s",
    toString(max_auxiliary_stamp_delta_s_));
  add_value(
    "cloud_sync_wait_timeout_s",
    toString(cloud_sync_wait_timeout_s_));
  add_value(
    "cloud_sync_reorder_window_s",
    toString(cloud_sync_reorder_window_s_));
  add_value(
    "cloud_processing_period_s",
    toString(cloud_processing_period_s_));
  add_value("executor_thread_count", toString(executor_thread_count_));
  add_value(
    "initialization_warmup_duration_s",
    toString(initialization_warmup_duration_s_));
  {
    std::lock_guard<std::mutex> cloud_lock(cloud_sync_mutex_);
    add_value(
      "primary_sync_queue_overflow_count",
      toString(primary_sync_queue_overflow_count_));
    add_value(
      "primary_sync_duplicate_count",
      toString(primary_sync_duplicate_count_));
    add_value(
      "primary_sync_out_of_order_count",
      toString(primary_sync_out_of_order_count_));
    add_value(
      "cloud_sync_watermark_resolution_count",
      toString(cloud_sync_watermark_resolution_count_));
    add_value(
      "cloud_sync_wall_timeout_count",
      toString(cloud_sync_wall_timeout_count_));
    add_value(
      "inter_lidar_wheel_wait_timeout_count",
      toString(inter_lidar_wheel_wait_timeout_count_));
    add_value(
      "deskew_imu_watermark_waited_count",
      toString(deskew_imu_watermark_waited_count_));
    add_value(
      "deskew_imu_wait_timeout_count",
      toString(deskew_imu_wait_timeout_count_));
    for (const auto & input : auxiliary_cloud_inputs_) {
      const auto prefix = input.name + "_cloud_";
      add_value(prefix + "enabled", input.enabled ? "true" : "false");
      add_value(prefix + "topic", input.topic);
      add_value(prefix + "queued", toString(input.queue.size()));
      add_value(prefix + "consumed_count", toString(input.consumed_count));
      add_value(prefix + "missing_count", toString(input.missing_count));
      add_value(prefix + "stale_drop_count", toString(input.stale_drop_count));
      add_value(prefix + "queue_overflow_count", toString(input.queue_overflow_count));
      add_value(prefix + "duplicate_count", toString(input.duplicate_count));
      add_value(prefix + "out_of_order_count", toString(input.out_of_order_count));
      add_value(
        prefix + "latest_stamp_ns",
        input.have_latest_stamp ? toString(input.latest_stamp_ns) : "unavailable");
    }
  }
  add_value(
    "map_publish_lock_skip_count",
    toString(map_publish_lock_skip_count_.load()));
  add_value(
    "map_publish_snapshot_update_count",
    toString(map_publish_snapshot_update_count_.load()));
  add_value("map_publish_callback_count", toString(map_publish_callback_count_.load()));
  add_value("map_publish_runtime_ms", toString(map_publish_runtime_ms_.load()));
  add_value("map_publish_runtime_ms_max", toString(map_publish_runtime_ms_max_.load()));
  add_value(
    "terrain_support_publish_runtime_ms",
    toString(terrain_support_publish_runtime_ms_.load()));
  add_value(
    "terrain_support_publish_runtime_ms_max",
    toString(terrain_support_publish_runtime_ms_max_.load()));
  add_value("latest_wheel_stamp_ns", toString(latest_wheel_stamp.nanoseconds()));
  add_value("last_used_wheel_stamp_ns", toString(last_used_wheel_stamp.nanoseconds()));
  add_value(
    "wheel_prediction_query_stamp_ns",
    toString(wheel_prediction_query_stamp.nanoseconds()));
  add_value(
    "wheel_prediction_lower_stamp_ns",
    toString(wheel_prediction_lower_stamp.nanoseconds()));
  add_value(
    "wheel_prediction_upper_stamp_ns",
    toString(wheel_prediction_upper_stamp.nanoseconds()));
  add_value("wheel_prediction_status", wheel_prediction_status);
  add_value(
    "wheel_prediction_max_bracket_delta_s",
    toString(wheel_prediction_max_bracket_delta_s));
  add_value("wheel_odom_buffer_size", toString(wheel_odom_buffer_size));
  add_value(
    "wheel_odom_buffer_duration_s",
    toString(wheel_odom_buffer_duration_s_));
  add_value(
    "wheel_odom_max_time_delta_s",
    toString(wheel_odom_max_time_delta_s_));
  add_value(
    "wheel_odom_max_interpolation_gap_s",
    toString(wheel_odom_max_interpolation_gap_s_));
  add_value(
    "wheel_odom_timestamp_reset_threshold_s",
    toString(wheel_odom_timestamp_reset_threshold_s_));
  add_value(
    "wheel_odom_require_expected_frames",
    wheel_odom_require_expected_frames_ ? "true" : "false");
  add_value("wheel_odom_expected_frame", wheel_odom_expected_frame_);
  add_value("wheel_odom_expected_child_frame", wheel_odom_expected_child_frame_);
  add_value(
    "wheel_odom_buffer_max_samples",
    toString(wheel_odom_buffer_max_samples_));
  add_value("wheel_odom_duplicate_count", toString(wheel_odom_duplicate_count));
  add_value(
    "wheel_odom_out_of_order_count",
    toString(wheel_odom_out_of_order_count));
  add_value(
    "wheel_odom_frame_rejected_count",
    toString(wheel_odom_frame_rejected_count));
  add_value(
    "wheel_odom_invalid_pose_rejected_count",
    toString(wheel_odom_invalid_pose_rejected_count));
  add_value(
    "wheel_odom_timestamp_rewind_rejected_count",
    toString(wheel_odom_timestamp_rewind_rejected_count));

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
