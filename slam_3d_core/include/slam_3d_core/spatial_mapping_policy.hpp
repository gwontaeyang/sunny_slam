#ifndef SLAM_3D_CORE__SPATIAL_MAPPING_POLICY_HPP_
#define SLAM_3D_CORE__SPATIAL_MAPPING_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace gole
{
namespace slam_3d_core
{

enum class MappingSpatialMode
{
  kPlanar,
  kSe3,
  kFloorAware,
};

inline std::optional<MappingSpatialMode> parseMappingSpatialMode(
  const std::string & value) noexcept
{
  if (value == "planar") {
    return MappingSpatialMode::kPlanar;
  }
  if (value == "se3") {
    return MappingSpatialMode::kSe3;
  }
  if (value == "floor_aware") {
    return MappingSpatialMode::kFloorAware;
  }
  return std::nullopt;
}

inline const char * mappingSpatialModeName(const MappingSpatialMode mode) noexcept
{
  switch (mode) {
    case MappingSpatialMode::kPlanar:
      return "planar";
    case MappingSpatialMode::kSe3:
      return "se3";
    case MappingSpatialMode::kFloorAware:
      return "floor_aware";
  }
  return "planar";
}

struct SpatialMappingPolicyConfig
{
  MappingSpatialMode mode{MappingSpatialMode::kPlanar};
  double keyframe_translation_m{0.8};
  double keyframe_yaw_deg{8.0};
  double keyframe_vertical_translation_m{0.0};
  double keyframe_rotation_3d_deg{0.0};
  double local_map_radius_m{35.0};
  double local_map_vertical_radius_m{0.0};
};

struct SpatialDeltaMetrics
{
  bool finite{false};
  double xy_translation_m{0.0};
  double translation_3d_m{0.0};
  double vertical_translation_m{0.0};
  double yaw_deg{0.0};
  double rotation_3d_deg{0.0};
};

inline double normalizedSpatialAngle(const double angle) noexcept
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

inline SpatialDeltaMetrics computeSpatialDelta(
  const Eigen::Isometry3d & pose,
  const Eigen::Isometry3d & reference) noexcept
{
  SpatialDeltaMetrics metrics;
  if (!pose.matrix().allFinite() || !reference.matrix().allFinite()) {
    return metrics;
  }

  const Eigen::Vector3d translation = pose.translation() - reference.translation();
  const Eigen::Matrix3d relative_rotation = reference.linear().transpose() * pose.linear();
  Eigen::Quaterniond quaternion(relative_rotation);
  if (!quaternion.coeffs().allFinite() || quaternion.norm() <= 1.0e-12) {
    return metrics;
  }
  quaternion.normalize();

  constexpr double radians_to_degrees = 180.0 / 3.14159265358979323846;
  const double pose_yaw = std::atan2(pose.linear()(1, 0), pose.linear()(0, 0));
  const double reference_yaw =
    std::atan2(reference.linear()(1, 0), reference.linear()(0, 0));
  metrics.xy_translation_m = std::hypot(translation.x(), translation.y());
  metrics.translation_3d_m = translation.norm();
  metrics.vertical_translation_m = std::abs(translation.z());
  metrics.yaw_deg =
    std::abs(normalizedSpatialAngle(pose_yaw - reference_yaw)) * radians_to_degrees;
  metrics.rotation_3d_deg =
    2.0 * std::acos(std::clamp(std::abs(quaternion.w()), 0.0, 1.0)) *
    radians_to_degrees;
  metrics.finite =
    std::isfinite(metrics.xy_translation_m) &&
    std::isfinite(metrics.translation_3d_m) &&
    std::isfinite(metrics.vertical_translation_m) &&
    std::isfinite(metrics.yaw_deg) &&
    std::isfinite(metrics.rotation_3d_deg);
  return metrics;
}

inline bool enabledThresholdExceeded(
  const double value,
  const double threshold) noexcept
{
  return std::isfinite(value) && std::isfinite(threshold) &&
         threshold > 0.0 && value >= threshold;
}

inline bool shouldAddSpatialKeyframe(
  const Eigen::Isometry3d & pose,
  const Eigen::Isometry3d & reference,
  const SpatialMappingPolicyConfig & config) noexcept
{
  const SpatialDeltaMetrics delta = computeSpatialDelta(pose, reference);
  if (!delta.finite) {
    return false;
  }

  const double rotation_3d_threshold =
    config.keyframe_rotation_3d_deg > 0.0 ?
    config.keyframe_rotation_3d_deg : config.keyframe_yaw_deg;
  switch (config.mode) {
    case MappingSpatialMode::kPlanar:
      // Preserve the historical zero-threshold behavior for this legacy mode.
      return delta.xy_translation_m >= std::max(0.0, config.keyframe_translation_m) ||
             delta.yaw_deg >= std::max(0.0, config.keyframe_yaw_deg);
    case MappingSpatialMode::kSe3:
      return delta.translation_3d_m >= std::max(0.0, config.keyframe_translation_m) ||
             delta.rotation_3d_deg >= std::max(0.0, rotation_3d_threshold) ||
             enabledThresholdExceeded(
        delta.vertical_translation_m,
        config.keyframe_vertical_translation_m);
    case MappingSpatialMode::kFloorAware:
      return delta.xy_translation_m >= std::max(0.0, config.keyframe_translation_m) ||
             delta.yaw_deg >= std::max(0.0, config.keyframe_yaw_deg) ||
             enabledThresholdExceeded(
        delta.vertical_translation_m,
        config.keyframe_vertical_translation_m) ||
             enabledThresholdExceeded(delta.rotation_3d_deg, rotation_3d_threshold);
  }
  return false;
}

inline double spatialTravelDistance(
  const Eigen::Isometry3d & pose,
  const Eigen::Isometry3d & reference,
  const MappingSpatialMode mode) noexcept
{
  const SpatialDeltaMetrics delta = computeSpatialDelta(pose, reference);
  if (!delta.finite) {
    return std::numeric_limits<double>::infinity();
  }
  return mode == MappingSpatialMode::kSe3 ?
         delta.translation_3d_m : delta.xy_translation_m;
}

inline double accumulatedSpatialTravel(
  const std::vector<Eigen::Isometry3d> & poses,
  const MappingSpatialMode mode) noexcept
{
  double distance_m = 0.0;
  for (std::size_t index = 1; index < poses.size(); ++index) {
    distance_m += spatialTravelDistance(poses[index], poses[index - 1U], mode);
  }
  return distance_m;
}

inline bool poseWithinLocalMapWindow(
  const Eigen::Isometry3d & pose,
  const Eigen::Isometry3d & center,
  const SpatialMappingPolicyConfig & config) noexcept
{
  const SpatialDeltaMetrics delta = computeSpatialDelta(pose, center);
  if (!delta.finite) {
    return false;
  }

  const double radius = std::max(0.0, config.local_map_radius_m);
  if (config.mode == MappingSpatialMode::kSe3) {
    return delta.translation_3d_m <= radius &&
           (config.local_map_vertical_radius_m <= 0.0 ||
           delta.vertical_translation_m <= config.local_map_vertical_radius_m);
  }
  if (delta.xy_translation_m > radius) {
    return false;
  }
  if (config.mode == MappingSpatialMode::kFloorAware &&
    config.local_map_vertical_radius_m > 0.0)
  {
    return delta.vertical_translation_m <= config.local_map_vertical_radius_m;
  }
  return true;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__SPATIAL_MAPPING_POLICY_HPP_
