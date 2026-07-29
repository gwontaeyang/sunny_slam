#ifndef SLAM_3D_CORE__REGISTRATION_POLICY_HPP_
#define SLAM_3D_CORE__REGISTRATION_POLICY_HPP_

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Geometry>

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

  const Eigen::Isometry3d prediction_T_place =
    prediction.inverse() * bounded_place_pose;
  const Eigen::Isometry3d map_correction =
    bounded_place_pose * prediction.inverse();
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

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__REGISTRATION_POLICY_HPP_
