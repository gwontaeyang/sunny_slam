#ifndef SLAM_3D_CORE__SUBMAP_POLICY_HPP_
#define SLAM_3D_CORE__SUBMAP_POLICY_HPP_

#include <algorithm>
#include <cstddef>

namespace gole
{
namespace slam_3d_core
{

enum class SubmapClosureReason
{
  kNone,
  kKeyframeLimit,
  kTravelLimit,
  kSpatialTransition,
};

struct SubmapPolicyConfig
{
  int max_keyframes{30};
  double max_travel_m{20.0};
  int overlap_keyframes{5};
};

inline SubmapClosureReason submapClosureReason(
  const std::size_t keyframe_count,
  const double travel_m,
  const SubmapPolicyConfig & config) noexcept
{
  if (config.max_keyframes > 0 &&
    keyframe_count >= static_cast<std::size_t>(config.max_keyframes))
  {
    return SubmapClosureReason::kKeyframeLimit;
  }
  if (config.max_travel_m > 0.0 && travel_m >= config.max_travel_m) {
    return SubmapClosureReason::kTravelLimit;
  }
  return SubmapClosureReason::kNone;
}

inline std::size_t boundedSubmapOverlap(
  const std::size_t completed_keyframe_count,
  const int requested_overlap) noexcept
{
  if (completed_keyframe_count <= 1U || requested_overlap <= 0) {
    return 0U;
  }
  return std::min(
    completed_keyframe_count - 1U,
    static_cast<std::size_t>(requested_overlap));
}

inline std::size_t submapOverlapStartIndex(
  const std::size_t completed_keyframe_count,
  const int requested_overlap) noexcept
{
  return completed_keyframe_count -
         boundedSubmapOverlap(completed_keyframe_count, requested_overlap);
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__SUBMAP_POLICY_HPP_
