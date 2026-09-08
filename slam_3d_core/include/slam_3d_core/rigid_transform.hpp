#ifndef SLAM_3D_CORE__RIGID_TRANSFORM_HPP_
#define SLAM_3D_CORE__RIGID_TRANSFORM_HPP_

#include <cmath>

#include <Eigen/Geometry>

namespace gole
{
namespace slam_3d_core
{

inline Eigen::Isometry3d projectToRigidIsometry(
  const Eigen::Isometry3d & transform) noexcept
{
  if (!transform.matrix().allFinite()) {
    return transform;
  }

  Eigen::Quaterniond rotation(transform.linear());
  const double norm = rotation.norm();
  if (!rotation.coeffs().allFinite() || !std::isfinite(norm) || norm <= 1.0e-12) {
    return transform;
  }

  Eigen::Isometry3d rigid = Eigen::Isometry3d::Identity();
  rigid.translation() = transform.translation();
  rigid.linear() = rotation.normalized().toRotationMatrix();
  return rigid;
}

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__RIGID_TRANSFORM_HPP_
