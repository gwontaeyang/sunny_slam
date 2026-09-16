#ifndef SLAM_3D_CORE__REGISTRATION_BACKEND_HPP_
#define SLAM_3D_CORE__REGISTRATION_BACKEND_HPP_

#include <limits>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace gole
{
namespace slam_3d_core
{

struct RegistrationBackendOptions
{
  int max_iterations{40};
  double max_correspondence_m{1.0};
  double quality_max_correspondence_m{1.0};
  bool use_trimmed{false};
  double trim_ratio{0.8};
  int min_correspondences{3};
  int fast_gicp_num_threads{1};
  double ndt_resolution_m{1.0};
  double ndt_step_size{0.1};
  double ndt_transformation_epsilon{0.01};
};

struct RegistrationBackendResult
{
  bool converged{false};
  double fitness{std::numeric_limits<double>::infinity()};
  int iterations{0};
  Eigen::Matrix4f target_T_source{Eigen::Matrix4f::Identity()};
  pcl::PointCloud<pcl::PointXYZ> aligned;
};

std::vector<std::string> availableRegistrationMethods();
void validateRegistrationMethod(const std::string & method);
void validateRegistrationOptions(
  const std::string & method, const RegistrationBackendOptions & options);

// voxel_surfel has its own observability-aware result and is dispatched by the node.
RegistrationBackendResult alignRegistration(
  const std::string & method,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & target,
  const Eigen::Matrix4f & guess,
  const RegistrationBackendOptions & options);

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__REGISTRATION_BACKEND_HPP_
