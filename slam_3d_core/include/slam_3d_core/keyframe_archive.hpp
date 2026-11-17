#ifndef SLAM_3D_CORE__KEYFRAME_ARCHIVE_HPP_
#define SLAM_3D_CORE__KEYFRAME_ARCHIVE_HPP_

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace gole
{
namespace slam_3d_core
{

struct ArchivedSensorScan
{
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud_base;
  Eigen::Vector3d origin_base{Eigen::Vector3d::Zero()};
};

struct ArchivedKeyframe
{
  std::uint64_t id{0};
  std::int64_t stamp_ns{0};
  Eigen::Isometry3d map_T_base{Eigen::Isometry3d::Identity()};
  std::vector<ArchivedSensorScan> scans;
};

struct KeyframeArchive
{
  std::string map_frame;
  std::string base_frame;
  bool map_z_filter_enabled{false};
  double map_z_min_m{-5.0};
  double map_z_max_m{8.0};
  std::uint64_t pruned_keyframes{0};
  std::vector<ArchivedKeyframe> keyframes;
};

// Refuses existing destinations. The manifest is committed only after all scans.
void writeKeyframeArchive(const std::string & directory, const KeyframeArchive & archive);

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__KEYFRAME_ARCHIVE_HPP_
