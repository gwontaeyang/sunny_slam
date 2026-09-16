#ifndef SLAM_3D_CORE__LOOP_CANDIDATE_ARCHIVE_HPP_
#define SLAM_3D_CORE__LOOP_CANDIDATE_ARCHIVE_HPP_

#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace gole
{
namespace slam_3d_core
{

struct LoopCandidateArchive
{
  std::string map_frame;
  std::string base_frame;
  std::int64_t stamp_ns{0};
  int candidate_keyframe_index{-1};
  int candidate_submap_id{-1};
  Eigen::Isometry3d reference_map_T_base{Eigen::Isometry3d::Identity()};
  Eigen::Isometry3d coarse_map_T_base{Eigen::Isometry3d::Identity()};
  std::optional<Eigen::Isometry3d> refined_map_T_base{std::nullopt};
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr source_base;
  pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_map;
  std::map<std::string, std::string> diagnostics;
};

// Opt-in diagnostic evidence, not an accepted loop or a backend constraint.
// Schema 1: status="complete", semantics="pre_backend_candidate_validation_not_accepted_loop".
// Each *_map_T_base is a row-major sequence of 16 doubles mapping base to map;
// refined_map_T_base is YAML null when no refinement proposal is available.
// transform_encoding="matrix4x4_row_major". source_base/target_map contain
// {pcd, frame, point_count}, with frame equal to base_frame/map_frame respectively.
// Non-null empty clouds are stored as binary XYZ PCDs with WIDTH 0, HEIGHT 1,
// POINTS 0 and no payload. Readers may use point_count=0 to bypass PCL loading.
// Diagnostic keys and values remain YAML strings, including nonfinite text.
// Frames must be nonempty and whitespace-free; submap -1 means unavailable.
// Validates before creating files and refuses any existing destination.
// manifest.yaml.partial is renamed last to manifest.yaml; failures retain evidence
// without a committed manifest. Nothing is deleted and incomplete outputs cannot be reused.
void writeLoopCandidateArchive(const std::string & directory, const LoopCandidateArchive & archive);

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__LOOP_CANDIDATE_ARCHIVE_HPP_
