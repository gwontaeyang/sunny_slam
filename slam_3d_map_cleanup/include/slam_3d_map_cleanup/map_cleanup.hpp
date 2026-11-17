#ifndef SLAM_3D_MAP_CLEANUP__MAP_CLEANUP_HPP_
#define SLAM_3D_MAP_CLEANUP__MAP_CLEANUP_HPP_

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "slam_3d_map_cleanup/temporal_static_map.hpp"

namespace gole
{
namespace slam_3d_map_cleanup
{

struct MapZFilter
{
  bool enabled{false};
  double min_m{0.0};
  double max_m{0.0};
};

struct CleanupResult
{
  TemporalStaticMapStats evidence;
  std::size_t keyframes{0U};
  std::size_t scans{0U};
  std::size_t archive_points{0U};
  std::size_t z_filtered_points{0U};
  double processing_ms{0.0};
  double output_ms{0.0};
  std::vector<std::string> warnings;
};

TemporalStaticMapConfig loadCleanupConfig(const std::filesystem::path & yaml_path);

// Endpoints are already deskewed in base, not in the individual sensor frame.
TemporalStaticMapSensorView projectSensorView(
  const Eigen::Isometry3d & map_from_base,
  const TemporalStaticMapSensorView & base_view,
  const MapZFilter & z_filter,
  double voxel_size_m);

// Throws on invalid input or IO failure. manifest.yaml is the last success marker.
CleanupResult cleanupArchive(
  const std::filesystem::path & input_dir,
  const std::filesystem::path & output_dir,
  const TemporalStaticMapConfig & config = TemporalStaticMapConfig{});

int runTemporalMapCleanupCli(int argc, char ** argv, std::ostream & out, std::ostream & err);

}  // namespace slam_3d_map_cleanup
}  // namespace gole

#endif  // SLAM_3D_MAP_CLEANUP__MAP_CLEANUP_HPP_
