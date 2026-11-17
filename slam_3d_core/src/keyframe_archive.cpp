#include "slam_3d_core/keyframe_archive.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

namespace gole
{
namespace slam_3d_core
{

void writeKeyframeArchive(const std::string & directory, const KeyframeArchive & archive)
{
  if (directory.empty() || archive.map_frame.empty() || archive.base_frame.empty() ||
    archive.keyframes.empty() || !std::isfinite(archive.map_z_min_m) ||
    !std::isfinite(archive.map_z_max_m) || archive.map_z_min_m > archive.map_z_max_m)
  {
    throw std::invalid_argument("invalid or empty keyframe archive");
  }
  for (std::size_t index = 0; index < archive.keyframes.size(); ++index) {
    const auto & keyframe = archive.keyframes[index];
    const Eigen::Matrix3d rotation = keyframe.map_T_base.rotation();
    if (keyframe.stamp_ns < 0 || keyframe.scans.empty() ||
      !keyframe.map_T_base.matrix().allFinite() ||
      !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1.0e-6) ||
      std::abs(rotation.determinant() - 1.0) > 1.0e-6 ||
      !keyframe.map_T_base.matrix().row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1)) ||
      (index > 0 && (keyframe.id <= archive.keyframes[index - 1].id ||
      keyframe.stamp_ns <= archive.keyframes[index - 1].stamp_ns)))
    {
      throw std::invalid_argument("invalid keyframe pose, order or sensor views");
    }
    for (const auto & scan : keyframe.scans) {
      if (!scan.cloud_base || scan.cloud_base->empty() || !scan.origin_base.allFinite()) {
        throw std::invalid_argument("invalid or empty archived sensor scan");
      }
      for (const auto & point : *scan.cloud_base) {
        if (!pcl::isFinite(point)) {
          throw std::invalid_argument("nonfinite archived sensor point");
        }
      }
    }
  }

  const std::filesystem::path root(directory);
  if (std::filesystem::exists(root)) {
    throw std::runtime_error("archive destination already exists: " + directory);
  }
  if (!root.parent_path().empty()) {
    std::filesystem::create_directories(root.parent_path());
  }
  if (!std::filesystem::create_directory(root)) {
    throw std::runtime_error("could not create new archive directory: " + directory);
  }
  std::filesystem::create_directory(root / "keyframes");

  YAML::Node manifest;
  manifest["schema_version"] = 1;
  manifest["map_frame"] = archive.map_frame;
  manifest["base_frame"] = archive.base_frame;
  manifest["snapshot_scope"] = "retained_keyframes";
  manifest["pruned_keyframes"] = archive.pruned_keyframes;
  manifest["complete_session"] = archive.pruned_keyframes == 0;
  manifest["point_frame"] = "base_frame";
  manifest["point_source"] = "deskewed_mapping_only_before_terrain_classification";
  manifest["map_z_filter"]["enabled"] = archive.map_z_filter_enabled;
  manifest["map_z_filter"]["min_m"] = archive.map_z_min_m;
  manifest["map_z_filter"]["max_m"] = archive.map_z_max_m;
  for (const auto & keyframe : archive.keyframes) {
    YAML::Node entry;
    entry["id"] = keyframe.id;
    entry["stamp_ns"] = keyframe.stamp_ns;
    Eigen::Quaterniond orientation(keyframe.map_T_base.rotation());
    orientation.normalize();
    const auto & position = keyframe.map_T_base.translation();
    entry["pose"] = std::vector<double>{position.x(), position.y(), position.z(),
      orientation.x(), orientation.y(), orientation.z(), orientation.w()};
    for (std::size_t index = 0; index < keyframe.scans.size(); ++index) {
      const auto & scan = keyframe.scans[index];
      std::ostringstream filename;
      filename << "keyframes/" << std::setfill('0') << std::setw(6) << keyframe.id <<
        "_sensor" << std::setw(2) << index << ".pcd";
      if (pcl::io::savePCDFileBinary((root / filename.str()).string(), *scan.cloud_base) < 0) {
        throw std::runtime_error("failed writing archived PCD: " + filename.str());
      }
      YAML::Node view;
      view["pcd"] = filename.str();
      view["origin_base"] = std::vector<double>{
        scan.origin_base.x(), scan.origin_base.y(), scan.origin_base.z()};
      entry["scans"].push_back(view);
    }
    manifest["keyframes"].push_back(entry);
  }
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << manifest;
  if (!emitter.good()) {
    throw std::runtime_error("failed encoding keyframe archive manifest");
  }
  const auto temporary_manifest = root / "manifest.yaml.partial";
  std::ofstream stream;
  stream.exceptions(std::ios::failbit | std::ios::badbit);
  stream.open(temporary_manifest);
  stream << emitter.c_str() << '\n';
  stream.close();
  std::filesystem::rename(temporary_manifest, root / "manifest.yaml");
}

}  // namespace slam_3d_core
}  // namespace gole
