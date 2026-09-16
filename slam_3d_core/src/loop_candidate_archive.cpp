#include "slam_3d_core/loop_candidate_archive.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <pcl/common/point_tests.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>

namespace gole
{
namespace slam_3d_core
{
namespace
{

bool validFrame(const std::string & frame)
{
  return !frame.empty() && frame.find_first_of(" \t\n\r\f\v") == std::string::npos &&
         frame.find('\0') == std::string::npos;
}

void validatePose(const Eigen::Isometry3d & pose)
{
  const Eigen::Matrix3d rotation = pose.linear();
  if (!pose.matrix().allFinite() ||
    !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1.0e-6) ||
    std::abs(rotation.determinant() - 1.0) > 1.0e-6 ||
    !pose.matrix().row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1.0e-12))
  {
    throw std::invalid_argument("invalid loop candidate rigid transform");
  }
}

void validateCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr & cloud)
{
  if (!cloud) {
    throw std::invalid_argument("null loop candidate cloud");
  }
  for (const auto & point : *cloud) {
    if (!pcl::isFinite(point)) {
      throw std::invalid_argument("nonfinite loop candidate point");
    }
  }
}

std::vector<double> rowMajorPose(const Eigen::Isometry3d & pose)
{
  std::vector<double> values;
  values.reserve(16);
  for (Eigen::Index row = 0; row < 4; ++row) {
    for (Eigen::Index column = 0; column < 4; ++column) {
      values.push_back(pose.matrix()(row, column));
    }
  }
  return values;
}

void writeCloud(const std::filesystem::path & path, const pcl::PointCloud<pcl::PointXYZ> & cloud)
{
  if (cloud.empty()) {
    // PCL versions differ in empty-cloud support; preserve sparse rejection evidence.
    std::ofstream stream;
    stream.exceptions(std::ios::failbit | std::ios::badbit);
    stream.open(path, std::ios::out | std::ios::binary);
    stream << "# .PCD v0.7 - Point Cloud Data file format\n"
      "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
      "WIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 0\nDATA binary\n";
    stream.close();
  } else if (pcl::io::savePCDFileBinary(path.string(), cloud) < 0) {
    throw std::runtime_error("failed writing loop candidate PCD: " + path.string());
  }
}

}  // namespace

void writeLoopCandidateArchive(const std::string & directory, const LoopCandidateArchive & archive)
{
  if (directory.empty() || directory.find('\0') != std::string::npos ||
    !validFrame(archive.map_frame) || !validFrame(archive.base_frame) ||
    archive.stamp_ns < 0 || archive.candidate_keyframe_index < 0 ||
    archive.candidate_submap_id < -1)
  {
    throw std::invalid_argument("invalid loop candidate archive metadata or directory");
  }
  validatePose(archive.reference_map_T_base);
  validatePose(archive.coarse_map_T_base);
  if (archive.refined_map_T_base) {
    validatePose(*archive.refined_map_T_base);
  }
  validateCloud(archive.source_base);
  validateCloud(archive.target_map);

  const std::filesystem::path root(directory);
  if (std::filesystem::exists(std::filesystem::symlink_status(root))) {
    throw std::runtime_error("archive destination already exists: " + directory);
  }
  if (!root.parent_path().empty()) {
    std::filesystem::create_directories(root.parent_path());
  }
  if (!std::filesystem::create_directory(root)) {
    throw std::runtime_error("could not create new archive directory: " + directory);
  }

  YAML::Node manifest;
  manifest["schema_version"] = 1;
  manifest["status"] = "complete";
  manifest["semantics"] = "pre_backend_candidate_validation_not_accepted_loop";
  manifest["map_frame"] = archive.map_frame;
  manifest["base_frame"] = archive.base_frame;
  manifest["stamp_ns"] = archive.stamp_ns;
  manifest["candidate_keyframe_index"] = archive.candidate_keyframe_index;
  manifest["candidate_submap_id"] = archive.candidate_submap_id;
  manifest["transform_encoding"] = "matrix4x4_row_major";
  manifest["reference_map_T_base"] = rowMajorPose(archive.reference_map_T_base);
  manifest["coarse_map_T_base"] = rowMajorPose(archive.coarse_map_T_base);
  manifest["refined_map_T_base"] = archive.refined_map_T_base ?
    YAML::Node(rowMajorPose(*archive.refined_map_T_base)) : YAML::Node(YAML::NodeType::Null);
  manifest["source_base"]["pcd"] = "source_base.pcd";
  manifest["source_base"]["frame"] = archive.base_frame;
  manifest["source_base"]["point_count"] = archive.source_base->size();
  manifest["target_map"]["pcd"] = "target_map.pcd";
  manifest["target_map"]["frame"] = archive.map_frame;
  manifest["target_map"]["point_count"] = archive.target_map->size();
  manifest["diagnostics"] = YAML::Node(YAML::NodeType::Map);
  for (const auto & diagnostic : archive.diagnostics) {
    YAML::Node key(diagnostic.first);
    YAML::Node value(diagnostic.second);
    key.SetTag("tag:yaml.org,2002:str");
    value.SetTag("tag:yaml.org,2002:str");
    manifest["diagnostics"][key] = value;
  }

  writeCloud(root / "source_base.pcd", *archive.source_base);
  writeCloud(root / "target_map.pcd", *archive.target_map);

  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << manifest;
  if (!emitter.good()) {
    throw std::runtime_error("failed encoding loop candidate archive manifest");
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
