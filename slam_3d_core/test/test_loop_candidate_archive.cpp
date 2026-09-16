#include "slam_3d_core/loop_candidate_archive.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

namespace
{
using gole::slam_3d_core::LoopCandidateArchive;
using gole::slam_3d_core::writeLoopCandidateArchive;
using Cloud = pcl::PointCloud<pcl::PointXYZ>;

const std::array<std::string, 3> kPoseNames = {
  "reference_map_T_base", "coarse_map_T_base", "refined_map_T_base"};
const std::array<Cloud::ConstPtr LoopCandidateArchive::*, 2> kClouds = {
  &LoopCandidateArchive::source_base, &LoopCandidateArchive::target_map};

Eigen::Isometry3d & poseAt(LoopCandidateArchive & archive, std::size_t index)
{
  if (index == 0) {
    return archive.reference_map_T_base;
  }
  if (index == 1) {
    return archive.coarse_map_T_base;
  }
  return archive.refined_map_T_base.value();
}

std::string readFile(const std::filesystem::path & path)
{
  std::ifstream stream(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

class LoopCandidateArchiveTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const std::string pattern =
      (std::filesystem::temp_directory_path() / "loop_candidate_archive_XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const auto directory = mkdtemp(buffer.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
  }

  void TearDown() override
  {
    if (!root_.empty()) {
      std::filesystem::remove_all(root_);
    }
  }

  LoopCandidateArchive fixture()
  {
    LoopCandidateArchive archive;
    archive.map_frame = "survey_map";
    archive.base_frame = "robot/base_link";
    archive.stamp_ns = 1780000000123456789LL;
    archive.candidate_keyframe_index = 8;
    archive.candidate_submap_id = 3;
    archive.refined_map_T_base = Eigen::Isometry3d::Identity();
    for (std::size_t index = 0; index < kPoseNames.size(); ++index) {
      auto & pose = poseAt(archive, index);
      const double offset = static_cast<double>(index);
      pose.translation() = Eigen::Vector3d(3.123456789012345 + offset, -4.2 - offset, 5.3 + offset);
      pose.linear() = (
        Eigen::AngleAxisd(0.31 + offset, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(-0.42, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.53, Eigen::Vector3d::UnitX())).toRotationMatrix();
    }
    auto source = std::make_shared<Cloud>();
    source->push_back(pcl::PointXYZ(1.25F, -2.5F, 3.75F));
    source->push_back(pcl::PointXYZ(-4.5F, 5.25F, -6.75F));
    archive.source_base = source;
    auto target = std::make_shared<Cloud>();
    target->push_back(pcl::PointXYZ(-8.25F, 9.5F, 10.75F));
    archive.target_map = target;
    archive.diagnostics = {
      {"decision", "rejected"}, {"fitness", "inf"}, {".inf", ".nan"},
      {"true", "false"}, {"42", "00123"}, {"null", "null"},
      {"", ""}, {"reason: detail", "sparse target\nnot accepted # evidence"}};
    return archive;
  }

  void expectInvalid(const LoopCandidateArchive & archive)
  {
    const auto output = root_ / "not_created" / "archive";
    EXPECT_THROW(writeLoopCandidateArchive(output.string(), archive), std::invalid_argument);
    EXPECT_FALSE(std::filesystem::exists(output.parent_path()));
  }

  std::filesystem::path root_;
};

TEST_F(LoopCandidateArchiveTest, RoundTripsFullPosesFramesAndUntransformedPoints)
{
  auto archive = fixture();
  const auto output = root_ / "nested" / "archive";
  writeLoopCandidateArchive(output.string(), archive);
  const auto manifest = YAML::LoadFile((output / "manifest.yaml").string());
  EXPECT_EQ(manifest["schema_version"].as<int>(), 1);
  EXPECT_EQ(manifest["status"].as<std::string>(), "complete");
  EXPECT_EQ(
    manifest["semantics"].as<std::string>(),
    "pre_backend_candidate_validation_not_accepted_loop");
  EXPECT_EQ(manifest["map_frame"].as<std::string>(), archive.map_frame);
  EXPECT_EQ(manifest["base_frame"].as<std::string>(), archive.base_frame);
  EXPECT_EQ(manifest["stamp_ns"].as<std::int64_t>(), archive.stamp_ns);
  EXPECT_EQ(manifest["candidate_keyframe_index"].as<int>(), archive.candidate_keyframe_index);
  EXPECT_EQ(manifest["candidate_submap_id"].as<int>(), archive.candidate_submap_id);
  EXPECT_EQ(manifest["transform_encoding"].as<std::string>(), "matrix4x4_row_major");
  for (std::size_t index = 0; index < kPoseNames.size(); ++index) {
    SCOPED_TRACE(kPoseNames[index]);
    const auto values = manifest[kPoseNames[index]].as<std::vector<double>>();
    ASSERT_EQ(values.size(), 16U);
    for (Eigen::Index row = 0; row < 4; ++row) {
      for (Eigen::Index column = 0; column < 4; ++column) {
        EXPECT_DOUBLE_EQ(values[4 * row + column], poseAt(archive, index).matrix()(row, column));
      }
    }
  }
  const std::array<std::string, 2> cloud_names = {"source_base", "target_map"};
  const std::array<std::string, 2> frames = {archive.base_frame, archive.map_frame};
  for (std::size_t index = 0; index < kClouds.size(); ++index) {
    SCOPED_TRACE(cloud_names[index]);
    const auto entry = manifest[cloud_names[index]];
    EXPECT_EQ(entry["frame"].as<std::string>(), frames[index]);
    EXPECT_EQ(entry["pcd"].as<std::string>(), cloud_names[index] + ".pcd");
    const auto & expected = *(archive.*kClouds[index]);
    EXPECT_EQ(entry["point_count"].as<std::size_t>(), expected.size());
    Cloud cloud;
    ASSERT_EQ(pcl::io::loadPCDFile((output / entry["pcd"].as<std::string>()).string(), cloud), 0);
    ASSERT_EQ(cloud.size(), expected.size());
    for (std::size_t point = 0; point < cloud.size(); ++point) {
      EXPECT_FLOAT_EQ(cloud[point].x, expected[point].x);
      EXPECT_FLOAT_EQ(cloud[point].y, expected[point].y);
      EXPECT_FLOAT_EQ(cloud[point].z, expected[point].z);
    }
  }
  const auto diagnostics = manifest["diagnostics"].as<std::map<std::string, std::string>>();
  EXPECT_EQ(diagnostics, archive.diagnostics);
  for (const auto & diagnostic : manifest["diagnostics"]) {
    EXPECT_EQ(diagnostic.first.Tag(), "tag:yaml.org,2002:str");
    EXPECT_EQ(diagnostic.second.Tag(), "tag:yaml.org,2002:str");
  }
  EXPECT_FALSE(std::filesystem::exists(output / "manifest.yaml.partial"));
}

TEST_F(LoopCandidateArchiveTest, RefusesOverwriteWithoutChangingAnyEvidence)
{
  const auto output = root_ / "archive";
  writeLoopCandidateArchive(output.string(), fixture());
  const auto manifest = readFile(output / "manifest.yaml");
  const auto source = readFile(output / "source_base.pcd");
  const auto target = readFile(output / "target_map.pcd");
  auto changed = fixture();
  changed.stamp_ns += 1;
  changed.source_base = std::make_shared<Cloud>();
  EXPECT_THROW(writeLoopCandidateArchive(output.string(), changed), std::runtime_error);
  EXPECT_EQ(readFile(output / "manifest.yaml"), manifest);
  EXPECT_EQ(readFile(output / "source_base.pcd"), source);
  EXPECT_EQ(readFile(output / "target_map.pcd"), target);
}

TEST_F(LoopCandidateArchiveTest, RefusesExistingEmptyDirectoryFileAndDanglingSymlink)
{
  const auto directory = root_ / "empty";
  std::filesystem::create_directory(directory);
  EXPECT_THROW(writeLoopCandidateArchive(directory.string(), fixture()), std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_empty(directory));
  const auto file = root_ / "file";
  std::ofstream(file) << "existing evidence";
  EXPECT_THROW(writeLoopCandidateArchive(file.string(), fixture()), std::runtime_error);
  EXPECT_EQ(readFile(file), "existing evidence");
  const auto link = root_ / "link";
  const auto missing = root_ / "missing";
  std::filesystem::create_symlink(missing, link);
  EXPECT_THROW(writeLoopCandidateArchive(link.string(), fixture()), std::runtime_error);
  EXPECT_EQ(std::filesystem::read_symlink(link), missing);
  EXPECT_FALSE(std::filesystem::exists(missing));
}

TEST_F(LoopCandidateArchiveTest, RefusesIncompleteDestinationWithoutDeletingPartialEvidence)
{
  const auto output = root_ / "incomplete";
  std::filesystem::create_directory(output);
  std::ofstream(output / "manifest.yaml.partial") << "interrupted manifest";
  std::ofstream(output / "source_base.pcd") << "partial source evidence";
  EXPECT_THROW(writeLoopCandidateArchive(output.string(), fixture()), std::runtime_error);
  EXPECT_FALSE(std::filesystem::exists(output / "manifest.yaml"));
  EXPECT_EQ(readFile(output / "manifest.yaml.partial"), "interrupted manifest");
  EXPECT_EQ(readFile(output / "source_base.pcd"), "partial source evidence");
}

TEST_F(LoopCandidateArchiveTest, RejectsInvalidMetadataBeforeCreatingDestination)
{
  EXPECT_THROW(writeLoopCandidateArchive("", fixture()), std::invalid_argument);
  const auto path = (root_ / "truncated").string() + std::string("\0suffix", 7);
  EXPECT_THROW(writeLoopCandidateArchive(path, fixture()), std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(root_ / "truncated"));
  for (const auto member : {&LoopCandidateArchive::map_frame, &LoopCandidateArchive::base_frame}) {
    for (const std::string frame : {"", " ", "bad\nframe"}) {
      auto archive = fixture();
      archive.*member = frame;
      expectInvalid(archive);
    }
  }
  auto archive = fixture();
  archive.stamp_ns = -1;
  expectInvalid(archive);
  archive = fixture();
  archive.candidate_keyframe_index = -1;
  expectInvalid(archive);
  archive = fixture();
  archive.candidate_submap_id = -2;
  expectInvalid(archive);
}

TEST_F(LoopCandidateArchiveTest, RejectsScalingShearReflectionAndInvalidHomogeneousRowInEveryPose)
{
  for (std::size_t index = 0; index < kPoseNames.size(); ++index) {
    SCOPED_TRACE(kPoseNames[index]);
    auto archive = fixture();
    poseAt(archive, index).linear() *= 2.0;
    expectInvalid(archive);
    archive = fixture();
    poseAt(archive, index).linear()(0, 1) += 0.3;
    expectInvalid(archive);
    archive = fixture();
    poseAt(archive, index).linear().col(0) *= -1.0;
    expectInvalid(archive);
    for (Eigen::Index column = 0; column < 4; ++column) {
      archive = fixture();
      poseAt(archive, index).matrix()(3, column) += 0.1;
      expectInvalid(archive);
    }
  }
}

TEST_F(LoopCandidateArchiveTest, RejectsNonfiniteValuesInEveryPoseMatrixElement)
{
  for (std::size_t index = 0; index < kPoseNames.size(); ++index) {
    SCOPED_TRACE(kPoseNames[index]);
    for (Eigen::Index row = 0; row < 4; ++row) {
      for (Eigen::Index column = 0; column < 4; ++column) {
        for (const double value : {
            std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()})
        {
          auto archive = fixture();
          poseAt(archive, index).matrix()(row, column) = value;
          expectInvalid(archive);
        }
      }
    }
  }
}

TEST_F(LoopCandidateArchiveTest, RejectsNullCloudsAndNonfiniteCoordinatesInBothClouds)
{
  for (const auto member : kClouds) {
    auto archive = fixture();
    (archive.*member).reset();
    expectInvalid(archive);
    for (Eigen::Index axis = 0; axis < 3; ++axis) {
      for (const float value : {
          std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity()})
      {
        archive = fixture();
        auto points = std::make_shared<Cloud>(*(archive.*member));
        points->back().getVector3fMap()[axis] = value;
        archive.*member = points;
        expectInvalid(archive);
      }
    }
  }
}

TEST_F(LoopCandidateArchiveTest, PreservesRejectedEmptyCloudsWithValidMetadata)
{
  for (int empty_mask = 1; empty_mask < 4; ++empty_mask) {
    auto archive = fixture();
    archive.stamp_ns = 0;
    archive.candidate_keyframe_index = 0;
    archive.candidate_submap_id = -1;
    archive.refined_map_T_base.reset();
    archive.diagnostics["reason"] = "sparse_target";
    const std::array<std::string, 2> names = {"source_base", "target_map"};
    for (std::size_t index = 0; index < kClouds.size(); ++index) {
      if (empty_mask & (1 << index)) {
        archive.*kClouds[index] = std::make_shared<Cloud>();
      }
    }
    const auto output = root_ / ("empty_" + std::to_string(empty_mask));
    writeLoopCandidateArchive(output.string(), archive);
    const auto manifest = YAML::LoadFile((output / "manifest.yaml").string());
    EXPECT_EQ(manifest["status"].as<std::string>(), "complete");
    EXPECT_EQ(manifest["candidate_submap_id"].as<int>(), -1);
    EXPECT_EQ(manifest["stamp_ns"].as<std::int64_t>(), 0);
    EXPECT_EQ(manifest["candidate_keyframe_index"].as<int>(), 0);
    EXPECT_TRUE(manifest["refined_map_T_base"].IsNull());
    EXPECT_EQ(manifest["diagnostics"]["reason"].as<std::string>(), "sparse_target");
    for (std::size_t index = 0; index < kClouds.size(); ++index) {
      const auto entry = manifest[names[index]];
      EXPECT_EQ(
        entry["frame"].as<std::string>(), index == 0 ? archive.base_frame : archive.map_frame);
      EXPECT_EQ(entry["point_count"].as<std::size_t>(), (archive.*kClouds[index])->size());
      if (!(empty_mask & (1 << index))) {
        continue;
      }
      const auto path = output / entry["pcd"].as<std::string>();
      const auto contents = readFile(path);
      EXPECT_NE(
        contents.find("FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"),
        std::string::npos);
      EXPECT_NE(contents.find("WIDTH 0\nHEIGHT 1\n"), std::string::npos);
      const std::string ending = "POINTS 0\nDATA binary\n";
      ASSERT_GE(contents.size(), ending.size());
      EXPECT_EQ(contents.substr(contents.size() - ending.size()), ending);
    }
    EXPECT_FALSE(std::filesystem::exists(output / "manifest.yaml.partial"));
  }
}

TEST_F(LoopCandidateArchiveTest, WritesEmptyDiagnosticsAsAMap)
{
  auto archive = fixture();
  archive.diagnostics.clear();
  const auto output = root_ / "archive";
  writeLoopCandidateArchive(output.string(), archive);
  const auto diagnostics = YAML::LoadFile((output / "manifest.yaml").string())["diagnostics"];
  EXPECT_TRUE(diagnostics.IsMap());
  EXPECT_EQ(diagnostics.size(), 0U);
}

TEST_F(LoopCandidateArchiveTest, DefaultsToNoRefinementAndWritesExplicitNull)
{
  EXPECT_FALSE(LoopCandidateArchive{}.refined_map_T_base.has_value());
  auto archive = fixture();
  archive.refined_map_T_base.reset();
  const auto output = root_ / "archive";
  writeLoopCandidateArchive(output.string(), archive);
  const auto manifest = YAML::LoadFile((output / "manifest.yaml").string());
  ASSERT_TRUE(manifest["refined_map_T_base"].IsDefined());
  EXPECT_TRUE(manifest["refined_map_T_base"].IsNull());
  EXPECT_EQ(manifest["status"].as<std::string>(), "complete");
}

}  // namespace
