#include "slam_3d_core/keyframe_archive.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <unistd.h>

namespace
{
using gole::slam_3d_core::ArchivedKeyframe;
using gole::slam_3d_core::ArchivedSensorScan;
using gole::slam_3d_core::KeyframeArchive;
using gole::slam_3d_core::writeKeyframeArchive;

class KeyframeArchiveTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::string pattern = (std::filesystem::temp_directory_path() / "keyframe_archive_XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const auto directory = mkdtemp(buffer.data());
    ASSERT_NE(directory, nullptr);
    root_ = directory;
  }

  void TearDown() override
  {
    std::filesystem::remove_all(root_);
  }

  KeyframeArchive fixture()
  {
    KeyframeArchive archive;
    archive.map_frame = "map";
    archive.base_frame = "base_link";
    auto points = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    points->push_back(pcl::PointXYZ(1.0F, 2.0F, 3.0F));
    ArchivedKeyframe keyframe;
    keyframe.id = 8;
    keyframe.stamp_ns = 1780000000123456789LL;
    keyframe.map_T_base.translation() = Eigen::Vector3d(3.1, -4.2, 5.3);
    keyframe.map_T_base.linear() = Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitY()).toRotationMatrix();
    keyframe.scans.push_back(ArchivedSensorScan{points, Eigen::Vector3d(0.4, -0.2, 1.5)});
    archive.keyframes.push_back(keyframe);
    return archive;
  }

  std::filesystem::path root_;
};

TEST_F(KeyframeArchiveTest, SavesFinalPoseAndUntransformedSensorEvidence)
{
  auto archive = fixture();
  archive.pruned_keyframes = 8;
  const auto output = root_ / "archive";
  writeKeyframeArchive(output.string(), archive);
  const auto manifest = YAML::LoadFile((output / "manifest.yaml").string());
  EXPECT_EQ(manifest["schema_version"].as<int>(), 1);
  EXPECT_EQ(manifest["pruned_keyframes"].as<int>(), 8);
  EXPECT_FALSE(manifest["complete_session"].as<bool>());
  const auto entry = manifest["keyframes"][0];
  EXPECT_EQ(entry["id"].as<int>(), 8);
  EXPECT_EQ(entry["stamp_ns"].as<std::int64_t>(), archive.keyframes[0].stamp_ns);
  const auto pose = entry["pose"].as<std::vector<double>>();
  ASSERT_EQ(pose.size(), 7U);
  EXPECT_DOUBLE_EQ(pose[0], 3.1);
  EXPECT_DOUBLE_EQ(pose[2], 5.3);
  const Eigen::Quaterniond rotation(pose[6], pose[3], pose[4], pose[5]);
  EXPECT_TRUE(rotation.toRotationMatrix().isApprox(archive.keyframes[0].map_T_base.rotation()));
  EXPECT_DOUBLE_EQ(entry["scans"][0]["origin_base"][2].as<double>(), 1.5);
  pcl::PointCloud<pcl::PointXYZ> cloud;
  ASSERT_EQ(pcl::io::loadPCDFile((output / entry["scans"][0]["pcd"].as<std::string>()).string(), cloud), 0);
  ASSERT_EQ(cloud.size(), 1U);
  EXPECT_FLOAT_EQ(cloud[0].x, 1.0F);
  EXPECT_FLOAT_EQ(cloud[0].z, 3.0F);
  EXPECT_FALSE(std::filesystem::exists(output / "manifest.yaml.partial"));
}

TEST_F(KeyframeArchiveTest, RefusesOverwriteWithoutChangingEvidence)
{
  const auto output = root_ / "archive";
  writeKeyframeArchive(output.string(), fixture());
  const auto size = std::filesystem::file_size(output / "manifest.yaml");
  EXPECT_THROW(writeKeyframeArchive(output.string(), fixture()), std::runtime_error);
  EXPECT_EQ(std::filesystem::file_size(output / "manifest.yaml"), size);
}

TEST_F(KeyframeArchiveTest, ValidatesAllFramesBeforeCreatingDestination)
{
  auto archive = fixture();
  archive.keyframes.push_back(archive.keyframes[0]);
  const auto output = root_ / "archive";
  EXPECT_THROW(writeKeyframeArchive(output.string(), archive), std::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(output));
}

TEST_F(KeyframeArchiveTest, RejectsMissingViewsAndNonRigidPose)
{
  auto archive = fixture();
  archive.keyframes[0].scans.clear();
  EXPECT_THROW(writeKeyframeArchive((root_ / "missing").string(), archive), std::invalid_argument);
  archive = fixture();
  archive.keyframes[0].map_T_base.linear()(0, 0) = 2.0;
  EXPECT_THROW(writeKeyframeArchive((root_ / "pose").string(), archive), std::invalid_argument);
}

TEST_F(KeyframeArchiveTest, RejectsNonfiniteOriginAndPoints)
{
  auto archive = fixture();
  archive.keyframes[0].scans[0].origin_base.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(writeKeyframeArchive((root_ / "origin").string(), archive), std::invalid_argument);
  archive = fixture();
  auto points = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  points->push_back(pcl::PointXYZ(std::numeric_limits<float>::infinity(), 0, 0));
  archive.keyframes[0].scans[0].cloud_base = points;
  EXPECT_THROW(writeKeyframeArchive((root_ / "point").string(), archive), std::invalid_argument);
}

TEST_F(KeyframeArchiveTest, ReportsCompleteSessionForUnprunedFrames)
{
  writeKeyframeArchive((root_ / "archive").string(), fixture());
  EXPECT_TRUE(YAML::LoadFile((root_ / "archive/manifest.yaml").string())["complete_session"].as<bool>());
}

}  // namespace
