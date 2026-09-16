#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "sensor_msgs/msg/point_field.hpp"
#include "slam_3d_lidar_ingest/hesai_pointcloud_contract.hpp"
#include "slam_3d_lidar_ingest/point_budget.hpp"

namespace
{

using gole::slam_3d_lidar_ingest::ClockPolicy;
using gole::slam_3d_lidar_ingest::CloudContractCode;
using gole::slam_3d_lidar_ingest::HesaiCloudContractOptions;
using gole::slam_3d_lidar_ingest::cloudContractCodeString;
using gole::slam_3d_lidar_ingest::clockPolicyString;
using gole::slam_3d_lidar_ingest::parseClockPolicy;
using gole::slam_3d_lidar_ingest::selectOutputPointOffset;
using gole::slam_3d_lidar_ingest::selectOutputStartStamp;
using gole::slam_3d_lidar_ingest::validateHesaiPointCloud;

constexpr std::uint32_t kPointStep = 26;

TEST(PointBudget, PreservesScanExtentOrderingAndEveryRing)
{
  using gole::slam_3d_lidar_ingest::stratifiedPointIndex;
  for (const std::size_t total : {128000U, 240000U, 460800U}) {
    for (const std::size_t budget : {2U, 31U, 2000U, 4000U, 6000U, 12000U}) {
      std::size_t previous = 0;
      std::set<std::size_t> rings;
      for (std::size_t i = 0; i < budget; ++i) {
        const auto index = stratifiedPointIndex(i, budget, total, 3);
        ASSERT_LT(index, total);
        if (i > 0) {
          ASSERT_GT(index, previous);
        }
        previous = index;
        rings.insert(index % 128);
      }
      EXPECT_EQ(stratifiedPointIndex(0, budget, total, 3), 0U);
      EXPECT_EQ(previous, total - 1);
      if (budget >= 2000U) {
        EXPECT_EQ(rings.size(), 128U);
      }
    }
  }
}

TEST(PointBudget, FullBudgetIsIdentityAndSeedChangesInteriorSamples)
{
  using gole::slam_3d_lidar_ingest::stratifiedPointIndex;
  std::size_t changed = 0;
  for (std::size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(stratifiedPointIndex(i, 1000, 1000, 0), i);
    const auto sample = stratifiedPointIndex(i, 1000, 128000, 1);
    EXPECT_EQ(sample, stratifiedPointIndex(i, 1000, 128000, 1));
    changed += sample != stratifiedPointIndex(i, 1000, 128000, 2);
  }
  EXPECT_GT(changed, 900U);
}

sensor_msgs::msg::PointField field(
  const std::string & name,
  const std::uint32_t offset,
  const std::uint8_t datatype)
{
  sensor_msgs::msg::PointField output;
  output.name = name;
  output.offset = offset;
  output.datatype = datatype;
  output.count = 1;
  return output;
}

template<typename T>
void writeValue(
  std::vector<std::uint8_t> & data,
  const std::size_t point_index,
  const std::uint32_t offset,
  const T value)
{
  std::memcpy(
    data.data() + point_index * kPointStep + offset,
    &value,
    sizeof(T));
}

sensor_msgs::msg::PointCloud2 makeCloud(
  const std::uint16_t max_ring = 127,
  const double duration_s = 0.1)
{
  sensor_msgs::msg::PointCloud2 cloud;
  cloud.header.stamp.sec = 100;
  cloud.header.frame_id = "hesai_test";
  cloud.height = 1;
  cloud.width = 4;
  cloud.fields = {
    field("x", 0, sensor_msgs::msg::PointField::FLOAT32),
    field("y", 4, sensor_msgs::msg::PointField::FLOAT32),
    field("z", 8, sensor_msgs::msg::PointField::FLOAT32),
    field("intensity", 12, sensor_msgs::msg::PointField::FLOAT32),
    field("ring", 16, sensor_msgs::msg::PointField::UINT16),
    field("timestamp", 18, sensor_msgs::msg::PointField::FLOAT64),
  };
  cloud.is_bigendian = false;
  cloud.point_step = kPointStep;
  cloud.row_step = cloud.width * cloud.point_step;
  cloud.data.resize(cloud.row_step);
  cloud.is_dense = true;

  for (std::size_t index = 0; index < cloud.width; ++index) {
    const float x = static_cast<float>(index + 1);
    const float y = static_cast<float>(index) * 0.25F;
    const float z = 0.5F;
    const float intensity = static_cast<float>(10 + index);
    const auto ring = static_cast<std::uint16_t>(
      index == cloud.width - 1 ? max_ring : index);
    const double timestamp =
      100.0 + duration_s * static_cast<double>(index) /
      static_cast<double>(cloud.width - 1);
    writeValue(cloud.data, index, 0, x);
    writeValue(cloud.data, index, 4, y);
    writeValue(cloud.data, index, 8, z);
    writeValue(cloud.data, index, 12, intensity);
    writeValue(cloud.data, index, 16, ring);
    writeValue(cloud.data, index, 18, timestamp);
  }
  return cloud;
}

TEST(HesaiPointCloudContract, AcceptsObservedJt128Layout)
{
  const auto result = validateHesaiPointCloud(
    makeCloud(127),
    HesaiCloudContractOptions{});

  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.code, CloudContractCode::kValid);
  EXPECT_EQ(result.point_count, 4U);
  EXPECT_EQ(result.finite_point_count, 4U);
  EXPECT_EQ(result.observed_max_ring, 127U);
  EXPECT_EQ(result.timestamp_change_count, 3U);
  EXPECT_EQ(result.max_consecutive_equal_timestamps, 1U);
  EXPECT_NEAR(
    static_cast<double>(result.scan_duration_ns) * 1.0e-9,
    0.1,
    1.0e-8);
  EXPECT_STREQ(cloudContractCodeString(result.code), "valid");
}

TEST(HesaiPointCloudContract, AcceptsXt32WithModelSpecificRingLimit)
{
  auto options = HesaiCloudContractOptions{};
  options.max_ring_id = 31;
  const auto result = validateHesaiPointCloud(makeCloud(31), options);

  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.observed_max_ring, 31U);
}

TEST(HesaiPointCloudContract, RejectsRingOutsideModelLimit)
{
  auto options = HesaiCloudContractOptions{};
  options.max_ring_id = 31;
  const auto result = validateHesaiPointCloud(makeCloud(32), options);

  EXPECT_EQ(result.code, CloudContractCode::kRingOutOfRange);
}

TEST(HesaiPointCloudContract, ReportsPacketTimestampPlateaus)
{
  auto cloud = makeCloud();
  const double first_packet_timestamp = 100.0;
  const double second_packet_timestamp = 100.1;
  writeValue(cloud.data, 0, 18, first_packet_timestamp);
  writeValue(cloud.data, 1, 18, first_packet_timestamp);
  writeValue(cloud.data, 2, 18, second_packet_timestamp);
  writeValue(cloud.data, 3, 18, second_packet_timestamp);
  const auto result = validateHesaiPointCloud(
    cloud,
    HesaiCloudContractOptions{});

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.timestamp_change_count, 1U);
  EXPECT_EQ(result.max_consecutive_equal_timestamps, 2U);
}

TEST(HesaiPointCloudContract, RejectsUnexpectedInputFrame)
{
  auto cloud = makeCloud();
  auto options = HesaiCloudContractOptions{};
  options.require_expected_frame = true;
  options.expected_frame_id = "hesai_front";
  const auto result = validateHesaiPointCloud(cloud, options);

  EXPECT_EQ(result.code, CloudContractCode::kUnexpectedFrameId);
}

TEST(HesaiPointCloudContract, RejectsPointCountOutsideModelProfile)
{
  auto options = HesaiCloudContractOptions{};
  options.min_points = 5;
  const auto result = validateHesaiPointCloud(makeCloud(), options);

  EXPECT_EQ(result.code, CloudContractCode::kTooFewPoints);
}

TEST(HesaiPointCloudContract, RejectsMissingTimestampField)
{
  auto cloud = makeCloud();
  cloud.fields.pop_back();
  const auto result = validateHesaiPointCloud(
    cloud,
    HesaiCloudContractOptions{});

  EXPECT_EQ(result.code, CloudContractCode::kMissingRequiredField);
  EXPECT_EQ(result.failed_field, "timestamp");
}

TEST(HesaiPointCloudContract, RejectsTimestampRegression)
{
  auto cloud = makeCloud();
  const double regressed_timestamp = 99.9;
  writeValue(cloud.data, 2, 18, regressed_timestamp);
  const auto result = validateHesaiPointCloud(
    cloud,
    HesaiCloudContractOptions{});

  EXPECT_EQ(result.code, CloudContractCode::kTimestampRegression);
  EXPECT_EQ(result.timestamp_regression_count, 1U);
}

TEST(HesaiPointCloudContract, RejectsNonfiniteXyzRatio)
{
  auto cloud = makeCloud();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  writeValue(cloud.data, 1, 0, nan);
  auto options = HesaiCloudContractOptions{};
  options.min_finite_point_ratio = 0.9;
  const auto result = validateHesaiPointCloud(cloud, options);

  EXPECT_EQ(result.code, CloudContractCode::kInsufficientFinitePoints);
  EXPECT_EQ(result.invalid_point_count, 1U);
}

TEST(HesaiPointCloudContract, RejectsHeaderFirstPointMismatch)
{
  auto cloud = makeCloud();
  cloud.header.stamp.nanosec = 10000000;
  const auto result = validateHesaiPointCloud(
    cloud,
    HesaiCloudContractOptions{});

  EXPECT_EQ(result.code, CloudContractCode::kHeaderTimestampMismatch);
}

TEST(HesaiPointCloudContract, SelectsSensorHeaderWithCalibratedOffset)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kSensorHeader,
    10000000000LL,
    20000000000LL,
    100000000ULL,
    2000000LL);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_start_ns, 10002000000LL);
}

TEST(HesaiPointCloudContract, ReanchorsLegacyBagAtReceiveTimeMinusDuration)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kRosReceiveTimeScanEnd,
    123LL,
    20000000000LL,
    100000000ULL,
    -1000000LL);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_start_ns, 19899000000LL);
}

TEST(HesaiPointCloudContract, AdvancesAnchoredOutputWithSensorHeaderDelta)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kAnchoredSensorTime,
    100250000000LL,
    900000000000LL,
    100000000ULL,
    0,
    100000000000LL,
    20000000000LL);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_start_ns, 20250000000LL);
}

TEST(HesaiPointCloudContract, ParsesAnchoredSensorTimePolicy)
{
  const auto parsed = parseClockPolicy("anchored_sensor_time");

  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(*parsed, ClockPolicy::kAnchoredSensorTime);
  EXPECT_STREQ(
    clockPolicyString(ClockPolicy::kAnchoredSensorTime),
    "anchored_sensor_time");
}

TEST(HesaiPointCloudContract, AppliesCalibratedSensorClockScale)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kCalibratedSensorTime,
    100250000000LL,
    900000000000LL,
    100000000ULL,
    1000LL,
    100000000000LL,
    20000000000LL,
    1.5);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_start_ns, 20375001000LL);
}

TEST(HesaiPointCloudContract, ParsesCalibratedSensorTimePolicy)
{
  const auto parsed = parseClockPolicy("calibrated_sensor_time");

  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(*parsed, ClockPolicy::kCalibratedSensorTime);
  EXPECT_STREQ(
    clockPolicyString(ClockPolicy::kCalibratedSensorTime),
    "calibrated_sensor_time");
}

TEST(HesaiPointCloudContract, RejectsInvalidCalibratedSensorClockScale)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kCalibratedSensorTime,
    100250000000LL,
    900000000000LL,
    100000000ULL,
    0,
    100000000000LL,
    20000000000LL,
    0.0);

  EXPECT_EQ(result.code, CloudContractCode::kClockMisaligned);
}

TEST(HesaiPointCloudContract, ScalesCalibratedPointOffset)
{
  const auto result = selectOutputPointOffset(
    0.1,
    ClockPolicy::kCalibratedSensorTime,
    1.5);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_offset_ns, 150000000U);
}

TEST(HesaiPointCloudContract, RejectsScaledPointOffsetOverflow)
{
  const auto result = selectOutputPointOffset(
    3.0,
    ClockPolicy::kCalibratedSensorTime,
    2.0);

  EXPECT_EQ(result.code, CloudContractCode::kTimestampOffsetOverflow);
}

TEST(HesaiPointCloudContract, RejectsAnchoredPolicyWithoutAnchor)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kAnchoredSensorTime,
    100000000000LL,
    20000000000LL,
    100000000ULL,
    0);

  EXPECT_EQ(result.code, CloudContractCode::kMissingHeaderTimestamp);
}

TEST(HesaiPointCloudContract, RejectsOutputOutsideRosTimeRange)
{
  const auto result = selectOutputStartStamp(
    ClockPolicy::kSensorHeader,
    1LL,
    1LL,
    0U,
    -2LL);

  EXPECT_EQ(result.code, CloudContractCode::kOutputTimeOutsideRosRange);
}

}  // namespace
