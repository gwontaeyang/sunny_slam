#include <cmath>
#include <limits>

#include "gtest/gtest.h"
#include "sensor_msgs/msg/imu.hpp"
#include "slam_3d_lidar_ingest/hesai_imu_normalizer.hpp"

namespace
{

constexpr double kPi = 3.14159265358979323846;

sensor_msgs::msg::Imu makeImu()
{
  sensor_msgs::msg::Imu message;
  message.header.stamp.sec = 10;
  message.header.frame_id = "hesai_front";
  message.angular_velocity.x = 180.0;
  message.angular_velocity.y = -90.0;
  message.angular_velocity.z = 45.0;
  message.linear_acceleration.x = 1.0;
  message.linear_acceleration.y = -0.5;
  message.linear_acceleration.z = -1.0;
  message.angular_velocity_covariance[0] = 4.0;
  message.linear_acceleration_covariance[0] = 9.0;
  return message;
}

}  // namespace

TEST(HesaiImuNormalizer, ConvertsLegacyUnitsAndCalibratedClock)
{
  auto options =
    gole::slam_3d_lidar_ingest::ImuNormalizationOptions{};
  options.clock_policy =
    gole::slam_3d_lidar_ingest::ClockPolicy::kCalibratedSensorTime;
  options.anchor_input_header_ns = 10000000000LL;
  options.anchor_output_header_ns = 20000000000LL;
  options.time_scale = 1.5;
  options.angular_velocity_scale = kPi / 180.0;
  options.linear_acceleration_scale = 9.80665;
  options.expected_input_frame_id = "hesai_front";
  options.output_frame_id = "base_imu";
  options.require_expected_input_frame = true;
  options.calibration_valid_min_input_header_ns = 9000000000LL;
  options.calibration_valid_max_input_header_ns = 11000000000LL;

  const auto result =
    gole::slam_3d_lidar_ingest::normalizeHesaiImu(
    makeImu(),
    100000000000LL,
    options);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.output_stamp_ns, 20000000000LL);
  EXPECT_EQ(result.message.header.frame_id, "base_imu");
  EXPECT_NEAR(result.message.angular_velocity.x, kPi, 1.0e-12);
  EXPECT_NEAR(result.message.angular_velocity.y, -kPi / 2.0, 1.0e-12);
  EXPECT_NEAR(result.message.linear_acceleration.z, -9.80665, 1.0e-12);
  EXPECT_NEAR(
    result.message.angular_velocity_covariance[0],
    4.0 * std::pow(kPi / 180.0, 2),
    1.0e-12);
  EXPECT_NEAR(
    result.message.linear_acceleration_covariance[0],
    9.0 * std::pow(9.80665, 2),
    1.0e-9);
}

TEST(HesaiImuNormalizer, PreservesUnknownCovariance)
{
  auto message = makeImu();
  message.angular_velocity_covariance[0] = -1.0;
  auto options =
    gole::slam_3d_lidar_ingest::ImuNormalizationOptions{};
  options.angular_velocity_scale = 0.5;

  const auto result =
    gole::slam_3d_lidar_ingest::normalizeHesaiImu(
    message,
    10000000000LL,
    options);

  ASSERT_TRUE(result.accepted());
  EXPECT_EQ(result.message.angular_velocity_covariance[0], -1.0);
}

TEST(HesaiImuNormalizer, RejectsUnexpectedFrame)
{
  auto options =
    gole::slam_3d_lidar_ingest::ImuNormalizationOptions{};
  options.expected_input_frame_id = "hesai_rear";
  options.require_expected_input_frame = true;

  const auto result =
    gole::slam_3d_lidar_ingest::normalizeHesaiImu(
    makeImu(),
    10000000000LL,
    options);

  EXPECT_EQ(
    result.code,
    gole::slam_3d_lidar_ingest::CloudContractCode::kUnexpectedFrameId);
}

TEST(HesaiImuNormalizer, RejectsNonfiniteMeasurement)
{
  auto message = makeImu();
  message.angular_velocity.z = std::numeric_limits<double>::quiet_NaN();

  const auto result =
    gole::slam_3d_lidar_ingest::normalizeHesaiImu(
    message,
    10000000000LL,
    gole::slam_3d_lidar_ingest::ImuNormalizationOptions{});

  EXPECT_EQ(
    result.code,
    gole::slam_3d_lidar_ingest::CloudContractCode::kNonFiniteMeasurement);
}

TEST(HesaiImuNormalizer, RejectsCalibrationExtrapolation)
{
  auto options =
    gole::slam_3d_lidar_ingest::ImuNormalizationOptions{};
  options.clock_policy =
    gole::slam_3d_lidar_ingest::ClockPolicy::kCalibratedSensorTime;
  options.anchor_input_header_ns = 10000000000LL;
  options.anchor_output_header_ns = 20000000000LL;
  options.calibration_valid_min_input_header_ns = 11000000000LL;

  const auto result =
    gole::slam_3d_lidar_ingest::normalizeHesaiImu(
    makeImu(),
    10000000000LL,
    options);

  EXPECT_EQ(
    result.code,
    gole::slam_3d_lidar_ingest::CloudContractCode::kClockCalibrationOutOfRange);
}
