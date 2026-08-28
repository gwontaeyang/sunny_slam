#ifndef SLAM_3D_LIDAR_INGEST__HESAI_IMU_NORMALIZER_HPP_
#define SLAM_3D_LIDAR_INGEST__HESAI_IMU_NORMALIZER_HPP_

#include <atomic>
#include <cstdint>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "slam_3d_lidar_ingest/hesai_pointcloud_contract.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

struct ImuNormalizationResult
{
  CloudContractCode code{CloudContractCode::kValid};
  sensor_msgs::msg::Imu message;
  std::int64_t input_stamp_ns{0};
  std::int64_t output_stamp_ns{0};

  bool accepted() const
  {
    return code == CloudContractCode::kValid;
  }
};

struct ImuNormalizationOptions
{
  ClockPolicy clock_policy{ClockPolicy::kSensorHeader};
  std::string expected_input_frame_id;
  std::string output_frame_id;
  bool require_expected_input_frame{false};
  std::int64_t time_offset_ns{0};
  double time_scale{1.0};
  std::int64_t anchor_input_header_ns{0};
  std::int64_t anchor_output_header_ns{0};
  std::int64_t calibration_valid_min_input_header_ns{0};
  std::int64_t calibration_valid_max_input_header_ns{0};
  double angular_velocity_scale{1.0};
  double linear_acceleration_scale{1.0};
};

ImuNormalizationResult normalizeHesaiImu(
  const sensor_msgs::msg::Imu & input,
  std::int64_t receive_time_ns,
  const ImuNormalizationOptions & options);

class HesaiImuNormalizer : public rclcpp::Node
{
public:
  explicit HesaiImuNormalizer(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void processMessage(const sensor_msgs::msg::Imu::SharedPtr message);
  void publishDiagnostics(
    const sensor_msgs::msg::Imu & input,
    const ImuNormalizationResult & result,
    std::int64_t receive_time_ns);

  ImuNormalizationOptions options_;
  std::string clock_domain_{"unverified"};
  std::string calibration_id_{"none"};
  double calibration_fit_rmse_ms_{-1.0};
  double calibration_fit_max_abs_ms_{-1.0};
  double max_output_stamp_step_s_{0.0};
  double diagnostic_period_s_{1.0};
  bool reject_non_increasing_output_stamp_{true};
  std::atomic<std::int64_t> last_output_stamp_ns_{0};
  std::atomic<std::int64_t> last_diagnostic_publish_ns_{0};
  std::uint64_t accepted_count_{0};
  std::uint64_t rejected_count_{0};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr
    health_publisher_;
};

}  // namespace slam_3d_lidar_ingest
}  // namespace gole

#endif  // SLAM_3D_LIDAR_INGEST__HESAI_IMU_NORMALIZER_HPP_
