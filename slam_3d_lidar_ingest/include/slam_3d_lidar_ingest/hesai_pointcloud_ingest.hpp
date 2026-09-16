#ifndef SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_INGEST_HPP_
#define SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_INGEST_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "builtin_interfaces/msg/time.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "slam_3d_lidar_ingest/hesai_pointcloud_contract.hpp"

namespace gole
{
namespace slam_3d_lidar_ingest
{

class HesaiPointCloudIngest : public rclcpp::Node
{
public:
  explicit HesaiPointCloudIngest(
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  struct SensorState
  {
    std::string name;
    std::string model;
    std::string clock_domain{"unverified"};
    ClockPolicy clock_policy{ClockPolicy::kSensorHeader};
    std::uint8_t sensor_id{0};
    std::int64_t time_offset_ns{0};
    double time_scale{1.0};
    std::int64_t anchor_input_header_ns{0};
    std::int64_t anchor_output_start_ns{0};
    std::int64_t calibration_valid_min_input_header_ns{0};
    std::int64_t calibration_valid_max_input_header_ns{0};
    double max_output_stamp_step_s{0.0};
    double calibration_fit_rmse_ms{-1.0};
    double calibration_fit_max_abs_ms{-1.0};
    std::string calibration_id{"none"};
    std::atomic<std::int64_t> last_output_start_ns{0};
    std::uint64_t accepted_scan_count{0};
    std::uint64_t rejected_scan_count{0};
    std::size_t last_output_point_count{0};
    HesaiCloudContractOptions contract_options;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher;
    rclcpp::CallbackGroup::SharedPtr callback_group;
  };

  void configureSensor(
    std::size_t sensor_index,
    const std::string & input_topic,
    const std::string & output_topic,
    const std::string & expected_frame_id,
    const std::string & model,
    std::uint8_t sensor_id,
    std::uint16_t max_ring_id,
    std::size_t min_points,
    std::size_t max_points,
    const rclcpp::QoS & input_qos,
    const rclcpp::QoS & output_qos);
  void processMessage(
    const sensor_msgs::msg::PointCloud2::SharedPtr & message,
    std::size_t sensor_index);
  sensor_msgs::msg::PointCloud2 buildCanonicalCloud(
    const sensor_msgs::msg::PointCloud2 & message,
    const SensorState & sensor,
    const HesaiCloudContractResult & result,
    std::int64_t output_start_ns) const;
  void publishDiagnostics(
    const sensor_msgs::msg::PointCloud2 & message,
    const SensorState & sensor,
    const HesaiCloudContractResult & result,
    std::int64_t receive_time_ns,
    std::int64_t output_start_ns,
    double peer_start_skew_ms) const;
  double peerStartSkewMs(const SensorState & sensor, std::int64_t output_start_ns) const;
  static std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time & stamp);

  std::array<SensorState, 3> sensors_;
  bool drop_nonfinite_points_{true};
  bool reject_non_increasing_output_stamp_{true};
  bool reject_clock_misalignment_{true};
  double max_sensor_header_to_receive_delta_s_{0.25};
  int input_depth_{10};
  int output_depth_{5};
  std::size_t max_output_points_{0};
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_publisher_;
};

}  // namespace slam_3d_lidar_ingest
}  // namespace gole

#endif  // SLAM_3D_LIDAR_INGEST__HESAI_POINTCLOUD_INGEST_HPP_
