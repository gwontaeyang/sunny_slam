#ifndef SLAM_3D_LIVOX_INGEST__MID360_INGEST_HPP_
#define SLAM_3D_LIVOX_INGEST__MID360_INGEST_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "slam_3d_livox_ingest/mid360_time_contract.hpp"

namespace gole
{
namespace slam_3d_livox_ingest
{

class Mid360Ingest : public rclcpp::Node
{
public:
  explicit Mid360Ingest(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  using CustomMsg = livox_ros_driver2::msg::CustomMsg;

  struct SensorState
  {
    std::string name;
    std::string frame_id;
    std::string clock_domain{"unverified"};
    std::int64_t time_offset_ns{0};
    std::uint64_t last_accepted_timebase_ns{0};
    std::int64_t last_accepted_corrected_timebase_ns{0};
    std::uint64_t accepted_scan_count{0};
    std::uint64_t rejected_scan_count{0};
    rclcpp::Subscription<CustomMsg>::SharedPtr subscription;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher;
  };

  void processMessage(const CustomMsg::SharedPtr & message, SensorState & sensor);
  sensor_msgs::msg::PointCloud2 buildCanonicalCloud(
    const CustomMsg & message,
    const SensorState & sensor,
    const TimeContractResult & result) const;
  void publishDiagnostics(
    const CustomMsg & message,
    const SensorState & sensor,
    const TimeContractResult & result,
    double header_timebase_delta_ms,
    double peer_start_skew_ms) const;
  static std::int64_t stampNanoseconds(const builtin_interfaces::msg::Time & stamp);
  static std::int64_t correctedTimebaseNanoseconds(
    std::uint64_t timebase_ns,
    std::int64_t offset_ns);
  const SensorState & peerOf(const SensorState & sensor) const;

  SensorState front_;
  SensorState rear_;
  TimeContractOptions contract_options_;
  bool reject_non_increasing_timebase_{true};
  bool drop_nonfinite_points_{true};
  double max_header_timebase_delta_ms_{5.0};
  int input_depth_{10};
  int output_depth_{5};
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr health_publisher_;
};

}  // namespace slam_3d_livox_ingest
}  // namespace gole

#endif  // SLAM_3D_LIVOX_INGEST__MID360_INGEST_HPP_
