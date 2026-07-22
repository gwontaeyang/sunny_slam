#include <exception>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include "slam_3d_core/scan_to_map_icp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    const auto node = std::make_shared<gole::slam_3d_core::ScanToMapIcp>(
      rclcpp::NodeOptions());
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("scan_to_map_icp_node"),
      "Unhandled exception: %s",
      ex.what());
  }

  rclcpp::shutdown();
  return 0;
}
