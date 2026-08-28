#include <exception>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
#include "slam_3d_core/scan_to_map_icp.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int exit_code = 0;

  try {
    const auto node = std::make_shared<gole::slam_3d_core::ScanToMapIcp>(
      rclcpp::NodeOptions());
    const auto executor_thread_count = static_cast<std::size_t>(
      node->get_parameter("executor_thread_count").as_int());
    rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), executor_thread_count);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("scan_to_map_icp_node"),
      "Unhandled exception: %s",
      ex.what());
    exit_code = 1;
  }

  rclcpp::shutdown();
  return exit_code;
}
