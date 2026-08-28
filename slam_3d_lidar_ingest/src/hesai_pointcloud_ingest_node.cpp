#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "slam_3d_lidar_ingest/hesai_pointcloud_ingest.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gole::slam_3d_lidar_ingest::HesaiPointCloudIngest>();
  rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(),
    3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
