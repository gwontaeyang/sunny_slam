#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "slam_3d_lidar_ingest/hesai_imu_normalizer.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(
    std::make_shared<gole::slam_3d_lidar_ingest::HesaiImuNormalizer>());
  rclcpp::shutdown();
  return 0;
}
