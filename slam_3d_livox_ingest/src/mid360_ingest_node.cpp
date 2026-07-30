#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "slam_3d_livox_ingest/mid360_ingest.hpp"

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<gole::slam_3d_livox_ingest::Mid360Ingest>());
  rclcpp::shutdown();
  return 0;
}
