#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "slam_2d_core/roughness_anchor_slam_toolbox.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<slam_2d_core::RoughnessAnchorSlamToolbox>(
      rclcpp::NodeOptions());
    node->configure();
    node->loadPoseGraphByParams();

    rclcpp::spin(node->get_node_base_interface());
  } catch (const karto::Exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("roughness_anchor_slam_toolbox_node"),
      "Unhandled Karto exception: %s",
      ex.GetErrorMessage().c_str());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("roughness_anchor_slam_toolbox_node"),
      "Unhandled exception: %s",
      ex.what());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return 0;
}
