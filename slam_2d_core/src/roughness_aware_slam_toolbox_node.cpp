#include <memory>

#include <sys/resource.h>

#include "rclcpp/rclcpp.hpp"
#include "slam_2d_core/roughness_aware_slam_toolbox.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int exit_code = 0;
  std::shared_ptr<slam_2d_core::RoughnessAwareSlamToolbox> node;

  try {
    int stack_size = 40000000;
    {
      auto parameter_node = std::make_shared<rclcpp::Node>("slam_toolbox");
      parameter_node->declare_parameter(
        "stack_size_to_use", rclcpp::ParameterType::PARAMETER_INTEGER);
      if (parameter_node->get_parameter("stack_size_to_use", stack_size)) {
        struct rlimit stack_limit;
        if (getrlimit(RLIMIT_STACK, &stack_limit) == 0 &&
          stack_limit.rlim_cur < static_cast<rlim_t>(stack_size))
        {
          stack_limit.rlim_cur = static_cast<rlim_t>(stack_size);
          setrlimit(RLIMIT_STACK, &stack_limit);
        }
      }
    }

    node = std::make_shared<slam_2d_core::RoughnessAwareSlamToolbox>(
      rclcpp::NodeOptions());
    node->configure();
    node->loadPoseGraphByParams();
    rclcpp::spin(node->get_node_base_interface());
  } catch (const karto::Exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("roughness_aware_slam_toolbox_node"),
      "Unhandled Karto exception: %s",
      ex.GetErrorMessage().c_str());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    exit_code = 1;
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(
      rclcpp::get_logger("roughness_aware_slam_toolbox_node"),
      "Unhandled exception: %s",
      ex.what());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    exit_code = 1;
  }

  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  if (node) {
    node->shutdown();
    node.reset();
  }
  return exit_code;
}
