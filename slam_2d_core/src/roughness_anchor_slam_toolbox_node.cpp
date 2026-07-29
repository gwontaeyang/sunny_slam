#include <atomic>
#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "slam_2d_core/roughness_anchor_slam_toolbox.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<slam_2d_core::RoughnessAnchorSlamToolbox>(
      rclcpp::NodeOptions());
    node->configure();

    std::atomic_bool processing_failed{false};
    std::thread processing_thread([node, &processing_failed]() {
        const auto handle_processing_exception =
        [&processing_failed](const std::string & message) {
          if (rclcpp::ok()) {
            processing_failed = true;
            RCLCPP_ERROR(
              rclcpp::get_logger("roughness_anchor_slam_toolbox_node"),
              "SLAM processing thread failed: %s",
              message.c_str());
            rclcpp::shutdown();
          } else {
            RCLCPP_DEBUG(
              rclcpp::get_logger("roughness_anchor_slam_toolbox_node"),
              "SLAM processing thread stopped during shutdown: %s",
              message.c_str());
          }
        };
        try {
          node->run();
        } catch (const karto::Exception & ex) {
          handle_processing_exception(ex.GetErrorMessage());
        } catch (const std::exception & ex) {
          handle_processing_exception(ex.what());
        } catch (...) {
          handle_processing_exception("unknown exception");
        }
      });

    rclcpp::spin(node->get_node_base_interface());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    if (processing_thread.joinable()) {
      processing_thread.join();
    }
    if (processing_failed) {
      return 1;
    }
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
