#ifndef SLAM_3D_CORE__PARAMETER_PROFILE_HPP_
#define SLAM_3D_CORE__PARAMETER_PROFILE_HPP_

#include <map>
#include <string>

#include "rclcpp/rclcpp.hpp"

namespace gole
{
namespace slam_3d_core
{

class ParameterProfile
{
public:
  explicit ParameterProfile(rclcpp::Node & node);

  template<typename T>
  auto declare(const std::string & name, const T & fallback)
  {
    const auto found = defaults_.find(name);
    if (found == defaults_.end()) {
      return node_.declare_parameter<T>(name, fallback);
    }
    const auto value = found->second;
    defaults_.erase(found);
    // ROS overrides retain their normal precedence over the profile default.
    return node_.declare_parameter<T>(name, value.get<T>());
  }

  void validateConsumed() const;

private:
  rclcpp::Node & node_;
  std::map<std::string, rclcpp::ParameterValue> defaults_;
};

}  // namespace slam_3d_core
}  // namespace gole

#endif  // SLAM_3D_CORE__PARAMETER_PROFILE_HPP_
