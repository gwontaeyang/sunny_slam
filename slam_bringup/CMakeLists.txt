cmake_minimum_required(VERSION 3.8)
project(slam_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pluginlib REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(slam_toolbox REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

find_library(SYNC_SLAM_TOOLBOX_LIBRARY
  NAMES sync_slam_toolbox
  HINTS ${slam_toolbox_LIBRARY_DIRS}
  REQUIRED
)
find_library(TOOLBOX_COMMON_LIBRARY
  NAMES toolbox_common
  HINTS ${slam_toolbox_LIBRARY_DIRS}
  REQUIRED
)
find_library(KARTO_SLAM_TOOLBOX_LIBRARY
  NAMES kartoSlamToolbox
  HINTS ${slam_toolbox_LIBRARY_DIRS}
  REQUIRED
)
find_library(CERES_SOLVER_PLUGIN_LIBRARY
  NAMES ceres_solver_plugin
  HINTS ${slam_toolbox_LIBRARY_DIRS}
  REQUIRED
)

include_directories(include)

# Create executable (test 폴더의 파일 사용)
add_executable(tf_to_odom_node test/tf_to_odom_node.cpp)
ament_target_dependencies(tf_to_odom_node
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# Install executable
install(TARGETS tf_to_odom_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(roughness_anchor_slam_toolbox_node
  src/karto_scan_matcher_vector.cpp
  src/roughness_anchor_slam_toolbox.cpp
  src/roughness_anchor_slam_toolbox_node.cpp
)
target_compile_features(roughness_anchor_slam_toolbox_node PUBLIC cxx_std_17)
ament_target_dependencies(roughness_anchor_slam_toolbox_node
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_sensor_msgs
  message_filters
  pluginlib
  visualization_msgs
  interactive_markers
  slam_toolbox
)
target_link_libraries(roughness_anchor_slam_toolbox_node
  ${SYNC_SLAM_TOOLBOX_LIBRARY}
  ${TOOLBOX_COMMON_LIBRARY}
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  ${CERES_SOLVER_PLUGIN_LIBRARY}
  Boost::serialization
  Eigen3::Eigen
  TBB::tbb
)

install(TARGETS roughness_anchor_slam_toolbox_node
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(slam_pose_graph_exporter
  src/slam_pose_graph_exporter.cpp
)
target_compile_features(slam_pose_graph_exporter PUBLIC cxx_std_17)
ament_target_dependencies(slam_pose_graph_exporter
  rclcpp
  slam_toolbox
)
target_link_libraries(slam_pose_graph_exporter
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  Boost::serialization
  Eigen3::Eigen
  TBB::tbb
)

install(TARGETS slam_pose_graph_exporter
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install test launch files
install(DIRECTORY test
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py"
)

# Install config files
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}/
)

# Install scripts
install(DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}/
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
