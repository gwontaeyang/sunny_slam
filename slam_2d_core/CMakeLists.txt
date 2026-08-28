cmake_minimum_required(VERSION 3.8)
project(slam_2d_core)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

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
find_package(Boost REQUIRED COMPONENTS serialization thread)
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

add_library(adaptive_roughness_memory STATIC
  src/adaptive_roughness_memory.cpp
)

add_library(karto_near_chain_observer STATIC
  src/karto_near_chain_observer.cpp
)
target_compile_features(karto_near_chain_observer PUBLIC cxx_std_17)
target_include_directories(karto_near_chain_observer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(karto_near_chain_observer
  slam_toolbox
)
target_link_libraries(karto_near_chain_observer
  adaptive_roughness_memory
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  Boost::serialization
  TBB::tbb
)

add_library(karto_loop_closure_observer STATIC
  src/karto_loop_closure_observer.cpp
)
target_compile_features(karto_loop_closure_observer PUBLIC cxx_std_17)
target_include_directories(karto_loop_closure_observer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(karto_loop_closure_observer
  slam_toolbox
)
target_link_libraries(karto_loop_closure_observer
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  Boost::serialization
  TBB::tbb
)

add_library(near_chain_rotation_consistency STATIC
  src/near_chain_rotation_consistency.cpp
)
target_compile_features(near_chain_rotation_consistency PUBLIC cxx_std_17)
target_include_directories(near_chain_rotation_consistency PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(adaptive_roughness_memory PUBLIC cxx_std_17)
target_include_directories(adaptive_roughness_memory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(roughness_innovation_policy STATIC
  src/roughness_innovation_policy.cpp
)
target_compile_features(roughness_innovation_policy PUBLIC cxx_std_17)
target_include_directories(roughness_innovation_policy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(roughness_place_recognition STATIC
  src/roughness_place_recognition.cpp
)
target_compile_features(roughness_place_recognition PUBLIC cxx_std_17)
target_include_directories(roughness_place_recognition PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(roughness_place_scan_chain STATIC
  src/roughness_place_scan_chain.cpp
)
target_compile_features(roughness_place_scan_chain PUBLIC cxx_std_17)
target_include_directories(roughness_place_scan_chain PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(roughness_place_scan_chain
  roughness_place_recognition
)

add_library(roughness_loop_memory STATIC
  src/roughness_loop_memory.cpp
)
target_compile_features(roughness_loop_memory PUBLIC cxx_std_17)
target_include_directories(roughness_loop_memory PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(roughness_scan_conditioner STATIC
  src/roughness_scan_conditioner.cpp
)
target_compile_features(roughness_scan_conditioner PUBLIC cxx_std_17)
target_include_directories(roughness_scan_conditioner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(roughness_scan_conditioner
  geometry_msgs
  sensor_msgs
)

add_library(ordered_stamp_audit STATIC
  src/ordered_stamp_audit.cpp
)
target_compile_features(ordered_stamp_audit PUBLIC cxx_std_17)
target_include_directories(ordered_stamp_audit PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(pose_graph_constraint_replay STATIC
  src/pose_graph_constraint_replay.cpp
)
target_compile_features(pose_graph_constraint_replay PUBLIC cxx_std_17)
target_include_directories(pose_graph_constraint_replay PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(tf_to_odom_node
  src/tf_to_odom_node.cpp
)
ament_target_dependencies(tf_to_odom_node
  rclcpp
  nav_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(roughness_anchor_slam_toolbox_node
  src/karto_scan_matcher_vector.cpp
  src/roughness_anchor_slam_toolbox.cpp
  src/roughness_anchor_slam_toolbox_node.cpp
)
target_compile_features(roughness_anchor_slam_toolbox_node PUBLIC cxx_std_17)
target_include_directories(roughness_anchor_slam_toolbox_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
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

add_executable(roughness_aware_slam_toolbox_node
  src/karto_scan_matcher_vector.cpp
  src/roughness_aware_slam_toolbox.cpp
  src/roughness_aware_slam_toolbox_node.cpp
)
target_compile_features(roughness_aware_slam_toolbox_node PUBLIC cxx_std_17)
target_include_directories(roughness_aware_slam_toolbox_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(roughness_aware_slam_toolbox_node
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
target_link_libraries(roughness_aware_slam_toolbox_node
  adaptive_roughness_memory
  karto_loop_closure_observer
  karto_near_chain_observer
  near_chain_rotation_consistency
  ordered_stamp_audit
  roughness_innovation_policy
  roughness_place_recognition
  roughness_place_scan_chain
  roughness_loop_memory
  roughness_scan_conditioner
  ${SYNC_SLAM_TOOLBOX_LIBRARY}
  ${TOOLBOX_COMMON_LIBRARY}
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  ${CERES_SOLVER_PLUGIN_LIBRARY}
  Boost::serialization
  Boost::thread
  Eigen3::Eigen
  TBB::tbb
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

add_executable(slam_pose_graph_edge_exporter
  src/slam_pose_graph_edge_exporter.cpp
)

add_executable(slam_pose_graph_constraint_replay
  src/slam_pose_graph_constraint_replay.cpp
)
target_compile_features(slam_pose_graph_constraint_replay PUBLIC cxx_std_17)
ament_target_dependencies(slam_pose_graph_constraint_replay
  pluginlib
  rclcpp
  slam_toolbox
)
target_link_libraries(slam_pose_graph_constraint_replay
  pose_graph_constraint_replay
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  ${CERES_SOLVER_PLUGIN_LIBRARY}
  Boost::serialization
  Eigen3::Eigen
  TBB::tbb
)
target_compile_features(slam_pose_graph_edge_exporter PUBLIC cxx_std_17)
ament_target_dependencies(slam_pose_graph_edge_exporter
  rclcpp
  slam_toolbox
)
target_link_libraries(slam_pose_graph_edge_exporter
  ${KARTO_SLAM_TOOLBOX_LIBRARY}
  Boost::serialization
  Eigen3::Eigen
  TBB::tbb
)

install(TARGETS
  tf_to_odom_node
  roughness_anchor_slam_toolbox_node
  roughness_aware_slam_toolbox_node
  slam_pose_graph_exporter
  slam_pose_graph_edge_exporter
  slam_pose_graph_constraint_replay
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS
  adaptive_roughness_memory
  karto_loop_closure_observer
  karto_near_chain_observer
  near_chain_rotation_consistency
  ordered_stamp_audit
  pose_graph_constraint_replay
  roughness_innovation_policy
  roughness_place_recognition
  roughness_place_scan_chain
  roughness_loop_memory
  roughness_scan_conditioner
  ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_pytest_test(
    test_roughness_anchor_node_contract
    test/test_roughness_anchor_node_contract.py
  )
  ament_add_pytest_test(
    test_roughness_aware_node_contract
    test/test_roughness_aware_node_contract.py
  )
  ament_add_gtest(
    test_adaptive_roughness_memory
    test/test_adaptive_roughness_memory.cpp
  )
  target_link_libraries(
    test_adaptive_roughness_memory
    adaptive_roughness_memory
  )
  ament_add_gtest(
    test_karto_loop_closure_observer
    test/test_karto_loop_closure_observer.cpp
  )
  target_link_libraries(
    test_karto_loop_closure_observer
    karto_loop_closure_observer
  )
  ament_add_gtest(
    test_karto_near_chain_observer
    test/test_karto_near_chain_observer.cpp
  )
  target_sources(
    test_karto_near_chain_observer
    PRIVATE src/karto_scan_matcher_vector.cpp
  )
  target_link_libraries(
    test_karto_near_chain_observer
    karto_near_chain_observer
  )
  ament_add_gtest(
    test_near_chain_rotation_consistency
    test/test_near_chain_rotation_consistency.cpp
  )
  target_link_libraries(
    test_near_chain_rotation_consistency
    near_chain_rotation_consistency
  )
  ament_add_gtest(
    test_roughness_innovation_policy
    test/test_roughness_innovation_policy.cpp
  )
  target_link_libraries(
    test_roughness_innovation_policy
    roughness_innovation_policy
  )
  ament_add_gtest(
    test_roughness_place_recognition
    test/test_roughness_place_recognition.cpp
  )
  target_link_libraries(
    test_roughness_place_recognition
    roughness_place_recognition
  )
  ament_add_gtest(
    test_roughness_place_scan_chain
    test/test_roughness_place_scan_chain.cpp
  )
  target_link_libraries(
    test_roughness_place_scan_chain
    roughness_place_scan_chain
  )
  ament_add_gtest(
    test_roughness_loop_memory
    test/test_roughness_loop_memory.cpp
  )
  target_link_libraries(
    test_roughness_loop_memory
    roughness_loop_memory
  )
  ament_add_gtest(
    test_ordered_stamp_audit
    test/test_ordered_stamp_audit.cpp
  )
  target_link_libraries(
    test_ordered_stamp_audit
    ordered_stamp_audit
  )
  ament_add_gtest(
    test_pose_graph_constraint_replay
    test/test_pose_graph_constraint_replay.cpp
  )
  target_link_libraries(
    test_pose_graph_constraint_replay
    pose_graph_constraint_replay
  )
  ament_add_gtest(
    test_roughness_scan_conditioner
    test/test_roughness_scan_conditioner.cpp
  )
  target_link_libraries(
    test_roughness_scan_conditioner
    roughness_scan_conditioner
  )
endif()

ament_package()
