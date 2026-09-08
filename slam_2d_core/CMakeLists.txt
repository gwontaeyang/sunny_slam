cmake_minimum_required(VERSION 3.8)
project(slam_2d_core)

find_package(ament_cmake REQUIRED)
set(runtime_dependencies rclcpp nav_msgs geometry_msgs sensor_msgs std_msgs std_srvs
  tf2 tf2_ros tf2_geometry_msgs tf2_sensor_msgs message_filters pluginlib
  visualization_msgs interactive_markers slam_toolbox)
foreach(dependency IN LISTS runtime_dependencies)
  find_package(${dependency} REQUIRED)
endforeach()
find_package(Boost REQUIRED COMPONENTS serialization thread)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)
foreach(library sync_slam_toolbox toolbox_common kartoSlamToolbox ceres_solver_plugin)
  find_library(${library}_PATH NAMES ${library} HINTS ${slam_toolbox_LIBRARY_DIRS} REQUIRED)
endforeach()

set(core ${CMAKE_CURRENT_SOURCE_DIR})
set(policies adaptive_roughness_memory karto_loop_closure_observer karto_near_chain_observer
  near_chain_rotation_consistency ordered_stamp_audit roughness_innovation_policy
  roughness_place_recognition roughness_place_scan_chain roughness_loop_memory
  roughness_scan_conditioner pose_graph_constraint_replay)
foreach(policy IN LISTS policies)
  add_library(${policy} STATIC ${core}/src/${policy}.cpp)
  target_compile_features(${policy} PUBLIC cxx_std_17)
  target_include_directories(${policy} PUBLIC ${core}/include)
endforeach()
ament_target_dependencies(karto_loop_closure_observer slam_toolbox)
ament_target_dependencies(karto_near_chain_observer slam_toolbox)
ament_target_dependencies(roughness_scan_conditioner geometry_msgs sensor_msgs)
target_link_libraries(karto_near_chain_observer adaptive_roughness_memory
  ${kartoSlamToolbox_PATH} Boost::serialization TBB::tbb)
target_link_libraries(karto_loop_closure_observer
  ${kartoSlamToolbox_PATH} Boost::serialization TBB::tbb)
target_link_libraries(roughness_place_scan_chain roughness_place_recognition)

add_executable(roughness_aware_slam_toolbox_node ${core}/src/karto_scan_matcher_vector.cpp
  ${core}/src/roughness_aware_slam_toolbox.cpp
  ${core}/src/roughness_aware_slam_toolbox_node.cpp)
target_compile_features(roughness_aware_slam_toolbox_node PUBLIC cxx_std_17)
target_include_directories(roughness_aware_slam_toolbox_node PRIVATE ${core}/include)
ament_target_dependencies(roughness_aware_slam_toolbox_node ${runtime_dependencies})
target_link_libraries(roughness_aware_slam_toolbox_node ${policies} ${sync_slam_toolbox_PATH}
  ${toolbox_common_PATH} ${kartoSlamToolbox_PATH} ${ceres_solver_plugin_PATH}
  Boost::serialization Boost::thread Eigen3::Eigen TBB::tbb)

add_executable(tf_to_odom_node ${core}/src/tf_to_odom_node.cpp)
target_compile_features(tf_to_odom_node PUBLIC cxx_std_17)
ament_target_dependencies(tf_to_odom_node rclcpp nav_msgs geometry_msgs
  tf2 tf2_ros tf2_geometry_msgs)

add_executable(slam_pose_graph_exporter ${core}/src/slam_pose_graph_exporter.cpp)
target_compile_features(slam_pose_graph_exporter PUBLIC cxx_std_17)
ament_target_dependencies(slam_pose_graph_exporter rclcpp slam_toolbox)
target_link_libraries(slam_pose_graph_exporter
  ${kartoSlamToolbox_PATH} Boost::serialization Eigen3::Eigen TBB::tbb)

add_executable(slam_pose_graph_edge_exporter ${core}/src/slam_pose_graph_edge_exporter.cpp)
target_compile_features(slam_pose_graph_edge_exporter PUBLIC cxx_std_17)
ament_target_dependencies(slam_pose_graph_edge_exporter rclcpp slam_toolbox)
target_link_libraries(slam_pose_graph_edge_exporter
  ${kartoSlamToolbox_PATH} Boost::serialization Eigen3::Eigen TBB::tbb)

install(TARGETS roughness_aware_slam_toolbox_node tf_to_odom_node
  slam_pose_graph_exporter slam_pose_graph_edge_exporter
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  foreach(policy IN LISTS policies)
    ament_add_gtest(test_${policy} ${core}/test/test_${policy}.cpp)
    target_link_libraries(test_${policy} ${policy})
  endforeach()
  target_sources(test_karto_near_chain_observer PRIVATE ${core}/src/karto_scan_matcher_vector.cpp)
  ament_add_pytest_test(aware_node_contract ${core}/test/test_roughness_aware_node_contract.py TIMEOUT 180)
endif()
ament_package()
