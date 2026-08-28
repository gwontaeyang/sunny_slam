cmake_minimum_required(VERSION 3.8)
project(slam_3d_core)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(
    CMAKE_BUILD_TYPE RelWithDebInfo
    CACHE STRING "Build type for slam_3d_core" FORCE
  )
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters registration)
find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(scan_to_map_icp
  src/scan_to_map_icp.cpp
)

add_library(indoor_terrain_evidence
  src/indoor_terrain_evidence.cpp
  src/terrain_preview_evidence.cpp
)

add_library(voxel_surfel_registration
  src/voxel_surfel_registration.cpp
)

target_include_directories(scan_to_map_icp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

target_include_directories(indoor_terrain_evidence PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

target_include_directories(voxel_surfel_registration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

ament_target_dependencies(scan_to_map_icp
  diagnostic_msgs
  geometry_msgs
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  visualization_msgs
)

target_link_libraries(scan_to_map_icp
  indoor_terrain_evidence
  voxel_surfel_registration
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

target_link_libraries(indoor_terrain_evidence
  Eigen3::Eigen
)

target_link_libraries(voxel_surfel_registration
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

add_executable(scan_to_map_icp_node
  src/scan_to_map_icp_node.cpp
)

target_include_directories(scan_to_map_icp_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(scan_to_map_icp_node
  rclcpp
)

target_link_libraries(scan_to_map_icp_node
  scan_to_map_icp
)

install(TARGETS scan_to_map_icp indoor_terrain_evidence voxel_surfel_registration
  EXPORT export_scan_to_map_icp
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS scan_to_map_icp_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_scan_to_map_icp HAS_LIBRARY_TARGET)
ament_export_dependencies(
  diagnostic_msgs
  geometry_msgs
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  visualization_msgs
  PCL
  Eigen3
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  # ament_xmllint resolves the package.xml schema through download.ros.org.
  # Keep this prototype testable in offline dataset-replay environments.
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_gtest(test_voxel_surfel_registration
    test/test_voxel_surfel_registration.cpp
  )
  target_link_libraries(test_voxel_surfel_registration
    voxel_surfel_registration
  )
  ament_add_gtest(test_registration_policy
    test/test_registration_policy.cpp
  )
  target_include_directories(test_registration_policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_registration_policy
    Eigen3::Eigen
  )
  ament_add_gtest(test_scan_context_descriptor
    test/test_scan_context_descriptor.cpp
  )
  target_include_directories(test_scan_context_descriptor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_add_gtest(test_ground_motion_constraint
    test/test_ground_motion_constraint.cpp
  )
  target_include_directories(test_ground_motion_constraint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_ground_motion_constraint
    Eigen3::Eigen
  )
  ament_add_gtest(test_imu_relative_attitude_constraint
    test/test_imu_relative_attitude_constraint.cpp
  )
  target_include_directories(test_imu_relative_attitude_constraint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_imu_relative_attitude_constraint
    Eigen3::Eigen
  )
  ament_add_gtest(test_imu_shock_policy
    test/test_imu_shock_policy.cpp
  )
  target_include_directories(test_imu_shock_policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_add_gtest(test_indoor_terrain_evidence
    test/test_indoor_terrain_evidence.cpp
  )
  target_include_directories(test_indoor_terrain_evidence PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_indoor_terrain_evidence
    indoor_terrain_evidence
    Eigen3::Eigen
  )
  ament_add_gtest(test_terrain_preview_evidence
    test/test_terrain_preview_evidence.cpp
  )
  target_include_directories(test_terrain_preview_evidence PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_terrain_preview_evidence
    indoor_terrain_evidence
    Eigen3::Eigen
  )
  ament_add_gtest(test_spatial_mapping_policy
    test/test_spatial_mapping_policy.cpp
  )
  target_include_directories(test_spatial_mapping_policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(test_spatial_mapping_policy
    Eigen3::Eigen
  )
  ament_add_gtest(test_submap_policy
    test/test_submap_policy.cpp
  )
  target_include_directories(test_submap_policy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  ament_add_pytest_test(test_scan_to_map_config test/test_scan_to_map_config.py)
  ament_add_pytest_test(test_indoor_profiles test/test_indoor_profiles.py)
  ament_add_pytest_test(
    test_submap_runtime
    test/test_submap_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_scan_to_map_runtime
    test/test_scan_to_map_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_imu_shock_runtime
    test/test_imu_shock_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_terrain_runtime
    test/test_terrain_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_multi_lidar_sync_runtime
    test/test_multi_lidar_sync_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_voxel_surfel_runtime
    test/test_voxel_surfel_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_pose_graph_markers_runtime
    test/test_pose_graph_markers_runtime.py
    TIMEOUT 60
  )
endif()

ament_package()
