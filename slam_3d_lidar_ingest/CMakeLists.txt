cmake_minimum_required(VERSION 3.8)
project(slam_3d_lidar_ingest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(hesai_pointcloud_contract
  src/hesai_pointcloud_contract.cpp
)
target_include_directories(hesai_pointcloud_contract PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hesai_pointcloud_contract
  sensor_msgs
)

add_library(hesai_pointcloud_ingest
  src/hesai_pointcloud_ingest.cpp
)
target_include_directories(hesai_pointcloud_ingest PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hesai_pointcloud_ingest
  builtin_interfaces
  diagnostic_msgs
  rclcpp
  sensor_msgs
)
target_link_libraries(hesai_pointcloud_ingest
  hesai_pointcloud_contract
)

add_library(hesai_imu_normalizer
  src/hesai_imu_normalizer.cpp
)
target_include_directories(hesai_imu_normalizer PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hesai_imu_normalizer
  builtin_interfaces
  diagnostic_msgs
  rclcpp
  sensor_msgs
)
target_link_libraries(hesai_imu_normalizer
  hesai_pointcloud_contract
)

add_executable(hesai_pointcloud_ingest_node
  src/hesai_pointcloud_ingest_node.cpp
)
target_include_directories(hesai_pointcloud_ingest_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hesai_pointcloud_ingest_node
  rclcpp
)
target_link_libraries(hesai_pointcloud_ingest_node
  hesai_pointcloud_ingest
)

add_executable(hesai_imu_normalizer_node
  src/hesai_imu_normalizer_node.cpp
)
target_include_directories(hesai_imu_normalizer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(hesai_imu_normalizer_node
  rclcpp
)
target_link_libraries(hesai_imu_normalizer_node
  hesai_imu_normalizer
)

install(TARGETS
  hesai_pointcloud_contract
  hesai_pointcloud_ingest
  hesai_imu_normalizer
  hesai_pointcloud_ingest_node
  hesai_imu_normalizer_node
  EXPORT export_hesai_pointcloud_contract
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/analyze_hesai_imu_alignment.py
  scripts/calibrate_geoje_xt32_replay.py
  scripts/generate_geoje_xt32_replay_params.py
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_hesai_pointcloud_contract HAS_LIBRARY_TARGET)
ament_export_dependencies(builtin_interfaces diagnostic_msgs rclcpp sensor_msgs)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_hesai_pointcloud_contract
    test/test_hesai_pointcloud_contract.cpp
  )
  ament_target_dependencies(test_hesai_pointcloud_contract
    sensor_msgs
  )
  target_link_libraries(test_hesai_pointcloud_contract
    hesai_pointcloud_contract
  )

  ament_add_gtest(test_hesai_imu_normalizer
    test/test_hesai_imu_normalizer.cpp
  )
  ament_target_dependencies(test_hesai_imu_normalizer
    sensor_msgs
  )
  target_link_libraries(test_hesai_imu_normalizer
    hesai_imu_normalizer
  )

  ament_add_pytest_test(
    test_hesai_pointcloud_ingest_runtime
    test/test_hesai_pointcloud_ingest_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_hesai_imu_normalizer_runtime
    test/test_hesai_imu_normalizer_runtime.py
    TIMEOUT 60
  )
  ament_add_pytest_test(
    test_geoje_clock_calibration
    test/test_geoje_clock_calibration.py
  )
endif()

ament_package()
