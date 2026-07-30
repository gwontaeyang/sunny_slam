cmake_minimum_required(VERSION 3.8)
project(slam_3d_livox_ingest)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(mid360_time_contract
  src/mid360_time_contract.cpp
)
target_include_directories(mid360_time_contract PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Livox ROS Driver 2 is a hardware-side dependency and is not vendored into this
# repository. Keep the validation library buildable for dataset development,
# while exposing the actual adapter whenever the driver messages are installed.
find_package(livox_ros_driver2 QUIET)
if(livox_ros_driver2_FOUND)
  add_library(mid360_ingest
    src/mid360_ingest.cpp
  )
  target_include_directories(mid360_ingest PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(mid360_ingest
    diagnostic_msgs
    livox_ros_driver2
    rclcpp
    sensor_msgs
  )
  target_link_libraries(mid360_ingest
    mid360_time_contract
  )

  add_executable(mid360_ingest_node
    src/mid360_ingest_node.cpp
  )
  target_include_directories(mid360_ingest_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(mid360_ingest_node
    rclcpp
  )
  target_link_libraries(mid360_ingest_node
    mid360_ingest
  )

  install(TARGETS mid360_ingest mid360_ingest_node
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
else()
  message(STATUS
    "livox_ros_driver2 was not found: building the Mid-360 time-contract "
    "library without the hardware message adapter")
endif()

install(TARGETS mid360_time_contract
  EXPORT export_mid360_time_contract
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_export_targets(export_mid360_time_contract HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_xmllint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_mid360_time_contract
    test/test_mid360_time_contract.cpp
  )
  target_link_libraries(test_mid360_time_contract
    mid360_time_contract
  )

  if(livox_ros_driver2_FOUND)
    ament_add_pytest_test(
      test_mid360_ingest_runtime
      test/test_mid360_ingest_runtime.py
      TIMEOUT 60
    )
  endif()
endif()

ament_package()
