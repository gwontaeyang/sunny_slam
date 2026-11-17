cmake_minimum_required(VERSION 3.16)
project(slam_3d_map_cleanup)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(yaml-cpp REQUIRED)

add_library(temporal_static_map src/temporal_static_map.cpp)
target_compile_features(temporal_static_map PUBLIC cxx_std_17)
target_include_directories(temporal_static_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(temporal_static_map PUBLIC Eigen3::Eigen)

add_library(map_cleanup src/map_cleanup.cpp)
target_link_libraries(map_cleanup PUBLIC temporal_static_map PRIVATE ${PCL_LIBRARIES} yaml-cpp)
target_include_directories(map_cleanup SYSTEM PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_options(map_cleanup PRIVATE ${PCL_DEFINITIONS})

add_executable(temporal_map_cleanup src/temporal_map_cleanup_main.cpp)
target_link_libraries(temporal_map_cleanup PRIVATE map_cleanup)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target temporal_static_map map_cleanup temporal_map_cleanup)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endforeach()
endif()

install(TARGETS temporal_static_map map_cleanup
  EXPORT export_map_cleanup
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS temporal_map_cleanup RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
ament_export_targets(export_map_cleanup HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 PCL yaml-cpp)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  ament_add_gtest(test_temporal_static_map test/test_temporal_static_map.cpp)
  target_link_libraries(test_temporal_static_map temporal_static_map)
  ament_add_gtest(test_map_cleanup test/test_map_cleanup.cpp)
  target_link_libraries(test_map_cleanup map_cleanup)
  add_test(NAME test_cleanup_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/test_cleanup_cli.py
      $<TARGET_FILE:temporal_map_cleanup>)
  set_tests_properties(test_cleanup_cli PROPERTIES TIMEOUT 60)
endif()

ament_package()
