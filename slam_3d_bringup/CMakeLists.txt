cmake_minimum_required(VERSION 3.8)
project(slam_3d_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  docs
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(test_launch_description test/test_launch_description.py)
  ament_add_pytest_test(test_pipeline_contract test/test_pipeline_contract.py)
endif()

ament_package()
