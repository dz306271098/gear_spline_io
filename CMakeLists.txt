cmake_minimum_required(VERSION 3.0.2)
project(gear_spline_lio)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  pcl_ros
  tf
  eigen_conversions
  livox_ros_driver
  rosbag
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs pcl_ros tf eigen_conversions livox_ros_driver rosbag
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Milestone 0: Pure algorithm test (no ROS dependencies)
add_executable(test_spline test/test_spline.cpp)
target_compile_definitions(test_spline PRIVATE STANDALONE_TEST)
target_link_libraries(test_spline
  ${EIGEN3_LIBRARIES}
  OpenMP::OpenMP_CXX
)
target_include_directories(test_spline PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
)

# Simulator test (IMU & LiDAR data generation)
add_executable(test_simulator test/test_simulator.cpp)
target_compile_definitions(test_simulator PRIVATE STANDALONE_TEST)
target_link_libraries(test_simulator
  ${EIGEN3_LIBRARIES}
  OpenMP::OpenMP_CXX
)
target_include_directories(test_simulator PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
)

# Milestone 1: Main LIO node with ROS1
add_executable(gear_spline_lio_node
  src/gear_spline_lio_node.cpp
  include/gear_spline/ikd-Tree/ikd_Tree.cpp
)
target_link_libraries(gear_spline_lio_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
target_include_directories(gear_spline_lio_node PRIVATE
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

# Offline rosbag test program for debugging
add_executable(test_rosbag_offline
  test/test_rosbag_offline.cpp
  include/gear_spline/ikd-Tree/ikd_Tree.cpp
)
target_link_libraries(test_rosbag_offline
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)
target_include_directories(test_rosbag_offline PRIVATE
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)
