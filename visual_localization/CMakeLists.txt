cmake_minimum_required(VERSION 3.8)
project(visual_localization)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(Thirdparty/g2o)

include_directories(
  inc
  learned_features_inference/inc
  ${PROJECT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

set(TensorRT_DIR "/home/zc/code/learned_features_inference-tensorRT/tensorRT/TensorRT-8.4.3.1")
include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)

add_library(FeaturesLib
  learned_features_inference/src/interface_tensorRT.cpp
  learned_features_inference/src/interface.cpp
  learned_features_inference/src/extractor.cpp
  learned_features_inference/src/utils.cpp)

target_link_libraries(FeaturesLib openvino::runtime ${OpenCV_LIBS} nvinfer nvinfer_plugin cudart)

add_library(VisualMappingLib
  src/camera.cpp
  src/matcher.cpp
  src/triangulation.cpp
  src/frame.cpp
  src/bundle_adjustment.cpp
  src/visualization.cpp
  src/detection.cpp
  src/map_point.cpp
  src/mapping.cpp
  src/map_save.cpp
  src/localization_helper.cpp
  src/localization_core.cpp)

target_link_libraries(
  VisualMappingLib
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${Pangolin_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
  FeaturesLib)

add_executable(VisualMapping main.cpp)
target_link_libraries(VisualMapping FeaturesLib VisualMappingLib)

add_executable(Localization Demo/localization.cpp)
target_link_libraries(Localization FeaturesLib VisualMappingLib)

add_executable(localization_node Demo/main_ros.cpp Demo/ros_localization_node.cpp)
ament_target_dependencies(localization_node
  rclcpp sensor_msgs std_msgs geometry_msgs nav_msgs cv_bridge message_filters tf2 tf2_ros pcl_ros pcl_conversions)
target_link_libraries(localization_node FeaturesLib VisualMappingLib)

install(TARGETS localization_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
