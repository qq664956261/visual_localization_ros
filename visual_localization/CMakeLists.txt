cmake_minimum_required(VERSION 3.1)
project(visual_localization)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

set(OpenVINO_DIR /home/zc/lib/l_openvino_toolkit_ubuntu20_2024.0.0.14509.34caeefd078_x86_64/runtime/cmake)
find_package(OpenVINO REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

find_package(Pangolin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        sensor_msgs
        geometry_msgs
        nav_msgs
        cv_bridge
        image_transport
        message_filters
        tf
)
catkin_package()
add_subdirectory(Thirdparty/g2o)
# add_subdirectory(${CMAKE_SOURCE_DIR}/Thirdparty/g2o
#                  ${CMAKE_BINARY_DIR}/g2o-build)

include_directories(
        inc
        learned_features_inference/inc
        ${PROJECT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)
set(TensorRT_DIR "/home/zc/code/learned_features_inference-tensorRT/tensorRT/TensorRT-8.4.3.1")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS}      )
include_directories( learned_features_inference/inc ${TensorRT_DIR}/include)
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
        FeaturesLib
)

add_executable(VisualMapping main.cpp)

target_link_libraries(VisualMapping
        FeaturesLib
        VisualMappingLib

)

add_executable(Localization Demo/localization.cpp)

target_link_libraries(Localization
        FeaturesLib
        VisualMappingLib
)

add_executable(localization_node Demo/main_ros.cpp Demo/ros_localization_node.cpp)
add_dependencies(localization_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(localization_node
        FeaturesLib
        VisualMappingLib
        ${catkin_LIBRARIES}
)

