cmake_minimum_required(VERSION 3.8)
project(crc_yolo)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

# TensorRT 8.2.1 public headers (Apache-2.0, github.com/NVIDIA/TensorRT, tag 8.2.1).
# CUDA types come from include/cuda_runtime_api.h, a minimal stand-in: the node loads
# libcudart and libnvinfer from the host at run time, so no CUDA toolkit is needed here.
set(TRT_INCLUDE_DIR "/opt/trt/include" CACHE PATH "TensorRT 8.2 headers")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(yolo_trt src/yolo_trt_node.cpp)
target_include_directories(yolo_trt PRIVATE include ${TRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_compile_options(yolo_trt PRIVATE -O2 -Wall -Wno-deprecated-declarations)
target_link_libraries(yolo_trt ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
ament_target_dependencies(yolo_trt rclcpp sensor_msgs vision_msgs)

install(TARGETS yolo_trt DESTINATION lib/${PROJECT_NAME})
ament_package()
