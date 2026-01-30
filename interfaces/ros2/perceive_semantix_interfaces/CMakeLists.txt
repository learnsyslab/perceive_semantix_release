cmake_minimum_required(VERSION 3.10)
project(perceive_semantix_interfaces)

# silence warnings produced by rosidl_default_generators
cmake_policy(SET CMP0148 OLD)

find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectDetection.msg"
  "msg/ObjectDetectionArray.msg"
  "srv/EnableHeatmap.srv"
  "srv/SetMapPose.srv"
  DEPENDENCIES geometry_msgs builtin_interfaces
)

ament_package()