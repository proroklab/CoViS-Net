cmake_minimum_required(VERSION 3.8)
project(sensing_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EncodedImage.msg"
  "msg/PolarPose.msg"
  "msg/PolarPoseWithCovariance.msg"
  "msg/PolarPoseWithCovarianceStamped.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_package()
