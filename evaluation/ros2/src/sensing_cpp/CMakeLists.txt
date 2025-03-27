cmake_minimum_required(VERSION 3.8)
project(sensing_cpp)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/python3.8/dist-packages/torch/share/cmake")

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensing_msgs REQUIRED)
find_package(Torch REQUIRED)

find_package(ament_cmake REQUIRED)

include_directories("${TORCH_INCLUDE_DIRS}")

# Torch TensorRT
include_directories("/usr/local/lib/python3.8/dist-packages/torch_tensorrt/include")
link_directories("/usr/local/lib/python3.8/dist-packages/torch_tensorrt/lib")
set(Torch_TensorRT_LIBS -Wl,--no-as-needed -ltorchtrt -Wl,--as-needed)

set(CMAKE_CXX_STANDARD 14)

include_directories("./include")

add_executable(encode_img src/encode_img.cpp)
target_link_directories(encode_img PRIVATE lib)
target_link_libraries(encode_img ${TORCH_LIBRARIES} ${Torch_TensorRT_LIBS} swmc_net)

add_executable(predict_pose src/predict_pose.cpp)
target_link_directories(predict_pose PRIVATE lib)
target_link_libraries(predict_pose ${TORCH_LIBRARIES} ${Torch_TensorRT_LIBS} swmc_net)

ament_target_dependencies(encode_img rclcpp std_msgs sensor_msgs image_transport cv_bridge OpenCV sensing_msgs Torch)
install(TARGETS encode_img DESTINATION lib/${PROJECT_NAME})

ament_target_dependencies(predict_pose rclcpp std_msgs sensor_msgs geometry_msgs sensing_msgs tf2 tf2_ros Torch)
install(TARGETS predict_pose DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY models launch cfg
  DESTINATION share/${PROJECT_NAME}
)

#install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()
