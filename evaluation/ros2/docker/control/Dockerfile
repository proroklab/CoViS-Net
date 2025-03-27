FROM dustynv/ros:humble-ros-base-l4t-r35.3.1

RUN apt-get update && apt-get install -y git vim

RUN pip3 install scipy

RUN mkdir -p /opt/robomaster/src
COPY src /opt/robomaster/src

WORKDIR /opt/robomaster

RUN . /opt/ros/humble/install/setup.sh && colcon build --symlink-install --packages-select robomaster_msgs freyja_msgs state_manager lqg_control robomaster_handler sensing_msgs control

COPY configs/cyclone_dds_profile_base.xml /opt/robomaster

ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/opt/robomaster/cyclone_dds_profile_base.xml
