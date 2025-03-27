FROM dustynv/ros:humble-pytorch-l4t-r35.3.1

RUN apt-get update && apt-get install -y git vim
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-arm64 -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

RUN mkdir -p /opt/inference/src
WORKDIR /opt/inference

# https://pytorch.org/TensorRT/getting_started/installation.html
RUN git clone https://github.com/pytorch/TensorRT.git --branch v1.4.0
WORKDIR /opt/inference/TensorRT
RUN cp toolchains/jp_workspaces/WORKSPACE.jp50 WORKSPACE
ADD docker/encoder/pr2118_version_fix.patch .
RUN git apply pr2118_version_fix.patch

RUN bazel build //:libtorchtrt --platforms //toolchains:jetpack_5.0
#RUN mkdir build && cmake -S . -B build -DTorch_DIR=/usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch -DCMAKE_BUILD_TYPE=Release
#RUN cmake --build build -j 8
WORKDIR /opt/inference/TensorRT/py
RUN python3 setup.py install --use-cxx11-abi

WORKDIR /opt/inference
RUN rm -rf TensorRT

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/python3.8/dist-packages/torch/lib"

###
RUN pip3 install scipy

RUN mkdir -p /opt/robomaster/src
COPY src /opt/robomaster/src
COPY configs/cyclone_dds_profile_base.xml /opt/robomaster

WORKDIR /opt/robomaster

RUN . /ros_deep_learning/install/setup.bash && colcon build --symlink-install --packages-select sensing_msgs sensing_cpp

ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV CYCLONEDDS_URI=/opt/robomaster/cyclone_dds_profile_base.xml
