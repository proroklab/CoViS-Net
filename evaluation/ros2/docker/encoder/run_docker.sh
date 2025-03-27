docker run -it --rm --runtime nvidia --net=host --ipc=host -v /dev/shm:/dev/shm --hostname $(cat /etc/hostname) -v /home/nvidia/ros2_panoptes:/opt/ros2_panoptes ros2_panoptes_encoder:latest
