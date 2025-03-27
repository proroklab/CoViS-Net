# CoViS-Net On-Robot evaluation
This directory contains the code for the on-robot evaluation of the CoViS-Net paper.

Prerequisites:
- A monocular forward facing camera with a field of view of 120 degree with a rectified image is expected. In our experiments, we used the Raspberry Pi HQ camera. We used the OpenCV fisheye module to calibrate and rectify all images.
- For adhoc wireless communication, we use a Netgear A6210 WiFi dongle. Run the adhoc setup script in `./util/adhoc_up.sh` to setup the adhoc network. You should be able to ping the other robots with their local adhoc IP address.

Run the following steps to run the pose control demo:
- To download the models, change directory to `./src/sensing_cpp/models` and run `./download.sh`. The models will be downloaded to the `./src/sensing_cpp/models` directory.
- Change directory to the root of this folder (`ros2`), and run `docker-compose build`.
- Run `docker-compose up encoder` to start the encoder process, and run `docker-compose up controller` to start the controller process in another terminal.
