import socket
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    namespace = socket.gethostname().replace("-", "_") + "/camera_0"
    return LaunchDescription(
        [
            Node(
                package="sensing_cpp",
                executable="encode_img",
                namespace=namespace,
                name="encode_img",
                parameters=[
                    {
                        "model_enc_file": "models/oyu1brtpe18_float16_trt_enc.ts",  # seq 128
                        #"model_enc_file": "models/0kc5po4ee18_float16_trt_enc.ts",  # seq 128, 6D
                        "image_inp_crop": 224,
                    }
                ],
            ),
        ]
    )
