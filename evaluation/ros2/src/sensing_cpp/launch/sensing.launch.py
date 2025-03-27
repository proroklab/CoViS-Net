import socket
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    base_hostname = socket.gethostname().replace("-", "_")
    own_base_ns = "/" + base_hostname
    lead = "robomaster" # "robomaster" or "unitree"
    if lead == "robomaster":
        other_cam_ns = {
            "/robomaster_0": None,
            "/robomaster_1": "/robomaster_0/camera_0",
            "/robomaster_2": "/robomaster_0/camera_0",
            "/robomaster_3": "/robomaster_0/camera_0",
            "/robomaster_4": "/robomaster_0/camera_0",
        }[own_base_ns]
    elif lead == "unitree":
        other_cam_ns = {
            "/robomaster_20": None,
            "/robomaster_0": "/robomaster_20/camera_0",
            "/robomaster_1": "/robomaster_20/camera_0",
            "/robomaster_2": "/robomaster_20/camera_0",
            "/robomaster_3": "/robomaster_20/camera_0",
            "/robomaster_4": "/robomaster_20/camera_0",
        }[own_base_ns]
    else:
        raise ValueError(f"Unknown lead: {lead}")

    own_cam_ns = own_base_ns + "/camera_0"
    swmc_config = f"cfg/swmc_net_config_adhoc_{base_hostname}.json"

    nodes = [
        Node(
            package="sensing_cpp",
            executable="encode_img",
            namespace=own_cam_ns,
            name="encode_img",
            parameters=[
                {
                    "model_enc_file": "models/oyu1brtpe18_float16_trt_enc.ts", # seq 128
                    #"model_enc_file": "models/0kc5po4ee18_float16_trt_enc.ts", # seq 128, 6D
                    "image_inp_crop": 224,
                    "swmc_config_file": swmc_config,
                }
            ],
        )
    ]
    if other_cam_ns is not None:
        pose_topic = "pose_" + other_cam_ns.replace("/", "").replace(
            "robomaster_", "r"
        ).replace("camera_", "c")

        nodes.append(
            Node(
                package="sensing_cpp",
                executable="predict_pose",
                namespace=own_cam_ns,
                name="predict_pose",
                parameters=[
                    {
                        "model_msg_file": "models/oyu1brtpe18_float16_trt_msg.ts", # 128
                        "model_post_file": "models/oyu1brtpe18_float32_jit_post.ts", # 128
                        #"model_msg_file": "models/0kc5po4ee18_float16_trt_msg.ts",  # 128, 6D
                        #"model_post_file": "models/0kc5po4ee18_float32_jit_cuda_post.ts",  # 128, 6D
                        "cam_namespace_other": other_cam_ns,
                        "pose_topic_name": pose_topic,
                        "swmc_config_file": swmc_config,
                    }
                ],
            )
        )

    return LaunchDescription(nodes)
