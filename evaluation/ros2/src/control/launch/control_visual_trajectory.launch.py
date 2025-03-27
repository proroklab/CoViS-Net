import socket
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    own_base_ns = "/" + socket.gethostname().replace("-", "_")
    nodes = [
        Node(
            package="control",
            executable="simple_pose_control",
            namespace=own_base_ns,
            name="simple_pose_control",
            parameters=[
                {
                    "ref_px": 0.0,
                    "ref_py": 0.0,
                    "ref_yaw": 0.0,
                }
            ],
        )
    ]

    return LaunchDescription(nodes)
