import socket
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    own_base_ns = "/" + socket.gethostname().replace("-", "_")
    nodes = []
    lead = "robomaster" # "robomaster" or "unitree"
    if lead == "robomaster":
        pose_topic = "camera_0/pose_r0c0"
        poses = {
            "/robomaster_1": {"px": 0.5, "py": -0.5, "yaw": 0.0}, # left
            "/robomaster_2": {"px": -0.5, "py": -0.5, "yaw": 0.0}, # right
            "/robomaster_3": {"px": 0.5, "py": 0.5, "yaw": 0.0}, # back left
        }
    elif lead == "unitree":
        pose_topic = "camera_0/pose_r20c0"
        poses = {
            "/robomaster_0": {"px": 0.5, "py": -0.5, "yaw": 0.0}, # left
            "/robomaster_1": {"px": -0.5, "py": -0.5, "yaw": 0.0}, # right
            "/robomaster_2": {"px": 0.5, "py": -1.0, "yaw": 0.0}, # back left
            "/robomaster_3": {"px": -0.5, "py": -1.0, "yaw": 0.0}, # back right
            "/robomaster_4": {"px": 0.0, "py": 1.0, "yaw": 0.0}, # front
        }
    else:
        raise ValueError(f"Unknown lead: {lead}")

    if own_base_ns == "/robomaster_0" and lead == "robomaster":
        node = Node(
            package="control",
            executable="remote_control",
            namespace=own_base_ns,
            name="remote_control",
        )
        nodes.append(node)
    elif own_base_ns in poses.keys():
        node = Node(
            package="control",
            executable="simple_pose_control",
            namespace=own_base_ns,
            name="simple_pose_control",
            parameters=[
                {
                    "pose_topic": pose_topic,
                    "ref_px": poses[own_base_ns]["px"],
                    "ref_py": poses[own_base_ns]["py"],
                    "ref_yaw": poses[own_base_ns]["yaw"],
                }
            ],
        )
        nodes.append(node)
    else:
        raise ValueError(f"Unknown base ns: {own_base_ns}")

    return LaunchDescription(nodes)
