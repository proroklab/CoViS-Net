import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robomaster_msgs.msg import WheelSpeed
from sensor_msgs.msg import Joy


class RemoteControl(Node):
    ROBOT_WHEEL_RADIUS = 0.15
    ROBOT_BASE_LEN_X = 0.3
    ROBOT_BASE_LEN_Y = 0.3

    def __init__(self):
        super().__init__("remote_control")

        self.create_subscription(
            Joy,
            "joy",
            self.joy_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.vel_pub = self.create_publisher(
            WheelSpeed,
            "cmd_wheels",
            qos_profile=qos_profile_sensor_data,
        )

        len_xy = self.ROBOT_BASE_LEN_X + self.ROBOT_BASE_LEN_Y
        self.dyn_inv = (
            np.array(
                [
                    [1, -1, -len_xy],
                    [1, 1, len_xy],
                    [1, 1, -len_xy],
                    [1, -1, len_xy],
                ]
            )
            / self.ROBOT_WHEEL_RADIUS
        )
        print("Init")

    def pub_vel(self, vx, vy, omega):
        vel = WheelSpeed()
        v = np.array([vx, vy, omega])
        wheel_speed = np.rad2deg(self.dyn_inv @ v)
        vel.fl = int(wheel_speed[0])
        vel.fr = int(wheel_speed[1])
        vel.rl = int(wheel_speed[2])
        vel.rr = int(wheel_speed[3])
        self.vel_pub.publish(vel)

    def joy_callback(self, joy):
        desired_vx = joy.axes[1]
        desired_vy = joy.axes[0]
        desired_w = joy.axes[2]

        max_w = 1.0
        max_v = 1.5
        omega = max(-max_w, min(max_w, desired_w))
        vy = max(-max_v, min(max_v, desired_vy))
        vx = max(-max_v, min(max_v, desired_vx))

        print(vx, vy, omega)
        self.pub_vel(vx, vy, omega)


def main(args=None):
    rclpy.init(args=args)
    node = RemoteControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
