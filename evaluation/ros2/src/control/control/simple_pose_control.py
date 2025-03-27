import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from robomaster_msgs.msg import WheelSpeed
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R


class SimplePoseControl(Node):
    ROBOT_WHEEL_RADIUS = 0.15
    ROBOT_BASE_LEN_X = 0.3
    ROBOT_BASE_LEN_Y = 0.3

    def __init__(self):
        super().__init__("simple_pose_control")
        self.declare_parameter("ref_px", 0.5)
        self.declare_parameter("ref_py", 0.0)
        self.declare_parameter("ref_yaw", 0.0)
        self.declare_parameter("pose_topic", "camera_0/pose_r0c0")

        self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            qos_profile=qos_profile_sensor_data,
        )

        self.vel_pub = self.create_publisher(
            WheelSpeed,
            "cmd_wheels",
            qos_profile=qos_profile_sensor_data,
        )

        self.control_timer = self.create_timer(1.0 / 15, self.control)
        self.cov = None
        self.p = None
        self.r = None
        self.prev_e_x = 0.0
        self.prev_e_y = 0.0
        self.prev_e_yaw = 0.0

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

    def control(self):
        if self.p is None or self.r is None or self.cov is None:
            return

        vx = 0.0
        vy = 0.0
        omega = 0.0
        if self.cov[0, 0] < 0.75 and self.cov[2, 2] < 0.75:
            max_vx = 0.75
            gain_pvx = 1.3
            gain_dvx = 0.08
            e_x = self.p.x - self.get_parameter("ref_px").value
            vy = -max(
                -max_vx,
                min(max_vx, gain_pvx * e_x + gain_dvx * (e_x - self.prev_e_x) * 15),
            )
            self.prev_e_x = e_x

            max_vy = 0.75
            gain_pvy = 1.45
            gain_dvy = 0.04
            e_y = self.p.z - self.get_parameter("ref_py").value
            vx = -max(
                -max_vy,
                min(max_vy, gain_pvy * e_y + gain_dvy * (e_y - self.prev_e_y) * 15),
            )
            self.prev_e_y = e_y

        if self.cov[3, 3] < 0.4:
            max_w = 0.8
            gain_pomega = 0.35
            gain_domega = 0.02

            e_yaw = self.r[1] - self.get_parameter("ref_yaw").value
            omega = max(
                -max_w,
                min(
                    max_w,
                    gain_pomega * e_yaw + gain_domega * (e_yaw - self.prev_e_yaw) * 15,
                ),
            )
            self.prev_e_yaw = e_yaw

        self.pub_vel(vx, vy, omega)

    def pose_callback(self, pose):
        self.p = pose.pose.pose.position
        q = pose.pose.pose.orientation
        self.cov = pose.pose.covariance.reshape(6, 6)
        self.r = R.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()


def main(args=None):
    rclpy.init(args=args)
    node = SimplePoseControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
