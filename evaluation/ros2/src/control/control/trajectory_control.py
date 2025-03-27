try:
    import rclpy
    from rclpy.node import Node
    from freyja_msgs.msg import ReferenceState
except ImportError:

    class Node:
        def __init__(self, name):
            pass


import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def state_figure_eight(t, amplitude=1.5, frequency=0.5, fixed_yaw=None):
    pe = amplitude * math.sin(frequency * t)
    pn = amplitude * math.sin(2 * frequency * t) / 2

    vn = amplitude * frequency * math.cos(2 * frequency * t)
    ve = amplitude * frequency * math.cos(frequency * t)

    if fixed_yaw is None:
        yaw = math.atan2(ve, vn)
    else:
        yaw = math.pi / 2

    return {"pe": pe, "pn": pn, "vn": vn, "ve": ve, "yaw": yaw}


def straight_segment(d, v, direction):
    """Compute position and velocity for a straight segment."""
    pe, pn = 0.0, 0.0
    ve, vn = 0.0, 0.0

    if direction == "v":
        pe = d
        ve = v
    elif direction == "h":
        pn = d
        vn = v

    return pe, pn, ve, vn


def curved_segment(d, radius, start_angle, speed):
    """Compute position and velocity for a curved (quarter-circle) segment."""
    angle = start_angle + d / radius
    pe = radius * math.cos(angle)
    pn = radius * math.sin(angle)
    ve = -math.sin(angle) * speed
    vn = math.cos(angle) * speed

    return pe, pn, ve, vn


def state_rounded_rectangle(t, height=6.0, width=3.0, corner_radius=1.0, speed=1.0):
    assert height >= 2 * corner_radius
    assert width >= 2 * corner_radius

    # Calculate lengths of straight and curved segments
    straight_height = height - 2 * corner_radius
    straight_width = width - 2 * corner_radius
    quarter_circle_length = math.pi * corner_radius / 2

    # Total length of the path
    total_length = 2 * (straight_height + straight_width) + 4 * quarter_circle_length

    # Time for one complete cycle
    total_time = total_length / speed

    # Normalize time
    t_normalized = t % total_time

    # Distance along the path
    d = t_normalized * speed
    if d < quarter_circle_length:  # Top right corner
        d_adjusted = d
        pe, pn, ve, vn = curved_segment(d_adjusted, corner_radius, math.pi / 2, speed)
        pe -= height / 2 - corner_radius
        pn += width / 2 - corner_radius
    elif d < quarter_circle_length + straight_width:  # Top edge
        d_adjusted = d - quarter_circle_length
        pe, pn, ve, vn = straight_segment(-d_adjusted, -speed, "h")
        pe -= height / 2
        pn += width / 2 - corner_radius
    elif d < 2 * quarter_circle_length + straight_width:  # Top left
        d_adjusted = d - quarter_circle_length - straight_width
        pe, pn, ve, vn = curved_segment(d_adjusted, corner_radius, -math.pi, speed)
        pn -= width / 2 - corner_radius
        pe -= height / 2 - corner_radius
    elif d < 2 * quarter_circle_length + straight_height + straight_width:  # Left edge
        d_adjusted = d - 2 * quarter_circle_length - straight_width
        pe, pn, ve, vn = straight_segment(d_adjusted, speed, "v")
        pn -= width / 2
        pe -= height / 2 - corner_radius
    elif (
        d < 3 * quarter_circle_length + straight_height + straight_width
    ):  # Bottom left
        d_adjusted = d - 2 * quarter_circle_length - straight_height - straight_width
        pe, pn, ve, vn = curved_segment(d_adjusted, corner_radius, -math.pi / 2, speed)
        pn -= width / 2 - corner_radius
        pe += height / 2 - corner_radius
    elif d < 3 * quarter_circle_length + straight_height + 2 * straight_width:  # Bottom
        d_adjusted = d - 3 * quarter_circle_length - straight_height - straight_width
        pe, pn, ve, vn = straight_segment(d_adjusted, speed, "h")
        pn -= width / 2 - corner_radius
        pe += height / 2
    elif (
        d < 4 * quarter_circle_length + straight_height + 2 * straight_width
    ):  # Bottom right
        d_adjusted = (
            d - 3 * quarter_circle_length - straight_height - 2 * straight_width
        )
        pe, pn, ve, vn = curved_segment(d_adjusted, corner_radius, 0, speed)
        pn += width / 2 - corner_radius
        pe += height / 2 - corner_radius
    elif d < 4 * quarter_circle_length + 2 * straight_height + 2 * straight_width:
        d_adjusted = (
            d - 4 * quarter_circle_length - straight_height - 2 * straight_width
        )
        pe, pn, ve, vn = straight_segment(-d_adjusted, -speed, "v")
        pn += width / 2
        pe += height / 2 - corner_radius

    yaw = math.pi / 2  # math.atan2(ve, vn)

    return {"pe": pe, "pn": pn, "vn": vn, "ve": ve, "yaw": yaw}


def state_oval_trajectory(t, minor=6.0, major=3.0, speed_fac=1.0):
    # Semi-axes of the oval
    a = major / 2  # semi-major axis
    b = minor / 2  # semi-minor axis

    # Period of the complete oval path
    # This is based on the approximate circumference of the ellipse
    #h = (a - b) ** 2 / (a + b) ** 2
    #circumference = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    #period = circumference / speed

    # Parametric angle for time t
    #angle = (2 * math.pi * t) / period
    #print(angle)

    # Position calculations
    pe = a * math.cos(t * speed_fac)  # x-coordinate
    pn = b * math.sin(t * speed_fac)  # y-coordinate

    # Velocity calculations (derivatives of the position functions)
    ve = -a * math.sin(t * speed_fac)  # derivative of x-coordinate
    vn = b * math.cos(t * speed_fac)  # derivative of y-coordinate

    # Yaw angle calculation
    # yaw = ((math.atan2(ve, vn) - math.pi / 2) % (2 * math.pi)) - math.pi
    yaw = math.atan2(ve, vn)

    return {"pe": pe, "pn": pn, "vn": vn, "ve": ve, "yaw": yaw}


def state_compose(state_a, state_b):
    rotation_a = R.from_euler("z", state_a["yaw"], degrees=False)
    pos_b_rotated = rotation_a.apply([state_b["pe"], state_b["pn"], 0.0])
    vel_b_rotated = rotation_a.apply([state_b["ve"], state_b["vn"], 0.0])

    return {
        "pe": state_a["pe"] + pos_b_rotated[0],
        "pn": state_a["pn"] + pos_b_rotated[1],
        "ve": state_a["ve"] + vel_b_rotated[0],
        "vn": state_a["vn"] + vel_b_rotated[1],
        "yaw": np.mod(state_a["yaw"] + state_b["yaw"] + np.pi, 2 * np.pi) - np.pi,
    }


class TrajectoryControl(Node):
    def __init__(self):
        super().__init__("trajectory_control")

        self.freq = 30.0
        self.pose_publisher_timer = self.create_timer(1 / self.freq, self.step_agent)
        self.elapsed_time = 0.0

        self.state_pub = self.create_publisher(
            ReferenceState,
            "reference_state",
            1,
        )

    def step_time(self):
        self.elapsed_time += 1.0 / self.freq

    def step_agent(self):
        self.step_time()

        state = ReferenceState()
        state.header.stamp = self.get_clock().now().to_msg()

        # state_dict = state_figure_eight(self.elapsed_time, amplitude=1.5, frequency=0.4) #, fixed_yaw=math.pi/2)
        # state_dict = state_rounded_rectangle(self.elapsed_time, height=3.0, width=2.0, corner_radius=0.75, speed=0.5)
        # state_dict = state_oval_trajectory(self.elapsed_time, minor=2.0, major=3.0, speed=0.5)

        # state_dict = state_rounded_rectangle(self.elapsed_time, height=1.5, width=1.5, corner_radius=0.3, speed=0.2)
        state_dict = state_oval_trajectory(
            self.elapsed_time, minor=1.25, major=1.25, speed=0.5
        )

        state.pe = state_dict["pe"] + 1.0
        state.pn = state_dict["pn"]
        state.vn = state_dict["vn"]
        state.ve = state_dict["ve"]
        state.yaw = 0.0  # state_dict['yaw']
        self.state_pub.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


def plot():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    states = []
    for timestep in range(0, 30):
        t = timestep / 5.0
        #state_a = state_figure_eight(
        #    t, amplitude=1.5, frequency=0.5, fixed_yaw=math.pi / 2
        #)
        state_a = state_oval_trajectory(
            t, minor=2.0, major=3.0, speed=1.5
        )
        state_b = state_oval_trajectory(t, minor=1.0, major=1.0, speed=0.5)
        #state_ab = state_compose(state_a, state_b)
        # state = state_figure_eight(t, amplitude=1.5, frequency=0.5, fixed_yaw=None)
        # state = state_rounded_rectangle(t, height=4.0, width=1.0, corner_radius=0.5, speed=1.0)
        # state = state_oval_trajectory(t, minor=3.0, major=6.0, speed=1.0)
        state_a["agent"] = "a"
        #state_ab["agent"] = "b"
        states.append(state_a)
        #states.append(state_ab)

    states = pd.DataFrame(states)
    states.v_norm = np.linalg.norm(states[["vn", "ve"]], axis=1)
    #print(states.v_norm.min(), states.v_norm.mean(), states.v_norm.max())
    plt.quiver(states.pn, -states.pe, states.vn, -states.ve)
    yaw_n = np.sin(states.yaw) * 2
    yaw_e = np.cos(states.yaw) * 2
    # plt.quiver(states.pn, -states.pe, yaw_n, -yaw_e, color='r')
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.plot(states.pn, -states.pe)
    plt.show()


if __name__ == "__main__":
    # main()
    plot()
