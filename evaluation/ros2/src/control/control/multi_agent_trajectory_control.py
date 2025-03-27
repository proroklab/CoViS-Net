try:
    import rclpy
    from rclpy.node import Node
    from freyja_msgs.msg import ReferenceState
except ImportError:

    class Node:
        def __init__(self, name):
            pass


import math
from .trajectory_control import (
    state_figure_eight,
    state_rounded_rectangle,
    state_oval_trajectory,
    state_compose,
)


class TrajectoryControl(Node):
    def __init__(self):
        super().__init__("trajectory_control")

        self.freq = 30.0
        self.pose_publisher_timer = self.create_timer(1 / self.freq, self.step_agent)
        self.elapsed_time = 0.0

        self.state_pubs = {}
        for uuid in [
            "robomaster_1",
            "robomaster_2",
        ]:
            self.state_pubs[uuid] = self.create_publisher(
                ReferenceState,
                f"/{uuid}/reference_state",
                1,
            )

    def step_time(self):
        self.elapsed_time += 1.0 / self.freq

    def step_agent(self):
        self.step_time()

        for i, (uuid, state_pub) in enumerate(self.state_pubs.items()):
            state = ReferenceState()
            state.header.stamp = self.get_clock().now().to_msg()

            state_dict = state_oval_trajectory(
                self.elapsed_time, minor=1.0, major=1.0, speed_fac=1.0
            )

            state.pe = state_dict["pe"] + 1.0 + i
            state.pn = state_dict["pn"]
            state.vn = state_dict["vn"]
            state.ve = state_dict["ve"]
            state.yaw = 0.0  # state_dict['yaw']
            state_pub.publish(state)


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
    for timestep in range(0, 60):
        t = timestep / 5.0
        state_a = state_figure_eight(
            t, amplitude=1.5, frequency=0.5, fixed_yaw=math.pi / 2
        )
        state_b = state_oval_trajectory(t, minor=1.0, major=1.0, speed=0.5)
        state_ab = state_compose(state_a, state_b)
        # state = state_figure_eight(t, amplitude=1.5, frequency=0.5, fixed_yaw=None)
        # state = state_rounded_rectangle(t, height=4.0, width=1.0, corner_radius=0.5, speed=1.0)
        # state = state_oval_trajectory(t, minor=3.0, major=6.0, speed=1.0)
        state_a["agent"] = "a"
        state_ab["agent"] = "b"
        states.append(state_a)
        states.append(state_ab)

    states = pd.DataFrame(states)
    states.v_norm = np.linalg.norm(states[["vn", "ve"]], axis=1)
    print(states.v_norm.min(), states.v_norm.mean(), states.v_norm.max())
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
