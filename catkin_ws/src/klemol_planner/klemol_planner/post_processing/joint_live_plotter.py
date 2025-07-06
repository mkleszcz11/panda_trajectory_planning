#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
from control_msgs.msg import JointTrajectoryControllerState
from threading import Lock
from matplotlib.animation import FuncAnimation
from collections import deque

class JointLivePlotter:
    def __init__(self):
        rospy.init_node('joint_live_plotter', anonymous=True)

        self.lock = Lock()
        self.max_len = 1000
        self.times = deque(maxlen=self.max_len)
        self.positions = deque(maxlen=self.max_len)
        self.velocities = deque(maxlen=self.max_len)

        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.joint_index = 0

        self.sub = rospy.Subscriber(
            # "/position_joint_trajectory_controller/state",
            "/effort_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.callback
        )

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)
        plt.tight_layout()
        plt.show()

    def callback(self, msg):
        with self.lock:
            now = rospy.Time.now().to_sec()
            self.times.append(now)
            self.positions.append(list(msg.actual.positions))
            self.velocities.append(list(msg.actual.velocities))

    def update_plot(self, _):
        with self.lock:
            if len(self.times) < 5:
                return

            times = np.array(self.times)
            pos = np.array(self.positions)
            vel = np.array(self.velocities)
            acc = np.gradient(vel, axis=0) / np.gradient(times)[:, None]
            jerk = np.gradient(acc, axis=0) / np.gradient(times)[:, None]

        self.axs[0].cla()
        self.axs[0].plot(times, pos[:, self.joint_index])
        self.axs[0].set_ylabel("Position")

        self.axs[1].cla()
        self.axs[1].plot(times, vel[:, self.joint_index])
        self.axs[1].set_ylabel("Velocity")

        self.axs[2].cla()
        self.axs[2].plot(times, jerk[:, self.joint_index])
        self.axs[2].set_ylabel("Jerk")
        self.axs[2].set_xlabel("Time [s]")

if __name__ == '__main__':
    JointLivePlotter()
