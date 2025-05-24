import rospy
import numpy as np
from control_msgs.msg import JointTrajectoryControllerState


class TrajectoryLogger:
    def __init__(self):
        self.time = []
        self.positions = []
        self.velocities = []
        self.modes = []
        self.start_time = rospy.Time.now().to_sec()
        self.current_mode = "undefined"

    def set_mode(self, mode: str):
        self.current_mode = mode

    def callback(self, msg: JointTrajectoryControllerState):
        t = rospy.Time.now().to_sec() - self.start_time
        self.time.append(t)
        self.positions.append(list(msg.actual.positions))
        self.velocities.append(list(msg.actual.velocities))
        self.modes.append(self.current_mode)

    def save(self, filename: str):
        np.savez(
            filename,
            time=np.array(self.time),
            pos=np.array(self.positions),
            vel=np.array(self.velocities),
            mode=np.array(self.modes)
        )
