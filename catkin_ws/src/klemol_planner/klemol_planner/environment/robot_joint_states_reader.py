from sensor_msgs.msg import JointState
import numpy as np
import rospy
import typing as t

class JointStatesReader:
    def __init__(self, joint_names: t.List[str]):
        self.joint_names = joint_names
        self.latest_state: t.Optional[JointState] = None
        self._sub = rospy.Subscriber("/joint_states", JointState, self._callback)

    def _callback(self, msg: JointState):
        self.latest_state = msg

    def get_current_positions(self) -> np.ndarray:
        if self.latest_state is None:
            raise RuntimeError("JointState not yet received.")
        idx_map = {name: i for i, name in enumerate(self.latest_state.name)}
        return np.array([self.latest_state.position[idx_map[j]] for j in self.joint_names])

    def get_current_velocities(self) -> np.ndarray:
        if self.latest_state is None:
            raise RuntimeError("JointState not yet received.")
        idx_map = {name: i for i, name in enumerate(self.latest_state.name)}
        return np.array([self.latest_state.velocity[idx_map[j]] for j in self.joint_names])
