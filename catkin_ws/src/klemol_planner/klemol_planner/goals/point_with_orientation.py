import numpy as np
import tf.transformations as tf_trans
from scipy.spatial.transform import Rotation as R

class PointWithOrientation:
    def __init__(self, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        self.set_position(x, y, z)
        self.set_orientation(roll, pitch, yaw)

    def __str__(self):
        return f"PointWithOrientation(x={self.x}, y={self.y}, z={self.z}, roll={self.roll}, pitch={self.pitch}, yaw={self.yaw})"

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def get_position(self):
        return np.array([self.x, self.y, self.z])

    def get_orientation(self):
        return np.array([self.roll, self.pitch, self.yaw])

    def as_matrix(self):
        rot = R.from_euler('xyz', [self.roll, self.pitch, self.yaw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [self.x, self.y, self.z]
        return T

    @classmethod
    def from_matrix(cls, T):
        position = T[:3, 3]
        rot = R.from_matrix(T[:3, :3]).as_euler('xyz')
        return cls(position[0], position[1], position[2], rot[0], rot[1], rot[2])
    
    def to_quaternion(self) -> np.ndarray:
        """
        Convert roll, pitch, yaw to a quaternion.

        Return:
            Quaternion as np.array([x, y, z, w])
        """
        return tf_trans.quaternion_from_euler(self.roll, self.pitch, self.yaw)
    