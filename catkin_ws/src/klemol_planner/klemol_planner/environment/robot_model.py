# robot_model.py

"""
Robot model abstraction for joint limits, FK, IK, and sampling logic.
This implementation assumes TRAC-IK is used for inverse kinematics.
"""
import typing as t
import numpy as np
import yaml
import os
from trac_ik_python.trac_ik import IK
from klemol_planner.goals.point_with_orientation import PointWithOrientation

import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion


class RobotModel:
    """
    Robot model wrapper for planning.

    Provides joint limits, inverse kinematics (via TRAC-IK), forward kinematics (via libfranka). #TODO -> use libfranka not moveit
    """

    def __init__(self, urdf_string: str, base_link: str, ee_link: str, joint_limits_path: str):
        """
        Initialize the robot model.

        Args:
            urdf_path: Path to the Panda URDF (or XACRO-generated URDF).
            base_link: Base link name (e.g., "panda_link0").
            ee_link: End-effector link name (e.g., "panda_link8").
            joint_limits_path: Path to the YAML file containing joint limits.
        """
        self.base_link = base_link
        self.ee_link = ee_link
        self.hand_z_offset = 0.0

        # Load joint limits from YAML
        with open(joint_limits_path, 'r') as f:
            data = yaml.safe_load(f)
            self.lower_bounds = np.array(data['joint_limits']['lower'])
            self.upper_bounds = np.array(data['joint_limits']['upper'])

        # Setup TRAC-IK
        self.ik_solver = IK(self.base_link, self.ee_link, urdf_string=urdf_string)
        self.num_joints = len(self.lower_bounds)

        # Initialize MoveIt Commander interface for FK
        moveit_commander.roscpp_initialize([])
        self.moveit_group = moveit_commander.MoveGroupCommander("panda_arm")

    def _load_urdf(self, path: str) -> str:
        """Load URDF content from a file."""
        with open(path, 'r') as f:
            return f.read()

    def sample_random_configuration(self) -> np.ndarray:
        """
        Sample a random joint configuration within the joint limits.

        Returns:
            Random joint configuration (7D np.ndarray)
        """
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def is_within_limits(self, config: np.ndarray) -> bool:
        """
        Check whether the joint configuration is within bounds.

        Args:
            config: Joint configuration.

        Returns:
            True if within joint limits.
        """
        return np.all(config >= self.lower_bounds) and np.all(config <= self.upper_bounds)

    def ik(self, pose: PointWithOrientation) -> t.Optional[np.ndarray]:
        """
        Compute inverse kinematics using TRAC-IK.

        Args:
            pose: Desired end-effector pose.

        Returns:
            Joint configuration if solution exists, else None.
        """
        quaternion = pose.to_quaternion()
        seed = self.sample_random_configuration()
        sol = self.ik_solver.get_ik(seed, pose.x, pose.y, pose.z, *quaternion)
        return np.array(sol) if sol is not None else None

    def fk(self, config: np.ndarray) -> t.Optional[PointWithOrientation]:
        """
        Compute the forward kinematics using MoveIt Commander. Do it for simplicity,
        might be slow but should be easy to debug.

        Args:
            config: Joint angles (7D array)

        Return:
            Pose of the end effector as PointWithOrientation.
        """
        self.moveit_group.set_joint_value_target(config.tolist())
        pose: Pose = self.moveit_group.get_current_pose().pose

        # Convert quaternion to roll, pitch, yaw
        quat = [pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        return PointWithOrientation(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z,
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )