# robot_model.py

"""
Robot model abstraction for joint limits, FK, IK, and sampling logic.
This implementation assumes TRAC-IK is used for inverse kinematics.
"""
import typing as t
import tf.transformations as tf_trans
import time
import numpy as np
import yaml
import geometry_msgs.msg
import os
from trac_ik_python.trac_ik import IK
from klemol_planner.goals.point_with_orientation import PointWithOrientation

import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion

import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory
import subprocess

from moveit_commander import RobotTrajectory
import copy

class Robot:
    """
    Robot model wrapper for planning.

    Provides joint limits, inverse kinematics (via TRAC-IK), forward kinematics (via libfranka). #TODO -> use libfranka not moveit
    """

    def __init__(self, urdf_string: str = None, base_link: str = None, ee_link: str = None, joint_limits_path: str = None):
        """
        Initialize the robot model.

        Args:
            urdf_path: Path to the Panda URDF (or XACRO-generated URDF).
            base_link: Base link name (e.g., "panda_link0").
            ee_link: End-effector link name (e.g., "panda_link8").
            joint_limits_path: Path to the YAML file containing joint limits.
        """

        self.group = moveit_commander.MoveGroupCommander("panda_arm")

        if base_link is not None:
            self.base_link = base_link
        else:
            self.base_link = "panda_link0"

        if ee_link is not None:
            self.ee_link = ee_link
        else:
            self.ee_link = "panda_link8"

        if urdf_string is not None:
            self.urdf_string = urdf_string
        else:
            pkg_root = rospy.get_param("/klemol_planner/package_path", default="/home/neurorobotic_student/panda_trajectory_planning/catkin_ws/src/klemol_planner")
            xacro_path = f"{pkg_root}/panda_description/panda.urdf.xacro"
            self.urdf_string = subprocess.check_output(["xacro", xacro_path]).decode("utf-8")

        # Load joint limits
        if joint_limits_path is None:
            pkg_root = rospy.get_param("/klemol_planner/package_path", default="/home/neurorobotic_student/panda_trajectory_planning/catkin_ws/src/klemol_planner")
            xacro_path = f"{pkg_root}/panda_description/panda.urdf.xacro"
            joint_limits_path = f"{pkg_root}/config/joint_limits.yaml"

        with open(joint_limits_path, 'r') as f:
            data = yaml.safe_load(f)
            self.lower_bounds = np.array(data['joint_limits']['lower'])
            self.upper_bounds = np.array(data['joint_limits']['upper'])
            self.velocity_limits = np.array(data['joint_limits']['velocity'])
            self.acceleration_limits = np.array(data['joint_limits']['acceleration'])
            self.effort_limits = np.array(data['joint_limits']['effort'])

        self.hand_z_offset = 0.0

        # Setup TRAC-IK
        self.ik_solver = IK(self.base_link, self.ee_link, urdf_string=self.urdf_string)
        self.num_joints = len(self.lower_bounds)

        # Initialize MoveIt Commander interface for FK
        moveit_commander.roscpp_initialize([])
        self.moveit_group = moveit_commander.MoveGroupCommander("panda_arm")

    def _load_urdf(self, path: str) -> str:
        """Load URDF content from a file."""
        with open(path, 'r') as f:
            return f.read()
        
    def get_current_joint_values(self):
        return self.moveit_group.get_current_joint_values()

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
    
    def ik_with_custom_solver(self, pose: PointWithOrientation, solver: IK, seed: np.ndarray = None) -> t.Optional[np.ndarray]:
        quaternion = pose.to_quaternion()
        if seed is None:
            seed = self.sample_random_configuration()
        sol = solver.get_ik(seed, pose.x, pose.y, pose.z, *quaternion)
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
    
    def move_gripper(self, open: bool = True):
        """
        Open or close the gripper.

        Args:
            open: If True, open the gripper. If False, close it.
        """
        if open:
            client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
            client.wait_for_server()

            goal = MoveGoal()
            goal.width = 0.08     # fully open (max 0.08 m)
            goal.speed = 0.1
            client.send_goal(goal)
            client.wait_for_result()

        else:
            client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
            client.wait_for_server()

            goal = GraspGoal()
            goal.width = 0.00     # fully closed
            goal.speed = 0.1
            goal.force = 30.0     # closing force (adjust based on object)
            goal.epsilon.inner = 0.005
            goal.epsilon.outer = 0.005
            client.send_goal(goal)
            client.wait_for_result()

    def send_trajectory_to_controller(self, trajectory: JointTrajectory):
        """
        Send a trajectory to the robot's controller.

        Args:
            trajectory: A JointTrajectory message.
        """
        client = actionlib.SimpleActionClient('/position_joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        # client = actionlib.SimpleActionClient('/effort_joint_trajectory_controller/follow_joint_trajectory', FollowJointTrajectoryAction)

        client.wait_for_server()
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = trajectory
        client.send_goal(goal)
        client.wait_for_result()

    def move_to_joint_config(self, joint_config):
        """Move the robot to a specific joint configuration."""
        self.group.set_max_velocity_scaling_factor(0.25)
        self.group.set_max_acceleration_scaling_factor(0.25)
        self.group.set_joint_value_target(joint_config)
        rospy.loginfo(f"Moving to joint configuration: {joint_config}")
        start_time = time.time()
        self.group.go(wait=True)
        end_time = time.time()

    def move_to_pose_trac_ik(self, position: PointWithOrientation):
        """Move the robot using TRAC-IK"""
        x, y, z = position.x, position.y, position.z
        roll, pitch, yaw = position.roll, position.pitch, position.yaw
        quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
        seed_state = np.random.uniform(self.lower_bounds, self.upper_bounds)  # Random seed
        joint_positions = self.ik_solver.get_ik(seed_state, x, y, z, *quaternion)

        if joint_positions:
            rospy.loginfo(f"TRAC-IK Solution Found for ({x}, {y}, {z})")
            self.robot_model.execute_joint_positions(joint_positions, "TRAC-IK")
        else:
            rospy.logerr("No IK solution found!")

    def execute_joint_positions(self, joint_positions, method):
        """Execute a joint position command and log the data"""
        self.group.set_max_velocity_scaling_factor(0.25)
        self.group.set_max_acceleration_scaling_factor(0.25)
        self.group.set_joint_value_target(joint_positions)
        start_time = time.time()
        self.group.go(wait=True)
        end_time = time.time()

    def move_to_pose_planner(self, pose: PointWithOrientation):
        """Move the robot using MoveIt's motion planner"""
        pose_target = geometry_msgs.msg.Pose()

        # Convert roll, pitch, yaw to quaternion
        quaternion = tf_trans.quaternion_from_euler(pose.roll, pose.pitch, pose.yaw)

        # Assign position and orientation
        pose_target.position.x = pose.x
        pose_target.position.y = pose.y
        pose_target.position.z = pose.z
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]

        # Set target and execute
        self.group.set_pose_target(pose_target)
        start_time = time.time()
        success = self.group.go(wait=True)
        end_time = time.time()

        if success:
            rospy.loginfo("Motion planning successful")
        else:
            rospy.logerr("Motion planning failed")

    def move_cartesian(self, pose: PointWithOrientation):
        """
        Move the robot using Cartesian path planning from current to target pose.

        Args:
            pose: Target pose as PointWithOrientation.
        """
        # Convert pose
        target_pose = geometry_msgs.msg.Pose()
        target_pose.position.x = pose.x
        target_pose.position.y = pose.y
        target_pose.position.z = pose.z
        quat = tf_trans.quaternion_from_euler(pose.roll, pose.pitch, pose.yaw)
        target_pose.orientation.x = quat[0]
        target_pose.orientation.y = quat[1]
        target_pose.orientation.z = quat[2]
        target_pose.orientation.w = quat[3]

        waypoints = []
        waypoints.append(copy.deepcopy(self.moveit_group.get_current_pose().pose))
        waypoints.append(copy.deepcopy(target_pose))

        (plan, fraction) = self.moveit_group.compute_cartesian_path(
            waypoints, 0.01  # waypoints to follow  # eef_step
        )

        print(f"Fraction of path computed: {fraction}")
        print(f"Number of waypoints: {len(plan.joint_trajectory.points)}")

        # if fraction < 0.99:
        #     rospy.logwarn(f"Cartesian path only computed {fraction*100:.1f}% of the way.")
        #     return
        # else:
        #     rospy.loginfo("Cartesian path planning successful.")

        # # Assign time_from_start manually
        # def assign_time_parametrization(trajectory, time_step=0.01):
        #     time = 0.0
        #     for point in trajectory.joint_trajectory.points:
        #         point.time_from_start = rospy.Duration.from_sec(time)
        #         time += time_step

        self.moveit_group.execute(plan, wait=True)
