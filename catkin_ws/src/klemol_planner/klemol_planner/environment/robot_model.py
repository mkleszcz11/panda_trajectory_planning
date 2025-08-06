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
from klemol_planner.environment.collision_checker import CollisionChecker
import rospy
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
from klemol_planner.tests.main_test_logger import MainTestLogger

import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory
import subprocess

from moveit_commander import RobotTrajectory
import copy
# from klemol_planner.post_processing.path_post_processing import PathPostProcessing
# from klemol_planner.planners.base import Planner
# from moveit_commander import RobotModel
from moveit_commander import RobotState, RobotCommander
from geometry_msgs.msg import Pose

import PyKDL as kdl
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import re

import rospy
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from tf.transformations import euler_from_quaternion


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


    def sample_random_valid_configuration(self, collision_checker: CollisionChecker) -> np.ndarray:
        """
        Sample a random joint configuration within joint limits that is collision-free
        and kinematically feasible.

        Returns:
            Random joint configuration (7D np.ndarray)
        """
        max_attempts = 100
        for _ in range(max_attempts):
            config = self.sample_random_configuration()

            if collision_checker.is_collision_free:
                return config
        
        raise RuntimeError(f"Could not find a valid configuration in {max_attempts} attempts.")


    def sample_random_configuration(self) -> np.ndarray:
        """
        Sample a random joint configuration within the joint limits.

        Returns:
            Random joint configuration (7D np.ndarray)
        """
        return np.random.uniform(self.lower_bounds, self.upper_bounds)
    
    def sample_random_configuration_in_task_space(self, obstacle = None) -> np.ndarray:
        """
        Sample a random joint configuration that is within the task space defined by the end-effector pose.

        Returns:
            Random joint configuration (7D np.ndarray)
        """
        x_range = (0.1, 0.7)
        y_range = (-0.4, 0.4)
        z_range = (0.17, 0.5)
        
        while True:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            roll = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi/2, np.pi/2)
            yaw = np.random.uniform(-np.pi, np.pi)

            # Get ik solution for random sampled pose
            pose = PointWithOrientation(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
            config = self.ik(pose)

            if config is not None and self.is_within_limits(config):
                return config

    def sample_random_configuration_in_task_space_with_limited_tool_orientation(self, obstacle = None) -> np.ndarray:
        """
        Sample a random joint configuration that is within the task space defined by the end-effector pose,
        with limited tool orientation.

        Returns:
            Random joint configuration (7D np.ndarray)
        """
        x_range = (0.1, 0.7)
        y_range = (-0.4, 0.4)
        z_range = (0.17, 0.5)
        
        while True:
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            roll = np.random.uniform(-np.pi-0.01, -np.pi+0.01)
            pitch = np.random.uniform(-0.01, 0.01)
            yaw = np.random.uniform(-np.pi, np.pi)

            # Get ik solution for random sampled pose
            pose = PointWithOrientation(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
            config = self.ik(pose)

            if config is not None and self.is_within_limits(config):
                return config

    def is_within_limits(self, config: np.ndarray) -> bool:
        """
        Check whether the joint configuration is within bounds.

        Args:
            config: Joint configuration.

        Returns:
            True if within joint limits.
        """
        return np.all(config >= self.lower_bounds) and np.all(config <= self.upper_bounds)
    
    def is_in_task_space(self, config: np.ndarray) -> bool:
        """
        Check if the joint configuration is within the task space defined by the end-effector pose.

        Args:
            config: Joint configuration.

        Returns:
            True if the end-effector pose is within the task space.
        """
        pose = self.fk(config)
        return (pose.x >= 0.1 and pose.x <= 0.7 and
                pose.y >= -0.4 and pose.y <= 0.4 and
                pose.z >= 0.17 and pose.z <= 0.5)

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
            seed = self.group.get_current_joint_values()
            # seed = self.sample_random_configuration()
        sol = solver.get_ik(seed, pose.x, pose.y, pose.z, *quaternion)
        return np.array(sol) if sol is not None else None

    def fk(self, config: np.ndarray) -> PointWithOrientation:
        """
        Compute FK using MoveIt's compute_fk service (official, stable).

        Args:
            config: Joint angles (7D np.ndarray)

        Returns:
            Pose of end effector as PointWithOrientation.
        """
        rospy.wait_for_service('/compute_fk')
        fk_service = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        
        fk_request = GetPositionFKRequest()
        fk_request.header.frame_id = self.base_link
        fk_request.fk_link_names = [self.ee_link]

        joint_state = JointState()
        joint_state.name = self.moveit_group.get_active_joints()
        joint_state.position = config.tolist()
        fk_request.robot_state.joint_state = joint_state

        try:
            response = fk_service.call(fk_request)
            if not response.error_code.val == response.error_code.SUCCESS:
                raise RuntimeError(f"FK failed: {response.error_code.val}")

            pose: PoseStamped = response.pose_stamped[0]
            quat = [pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w]
            roll, pitch, yaw = euler_from_quaternion(quat)

            return PointWithOrientation(
                x=pose.pose.position.x,
                y=pose.pose.position.y,
                z=pose.pose.position.z,
                roll=roll,
                pitch=pitch,
                yaw=yaw
            )
        except rospy.ServiceException as e:
            raise RuntimeError(f"Service call failed: {e}")

    
    def _move_gripper(self, open: bool = True):
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
            goal.speed = 1.5
            client.send_goal(goal)
            client.wait_for_result()

        else:
            client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
            client.wait_for_server()

            goal = GraspGoal()
            goal.width = 0.065     # fully closed
            goal.speed = 1.5
            goal.force = 5.0     # closing force (adjust based on object)
            goal.epsilon.inner = 0.08
            goal.epsilon.outer = 0.08
            client.send_goal(goal)
            client.wait_for_result()

    def close_gripper(self):
        self._move_gripper(False)

    def open_gripper(self):
        self._move_gripper(True)

    def move_with_trajectory_planner(self,
                                     planner,
                                     goal: PointWithOrientation,
                                     post_processing = None,
                                     post_goal_path: list=None,
                                     pre_start_path: list=None,
                                     post_processing_method: t.Callable = None,
                                     logger: MainTestLogger= None):
        """
        Move to the goal using a defined planer. If post goal path is specified it will be added to the path
        that leads to the goal.

        Params:
            planner: planner object instance
            goal: Goal pose
            post_goal_path: List of waypoints to append after reaching a goal
        """
        path = []
        current_config = np.array(self.group.get_current_joint_values())

        path.append(current_config)

        ik_solver = IK(
            base_link=self.base_link,
            tip_link=self.ee_link,
            urdf_string=self.urdf_string,
            timeout=0.1,
            solve_type="Distance",
        )

        ### PRE START PATH ###
        if pre_start_path:
            for waypoint in pre_start_path:
                seed = path[-1]
                pose_to_append = self.ik_with_custom_solver(
                    pose = waypoint,
                    solver=ik_solver,
                    seed=seed
                )
                path.append(pose_to_append)

        planner.set_start(path[-1])
        planner.set_goal(goal)
        planned_path, success = planner.plan()
        if not success and logger is not None:
            logger.planning_successful = False

        # path += planned_path
        path += planned_path[1:]

        ### ADDING INTERMEDIATE NODES UP TO THE OBJECT ###
        if post_goal_path:
            for waypoint in post_goal_path:
                seed = path[-1]
                pose_to_append = self.ik_with_custom_solver(
                    pose = waypoint,
                    solver=ik_solver,
                    seed=seed
                )
                path.append(pose_to_append)
 
        if logger is not None:
            logger.planning_time = logger.stop_timer()
            logger.number_of_waypoints_before_post_processing = len(path)

        #### Call shortcutting function (edit path)
        if success:
            rospy.loginfo(f"Planner found path with {len(path)} waypoints.")
            rospy.loginfo(f"Fitting spline to the path...")
            # Smooth the path and execute smooth trajectory
            if logger is not None:
                logger.spline_fitting_start_time = logger.stop_timer()
            if post_processing_method is not None:
                trajectory = post_processing_method(
                    path=path,
                    joint_names=self.group.get_active_joints(),
                    velocity_limits=self.velocity_limits,
                    acceleration_limits=self.acceleration_limits,
                    max_vel_acc_multiplier = 0.3,
                )
            else:
                rospy.logwarn("No post-processing method provided, using default quintic BSpline.")
            # trajectory = post_processing.interpolate_quintic_bspline_trajectory(
            #     path=path,
            #     joint_names=self.group.get_active_joints(),
            #     velocity_limits=self.velocity_limits,
            #     acceleration_limits=self.acceleration_limits,
            #     max_vel_acc_multiplier = 0.3
            #     )
            # trajectory = post_processing.mock_interpolate(
            #     path=path,
            #     joint_names=self.group.get_active_joints(),
            #     velocity_limits=self.velocity_limits,
            #     acceleration_limits=self.acceleration_limits,
            #     segment_time = 2.0
            #     )
            if logger is not None:
                logger.spline_fitting_time = logger.stop_timer()
            self.send_trajectory_to_controller(trajectory)
        else:
            rospy.logwarn("Planner failed to find a path.")


    def move_with_trajectory_planner_predefined_time(self,
                                                     planner,
                                                     goal: PointWithOrientation,
                                                     post_processing = None,
                                                     post_goal_path: list=None,
                                                     pre_start_path: list=None,
                                                     post_processing_method: t.Callable = None,
                                                     logger: MainTestLogger= None,
                                                     duration: float = None):
        """
        Move to the goal using a defined planer. If post goal path is specified it will be added to the path
        that leads to the goal.

        Params:
            planner: planner object instance
            goal: Goal pose
            post_goal_path: List of waypoints to append after reaching a goal
        """
        path = []
        current_config = np.array(self.group.get_current_joint_values())

        path.append(current_config)

        ik_solver = IK(
            base_link=self.base_link,
            tip_link=self.ee_link,
            urdf_string=self.urdf_string,
            timeout=0.05,
            solve_type="Distance",
        )

        ### PRE START PATH ###
        if pre_start_path:
            for waypoint in pre_start_path:
                seed = path[-1]
                pose_to_append = self.ik_with_custom_solver(
                    pose = waypoint,
                    solver=ik_solver,
                    seed=seed
                )
                path.append(pose_to_append)

        planner.set_start(path[-1])
        planner.set_goal(goal)
        planned_path, success = planner.plan()
        if not success and logger is not None:
            logger.planning_successful = False

        # path += planned_path
        path += planned_path[1:]

        ### ADDING INTERMEDIATE NODES UP TO THE OBJECT ###
        if post_goal_path:
            for waypoint in post_goal_path:
                seed = path[-1]
                pose_to_append = self.ik_with_custom_solver(
                    pose = waypoint,
                    solver=ik_solver,
                    seed=seed
                )
                path.append(pose_to_append)
 
        if logger is not None:
            logger.planning_time = logger.stop_timer()
            logger.number_of_waypoints_before_post_processing = len(path)

        #### Call shortcutting function (edit path)
        if success:
            rospy.loginfo(f"Planner found path with {len(path)} waypoints.")
            rospy.loginfo(f"Fitting spline to the path...")
            # Smooth the path and execute smooth trajectory
            if logger is not None:
                logger.spline_fitting_start_time = logger.stop_timer()
            if post_processing_method is not None:
                trajectory = post_processing_method(
                    path=path,
                    joint_names=self.group.get_active_joints(),
                    velocity_limits=self.velocity_limits,
                    acceleration_limits=self.acceleration_limits,
                    max_vel_acc_multiplier = 0.3,
                    duration = 5.0
                )
            else:
                rospy.logwarn("No post-processing method provided, using default quintic BSpline.")
            # trajectory = post_processing.interpolate_quintic_bspline_trajectory(
            #     path=path,
            #     joint_names=self.group.get_active_joints(),
            #     velocity_limits=self.velocity_limits,
            #     acceleration_limits=self.acceleration_limits,
            #     max_vel_acc_multiplier = 0.3
            #     )
            # trajectory = post_processing.mock_interpolate(
            #     path=path,
            #     joint_names=self.group.get_active_joints(),
            #     velocity_limits=self.velocity_limits,
            #     acceleration_limits=self.acceleration_limits,
            #     segment_time = 2.0
            #     )
            if logger is not None:
                logger.spline_fitting_time = logger.stop_timer()
            self.send_trajectory_to_controller(trajectory)
        else:
            rospy.logwarn("Planner failed to find a path.")


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
