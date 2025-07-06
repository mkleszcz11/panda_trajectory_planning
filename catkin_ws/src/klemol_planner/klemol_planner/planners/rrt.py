import typing as t
import numpy as np
import time
import random

from klemol_planner.planners.base import Planner
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.planners.nodes import TreeNode

import rospy

from trac_ik_python.trac_ik import IK

class RRTPlanner(Planner):
    """
    Rapidly-exploring Random Tree (RRT) planner implementation.

    This class extends the base Planner and implements the core RRT algorithm for generating a
    collision-free path from a start configuration to a goal pose using random sampling.
    """

    def __init__(self,
                 robot_model: Robot,
                 collision_checker: CollisionChecker,
                 parameters: dict):
        """
        Initialize RRT planner.

        Args:
            robot_model: Provides joint limits, FK, IK.
            collision_checker: For checking collisions along the path.
            parameters: Dict of planner-specific parameters like step_size, max_iterations.
        """
        super().__init__(robot_model, collision_checker, parameters)
        self.step_size: float = parameters.get("step_size", 0.1)           # [rad], max distance between nodes
        self.max_iterations: int = parameters.get("max_iterations", 1000)
        self.goal_bias: float = parameters.get("bias", 0.05)               # Probability of sampling the goal
        self.max_time: float = parameters.get("max_time", 5.0)             # Max planning time in seconds

    def plan(self) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Execute the RRT planning algorithm.

        Return:
            A tuple of:
            - List of joint configurations (np.ndarray) forming the planned path.
            - Boolean indicating if planning succeeded.
        """
        rospy.loginfo("Starting RRT planning...")
        rospy.loginfo(f"Parameters: {self.parameters}")
        if self.start_config is None or self.goal_pose is None:
            raise ValueError("Start configuration and goal pose must be set before planning.")

        # Inverse kinematics to find a goal configuration
        # goal_config = self.robot_model.ik(self.goal_pose)

        custom_solver = IK(base_link = self.robot_model.base_link,
                           tip_link = self.robot_model.ee_link,
                           urdf_string = self.robot_model.urdf_string,
                           timeout = 1.0,
                           solve_type="Distance")

        # random_seed = np.random.uniform(self.robot_model.lower_bounds, self.robot_model.upper_bounds)
        goal_config = self.robot_model.ik_with_custom_solver(self.goal_pose, solver = custom_solver)

        if goal_config is None:
            rospy.logerr("No valid goal configuration found.")
            return [], False

        root = TreeNode(self.start_config)
        nodes: t.List[TreeNode] = [root]
        start_time = time.time()

        for i in range(self.max_iterations):
            if time.time() - start_time > self.max_time:
                break

            # Biased sampling towards goal
            if random.random() < self.goal_bias:
                sample = goal_config
            else:
                sample = self.robot_model.sample_random_configuration()

            # Find nearest node
            nearest = min(nodes, key=lambda node: np.linalg.norm(node.config - sample))

            # Steer towards sample
            direction = sample - nearest.config
            distance = np.linalg.norm(direction)
            if distance == 0:
                continue
            direction = direction / distance
            new_config = nearest.config + self.step_size * direction

            # Check joint limits and collision
            if not self.robot_model.is_within_limits(new_config):
                continue
            if not self.collision_checker.is_collision_free(start_config=nearest.config, goal_config=new_config):
                continue

            # Add to tree
            new_node = TreeNode(new_config, nearest)
            nodes.append(new_node)

            # Check goal reached
            if np.linalg.norm(new_config - goal_config) < self.step_size:
                goal_node = TreeNode(goal_config, new_node)
                nodes.append(goal_node)
                return self._reconstruct_path(goal_node), True

        return [], False

    def _reconstruct_path(self, node: TreeNode) -> t.List[np.ndarray]:
        """
        Reconstruct path by backtracking from goal to root.

        Args:
            node: The goal node.

        Return:
            List of configurations from start to goal.
        """
        path = []
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]
