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

class RRTStarPlanner(Planner):
    """
    Rapidly-exploring Random Tree Star (RRT*) planner implementation.

    This planner improves paths over time by rewiring the tree based on cost.
    """

    def __init__(self,
                 robot_model: Robot,
                 collision_checker: CollisionChecker,
                 parameters: dict):
        """
        Initialize RRT* planner.
        """
        super().__init__(robot_model, collision_checker, parameters)
        self.step_size: float = parameters.get("step_size", 0.1)
        self.max_iterations: int = parameters.get("max_iterations", 1000)
        self.goal_bias: float = parameters.get("bias", 0.05)
        self.max_time: float = parameters.get("max_time", 5.0)
        self.rewire_radius: float = parameters.get("rewire_radius", 0.5)

    def plan(self) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Execute the RRT* planning algorithm.

        Return:
            A tuple of:
            - List of joint configurations (np.ndarray) forming the planned path.
            - Boolean indicating if planning succeeded.
        """
        rospy.loginfo("Starting RRT* planning...")
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
        root.cost = 0.0
        nodes: t.List[TreeNode] = [root]
        start_time = time.time()

        for i in range(self.max_iterations):
            if time.time() - start_time > self.max_time:
                break

            # Sample random configuration
            if random.random() < self.goal_bias:
                sample = goal_config
            else:
                sample = self.robot_model.sample_random_configuration()

            nearest = min(nodes, key=lambda node: np.linalg.norm(node.config - sample))

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

            # Create new node
            new_node = TreeNode(new_config)
            new_node.cost = nearest.cost + np.linalg.norm(new_node.config - nearest.config)
            new_node.parent = nearest

            # Find nearby nodes within rewire radius
            near_nodes = self._find_near_nodes(nodes, new_node)

            # Choose best parent
            for near_node in near_nodes:
                if not self.collision_checker.is_collision_free(start_config = near_node.config, goal_config=new_node.config):
                    continue
                cost_through_near = near_node.cost + np.linalg.norm(near_node.config - new_node.config)
                if cost_through_near < new_node.cost:
                    new_node.parent = near_node
                    new_node.cost = cost_through_near

            nodes.append(new_node)

            # Rewire near nodes if beneficial
            for near_node in near_nodes:
                if not self.collision_checker.is_collision_free(start_config = new_node.config, goal_config=near_node.config):
                    continue
                cost_through_new = new_node.cost + np.linalg.norm(new_node.config - near_node.config)
                if cost_through_new < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = cost_through_new

            # Goal checking
            if np.linalg.norm(new_config - goal_config) < self.step_size:
                goal_node = TreeNode(goal_config)
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + np.linalg.norm(new_node.config - goal_config)
                nodes.append(goal_node)
                return self._reconstruct_path(goal_node), True

        return [], False

    def _find_near_nodes(self, nodes: t.List[TreeNode], new_node: TreeNode) -> t.List[TreeNode]:
        """
        Find all nodes within the rewire radius.
        """
        near_nodes = []
        for node in nodes:
            if np.linalg.norm(node.config - new_node.config) <= self.rewire_radius:
                near_nodes.append(node)
        return near_nodes

    def _reconstruct_path(self, node: TreeNode) -> t.List[np.ndarray]:
        """
        Reconstruct path by backtracking from goal to root.
        """
        path = []
        while node is not None:
            path.append(node.config)
            node = node.parent
        return path[::-1]
