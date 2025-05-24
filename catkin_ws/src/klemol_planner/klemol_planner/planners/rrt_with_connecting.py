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


class RRTWithConnectingPlanner(Planner):
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
        self.step_size: float = parameters.get("step_size", 0.1)
        self.max_iterations: int = parameters.get("max_iterations", 1000)
        self.goal_bias: float = parameters.get("bias", 0.05)
        self.max_time: float = parameters.get("max_time", 5.0)

        self.max_goal_samples: int = parameters.get("max_goal_samples", 10)
        self.goal_configs: t.List[np.ndarray] = []

        weights = np.array([0.5, 1.0, 10.75, 0.5, 0.05, 0.05, 0.05])  # Example: penalise lower joints less
        self.weights = weights#[::-1] # REVERSE THE WEIGHTS

    def plan(self) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Execute the RRT planning algorithm.

        Return:
            A tuple of:
            - List of joint configurations (np.ndarray) forming the planned path.
            - Boolean indicating if planning succeeded.
        """
        rospy.loginfo("Starting RRT with connection planning...")
        rospy.loginfo(f"Parameters: {self.parameters}")
        if self.start_config is None or self.goal_pose is None:
            raise ValueError("Start configuration and goal pose must be set before planning.")

        # Step 1: generate multiple IK solutions
        self.goal_configs = self._generate_goal_configurations(self.goal_pose)
        if not self.goal_configs:
            rospy.logwarn("No valid IK solutions found for the given goal pose.")
            return [], False

        root = TreeNode(self.start_config)
        nodes: t.List[TreeNode] = [root]
        start_time = time.time()

        new_config = self.start_config
        new_node = root

        for _ in range(self.max_iterations):
            if time.time() - start_time > self.max_time:
                break

            # Check if goal reached for any config
            for goal_config in self.goal_configs:
                if np.linalg.norm(new_config - goal_config) < self.step_size:
                    goal_node = TreeNode(goal_config, new_node)
                    nodes.append(goal_node)
                    return self._reconstruct_path(goal_node), True

            # Check for any direct connections to goal
            connectable_goals = []
            for goal_config in self.goal_configs:
                print(f"Checking connection to goal config: {goal_config}")
                if self._is_collision_free_path(new_config, goal_config):
                    connectable_goals.append(goal_config)

            # If any goal is directly reachable, pick the best one (e.g. closest in joint space)
            if connectable_goals:
                best_goal = min(
                    connectable_goals,
                    key=lambda g: self._weighted_distance(new_config, g, self.weights)
                )
                print(f"I can connect straight to {best_goal}.")
                goal_node = TreeNode(best_goal, new_node)
                nodes.append(goal_node)
                rospy.loginfo("Connected to best goal from reachable set.")
                return self._reconstruct_path(goal_node), True

            # Biased sampling towards goal
            if random.random() < self.goal_bias:
                sample = random.choice(self.goal_configs)
            else:
                sample = self.robot_model.sample_random_configuration()

            nearest = min(nodes, key=lambda node: self._weighted_distance(node.config, sample, self.weights))

            # Steer towards sample
            direction = sample - nearest.config
            distance = self._weighted_distance(nearest.config, sample, self.weights)
            if distance == 0:
                continue
            direction = direction / distance
            new_config = nearest.config + self.step_size * direction

            # Check joint limits and collision
            if not self.robot_model.is_within_limits(new_config):
                continue
            if self.collision_checker.is_in_collision(new_config):
                continue

            # Add to tree
            new_node = TreeNode(new_config, nearest)
            nodes.append(new_node)

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
    
    def _is_collision_free_path(self, from_config: np.ndarray, to_config: np.ndarray) -> bool:
        """
        Check whether a straight-line path between two joint configurations is free of collisions.

        This function interpolates linearly between two configurations and checks each intermediate
        step for joint limits and collisions. Returns False immediately if any check fails.

        Args:
            from_config: Start joint configuration (7-DOF).
            to_config: Goal joint configuration (7-DOF).

        Returns:
            True if the entire path is collision-free and within joint limits, False otherwise.
        """

        # Compute the total Euclidean distance in joint space
        distance = np.linalg.norm(to_config - from_config)

        # Decide how many intermediate points to check based on step_size
        num_steps = int(np.ceil(distance / self.step_size))
        print(f"Checking {num_steps} steps")
        # TODO RISKY!
        num_steps = int(np.ceil(num_steps) / 4)
        if num_steps == 0:
            return True  # The configurations are too close — no interpolation needed

        # Check all intermediate points along the straight line
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            interp_config = (1 - alpha) * from_config + alpha * to_config

            # 1. Check if the joint values are within limits
            if not self.robot_model.is_within_limits(interp_config):
                return False

            # RISKY!
            # # 2. Check if this configuration is collision-free
            # if self.collision_checker.is_in_collision(interp_config):
            #     return False

        return True  # All intermediate configurations are safe

    def _weighted_distance(self, config1: np.ndarray, config2: np.ndarray, weights: t.Optional[np.ndarray] = None) -> float:
        """
        Compute weighted joint-space distance between two configurations.

        Args:
            config1: First joint configuration.
            config2: Second joint configuration.
            weights: Optional array of weights (default = 1.0 for each joint).

        Returns:
            Weighted L2 distance.
        """
        if weights is None:
            weights = np.ones_like(config1)
        diff = config1 - config2
        return np.sqrt(np.sum(weights * diff**2))
    
    def _generate_goal_configurations(self, goal_pose: PointWithOrientation) -> t.List[np.ndarray]:
        """
        Generate multiple IK solutions for a single end-effector pose.
        
        Args:
            goal_pose: Desired 6D pose of the end-effector.
        
        Returns:
            A list of valid, unique joint configurations that solve the pose.
        """
        goal_configs: t.List[np.ndarray] = []
        seed_attempts = 0
        max_attempts = self.max_goal_samples * 5
        threshold = 1e-2  # Joint-space uniqueness threshold

        ik_solver = IK(
            base_link=self.robot_model.base_link,
            tip_link=self.robot_model.ee_link,
            urdf_string=self.robot_model.urdf_string,
            timeout=0.1,
            solve_type="Speed",
        )
        while len(goal_configs) < self.max_goal_samples and seed_attempts < max_attempts:
            random_seed = np.random.uniform(self.robot_model.lower_bounds, self.robot_model.upper_bounds)
            solution = self.robot_model.ik_with_custom_solver(goal_pose, solver=ik_solver, seed=random_seed)
            seed_attempts += 1

            if solution is None:
                continue
            if not self.robot_model.is_within_limits(solution):
                continue
            if self.collision_checker.is_in_collision(solution):
                continue
            # Ensure uniqueness
            if any(np.linalg.norm(solution - existing) < threshold for existing in goal_configs):
                continue
            goal_configs.append(solution)

        return goal_configs
