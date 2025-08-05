"""
Base interface for all motion planners.
Defines the common API for sampling-based planners (plan(), set_start(), set_goal(), etc.).
"""

import typing as t
import numpy as np
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation

from trac_ik_python.trac_ik import IK

class Planner:
    def __init__(self,
                 robot_model: Robot,
                 collision_checker: CollisionChecker,
                 parameters: dict):
        """
        Initialize the planner with robot model, collision checker, and parameters.

        Args:
            robot_model: Object providing joint limits and FK/IK.
            collision_checker: Object to evaluate collisions during planning.
            parameters: Dictionary containing planner-specific parameters.
        """
        self.robot_model = robot_model
        self.collision_checker = collision_checker
        self.parameters = parameters
        self.start_config: t.Optional[np.ndarray] = None
        self.goal_pose: t.Optional[PointWithOrientation] = None

    def set_start(self, start_config: np.ndarray) -> None:
        """
        Define the starting configuration of the robot.

        Args:
            start_config: Joint angles representing the start configuration.
        """
        self.start_config = start_config

    def set_goal(self, goal_pose: PointWithOrientation) -> None:
        """
        Define the Cartesian goal pose for the end-effector.

        Args:
            goal_pose: Desired pose (position + orientation) of the end-effector.
        """
        self.goal_pose = goal_pose

    def plan(self) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Execute the planner to compute a trajectory.

        Return:
            A tuple containing:
              - List of joint configurations representing the path.
              - Boolean flag indicating success.

        Note:
            Example (single) joint configurations:
            np.array([ 0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.5 ])
        """
        raise NotImplementedError("Each planner must implement the plan() method.")
    

    def generate_goal_configurations(self, goal_pose: PointWithOrientation) -> t.List[np.ndarray]:
        """
        Generate multiple IK solutions for a single end-effector pose.
        
        Args:
            goal_pose: Desired 6D pose of the end-effector.
        
        Returns:
            A list of valid, unique joint configurations that solve the pose.
        """
        goal_configs: t.List[np.ndarray] = []
        seed_attempts = 0
        max_attempts = 20 * self.max_goal_samples
        threshold = 1e-2  # Joint-space uniqueness threshold

        ik_solver = IK(
            base_link=self.robot_model.base_link,
            tip_link=self.robot_model.ee_link,
            urdf_string=self.robot_model.urdf_string,
            timeout=0.1,
            solve_type="Manipulation1",
        )
        while len(goal_configs) < self.max_goal_samples and seed_attempts < max_attempts:
            random_seed = np.random.uniform(self.robot_model.lower_bounds, self.robot_model.upper_bounds)
            solution = self.robot_model.ik_with_custom_solver(goal_pose, solver=ik_solver, seed=random_seed)
            seed_attempts += 1

            if solution is None:
                continue
            if not self.robot_model.is_within_limits(solution):
                continue
            if self.collision_checker.is_joint_config_in_collision(solution):
                continue
            # Ensure uniqueness
            if any(np.linalg.norm(solution - existing) < threshold for existing in goal_configs):
                continue
            goal_configs.append(solution)

        return goal_configs

