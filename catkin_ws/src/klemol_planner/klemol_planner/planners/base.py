"""
Base interface for all motion planners.
Defines the common API for sampling-based planners (plan(), set_start(), set_goal(), etc.).
"""

import typing as t
import numpy as np
from klemol_planner.environment.robot_model import RobotModel
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation

class Planner:
    def __init__(self,
                 robot_model: RobotModel,
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
