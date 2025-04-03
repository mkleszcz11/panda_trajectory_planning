import typing as t
import numpy as np
import rospy
import moveit_commander

from klemol_planner.environment.robot_model import RobotModel

class CollisionChecker:
    """
    Collision checker using MoveIt for self-collision and environment collision checking.

    This class provides an interface to check whether a given joint configuration results
    in a collision using the current MoveIt planning scene.
    """

    def __init__(self, robot_model: RobotModel, group_name: str = "panda_arm"):
        """
        Initialize the collision checker.

        Args:
            group_name: Name of the MoveIt planning group.
        """
        moveit_commander.roscpp_initialize([])
        self.robot_model = robot_model
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_planning_time(0.5)

    def is_in_collision(self, joint_config: np.ndarray) -> bool:
        """
        Check if the given joint configuration is in collision.

        Args:
            joint_config: Joint angles as a NumPy array.

        Returns:
            True if the configuration results in a collision, False otherwise.
        """
        # Set the joint target
        self.group.set_joint_value_target(joint_config.tolist())

        # Plan to the joint state (we do not execute)
        success, plan, _, _ = self.group.plan() 

        # If the plan result is empty or incomplete, assume collision or failure
        if not success or not plan.joint_trajectory.points:
            return True

        return False
