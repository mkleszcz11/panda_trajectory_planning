import typing as t
import numpy as np
import rospy
import moveit_commander
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState




class CollisionChecker:
    """
    Collision checker using MoveIt for self-collision and environment collision checking.

    This class provides an interface to check whether a given joint configuration results
    in a collision using the current MoveIt planning scene.
    """

    def __init__(self, group_name: str = "panda_arm"):
        """
        Initialize the collision checker.

        Args:
            group_name: Name of the MoveIt planning group.
        """
        moveit_commander.roscpp_initialize([])
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_planning_time(0.5)

        # Get joint names once
        self.joint_names = self.group.get_active_joints()

        # Prepare collision checking service
        rospy.wait_for_service("/check_state_validity")
        self.state_validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)

    def is_collision_free(self, start_config: np.ndarray, goal_config: np.ndarray) -> bool:
        """
        Check if the movement from start config to joint config is collision-free.
        """
        return (not self.is_joint_config_in_collision(goal_config) and self.is_path_valid(start_config, goal_config))
    
    def is_joint_config_in_collision(self, joint_config: np.ndarray) -> bool:
        """
        Check if the given joint configuration is in collision.

        Args:
            joint_config: Joint angles as a NumPy array.

        Returns:
            True if the configuration results in a collision, False otherwise.
        """
        joint_state = JointState()
        joint_state.name = self.joint_names
        joint_state.position = joint_config.tolist()

        robot_state = RobotState()
        robot_state.joint_state = joint_state

        req = GetStateValidityRequest()
        req.robot_state = robot_state
        req.group_name = self.group.get_name()

        res = self.state_validity(req)
        return not res.valid

    def is_path_valid(self, from_q: np.ndarray, to_q: np.ndarray) -> bool:
        """
        Check if the path from `from_q` to `to_q` is in collision.
        Args:
            from_q: Starting joint configuration as a NumPy array.
            to_q: Ending joint configuration as a NumPy array.
        
        Returns:
            True if the path is in collision, False otherwise.
        """
        dist = np.linalg.norm(to_q - from_q)
        # if dist > 0.3:
        #     print(f"Path too long: {dist:.2f} m, skipping collision check")
        #     return False

        steps = max(2, int(np.ceil(dist / 0.02)))
        for i in range(1, steps):
            alpha = i / steps
            interp = (1 - alpha) * from_q + alpha * to_q

            # Check collision
            if self.is_joint_config_in_collision(interp):
                return False
        return True

# class CollisionChecker:
#     """
#     Collision checker using MoveIt for self-collision and environment collision checking.

#     This class provides an interface to check whether a given joint configuration results
#     in a collision using the current MoveIt planning scene.
#     """

#     def __init__(self, group_name: str = "panda_arm"):
#         """
#         Initialize the collision checker.

#         Args:
#             group_name: Name of the MoveIt planning group.
#         """
#         moveit_commander.roscpp_initialize([])
#         self.scene = moveit_commander.PlanningSceneInterface()
#         self.group = moveit_commander.MoveGroupCommander(group_name)
#         self.group.set_planning_time(0.5)

#     def is_in_collision(self, joint_config: np.ndarray) -> bool:
#         """
#         Check if the given joint configuration is in collision.

#         Args:
#             joint_config: Joint angles as a NumPy array.

#         Returns:
#             True if the configuration results in a collision, False otherwise.
#         """

#         # # Set the joint target
#         # self.group.set_joint_value_target(joint_config.tolist())

#         # # Plan to the joint state (we do not execute)
#         # success, plan, _, _ = self.group.plan() 

#         # # If the plan result is empty or incomplete, assume collision or failure
#         # if not success or not plan.joint_trajectory.points:
#         #     return True

#         # return False
        
