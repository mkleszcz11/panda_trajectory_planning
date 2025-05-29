import rospy
import numpy as np
from klemol_planner.post_processing.path_post_processing import PathPostProcessing
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from control_msgs.msg import JointTrajectoryControllerState
from klemol_planner.tests.main_test_logger import MainTestLogger
from klemol_planner.utils.config_loader import load_planner_params
from klemol_planner.environment.robot_model import Robot

import math

from trac_ik_python.trac_ik import IK


class PlannersTestExecutor:
    def __init__(self,
                 robot_model: Robot,
                 collision_checker: CollisionChecker,
                 post_processing: PathPostProcessing,
                 logger: MainTestLogger):
        self.robot_model = robot_model
        self.collision_checker = collision_checker
        self.logger = logger
        self.post_processing = post_processing

        # TODO - change accordingly !!!
        self.logger_sub = rospy.Subscriber(
            # "/position_joint_trajectory_controller/state",
            "/effort_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.logger.callback,
        )

    def run_test(self,
                 planner_type: str,
                 start_joint_config: np.ndarray,
                 goal_end_effector_pose: PointWithOrientation,
                 velocity_limits: np.ndarray = np.ones(7)):
        rospy.loginfo(f"Running test for: {planner_type}")
        # TODO - integrate with logger

        ### Test setup
        self.logger.reset()
        self.logger.set_planner(planner_type)
        planner = self._return_new_planner(planner_type)

        ### Move robot to start configuration
        self.robot_model.move_to_joint_config(start_joint_config)
        rospy.sleep(1)  # Allow the robot to stabilize


        ### Set up the planner and execute the path
        # planner.set_start(start_joint_config)
        # planner.set_goal(goal_end_effector_pose)
        self.logger.start_timer()
        self.robot_model.move_with_trajectory_planner(planner = planner,
                                                      post_processing=self.post_processing,
                                                      goal = goal_end_effector_pose)

        rospy.sleep(1.0)  # Let the robot stabilize after each move

    def _return_new_planner(self, planner_type: str):
        """
        Returns a new planner instance based on the planner type.
        """
        params = load_planner_params(planner_name=planner_type)

        if planner_type == "rrt":
            from klemol_planner.planners.rrt import RRTPlanner
            print(f"RRT planner initialized with parameters: {params}")
            return RRTPlanner(robot_model=self.robot_model,
                              collision_checker=self.collision_checker,
                              parameters=params)

        elif planner_type == "rrt_with_connecting":
            from klemol_planner.planners.rrt_with_connecting import RRTWithConnectingPlanner
            print(f"RRT with connecting planner initialized with parameters: {params}")
            return RRTWithConnectingPlanner(robot_model=self.robot_model,
                                            collision_checker=self.collision_checker,
                                            parameters=params)

        elif planner_type == "rrt_star":
            from klemol_planner.planners.rrt_star import RRTStarPlanner
            print(f"RRT* planner initialized with parameters: {params}")
            return RRTStarPlanner(robot_model=self.robot_model,
                                  collision_checker=self.collision_checker,
                                  parameters=params)

        else:
            raise ValueError(f"Unknown planner type: {planner_type}")
