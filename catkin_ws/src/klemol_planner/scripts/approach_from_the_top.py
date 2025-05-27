#!/usr/bin/env python3

########################################################
# Simulate that we see the object in the camera.
# Then we move the robot to the object.
# Goal of this code is to validate that transformations
# are working correctly.
########################################################

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf_trans
import numpy as np
import csv
import time
from trac_ik_python.trac_ik import IK
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.post_processing.path_post_processing import PathPostProcessing
import math
import subprocess
import typing as t

from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.planners.rrt import RRTPlanner
from klemol_planner.planners.rrt_star import RRTStarPlanner
from klemol_planner.planners.rrt_with_connecting import RRTWithConnectingPlanner
from klemol_planner.utils.config_loader import load_planner_params
from klemol_planner.tests.trajectory_logger import TrajectoryLogger
from control_msgs.msg import JointTrajectoryControllerState

import copy
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

from klemol_planner.camera_utils.camera_operations import CameraOperations

class FrankaMotionController:
    def __init__(self):
        rospy.init_node("franka_motion_controller")
        self.logger = TrajectoryLogger()

        ##################################
        ## MOVEIT STUFF - TO BE REMOVED ##
        ##################################
        moveit_commander.roscpp_initialize([])

        # RobotModel initialization
        self.robot_model = Robot()

        # Initialize robot model and collision checker
        self.collision_checker = CollisionChecker(self.robot_model, group_name="panda_arm")
        self.post_processing = PathPostProcessing(collision_checker=self.collision_checker)


        # Load RRT-specific parameters from config
        algorithm = "rrt_with_connecting" #"rrt" / "rrt_star" / "rrt_with_connecting" / "prm"
        algorithm_params = load_planner_params(algorithm)
        # self.custom_planner = RRTPlanner(self.robot_model, self.collision_checker, algorithm_params)
        # self.custom_planner = RRTStarPlanner(self.robot_model, self.collision_checker, algorithm_params)
        self.custom_planner = RRTWithConnectingPlanner(self.robot_model, self.collision_checker, algorithm_params)

        camera_operations = CameraOperations()
        panda_transformations = PandaTransformations(cam_operations=camera_operations)


        ###############################################
        ######### Camera calibration fo sim ###########
        ###############################################
        panda_transformations.T_base_to_camera = np.array([[0, 1,  0, 0.35],
                                                           [1, 0,  0, 0.0],
                                                           [0, 0, -1, 1.4],
                                                           [0, 0,  0, 1]])
        panda_transformations.table_corners_translations = {
            "corner_0": np.array([0.7, -0.4, 0.06]),# - self.z_calibration_constant]),
            "corner_1": np.array([0.7,  0.4, 0.06]),# - self.z_calibration_constant]),
            "corner_2": np.array([0.1,  0.4, 0.06]),# - self.z_calibration_constant]),
            "corner_3": np.array([0.1, -0.4, 0.06])# - self.z_calibration_constant])
        }
        panda_transformations.calibrate_corners_relative_to_base()

        #####################################
        # WE WILL BE MOVING TO THESE POINTS #
        #####################################
        # Define fixed joint configuration for consistent execution
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Joint angles in radians

        point_1 = PointWithOrientation(0.0, 0.0, 0.9, 0.0, 0.0, math.pi * 0.75)

        object_in_camera_frame = PointWithOrientation(0.3, 0.1, 1.2, 0.0, 0.05, math.pi * 0.75)
        self.object_in_base_frame = panda_transformations.transform_point(object_in_camera_frame, 'camera', 'base')

        self.point_above_object_in_base_frame = PointWithOrientation(
            self.object_in_base_frame.x,
            self.object_in_base_frame.y,
            self.object_in_base_frame.z + 0.12,
            self.object_in_base_frame.roll,
            self.object_in_base_frame.pitch,
            self.object_in_base_frame.yaw
        )

        # Prepare a dictionary for visualization
        visualisation_frames = {}

        # Optional: add any extra objects (e.g. a detected tennis ball)
        visualisation_frames["object"] = self.object_in_base_frame.as_matrix()
        visualisation_frames["above_object"] = self.point_above_object_in_base_frame.as_matrix()

        # Visualise
        # panda_transformations.visusalise_environment(visualisation_frames)

    def execute(self):
        """Main execution sequence"""
        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.robot_model.move_to_joint_config(self.start_joint_config)
        self.logger_sub = rospy.Subscriber(
            "/position_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.logger.callback,
        )

        ### Open gripper
        self.robot_model.open_gripper()

        self.logger.set_mode("Picking banana quintic | RRT with shortcutting")

        post_goal_path = [self.object_in_base_frame]

        self.robot_model.move_with_trajectory_planner(planner = self.custom_planner,
                                                      post_processing = self.post_processing,
                                                      goal = self.point_above_object_in_base_frame,
                                                      post_goal_path = post_goal_path)

        ### Close gripper
        self.robot_model.close_gripper()


        ### Move back up

        self.robot_model.move_with_trajectory_planner(planner = self.custom_planner,
                                                      post_processing = self.post_processing,
                                                      goal = self.point_above_object_in_base_frame,
                                                      post_goal_path = post_goal_path)

        # self.robot_model.move_to_pose_planner(self.point_above_object_in_base_frame)

        ### Move above box 1
        # self.move_with_trajectory_planner(self.point_above_box1_in_base_frame) # self.robot_model.move_to_pose_planner(self.point_above_box1_in_base_frame)

        # ### Open gripper
        # self.robot_model.move_gripper(True)



        self.logger.save("/tmp/pick_banana_demo.npz")


if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(0.5)  # Allow ROS to initialize
    controller.execute()
