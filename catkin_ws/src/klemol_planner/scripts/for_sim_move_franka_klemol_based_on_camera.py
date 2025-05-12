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

from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.planners.rrt import RRTPlanner
from klemol_planner.planners.rrt_star import RRTStarPlanner
from klemol_planner.planners.rrt_with_connecting import RRTWithConnectingPlanner
from klemol_planner.planners.prm import PRMPlanner
from klemol_planner.utils.config_loader import load_planner_params

import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

from klemol_planner.camera_utils.camera_operations import CameraOperations

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory

class FrankaMotionController:
    def __init__(self):
        rospy.init_node("franka_motion_controller")

        ##################################
        ## MOVEIT STUFF - TO BE REMOVED ##
        ##################################
        moveit_commander.roscpp_initialize([])

        # RobotModel initialization
        self.robot_model = Robot()

        # Initialize robot model and collision checker
        self.collision_checker = CollisionChecker(self.robot_model, group_name="panda_arm")

        # Load RRT-specific parameters from config
        algorithm = "rrt"  # or "rrt_star", "rrt_with_connecting", "prm"
        algorithm_params = load_planner_params(algorithm)
        self.custom_planner = RRTPlanner(self.robot_model, self.collision_checker, algorithm_params)

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

        object_in_camera_frame = PointWithOrientation(0.15, 0.15, 1.2, 0.0, 0.05, math.pi * 0.75)
        object_in_base_frame = panda_transformations.transform_point(object_in_camera_frame, 'camera', 'base')

        point_above_object_in_base_frame = PointWithOrientation(
            object_in_base_frame.x,
            object_in_base_frame.y,
            object_in_base_frame.z + 0.12,
            object_in_base_frame.roll,
            object_in_base_frame.pitch,
            object_in_base_frame.yaw
        )

        # Prepare a dictionary for visualization
        visualisation_frames = {}

        # Optional: add any extra objects (e.g. a detected tennis ball)
        visualisation_frames["tennis"] = object_in_base_frame.as_matrix()

        # Visualise
        panda_transformations.visusalise_environment(visualisation_frames)

        self.target_positions = [
            point_above_object_in_base_frame,
            object_in_base_frame,
            point_above_object_in_base_frame,
            panda_transformations.transform_point(point_1, 'camera', 'base')
        ]

    def execute(self):
        """Main execution sequence"""

        ####################################
        #### CUSTOM TRAJECTORY PLANNER #####
        ####################################
        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.robot_model.move_to_joint_config(self.start_joint_config)

        # Open gripper
        self.robot_model.move_gripper(True)

        rospy.loginfo("Executing predefined movements using custom Trajectory Planner")
        for i,pos in enumerate(self.target_positions):
            if i == 2:
                self.robot_model.move_gripper(False)
                rospy.sleep(1)

            # THIS IS CUSTOM PLANNER
            current_config = np.array(self.robot_model.group.get_current_joint_values())
            self.custom_planner.set_start(current_config)
            self.custom_planner.set_goal(pos)
            path, success = self.custom_planner.plan()

            # # Call shortcutting function (edit path)
            path_post_processing = PathPostProcessing(self.collision_checker)
            path = path_post_processing.generate_a_shortcutted_path(path)

            if success:
                rospy.loginfo(f"RRT path found with {len(path)} waypoints.")

                # Smooth the path and execute smooth trajectory
                trajectory = path_post_processing.interpolate_trajectory_time_parameterised(path, joint_names=self.robot_model.group.get_active_joints())
                self.robot_model.send_trajectory_to_controller(trajectory)

                # for config in path:
                #     self.robot_model.execute_joint_positions(config, "Custom RRT")
            else:
                rospy.logwarn("RRT planner failed to find a path.")

if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(1)  # Allow ROS to initialize
    controller.execute()
