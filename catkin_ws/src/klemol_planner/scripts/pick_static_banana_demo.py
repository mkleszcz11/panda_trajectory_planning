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

from copy import deepcopy

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

        # REBASE
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785 - math.pi / 2.0]  # Joint angles in radians
        self.robot_model.open_gripper()
        self.robot_model.move_to_joint_config(self.start_joint_config)

        camera_operations = CameraOperations()
        panda_transformations = PandaTransformations(cam_operations=camera_operations)
        panda_transformations.calibrate_camera()

        #####################################
        # WE WILL BE MOVING TO THESE POINTS #
        #####################################
        # Define fixed joint configuration for consistent execution
        point_1 = PointWithOrientation(0.0, 0.0, 0.9, 0.0, 0.0, math.pi * 0.75)

        print("TRYING TO FIND A CUSTOM OBJECT")
        print(F"DEBUG DEBUG DEBUG ----->>>>> {camera_operations.USE_REALSENSE}")
        if camera_operations.USE_REALSENSE:
            success, x, y, z, yaw = camera_operations.find_grasp()
            yaw = yaw % math.pi

        else:
            success = True
            x, y, z, yaw = 0.01, 0.01, 1.0, 0.0
        if success:
            print(f"YAW YAW YAW YAW YAW YAW YAW ======>>>>> {yaw}")
            object_in_camera_frame = PointWithOrientation(x, y, z, 0.0, 0.0, (-math.pi * 0.25) + yaw)
            self.object_in_base_frame = panda_transformations.transform_point(object_in_camera_frame, 'camera', 'base')

            print(f"X = {x} | Y = {y} | Z = {z}")
        else:
            object_in_camera_frame = point_1
            print("NO OBJECT DETECTED")
        print("OBJECT DETECTION DONE")

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

        ###################################
        ###################################
        ###################################

        # Get all marker transforms in camera frame
        marker_transforms = camera_operations.get_marker_transforms()

        # Prepare a dictionary for visualization
        visualisation_frames = {}

        # Iterate over all detected corners
        for corner_name in ["corner_0", "corner_1", "corner_2", "corner_3"]:
            if corner_name not in marker_transforms:
                print(f"[WARN] {corner_name} not detected.")
                continue

            # Extract translation
            x, y, z = marker_transforms[corner_name][:3, 3]

            # Construct a point in the camera frame
            corner_cam = PointWithOrientation(x, y, z, 0.0, 0.0, 0.0)

            # Transform to base frame
            corner_base = panda_transformations.transform_point(corner_cam, 'camera', 'base')

            # Store for visualization
            visualisation_frames[f"{corner_name}_in_camera_frame"] = corner_base.as_matrix()

        # Process box markers (10 and 11)
        # Default to point_1 if not detected
        point_box_1 = panda_transformations.transform_point(point_1, 'camera', 'base')
        point_box_2 = panda_transformations.transform_point(point_1, 'camera', 'base')

        # Check for marker 10 (box_1)
        if "corner_10" in marker_transforms:
            x, y, z = marker_transforms["corner_10"][:3, 3]
            box_cam = PointWithOrientation(x, y, z - 0.12, 0.0, 0.0, 0.0)
            point_box_1 = panda_transformations.transform_point(box_cam, 'camera', 'base')
            self.point_above_box1_in_base_frame = PointWithOrientation(
                point_box_1.x,
                point_box_1.y,
                point_box_1.z + 0.1,
                point_box_1.roll,
                point_box_1.pitch,
                point_box_1.yaw
            )
            visualisation_frames["box_1_in_camera_frame"] = point_box_1.as_matrix()
            print("Box 1 Found")
        else:
            print("[WARN] marker_10 (box_1) not detected. Using point_1 instead.")

        # Check for marker 11 (box_2)
        if "corner_11" in marker_transforms:
            x, y, z = marker_transforms["corner_11"][:3, 3]
            box_cam = PointWithOrientation(x, y, z - 0.12, 0.0, 0.0, 0.0)
            point_box_2 = panda_transformations.transform_point(box_cam, 'camera', 'base')
            visualisation_frames["box_2_in_camera_frame"] = point_box_2.as_matrix()
            print("Box 2 Found")
        else:
            print("[WARN] marker_11 (box_2) not detected. Using point_1 instead.")

        # Add tennis ball to visualization
        visualisation_frames["banana"] = self.object_in_base_frame.as_matrix()

        # Visualise
        # panda_transformations.visusalise_environment(visualisation_frames)

    # def move_with_trajectory_planner(self, goal):
    #     current_config = np.array(self.robot_model.group.get_current_joint_values())
    #     self.custom_planner.set_start(current_config)
    #     self.custom_planner.set_goal(goal)
    #     path, success = self.custom_planner.plan()

    #     #### Call shortcutting function (edit path)
    #     path_post_processing = PathPostProcessing(self.collision_checker)
    #     # path = path_post_processing.generate_a_shortcutted_path(path)

    #     ### 
    #     if success:
    #         rospy.loginfo(f"Planner found path with {len(path)} waypoints.")
    #         rospy.loginfo(f"Fitting spline to the path...")
    #         # Smooth the path and execute smooth trajectory
    #         trajectory = path_post_processing.interpolate_quintic_trajectory(
    #             path=path,
    #             joint_names=self.robot_model.group.get_active_joints(),
    #             velocity_limits=self.robot_model.velocity_limits,
    #             acceleration_limits=self.robot_model.acceleration_limits,
    #             max_vel_acc_multiplier = 0.6
    #             )
    #         self.robot_model.send_trajectory_to_controller(trajectory)
    #     else:
    #         rospy.logwarn("RRT planner failed to find a path.")

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

        # #### Move above the object
        # self.move_with_trajectory_planner(self.point_above_object_in_base_frame)

        ### Move to the object, last part straightish
        post_goal_path = [self.object_in_base_frame]
        self.robot_model.move_with_trajectory_planner(planner = self.custom_planner,
                                                      post_processing = self.post_processing,
                                                      goal = self.point_above_object_in_base_frame,
                                                      post_goal_path = post_goal_path)

        ### Close gripper
        self.robot_model.close_gripper()

        ### Build waypoints before moving above the box
        waypoint_1 = deepcopy(self.object_in_base_frame)
        waypoint_1.z += 0.05 

        waypoint_2 = deepcopy(self.object_in_base_frame)

        # Find direction
        dx = self.point_above_box1_in_base_frame.x - self.object_in_base_frame.x
        dy = self.point_above_box1_in_base_frame.y - self.object_in_base_frame.y
        distance = math.hypot(dx, dy)  # sqrt(dx^2 + dy^2)
        dx_unit = dx / distance
        dy_unit = dy / distance
        step_size = 0.05
        new_x = self.object_in_base_frame.x + dx_unit * step_size
        new_y = self.object_in_base_frame.y + dy_unit * step_size

        waypoint_2.x += new_x
        waypoint_2.y += new_y
        waypoint_2.z = waypoint_1.z + step_size

        pre_start_path = [waypoint_1, waypoint_2]

        self.robot_model.move_with_trajectory_planner(planner = self.custom_planner,
                                                      post_processing = self.post_processing,
                                                      goal = self.point_above_box1_in_base_frame,
                                                      pre_start_path = pre_start_path)

        ### Open gripper
        self.robot_model.open_gripper

        self.logger.save("/tmp/pick_banana_demo.npz")


if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(0.5)  # Allow ROS to initialize
    controller.execute()
