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
from klemol_planner.utils.config_loader import load_planner_params

import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

from klemol_planner.camera_utils.camera_operations import CameraOperations

class FrankaMotionController:
    def __init__(self):
        rospy.init_node("franka_motion_controller")

        ##################################
        ## MOVEIT STUFF - TO BE REMOVED ##
        ##################################
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")

        # TRAC-IK Solver
        self.ik_solver = IK("panda_link0", "panda_link8")
        self.lower_bounds, self.upper_bounds = self.ik_solver.get_joint_limits()

        # Load config paths
        pkg_root = rospy.get_param("/klemol_planner/package_path", default="/home/neurorobotic_student/panda_trajectory_planning/catkin_ws/src/klemol_planner")
        xacro_path = f"{pkg_root}/panda_description/panda.urdf.xacro"
        urdf_string = subprocess.check_output(["xacro", xacro_path]).decode("utf-8")
        joint_limits_path = f"{pkg_root}/config/joint_limits.yaml"

        # RobotModel initialization
        self.robot_model = Robot(
            urdf_string=urdf_string,
            base_link="panda_link0",
            ee_link="panda_link8",
            joint_limits_path=joint_limits_path
        )

        # Initialize robot model and collision checker
        self.collision_checker = CollisionChecker(self.robot_model, group_name="panda_arm")

        # Load RRT-specific parameters from config
        rrt_params = load_planner_params("rrt")
        rrt_star_params = load_planner_params("rrt_star")

        # self.custom_planner = RRTPlanner(self.robot_model, self.collision_checker, rrt_params)
        # self.custom_planner = RRTStarPlanner(self.robot_model, self.collision_checker, rrt_star_params)
        self.custom_planner = RRTWithConnectingPlanner(self.robot_model, self.collision_checker, rrt_params)

        # Storage for data comparison
        self.data_log = []

        ###################
        ## MOVE TO POSE 0 #
        ###################
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Standard base pose
        # self.start_joint_config = [0, 0, 0, -0.1, 0, math.pi / 2.0, 0] # Vertical starting pose
        self.robot_model.move_to_joint_config(self.start_joint_config)


        ##################################
        ######### CUSTOM STUFF ###########
        ##################################
        camera_operations = CameraOperations()
        panda_transformations = PandaTransformations(cam_operations=camera_operations)
        panda_transformations.calibrate_camera()

        #####################################
        # WE WILL BE MOVING TO THESE POINTS #
        #####################################
        # Define fixed joint configuration for consistent execution
        # Points in camera frame
        # point_1 = PointWithOrientation(0.0, 0.0, 1.1, 0.0, 0.0, -math.pi/4.0)
        # point_2 = PointWithOrientation(0.3, 0.3, 1.0, 0.0, 0.0, -math.pi/4.0)
        # point_3 = PointWithOrientation(-0.3, -0.3, 1.0, 0.0, 0.0, -math.pi/4.0)
        # point_4 = PointWithOrientation(0.0, 0.0, 1.0, 0.0, 0.0, -math.pi/4.0)
        table_corner_0 = PointWithOrientation(0.0, 0.0, 0.05, 0.0, math.pi, -math.pi)
        point_1 = PointWithOrientation(0.0, 0.0, 0.9, 0.0, 0.0, -math.pi / 4.0)
        point_1 = PointWithOrientation(0.0, 0.0, 0.9, 0.0, 0.0, math.pi * 0.75)

        print("TRYING TO FIND A CUSTOM OBJECT")
        print(F"DEBUG DEBUG DEBUG ----->>>>> {camera_operations.USE_REALSENSE}")
        if camera_operations.USE_REALSENSE:
            success, x, y, z = camera_operations.find_tennis()
        else:
            success = True
            x, y, z = 0.01, 0.01, 1.0
        if success:
            object_in_camera_frame = PointWithOrientation(x, y, z, 0.0, 0.0, math.pi * 0.75)
            object_in_base_frame = panda_transformations.transform_point(object_in_camera_frame, 'camera', 'base')

            print(f"X = {x} | Y = {y} | Z = {z}")
        else:
            object_in_camera_frame = point_1
            print("NO OBJECT DETECTED")
        print("OBJECT DETECTION DONE")
        # object_in_camera_frame = PointWithOrientation(0.15, 0.15, 1.2, 0.0, 0.0, -math.pi/4.0)
        # object_in_base_frame = panda_transformations.transform_point(object_in_camera_frame, 'camera', 'base')
        point_above_object_in_base_frame = PointWithOrientation(
            object_in_base_frame.x,
            object_in_base_frame.y,
            object_in_base_frame.z + 0.12,
            object_in_base_frame.roll,
            object_in_base_frame.pitch,
            object_in_base_frame.yaw
        )

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
        visualisation_frames["tennis"] = object_in_base_frame.as_matrix()

        # Visualise
        # panda_transformations.visusalise_environment(visualisation_frames)

        # Define target positions including boxes
        self.target_positions = [
            # panda_transformations.transform_point(table_corner_0, 'table', 'base'),
            point_above_object_in_base_frame,
            object_in_base_frame,
            point_above_object_in_base_frame,
            panda_transformations.transform_point(point_1, 'camera', 'base'),
            point_box_1,
            point_box_2
        ]

    def execute(self):
        """Main execution sequence"""

        ####################
        ##### TRACK IK #####
        ####################
        # # # Move to fixed starting joint configuration before each method
        # rospy.loginfo("Moving to Start Joint Configuration before TRAC-IK execution")
        # self.robot_model.move_to_joint_config(self.start_joint_config)

        # rospy.loginfo("Executing predefined movements using TRAC-IK")
        # for pos in self.target_positions:
        #     self.robot_model.move_to_pose_trac_ik(pos)

        ##########################
        ##### MOVEIT PLANNER #####
        ##########################
        # rospy.loginfo("Returning to Start Joint Configuration before trajectory planner execution")
        # self.robot_model.move_to_joint_config(self.start_joint_config)

        # # Add a table as an obstacle
        # box_pose = geometry_msgs.msg.PoseStamped()
        # box_pose.header.frame_id = self.robot.get_planning_frame()  # typically "panda_link0" or "world"
        # box_pose.pose.position.x = 0.4
        # box_pose.pose.position.y = 0.0
        # box_pose.pose.position.z = 0.19  # box center height
        # box_pose.pose.orientation.w = 1.0  # neutral orientation

        # # self.scene.add_box("table_box", box_pose, size=(0.6, 0.8, 0.02))  # (x, y, z) dimensions in meters
        # rospy.sleep(1.0)  # Give time for the scene to update

        # rospy.loginfo("Executing predefined movements using MoveIt Trajectory Planner")
        # self.robot_model.move_gripper(True)

        # for i, pos in enumerate(self.target_positions):
        #     print(f"Moving to position: {pos}, type: {type(pos)}")
        #     rospy.loginfo(f"Moving to position: {pos}")
        #     if i == 4:
        #         self.robot_model.move_gripper(False)
        #         rospy.sleep(1)
        #     self.robot_model.move_to_pose_planner(pos)

        ####################################
        #### CUSTOM TRAJECTORY PLANNER #####
        ####################################
        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.robot_model.move_to_joint_config(self.start_joint_config)

        # Close gripper, wait 3s, open gripper
        self.robot_model.move_gripper(False)
        rospy.sleep(0.5)
        self.robot_model.move_gripper(True)

        rospy.loginfo("Executing predefined movements using custom Trajectory Planner")
        for i,pos in enumerate(self.target_positions):
            current_config = np.array(self.group.get_current_joint_values())
            self.custom_planner.set_start(current_config)
            self.custom_planner.set_goal(pos)
            path, success = self.custom_planner.plan()

            # Call shortcutting function (edit path)
            path_shortcutter = PathPostProcessing(self.collision_checker)
            path = path_shortcutter.generate_a_shortcutted_path(path)

            if i == 2:
                self.robot_model.move_gripper(False)
                rospy.sleep(0.5)

            if i == 5:
                self.robot_model.move_gripper(True)
                rospy.sleep(0.5)

            if success:
                rospy.loginfo(f"RRT path found with {len(path)} waypoints.")
                for config in path:
                    self.robot_model.execute_joint_positions(config, "Custom RRT")
            else:
                rospy.logwarn("RRT planner failed to find a path.")

        self.robot_model.move_to_joint_config(self.start_joint_config)
        # # Save data for comparison
        # self.save_data()
        # rospy.loginfo("Execution complete.")

if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(0.5)  # Allow ROS to initialize
    controller.execute()
