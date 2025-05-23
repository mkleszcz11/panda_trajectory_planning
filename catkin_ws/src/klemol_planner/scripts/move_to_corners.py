#!/usr/bin/env python3

import rospy
import math
import geometry_msgs.msg
import tf.transformations as tf_trans
import moveit_commander
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.camera_utils.camera_operations import CameraOperations
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import actionlib
import numpy as np

class TableCornerMover:
    def __init__(self):
        rospy.init_node("move_to_table_corners")

        # Initialize MoveIt
        moveit_commander.roscpp_initialize([])
        self.group = moveit_commander.MoveGroupCommander("panda_arm")

        # # Reset speed/acceleration scaling
        # self.group.set_max_velocity_scaling_factor(0.6)
        # self.group.set_max_acceleration_scaling_factor(0.4)

        # Robot frame transformation utility
        self.camera_operations = CameraOperations()
        self.panda_transformations = PandaTransformations(cam_operations=self.camera_operations)
        self.panda_transformations.calibrate_camera()
        # self.panda_transformations.visusalise_environment()
        
        # Calculate and print reprojection error
        mean_error, corner_errors = self.calculate_mean_reprojection_error()
        rospy.loginfo(f"Mean reprojection error: {mean_error:.4f} meters")
        for corner, error in corner_errors.items():
            rospy.loginfo(f"  {corner}: {error:.4f} meters")
        
        # Calculate and print reprojection error
        mean_error, corner_errors = self.calculate_mean_reprojection_error()
        rospy.loginfo(f"Mean reprojection error: {mean_error:.4f} meters")
        for corner, error in corner_errors.items():
            rospy.loginfo(f"  {corner}: {error:.4f} meters")

        # Define the table corners in camera frame (IDs 0 to 3)
        marker_transforms = self.panda_transformations.camera_operations.get_marker_transforms()
        #self.corner_names = ["corner_0", "corner_1", "corner_2", "corner_3"]
        self.corner_names = ["corner_0", "corner_1"]
        self.target_positions = []

        cam_frame_point_under_camera = PointWithOrientation(0, 0, 1.0, 0.0, 0.0, math.pi * 0.75)
        self.point_under_camera = self.panda_transformations.transform_point(cam_frame_point_under_camera, 'camera', 'base')

        for name in self.corner_names:
            if name not in marker_transforms:
                rospy.logwarn(f"Table corner '{name}' not detected. Skipping.")
                continue

            mat = marker_transforms[name]
            x, y, z = mat[:3, 3]
            # Define approach pose (above corner, facing down)
            corner_cam = PointWithOrientation(x, y, z, 0.0, 0.0, math.pi)#math.pi * 0.75)
            corner_base = self.panda_transformations.transform_point(corner_cam, 'camera', 'base')
            corner_cam_5_cm_above_table = PointWithOrientation(x, y, z - 0.05, 0.0, 0.0, math.pi)#math.pi * 0.75)
            corner_cam_5_cm_above_table = self.panda_transformations.transform_point(corner_cam_5_cm_above_table, 'camera', 'base')
            
            corner_x = self.panda_transformations.table_corners_translations[name][0]
            corner_y = self.panda_transformations.table_corners_translations[name][1]
            corner_z = self.panda_transformations.table_corners_translations[name][2]
            print(f"CALIBRATED CORENER TRANSLATION -> {corner_x}, {corner_y}, {corner_z}")
            print(f"CORENER BASED ON CAMERA        -> {corner_base.x}, {corner_base.y}, {corner_base.z}")
            corner_as_calibrated = PointWithOrientation(corner_x, corner_y, corner_z, math.pi, 0.0, 0.0)# math.pi / 2.0)
            corner_as_calibrated.yaw -= math.pi / 2.0

            self.target_positions.append(self.point_under_camera)
            self.target_positions.append(corner_cam_5_cm_above_table)
            self.target_positions.append(corner_base)
            self.target_positions.append(corner_as_calibrated)
            self.target_positions.append(corner_cam_5_cm_above_table)

        rospy.loginfo(f"Prepared {len(self.target_positions)} table corner targets.")

    def calculate_mean_reprojection_error(self):
        """
        Calculate the mean reprojection error between calibrated corner positions and camera-detected positions.
        
        Returns:
            mean_error: Mean Euclidean distance error across all corners
            errors: Dictionary of individual errors for each corner
        """
        # Get marker transforms from camera
        marker_transforms = self.panda_transformations.camera_operations.get_marker_transforms()
        
        # Dictionary to store individual errors for each corner
        errors = {}
        total_error = 0.0
        corner_count = 0
        
        # Corner names to process
        corner_names = ["corner_0", "corner_1", "corner_2", "corner_3"]
        
        # Process each corner
        for name in corner_names:
            if name not in marker_transforms:
                rospy.logwarn(f"Corner '{name}' not detected by camera. Skipping.")
                continue
                
            # Get camera-detected position
            mat = marker_transforms[name]
            x_cam, y_cam, z_cam = mat[:3, 3]
            
            # Transform from camera frame to base frame
            corner_cam = PointWithOrientation(x_cam, y_cam, z_cam, 0.0, 0.0, math.pi)
            corner_base = self.panda_transformations.transform_point(corner_cam, 'camera', 'base')
            
            # Get calibrated position from panda_transformations
            if name in self.panda_transformations.table_corners_translations:
                x_calib = self.panda_transformations.table_corners_translations[name][0]
                y_calib = self.panda_transformations.table_corners_translations[name][1]
                z_calib = self.panda_transformations.table_corners_translations[name][2]
                
                # Calculate Euclidean distance error
                error = np.sqrt((corner_base.x - x_calib)**2 + 
                                (corner_base.y - y_calib)**2 + 
                                (corner_base.z - z_calib)**2)
                
                errors[name] = error
                total_error += error
                corner_count += 1
                
                rospy.loginfo(f"{name} - Camera detected: ({corner_base.x:.4f}, {corner_base.y:.4f}, {corner_base.z:.4f})")
                rospy.loginfo(f"{name} - Calibrated: ({x_calib:.4f}, {y_calib:.4f}, {z_calib:.4f})")
                rospy.loginfo(f"{name} - Error: {error:.4f} meters")
            else:
                rospy.logwarn(f"No calibration data for '{name}'")
        
        # Calculate mean error
        mean_error = total_error / corner_count if corner_count > 0 else 0.0
        
        return mean_error, errors
    
    def robot_model.move_to_pose_planner(self, pose: PointWithOrientation):
        """Move the robot using MoveIt's motion planner"""
        pose_target = geometry_msgs.msg.Pose()
        quaternion = tf_trans.quaternion_from_euler(pose.roll, pose.pitch, pose.yaw)

        pose_target.position.x = pose.x
        pose_target.position.y = pose.y
        pose_target.position.z = pose.z
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]

        self.group.set_pose_target(pose_target)
        success = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

        if success:
            rospy.loginfo("Motion succeeded.")
        else:
            rospy.logwarn("Motion failed.")

    def robot_model.move_gripper(self, open_gripper: bool):
        """
        Open or close the gripper
        Args:
            open_gripper: True to open, False to close
        """
        if open_gripper:
            print(f"OPENING GRIPPER")
            client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
            client.wait_for_server()

            goal = MoveGoal()
            goal.width = 0.08  # fully open
            goal.speed = 0.1
            client.send_goal(goal)
            client.wait_for_result()

        else:
            print(f"CLOSING GRIPPER")
            client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
            client.wait_for_server()

            goal = GraspGoal()
            goal.width = 0.02
            goal.speed = 0.1
            goal.force = 5
            goal.epsilon.inner = 0.0
            goal.epsilon.outer = 0.06

            client.send_goal(goal)
            client.wait_for_result()
            result = client.get_result()

            if not result.success:
                rospy.logwarn("Gripper: grasp failed.")
            else:
                rospy.loginfo("Gripper: grasp succeeded.")

    def execute(self):
        # Close gripper:
        # self.robot_model.move_gripper(False)

        rospy.loginfo("Moving to all detected table corners.")
        for i, target_pose in enumerate(self.target_positions):
            rospy.loginfo(f"Moving to pose num {i}")
            self.robot_model.move_to_pose_planner(target_pose)
            rospy.sleep(1.0)  # small pause

        self.robot_model.move_to_pose_planner(self.point_under_camera)

        rospy.loginfo("Finished all table corner moves.")


if __name__ == "__main__":
    mover = TableCornerMover()
    rospy.sleep(1)
    mover.execute()
