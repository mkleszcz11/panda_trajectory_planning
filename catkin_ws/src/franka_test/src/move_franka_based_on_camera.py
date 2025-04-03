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
from PandaTransformations import PandaTransformations
from PointWithOrientation import PointWithOrientation
import math

class FrankaMotionController:
    def __init__(self):
        rospy.init_node("franka_motion_controller")

        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")

        # TRAC-IK Solver
        self.ik_solver = IK("panda_link0", "panda_link8")
        self.lower_bounds, self.upper_bounds = self.ik_solver.get_joint_limits()

        # Define fixed joint configuration for consistent execution
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]  # Joint angles in radians

        panda_transformations = PandaTransformations()

        # Points in camera frame
        point_1 = PointWithOrientation(0.0, 0.0, 1.1, 0.0, 0.0, -math.pi/4.0)
        point_2 = PointWithOrientation(0.3, 0.3, 1.0, 0.0, 0.0, -math.pi/4.0)
        point_3 = PointWithOrientation(-0.3, -0.3, 1.0, 0.0, 0.0, -math.pi/4.0)
        point_4 = PointWithOrientation(0.0, 0.0, 1.0, 0.0, 0.0, -math.pi/4.0)

        self.target_positions = [
            panda_transformations.transform_point(point_1, 'camera', 'base'),
            panda_transformations.transform_point(point_2, 'camera', 'base'),
            panda_transformations.transform_point(point_3, 'camera', 'base'),
            panda_transformations.transform_point(point_4, 'camera', 'base')
        ]

        # Storage for data comparison
        self.data_log = []

    def move_to_joint_config(self, joint_config):
        """Move the robot to a specific joint configuration."""
        self.group.set_joint_value_target(joint_config)
        rospy.loginfo(f"Moving to joint configuration: {joint_config}")
        start_time = time.time()
        self.group.go(wait=True)
        end_time = time.time()
        self.log_data(joint_config, "Fixed Joint Configuration", start_time, end_time)

    def move_to_pose_trac_ik(self, position: PointWithOrientation):
        """Move the robot using TRAC-IK"""
        x, y, z = position.x, position.y, position.z
        roll, pitch, yaw = position.roll, position.pitch, position.yaw
        quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
        seed_state = np.random.uniform(self.lower_bounds, self.upper_bounds)  # Random seed
        joint_positions = self.ik_solver.get_ik(seed_state, x, y, z, *quaternion)

        if joint_positions:
            rospy.loginfo(f"TRAC-IK Solution Found for ({x}, {y}, {z})")
            self.execute_joint_positions(joint_positions, "TRAC-IK")
        else:
            rospy.logerr("No IK solution found!")

    def execute_joint_positions(self, joint_positions, method):
        """Execute a joint position command and log the data"""
        self.group.set_joint_value_target(joint_positions)
        start_time = time.time()
        self.group.go(wait=True)
        end_time = time.time()

        self.log_data(joint_positions, method, start_time, end_time)

    def move_to_pose_planner(self, pose: PointWithOrientation):
        """Move the robot using MoveIt's motion planner"""
        pose_target = geometry_msgs.msg.Pose()

        # Convert roll, pitch, yaw to quaternion
        quaternion = tf_trans.quaternion_from_euler(pose.roll, pose.pitch, pose.yaw)

        # Assign position and orientation
        pose_target.position.x = pose.x
        pose_target.position.y = pose.y
        pose_target.position.z = pose.z
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]

        # Set target and execute
        self.group.set_pose_target(pose_target)
        start_time = time.time()
        success = self.group.go(wait=True)
        end_time = time.time()

        if success:
            rospy.loginfo("Motion planning successful")
        else:
            rospy.logerr("Motion planning failed")

        # Log data (optional)
        self.log_data(self.group.get_current_joint_values(), "Trajectory Planner", start_time, end_time)

    def log_data(self, joint_positions, method, start_time, end_time):
        """Save movement data for analysis"""
        self.data_log.append({
            "method": method,
            "joint_positions": joint_positions,
            "execution_time": end_time - start_time
        })

    def save_data(self):
        """Save the logged data to a CSV file"""
        filename = "/tmp/franka_movement_log.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["method", "joint_positions", "execution_time"])
            writer.writeheader()
            for row in self.data_log:
                writer.writerow(row)
        rospy.loginfo(f"Data saved to {filename}")

    def execute(self):
        """Main execution sequence"""

        ####################
        ##### TRACK IK #####
        ####################
        # # # Move to fixed starting joint configuration before each method
        # rospy.loginfo("Moving to Start Joint Configuration before TRAC-IK execution")
        # self.move_to_joint_config(self.start_joint_config)

        # rospy.loginfo("Executing predefined movements using TRAC-IK")
        # for pos in self.target_positions:
        #     self.move_to_pose_trac_ik(pos)

        ##########################
        ##### MOVEIT PLANNER #####
        ##########################
        rospy.loginfo("Returning to Start Joint Configuration before trajectory planner execution")
        self.move_to_joint_config(self.start_joint_config)

        # Add a table as an obstacle
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = self.robot.get_planning_frame()  # typically "panda_link0" or "world"
        box_pose.pose.position.x = 0.4
        box_pose.pose.position.y = 0.0
        box_pose.pose.position.z = 0.19  # box center height
        box_pose.pose.orientation.w = 1.0  # neutral orientation

        self.scene.add_box("table_box", box_pose, size=(0.6, 0.8, 0.02))  # (x, y, z) dimensions in meters
        rospy.sleep(1.0)  # Give time for the scene to update

        rospy.loginfo("Executing predefined movements using MoveIt Trajectory Planner")
        for pos in self.target_positions:
            print(f"Moving to position: {pos}, type: {type(pos)}")
            rospy.loginfo(f"Moving to position: {pos}")
            self.move_to_pose_planner(pos)

        ####################################
        #### CUSTOM TRAJECTORY PLANNER #####
        ####################################
        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.move_to_joint_config(self.start_joint_config)

        rospy.loginfo("Executing predefined movements using custom Trajectory Planner")
        for pos in self.target_positions:
            pass


        # # Save data for comparison
        # self.save_data()
        # rospy.loginfo("Execution complete.")

if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(1)  # Allow ROS to initialize
    controller.execute()
