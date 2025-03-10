#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf_trans
import numpy as np
import csv
import time
from trac_ik_python.trac_ik import IK

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

        # Define points in (x, y, z, roll, pitch, yaw)
        self.target_positions = [
            (0.5, 0.2, 0.4, 0, 0, 0),  # A
            (0.4, -0.2, 0.5, 0, 0, 1.57),  # B
            (0.3, 0.1, 0.6, 0, -0.5, 0),  # C
            (0.6, -0.1, 0.3, 1.57, 0, 0),  # D
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

    def move_to_pose_trac_ik(self, x, y, z, roll, pitch, yaw):
        """Move the robot using TRAC-IK"""
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

    def move_to_pose_planner(self, x, y, z, roll, pitch, yaw):
        """Move the robot using MoveIt's motion planner"""
        pose_target = geometry_msgs.msg.Pose()
        quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
        pose_target.position.x = x
        pose_target.position.y = y
        pose_target.position.z = z
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        pose_target.orientation.w = quaternion[3]

        self.group.set_pose_target(pose_target)
        start_time = time.time()
        self.group.go(wait=True)
        end_time = time.time()

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

        # Move to fixed starting joint configuration before each method
        rospy.loginfo("Moving to Start Joint Configuration before TRAC-IK execution")
        self.move_to_joint_config(self.start_joint_config)

        rospy.loginfo("Executing predefined movements using TRAC-IK")
        for pos in self.target_positions:
            self.move_to_pose_trac_ik(*pos)

        rospy.loginfo("Returning to Start Joint Configuration before trajectory planner execution")
        self.move_to_joint_config(self.start_joint_config)

        rospy.loginfo("Executing predefined movements using a Trajectory Planner")
        for pos in self.target_positions:
            self.move_to_pose_planner(*pos)

        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.move_to_joint_config(self.start_joint_config)

        # Save data for comparison
        self.save_data()
        rospy.loginfo("Execution complete.")

if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(1)  # Allow ROS to initialize
    controller.execute()
