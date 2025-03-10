#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf_trans
from trac_ik_python.trac_ik import IK
import numpy as np
from random import random

class FrankaIKController:
    def __init__(self):
        rospy.init_node("franka_ik_controller")

        # Initialize MoveIt Commander
        moveit_commander.roscpp_initialize([])
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")

        # TRAC-IK Solver: Base link -> End Effector
        self.ik_solver = IK("panda_link0", "panda_link8")

        # Get joint limits
        self.lower_bounds, self.upper_bounds = self.ik_solver.get_joint_limits()

    def move_to_pose(self, x, y, z, roll, pitch, yaw):
        # Convert (roll, pitch, yaw) to quaternion
        quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)

        # Solve IK for desired pose
        seed_state = np.random.uniform(self.lower_bounds, self.upper_bounds)  # Random seed state
        joint_positions = self.ik_solver.get_ik(seed_state, x, y, z, *quaternion)

        if joint_positions:
            rospy.loginfo("IK Solution Found!")
            self.execute_joint_positions(joint_positions)
        else:
            rospy.logerr("No IK solution found!")

    def execute_joint_positions(self, joint_positions):
        # Move the robot to the computed joint values
        self.group.set_joint_value_target(joint_positions)
        self.group.go(wait=True)

if __name__ == "__main__":
    controller = FrankaIKController()

    # Example: Move to (x=0.4, y=0, z=0.5) with no rotation
    rospy.sleep(1)  # Allow ROS to initialize
    x = 0.4#random()
    y = 0.4#random()
    z = 0.2#random()
    roll = 0#random()
    pitch = 3.14#random()
    yaw = 0#random()
    controller.move_to_pose(x, y, z, roll, pitch, yaw)

    rospy.spin()
