#!/usr/bin/env python3

# import rospy
# import moveit_commander
# import moveit_msgs.msg
# import geometry_msgs.msg
# import tf.transformations as tf_trans
# from trac_ik_python.trac_ik import IK
# import numpy as np
# from random import random
# import math

from PandaTransformations import PandaTransformations
from PointWithOrientation import PointWithOrientation

#!/usr/bin/env python3

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf.transformations as tf_trans
from trac_ik_python.trac_ik import IK
import numpy as np
import math

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
            self.robot_model.execute_joint_positions(joint_positions)
        else:
            rospy.logerr("No IK solution found!")

    def robot_model.execute_joint_positions(self, joint_positions):
        # Move the robot to the computed joint values
        self.group.set_joint_value_target(joint_positions)
        self.group.go(wait=True)

if __name__ == "__main__":
    controller = FrankaIKController()

    rospy.sleep(1)  # Allow ROS to initialize

    # Initialize transformation handler
    my_transformations = PandaTransformations()

    # Set up a point in a table coordinate system
    table_point = PointWithOrientation(0.1, 0.1, 0.1, 0.0, math.pi , math.pi/4.0)
    # table_point.set_position(-0.1, -0.1, 0.0)  # Relative to table origin
    # table_point.set_orientation(0.0, math.pi , math.pi/4.0)  # Example orientation

    print(f"table point -> {table_point.get_position()}")
    print(f"table point -> {table_point.get_orientation()}")

    # Transform point to base frame
    transformed_point = my_transformations.transform_point(table_point, 'table', 'base')



    # # Transform to base frame
    # transformed_point = my_transformations.transform_point(
    #     table_point.get_position(),
    #     'table',
    #     'base'
    # )
    # # get rotation matrix from transformation matrix
    # orientation = my_transformations.get_transform('table', 'base')[:3, :3]
    # print(f"orientation -> {orientation}")
    # euler_angles = tf_trans.euler_from_matrix(orientation)

    # # Update transformed point with rotation
    # transformed_point = PointWithOrientation(
    #     transformed_point[0], transformed_point[1], transformed_point[2],
    #     euler_angles[0], euler_angles[1], euler_angles[2]
    # )

    # rospy.loginfo(f"Moving to: {transformed_point.get_position()} with orientation {transformed_point.get_orientation()}")

    # Move robot to target pose
    controller.move_to_pose(
        transformed_point.x,
        transformed_point.y,
        transformed_point.z,
        transformed_point.roll,
        transformed_point.pitch,
        transformed_point.yaw
    )

    rospy.spin()


# class FrankaIKController:
#     def __init__(self):
#         rospy.init_node("franka_ik_controller")

#         # Initialize MoveIt Commander
#         moveit_commander.roscpp_initialize([])
#         self.robot = moveit_commander.RobotCommander()
#         self.scene = moveit_commander.PlanningSceneInterface()
#         self.group = moveit_commander.MoveGroupCommander("panda_arm")

#         # TRAC-IK Solver: Base link -> End Effector
#         self.ik_solver = IK("panda_link0", "panda_link8")

#         # Get joint limits
#         self.lower_bounds, self.upper_bounds = self.ik_solver.get_joint_limits()

#     def move_to_pose(self, x, y, z, roll, pitch, yaw):
#         # Convert (roll, pitch, yaw) to quaternion
#         quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)

#         # Solve IK for desired pose
#         seed_state = np.random.uniform(self.lower_bounds, self.upper_bounds)  # Random seed state
#         joint_positions = self.ik_solver.get_ik(seed_state, x, y, z, *quaternion)

#         if joint_positions:
#             rospy.loginfo("IK Solution Found!")
#             self.robot_model.execute_joint_positions(joint_positions)
#         else:
#             rospy.logerr("No IK solution found!")

#     def robot_model.execute_joint_positions(self, joint_positions):
#         # Move the robot to the computed joint values
#         self.group.set_joint_value_target(joint_positions)
#         self.group.go(wait=True)

# if __name__ == "__main__":
#     controller = FrankaIKController()

#     # Example: Move to (x=0.4, y=0, z=0.5) with no rotation
#     # rospy.sleep(1)  # Allow ROS to initialize
#     # x = 0.6866172576844853
#     # y = -0.40264805670207643
#     # z = 0.1882167138629311
#     # roll = 0#random()
#     # pitch = 3.14#random()
#     # yaw = 0#random()
#     # controller.move_to_pose(x, y, z, roll, pitch, yaw)
    
    
#     # my_transformations = PandaTransformations()
#     # table_point = PointWithOrientation()
    
#     # table_point.set_position(0.0, 0.0, 0.0)
#     # table_point.set_orientation(0.0, math.pi-0.01, 0.0)
#     # transformed_point = my_transformations.transform_table_to_base_link(table_point)
#     # print(transformed_point.get_position())
#     # print(transformed_point.get_orientation())
#     rospy.sleep(1)

#     controller.move_to_pose(transformed_point.x, transformed_point.y, transformed_point.z, transformed_point.roll, transformed_point.pitch, transformed_point.yaw)

#     rospy.spin()
