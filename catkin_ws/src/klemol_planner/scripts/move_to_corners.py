#!/usr/bin/env python3

import rospy
import math
import geometry_msgs.msg
import tf.transformations as tf_trans
import moveit_commander
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.camera_utils.camera_operations import CameraOperations


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
        self.panda_transformations.visusalise_environment()

        # Define the table corners in camera frame (IDs 0 to 3)
        marker_transforms = self.panda_transformations.camera_operations.get_marker_transforms()
        self.corner_names = ["corner_0", "corner_1", "corner_2", "corner_3"]
        self.target_positions = []

        for name in self.corner_names:
            if name not in marker_transforms:
                rospy.logwarn(f"Table corner '{name}' not detected. Skipping.")
                continue

            mat = marker_transforms[name]
            x, y, z = mat[:3, 3]
            # Define approach pose (above corner, facing down)
            corner_cam = PointWithOrientation(x, y, z - 0.02, 0.0, 0.0, math.pi * 0.75)
            corner_base = self.panda_transformations.transform_point(corner_cam, 'camera', 'base')
            self.target_positions.append(corner_base)

        rospy.loginfo(f"Prepared {len(self.target_positions)} table corner targets.")

    def move_to_pose_planner(self, pose: PointWithOrientation):
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

    def execute(self):
        rospy.loginfo("Moving to all detected table corners.")
        for i, corner_pose in enumerate(self.target_positions):
            rospy.loginfo(f"Moving to corner {i}")
            self.move_to_pose_planner(corner_pose)
            rospy.sleep(1.0)  # small pause

        rospy.loginfo("Finished all table corner moves.")


if __name__ == "__main__":
    mover = TableCornerMover()
    rospy.sleep(1)
    mover.execute()
