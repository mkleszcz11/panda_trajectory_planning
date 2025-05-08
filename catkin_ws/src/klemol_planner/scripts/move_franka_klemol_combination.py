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
from klemol_planner.post_processing.path_shortcutter import PathShortcutter
import math
import subprocess
import itertools

from klemol_planner.environment.robot_model import RobotModel
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
        self.robot_model = RobotModel(
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
        self.move_to_joint_config(self.start_joint_config)


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

    def move_to_joint_config(self, joint_config):
        """Move the robot to a specific joint configuration."""
        self.group.clear_pose_targets()
        self.group.set_joint_value_target(joint_config)
        rospy.loginfo(f"Moving to joint configuration: {joint_config}")
        self.group.go(wait=True)
        self.group.plan()

    # def move_to_pose_trac_ik(self, position: PointWithOrientation):
    #     """Move the robot using TRAC-IK"""
    #     x, y, z = position.x, position.y, position.z
    #     roll, pitch, yaw = position.roll, position.pitch, position.yaw
    #     quaternion = tf_trans.quaternion_from_euler(roll, pitch, yaw)
    #     seed_state = np.random.uniform(self.lower_bounds, self.upper_bounds)  # Random seed
    #     # joint_positions = self.ik_solver.get_ik(seed_state, x, y, z, *quaternion)
    #     joint_positions = self.group.get_current_joint_values()
    #     print(f"JOINT POSITIONS: {joint_positions}")

    #     if joint_positions:
    #         rospy.loginfo(f"TRAC-IK Solution Found for ({x}, {y}, {z})")
    #         self.execute_joint_positions(joint_positions, "TRAC-IK")
    #     else:
    #         rospy.logerr("No IK solution found!")

    def execute_joint_positions(self, joint_positions, method):
        """Execute a joint position command and log the data"""
        self.group.clear_pose_targets()
        rospy.loginfo(f"====== Moving to joint configurations: {joint_positions}")
        self.group.set_max_velocity_scaling_factor(0.75)      # scale speed
        self.group.set_max_acceleration_scaling_factor(0.1)  # scale acceleration

        for pos in joint_positions:
            print(f"executing position: {pos}")
            self.group.set_joint_value_target(pos)
            self.group.plan()
            self.group.go(wait=True)


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
        # self.group.set_max_velocity_scaling_factor(0.6)      # scale speed
        # self.group.set_max_acceleration_scaling_factor(0.2)  # scale acceleration
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

    def move_gripper(self, open_gripper: bool):
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

    def compute_goal_configurations_for_poses(self, poses: t.List[PointWithOrientation]) -> t.List[t.List[np.ndarray]]:
        """
        For each pose, use the planner's internal IK generation method
        (which calls ik_with_custom_solver) to get multiple valid joint configurations.

        Returns:
            A list of lists, where each inner list contains valid IK solutions for a given pose.
        """
        all_goal_configs = []
        for i, pose in enumerate(poses):
            rospy.loginfo(f"Generating goal configurations for pose {i}")
            self.custom_planner.set_goal(pose)  # sets self.custom_planner.goal_pose
            goal_configs = self.custom_planner.generate_goal_configurations(pose)
            if not goal_configs:
                rospy.logwarn(f"[Pose {i}] No IK solutions found.")
            else:
                rospy.loginfo(f"[Pose {i}] Found {len(goal_configs)} IK solutions.")
            all_goal_configs.append(goal_configs)
        return all_goal_configs

    def find_best_joint_sequence_reverse(self, config_sets):
        """
        Greedy backward search: start from final pose and iteratively pick the best previous config.
        """
        num_poses = len(config_sets)
        best_path = []
        current = None  # current end configuration

        for i in reversed(range(num_poses)):
            best_prev = None
            best_cost = float("inf")
            options = config_sets[i]

            if not options:
                rospy.logwarn(f"No IK options available for pose {i}")
                return None, None

            if current is None:
                # We're at the final pose; pick the most manipulable or any valid one
                best_prev = options[0]
                best_path.insert(0, best_prev)
                current = best_prev
                continue

            for option in options:
                self.custom_planner.set_start(option)
                self.custom_planner.goal_configs = [current]
                path, success = self.custom_planner.plan()

                if not success or not path:
                    continue

                cost = len(path)
                if cost < best_cost:
                    best_cost = cost
                    best_prev = option
                    best_path.insert(0, option)

            if best_prev is None:
                rospy.logwarn(f"Could not find a valid connection for pose {i}")
                return None, None

            current = best_prev

        return best_path, None

    def find_best_joint_sequence_reverse_horizonN(self, config_sets, horizon: int):
        """
        Reverse planning with tunable N-step horizon and speed optimisations.

        Args:
            config_sets: list of IK solutions per pose
            horizon: number of previous poses to consider (must be ≥1)

        Returns:
            best_path: list of joint configs in execution order
        """
        assert horizon >= 1, "Horizon must be ≥ 1"
        num_poses = len(config_sets)
        if num_poses < horizon:
            rospy.logerr("Not enough poses for given horizon.")
            return None, None

        planner = self.custom_planner
        cost_cache = {}  # cache for (start_bytes, goal_bytes) → cost
        fail_cache = set()  # set of (start_bytes, goal_bytes) that failed

        def get_plan_cost(q_start: np.ndarray, q_goal: np.ndarray) -> t.Optional[int]:
            key = (q_start.tobytes(), q_goal.tobytes())
            if key in cost_cache:
                return cost_cache[key]
            if key in fail_cache:
                return None

            planner.set_start(q_start)
            planner.goal_configs = [q_goal]
            path, success = planner.plan()
            if not success or not path:
                fail_cache.add(key)
                return None

            cost = len(path)
            cost_cache[key] = cost
            return cost

        current = config_sets[-1][0] if config_sets[-1] else None
        if current is None:
            rospy.logerr("No IK solutions at final pose.")
            return None, None

        best_path = [current]
        i = num_poses - 1 - horizon

        while i >= 0:
            best_seq = None
            best_score = float("inf")

            try_sequences = itertools.product(*config_sets[i:i + horizon])

            for seq in try_sequences:
                segment = list(seq) + [current]
                total_cost = 0
                valid = True

                for a, b in zip(segment[:-1], segment[1:]):
                    cost = get_plan_cost(a, b)
                    if cost is None:
                        valid = False
                        break
                    total_cost += cost

                if valid and total_cost < best_score:
                    best_score = total_cost
                    best_seq = seq

            if best_seq is None:
                rospy.logerr(f"No valid sequence found at i={i} with horizon={horizon}")
                return None, None

            best_path = list(best_seq) + best_path
            current = best_seq[0]
            i -= horizon

        return best_path, None


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
        # rospy.loginfo("Returning to Start Joint Configuration before trajectory planner execution")
        # self.move_to_joint_config(self.start_joint_config)

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
        # self.move_gripper(True)

        # for i, pos in enumerate(self.target_positions):
        #     print(f"Moving to position: {pos}, type: {type(pos)}")
        #     rospy.loginfo(f"Moving to position: {pos}")
        #     if i == 4:
        #         self.move_gripper(False)
        #         rospy.sleep(1)
        #     self.move_to_pose_planner(pos)

        ####################################
        #### CUSTOM TRAJECTORY PLANNER #####
        ####################################
        rospy.loginfo("Returning to Start Joint Configuration after execution")
        self.move_to_joint_config(self.start_joint_config)

        # Close gripper, wait 3s, open gripper
        self.move_gripper(False)
        rospy.sleep(0.5)
        self.move_gripper(True)

        rospy.loginfo("Executing predefined movements using custom Trajectory Planner")

        rospy.loginfo("Generating IK options for all target poses...")
        config_sets = self.compute_goal_configurations_for_poses(self.target_positions)

        rospy.loginfo("Evaluating all IK combinations...")
        best_path, _ = self.find_best_joint_sequence_reverse_horizonN(config_sets, horizon=3)

        if best_path:
            rospy.loginfo(f"Best joint sequence found with total cost: {len(best_path)}")
            for i, config in enumerate(best_path):
                if i == 2:
                    self.move_gripper(False)
                    rospy.sleep(0.5)
                if i == 5:
                    self.move_gripper(True)
                    rospy.sleep(0.5)
                self.execute_joint_positions(config, "Custom Optimised RRT")
        else:
            rospy.logerr("Failed to find valid full-sequence trajectory.")

        self.move_to_joint_config(self.start_joint_config)
        # # Save data for comparison
        # self.save_data()
        # rospy.loginfo("Execution complete.")

if __name__ == "__main__":
    controller = FrankaMotionController()
    rospy.sleep(0.5)  # Allow ROS to initialize
    controller.execute()
