import rospy
from klemol_planner.tests.test_planners.planners_test_executor import PlannersTestExecutor
from klemol_planner.tests.main_test_logger import MainTestLogger
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.post_processing.path_post_processing import PathPostProcessing

import math
import moveit_commander
import copy
import os
import yaml
from geometry_msgs.msg import PoseStamped


class PlannersTestRunner:
    def __init__(self):
        """
        Initializes the PlannersTestRunner.
        """
        self.NUMBER_OF_LOOPS = 30

        rospy.init_node("franka_motion_controller")
        moveit_commander.roscpp_initialize([])
        self.robot_model = Robot()
        self.collision_checker = CollisionChecker(group_name="panda_arm")
        self.post_processing = PathPostProcessing(collision_checker=self.collision_checker)

        self.logger = MainTestLogger(robot_model = self.robot_model)
        self.executor = PlannersTestExecutor(robot_model=self.robot_model,
                                             collision_checker=self.collision_checker,
                                             post_processing=self.post_processing,
                                             logger=self.logger)

        self.test_config_file = os.path.join(os.path.dirname(__file__), "test_planner_params.yaml")
        with open(self.test_config_file) as f:
            self.planner_configs = yaml.safe_load(f)

        # Initialize MoveIt planning scene
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0)  # Give time for the scene to initialize

        self.add_box_obstacle(
            name="inflated_stick",
            size=(0.02, 0.02, 0.8),
            position=(0.42, 0.0, 0.4),
            collision_margin=0.03  # Add 3 cm clearance on all sides
        )

    def run_tests(self):
        """
        Main function to run the tests.
        """
        for i in range(self.NUMBER_OF_LOOPS):
            rospy.loginfo(f"#########################################")
            rospy.loginfo(f"Starting loop {i + 1}/{self.NUMBER_OF_LOOPS}")
            rospy.loginfo(f"#########################################")
            self.logger.current_loop_index = i
            start_joint_config = [0, -0.786, 0, -2.356, 0, 1.572, 0.785 - math.pi]# - math.pi] 
            # We should test achieving the cartesian pose, we don't know what will be the joint configuration
            goal_end_effector_pose = PointWithOrientation(0.65, 0.0, 0.3, math.pi, 0.0, math.pi * 0.75)

            # Be sure the goal is PointWithOrientation
            if not isinstance(goal_end_effector_pose, PointWithOrientation):
                raise TypeError("Goal end-effector pose must be of type PointWithOrientation.")

            # Load config file:

            for planner_name, planner_config in self.planner_configs.items():
                planner_type = planner_config['planner_type']
                planner_params = planner_config['planner_params']
                rospy.loginfo(f"Running {planner_name} ({planner_type}) with parameters: {planner_params}")
                # Run the planner test
                self.executor.run_test(test_name=planner_name,
                                       planner_type=planner_type,
                                       planner_params=planner_params,
                                       start_joint_config=start_joint_config,
                                       goal_end_effector_pose=goal_end_effector_pose)

                rospy.sleep(0.5)

        # Save all results to a single file after all tests
        self.logger.save('/home/marcin/results/planner_prm_temporary.npz')
        rospy.loginfo("All results saved.")

    def add_box_obstacle(self, name, size, position, orientation=(0, 0, 0, 1), collision_margin=0.0):
        """
        Add a box obstacle to the planning scene with an optional collision margin.

        Args:
            name (str): Name of the obstacle.
            size (tuple): (x, y, z) dimensions of the actual box (meters).
            position (tuple): (x, y, z) center of the box (meters).
            orientation (tuple): Quaternion (x, y, z, w) orientation.
            collision_margin (float): Amount to inflate each dimension symmetrically (meters).
        """
        # Inflate size symmetrically
        inflated_size = tuple(s + 2 * collision_margin for s in size)

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.robot_model.base_link
        box_pose.pose.position.x = position[0]
        box_pose.pose.position.y = position[1]
        box_pose.pose.position.z = position[2]
        box_pose.pose.orientation.x = orientation[0]
        box_pose.pose.orientation.y = orientation[1]
        box_pose.pose.orientation.z = orientation[2]
        box_pose.pose.orientation.w = orientation[3]

        self.scene.add_box(name, box_pose, size=inflated_size)
        rospy.sleep(1.0)
        rospy.loginfo(f"Added box '{name}' at {position} with inflated size {inflated_size} (original: {size}, margin: {collision_margin})")

if __name__ == "__main__":
    test_runner = PlannersTestRunner()
    test_runner.run_tests()
    rospy.loginfo("All tests completed.")
