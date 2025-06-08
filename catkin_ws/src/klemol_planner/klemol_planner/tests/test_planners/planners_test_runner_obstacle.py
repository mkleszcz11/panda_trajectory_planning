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
        self.NUMBER_OF_LOOPS = 2

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

        # Add a box obstacle
        self.add_box_obstacle(name="table_box",
                              size=(0.02, 0.02, 0.8),  # dimensions (x, y, z) in meters
                              position=(0.40, 0.0, 0.4))  # center of box relative to world

    def run_tests(self):
        """
        Main function to run the tests.
        """
        for i in range(self.NUMBER_OF_LOOPS):
            rospy.loginfo(f"Starting loop {i + 1}/{self.NUMBER_OF_LOOPS}")
            self.logger.current_loop_index = i
            start_joint_config = [0, -0.985, 0, -2.356, 0, 1.571, 0.785]# - math.pi] 
            # We should test achieving the cartesian pose, we don't know what will be the joint configuration
            goal_end_effector_pose = PointWithOrientation(0.6, 0.0, 0.3, math.pi, 0.0, math.pi * 0.75)

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
        self.logger.save('/tmp/planner_test_results.npz')
        rospy.loginfo("All results saved.")

    def add_box_obstacle(self, name, size, position, orientation=(0, 0, 0, 1)):
        """
        Add a box obstacle to the planning scene.
        Args:
            name (str): Name of the obstacle.
            size (tuple): (x, y, z) dimensions of the box.
            position (tuple): (x, y, z) center of the box.
            orientation (tuple): Quaternion (x, y, z, w) orientation. Defaults to no rotation.
        """
        box_pose = PoseStamped()
        box_pose.header.frame_id = self.robot_model.base_link  # or "panda_link0" / "world"
        box_pose.pose.position.x = position[0]
        box_pose.pose.position.y = position[1]
        box_pose.pose.position.z = position[2]
        box_pose.pose.orientation.x = orientation[0]
        box_pose.pose.orientation.y = orientation[1]
        box_pose.pose.orientation.z = orientation[2]
        box_pose.pose.orientation.w = orientation[3]

        self.scene.add_box(name, box_pose, size=size)
        rospy.sleep(1.0)  # Allow time for the scene to update
        rospy.loginfo(f"Added box obstacle '{name}' with size {size} at {position}")

if __name__ == "__main__":
    test_runner = PlannersTestRunner()
    test_runner.run_tests()
    rospy.loginfo("All tests completed.")
