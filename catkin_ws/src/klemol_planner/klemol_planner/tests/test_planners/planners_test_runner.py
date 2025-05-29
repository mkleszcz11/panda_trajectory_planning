import rospy
from klemol_planner.tests.test_planners.planners_test_executor import PlannersTestExecutor
from klemol_planner.tests.main_test_logger import MainTestLogger
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.post_processing.path_post_processing import PathPostProcessing

import moveit_commander
import copy


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

        self.logger = MainTestLogger()
        self.executor = PlannersTestExecutor(robot_model=self.robot_model,
                                             collision_checker=self.collision_checker,
                                             post_processing=self.post_processing,
                                             logger=self.logger)
        

        self.planners = [
            "rrt",
            "rrt_with_connecting",
            # "rrt_star",
            # "prm",
            # "prm_star",
            # "sbl",
            # "est",
            # "birrt",
            # "kpiece"
        ]

    def run_tests(self):
        """
        Main function to run the tests.
        """
        for i in range(self.NUMBER_OF_LOOPS):
            rospy.loginfo(f"Starting loop {i + 1}/{self.NUMBER_OF_LOOPS}")
            start_joint_config = self.robot_model.sample_random_valid_configuration(collision_checker=self.collision_checker)
            goal_joint_config = self.robot_model.sample_random_valid_configuration(collision_checker=self.collision_checker)

            # We should test achieving the cartesian pose, we don't know what will be the joint configuration
            goal_end_effector_pose = self.robot_model.fk(goal_joint_config)
            print((f"GOAL END-EFFECTOR POSE: {goal_end_effector_pose}"))
            # Be sure the goal is PointWithOrientation
            if not isinstance(goal_end_effector_pose, PointWithOrientation):
                raise TypeError("Goal end-effector pose must be of type PointWithOrientation.")

            for planner in self.planners:
                rospy.loginfo(f"Running planner: {planner}")
                # self.logger.set_planner(planner)

                # Run the planner test
                self.executor.run_test(planner_type=planner,
                                       start_joint_config=start_joint_config,
                                       goal_end_effector_pose=goal_end_effector_pose)

                rospy.sleep(0.5)

        # Save all results to a single file after all tests
        self.logger.save_all('/tmp/planner_test_results.npz')
        rospy.loginfo("All results saved.")

if __name__ == "__main__":
    test_runner = PlannersTestRunner()
    test_runner.run_tests()
    rospy.loginfo("All tests completed.")
