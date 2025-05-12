import rospy
from klemol_planner.tests.splines_test_executor import TestExecutor
from klemol_planner.tests.trajectory_logger import TrajectoryLogger
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
import moveit_commander


if __name__ == "__main__":
    rospy.init_node("franka_motion_controller")
    moveit_commander.roscpp_initialize([])
    # controller = FrankaMotionController()
    robot_model = Robot()
    collision_checker = CollisionChecker(robot_model, group_name="panda_arm")
    logger = TrajectoryLogger()
    executor = TestExecutor(collision_checker, logger)

    start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

    # Standard motion
    robot_model.move_to_joint_config(start_joint_config)
    rospy.sleep(2)
    executor.run_test(mode="raw")

    # robot_model.move_to_joint_config(start_joint_config)
    # rospy.sleep(2)
    # executor.run_test(mode="spline_time_parametrised")

    # Spline motion with constraints
    robot_model.move_to_joint_config(start_joint_config)
    rospy.sleep(2)
    executor.run_test(mode="spline_time_parametrised_with_constraints")

    logger.save("/tmp/franka_motion_comparison.npz")
    rospy.loginfo("Data saved for post-analysis.")