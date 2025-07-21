import rospy
from klemol_planner.tests.test_splines.splines_test_executor import SplinesTestExecutor
from klemol_planner.tests.trajectory_logger import TrajectoryLogger
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
import moveit_commander
import copy
import os


if __name__ == "__main__":
    ALPHA = 1
    rospy.init_node("franka_motion_controller")
    moveit_commander.roscpp_initialize([])
    robot_model = Robot()
    collision_checker = CollisionChecker(group_name="panda_arm")
    logger = TrajectoryLogger(robot_model=robot_model)
    executor = SplinesTestExecutor(robot_model, collision_checker, logger, alpha=ALPHA)

    start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

    # Use 0.25 velocity limits:
    velocity_limits = copy.deepcopy(robot_model.velocity_limits)
    # for i, limit in enumerate(velocity_limits):
    #     velocity_limits[i] = limit * 0.25

    # # Standard motion
    # robot_model.move_to_joint_config(start_joint_config)
    # rospy.sleep(2)
    # executor.run_test(mode="raw")

    # Cubic splines
    # robot_model.move_to_joint_config(start_joint_config)
    # rospy.sleep(2)
    # executor.run_test(mode="spline_cubic_hermite")

    # Quintic splines (bsplines)
    robot_model.move_to_joint_config(start_joint_config)
    rospy.sleep(2)
    executor.run_test(mode="spline_quintic_bsplines")

    # # Quintic polynomial splines
    # robot_model.move_to_joint_config(start_joint_config)
    # rospy.sleep(2)
    # executor.run_test(mode="spline_quintic_polynomial")

    file_path = f"/home/marcin/panda_trajectory_planning/catkin_ws/src/klemol_planner/klemol_planner/tests/splines_results/alpha_{ALPHA}/splines_results.npz"
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    logger.save(file_path)
