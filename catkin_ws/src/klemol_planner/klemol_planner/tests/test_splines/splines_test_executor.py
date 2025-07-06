import rospy
import numpy as np
from klemol_planner.post_processing.path_post_processing import PathPostProcessing
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from control_msgs.msg import JointTrajectoryControllerState

import math

from trac_ik_python.trac_ik import IK


class SplinesTestExecutor:
    def __init__(self, collision_checker: CollisionChecker, logger):
        self.robot_model = collision_checker.robot_model
        self.collision_checker = collision_checker
        self.logger = logger
        self.logger_sub = rospy.Subscriber(
            # "/position_joint_trajectory_controller/state",
            "/effort_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.logger.callback,
        )

    def run_test(self, mode: str = "raw", velocity_limits: np.ndarray = np.ones(7)):
        rospy.loginfo(f"Running test in mode: {mode}")
        self.logger.set_mode(mode)

        # for i, pos in enumerate(self.controller.target_positions):
        # current = np.array(self.controller.robot_model.group.get_current_joint_values())
        # self.controller.custom_planner.set_start(current)
        # self.controller.custom_planner.set_goal(pos)

        # path = [
        #     np.array([0, -0.785, 0.0, -2.356, 0, 1.571, 0.785]),
        #     np.array([0.0, -0.6, 0.5, -2.4, 0, 1.5, 0.8]),
        #     np.array([0.5, -0.5, 1.0, -1.2, 0, 1.6, 0.7]),
        #     np.array([0.6, -0.4, 0.5, -2.4, 0, 1.7, 0.9]),
        #     np.array([1.1, -0.4, 1.0, -1.2, 0, 1.8, 0.6]),
        #     np.array([0.7, -0.4, 0.5, -2.4, 0, 1.9, 1.0]),
        #     np.array([0.5, -0.4, 1.0, -1.2, 0, 2.0, 0.5]),
        # ]

        path = [
            np.array([0.5315303232882735, -0.6880944614953183, -1.295222514333675, -2.7098521010502994, -0.8629544292895819, 2.304309252112395, 0.7451632836879227]),
            np.array([0.34066860874413685, -0.7676133076880172, -0.6526554741155994, -2.1289542294024066, -0.4365001806550408, 1.486466334709319, 0.582596192755382]),
            np.array([-0.7193406015171879, 0.7030311349809253, 1.7279603680764843, -1.462167062832389, -0.7095803065350061, 1.3887629938557762, 1.685751010149957]),
            np.array([0.031114206971786906, 0.280148668136432, 0.1660746720242674, -1.7795346119093702, 0.5708601494733205, 1.69713915906316, 0.18080576717360675]),
            np.array([0.5315303232882735, -0.6880944614953183, -1.295222514333675, -2.7098521010502994, -0.8629544292895819, 2.304309252112395, 0.7451632836879227]),
            np.array([0.031114206971786906, 0.280148668136432, 0.1660746720242674, -1.7795346119093702, 0.5708601494733205, 1.69713915906316, 0.18080576717360675]),
            np.array([0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])
        ]

        post_processing = PathPostProcessing(self.collision_checker)
        # path = post_processing.generate_a_shortcutted_path(path)

        if mode == "raw":
            for idx, config in enumerate(path):
                self.robot_model.execute_joint_positions(config, "Raw")
                print(f"RAW MODE - moving to {idx}/{len(path)}")

        elif mode == "spline_time_parametrised":
            traj = post_processing.interpolate_trajectory_time_parameterised(
                path,
                joint_names=self.robot_model.group.get_active_joints(),
            )
            self.robot_model.send_trajectory_to_controller(traj)

        elif mode == "spline_cubic_hermite":
            traj = post_processing.interpolate_trajectory_with_cubic_hermite_splines(
                path=path,
                joint_names=self.robot_model.group.get_active_joints(),
                velocity_limits=velocity_limits,
            )
            self.robot_model.send_trajectory_to_controller(traj)

        elif mode == "spline_quintic":
            traj = post_processing.interpolate_quintic_trajectory(
                path=path,
                joint_names=self.robot_model.group.get_active_joints(),
                velocity_limits=velocity_limits,
            )
            self.robot_model.send_trajectory_to_controller(traj)


        rospy.sleep(1.0)  # Let the robot stabilize after each move
