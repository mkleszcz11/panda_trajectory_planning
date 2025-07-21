import rospy
import numpy as np
from klemol_planner.post_processing.path_post_processing import PathPostProcessing
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from control_msgs.msg import JointTrajectoryControllerState
from klemol_planner.environment.robot_model import Robot

import math

from trac_ik_python.trac_ik import IK


class SplinesTestExecutor:
    def __init__(self, robot_model: Robot, collision_checker: CollisionChecker, logger, alpha: int = 1):
        self.alpha = alpha
        self.robot_model = robot_model
        self.collision_checker = collision_checker
        self.logger = logger
        self.logger_sub = rospy.Subscriber(
            # "/position_joint_trajectory_controller/state",
            "/effort_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.logger.callback,
        )
        # self.path = [
        #     np.array([0.5315303232882735, -0.6880944614953183, -1.295222514333675, -2.7098521010502994, -0.8629544292895819, 2.304309252112395, 0.7451632836879227]),
        #     np.array([0.34066860874413685, -0.7676133076880172, -0.6526554741155994, -2.1289542294024066, -0.4365001806550408, 1.486466334709319, 0.582596192755382]),
        #     np.array([-0.7193406015171879, 0.7030311349809253, 1.7279603680764843, -1.462167062832389, -0.7095803065350061, 1.3887629938557762, 1.685751010149957]),
        #     np.array([0.031114206971786906, 0.280148668136432, 0.1660746720242674, -1.7795346119093702, 0.5708601494733205, 1.69713915906316, 0.18080576717360675]),
        #     np.array([0.5315303232882735, -0.6880944614953183, -1.295222514333675, -2.7098521010502994, -0.8629544292895819, 2.304309252112395, 0.7451632836879227]),
        #     np.array([0.031114206971786906, 0.280148668136432, 0.1660746720242674, -1.7795346119093702, 0.5708601494733205, 1.69713915906316, 0.18080576717360675]),
        #     np.array([0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])
        # ]
        self.path = [
            # np.array([-1.45, -0.15, 0.96, -2.64, 0.29, 2.54, 0.10]),
            np.array([0.682, 0.025, -0.038, -2.118, -0.002, 2.145, 1.507]),
            np.array([0.579, 0.303, 0.062, -2.237, -0.038, 2.542, 1.533]),
            np.array([-0.456, 0.084, 0.498, -1.988, -0.042, 2.065, 0.858]), # np.array([-0.982, 0.356, 1.240, -2.018, -0.394, 2.105, 1.231]),
            np.array([-0.542, -0.626, 0.055, -2.502, 0.831, 2.165, 0.816]),
            np.array([-0.261, -0.068, -0.534, -2.334, -0.117, 2.260, -0.060]),
        ]
        self.extended_path = self.load_extended_path(alpha=self.alpha)        # Set logger parameters# REMOVE THE FIRST WAYPOINT (self.extended_path[0]) IF IT IS NOT A STARTING POINT (start_joint_config)
        self.extended_path = self.extended_path[1:]  # Remove the first waypoint if it's not a starting point

        self.logger.final_joint_target = self.extended_path[-1]  # needed for end condition
        self.logger.position_threshold = 0.02  # radians, tune this if needed

    def load_extended_path(self, alpha: int) -> np.ndarray:
        path_file = f"/home/marcin/panda_trajectory_planning/catkin_ws/src/klemol_planner/klemol_planner/tests/test_splines/extended_paths/extended_path_alpha_{alpha}.npy"
        return np.load(path_file, allow_pickle=True)

    def run_test(self, mode: str = "raw"):

        # COMMENT OUT TO DISABLE EXTENDED PATHS
        self.path = self.extended_path

        rospy.loginfo(f"Running test in mode: {mode}")
        self.logger.set_mode(mode)
        self.logger.final_joint_target = self.path[-1]

        post_processing = PathPostProcessing(self.collision_checker)

        if mode == "raw":
            # Raw movement should not be logged as a spline trajectory
            for idx, config in enumerate(self.path):
                self.robot_model.execute_joint_positions(config, "Raw")
                print(f"RAW MODE - moving to {idx}/{len(self.path)}")
            return

        # --------- Start PLANNING timer ---------
        self.logger.start_planning_timer()

        if mode == "spline_cubic_hermite":
            traj = post_processing.generate_cubic_trajectory(
                path=self.path,
                joint_names=self.robot_model.group.get_active_joints(),
                velocity_limits=self.robot_model.velocity_limits,
                acceleration_limits=self.robot_model.acceleration_limits,
                max_vel_acc_multiplier=0.25
            )

        elif mode == "spline_quintic_bsplines":
            traj = post_processing.generate_quintic_bspline_trajectory(
                path=self.path,
                joint_names=self.robot_model.group.get_active_joints(),
                velocity_limits=self.robot_model.velocity_limits,
                acceleration_limits=self.robot_model.acceleration_limits,
                max_vel_acc_multiplier=0.25
            )

        elif mode == "spline_quintic_polynomial":
            traj = post_processing.generate_quintic_polynomial_trajectory(
                path=self.path,
                joint_names=self.robot_model.group.get_active_joints(),
                velocity_limits=self.robot_model.velocity_limits,
                acceleration_limits=self.robot_model.acceleration_limits,
                max_vel_acc_multiplier=0.25
            )

        else:
            rospy.logwarn(f"Unsupported mode: {mode}")
            return

        # --------- End PLANNING timer ---------
        self.logger.stop_planning_timer()

        
        # --------- Start RECORDING ---------
        self.logger.start_recording()

        # --------- Send trajectory ---------
        self.robot_model.send_trajectory_to_controller(traj)

        # Let logger decide when to stop based on velocity threshold
        rospy.loginfo("Waiting for logger to auto-stop after execution...")
        while self.logger.recording_active:
            rospy.sleep(0.1)

        rospy.sleep(0.5)  # Let robot settle