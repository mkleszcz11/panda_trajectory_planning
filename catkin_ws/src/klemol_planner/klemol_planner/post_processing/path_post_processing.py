import numpy as np
import typing as t

from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.environment.robot_joint_states_reader import JointStatesReader
from klemol_planner.environment.robot_joint_states_reader import JointStatesReader
import trajectory_msgs.msg
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import scipy.interpolate
from std_msgs.msg import Header
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicHermiteSpline

import rospy
import math

class PathPostProcessing:
    """
    Class to shortcut a path.
    """
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker

    def interpolate_quintic_trajectory(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        dt: float = 0.005,
        max_vel_acc_multiplier: float = 0.1
    ) -> JointTrajectory:
        """
        True quintic spline interpolation for joint-space trajectory.

        Estimates position, velocity, and acceleration at waypoints, and interpolates
        with 5th-order splines.

        Args:
            path: List of joint configurations.
            joint_names: Joint names.
            velocity_limits: Max joint velocities.
            get_current_joint_values: Callable to get current joint config.
            dt: Sampling interval.
            safety_margin: Multiplier <1.0 to stay below limits.

        Returns:
            JointTrajectory message.
        """
        joint_reader = JointStatesReader(joint_names)

        # Wait until the first message arrives
        while joint_reader.latest_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)


        q_current = joint_reader.get_current_positions()
        qdot_current = joint_reader.get_current_velocities()

        # Combine initial position with path
        q = np.array(list(path))
        n_waypoints = len(q)
        n_joints = len(joint_names)

        # --- Time allocation ---
        times = [0.0]
        v_max = velocity_limits * max_vel_acc_multiplier
        a_max = acceleration_limits * max_vel_acc_multiplier

        for i in range(1, n_waypoints):
            delta_q = np.abs(q[i] - q[i - 1])
            t_required = self._calculate_min_time_linear_movement(delta_q=delta_q,
                                                                  v_max = v_max,
                                                                  a_max = a_max) #np.max(np.maximum(t_vel, t_acc))
            times.append(times[-1] + max(t_required, 0.4))

        times = np.array(times)
        print(f"======== TIMES TIMES TIMES ========\n{times}\n======== TIMES TIMES TIMES ========")

        splines = []
        for j in range(n_joints):
            spline = make_interp_spline(
                times,
                q[:, j],
                k=5,
                bc_type=(
                    [(1, 0.0), (2, 0.0)],
                    [(1, 0.0), (2, 0.0)]
                )
            )
            splines.append(spline)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = Header()

        # # Sample from 0.1s onward
        # # start_time_offset = 0.1
        # times = 5
        t_samples = np.arange(0, times[-1] + dt, dt)

        for t_val in t_samples:
            point = JointTrajectoryPoint()
            point.positions = [spline(t_val) for spline in splines]
            point.velocities = [spline.derivative(1)(t_val) for spline in splines]
            point.accelerations = [spline.derivative(2)(t_val) for spline in splines]
            point.time_from_start = rospy.Duration.from_sec(t_val)
            traj_msg.points.append(point)

        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.5)

        return traj_msg


    # SIMPLIFIED
    def _calculate_min_time_linear_movement(self,
                                       delta_q: np.ndarray,
                                       v_max: np.ndarray,
                                       a_max: np.ndarray) -> float: #t.Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate the minimum time required to move delta_q (float - one joint at the time),
        considering joint velocity and acceleration limits. Minimum required time is defined
        in the following way:
        1. Calculate the minimum moving time for each joint
        2. Get the max out of the min times.

        Acceleration limits are in the robot model itself (each joint has a different limits).

        Note:
            Equation to find a time in linear motion with vel and acc has 2 solutions
            We know vel and acc will be positive, therefore we should always add a square root,
            thus taking a solution "from the future"

        Returns:
            time: time required for a movement
            velocities: velocities at the end of movement
            accelerations: accelerations at the end of movement
        """
        # Check if we will reach a max speed while moving between waypoints.
        print(f"A MAX => {a_max} | DELTA Q = {delta_q}")

        times_required = np.zeros(7)

        for joint in range(7):
            possible_velocity = math.sqrt(2 * a_max[joint] * delta_q[joint])

            # If v_possible <= v_max we will be accelerating for the entire movement duration, calculate this time.
            if possible_velocity <= v_max[joint]:
                t_required = math.sqrt((2 * a_max[joint]) / (delta_q[joint]))
            else:
                # Check how much time does it take to accelerate to the max speed
                t_first_part = v_max[joint] / a_max[joint]

                # Check how much way does it take to accelerate to the max speed
                s_first_part = (a_max[joint] * t_first_part * t_first_part) / 2.0

                # Check how much time does it take to move the remaining distance with the max speed
                s_second_part = delta_q[joint] - s_first_part
                t_second_part = s_second_part / v_max[joint]

                times_required[joint] = t_first_part + t_second_part


        # Considering the longest time, what would be the final speed and acceleration for every joint
        # TODO

        # t_required - float
        # end_v - numpy array (7 elements)
        # end_acc - numpy array (7 elements)

        print(f"TIMES REQUIRED: {times_required}")
        t_required = times_required.max()
        return t_required


    # # SIMPLIFIED
    # def _calculate_min_time_linear_movement(self,
    #                                    delta_q: float,
    #                                    v_max: float,
    #                                    a_max: float) -> float: #t.Tuple[float, np.ndarray, np.ndarray]:
    #     """
    #     Calculate the minimum time required to move delta_q (float - one joint at the time),
    #     considering joint velocity and acceleration limits. Minimum required time is defined
    #     in the following way:
    #     1. Calculate the minimum moving time for each joint
    #     2. Get the max out of the min times.

    #     Acceleration limits are in the robot model itself (each joint has a different limits).

    #     Note:
    #         Equation to find a time in linear motion with vel and acc has 2 solutions
    #         We know vel and acc will be positive, therefore we should always add a square root,
    #         thus taking a solution "from the future"

    #     Returns:
    #         time: time required for a movement
    #         velocities: velocities at the end of movement
    #         accelerations: accelerations at the end of movement
    #     """
    #     # Check if we will reach a max speed while moving between waypoints.
    #     print(f"A MAX => {a_max} | DELTA Q = {delta_q}")
    #     possible_velocity = math.sqrt(2 * a_max * delta_q)

    #     # If v_possible <= v_max we will be accelerating for the entire movement duration, calculate this time.
    #     if possible_velocity <= v_max:
    #         t_required = math.sqrt((2 * a_max) / (delta_q))
    #     else:
    #         # Check how much time does it take to accelerate to the max speed
    #         t_first_part = v_max / a_max

    #         # Check how much way does it take to accelerate to the max speed
    #         s_first_part = (a_max * t_first_part * t_first_part) / 2

    #         # Check how much time does it take to move the remaining distance with the max speed
    #         s_second_part = delta_q - s_first_part
    #         t_second_part = s_second_part / v_max

    #         t_required = t_first_part + t_second_part

    #         # # Get end velocities and accelerations
    #         # velocities[i] = 
    #         # accelerations[i] = 


    #     # Considering the longest time, what would be the final speed and acceleration for every joint
    #     # TODO

    #     # t_required - float
    #     # end_v - numpy array (7 elements)
    #     # end_acc - numpy array (7 elements)
    #     return t_required

    # def _calculate_min_time_linear_movement(self,
    #                                    delta_q: np.ndarray,
    #                                    start_v: np.ndarray,
    #                                    v_max: np.ndarray,
    #                                    a_max: np.ndarray) -> t.Tuple[float, np.ndarray, np.ndarray]:
    #     """
    #     Calculate the minimum time required to move delta_q (joint state),
    #     considering joint velocity and acceleration limits. Minimum required time is defined
    #     in the following way:
    #     1. Calculate the minimum moving time for each joint
    #     2. Get the max out of the min times.

    #     Acceleration limits are in the robot model itself (each joint has a different limits).

    #     Note:
    #         Equation to find a time in linear motion with vel and acc has 2 solutions
    #         We know vel and acc will be positive, therefore we should always add a square root,
    #         thus taking a solution "from the future"

    #     Returns:
    #         time: time required for a movement
    #         velocities: velocities at the end of movement
    #         accelerations: accelerations at the end of movement
    #     """
    #     times_required = np.zeros(7) # Min time for every joint
    #     velocities = np.zeros(7) # Velocities at the end of movement
    #     accelerations = np.zeros(7) # Accelerations at the end of movement

    #     # Find the min required time for every joint.
    #     for i in range(7):
    #         # Check if we will reach a max speed while moving between waypoints.
    #         possible_velocity = math.sqrt(start_v[i] + 2 * a_max[i] * delta_q[i])

    #         # If v_possible <= v_max we will be accelerating for the entire movement duration, calculate this time.
    #         if possible_velocity <= v_max:
    #             times_required[i] = 
    #             velocities = 
    #             accelerations = 
    #         else:
    #             # Check how much time does it take to accelerate to the max speed
    #             t_first_part = 

    #             # Check how much way does it take to accelerate to the max speed
    #             s_first_part = 

    #             # Check how much time does it take to move the remaining distance with the max speed
    #             t_second_part = 

    #             times_required[i] =  = t_first_part + t_second_part

    #             # Get end velocities and accelerations
    #             velocities[i] = 
    #             accelerations[i] = 

    #     # Get the longest time:
    #     t_required = times_required.max()

    #     # Considering the longest time, what would be the final speed and acceleration for every joint
    #     # TODO

    #     # t_required - float
    #     # end_v - numpy array (7 elements)
    #     # end_acc - numpy array (7 elements)
    #     return min_time, times_required, velocities, accelerations


    def interpolate_trajectory_with_cubic_hermite_splines(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        dt: float = 0.005,
        safety_margin: float = 0.9
    ) -> JointTrajectory:
        """
        Time-parameterized Cubic spline interpolation with velocity constraints,
        using smooth estimated velocities at waypoints.

        Args:
            path: List of joint configurations.
            joint_names: List of joint names.
            velocity_limits: Max joint velocities (rad/s).
            get_current_joint_values: Callable that returns current joint config.
            effort_limits: Not used yet.
            dt: Sampling interval.
            safety_margin: Fraction of max velocity to stay under (e.g., 0.9).

        Returns:
            JointTrajectory message.
        """
        joint_reader = JointStatesReader(joint_names)

        # Wait until state received
        while joint_reader.latest_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)

        q_current = joint_reader.get_current_positions()
        qdot_current = joint_reader.get_current_velocities()

        # Combine with path
        q = np.array([q_current] + list(path))
        n_waypoints = len(q)
        n_joints = len(joint_names)

        # --- Time allocation ---
        times = [0.0]
        for i in range(1, n_waypoints):
            delta_q = np.abs(q[i] - q[i - 1])
            t_required = np.max(delta_q / (velocity_limits * safety_margin))
            times.append(times[-1] + max(t_required, 0.4))  # min segment time
        times = np.array(times)

        # --- Estimate velocities ---
        vel = np.zeros_like(q)
        vel[0] = np.clip(qdot_current, -velocity_limits * safety_margin, velocity_limits * safety_margin)

        for i in range(1, n_waypoints - 1):
            dt1 = times[i] - times[i - 1]
            dt2 = times[i + 1] - times[i]
            vel[i] = (q[i + 1] - q[i - 1]) / (dt1 + dt2)

        vel[-1] = np.zeros(n_joints) # Final velocity (finite difference)

        # Clamp intermediate velocities
        for i in range(1, n_waypoints):
            vel[i] = np.clip(vel[i], -velocity_limits * safety_margin, velocity_limits * safety_margin)

        # --- Fit cubic Hermite splines per joint ---
        splines = [
            CubicHermiteSpline(times, q[:, j], vel[:, j])
            for j in range(n_joints)
        ]

        # --- Sample the trajectory ---
        start_time_offset = 0.1
        t_samples = np.arange(start_time_offset, times[-1] + dt, dt)

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = Header()

        for t_val in t_samples:
            point = JointTrajectoryPoint()
            point.positions = [spline(t_val) for spline in splines]
            point.velocities = [spline.derivative()(t_val) for spline in splines]
            point.accelerations = [spline.derivative(nu=2)(t_val) for spline in splines]
            point.time_from_start = rospy.Duration.from_sec(t_val)
            traj_msg.points.append(point)

        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.5)

        print("Trajectory starts at:", traj_msg.header.stamp.to_sec())
        print("First point time_from_start:", traj_msg.points[0].time_from_start.to_sec())
        print("ROS now:", rospy.Time.now().to_sec())
        return traj_msg

    def interpolate_trajectory_time_parameterised(self, path: t.List[np.ndarray], joint_names: t.List[str], dt: float = 1.05) -> JointTrajectory:
        """
        Create a time-parameterized trajectory using quintic interpolation between waypoints.

        Args:
            path: List of joint configurations (7-DOF).
            joint_names: List of joint names in correct order.
            dt: Time step between interpolated points [s].

        Returns:
            JointTrajectory message with positions, velocities, and accelerations.
        """
        n_joints = len(joint_names)
        n_points = len(path)
        times = np.linspace(0, (n_points - 1) * dt, n_points)

        # Prepare interpolation for each joint
        q = np.array(path).T  # shape: (n_joints, n_points)
        splines = [scipy.interpolate.CubicHermiteSpline(times, qj, np.zeros_like(times)) for qj in q]

        # Sample at fine resolution
        t_samples = np.arange(0, times[-1] + dt, dt)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = std_msgs.msg.Header()
        traj_msg.header.stamp = rospy.Time.now()

        for t in t_samples:
            point = JointTrajectoryPoint()
            point.positions = [spline(t) for spline in splines]
            point.velocities = [spline.derivative()(t) for spline in splines]
            point.accelerations = [spline.derivative(nu=2)(t) for spline in splines]
            point.time_from_start = rospy.Duration.from_sec(float(t))
            traj_msg.points.append(point)

        return traj_msg

    def interpolate_linear(self, config_a: np.ndarray, config_b: np.ndarray, num_points: int = 10) -> t.List[np.ndarray]:
        """
        Linearly interpolate between two configurations.

        Args:
            config_a: Start joint configuration.
            config_b: End joint configuration.
            num_points: Number of intermediate samples.

        Returns:
            List of interpolated joint configurations.
        """
        return [config_a + (config_b - config_a) * float(i) / (num_points - 1) for i in range(num_points)]

    def is_collision_free(self, start: np.ndarray, end: np.ndarray, num_samples: int = 10) -> bool:
        """
        Check if straight-line interpolation between two joint configurations is collision-free.

        Args:
            start: Start configuration.
            end: End configuration.
            num_samples: Number of interpolation points.

        Returns:
            True if all interpolated points are collision free.
        """
        for point in self.interpolate_linear(start, end, num_samples):
            if self.collision_checker.is_in_collision(point):
                return False
        return True

    def generate_a_shortcutted_path(self, path: t.List[np.ndarray]) -> t.List[np.ndarray]:
        """
        Analyze a path and shortcut it as much as possible using straight-line segments.

        The shortcutting logic checks if intermediate points can be skipped without collision.

        Args:
            path: List of joint configurations representing the original path.

        Returns:
            List of joint configurations forming a shorter, still collision-free path.
        """
        if not path:
            return []
        print("GENERATING A SHORTCUTTED PATH")
        print(f"Original path length: {len(path)}")
        i = 0
        new_path = [path[0]]

        while i < len(path) - 1:
            found = False
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(path[i], path[j]):
                    new_path.append(path[j])
                    i = j
                    found = True
                    break
            if not found:
                new_path.append(path[i + 1])
                i += 1
        print(f"Shortcutted path length: {len(new_path)}")
        return new_path
