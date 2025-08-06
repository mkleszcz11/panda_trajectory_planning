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
from scipy.interpolate import make_interp_spline, CubicHermiteSpline, BSpline


from scipy.interpolate import BPoly

import rospy
import math

class PathPostProcessing:
    """
    Class to shortcut a path.
    """
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker

    def mock_interpolate(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        dt: float = 0.005,
        segment_time: float = 1.0,
    ) -> JointTrajectory:
        """
        Simple mock interpolator: piecewise linear segments with stop at each waypoint.

        Args:
            path: List of joint configurations.
            joint_names: Joint names.
            velocity_limits: Max joint velocities (not used here).
            acceleration_limits: Max joint accelerations (not used here).
            dt: Sampling interval (not used here, as we stop at waypoints).
            segment_time: Time between consecutive waypoints.

        Returns:
            JointTrajectory message.
        """
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = Header()
        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.3)

        for i, config in enumerate(path):
            point = JointTrajectoryPoint()
            point.positions = config.tolist()
            point.velocities = [0.0] * len(config)
            point.accelerations = [0.0] * len(config)
            point.time_from_start = rospy.Duration.from_sec(i * segment_time)
            traj_msg.points.append(point)

        return traj_msg

    def generate_quintic_bspline_trajectory(
            self,
            path: t.List[np.ndarray],
            joint_names: t.List[str],
            velocity_limits: np.ndarray,
            acceleration_limits: np.ndarray,
            dt: float = 0.005,
            max_vel_acc_multiplier: float = 0.1,
            duration: float = None
        ) -> JointTrajectory:
        """
        Approximates joint-space trajectory using a quintic B-spline (order 5, degree 5).
        This is NOT interpolating; the path is approximated using control points.

        Args:
            path: List of joint configurations (control points).
            joint_names: Joint names.
            velocity_limits: Max joint velocities.
            acceleration_limits: Max joint accelerations.
            dt: Sampling interval.
            max_vel_acc_multiplier: Multiplier to reduce speed/acceleration.
            duration: Optional. Force the trajectory to last exactly this long [s].

        Returns:
            JointTrajectory message.
        """
        joint_reader = JointStatesReader(joint_names)

        while joint_reader.latest_state is None and not rospy.is_shutdown():
            rospy.sleep(0.05)

        q_current = joint_reader.get_current_positions()
        q = np.array([q_current]*3 + list(path) + [path[-1]]*2)

        n_ctrl_points = q.shape[0]
        n_joints = q.shape[1]
        degree = 5
        k = degree

        if n_ctrl_points <= k:
            raise ValueError(f"Need at least {k+1} control points for quintic B-spline (got {n_ctrl_points})")

        # --- Knot vector ---
        knots = np.concatenate((
            np.zeros(k),
            np.linspace(0, 1, n_ctrl_points - k + 1),
            np.ones(k)
        ))

        splines = [BSpline(knots, q[:, j], k) for j in range(n_joints)]

        # --- Time allocation ---
        if duration is not None:
            total_time = duration
        else:
            total_time = 0.0
            v_max = velocity_limits * max_vel_acc_multiplier
            a_max = acceleration_limits * max_vel_acc_multiplier
            for i in range(1, len(q)):
                delta_q = np.abs(q[i] - q[i - 1])
                t_required = self._calculate_min_time_linear_movement(delta_q, v_max, a_max)
                total_time += max(t_required, 0.1)

        t_samples = np.arange(0, total_time + dt, dt)
        u_samples = np.linspace(0, 1, len(t_samples))  # parametric domain
        scale = 1.0 / total_time

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = Header()

        for i, (t, u) in enumerate(zip(t_samples, u_samples)):
            point = JointTrajectoryPoint()
            point.positions = [s(u) for s in splines]
            point.velocities = [s.derivative(1)(u) * scale for s in splines]
            point.accelerations = [s.derivative(2)(u) * scale**2 for s in splines]
            point.time_from_start = rospy.Duration.from_sec(t)
            traj_msg.points.append(point)

        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.3)
        return traj_msg

    def generate_quintic_polynomial_trajectory(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        dt: float = 0.005,
        max_vel_acc_multiplier: float = 0.1,
        duration: float = None
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
        q = np.array([q_current] + list(path)) # TODO REMOVE q_current if real robot is not working
        n_waypoints = len(q)
        n_joints = len(joint_names)

        # --- Time allocation ---
        times = [0.0]
        v_max = velocity_limits * max_vel_acc_multiplier
        a_max = acceleration_limits * max_vel_acc_multiplier

        for i in range(1, n_waypoints):
            delta_q = np.abs(q[i] - q[i - 1])

            if delta_q.max() < 1e-6:
                # If the joint positions are very close, skip this waypoint
                times.append(times[-1] + 0.1)
                rospy.logwarn(f"Skipping waypoint {i} due to negligible joint movement.")
                continue

            t_required = self._calculate_min_time_linear_movement(delta_q=delta_q,
                                                                  v_max = v_max,
                                                                  a_max = a_max) #np.max(np.maximum(t_vel, t_acc))
            times.append(times[-1] + max(t_required, 0.1))

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

        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(0.3) #TODO Increase this in case of weird robot noises

        return traj_msg


    def generate_cubic_trajectory(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        acceleration_limits: np.ndarray,
        dt: float = 0.005,
        max_vel_acc_multiplier: float = 0.1
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
        v_max = velocity_limits * max_vel_acc_multiplier
        a_max = acceleration_limits * max_vel_acc_multiplier

        for i in range(1, n_waypoints):
            delta_q = np.abs(q[i] - q[i - 1])

            if delta_q.max() < 1e-6:
                # If the joint positions are very close, skip this waypoint
                times.append(0.1)
                rospy.logwarn(f"Skipping waypoint {i} due to negligible joint movement.")
                continue

            t_required = self._calculate_min_time_linear_movement(delta_q=delta_q,
                                                                  v_max = v_max,
                                                                  a_max = a_max) #np.max(np.maximum(t_vel, t_acc))
            times.append(times[-1] + max(t_required, 0.1))


        times = np.array(times)

        # --- Estimate velocities ---
        vel = np.zeros_like(q)
        vel[0] = np.clip(qdot_current, -velocity_limits * max_vel_acc_multiplier, velocity_limits * max_vel_acc_multiplier)

        for i in range(1, n_waypoints - 1):
            dt1 = times[i] - times[i - 1]
            dt2 = times[i + 1] - times[i]
            vel[i] = (q[i + 1] - q[i - 1]) / (dt1 + dt2)

        vel[-1] = np.zeros(n_joints) # Final velocity (finite difference)

        # Clamp intermediate velocities
        for i in range(1, n_waypoints):
            vel[i] = np.clip(vel[i], -velocity_limits * max_vel_acc_multiplier, velocity_limits * max_vel_acc_multiplier)

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

        traj_msg.header.stamp = rospy.Time.now() #+ rospy.Duration(0.3)

        # print("Trajectory starts at:", traj_msg.header.stamp.to_sec())
        # print("First point time_from_start:", traj_msg.points[0].time_from_start.to_sec())
        # print("ROS now:", rospy.Time.now().to_sec())
        return traj_msg

    # SIMPLIFIED
    def _calculate_min_time_linear_movement(self,
                                            delta_q: np.ndarray,
                                            v_max: np.ndarray,
                                            a_max: np.ndarray) -> float:
        """
        Compute minimum duration to move through delta_q, accounting for acceleration + cruise.

        Returns:
            Maximum joint-wise time required to complete the motion safely.
        """
        times_required = np.zeros(7)

        for joint in range(7):
            dq = abs(delta_q[joint])
            vmax = v_max[joint]
            amax = a_max[joint]

            # Velocity we could reach if only accelerating
            v_possible = math.sqrt(2 * amax * dq)

            if v_possible <= vmax:
                # Triangular profile (accelerate then decelerate)
                t_required = math.sqrt(2 * dq / amax)
            else:
                # Trapezoidal profile (accel + cruise + decel)
                t_accel = vmax / amax
                s_accel = 0.5 * amax * t_accel ** 2
                s_cruise = dq - 2 * s_accel
                t_cruise = s_cruise / vmax if s_cruise > 0 else 0.0
                t_required = 2 * t_accel + t_cruise

            times_required[joint] = t_required

        return np.max(times_required)

        # Considering the longest time, what would be the final speed and acceleration for every joint
        # TODO

        # t_required - float
        # end_v - numpy array (7 elements)
        # end_acc - numpy array (7 elements)

        # print(f"TIMES REQUIRED: {times_required}")
        t_required = times_required.max()
        return t_required


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
        # print("GENERATING A SHORTCUTTED PATH")
        # print(f"Original path length: {len(path)}")
        i = 0
        new_path = [path[0]]

        while i < len(path) - 1:
            found = False
            for j in range(len(path) - 1, i, -1):
                if self.collision_checker.is_collision_free(path[i], path[j]):
                    new_path.append(path[j])
                    i = j
                    found = True
                    break
            if not found:
                new_path.append(path[i + 1])
                i += 1
        # print(f"Shortcutted path length: {len(new_path)}")
        return new_path
