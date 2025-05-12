import numpy as np
import typing as t

from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
import trajectory_msgs.msg
import std_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import scipy.interpolate
from std_msgs.msg import Header

import rospy

class PathPostProcessing:
    """
    Class to shortcut a path.
    """
    def __init__(self, collision_checker: CollisionChecker):
        self.collision_checker = collision_checker

    def interpolate_trajectory_with_constraints(
        self,
        path: t.List[np.ndarray],
        joint_names: t.List[str],
        velocity_limits: np.ndarray,
        get_current_joint_values: t.Callable[[], np.ndarray], 
        effort_limits: t.Optional[np.ndarray] = None,  # placeholder for future extension
        dt: float = 0.01,
        safety_margin: float = 0.9
    ) -> JointTrajectory:
        """
        Time-parameterized quintic spline interpolation with velocity constraints,
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
        # --- Prepend current joint config ---
        q_current = np.array(get_current_joint_values())
        q = np.array([q_current] + list(path))
        n_joints = len(joint_names)

        # --- Estimate time allocation based on velocity limits ---
        times = [0.0]
        for i in range(1, len(q)):
            delta_q = np.abs(q[i] - q[i - 1])
            required_time = np.max(delta_q / (velocity_limits * safety_margin))
            times.append(times[-1] + max(required_time, 0.4))  # min duration
        times = np.array(times)

        # --- Estimate velocities using finite differences ---
        vel = np.zeros_like(q)
        for i in range(1, len(q) - 1):
            dt1 = times[i] - times[i - 1]
            dt2 = times[i + 1] - times[i]
            vel[i] = (q[i + 1] - q[i - 1]) / (dt1 + dt2)
        vel[0] = (q[1] - q[0]) / (times[1] - times[0])
        vel[-1] = (q[-1] - q[-2]) / (times[-1] - times[-2])

        # --- Clamp velocities to limits ---
        for i in range(len(vel)):
            for j in range(n_joints):
                limit = velocity_limits[j] * safety_margin
                if abs(vel[i, j]) > limit:
                    vel[i, j] = np.sign(vel[i, j]) * limit

        # --- Fit splines with estimated velocities ---
        splines = [
            scipy.interpolate.CubicHermiteSpline(times, q[:, j], vel[:, j])
            for j in range(n_joints)
        ]

        # --- Sample the trajectory ---
        t_samples = np.arange(0, times[-1] + dt, dt)
        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header = Header()

        start_delay = 0.5  # seconds
        traj_msg.header.stamp = rospy.Time.now() + rospy.Duration(start_delay)

        for t in t_samples:
            point = JointTrajectoryPoint()
            point.positions = [spline(t) for spline in splines]
            point.velocities = [spline.derivative()(t) for spline in splines]
            point.accelerations = [spline.derivative(nu=2)(t) for spline in splines]
            point.time_from_start = rospy.Duration.from_sec(float(t))
            traj_msg.points.append(point)

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
