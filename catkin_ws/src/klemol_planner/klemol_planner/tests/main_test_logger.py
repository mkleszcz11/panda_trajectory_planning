import rospy
import numpy as np
from control_msgs.msg import JointTrajectoryControllerState


class MainTestLogger:
    def __init__(self):
        self.results = []
        self.reset()

    def reset(self):
        # Reset all logged data for a new test run
        self.planning_time = None
        self.execution_time = None
        self.number_of_steps = None
        self.joint_travel_distances = [0.0 for _ in range(7)]
        self.cartesian_path_length = None

        # Real-time tracking during execution
        self.start_time = None
        self.movement_started = False
        self.time_stamps = []
        self.positions = []
        self.velocities = []

        self.current_planner = "undefined"

    def set_planner(self, planner_name: str):
        self.current_planner = planner_name

    def start_timer(self):
        self.start_time = rospy.Time.now().to_sec()

    def stop_timer(self):
        if self.start_time is not None:
            return rospy.Time.now().to_sec() - self.start_time
        return None

    def callback(self, msg: JointTrajectoryControllerState):
        # Record data from trajectory controller state
        current_time = rospy.Time.now().to_sec()
        if self.start_time is None:
            self.start_time = current_time  # Fallback if not explicitly started

        elapsed_time = current_time - self.start_time
        self.time_stamps.append(elapsed_time)
        self.positions.append(list(msg.actual.positions[:7]))  # Only joints 1-7
        self.velocities.append(list(msg.actual.velocities[:7]))

        # Detect when movement starts (if any joint velocity exceeds threshold)
        if not self.movement_started:
            if any(abs(v) > 1e-3 for v in msg.actual.velocities[:7]):
                self.movement_started = True
                self.movement_start_time = elapsed_time

    def compute_metrics(self):
        if not self.positions:
            return

        positions_array = np.array(self.positions)
        diffs = np.diff(positions_array, axis=0)
        joint_distances = np.sum(np.abs(diffs), axis=0)
        self.joint_travel_distances = joint_distances.tolist()
        self.number_of_steps = len(self.positions)

        self.cartesian_path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        if self.movement_started:
            self.execution_time = self.time_stamps[-1] - self.movement_start_time
        else:
            self.execution_time = None

        # Append results
        self.results.append({
            "planner": self.current_planner,
            "planning_time": self.planning_time,
            "execution_time": self.execution_time,
            "number_of_steps": self.number_of_steps,
            "joint_travel_distances": self.joint_travel_distances,
            "cartesian_path_length": self.cartesian_path_length,
            "time_stamps": self.time_stamps,
            "positions": self.positions,
            "velocities": self.velocities
        })

    def save(self, filename: str):
        np.savez(filename, results=self.results)