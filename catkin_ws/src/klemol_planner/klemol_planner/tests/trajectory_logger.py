import rospy
import numpy as np
from control_msgs.msg import JointTrajectoryControllerState


class TrajectoryLogger:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.results = []

        self.reset()

        self.sub = rospy.Subscriber(
            "/effort_joint_trajectory_controller/state",
            JointTrajectoryControllerState,
            self.callback,
        )

    def reset(self):
        self.time = []
        self.positions = []
        self.velocities = []
        self.ee_positions = []

        self.absolute_start_time = rospy.Time.now().to_sec()
        self.planning_start_time = None
        self.planning_end_time = None
        self.execution_start_time = None
        self.execution_end_time = None

        self.recording_active = False
        self.movement_started = False

        self.current_mode = "undefined"
        self.final_joint_target = None
        self.position_threshold = 0.015  # radians
        self.velocity_threshold = 0.05   # rad/s

    def set_mode(self, mode: str):
        self.current_mode = mode

    def start_planning_timer(self):
        self.planning_start_time = rospy.Time.now().to_sec()

    def stop_planning_timer(self):
        self.planning_end_time = rospy.Time.now().to_sec()

    def start_recording(self):
        rospy.loginfo(f"[{self.current_mode}] Start recording")
        self.execution_start_time = rospy.Time.now().to_sec()
        self.recording_active = True
        self.movement_started = False

    def stop_recording(self):
        rospy.loginfo(f"[{self.current_mode}] Stop recording")
        self.execution_end_time = rospy.Time.now().to_sec()
        self.recording_active = False
        self._finalize_segment()

    def callback(self, msg: JointTrajectoryControllerState):
        if not self.recording_active:
            return

        current_time = msg.header.stamp.to_sec()
        elapsed = current_time - self.absolute_start_time

        joints = np.array(msg.actual.positions[:7])
        ee_pose = self.robot_model.fk(joints)

        self.time.append(elapsed)
        self.positions.append(list(joints))
        self.velocities.append(list(msg.actual.velocities[:7]))
        self.ee_positions.append([ee_pose.x, ee_pose.y, ee_pose.z])

        # Detect motion start
        if not self.movement_started and any(abs(v) > self.velocity_threshold for v in msg.actual.velocities[:7]):
            self.movement_started = True
            rospy.loginfo(f"[{self.current_mode}] Movement started")

        # Detect end of motion
        if self.movement_started:
            last_vels = np.array(self.velocities[-20:])
            if last_vels.shape[0] >= 10:
                avg_speed = np.mean(np.abs(last_vels))
                current_pos = np.array(self.positions[-1])
                joint_error = np.linalg.norm(current_pos - self.final_joint_target)

                if avg_speed < self.velocity_threshold and joint_error < self.position_threshold:
                    rospy.loginfo(f"[{self.current_mode}] Auto-stop: avg_speed={avg_speed:.4f}, joint_error={joint_error:.4f}")
                    self.stop_recording()

    def _finalize_segment(self):
        planning_time = (
            self.planning_end_time - self.planning_start_time
            if self.planning_start_time is not None and self.planning_end_time is not None
            else None
        )
        execution_time = (
            self.execution_end_time - self.execution_start_time
            if self.execution_start_time is not None and self.execution_end_time is not None
            else None
        )

        self.results.append({
            "planner": self.current_mode,
            "time": np.array(self.time),
            "pos": np.array(self.positions),
            "vel": np.array(self.velocities),
            "ee": np.array(self.ee_positions),
            "planning_time": planning_time,
            "execution_time": execution_time
        })

        # Prepare for next segment
        self.reset()

    def save(self, filename: str):
        np.savez(filename, results=np.array(self.results, dtype=object))
        rospy.loginfo(f"Saved {len(self.results)} trajectory segments to: {filename}")
