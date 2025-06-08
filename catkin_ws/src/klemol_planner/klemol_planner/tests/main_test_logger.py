import rospy
import numpy as np
from control_msgs.msg import JointTrajectoryControllerState
# from klemol_planner.environment.robot_model import RobotModel # Circular import


class MainTestLogger:
    def __init__(self, robot_model):
        self.active = False # When true logger is recording (callback is active)
        self.results = []
        self.robot_model = robot_model
        self.current_loop_index = 0
        self.reset()

    def reset(self):
        # Reset all logged data for a new test run
        self.planning_successful = True # If the test run was successful, set to False in case of too long planning time etc.
        self.planning_time = None # Only the time when we are planning
        self.spline_fitting_start_time = None # Start time of spline fitting
        self.spline_fitting_time = None # Only the time when we are fitting the spline
        self.execution_time = None # Only the time when we are moving
        self.number_of_steps = None # 
        self.number_of_waypoints_before_post_processing = None # Number of waypoints before any post-processing
        self.joint_travel_distances = [0.0 for _ in range(7)] # Each joint's total travel distance
        self.cartesian_path_length = None # Total length of the Cartesian path traveled by the end-effector

        # Real-time tracking during execution
        self.start_time = None
        self.movement_started = False
        self.time_stamps = []
        self.joint_positions = [] # List of joint positions for each time (list of 7 elemnt lists)
        self.ee_positions = [] # List of end-effector positions for each time (list of 3 elemnt lists)
        self.velocities = []

        self.current_planner = "undefined"
        self.active = False

    def activate(self):
        if not self.active:
            rospy.loginfo("Activating MainTestLogger")
            self.active = True
        else:
            rospy.logwarn("MainTestLogger is already active")

    def deactivate(self):
        if self.active:
            rospy.loginfo("Deactivating MainTestLogger")
            self.active = False
        else:
            rospy.logwarn("MainTestLogger is already inactive")

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
        if self.active:
            # current_time = rospy.Time.now().to_sec()
            current_time = msg.header.stamp.to_sec() # USE SIM TIME

            if self.start_time is None:
                # self.start_time = current_time  # Fallback if not explicitly started
                rospy.logwarn("start_time not explicitly set, falling back to first sim timestamp.") # USE SIM TIME
                self.start_time = current_time                                                       # USE SIM TIME

            elapsed_time = current_time - self.start_time
            self.time_stamps.append(elapsed_time)
            self.joint_positions.append(list(msg.actual.positions[:7]))  # Only joints 1-7
            
            ee_pose = self.robot_model.fk(np.array(self.joint_positions[-1]))
            ee_list_to_append = [ee_pose.x, ee_pose.y, ee_pose.z]
            self.ee_positions.append(ee_list_to_append)

            self.velocities.append(list(msg.actual.velocities[:7]))

            # Detect when movement starts (if any joint velocity exceeds threshold)
            if not self.movement_started:
                if any(abs(v) > 1e-2 for v in msg.actual.velocities[:7]):
                    self.movement_started = True
                    self.movement_start_time = elapsed_time
        # else:
        #     rospy.logwarn("MainTestLogger is not active. Callback data will not be recorded.")

    def compute_metrics(self):
        if not self.joint_positions:
            return

        joint_positions_array = np.array(self.joint_positions) # 
        ee_positions_array = np.array(self.ee_positions)
        joint_diffs = np.diff(joint_positions_array, axis=0)
        joint_distances = np.sum(np.abs(joint_diffs), axis=0)
        ee_diffs = np.diff(ee_positions_array, axis=0)
        ee_distances = np.linalg.norm(ee_diffs, axis=1)
        self.joint_travel_distances = joint_distances.tolist()
        self.number_of_steps = len(self.joint_positions)

        self.cartesian_path_length = np.sum(ee_distances) if len(ee_distances) > 0 else 0.0


        if self.movement_started:
            self.execution_time = self.time_stamps[-1] - self.movement_start_time
        else:
            self.execution_time = None

        if self.spline_fitting_time is not None and self.spline_fitting_start_time is not None:
            self.spline_fitting_time -= self.spline_fitting_start_time
        else:
            self.spline_fitting_time = None

        # Append results
        self.results.append({
            "planner": self.current_planner,
            "loop_index": self.current_loop_index,
            "planning_successful": self.planning_successful,
            "planning_time": self.planning_time,
            "spline_fitting_time": self.spline_fitting_time,
            "execution_time": self.execution_time,
            "number_of_steps": self.number_of_steps,
            "number_of_waypoints_before_post_processing": self.number_of_waypoints_before_post_processing,
            "joint_travel_distances": self.joint_travel_distances,
            "cartesain_positions": self.ee_positions,
            "cartesian_path_length": self.cartesian_path_length,
            "time_stamps": self.time_stamps,
            "positions": self.joint_positions,
            "velocities": self.velocities
        })

    def save(self, filename: str):
        np.savez(filename, results=self.results)