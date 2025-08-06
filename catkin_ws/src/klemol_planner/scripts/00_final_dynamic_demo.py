import typing as t
from geometry_msgs.msg import PoseStamped
import rospy
import copy
import moveit_commander
import time

from klemol_planner.planners.rrt import RRTPlanner
from klemol_planner.planners.rrt_star import RRTStarPlanner
from klemol_planner.planners.rrt_with_connecting import RRTWithConnectingPlanner
from klemol_planner.planners.prm import PRMPlanner

from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.post_processing.path_post_processing import PathPostProcessing

from klemol_planner.tests.main_test_logger import MainTestLogger

from klemol_planner.utils.config_loader import load_planner_params

from klemol_planner.camera_utils.kalman_filter import AsynchronousPredictiveKalmanFilter
import numpy as np

class DynamicDemo:
    def __init__(self, planner_name: str, post_processing_method: str, objects_names: t.List[str], include_obstacle: bool = False):
        """
        Demo initialization.

        Args:
            planner_name (str): Name of the planner to use for the demo.
            post_processing_method (str): Name of the post processing method to use.
            objects_names (t.List[str]): List of object names to be used in the demo.
        """
        KALMAN_FILTER_TIME_HORIZON = 5.0

        # List of objects that should be cleaned
        self.objects_to_clean = objects_names

        # Start joint config
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]

        # List of all possible objects the system can clean - names as defined
        # in yolo, must be associated with an aruco code (one in the box)
        # Note: return values from camera operations are corner_XY, where XY is the aruco code number (1, 2, 14, 200, ...)
        self.object_name_to_aruco = {
            "banana": "corner_11",
            "sports ball": "corner_10",
            "scissors": "corner_10",
            "spoon": "corner_10",
            "fork": "corner_10",
            "carrot": "corner_10",
            "knife": "corner_10",
        }

        # Validate if object names are a subset of available names in object_name_to_aruco
        if not set(objects_names).issubset(set(self.object_name_to_aruco.keys())):
            raise ValueError("Some object names are not defined in the object_name_to_aruco dictionary.")

        # Mandatory classes to run the demo
        self.robot_model = Robot()
        self.logger = MainTestLogger(robot_model = self.robot_model)
        self.collision_checker = CollisionChecker(group_name="panda_arm")
        self.post_processing = PathPostProcessing(collision_checker=self.collision_checker)

        # All possible planners available for the demo
        self.available_planners = {
            "rrt": RRTPlanner,
            "rrt_star": RRTStarPlanner,
            "rrt_with_connecting": RRTWithConnectingPlanner,
            "prm": PRMPlanner
        }

        # All possible post processing methods available for the demo
        self.available_post_processing_methods = {
            "qubic_spline": self.post_processing.generate_cubic_trajectory,
            "quintic_polynomial": self.post_processing.generate_quintic_polynomial_trajectory,
            "quintic_bspline": self.post_processing.generate_quintic_bspline_trajectory,
        }
        self.post_processing_method = self.available_post_processing_methods.get(post_processing_method, None)

        # Initialize the selected planner
        if planner_name not in self.available_planners:
            raise ValueError(f"Planner {planner_name} is not available. Choose from {list(self.available_planners.keys())}.")
        algorithm_params = load_planner_params(planner_name)
        self.demo_planner = self.available_planners[planner_name](self.robot_model, self.collision_checker, algorithm_params)

        # Move to the initial position
        self.robot_model.move_to_joint_config(self.start_joint_config)

        # Camera operations and transformations
        self.camera_operations = CameraOperations()
        self.panda_transformations = PandaTransformations(cam_operations=self.camera_operations)
        self.panda_transformations.calibrate_camera()

        # Dictionary of object dropping points, defined as {"object_name": PointWithOrientation}
        self.drop_points = dict()
        self.drop_points = self.localise_dropping_locations(objects_names, vertical_offset=0.0)

        # List of tuples, where each tuple is defined as (object_name, PointWithOrientation)
        # this is not a dictionary, as we might have multiple objects with the same name
        self.pick_points = list()
        # self.pick_points = self.localise_picking_locations(objects_names, vertical_offset=0.0)

        # Add an obstacle if specified
        if include_obstacle:
            moveit_commander.roscpp_initialize([])
            self.scene = moveit_commander.PlanningSceneInterface()
            rospy.sleep(1.0)  # Give time for the scene to initialize
            self.add_box_obstacle(
                name="obstacle",
                size=(0.02, 0.02, 0.8),
                position=(0.42, 0.0, 0.4),
                collision_margin=0.03
            )

        # self.kalman_filter = AsynchronousPredictiveKalmanFilter(
        #     N=30.0 * KALMAN_FILTER_TIME_HORIZON,  # PREDICTION_HORIZON
        #     dt=1.0 / 30.0,
        #     process_noise_std=0.02,
        #     initial_estimate_covariance_diag=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        # )
        CAM_FPS: int = 10
        self.PREDICTION_HORIZON: int = 60; KF_DT: float = 1.0 / CAM_FPS
        KF_PROCESS_NOISE_STD: float = 0.02; KF_INITIAL_COV_DIAG: t.List[float] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.kalman_filter = AsynchronousPredictiveKalmanFilter(N=self.PREDICTION_HORIZON, dt=KF_DT, process_noise_std=KF_PROCESS_NOISE_STD, initial_estimate_covariance_diag=KF_INITIAL_COV_DIAG)
        measurement_noise_vars = np.array([0.003, 0.003, 0.01]) ** 2
        self.kalman_filter.R = np.diag(measurement_noise_vars)

    def localise_dropping_locations(self, objects_names: t.List[str], vertical_offset: float = 0.0) -> t.Dict[str, PointWithOrientation]:
        """
        Localise the dropping locations for the given objects.
        Should be called once at the beginning, as later objects may overlap with the code in the box.
        
        Return:
            A dictionary of object dropping points, defined as {"object_name": PointWithOrientation}
        """
        return_dict = dict()
        marker_transforms = self.camera_operations.get_marker_transforms()

        # Get an object, find assosiated aruco code, and transform it to the base frame
        for object_name in objects_names:
            if object_name not in self.object_name_to_aruco:
                raise ValueError(f"Object name {object_name} not found in object_name_to_aruco mapping.")

            aruco_code = self.object_name_to_aruco[object_name]
            if aruco_code not in marker_transforms:
                raise ValueError(f"Aruco code {aruco_code} for object {object_name} not found in marker transforms.")

            # Get the transform for the aruco code
            x, y, z = marker_transforms[aruco_code][:3, 3]
            # Create a PointWithOrientation for the dropping point
            dropping_point_in_camera_frame = PointWithOrientation(
                x=x,
                y=y,
                z=z,  # Add vertical offset if neededobject_to
                roll=0.0,
                pitch=0.0,
                yaw=0.0  # Assuming no specific orientation for dropping
            )
            dropping_point_in_robot_frame = self.panda_transformations.transform_point(dropping_point_in_camera_frame, 'camera', 'base')
            rospy.loginfo(f"Object {object_name} dropping point in robot frame: {dropping_point_in_robot_frame}")

            return_dict[object_name] = dropping_point_in_robot_frame

        rospy.loginfo(f"Localised dropping points: {return_dict}")
        return return_dict

    def localise_picking_locations(self, objects_names: t.List[str], vertical_offset: float = 0.0) -> t.List[t.Tuple[str, PointWithOrientation]]:
        """
        Localise the picking locations for the given objects. This is to be solved with a camera.
        Transfrom the points in camera frame to base

        Return:
            A List of tuples, where each tuple is defined as (object_name, PointWithOrientation)
        """
        list_of_picking_points_in_camera_frame = self.camera_operations.get_list_of_picking_points_in_camera_frame(objects_names, points_to_not_focus_on=self.drop_points.values())
        # list_of_picking_points_in_base_frame = list()
        for idx, (_, point_in_camera_frame) in enumerate(list_of_picking_points_in_camera_frame):
            list_of_picking_points_in_camera_frame[idx] = (_, self.panda_transformations.transform_point(point_in_camera_frame, "camera", "base"))

        return list_of_picking_points_in_camera_frame # it is already transformed

    def find_picking_point(self, object_name: str, vertical_offset: float = 0.0) -> PointWithOrientation:
        """
        Return only one picking point for the specified object.
        Based on the list of picking points, choose one and return it.
        """
        for name, point in self.pick_points:
            if name == object_name:
                # PRINT POINT AND IT"S ATTRIBUTES
                print(f"POINT -> {point}")
                return PointWithOrientation(
                    x=point.x,
                    y=point.y,
                    z=point.z + vertical_offset,  # Add vertical offset if needed
                    roll=point.roll,
                    pitch=point.pitch,
                    yaw=point.yaw
                )

        raise ValueError(f"Object {object_name} not found in picking points.")

    def kalman_wrapper_predict_xy(self, seconds_ahead: float = 5.0, predicted_states: np.ndarray = None) -> t.Tuple[float, float]:
        """
        Predict XY coordinates of object after `seconds_ahead` seconds using Kalman Filter.
        """
        # N = int(seconds_ahead / self.kalman_filter.dt)
        # predicted_states, _ = self.kalman_filter.update(N)
        print(f"ALL PREDICTED STATES -> {predicted_states}")
        step = self.PREDICTION_HORIZON - 1
        state_dim = 6
        current_pred_state = predicted_states[step*state_dim:(step+1)*state_dim]
        pred_pos_3d = current_pred_state[0:3]

        print(f"PREDICTED POSITION IS -> {pred_pos_3d}")
        return pred_pos_3d[0], pred_pos_3d[1]  # x, y

    def kalman_wrapper_update_filter_about_the_object(self, object_name: str, duration: float = 3.0):
        """
        Update Kalman Filter with XY position measurements of the object for a given duration.
        """
        start_time = time.time()
        while time.time() - start_time < duration and not rospy.is_shutdown():
            picking_points = self.camera_operations.get_list_of_bb_centers_for_picking_points_in_camera_frame(
                [object_name], points_to_not_focus_on=self.drop_points.values()
            )
            for name, point in picking_points:
                if name != object_name:
                    continue
                transformed = self.panda_transformations.transform_point(point, "camera", "base")
                measurement = np.array([transformed.x, transformed.y, transformed.z])
                print(f"updating kalman filter with x = {transformed.x}, y = {transformed.y}, z = {transformed.z}")
                predicted_states, _ = self.kalman_filter.update(measurement)
                break
            rospy.sleep(0.05)
        return predicted_states

    def pick_and_drop_specified_time(self, object_name: str, approach_vertical_offset: float = 0.0, duration: float = 5.0):
        """
        Pick and drop the specified object.

        1. Open gripper.
        2. Find the picking point for the object.
        3. Move to the picking point - if approaching from above, add post_goal_path
        4. Close the gripper to pick the object.
        5. Move to the dropping point.
        6. Open the gripper to drop the object.

        Args:
            object_name (str): Name of the object to pick and drop.
            approach_vertical_offset (float): Vertical offset to approach the object from above.
        """
        if object_name not in self.objects_to_clean:
            raise ValueError(f"Object {object_name} is not in the list of objects to clean.")

        print("ENTERING KALMAN WRAPPER UPDATE ABOUT THE OBJECT")
        # Track object for 3 seconds
        predicted_states = self.kalman_wrapper_update_filter_about_the_object(object_name=object_name, duration=5.0)

        # Open gripper
        self.robot_model.open_gripper()

        # Find the picking point for the object
        picking_point = self.find_picking_point(object_name, vertical_offset=approach_vertical_offset)
        rospy.loginfo(f"Picking point for {object_name}: {picking_point}")

        print("ENTERING KALMAN WRAPPER PREDICT XY")
        # Find out where will the object be in 5 seconds
        x_future, y_future = self.kalman_wrapper_predict_xy(seconds_ahead=duration, predicted_states = predicted_states)
        print(f"OF PICKING UP POINT OF {object_name} WAS AT X = {picking_point.x}, Y = {picking_point.y}")
        picking_point.x = x_future
        picking_point.y = y_future
        picking_point.z = 0.348 # 33 was ok for only conveyor belt
        print(f"I WILL PICK UP {object_name} IN {duration} seconds")

        # Move to the picking point
        if approach_vertical_offset > 0:
            actual_picking_point = copy.deepcopy(picking_point)
            actual_picking_point.z -= approach_vertical_offset
            post_goal_path = [actual_picking_point]
        else:
            post_goal_path = None

        self.robot_model.move_with_trajectory_planner_predefined_time(
            planner=self.demo_planner,
            post_processing=self.post_processing,
            goal=picking_point,
            post_goal_path=post_goal_path,
            post_processing_method=self.post_processing_method,
            duration = 5.0
        )

        # Close the gripper to pick the object
        self.robot_model.close_gripper()
        rospy.loginfo(f"Picked {object_name} at {picking_point}")

        # Move to the dropping point
        dropping_point = self.drop_points.get(object_name)
        if not dropping_point:
            raise ValueError(f"Dropping point for {object_name} not found.")

        drop_point = PointWithOrientation(
            x=dropping_point.x,
            y=dropping_point.y,
            z=dropping_point.z + approach_vertical_offset,  # Move Above
            roll=dropping_point.roll,
            pitch=dropping_point.pitch,
            yaw=dropping_point.yaw
        )

        # SIMPLIFY 
        point_above_picking_point = copy.deepcopy(picking_point)
        point_above_picking_point.z += 0.1

        intermediate_point = copy.deepcopy(picking_point)
        intermediate_point.z -= approach_vertical_offset / 2.0

        pre_start_path = [intermediate_point, picking_point, point_above_picking_point]

        point_above_drop_point = copy.deepcopy(drop_point)
        point_above_drop_point.z += 0.1
        post_goal_path = [drop_point]

        self.robot_model.move_with_trajectory_planner(
            planner=self.demo_planner,
            post_processing=self.post_processing,
            goal=point_above_drop_point,
            pre_start_path=pre_start_path,
            post_goal_path=post_goal_path,
            post_processing_method=self.post_processing_method
        )

        # Open the gripper to drop the object
        self.robot_model.open_gripper()
        rospy.loginfo(f"Dropped {object_name} at {dropping_point}")

    def run(self):
        """
        1. Look for objects to clean and generate picking points.
        2. Find an object position 5s in the future
        3. Pick and drop from that point, execute in such a way to pick at specified time .
        Repeat till localise_picking_location returns an empty list num_of_tries_to_find times in the row.
        """
        num_of_tries_to_find = 30
        failed_tries = 0
        rospy.loginfo("Starting the dynamic demo for picking and dropping objects.")

        while failed_tries < num_of_tries_to_find:
            self.pick_points = self.localise_picking_locations(self.objects_to_clean, vertical_offset=0.25)
            rospy.loginfo(f"=====================================================")
            rospy.loginfo(f"All objects which can be picked -> {self.pick_points}")
            if not self.pick_points:
                rospy.logwarn("No picking points found. Retrying...")
                failed_tries += 1
            else:
                object_name, _ = self.pick_points[0]
                rospy.loginfo(f"Trying to pick and drop object: {object_name}")
                self.pick_and_drop_specified_time(object_name, approach_vertical_offset=0.08, duration=5.0)
                failed_tries = 0

            if failed_tries >= int(num_of_tries_to_find/2):
                self.robot_model.move_to_joint_config(self.start_joint_config)

        rospy.loginfo("dynamic demo finished.")

    def add_box_obstacle(self, name, size, position, orientation=(0, 0, 0, 1), collision_margin=0.0):
        """
        Add a box obstacle to the planning scene with an optional collision margin.

        Args:
            name (str): Name of the obstacle.
            size (tuple): (x, y, z) dimensions of the actual box (meters).
            position (tuple): (x, y, z) center of the box (meters).
            orientation (tuple): Quaternion (x, y, z, w) orientation.
            collision_margin (float): Amount to inflate each dimension symmetrically (meters).
        """
        # Inflate size symmetrically
        inflated_size = tuple(s + 2 * collision_margin for s in size)

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.robot_model.base_link
        box_pose.pose.position.x = position[0]
        box_pose.pose.position.y = position[1]
        box_pose.pose.position.z = position[2]
        box_pose.pose.orientation.x = orientation[0]
        box_pose.pose.orientation.y = orientation[1]
        box_pose.pose.orientation.z = orientation[2]
        box_pose.pose.orientation.w = orientation[3]

        self.scene.add_box(name, box_pose, size=inflated_size)
        rospy.sleep(1.0)
        rospy.loginfo(f"Added box '{name}' at {position} with inflated size {inflated_size} (original: {size}, margin: {collision_margin})")

def main():
    rospy.init_node("static_demo_node", anonymous=True)

    # Example usage
    planner_name = "rrt_with_connecting"  # Choose from available planners
    post_processing_method = "quintic_bspline"  # Choose from available post processing methods
    objects_to_clean = ["scissors", "fork", "knife"]  # Define the objects to clean
    # objects_to_clean = ["banana"]

    demo = DynamicDemo(planner_name, post_processing_method, objects_to_clean, include_obstacle=False)
    demo.run()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
